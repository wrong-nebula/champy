// ============================================================================
// SPOROCARP — Instanced Micro-Botanical Forest-Floor Scatter Pass (WebGPU)
// Biome-clustered ecological forest floor with 6 botanical archetypes:
//   0: Contoured multi-lobed fallen oak leaves (midrib, lateral veins, margin blotches)
//   1: Convex Leucobryum pillow-moss cushions (deep emerald base -> chartreuse tips)
//   2: Slender conifer needles & fibrous cedar bark slivers
//   3: Rounded speckled mineral pebbles & quartz/granite grit
//   4: Hero micro-landmarks: moss-draped fallen twigs & curled oak bark arches
//   5: Tiny radial fern / liverwort micro-rosettes along moss-island edges
// Shaded in linear HDR with PCF + Fast-SDF-AO/GI, lighting moods & canopy gobo.
// ============================================================================
import {
  SoilMaterialId,
  type FruitingBodyState,
  type SimulationSnapshot,
  type SolidObstacle,
} from '../types/contract.ts';
import { evalSoilSurfaceHeightM, evalSoilSurfaceNormal } from './terrainHeight.ts';
import { WGSL_COMMON, makeWgslSdfGiHelpers, makeWgslShadowHelpers } from './shaders.ts';

const MAX_INSTANCES = 4500;
const FLOATS_PER_INST = 16; // instPosScale(4), instQuat(4), instColWet(4), instExtra(4)
const FLOATS_PER_VERT = 10; // pos(3), normal(3), uv(2), vertMeta(2: kind, localParam)
const SUBMESH_COUNT = 8; // 0:leaf, 1:moss, 2:needle, 3:barkSliver, 4:pebble, 5:twig, 6:barkArch, 7:rosette

const BOTANICAL_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;
${makeWgslSdfGiHelpers(1, 'sdfSamp', 3, 2)}
${makeWgslShadowHelpers(4, 5)}

struct BotVertIn {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) vertMeta: vec2<f32>,     // x = archetype kind (0..5), y = local height/margin param
  @location(4) instPosScale: vec4<f32>, // xyz = world pos, w = scale
  @location(5) instQuat: vec4<f32>,     // qx, qy, qz, qw
  @location(6) instColWet: vec4<f32>,   // rgb = sRGB base tint, w = wetness
  @location(7) instExtra: vec4<f32>,    // x = seed, y = subVariant, z = biomeLush, w = aoBias
};

struct BotVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) vertMeta: vec2<f32>,
  @location(4) colWet: vec4<f32>,
  @location(5) instExtra: vec4<f32>,
};

fn rotateQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

fn hash21(p: vec2<f32>) -> f32 {
  var q = fract(p * vec2<f32>(123.34, 456.21));
  q += dot(q, q + vec2<f32>(45.32));
  return fract(q.x * q.y);
}

fn valueNoise2D(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u2 = f * f * (vec2<f32>(3.0) - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2<f32>(1.0, 0.0));
  let c = hash21(i + vec2<f32>(0.0, 1.0));
  let d = hash21(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u2.x), mix(c, d, u2.x), u2.y);
}

fn fbm2D(p: vec2<f32>) -> f32 {
  var v = 0.0;
  var amp = 0.5;
  var freq = 1.0;
  for (var i = 0; i < 3; i = i + 1) {
    v += amp * valueNoise2D(p * freq);
    freq *= 2.07;
    amp *= 0.5;
  }
  return v * 1.142857;
}

@vertex
fn vs_main(v: BotVertIn) -> BotVertOut {
  var out: BotVertOut;
  let scaled = v.pos * v.instPosScale.w;
  let worldPos = v.instPosScale.xyz + rotateQuat(scaled, v.instQuat);
  let worldN = normalize(rotateQuat(v.normal, v.instQuat));
  out.clipPos = u.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.worldNormal = worldN;
  out.uv = v.uv;
  out.vertMeta = v.vertMeta;
  out.colWet = v.instColWet;
  out.instExtra = v.instExtra;
  return out;
}

@fragment
fn fs_main(in: BotVertOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4<f32> {
  if (isClippedBySlice(in.worldPos, u.sliceParams)) {
    discard;
  }

  let rawN = normalize(in.worldNormal);
  let nBase = select(-rawN, rawN, frontFacing);
  let l = normalize(u.lightDir.xyz);
  let v = normalize(u.cameraPos.xyz - in.worldPos);
  let h = normalize(l + v);

  let kind = i32(floor(in.vertMeta.x + 0.5));
  let param = clamp(in.vertMeta.y, 0.0, 1.0);
  let uv = in.uv;
  let seed = in.instExtra.x * 37.0;
  let subVar = in.instExtra.y;
  let lush = in.instExtra.z;
  let wet = clamp(in.colWet.w, 0.0, 1.0);

  // Tangent basis for procedural normal micro-relief
  let upRef = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(nBase.y) > 0.9);
  let t1 = normalize(cross(upRef, nBase));
  let t2 = cross(nBase, t1);

  var srgbAlbedo = in.colWet.rgb;
  var bumpGrad = vec2<f32>(0.0);
  var specStr = 0.02 + 0.08 * wet;
  var specPow = 28.0;
  var sssWeight = 0.35;
  var sssHue = vec3<f32>(1.15, 1.25, 0.65);
  var contactAo = 1.0;

  if (kind == 0) {
    // ------------------------------------------------------------------------
    // 0: Contoured Multi-Lobed Fallen Oak Leaf (midrib, lateral veins, blotches)
    // ------------------------------------------------------------------------
    let midDist = abs(uv.y - 0.5) * 2.0; // 0 at midrib, 1 at margin
    let petioleMask = 1.0 - smoothstep(0.06, 0.22, uv.x);
    let midWidth = mix(0.055, 0.018, uv.x);
    let midrib = (1.0 - smoothstep(0.0, midWidth, midDist)) * (1.0 - smoothstep(0.84, 0.98, uv.x));

    // Branching forward-angled lateral veins
    let veinPhase = (uv.x - midDist * 0.42 + valueNoise2D(uv * 8.0 + vec2<f32>(seed, seed * 0.5)) * 0.04) * 22.0;
    let veinLine = pow(1.0 - abs(sin(veinPhase * 3.14159)), 8.0);
    let subVein = pow(1.0 - abs(sin((veinPhase * 2.0 + 0.5) * 3.14159)), 12.0) * 0.45;
    let veinFade = smoothstep(0.03, 0.16, midDist) * (1.0 - smoothstep(0.65, 0.96, param)) * (1.0 - petioleMask);
    let veins = clamp((veinLine + subVein) * veinFade, 0.0, 1.0);

    // Organic blotches & weathered dark brown margin necrosis
    let blotchNoise = fbm2D(uv * vec2<f32>(7.5, 6.5) + vec2<f32>(seed * 1.3, seed * 2.1));
    let fineMottle = valueNoise2D(uv * 24.0 + vec2<f32>(seed * 3.7, 1.9));
    let marginRot = smoothstep(0.54, 0.98, param + (blotchNoise - 0.5) * 0.42);
    let spotRot = smoothstep(0.63, 0.80, blotchNoise * 0.72 + fineMottle * 0.28);

    // Base warm oak color + sunlit amber patches + dark umber veins/margins
    let sunlitAmber = srgbAlbedo * vec3<f32>(1.18, 1.12, 0.86) + vec3<f32>(0.05, 0.03, 0.0);
    let darkUmber = vec3<f32>(0.15, 0.085, 0.045);
    let midribCol = vec3<f32>(0.19, 0.11, 0.055);
    let darkSpotCol = vec3<f32>(0.17, 0.10, 0.055);

    srgbAlbedo = mix(srgbAlbedo, sunlitAmber, smoothstep(0.42, 0.72, fineMottle) * 0.45);
    srgbAlbedo = mix(srgbAlbedo, darkSpotCol, spotRot * 0.65);
    srgbAlbedo = mix(srgbAlbedo, darkUmber, marginRot * 0.72);
    srgbAlbedo = mix(srgbAlbedo, midribCol, max(midrib, petioleMask * 0.75) * 0.82);
    srgbAlbedo = mix(srgbAlbedo, darkUmber, veins * 0.58);

    // Subtle vein furrow normal bump
    let epsV = 0.02;
    let vpX = (uv.x + epsV - midDist * 0.42) * 22.0;
    let vpY = (uv.x - (midDist + epsV) * 0.42) * 22.0;
    let vGx = (pow(1.0 - abs(sin(vpX * 3.14159)), 6.0) - veinLine) * veinFade;
    let vGy = (pow(1.0 - abs(sin(vpY * 3.14159)), 6.0) - veinLine) * veinFade;
    bumpGrad = vec2<f32>(vGx, vGy) * 0.45;

    specStr = 0.03 + 0.14 * wet;
    specPow = mix(24.0, 56.0, wet);
    sssWeight = mix(0.85, 0.42, marginRot);
    sssHue = vec3<f32>(1.55, 1.02, 0.42);
    contactAo = mix(0.84, 1.0, smoothstep(0.0, 0.45, uv.x));
  } else if (kind == 1) {
    // ------------------------------------------------------------------------
    // 1: Convex Pillow-Moss Cushion (deep emerald base -> bright chartreuse tips)
    // ------------------------------------------------------------------------
    let pileCoord = uv * 42.0 + vec2<f32>(seed * 3.1, seed * 1.7);
    let tuftA = valueNoise2D(pileCoord);
    let tuftB = valueNoise2D(pileCoord * 2.25 + vec2<f32>(4.2, 8.7));
    let pileHeight = tuftA * 0.55 + tuftB * 0.45;

    // Fine surface micro-bump for velvety bryophyte gametophyte pile
    let epsM = 0.22;
    let hX = valueNoise2D(pileCoord + vec2<f32>(epsM, 0.0)) * 0.55 + valueNoise2D((pileCoord + vec2<f32>(epsM, 0.0)) * 2.25 + vec2<f32>(4.2, 8.7)) * 0.45;
    let hZ = valueNoise2D(pileCoord + vec2<f32>(0.0, epsM)) * 0.55 + valueNoise2D((pileCoord + vec2<f32>(0.0, epsM)) * 2.25 + vec2<f32>(4.2, 8.7)) * 0.45;
    bumpGrad = vec2<f32>(hX - pileHeight, hZ - pileHeight) * 1.25;

    let baseDeepEmerald = vec3<f32>(0.055, 0.145, 0.050);
    let midVelvetMoss = srgbAlbedo * vec3<f32>(0.92, 0.96, 0.88);
    let brightChartreuse = vec3<f32>(0.36, 0.56, 0.14) + vec3<f32>(0.04, 0.06, 0.01) * lush;

    let tipBand = smoothstep(0.18, 0.86, param + (pileHeight - 0.5) * 0.28);
    let microTips = smoothstep(0.54, 0.86, pileHeight) * smoothstep(0.30, 0.88, param);

    srgbAlbedo = mix(baseDeepEmerald, midVelvetMoss, tipBand);
    srgbAlbedo = mix(srgbAlbedo, brightChartreuse, microTips * (0.52 + 0.26 * lush));
    // Soft micro-crevice shading between shoot tips
    srgbAlbedo *= 0.86 + 0.26 * pileHeight;

    specStr = 0.015 + 0.05 * wet;
    specPow = 18.0;
    sssWeight = 0.72 + 0.35 * microTips;
    sssHue = vec3<f32>(1.10, 1.45, 0.42);
    contactAo = mix(0.62, 1.0, smoothstep(0.0, 0.38, param));
  } else if (kind == 2) {
    // ------------------------------------------------------------------------
    // 2: Slender Conifer Needles (subVar < 0.5) & Cedar Bark Slivers (>= 0.5)
    // ------------------------------------------------------------------------
    if (subVar < 0.5) {
      // Conifer needle: central stomatal keel + subtle golden-bronze tip weathering
      let centerRidge = 1.0 - smoothstep(0.0, 0.28, abs(uv.y - 0.5) * 2.0);
      let stria = valueNoise2D(vec2<f32>(uv.x * 18.0 + seed, uv.y * 42.0));
      let tipDry = smoothstep(0.68, 1.0, uv.x);
      srgbAlbedo = mix(srgbAlbedo, srgbAlbedo * 1.24 + vec3<f32>(0.04, 0.05, 0.01), centerRidge * 0.45);
      srgbAlbedo = mix(srgbAlbedo, vec3<f32>(0.46, 0.29, 0.12), tipDry * 0.42);
      srgbAlbedo *= 0.88 + 0.24 * stria;
      bumpGrad = vec2<f32>(0.0, (stria - 0.5) * 0.45);
      specStr = 0.05 + 0.14 * wet;
      specPow = 42.0;
      sssWeight = 0.48;
      sssHue = vec3<f32>(1.30, 1.20, 0.52);
    } else {
      // Cedar / rotting oak bark sliver: longitudinal fibrous grooves + lichen specks
      let barkCoord = vec2<f32>(uv.x * 5.0 + seed, uv.y * 28.0 + seed * 0.7);
      let fiber = fbm2D(barkCoord);
      let grain = valueNoise2D(vec2<f32>(uv.x * 26.0, uv.y * 55.0 + seed));
      let epsB = 0.18;
      let fY = fbm2D(barkCoord + vec2<f32>(0.0, epsB));
      bumpGrad = vec2<f32>((grain - 0.5) * 0.5, (fY - fiber) * 2.8);

      let grooveCol = vec3<f32>(0.13, 0.08, 0.05);
      let weatheredRidge = srgbAlbedo * vec3<f32>(1.15, 1.10, 1.05) + vec3<f32>(0.06, 0.05, 0.04);
      srgbAlbedo = mix(grooveCol, weatheredRidge, smoothstep(0.28, 0.76, fiber));
      let lichen = smoothstep(0.76, 0.90, fbm2D(uv * 11.0 + vec2<f32>(seed * 2.3, 4.1)));
      srgbAlbedo = mix(srgbAlbedo, vec3<f32>(0.36, 0.44, 0.28), lichen * 0.55);

      specStr = 0.01 + 0.05 * wet;
      specPow = 20.0;
      sssWeight = 0.08;
    }
    contactAo = mix(0.78, 1.0, param);
  } else if (kind == 3) {
    // ------------------------------------------------------------------------
    // 3: Rounded Speckled Mineral Pebbles (quartz, mica, feldspar speckles)
    // ------------------------------------------------------------------------
    let rockCoord = uv * 18.0 + vec2<f32>(seed * 2.1, seed * 1.4);
    let mineralA = valueNoise2D(rockCoord);
    let mineralB = valueNoise2D(rockCoord * 2.7 + vec2<f32>(7.1, 3.3));
    let micaSpeck = smoothstep(0.74, 0.92, mineralB);
    let veinBand = 1.0 - smoothstep(0.0, 0.055, abs(fbm2D(uv * 5.5 + vec2<f32>(seed, 2.0)) - 0.52));

    let epsP = 0.22;
    let mX = valueNoise2D(rockCoord + vec2<f32>(epsP, 0.0));
    let mZ = valueNoise2D(rockCoord + vec2<f32>(0.0, epsP));
    bumpGrad = vec2<f32>(mX - mineralA, mZ - mineralA) * 1.4;

    let darkMica = srgbAlbedo * 0.42;
    let paleQuartz = min(vec3<f32>(0.90), srgbAlbedo * 1.38 + vec3<f32>(0.10));
    srgbAlbedo = mix(srgbAlbedo * 0.85, srgbAlbedo * 1.14, mineralA);
    srgbAlbedo = mix(srgbAlbedo, darkMica, micaSpeck * 0.72);
    srgbAlbedo = mix(srgbAlbedo, paleQuartz, veinBand * 0.55);
    // Damp dark soil collar near base
    srgbAlbedo *= mix(0.68, 1.0, smoothstep(0.0, 0.35, param));
    specStr = mix(0.03 + 0.14 * wet, 0.06 + 0.18 * wet, veinBand);
    specPow = mix(32.0, 64.0, wet);
    sssWeight = 0.04;
    contactAo = mix(0.58, 1.0, smoothstep(0.0, 0.40, param));
  } else if (kind == 4) {
    // ------------------------------------------------------------------------
    // 4: Hero Micro-Landmarks: Moss-Draped Fallen Twigs & Curled Oak Bark Arches
    // ------------------------------------------------------------------------
    let barkCoord = vec2<f32>(uv.x * 7.0 + seed, uv.y * 22.0 + seed * 0.5);
    let furrow = fbm2D(barkCoord);
    let fineBark = valueNoise2D(barkCoord * 2.6);
    let epsT = 0.16;
    let bX = fbm2D(barkCoord + vec2<f32>(epsT, 0.0));
    let bY = fbm2D(barkCoord + vec2<f32>(0.0, epsT));

    let deepBark = vec3<f32>(0.14, 0.09, 0.055);
    let midBark = srgbAlbedo;
    let paleBarkRidge = srgbAlbedo * 1.30 + vec3<f32>(0.07, 0.06, 0.05);
    var twigSrgb = mix(deepBark, midBark, smoothstep(0.24, 0.64, furrow));
    twigSrgb = mix(twigSrgb, paleBarkRidge, smoothstep(0.68, 0.90, furrow * 0.7 + fineBark * 0.3) * 0.55);

    // Draped emerald-chartreuse moss patches clinging onto top shoulders (nBase.y > 0.25)
    let topFacing = smoothstep(0.18, 0.68, nBase.y);
    let mossPatchNoise = fbm2D(vec2<f32>(uv.x * 3.6 + seed * 1.4, uv.y * 3.2 - seed * 0.8));
    let mossTipGrain = valueNoise2D(uv * 24.0 + vec2<f32>(seed * 3.1, 2.7));
    let mossThreshold = mix(0.56, 0.32, lush * 0.65 + wet * 0.35);
    let mossMask = topFacing * smoothstep(mossThreshold, mossThreshold + 0.20, mossPatchNoise);

    let drapedEmerald = vec3<f32>(0.075, 0.21, 0.055);
    let drapedChartreuse = vec3<f32>(0.29, 0.48, 0.11);
    let drapedMoss = mix(drapedEmerald, drapedChartreuse, smoothstep(0.36, 0.86, mossTipGrain) * topFacing);

    srgbAlbedo = mix(twigSrgb, drapedMoss, mossMask * 0.92);
    let barkBump = vec2<f32>(bX - furrow, bY - furrow) * 2.2;
    let mossBump = vec2<f32>(mossTipGrain - 0.5, (fineBark - 0.5)) * 1.4;
    bumpGrad = mix(barkBump, mossBump, mossMask);

    specStr = mix(0.015 + 0.06 * wet, 0.02 + 0.05 * wet, mossMask);
    specPow = 22.0;
    sssWeight = mix(0.12, 0.92, mossMask);
    sssHue = vec3<f32>(1.15, 1.50, 0.44);
    contactAo = mix(0.55, 1.0, smoothstep(0.0, 0.32, param));
  } else {
    // ------------------------------------------------------------------------
    // 5: Tiny Radial Fern / Liverwort Micro-Rosettes at Moss-Island Edges
    // ------------------------------------------------------------------------
    let centerDist = param; // 0 at heart -> 1 at frond tip
    let midDist = abs(uv.y - 0.5) * 2.0;
    let midvein = 1.0 - smoothstep(0.0, 0.14, midDist);
    // Delicate chevron pinnae / thallus lobes along frond
    let pinnaPhase = (uv.x * 10.0 - midDist * 2.4) * 3.14159;
    let pinnae = smoothstep(0.15, 0.85, sin(pinnaPhase) * 0.5 + 0.5);
    let cellGrain = valueNoise2D(uv * 22.0 + vec2<f32>(seed, seed * 1.7));

    let heartCol = vec3<f32>(0.11, 0.22, 0.06);
    let frondEmerald = srgbAlbedo;
    let translucentTip = vec3<f32>(0.42, 0.68, 0.19);

    srgbAlbedo = mix(heartCol, frondEmerald, smoothstep(0.08, 0.45, centerDist));
    srgbAlbedo = mix(srgbAlbedo, translucentTip, smoothstep(0.55, 0.98, centerDist) * 0.62);
    srgbAlbedo = mix(srgbAlbedo * 0.86, srgbAlbedo * 1.08, pinnae);
    srgbAlbedo = mix(srgbAlbedo, vec3<f32>(0.14, 0.25, 0.07), midvein * 0.45);
    srgbAlbedo *= 0.90 + 0.20 * cellGrain;

    bumpGrad = vec2<f32>(cos(pinnaPhase) * 0.35, (midDist - 0.5) * 0.25);
    specStr = 0.04 + 0.10 * wet;
    specPow = 34.0;
    sssWeight = 1.18;
    sssHue = vec3<f32>(1.25, 1.65, 0.45);
    contactAo = mix(0.68, 1.0, smoothstep(0.0, 0.38, centerDist));
  }

  let n = normalize(nBase + t1 * bumpGrad.x + t2 * bumpGrad.y);

  let aoBent = evalFastSdfAO(in.worldPos + n * 0.004, n);
  let bentN = aoBent.xyz;
  let sdfAo = aoBent.w * contactAo;
  let shadow = min(evalPcfShadow(in.worldPos, n), mix(0.35, 1.0, evalSdfSoftShadow(in.worldPos + n * 0.006, l)));
  let gi = evalSdfGI(in.worldPos + n * 0.005, n, bentN);

  // Lighting mood & drifting warm dappled leaf-canopy gobo from u.padExtra
  // u.padExtra: x = lightingMood (0 Studio, 1 Canopy, 2 Dark-Field Backlit), y = realTimeSec, z = goboStrength
  let mood = u.padExtra.x;
  let realTimeSec = u.padExtra.y;
  let goboStrength = clamp(u.padExtra.z, 0.0, 1.0);

  var goboTint = vec3<f32>(1.0);
  if (goboStrength > 0.01) {
    let lUp = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(l.y) > 0.9);
    let lTan = normalize(cross(lUp, l));
    let lBit = cross(l, lTan);
    let goboUv = vec2<f32>(dot(in.worldPos, lTan), dot(in.worldPos, lBit));
    let drift = vec2<f32>(realTimeSec * 0.016, realTimeSec * 0.010);
    let g1 = valueNoise2D(goboUv * 12.5 + drift + vec2<f32>(2.1, 5.7));
    let g2 = valueNoise2D(goboUv * 25.0 - drift * 0.72 + vec2<f32>(9.3, 1.4));
    let sunfleck = smoothstep(0.30, 0.74, g1 * 0.64 + g2 * 0.36);
    let canopyShade = vec3<f32>(0.34, 0.44, 0.40);
    let canopyFlecks = vec3<f32>(1.24, 1.06, 0.78);
    goboTint = mix(vec3<f32>(1.0), mix(canopyShade, canopyFlecks, sunfleck), goboStrength);
  }

  let isCanopy = mood >= 0.5 && mood <= 1.5;
  let isBacklit = mood > 1.5;

  let sunBaseCol = select(
    vec3<f32>(0.96, 0.90, 0.78),
    select(vec3<f32>(1.08, 0.94, 0.70), vec3<f32>(1.04, 0.96, 0.84), isBacklit),
    isCanopy || isBacklit,
  );
  let skyLow = select(vec3<f32>(0.032, 0.038, 0.044), vec3<f32>(0.018, 0.024, 0.030), isBacklit);
  let skyHigh = select(
    vec3<f32>(0.11, 0.14, 0.18),
    select(vec3<f32>(0.10, 0.15, 0.14), vec3<f32>(0.068, 0.088, 0.115), isBacklit),
    isCanopy || isBacklit,
  );

  // 3-point macro botanical lighting (sun key + bent sky fill + grazing rim SSS)
  let wrapNdl = clamp((dot(n, l) + 0.42) / 1.42, 0.0, 1.0);
  let sun = sunBaseCol * goboTint * wrapNdl * mix(0.22, 1.0, shadow);
  let sky = mix(skyLow, skyHigh, bentN.y * 0.5 + 0.5);

  let sssAlign = pow(clamp(dot(v, -l + n * 0.35), 0.0, 1.0), 2.3);
  let grazingFresnel = pow(1.0 - max(dot(n, v), 0.0), 2.6);
  let moodSssMult = select(select(1.0, 1.32, isCanopy), 2.65, isBacklit);
  let backSss = (sssAlign * 0.22 + grazingFresnel * select(0.03, 0.20, isBacklit))
    * moodSssMult * sssWeight * (0.35 + 0.65 * shadow);

  // Linear HDR output parity (strictly srgb -> linear, no premature gamma/ACES)
  let albedo = srgbToLinear(srgbAlbedo);
  let sssTint = albedo * sssHue;
  let spec = pow(max(dot(n, h), 0.0), specPow) * specStr * shadow * sdfAo * sunBaseCol * goboTint;

  let color = albedo * ((sky + sun) * sdfAo + gi * 0.52) + sssTint * backSss + spec;
  return vec4<f32>(color, 1.0);
}
`;

function hash1(n: number): number {
  const x = Math.sin(n * 12.9898 + 78.233) * 43758.5453123;
  return x - Math.floor(x);
}

function hash2(ix: number, iz: number): number {
  const n = Math.sin(ix * 127.1 + iz * 311.7) * 43758.5453123;
  return n - Math.floor(n);
}

function noise2(x: number, z: number): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uz = fz * fz * (3 - 2 * fz);
  const a = hash2(ix, iz);
  const b = hash2(ix + 1, iz);
  const c = hash2(ix, iz + 1);
  const d = hash2(ix + 1, iz + 1);
  return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
}

function fbm2(x: number, z: number): number {
  let v = 0;
  let amp = 0.5;
  let freq = 1;
  for (let i = 0; i < 3; i++) {
    v += amp * (noise2(x * freq, z * freq) - 0.5);
    freq *= 2.07;
    amp *= 0.5;
  }
  return v;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / Math.max(1e-6, edge1 - edge0));
  return t * t * (3 - 2 * t);
}

/**
 * Write unit quaternion aligning +Y with terrain normal `nrm` composed with Y-up yaw,
 * X-pitch, and Z-roll: qSlope * qYaw(Y) * qPitch(X) * qRoll(Z).
 */
function writeSlopeAlignedQuat(
  out: Float32Array,
  off: number,
  nrm: readonly [number, number, number],
  yaw: number,
  pitch: number,
  roll: number,
): void {
  const cy = Math.cos(yaw * 0.5), sy = Math.sin(yaw * 0.5);
  const cp = Math.cos(pitch * 0.5), sp = Math.sin(pitch * 0.5);
  const cr = Math.cos(roll * 0.5), sr = Math.sin(roll * 0.5);
  // qYaw(0, sy, 0, cy) * qPitch(sp, 0, 0, cp)
  const ypx = cy * sp;
  const ypy = sy * cp;
  const ypz = -sy * sp;
  const ypw = cy * cp;
  // * qRoll(0, 0, sr, cr)
  const lx = ypx * cr + ypy * sr;
  const ly = ypy * cr - ypx * sr;
  const lz = ypz * cr + ypw * sr;
  const lw = ypw * cr - ypz * sr;

  // Shortest-arc quaternion from (0, 1, 0) -> (nx, ny, nz)
  const sx = nrm[2];
  const sz = -nrm[0];
  const sw = 1 + nrm[1];
  const sInv = 1 / (Math.hypot(sx, sz, sw) || 1);
  const qax = sx * sInv;
  const qaz = sz * sInv;
  const qaw = sw * sInv;

  // qSlope * qLocal (qay = 0)
  const qx = qaw * lx + qax * lw - qaz * ly;
  const qy = qaw * ly - qax * lz + qaz * lx;
  const qz = qaw * lz + qax * ly + qaz * lw;
  const qw = qaw * lw - qax * lx - qaz * lz;
  const qInv = 1 / (Math.hypot(qx, qy, qz, qw) || 1);
  out[off] = qx * qInv;
  out[off + 1] = qy * qInv;
  out[off + 2] = qz * qInv;
  out[off + 3] = qw * qInv;
}

interface SubMeshSlice {
  firstIndex: number;
  indexCount: number;
}

interface GridPoint {
  px: number;
  py: number;
  pz: number;
  u: number;
  v: number;
  param: number;
}

function buildArchetypeGeometry(): {
  verts: Float32Array;
  indices: Uint16Array;
  slices: SubMeshSlice[];
} {
  const v: number[] = [];
  const idx: number[] = [];
  const slices: SubMeshSlice[] = [];

  const pushVert = (
    px: number, py: number, pz: number,
    nx: number, ny: number, nz: number,
    u: number, vUv: number,
    kind: number, localParam: number,
  ): number => {
    const l = Math.hypot(nx, ny, nz) || 1;
    const vi = v.length / FLOATS_PER_VERT;
    v.push(px, py, pz, nx / l, ny / l, nz / l, u, vUv, kind, localParam);
    return vi;
  };
  const pushTri = (a: number, b: number, c: number) => idx.push(a, b, c);

  // Build an open curved grid surface with finite-difference upward normals and CCW (+Y) winding.
  const emitGridSurface = (
    rows: number,
    cols: number,
    grid: GridPoint[][],
    kind: number,
  ): void => {
    const vertIds: number[][] = [];
    for (let i = 0; i <= rows; i++) {
      const rowIds: number[] = [];
      const i0 = Math.max(0, i - 1);
      const i1 = Math.min(rows, i + 1);
      for (let j = 0; j <= cols; j++) {
        const j0 = Math.max(0, j - 1);
        const j1 = Math.min(cols, j + 1);
        const p = grid[i][j];
        // dP/ds (along +Z) x dP/dt (along +X) -> +Y outward
        const sx = grid[i][j1].px - grid[i][j0].px;
        const sy = grid[i][j1].py - grid[i][j0].py;
        const sz = grid[i][j1].pz - grid[i][j0].pz;
        const tx = grid[i1][j].px - grid[i0][j].px;
        const ty = grid[i1][j].py - grid[i0][j].py;
        const tz = grid[i1][j].pz - grid[i0][j].pz;
        let nx = sy * tz - sz * ty;
        let ny = sz * tx - sx * tz;
        let nz = sx * ty - sy * tx;
        if (Math.hypot(nx, ny, nz) < 1e-6) {
          nx = 0; ny = 1; nz = 0;
        }
        rowIds.push(pushVert(p.px, p.py, p.pz, nx, ny, nz, p.u, p.v, kind, p.param));
      }
      vertIds.push(rowIds);
    }
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const a = vertIds[i][j];
        const b = vertIds[i][j + 1];
        const c = vertIds[i + 1][j + 1];
        const d = vertIds[i + 1][j];
        pushTri(a, b, c);
        pushTri(a, c, d);
      }
    }
  };

  // 0: Contoured Multi-Lobed Fallen Oak Leaf (petiole + 3 rounded lobe pairs + cup curvature)
  {
    const firstIndex = idx.length;
    const rows = 10;
    const cols = 4; // 5 verts across: s = -1, -0.5, 0, 0.5, 1
    const grid: GridPoint[][] = [];
    for (let i = 0; i <= rows; i++) {
      const t = i / rows;
      const px = -1.0 + t * 1.96;
      const row: GridPoint[] = [];
      for (let j = 0; j <= cols; j++) {
        const s = (j / cols) * 2 - 1;
        let halfW = 0.032;
        if (t > 0.12) {
          const uBlade = (t - 0.12) / 0.88;
          const asym = s >= 0 ? 0.035 : -0.035;
          const env = Math.sin(Math.pow(uBlade, 0.72) * Math.PI);
          const lobes = 0.56 + 0.44 * Math.cos(((uBlade + asym) * 3.10 - 0.42) * Math.PI * 2);
          halfW = 0.032 + 0.54 * env * lobes;
        }
        const pz = s * halfW;
        const arch = 0.030 * Math.sin(t * Math.PI);
        const cup = 0.056 * (s * s) * smoothstep(0.08, 0.30, t);
        const tipCurl = 0.048 * Math.pow(Math.max(0, t - 0.64) / 0.36, 1.6);
        const twist = 0.022 * s * (t - 0.45);
        const py = 0.008 + arch + cup + tipCurl + twist;
        row.push({ px, py, pz, u: t, v: s * 0.5 + 0.5, param: Math.abs(s) });
      }
      grid.push(row);
    }
    emitGridSurface(rows, cols, grid, 0);
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 1: Convex Leucobryum Pillow-Moss Cushion Mound (CCW outward winding)
  {
    const firstIndex = idx.length;
    const crownH = 0.46;
    const crown = pushVert(0.0, crownH, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1, 1.0);
    const segs = 10;
    const rSteps = [0.34, 0.68, 1.0];
    const rings: number[][] = [];
    for (let ri = 0; ri < rSteps.length; ri++) {
      const rNorm = rSteps[ri];
      const ring: number[] = [];
      for (let k = 0; k < segs; k++) {
        const a = (k / segs) * Math.PI * 2;
        const ca = Math.cos(a), sa = Math.sin(a);
        const lobeMod = 1.0 + 0.14 * Math.sin(a * 3.0 + 0.4) + 0.08 * Math.cos(a * 5.0 - 0.8);
        const r = rNorm * lobeMod;
        const px = r * ca;
        const pz = r * sa;
        const baseDome = ri === rSteps.length - 1
          ? 0.0
          : crownH * Math.pow(Math.cos(rNorm * Math.PI * 0.49), 0.72) * (1.0 + 0.07 * Math.sin(a * 3.0 + 0.4) * rNorm);
        const ny = Math.max(0.24, Math.cos(rNorm * Math.PI * 0.42));
        const nr = Math.sin(rNorm * Math.PI * 0.42);
        ring.push(pushVert(
          px, baseDome, pz,
          nr * ca, ny, nr * sa,
          px * 0.42 + 0.5, pz * 0.42 + 0.5,
          1, clamp01(baseDome / crownH),
        ));
      }
      rings.push(ring);
    }
    for (let k = 0; k < segs; k++) {
      const k1 = (k + 1) % segs;
      pushTri(crown, rings[0][k1], rings[0][k]);
    }
    for (let ri = 0; ri < rings.length - 1; ri++) {
      const inner = rings[ri];
      const outer = rings[ri + 1];
      for (let k = 0; k < segs; k++) {
        const k1 = (k + 1) % segs;
        pushTri(inner[k], inner[k1], outer[k1]);
        pushTri(inner[k], outer[k1], outer[k]);
      }
    }
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 2: Slender Conifer Needle (curved central keel blade)
  {
    const firstIndex = idx.length;
    const rows = 6;
    const cols = 2; // s = -1, 0, 1
    const grid: GridPoint[][] = [];
    for (let i = 0; i <= rows; i++) {
      const t = i / rows;
      const px = -0.95 + t * 1.90;
      const halfW = 0.056 * (0.42 + 0.58 * Math.sin(t * Math.PI));
      const bendZ = 0.09 * Math.sin(t * Math.PI);
      const archY = 0.012 + 0.032 * Math.sin(t * Math.PI);
      const row: GridPoint[] = [];
      for (let j = 0; j <= cols; j++) {
        const s = j - 1;
        const pz = bendZ + s * halfW;
        const py = archY + (s === 0 ? 0.016 : 0.0);
        row.push({ px, py, pz, u: t, v: s * 0.5 + 0.5, param: Math.abs(s) });
      }
      grid.push(row);
    }
    emitGridSurface(rows, cols, grid, 2);
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 3: Fibrous Cedar / Oak Bark Sliver (wider curved weathered strip)
  {
    const firstIndex = idx.length;
    const rows = 7;
    const cols = 3; // s = -1, -1/3, +1/3, 1
    const grid: GridPoint[][] = [];
    for (let i = 0; i <= rows; i++) {
      const t = i / rows;
      const px = -0.96 + t * 1.92;
      const halfW = 0.165 * (0.62 + 0.38 * Math.sin(t * Math.PI)) * (1.0 + 0.14 * Math.sin(t * 11.0));
      const archY = 0.010 + 0.038 * Math.sin(t * Math.PI);
      const row: GridPoint[] = [];
      for (let j = 0; j <= cols; j++) {
        const s = (j / cols) * 2 - 1;
        const pz = s * halfW + 0.035 * Math.sin(t * Math.PI * 1.3);
        const py = archY + 0.038 * (1.0 - s * s);
        row.push({ px, py, pz, u: t, v: s * 0.5 + 0.5, param: 1.0 - Math.abs(s) * 0.5 });
      }
      grid.push(row);
    }
    emitGridSurface(rows, cols, grid, 2);
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 4: Rounded Speckled Mineral Pebble (smooth triaxial river pebble, CCW outward winding)
  {
    const firstIndex = idx.length;
    const crownH = 0.42;
    const crown = pushVert(0.0, crownH, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 3, 1.0);
    const segs = 8;
    const rSteps = [0.38, 0.72, 1.0];
    const rings: number[][] = [];
    for (let ri = 0; ri < rSteps.length; ri++) {
      const rNorm = rSteps[ri];
      const ring: number[] = [];
      for (let k = 0; k < segs; k++) {
        const a = (k / segs) * Math.PI * 2;
        const ca = Math.cos(a), sa = Math.sin(a);
        const rMod = 0.92 + 0.11 * Math.sin(a * 2.0 + 0.7) + 0.06 * Math.cos(a * 3.0 - 0.4);
        const rx = 0.98 * rMod * rNorm;
        const rz = 0.76 * rMod * rNorm;
        const h = ri === rSteps.length - 1
          ? 0.0
          : crownH * Math.sqrt(Math.max(0.02, 1.0 - rNorm * rNorm * 0.96));
        const ny = Math.max(0.22, Math.cos(rNorm * Math.PI * 0.44));
        const nr = Math.sin(rNorm * Math.PI * 0.44);
        ring.push(pushVert(
          rx * ca, h, rz * sa,
          nr * ca / 0.98, ny, nr * sa / 0.76,
          rx * ca * 0.45 + 0.5, rz * sa * 0.45 + 0.5,
          3, clamp01(h / crownH),
        ));
      }
      rings.push(ring);
    }
    for (let k = 0; k < segs; k++) {
      const k1 = (k + 1) % segs;
      pushTri(crown, rings[0][k1], rings[0][k]);
    }
    for (let ri = 0; ri < rings.length - 1; ri++) {
      const inner = rings[ri];
      const outer = rings[ri + 1];
      for (let k = 0; k < segs; k++) {
        const k1 = (k + 1) % segs;
        pushTri(inner[k], inner[k1], outer[k1]);
        pushTri(inner[k], outer[k1], outer[k]);
      }
    }
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 5: Hero Micro-Landmark A — Moss-Draped Gnarled Fallen Twig w/ Forked Side Stub
  {
    const firstIndex = idx.length;
    const emitTube = (
      spine: Array<{ x: number; y: number; z: number; r: number }>,
      radSegs: number,
      uScale: number,
    ) => {
      const rings: number[][] = [];
      for (let i = 0; i < spine.length; i++) {
        const p = spine[i];
        const prev = spine[Math.max(0, i - 1)];
        const next = spine[Math.min(spine.length - 1, i + 1)];
        let tx = next.x - prev.x;
        let ty = next.y - prev.y;
        let tz = next.z - prev.z;
        const tl = Math.hypot(tx, ty, tz) || 1;
        tx /= tl; ty /= tl; tz /= tl;

        // Orthonormal frame (T, N_up, B_side) where T x N_up = B_side
        const refX = Math.abs(ty) > 0.9 ? 1 : 0;
        const refY = Math.abs(ty) > 0.9 ? 0 : 1;
        const refZ = 0;
        let bx = ty * refZ - tz * refY;
        let by = tz * refX - tx * refZ;
        let bz = tx * refY - ty * refX;
        const bl = Math.hypot(bx, by, bz) || 1;
        bx /= bl; by /= bl; bz /= bl;
        const ux = by * tz - bz * ty;
        const uy = bz * tx - bx * tz;
        const uz = bx * ty - by * tx;

        const tNorm = i / (spine.length - 1);
        const ring: number[] = [];
        for (let k = 0; k < radSegs; k++) {
          const a = (k / radSegs) * Math.PI * 2;
          const ca = Math.cos(a), sa = Math.sin(a);
          const nx = ca * ux + sa * bx;
          const ny = ca * uy + sa * by;
          const nz = ca * uz + sa * bz;
          const px = p.x + nx * p.r;
          const py = p.y + ny * p.r;
          const pz = p.z + nz * p.r;
          ring.push(pushVert(
            px, py, pz,
            nx, ny, nz,
            tNorm * uScale, k / radSegs,
            4, clamp01((py + 0.04) / 0.28),
          ));
        }
        rings.push(ring);
      }
      for (let i = 0; i < rings.length - 1; i++) {
        for (let k = 0; k < radSegs; k++) {
          const k1 = (k + 1) % radSegs;
          pushTri(rings[i][k], rings[i][k1], rings[i + 1][k1]);
          pushTri(rings[i][k], rings[i + 1][k1], rings[i + 1][k]);
        }
      }
    };

    const mainSpine: Array<{ x: number; y: number; z: number; r: number }> = [];
    const mainSegs = 8;
    for (let i = 0; i <= mainSegs; i++) {
      const t = i / mainSegs;
      mainSpine.push({
        x: -1.10 + t * 2.20,
        y: 0.18 * Math.sin(t * Math.PI) - 0.03,
        z: 0.14 * Math.sin(t * Math.PI * 1.35 - 0.25),
        r: 0.115 * (1.0 - 0.22 * t) * (1.0 + 0.15 * Math.sin(t * Math.PI * 5.0)),
      });
    }
    emitTube(mainSpine, 7, 2.5);

    // Short gnarled side-twig stub sprouting cleanly from main stem
    const stubSpine: Array<{ x: number; y: number; z: number; r: number }> = [];
    const origin = mainSpine[3];
    for (let i = 0; i <= 3; i++) {
      const s = i / 3;
      stubSpine.push({
        x: origin.x + s * 0.34,
        y: origin.y + s * 0.16,
        z: origin.z + s * 0.32,
        r: 0.072 * (1.0 - 0.55 * s),
      });
    }
    emitTube(stubSpine, 6, 1.0);
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 6: Hero Micro-Landmark B — Curled Oak Bark Arch
  {
    const firstIndex = idx.length;
    const rows = 8;
    const cols = 4;
    const grid: GridPoint[][] = [];
    for (let i = 0; i <= rows; i++) {
      const t = i / rows;
      const px = -1.06 + t * 2.12;
      const archY = 0.22 * Math.sin(t * Math.PI) - 0.02;
      const halfW = 0.34 * (0.78 + 0.22 * Math.sin(t * Math.PI)) * (1.0 + 0.10 * Math.cos(t * 9.0));
      const row: GridPoint[] = [];
      for (let j = 0; j <= cols; j++) {
        const s = (j / cols) * 2 - 1;
        const pz = s * halfW + 0.06 * Math.sin(t * Math.PI);
        const curlY = 0.085 * (1.0 - s * s);
        const twistY = 0.045 * s * (t - 0.5);
        const py = archY + curlY + twistY;
        row.push({
          px, py, pz,
          u: t * 2.0,
          v: s * 0.5 + 0.5,
          param: clamp01((py + 0.03) / 0.30),
        });
      }
      grid.push(row);
    }
    emitGridSurface(rows, cols, grid, 4);
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 7: Tiny Radial Fern / Liverwort Micro-Rosette (7 overlapping arched fronds)
  {
    const firstIndex = idx.length;
    const frondConfigs: Array<{ angle: number; len: number; width: number; lift: number; droop: number }> = [
      { angle: 0.05, len: 1.00, width: 0.26, lift: 0.15, droop: 0.06 },
      { angle: 1.62, len: 0.94, width: 0.25, lift: 0.14, droop: 0.05 },
      { angle: 3.18, len: 1.02, width: 0.27, lift: 0.16, droop: 0.07 },
      { angle: 4.78, len: 0.96, width: 0.25, lift: 0.14, droop: 0.05 },
      { angle: 0.84, len: 0.68, width: 0.21, lift: 0.26, droop: 0.16 },
      { angle: 2.86, len: 0.72, width: 0.22, lift: 0.25, droop: 0.15 },
      { angle: 5.02, len: 0.66, width: 0.20, lift: 0.27, droop: 0.17 },
    ];
    for (let f = 0; f < frondConfigs.length; f++) {
      const fc = frondConfigs[f];
      const ca = Math.cos(fc.angle), sa = Math.sin(fc.angle);
      // Tangent along frond: (ca, 0, sa), side perpendicular (-sa, 0, ca) -> side x tangent = +Y
      const rows = 3;
      const cols = 2; // s = -1, 0, 1
      const grid: GridPoint[][] = [];
      for (let i = 0; i <= rows; i++) {
        const t = i / rows;
        const rDist = 0.04 + t * fc.len;
        const halfW = fc.width * Math.sin(Math.pow(Math.max(0.06, t), 0.62) * Math.PI * 0.94);
        const archY = 0.02 + fc.lift * Math.sin(t * Math.PI * 0.75) + fc.droop * t;
        const row: GridPoint[] = [];
        for (let j = 0; j <= cols; j++) {
          const s = j - 1;
          const px = rDist * ca - s * halfW * sa;
          const pz = rDist * sa + s * halfW * ca;
          const py = archY + Math.abs(s) * 0.018;
          row.push({ px, py, pz, u: t, v: s * 0.5 + 0.5, param: t });
        }
        grid.push(row);
      }
      emitGridSurface(rows, cols, grid, 5);
    }
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  return {
    verts: new Float32Array(v),
    indices: new Uint16Array(idx),
    slices,
  };
}

export class BotanicalScatterPass {
  private device: GPUDevice;
  private pipe: GPURenderPipeline;
  private geomVertBuf: GPUBuffer;
  private geomIdxBuf: GPUBuffer;
  private instBuf: GPUBuffer;
  private instScratch = new Float32Array(MAX_INSTANCES * FLOATS_PER_INST);
  private slices: SubMeshSlice[];
  private counts = new Int32Array(SUBMESH_COUNT);
  private lastSig = '';

  constructor(
    device: GPUDevice,
    hdrFormat: GPUTextureFormat,
    litBindGroupLayout: GPUBindGroupLayout,
  ) {
    this.device = device;
    const geom = buildArchetypeGeometry();
    this.slices = geom.slices;

    this.geomVertBuf = device.createBuffer({
      size: geom.verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      this.geomVertBuf,
      0,
      geom.verts.buffer as ArrayBuffer,
      geom.verts.byteOffset,
      geom.verts.byteLength,
    );

    const paddedIdxBytes = Math.ceil(geom.indices.byteLength / 4) * 4;
    const paddedIndices = new Uint16Array(paddedIdxBytes / 2);
    paddedIndices.set(geom.indices);
    this.geomIdxBuf = device.createBuffer({
      size: paddedIdxBytes,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      this.geomIdxBuf,
      0,
      paddedIndices.buffer as ArrayBuffer,
      paddedIndices.byteOffset,
      paddedIndices.byteLength,
    );

    this.instBuf = device.createBuffer({
      size: MAX_INSTANCES * FLOATS_PER_INST * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const mod = device.createShaderModule({ code: BOTANICAL_WGSL });
    this.pipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [litBindGroupLayout] }),
      vertex: {
        module: mod,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: FLOATS_PER_VERT * 4,
            stepMode: 'vertex',
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x2' },
              { shaderLocation: 3, offset: 32, format: 'float32x2' },
            ],
          },
          {
            arrayStride: FLOATS_PER_INST * 4,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 4, offset: 0, format: 'float32x4' },
              { shaderLocation: 5, offset: 16, format: 'float32x4' },
              { shaderLocation: 6, offset: 32, format: 'float32x4' },
              { shaderLocation: 7, offset: 48, format: 'float32x4' },
            ],
          },
        ],
      },
      fragment: {
        module: mod,
        entryPoint: 'fs_main',
        targets: [{ format: hdrFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: 4 },
    });
  }

  update(snapshot: SimulationSnapshot): void {
    const soil = snapshot.soil;
    let sigNum = soil.nx * 73856093 ^ soil.nz * 83492791 ^ snapshot.bodies.length * 19349663 ^ snapshot.obstacles.length * 2654435761;
    const total = soil.material.length;
    const stride = Math.max(1, Math.floor(total / 192));
    for (let i = 0; i < total; i += stride) {
      sigNum = (sigNum * 31 + soil.material[i] * 17 + Math.round(soil.theta[i] * 20)) | 0;
    }
    const bodies: readonly FruitingBodyState[] = snapshot.bodies;
    for (let i = 0; i < bodies.length; i++) {
      sigNum = (sigNum * 31 + Math.round(bodies[i].baseX * 200) * 13 + Math.round(bodies[i].baseZ * 200) * 7) | 0;
    }
    const sig = `${sigNum}:${snapshot.species.id}`;
    if (sig === this.lastSig) return;
    this.lastSig = sig;

    const buckets: number[][] = Array.from({ length: SUBMESH_COUNT }, () => []);
    const halfX = soil.sizeX * 0.5;
    const halfZ = soil.sizeZ * 0.5;
    const iyTop = soil.ny - 1;
    const iyShallowMin = Math.max(0, soil.ny - 5);
    const obstacles: readonly SolidObstacle[] = snapshot.obstacles;

    const pushProp = (
      sliceId: number,
      wx: number, wy: number, wz: number, scale: number,
      nrm: readonly [number, number, number],
      yaw: number, pitch: number, rollA: number,
      r: number, g: number, b: number, wet: number,
      seed: number, subVar: number, biomeLush: number, aoBias: number,
    ): void => {
      buckets[sliceId].push(
        wx, wy, wz, scale,
        nrm[0], nrm[1], nrm[2], yaw,
        pitch, rollA, r, g,
        b, wet, seed, subVar,
        biomeLush, aoBias,
      );
    };

    const isClearOfStipesAndObstacles = (wx: number, wz: number, extraMargin: number): boolean => {
      for (let b = 0; b < bodies.length; b++) {
        const minR = Math.max(0.015, bodies[b].stipeRadiusM + 0.0085) + extraMargin;
        if (Math.hypot(wx - bodies[b].baseX, wz - bodies[b].baseZ) < minR) {
          return false;
        }
      }
      for (let o = 0; o < obstacles.length; o++) {
        const obs = obstacles[o];
        if (obs.kind === 'stone' && Math.hypot(wx - obs.x, wz - obs.z) < obs.radius * 0.78 + extraMargin) {
          return false;
        }
      }
      return true;
    };

    const heroPositions: Array<[number, number]> = [];
    const gridN = 56;
    const spanX = soil.sizeX - 0.016;
    const spanZ = soil.sizeZ - 0.016;

    for (let gz = 0; gz < gridN; gz++) {
      for (let gx = 0; gx < gridN; gx++) {
        const cellIdx = gz * gridN + gx;
        const rx = (gx + 0.14 + 0.72 * hash1(cellIdx * 23 + 1)) / gridN;
        const rz = (gz + 0.14 + 0.72 * hash1(cellIdx * 23 + 2)) / gridN;
        const wx = (rx - 0.5) * spanX;
        const wz = (rz - 0.5) * spanZ;

        // Strictly clear of fruiting-body stipe bases (r >= 0.015 m)
        if (!isClearOfStipesAndObstacles(wx, wz, 0.0)) continue;

        const ix = Math.max(0, Math.min(soil.nx - 1, Math.floor((wx + halfX) / soil.dx)));
        const iz = Math.max(0, Math.min(soil.nz - 1, Math.floor((wz + halfZ) / soil.dx)));
        const vIdx = soil.voxelIndex(ix, iyTop, iz);
        const mat = soil.material[vIdx];
        const wet = clamp01(soil.theta[vIdx] / 0.55);

        // Detect shallow buried/adjacent WOOD, STONE, and SAND influence (captures buried oak logs & stone)
        let woodProx = mat === SoilMaterialId.WOOD ? 1.0 : 0.0;
        let stoneProx = mat === SoilMaterialId.STONE ? 1.0 : 0.0;
        let sandProx = mat === SoilMaterialId.SAND ? 1.0 : 0.0;
        for (let dz = -1; dz <= 1; dz++) {
          const nz = Math.max(0, Math.min(soil.nz - 1, iz + dz));
          for (let dx = -1; dx <= 1; dx++) {
            const nx = Math.max(0, Math.min(soil.nx - 1, ix + dx));
            const hWeight = (dx === 0 && dz === 0) ? 1.0 : 0.45;
            for (let iy = iyTop; iy >= iyShallowMin; iy--) {
              const m = soil.material[soil.voxelIndex(nx, iy, nz)];
              const dWeight = (1.0 - (iyTop - iy) * 0.19) * hWeight * 0.42;
              if (m === SoilMaterialId.WOOD) woodProx += dWeight;
              else if (m === SoilMaterialId.STONE) stoneProx += dWeight;
              else if (m === SoilMaterialId.SAND) sandProx += dWeight * 0.65;
            }
          }
        }
        woodProx = Math.min(1.0, woodProx);
        stoneProx = Math.min(1.0, stoneProx);
        sandProx = Math.min(1.0, sandProx);

        const wy = evalSoilSurfaceHeightM(wx, wz, soil, bodies);
        const nrm = evalSoilSurfaceNormal(wx, wz, soil, bodies);

        // Domain-warped low-frequency fBm biome clustering (normalized ~[0.1..0.9] contrast)
        const warpX = fbm2(wx * 5.6 + 2.3, wz * 5.6 - 1.7) * 0.14;
        const warpZ = fbm2(wx * 5.6 - 6.1, wz * 5.6 + 4.4) * 0.14;
        const px = wx + warpX;
        const pz = wz + warpZ;

        const mossField = clamp01(0.5 + fbm2(px * 6.8 + 11.4, pz * 6.8 + 3.2) * 2.55);
        const litterField = clamp01(0.5 + fbm2(px * 7.4 - 4.8, pz * 7.4 + 9.6) * 2.55);
        const gravelField = clamp01(0.5 + fbm2(px * 8.2 + 1.9, pz * 8.2 - 7.3) * 2.55);
        const hollowBoost = clamp01((0.0024 - wy) / 0.0060);
        const moistRel = Math.max(-0.14, Math.min(0.11, (wet - 0.58) * 0.28));

        const mossScore = mossField + moistRel + woodProx * 0.18 - sandProx * 0.50;
        const litterScore = litterField * 0.68 + hollowBoost * 0.36 - woodProx * 0.14 - sandProx * 0.35;

        const roll = hash1(cellIdx * 23 + 3);
        const roll2 = hash1(cellIdx * 23 + 4);
        const seed = hash1(cellIdx * 23 + 5);
        const tone = (hash1(cellIdx * 23 + 6) - 0.5) * 0.14;
        const yaw = hash1(cellIdx * 23 + 7) * Math.PI * 2;
        const pitch = (hash1(cellIdx * 23 + 8) - 0.5) * 0.14;
        const rollA = (hash1(cellIdx * 23 + 9) - 0.5) * 0.14;
        const sizeJ = hash1(cellIdx * 23 + 10);

        const dark = 1.0 - wet * 0.25;

        // 1. Sparse hero micro-landmarks: moss-draped twigs & curled oak bark arches
        const heroChance = woodProx > 0.22 ? 0.032 : mat === SoilMaterialId.HUMUS ? 0.011 : 0.0;
        if (
          buckets[5].length / 18 + buckets[6].length / 18 < 54
          && roll2 < heroChance
          && isClearOfStipesAndObstacles(wx, wz, 0.009)
        ) {
          let spaced = true;
          for (let h = 0; h < heroPositions.length; h++) {
            if (Math.hypot(wx - heroPositions[h][0], wz - heroPositions[h][1]) < 0.044) {
              spaced = false;
              break;
            }
          }
          if (spaced) {
            heroPositions.push([wx, wz]);
            const isBarkArch = roll > (woodProx > 0.22 ? 0.44 : 0.56);
            const sliceId = isBarkArch ? 6 : 5;
            const scale = (isBarkArch ? 0.0155 : 0.0175) + sizeJ * 0.007;
            const br = (isBarkArch ? 0.36 : 0.32) + tone * 0.45;
            const bg = (isBarkArch ? 0.23 : 0.21) + tone * 0.30;
            const bb = (isBarkArch ? 0.14 : 0.13) + tone * 0.18;
            pushProp(
              sliceId, wx, wy - 0.0003, wz, scale,
              nrm, yaw, pitch * 0.5, rollA * 0.5,
              br * dark, bg * dark, bb * dark, wet,
              seed, isBarkArch ? 1 : 0, clamp01((mossScore - 0.3) * 1.4), 1.0,
            );
            continue;
          }
        }

        // 2. SAND or strong STONE regions: clustered mineral pebbles & grit (+ moss pockets on stone)
        if (mat === SoilMaterialId.SAND || (mat === SoilMaterialId.STONE && mossScore < 0.60)) {
          const isStone = mat === SoilMaterialId.STONE;
          const gravelPocket = smoothstep(0.32, 0.72, gravelField);
          const densityGate = isStone ? (0.42 + 0.30 * gravelPocket) : (0.14 + 0.48 * gravelPocket);
          if (roll > densityGate) continue;

          if (isStone && roll < 0.20 && wet > 0.28) {
            const scale = 0.0048 + sizeJ * 0.0042;
            pushProp(
              1, wx, wy - 0.0003, wz, scale, nrm, yaw, pitch, rollA,
              (0.18 + tone * 0.35) * dark, (0.38 + tone * 0.55) * dark, (0.12 + tone * 0.2) * dark,
              wet, seed, 0, clamp01(mossScore), 1.0,
            );
          } else if (roll < densityGate - 0.06) {
            const scale = (isStone ? 0.0031 : 0.0020) + sizeJ * 0.0025;
            const warmSand = !isStone && roll2 > 0.42;
            const r = warmSand ? 0.64 + tone * 0.35 : 0.45 + tone * 0.35;
            const g = warmSand ? 0.54 + tone * 0.30 : 0.46 + tone * 0.35;
            const b = warmSand ? 0.39 + tone * 0.25 : 0.48 + tone * 0.35;
            pushProp(
              4, wx, wy - 0.0002, wz, scale, nrm, yaw, pitch, rollA,
              r * dark, g * dark, b * dark, wet, seed, 0, 0.0, 1.0,
            );
          } else {
            const scale = 0.0062 + sizeJ * 0.0034;
            pushProp(
              2, wx, wy + 0.0002, wz, scale, nrm, yaw, pitch, rollA,
              (0.42 + tone * 0.4) * dark, (0.31 + tone * 0.25) * dark, 0.14 * dark,
              wet, seed, 0, 0.0, 1.0,
            );
          }
          continue;
        }

        // 3. Rotting WOOD zones: bark flakes, moss patches & pine needles
        if (mat === SoilMaterialId.WOOD || woodProx > 0.24) {
          if (roll > 0.86) continue; // subtle gaps for exposed wood grain
          if (mossScore > 0.56 && roll < 0.46) {
            const scale = 0.0052 + sizeJ * 0.0046;
            pushProp(
              1, wx, wy - 0.0003, wz, scale, nrm, yaw, pitch, rollA,
              (0.18 + tone * 0.35) * dark, (0.41 + tone * 0.55) * dark, (0.12 + tone * 0.2) * dark,
              wet, seed, 0, clamp01((mossScore - 0.45) * 1.8), 1.0,
            );
          } else if (roll < 0.74) {
            const scale = 0.0066 + sizeJ * 0.0046;
            pushProp(
              3, wx, wy + 0.00015, wz, scale, nrm, yaw, pitch, rollA,
              (0.41 + tone * 0.45) * dark, (0.24 + tone * 0.30) * dark, (0.12 + tone * 0.15) * dark,
              wet, seed, 1, clamp01(mossScore), 1.0,
            );
          } else {
            const scale = 0.0070 + sizeJ * 0.0038;
            pushProp(
              2, wx, wy + 0.0002, wz, scale, nrm, yaw, pitch, rollA,
              (0.38 + tone * 0.4) * dark, (0.25 + tone * 0.3) * dark, 0.12 * dark,
              wet, seed, 0, 0.2, 1.0,
            );
          }
          continue;
        }

        // 4. Moss-island edge micro-rosettes (fern / liverwort sprouts nestled along moss margins)
        const onMossEdge = mossScore >= 0.50 && mossScore <= 0.63;
        if (
          onMossEdge
          && roll2 < 0.21
          && buckets[7].length / 18 < 64
          && isClearOfStipesAndObstacles(wx, wz, 0.004)
        ) {
          const scale = 0.0070 + sizeJ * 0.0044;
          pushProp(
            7, wx, wy, wz, scale, nrm, yaw, pitch * 0.6, rollA * 0.6,
            (0.18 + tone * 0.3) * dark, (0.43 + tone * 0.45) * dark, (0.12 + tone * 0.2) * dark,
            wet, seed, 0, 0.85, 1.0,
          );
          continue;
        }

        // 5. Lush clustered pillow-moss islands in moist humus areas
        if (mossScore >= 0.59 && mat === SoilMaterialId.HUMUS) {
          const islandCore = smoothstep(0.59, 0.84, mossScore);
          if (roll > 0.72 + 0.24 * islandCore) continue;
          if (roll2 < 0.12 * (1.0 - islandCore)) {
            // Occasional leaf/needle caught on moss island fringe
            const isLeaf = roll < 0.55;
            const scale = (isLeaf ? 0.0080 : 0.0068) + sizeJ * 0.0036;
            pushProp(
              isLeaf ? 0 : 2, wx, wy + 0.00025, wz, scale, nrm, yaw, pitch, rollA,
              (isLeaf ? 0.54 + tone * 0.5 : 0.36 + tone * 0.35) * dark,
              (isLeaf ? 0.29 + tone * 0.35 : 0.25 + tone * 0.25) * dark,
              (isLeaf ? 0.11 : 0.12) * dark,
              wet, seed, 0, islandCore, 1.0,
            );
          } else {
            const scale = 0.0050 + (0.42 + 0.58 * islandCore) * (0.0024 + sizeJ * 0.0044);
            pushProp(
              1, wx, wy - 0.00035, wz, scale, nrm, yaw, pitch, rollA,
              (0.18 + tone * 0.35) * dark,
              (0.40 + tone * 0.55 + islandCore * 0.05) * dark,
              (0.11 + tone * 0.20) * dark,
              wet, seed, 0, islandCore, 1.0,
            );
          }
          continue;
        }

        // 6. Packed russet/amber oak-leaf litter pockets in hollows
        if (litterScore >= 0.51) {
          const pocketCore = smoothstep(0.51, 0.78, litterScore);
          if (roll > 0.68 + 0.24 * pocketCore) continue;

          if (roll2 < 0.74) {
            // Layered oak leaves in russet, amber, or dark weathered brown
            const paletteRoll = hash1(cellIdx * 23 + 11);
            let lr = 0.52, lg = 0.27, lb = 0.10;
            if (paletteRoll > 0.66) {
              lr = 0.64; lg = 0.38; lb = 0.12; // warm amber
            } else if (paletteRoll > 0.32) {
              lr = 0.52; lg = 0.25; lb = 0.10; // russet oak
            } else {
              lr = 0.36; lg = 0.20; lb = 0.10; // weathered umber
            }
            const scale = 0.0080 + sizeJ * 0.0052;
            pushProp(
              0, wx, wy + 0.00042 + pocketCore * 0.00028, wz, scale, nrm, yaw, pitch * 1.15, rollA * 1.15,
              (lr + tone * 0.45) * dark, (lg + tone * 0.30) * dark, (lb + tone * 0.15) * dark,
              wet, seed, 0, 0.2, 1.0,
            );
          } else if (roll2 < 0.92) {
            // Conifer needle
            const scale = 0.0068 + sizeJ * 0.0040;
            pushProp(
              2, wx, wy + 0.0002, wz, scale, nrm, yaw, pitch, rollA,
              (0.40 + tone * 0.35) * dark, (0.27 + tone * 0.25) * dark, 0.12 * dark,
              wet, seed, 0, 0.1, 1.0,
            );
          } else {
            // Occasional bark sliver in leaf pocket
            const scale = 0.0062 + sizeJ * 0.0038;
            pushProp(
              3, wx, wy + 0.0001, wz, scale, nrm, yaw, pitch, rollA,
              (0.36 + tone * 0.35) * dark, (0.22 + tone * 0.25) * dark, 0.12 * dark,
              wet, seed, 1, 0.1, 1.0,
            );
          }
          continue;
        }

        // 7. Exposed bare damp dark-humus troughs between islands (keep ~90% bare so dark soil shows cleanly)
        if (stoneProx > 0.24 && roll < 0.26) {
          const scale = 0.0024 + sizeJ * 0.0020;
          pushProp(
            4, wx, wy - 0.0002, wz, scale, nrm, yaw, pitch, rollA,
            (0.44 + tone * 0.3) * dark, (0.45 + tone * 0.3) * dark, (0.47 + tone * 0.3) * dark,
            wet, seed, 0, 0.0, 1.0,
          );
        } else if (roll < 0.10) {
          const isNeedle = roll2 > 0.35;
          const scale = (isNeedle ? 0.0062 : 0.0022) + sizeJ * 0.0024;
          pushProp(
            isNeedle ? 2 : 4, wx, wy + (isNeedle ? 0.00015 : -0.0002), wz, scale, nrm, yaw, pitch, rollA,
            (isNeedle ? 0.35 + tone * 0.3 : 0.42 + tone * 0.25) * dark,
            (isNeedle ? 0.23 + tone * 0.2 : 0.42 + tone * 0.25) * dark,
            (isNeedle ? 0.12 : 0.43 + tone * 0.25) * dark,
            wet, seed, 0, 0.0, 1.0,
          );
        }
      }
    }

    let cursor = 0;
    const dst = this.instScratch;
    const qTmp = new Float32Array(4);
    for (let k = 0; k < SUBMESH_COUNT; k++) {
      const src = buckets[k];
      let c = 0;
      for (let i = 0; i < src.length && cursor < MAX_INSTANCES; i += 18) {
        const o = cursor * FLOATS_PER_INST;
        dst[o] = src[i];
        dst[o + 1] = src[i + 1];
        dst[o + 2] = src[i + 2];
        dst[o + 3] = src[i + 3];
        writeSlopeAlignedQuat(
          qTmp, 0,
          [src[i + 4], src[i + 5], src[i + 6]],
          src[i + 7], src[i + 8], src[i + 9],
        );
        dst[o + 4] = qTmp[0];
        dst[o + 5] = qTmp[1];
        dst[o + 6] = qTmp[2];
        dst[o + 7] = qTmp[3];
        dst[o + 8] = src[i + 10];
        dst[o + 9] = src[i + 11];
        dst[o + 10] = src[i + 12];
        dst[o + 11] = src[i + 13];
        dst[o + 12] = src[i + 14];
        dst[o + 13] = src[i + 15];
        dst[o + 14] = src[i + 16];
        dst[o + 15] = src[i + 17];
        cursor++;
        c++;
      }
      this.counts[k] = c;
    }

    if (cursor > 0) {
      this.device.queue.writeBuffer(
        this.instBuf,
        0,
        dst.buffer,
        dst.byteOffset,
        cursor * FLOATS_PER_INST * 4,
      );
    }
  }

  draw(pass: GPURenderPassEncoder, litBindGroup: GPUBindGroup): void {
    let total = 0;
    for (let k = 0; k < SUBMESH_COUNT; k++) {
      total += this.counts[k];
    }
    if (total === 0) return;

    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, litBindGroup);
    pass.setVertexBuffer(0, this.geomVertBuf);
    pass.setVertexBuffer(1, this.instBuf);
    pass.setIndexBuffer(this.geomIdxBuf, 'uint16');

    let firstInst = 0;
    for (let k = 0; k < SUBMESH_COUNT; k++) {
      const cnt = this.counts[k];
      if (cnt > 0) {
        const s = this.slices[k];
        pass.drawIndexed(s.indexCount, cnt, s.firstIndex, 0, firstInst);
        firstInst += cnt;
      }
    }
  }

  dispose(): void {
    this.geomVertBuf.destroy();
    this.geomIdxBuf.destroy();
    this.instBuf.destroy();
  }
}
