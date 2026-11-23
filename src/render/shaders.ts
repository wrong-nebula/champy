// ============================================================================
// SPOROCARP — Hand-Crafted WGSL (WebGPU HDR) & GLSL ES 3.00 (WebGL2 Fallback)
// Implements:
//   0. Studio Gradient Backdrop
//   1. Domain-Warped Voronoi/POM Soil Cross-Section, Razor Slice & Fast-SDF GI
//   2. Depth-Cued 3D Branching Mycelial Network (HDR linear glow)
//   3. Botanical Species PBR + Waxy Chitin SSS + PCF Shadows + Fast-SDF-AO/GI
//   4. Solid Obstacles (Twigs/Stones) & Physical Glass w/ Humidity Condensation
//   5. Airborne Spore Cloud (WebGL2 fallback / non-splat path)
//   6. Dressaire Evaporative Buoyant Wind Streamlines
// ============================================================================

export const WGSL_COMMON = /* wgsl */ `
struct SceneUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,       // xyz = eye, w = timeSec
  lightDir: vec4<f32>,        // xyz = normalized light dir, w = tempC
  soilBoundsMin: vec4<f32>,   // xyz = minX, minY, minZ, w = dx
  soilBoundsMax: vec4<f32>,   // xyz = maxX, maxY, maxZ, w = xrayMode (0..5)
  sliceParams: vec4<f32>,     // x = enabled, y = axis (0=x,1=z), z = worldCoord, w = auxAlpha
  sssColor: vec4<f32>,        // rgb = species sssColor, w = showMycelium
  sporePrintColor: vec4<f32>, // rgb = spore color, w = selectedBodyId
  soilDims: vec4<f32>,        // x = nx, y = ny, z = nz, w = humidity
  sdfVolMin: vec4<f32>,       // xyz = sdf min, w = sdfVoxelSize
  sdfVolMax: vec4<f32>,       // xyz = sdf max, w = giEnabled (1.0 = active)
  camRight: vec4<f32>,        // xyz = camera right, w = viewportWidthPx
  camUp: vec4<f32>,           // xyz = camera up, w = viewportHeightPx
  lightViewProj: mat4x4<f32>, // directional 2048x2048 shadow matrix
  speciesExtra: vec4<f32>,  // x = speciesIdx (0..4), y = shadowEnabled, z = near, w = far
  postParams: vec4<f32>,    // x = focusDistM, y = dofStrength, z = bloomStrength, w = rimStrength
  padExtra: vec4<f32>,
  biolumInfo: vec4<f32>,    // x = biolumIntensity 0..2.5 (bodies+foxfire), y = nocturnalFactor 0..1, z = meanO2Factor, w = bodyGlow 0..2.5
};

fn isClippedBySlice(p: vec3<f32>, sliceParams: vec4<f32>) -> bool {
  if (sliceParams.x < 0.5) {
    return false;
  }
  if (sliceParams.y < 0.5) {
    return p.x > sliceParams.z + 0.0005;
  }
  return p.z > sliceParams.z + 0.0005;
}

fn hash31(p: vec3<f32>) -> f32 {
  var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
  q += vec3<f32>(dot(q, q.yxz + vec3<f32>(33.33)));
  return fract((q.x + q.y) * q.z);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
  var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yxz + vec3<f32>(33.33));
  return fract((q.xxy + q.yxx) * q.zyx);
}

fn valueNoise3D(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (vec3<f32>(3.0) - 2.0 * f);
  let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
  let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
  let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
  let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
  let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
  let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
  let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
  let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
  let nx00 = mix(n000, n100, u.x);
  let nx10 = mix(n010, n110, u.x);
  let nx01 = mix(n001, n101, u.x);
  let nx11 = mix(n011, n111, u.x);
  let nxy0 = mix(nx00, nx10, u.y);
  let nxy1 = mix(nx01, nx11, u.y);
  return mix(nxy0, nxy1, u.z);
}

fn fbm3D(p: vec3<f32>) -> f32 {
  var v = 0.0;
  var amp = 0.5;
  var freq = 1.0;
  for (var i = 0; i < 3; i = i + 1) {
    v += amp * valueNoise3D(p * freq);
    freq *= 2.07;
    amp *= 0.5;
  }
  return v;
}

// Cellular 3D Voronoi returning vec2(F1, F2 - F1) for soil pores & morel pits
fn voronoi3D(p: vec3<f32>) -> vec2<f32> {
  let n = floor(p);
  let f = fract(p);
  var f1 = 8.0;
  var f2 = 8.0;
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let g = vec3<f32>(f32(dx), f32(dy), f32(dz));
        let o = hash33(n + g);
        let r = g + o - f;
        let d = dot(r, r);
        if (d < f1) {
          f2 = f1;
          f1 = d;
        } else if (d < f2) {
          f2 = d;
        }
      }
    }
  }
  let s1 = sqrt(f1);
  let s2 = sqrt(f2);
  return vec2<f32>(s1, s2 - s1);
}

fn srgbToLinear(c: vec3<f32>) -> vec3<f32> {
  return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn colormapViridis(tIn: f32) -> vec3<f32> {
  let t = clamp(tIn, 0.0, 1.0);
  let c0 = vec3<f32>(0.2777, 0.0054, 0.3340);
  let c1 = vec3<f32>(0.1050, 1.4046, 1.3845);
  let c2 = vec3<f32>(-0.3308, 0.2148, 0.0950);
  let c3 = vec3<f32>(-4.6342, -5.7991, -19.3324);
  let c4 = vec3<f32>(6.2282, 14.1799, 56.6905);
  let c5 = vec3<f32>(4.7763, -13.7451, -65.3530);
  let c6 = vec3<f32>(-5.4354, 4.6458, 26.3124);
  return srgbToLinear(clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3<f32>(0.0), vec3<f32>(1.0)));
}

fn colormapInferno(tIn: f32) -> vec3<f32> {
  let t = clamp(tIn, 0.0, 1.0);
  let c0 = vec3<f32>(0.0002, 0.0016, 0.0139);
  let c1 = vec3<f32>(0.1065, 0.0639, 0.2297);
  let c2 = vec3<f32>(0.4240, 0.0920, 0.4320);
  let c3 = vec3<f32>(0.7356, 0.2159, 0.3302);
  let c4 = vec3<f32>(0.9554, 0.4876, 0.1072);
  let c5 = vec3<f32>(0.9883, 0.9983, 0.6449);
  let s = t * 5.0;
  var c = c5;
  if (s < 1.0) { c = mix(c0, c1, s); }
  else if (s < 2.0) { c = mix(c1, c2, s - 1.0); }
  else if (s < 3.0) { c = mix(c2, c3, s - 2.0); }
  else if (s < 4.0) { c = mix(c3, c4, s - 3.0); }
  else { c = mix(c4, c5, s - 4.0); }
  return srgbToLinear(c);
}

fn colormapCoolwarm(tIn: f32) -> vec3<f32> {
  let t = clamp(tIn, 0.0, 1.0);
  let cool = vec3<f32>(0.16, 0.46, 0.88);
  let mid = vec3<f32>(0.84, 0.87, 0.90);
  let warm = vec3<f32>(0.92, 0.34, 0.16);
  if (t < 0.5) {
    return srgbToLinear(mix(cool, mid, t * 2.0));
  }
  return srgbToLinear(mix(mid, warm, (t - 0.5) * 2.0));
}

fn evalCanopyGobo(worldPos: vec3<f32>, lightDir: vec3<f32>, padExtra: vec4<f32>) -> f32 {
  if (padExtra.z <= 0.01) {
    return 1.0;
  }
  let l = normalize(lightDir + vec3<f32>(0.0, 1e-5, 0.0));
  let lUp = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(l.y) > 0.94);
  let tU = normalize(cross(lUp, l));
  let tV = cross(l, tU);
  let pCanopy = vec2<f32>(dot(worldPos, tU), dot(worldPos, tV));
  let drift = vec2<f32>(padExtra.y * 0.032, padExtra.y * 0.019);
  let q = pCanopy * 10.5 + drift;
  let nCoarse = valueNoise3D(vec3<f32>(q * 0.65, padExtra.y * 0.04));
  let nMid = valueNoise3D(vec3<f32>(q * 1.85 + vec2<f32>(3.7, -2.1), padExtra.y * 0.07));
  let nFine = valueNoise3D(vec3<f32>(q * 4.2 - drift * 1.6, 1.3));
  let gap = smoothstep(0.28, 0.74, nCoarse * 0.56 + nMid * 0.30 + nFine * 0.14);
  let gobo = mix(0.30, 1.15, gap);
  return mix(1.0, gobo, clamp(padExtra.z, 0.0, 1.0));
}
`;

export function makeWgslSdfGiHelpers(
  sdfTexBinding: number,
  sampName: string,
  radTexBinding: number,
  declareSamplerBinding?: number,
): string {
  const sampDecl = declareSamplerBinding !== undefined
    ? `@group(0) @binding(${declareSamplerBinding}) var ${sampName}: sampler;\n`
    : '';
  return /* wgsl */ `
@group(0) @binding(${sdfTexBinding}) var sdfVolTex: texture_3d<f32>;
${sampDecl}@group(0) @binding(${radTexBinding}) var radVolTex: texture_3d<f32>;

fn sdfWorldToUvw(p: vec3<f32>) -> vec3<f32> {
  return (p - u.sdfVolMin.xyz) / max(u.sdfVolMax.xyz - u.sdfVolMin.xyz, vec3<f32>(1e-4));
}

fn sampleSdfDist(p: vec3<f32>) -> f32 {
  let uvw = sdfWorldToUvw(p);
  if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
    return 0.25;
  }
  return textureSampleLevel(sdfVolTex, ${sampName}, uvw, 0.0).x;
}

fn sdfGradNormal(p: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let eps = max(u.sdfVolMin.w * 0.9, 0.005);
  let dx = sampleSdfDist(p + vec3<f32>(eps, 0.0, 0.0)) - sampleSdfDist(p - vec3<f32>(eps, 0.0, 0.0));
  let dy = sampleSdfDist(p + vec3<f32>(0.0, eps, 0.0)) - sampleSdfDist(p - vec3<f32>(0.0, eps, 0.0));
  let dz = sampleSdfDist(p + vec3<f32>(0.0, 0.0, eps)) - sampleSdfDist(p - vec3<f32>(0.0, 0.0, eps));
  let g = vec3<f32>(dx, dy, dz);
  let len = length(g);
  return select(fallback, g / len, len > 1e-5);
}

fn evalFastSdfAO(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
  if (u.sdfVolMax.w < 0.5) {
    return vec4<f32>(n, 1.0);
  }
  let h = array<f32, 4>(0.008, 0.020, 0.044, 0.090);
  let w = array<f32, 4>(0.42, 0.28, 0.18, 0.12);
  var occ = 0.0;
  for (var i = 0; i < 4; i = i + 1) {
    let d = sampleSdfDist(p + n * h[i]);
    occ += w[i] * max(0.0, h[i] - d) / h[i];
  }
  let ao = clamp(1.0 - 0.90 * occ, 0.14, 1.0);
  let gradN = sdfGradNormal(p + n * 0.012, n);
  let bentN = normalize(mix(n, gradN, (1.0 - ao) * 0.65));
  return vec4<f32>(bentN, ao);
}

fn evalSdfSoftShadow(p: vec3<f32>, l: vec3<f32>) -> f32 {
  if (u.sdfVolMax.w < 0.5) {
    return 1.0;
  }
  let vSize = max(u.sdfVolMin.w, 0.007);
  let ro = p + l * (vSize * 1.15);
  var res = 1.0;
  var t = vSize * 1.25;
  for (var i = 0; i < 12; i = i + 1) {
    let d = sampleSdfDist(ro + l * t);
    if (d < vSize * 0.12) {
      return 0.08;
    }
    res = min(res, clamp(8.5 * d / t, 0.08, 1.0));
    t += clamp(d, vSize * 0.70, 0.042);
    if (t > 0.44) {
      break;
    }
  }
  return res;
}

fn evalSdfGI(p: vec3<f32>, n: vec3<f32>, bentN: vec3<f32>) -> vec3<f32> {
  if (u.sdfVolMax.w < 0.5) {
    return vec3<f32>(0.0);
  }
  let upRef = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(bentN.y) > 0.9);
  let t1 = normalize(cross(upRef, bentN));
  let t2 = cross(bentN, t1);

  let uv0 = clamp(sdfWorldToUvw(p + bentN * 0.024), vec3<f32>(0.01), vec3<f32>(0.99));
  let uv1 = clamp(sdfWorldToUvw(p + bentN * 0.062), vec3<f32>(0.01), vec3<f32>(0.99));

  let dA = normalize(bentN * 0.75 + t1 * 0.66);
  let dB = normalize(bentN * 0.75 - t1 * 0.33 + t2 * 0.57);
  let dC = normalize(bentN * 0.75 - t1 * 0.33 - t2 * 0.57);
  let uvA = clamp(sdfWorldToUvw(p + dA * 0.045), vec3<f32>(0.01), vec3<f32>(0.99));
  let uvB = clamp(sdfWorldToUvw(p + dB * 0.045), vec3<f32>(0.01), vec3<f32>(0.99));
  let uvC = clamp(sdfWorldToUvw(p + dC * 0.045), vec3<f32>(0.01), vec3<f32>(0.99));

  let c0 = textureSampleLevel(radVolTex, ${sampName}, uv0, 0.0).rgb;
  let c1 = textureSampleLevel(radVolTex, ${sampName}, uv1, 0.0).rgb;
  let cA = textureSampleLevel(radVolTex, ${sampName}, uvA, 0.0).rgb;
  let cB = textureSampleLevel(radVolTex, ${sampName}, uvB, 0.0).rgb;
  let cC = textureSampleLevel(radVolTex, ${sampName}, uvC, 0.0).rgb;

  let normalWeight = 0.7 + 0.3 * max(dot(n, bentN), 0.0);
  return (c0 * 0.34 + c1 * 0.22 + (cA + cB + cC) * 0.146) * normalWeight;
}
`;
}

export function makeWgslShadowHelpers(
  shadowMapBinding: number,
  shadowSampBinding: number,
): string {
  return /* wgsl */ `
@group(0) @binding(${shadowMapBinding}) var shadowMapTex: texture_depth_2d;
@group(0) @binding(${shadowSampBinding}) var shadowSamp: sampler_comparison;

fn evalPcfShadow(worldPos: vec3<f32>, n: vec3<f32>) -> f32 {
  if (u.speciesExtra.y < 0.5) {
    return 1.0;
  }
  let biasPos = worldPos + n * 0.0018;
  let lp = u.lightViewProj * vec4<f32>(biasPos, 1.0);
  let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
  let refZ = lp.z - 0.00055;
  let inBounds = all(uv >= vec2<f32>(0.005)) && all(uv <= vec2<f32>(0.995)) && refZ > 0.0 && refZ < 1.0;
  let safeUv = clamp(uv, vec2<f32>(0.005), vec2<f32>(0.995));
  let safeZ = clamp(refZ, 0.0, 0.9999);
  let texel = 1.0 / 2048.0;
  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let off = vec2<f32>(f32(dx), f32(dy)) * texel * 1.3;
      sum += textureSampleCompareLevel(shadowMapTex, shadowSamp, safeUv + off, safeZ);
    }
  }
  return select(1.0, sum / 9.0, inBounds);
}
`;
}

export function makeWgslIblHelpers(
  cubeBinding: number,
  sampBinding: number,
): string {
  return /* wgsl */ `
@group(0) @binding(${cubeBinding}) var iblCubeTex: texture_cube<f32>;
@group(0) @binding(${sampBinding}) var iblSamp: sampler;

fn evalStudioIbl(reflDir: vec3<f32>, roughness: f32, iblStrength: f32) -> vec3<f32> {
  if (iblStrength <= 0.001) {
    return vec3<f32>(0.0);
  }
  let lod = clamp(roughness, 0.0, 1.0) * 4.0;
  return textureSampleLevel(iblCubeTex, iblSamp, normalize(reflDir), lod).rgb * iblStrength;
}
`;
}

// ----------------------------------------------------------------------------
// 0. STUDIO GRADIENT BACKDROP SHADER (WGSL HDR)
// ----------------------------------------------------------------------------
export const BACKDROP_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;

struct BackVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> BackVertOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var out: BackVertOut;
  out.clipPos = vec4<f32>(pos[vid], 0.99999, 1.0);
  out.uv = pos[vid] * 0.5 + 0.5;
  return out;
}

@fragment
fn fs_main(in: BackVertOut) -> @location(0) vec4<f32> {
  let d = length((in.uv - vec2<f32>(0.5, 0.45)) * vec2<f32>(1.18, 1.0));
  let glow = exp(-d * d * 2.6);
  let mood = u.padExtra.x;

  // 0: Botanical dark blue-slate studio
  let edgeStudio = vec3<f32>(0.011, 0.015, 0.023);
  let centerStudio = vec3<f32>(0.042, 0.060, 0.084);

  // 1: Warm deep woodland dusk canopy
  let edgeCanopy = vec3<f32>(0.010, 0.014, 0.011);
  let centerCanopy = vec3<f32>(0.044, 0.052, 0.034);

  // 2: Ultra-dark dramatic obsidian dark-field spotlight
  let edgeBacklit = vec3<f32>(0.0025, 0.0030, 0.0042);
  let centerBacklit = vec3<f32>(0.018, 0.022, 0.030);
  let spotGlow = exp(-d * d * 4.4);

  // 3: Deep midnight forest indigo nocturnal gradient
  let edgeNocturnal = vec3<f32>(0.002, 0.004, 0.009);
  let centerNocturnal = vec3<f32>(0.008, 0.018, 0.032);

  let isNocturnal = smoothstep(2.4, 2.9, mood);
  let isCanopy = smoothstep(0.5, 1.0, mood) * (1.0 - smoothstep(1.2, 1.8, mood));
  let isBacklit = smoothstep(1.2, 1.8, mood) * (1.0 - isNocturnal);

  var edge = mix(edgeStudio, edgeCanopy, isCanopy);
  edge = mix(edge, edgeBacklit, isBacklit);
  edge = mix(edge, edgeNocturnal, isNocturnal);
  var center = mix(centerStudio, centerCanopy, isCanopy);
  center = mix(center, centerBacklit, isBacklit);
  center = mix(center, centerNocturnal, isNocturnal);

  let g = mix(glow, spotGlow, isBacklit);
  return vec4<f32>(mix(edge, center, g), 1.0);
}
`;

// ----------------------------------------------------------------------------
// 1. SOIL CROSS-SECTION, HEIGHTFIELD & RAZOR SLICE SHADER (WGSL HDR)
// ----------------------------------------------------------------------------
export const SOIL_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var soilTex: texture_3d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var sporePrintTex: texture_2d<f32>;
${makeWgslSdfGiHelpers(4, 'linearSampler', 5)}
${makeWgslShadowHelpers(6, 7)}
${makeWgslIblHelpers(8, 9)}

struct SoilVertIn {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) flags: f32, // 0 = outer soil, 1 = razor slice plane, 2 = plinth
};

struct SoilVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) flags: f32,
};

@vertex
fn vs_main(v: SoilVertIn) -> SoilVertOut {
  var out: SoilVertOut;
  out.clipPos = u.viewProj * vec4<f32>(v.pos, 1.0);
  out.worldPos = v.pos;
  out.worldNormal = v.normal;
  out.flags = v.flags;
  return out;
}

fn soilHeightAt(p: vec3<f32>) -> f32 {
  let f = fbm3D(p * 110.0);
  let v = voronoi3D(p * 185.0);
  return f * 0.55 + (1.0 - smoothstep(0.0, 0.65, v.x)) * 0.45;
}

fn soilMaterialColor(
  matId: i32,
  worldPos: vec3<f32>,
  wetness: f32,
  o2Norm: f32,
  isTop: bool,
  vor: vec2<f32>,
  f3: f32,
) -> vec3<f32> {
  let gFine = valueNoise3D(worldPos * 210.0);
  let gStrata = valueNoise3D(vec3<f32>(worldPos.x * 16.0, worldPos.y * 85.0, worldPos.z * 16.0));
  let pores = smoothstep(0.05, 0.55, vor.x);

  var dryCol = vec3<f32>(0.24, 0.17, 0.11);
  var wetCol = vec3<f32>(0.08, 0.055, 0.035);

  if (matId == 0) {
    // Sand: crisp high-frequency quartz grit grains
    let qGrain = valueNoise3D(worldPos * 520.0);
    let qSparkle = hash31(floor(worldPos * 560.0));
    let speck = (qGrain - 0.5) * 0.24 + (qSparkle - 0.5) * 0.20 + (vor.x - 0.45) * 0.16 + (f3 - 0.5) * 0.08;
    dryCol = vec3<f32>(0.68, 0.56, 0.39) + vec3<f32>(speck);
    wetCol = vec3<f32>(0.38, 0.30, 0.19) + vec3<f32>(speck * 0.72);
  } else if (matId == 1) {
    // Forest Silt-Loam: stratified earth with micro-aggregates
    let band = (gStrata - 0.5) * 0.12 + (f3 - 0.5) * 0.10;
    dryCol = vec3<f32>(0.40, 0.30, 0.21) * (0.88 + 0.24 * pores) + vec3<f32>(band);
    wetCol = vec3<f32>(0.26, 0.19, 0.13) * (0.88 + 0.26 * pores) + vec3<f32>(band * 0.75);
  } else if (matId == 2) {
    // Heavy Clay: dense terracotta-slate slicks with mineral cracking
    let crack = smoothstep(0.0, 0.09, vor.y);
    let slick = (f3 - 0.5) * 0.08;
    dryCol = (vec3<f32>(0.48, 0.32, 0.25) + vec3<f32>(slick)) * (0.84 + 0.16 * crack);
    wetCol = (vec3<f32>(0.33, 0.22, 0.17) + vec3<f32>(slick * 0.70)) * (0.84 + 0.18 * crack);
  } else if (matId == 3) {
    // Forest Humus / Litter: rich dark organic mulch + high-frequency forest moss micro-mottling
    let litter = (gFine - 0.5) * 0.10 + (f3 - 0.5) * 0.08;
    dryCol = vec3<f32>(0.26, 0.19, 0.13) * (0.84 + 0.28 * pores) + vec3<f32>(litter);
    wetCol = vec3<f32>(0.15, 0.11, 0.075) * (0.84 + 0.28 * pores) + vec3<f32>(litter * 0.55);
    if (isTop) {
      let fineMoss = fbm3D(worldPos * 145.0) * 0.65 + gFine * 0.35;
      let moss = smoothstep(0.42, 0.78, fineMoss);
      dryCol = mix(dryCol, vec3<f32>(0.24, 0.42, 0.15), moss * 0.62);
      wetCol = mix(wetCol, vec3<f32>(0.16, 0.30, 0.10), moss * 0.68);
    }
  } else if (matId == 4) {
    // Rotting Oak Log: concentric heartwood growth rings + fibrous grain
    let rings = sin(length(worldPos.yz * 76.0 + vec2<f32>(3.0, 5.0)) + f3 * 4.8) * 0.5 + 0.5;
    let fiber = valueNoise3D(vec3<f32>(worldPos.x * 24.0, worldPos.y * 165.0, worldPos.z * 165.0));
    let woodVar = (rings * 0.58 + fiber * 0.42 - 0.5) * 0.18;
    dryCol = vec3<f32>(0.46, 0.29, 0.16) + vec3<f32>(woodVar);
    wetCol = vec3<f32>(0.27, 0.16, 0.09) + vec3<f32>(woodVar * 0.65);
  } else {
    // Impermeable Stone: speckled granite + moss cap near surface
    let grain = (gFine - 0.5) * 0.14 + (vor.y - 0.2) * 0.12;
    dryCol = vec3<f32>(0.36, 0.38, 0.40) + vec3<f32>(grain);
    wetCol = vec3<f32>(0.21, 0.23, 0.25) + vec3<f32>(grain * 0.7);
    if (worldPos.y > -0.025) {
      let moss = smoothstep(0.45, 0.76, f3);
      dryCol = mix(dryCol, vec3<f32>(0.11, 0.23, 0.08), moss * 0.62);
    }
  }

  var col = mix(dryCol, wetCol, clamp(wetness, 0.0, 1.0));
  let hypoxia = smoothstep(0.32, 0.05, o2Norm) * step(f32(matId), 4.5);
  col = mix(col, vec3<f32>(0.05, 0.10, 0.15), hypoxia * 0.35);
  return col;
}

@fragment
fn fs_main(in: SoilVertOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4<f32> {
  // Obsidian laboratory plinth underneath terrarium
  if (in.flags > 1.5) {
    let n = normalize(in.worldNormal);
    let l = normalize(u.lightDir.xyz);
    let v = normalize(u.cameraPos.xyz - in.worldPos);
    let h = normalize(l + v);
    let ndl = max(dot(n, l), 0.0);
    let spec = pow(max(dot(n, h), 0.0), 48.0) * 0.06;
    let base = srgbToLinear(vec3<f32>(0.055, 0.072, 0.095));
    let noctPlinth = max(clamp(u.biolumInfo.y, 0.0, 1.0), smoothstep(2.4, 2.9, u.padExtra.x));
    let moonTint = mix(vec3<f32>(1.0), vec3<f32>(0.42, 0.56, 0.86), noctPlinth);
    let plinthCol = (base * (0.45 + 0.75 * ndl) + vec3<f32>(spec)) * moonTint * mix(1.0, 0.09, noctPlinth);
    return vec4<f32>(plinthCol, 1.0);
  }

  let isSlice = in.flags > 0.5;
  if (!isSlice && isClippedBySlice(in.worldPos, u.sliceParams)) {
    discard;
  }

  let span = max(u.soilBoundsMax.xyz - u.soilBoundsMin.xyz, vec3<f32>(1e-4));
  let normPos = clamp((in.worldPos - u.soilBoundsMin.xyz) / span, vec3<f32>(0.0), vec3<f32>(0.9999));
  let uvw = vec3<f32>(normPos.x, normPos.z, normPos.y);
  let dimsF = vec3<f32>(u.soilDims.x, u.soilDims.z, u.soilDims.y);
  let dimsI = vec3<i32>(i32(u.soilDims.x), i32(u.soilDims.z), i32(u.soilDims.y));

  let smoothSample = textureSampleLevel(soilTex, linearSampler, uvw, 0.0);
  let wetness = smoothSample.g;
  let xrayVal = smoothSample.b;
  let o2Norm = smoothSample.a;

  let xrayMode = i32(floor(u.soilBoundsMax.w + 0.5));
  let rawN = normalize(in.worldNormal);
  let n = select(rawN, -rawN, isSlice && !frontFacing);
  let isTop = n.y > 0.5;
  let v = normalize(u.cameraPos.xyz - in.worldPos);

  // Subtle 3-step tangent-space Parallax Occlusion Mapping (POM) on natural vertical cutaway walls
  var samplePos = in.worldPos;
  if (!isTop && !isSlice && xrayMode == 0) {
    let upRef = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t1 = normalize(cross(upRef, n));
    let t2 = cross(n, t1);
    let viewTan = vec3<f32>(dot(v, t1), dot(v, t2), max(dot(v, n), 0.25));
    let pomDir = (t1 * viewTan.x + t2 * viewTan.y) / viewTan.z;
    var hWalk = 0.0;
    for (var step = 0; step < 3; step = step + 1) {
      let pStep = in.worldPos - pomDir * (hWalk * 0.0020);
      hWalk = mix(hWalk, 1.0 - soilHeightAt(pStep), 0.55);
    }
    samplePos = in.worldPos - pomDir * (hWalk * 0.0018);
  }

  // Domain-warped 8-corner trilinear material blending
  let w1 = valueNoise3D(samplePos * 26.0 + vec3<f32>(1.7, 3.1, 5.3));
  let w2 = valueNoise3D(samplePos * 26.0 + vec3<f32>(11.2, 7.4, 2.9));
  let w3 = valueNoise3D(samplePos * 26.0 + vec3<f32>(4.6, 9.8, 13.1));
  let warp = (vec3<f32>(w1, w2, w3) - vec3<f32>(0.5)) * (0.95 / dimsF);
  let uvwMat = clamp(uvw + warp, vec3<f32>(0.0), vec3<f32>(0.9999));

  let gridPos = uvwMat * dimsF - vec3<f32>(0.5);
  let i0 = clamp(vec3<i32>(floor(gridPos)), vec3<i32>(0), dimsI - vec3<i32>(1));
  let i1 = min(i0 + vec3<i32>(1), dimsI - vec3<i32>(1));
  let f = smoothstep(vec3<f32>(0.0), vec3<f32>(1.0), clamp(fract(gridPos), vec3<f32>(0.0), vec3<f32>(1.0)));

  var wMat = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var dz = 0; dz < 2; dz = dz + 1) {
    let tz = select(1.0 - f.z, f.z, dz == 1);
    let cz = select(i0.z, i1.z, dz == 1);
    for (var dy = 0; dy < 2; dy = dy + 1) {
      let ty = select(1.0 - f.y, f.y, dy == 1);
      let cy = select(i0.y, i1.y, dy == 1);
      for (var dx = 0; dx < 2; dx = dx + 1) {
        let tx = select(1.0 - f.x, f.x, dx == 1);
        let cx = select(i0.x, i1.x, dx == 1);
        let m = clamp(i32(floor(textureLoad(soilTex, vec3<i32>(cx, cy, cz), 0).r * 255.0 + 0.5)), 0, 5);
        wMat[m] += tx * ty * tz;
      }
    }
  }

  let f3 = fbm3D(samplePos * 62.0);
  let vor = select(vec2<f32>(0.45, 0.25), voronoi3D(samplePos * 185.0), xrayMode == 0);

  var albedo = vec3<f32>(0.0);
  for (var m = 0; m < 6; m = m + 1) {
    if (wMat[m] > 0.001) {
      albedo += soilMaterialColor(m, samplePos, wetness, o2Norm, isTop, vor, f3) * wMat[m];
    }
  }
  let stoneFrac = wMat[5];

  if (xrayMode == 0) {
    // Crisp horizontal sedimentary micro-banding & pale mineral grit flecks on cutaway strata
    let earthMask = clamp(1.0 - stoneFrac - wMat[4], 0.0, 1.0);
    let strataWave = sin(samplePos.y * 540.0 + fbm3D(samplePos * 42.0) * 6.5) * 0.5 + 0.5;
    let strataDetail = valueNoise3D(vec3<f32>(samplePos.x * 16.0, samplePos.y * 380.0, samplePos.z * 16.0));
    let strataAmp = select(0.08, 0.36, !isTop);
    albedo *= mix(1.0, (1.0 - strataAmp * 0.5) + strataAmp * (strataWave * 0.58 + strataDetail * 0.42), earthMask);

    if (!isTop) {
      let paleSeed = hash31(floor(samplePos * 480.0));
      let paleVor = voronoi3D(samplePos * 460.0);
      let paleFlecks = step(0.92, paleSeed) * (1.0 - smoothstep(0.06, 0.22, paleVor.x)) * (1.0 - wMat[4]);
      albedo = mix(albedo, vec3<f32>(0.74, 0.68, 0.56), paleFlecks * 0.55);
    }
  }

  // Triplanar fBm + Voronoi micro-bump normal
  let upRef = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t1 = normalize(cross(upRef, n));
  let t2 = cross(n, t1);
  let epsB = 0.0020;
  let h0 = fbm3D(samplePos * 145.0);
  let hx = fbm3D((samplePos + t1 * epsB) * 145.0);
  let hz = fbm3D((samplePos + t2 * epsB) * 145.0);
  let bumpStrength = mix(0.85, 0.35, wetness * 0.5);
  let bumpedN = normalize(n - (t1 * (hx - h0) + t2 * (hz - h0)) * bumpStrength);

  albedo = srgbToLinear(albedo);
  var emissive = vec3<f32>(0.0);
  if (xrayMode == 1) {
    let mapped = colormapViridis(xrayVal);
    albedo = mix(albedo * 0.22, mapped, select(0.78, 0.92, isSlice));
    emissive = mapped * select(0.08, 0.18, isSlice);
  } else if (xrayMode == 2) {
    let mapped = colormapInferno(xrayVal);
    albedo = mix(albedo * 0.20, mapped, select(0.80, 0.94, isSlice));
    emissive = mapped * select(0.10, 0.20, isSlice);
  } else if (xrayMode == 3) {
    let mapped = colormapViridis(xrayVal);
    albedo = mix(albedo * 0.35, mapped * 0.55, select(0.45, 0.65, isSlice));
  } else if (xrayMode == 4) {
    albedo = mix(albedo * 0.35, vec3<f32>(0.015, 0.025, 0.045), 0.55);
  } else if (xrayMode == 5) {
    albedo = albedo * 0.72;
  }

  // Ground spore print overlay on soil top surface
  if (isTop && xrayMode == 0) {
    let spVal = textureSampleLevel(sporePrintTex, linearSampler, vec2<f32>(normPos.x, normPos.z), 0.0).r;
    let printAlpha = clamp(spVal * 1.45, 0.0, 0.92);
    albedo = mix(albedo, srgbToLinear(u.sporePrintColor.rgb), printAlpha);
  }

  // Hair-thin dark reddish-brown rootlets on cutaway faces
  var rootMask = 0.0;
  if (!isTop && xrayMode == 0 && samplePos.y > -0.105 && samplePos.y < -0.001) {
    let depthRoot = smoothstep(-0.105, -0.015, samplePos.y) * (1.0 - smoothstep(-0.006, -0.001, samplePos.y));
    let rootWarp = fbm3D(samplePos * 48.0) * 0.016;
    let uRoot = (samplePos.x + samplePos.z * 0.72) + rootWarp;
    let vRoot = samplePos.y * 1.4 + valueNoise3D(samplePos * 75.0) * 0.014;
    let p1 = abs(fract(uRoot * 38.0 + sin(vRoot * 160.0) * 0.35) - 0.5);
    let p2 = abs(fract((samplePos.z - samplePos.x * 0.55 + rootWarp) * 52.0 + cos(vRoot * 190.0) * 0.30) - 0.5);
    let patch1 = smoothstep(0.32, 0.70, valueNoise3D(samplePos * 26.0 + vec3<f32>(1.9, 0.4, 3.1)));
    let patch2 = smoothstep(0.38, 0.74, valueNoise3D(samplePos * 34.0 + vec3<f32>(7.1, 2.3, 5.7)));
    let rLine1 = (1.0 - smoothstep(0.012, 0.036, p1)) * patch1;
    let rLine2 = (1.0 - smoothstep(0.010, 0.030, p2)) * patch2 * 0.85;
    rootMask = clamp(max(rLine1, rLine2) * depthRoot * (1.0 - stoneFrac), 0.0, 1.0);
    let rootCol = srgbToLinear(vec3<f32>(0.34, 0.14, 0.065));
    albedo = mix(albedo, rootCol, rootMask * 0.88);
  }

  // Sharper wet capillary darkening fingers along cutaway borders
  var meniscus = 0.0;
  if (!isTop && xrayMode == 0) {
    let fN1 = valueNoise3D(vec3<f32>(in.worldPos.x * 78.0, in.worldPos.y * 22.0, in.worldPos.z * 78.0));
    let fN2 = valueNoise3D(vec3<f32>(in.worldPos.x * 165.0, in.worldPos.y * 42.0, in.worldPos.z * 165.0));
    let fingerDepth = mix(0.008, 0.038, smoothstep(0.30, 0.78, fN1 * 0.65 + fN2 * 0.35));
    meniscus = (1.0 - smoothstep(-fingerDepth, 0.001, in.worldPos.y)) * smoothstep(0.16, 0.74, wetness);
    let poreCapillary = mix(0.85, 1.15, smoothstep(0.18, 0.72, vor.x));
    albedo *= 1.0 - clamp(meniscus * 0.40 * poreCapillary, 0.0, 0.48);
  }

  let l = normalize(u.lightDir.xyz);
  let h = normalize(l + v);
  let ndl = max(dot(bumpedN, l), 0.0);
  let depthAo = mix(0.78, 1.0, smoothstep(0.0, 0.28, normPos.y)) * (1.0 - rootMask * 0.26);

  var color: vec3<f32>;
  if (xrayMode == 0) {
    let giBlend = smoothstep(-0.048, -0.003, in.worldPos.y);
    let aoBent = evalFastSdfAO(in.worldPos + n * 0.005, bumpedN);
    let bentN = aoBent.xyz;
    let sdfAo = mix(1.0, aoBent.w, giBlend);
    let pcfSh = evalPcfShadow(in.worldPos, n);
    let sdfSh = evalSdfSoftShadow(in.worldPos + n * 0.007, l);
    let shadow = mix(1.0, min(pcfSh, mix(0.36, 1.0, sdfSh)), giBlend);
    let gi = evalSdfGI(in.worldPos + n * 0.008, bumpedN, bentN) * giBlend;

    let gobo = select(1.0, evalCanopyGobo(in.worldPos, l, u.padExtra), isTop);
    let mood = u.padExtra.x;
    let noct = max(clamp(u.biolumInfo.y, 0.0, 1.0), smoothstep(2.4, 2.9, mood));
    let isCanopyMood = smoothstep(0.5, 1.0, mood) * (1.0 - smoothstep(1.2, 1.8, mood));
    let isBacklitMood = smoothstep(1.2, 1.8, mood) * (1.0 - smoothstep(2.4, 2.9, mood));

    let skyScale = mix(mix(1.0, 0.38, isBacklitMood), 0.07, noct);
    let skyFloor = select(vec3<f32>(0.032, 0.038, 0.044), vec3<f32>(0.062, 0.068, 0.078), !isTop);
    let skyDay = mix(skyFloor, vec3<f32>(0.13, 0.16, 0.21), bentN.y * 0.5 + 0.5);
    let skyMoon = mix(vec3<f32>(0.015, 0.022, 0.048), vec3<f32>(0.055, 0.085, 0.165), bentN.y * 0.5 + 0.5);
    let sky = mix(skyDay, skyMoon, noct) * skyScale;
    let sunBase = mix(vec3<f32>(0.96, 0.90, 0.80), vec3<f32>(1.04, 0.88, 0.62), isCanopyMood);
    let sunTint = mix(sunBase, vec3<f32>(0.36, 0.48, 0.78), noct) * mix(1.0, 0.07, noct);
    let sun = sunTint * ndl * mix(0.22, 1.0, shadow) * gobo;
    let rimLight = pow(1.0 - max(dot(bumpedN, v), 0.0), 3.2) * u.postParams.w * 0.04 * mix(1.0, 0.12, noct);
    let specDim = mix(1.0, 0.08, noct);
    let wetSpec = pow(max(dot(bumpedN, h), 0.0), 42.0) * wetness * 0.14 * (1.0 - stoneFrac) * shadow * sdfAo * gobo * specDim;

    // Tiny glinting quartz grit flecks in sand / silt-loam / granite
    let qFrac = clamp(wMat[0] + wMat[1] * 0.55 + wMat[5] * 0.70, 0.0, 1.0);
    var quartzGlint = 0.0;
    if (qFrac > 0.05) {
      let qCellPos = samplePos * 460.0;
      let qVor = voronoi3D(qCellPos);
      let qId = floor(qCellPos);
      let qSeed = hash31(qId);
      let qSpark = step(0.952, qSeed) * (1.0 - smoothstep(0.05, 0.18, qVor.x));
      let qFacetN = normalize(bumpedN + (hash33(qId) - vec3<f32>(0.5)) * 1.15);
      quartzGlint = pow(max(dot(qFacetN, h), 0.0), 96.0) * qSpark * qFrac * 0.85 * shadow * sdfAo * gobo * specDim;
    }

    // Wet soil IBL reflection sheen on surface & meniscus (plus emerald GI catchlight at night)
    let wetIblMask = select(meniscus * 0.72, smoothstep(0.26, 0.86, wetness) * (1.0 - stoneFrac * 0.5), isTop);
    let soilRefl = reflect(-v, bumpedN);
    let soilFresnel = 0.02 + 0.98 * pow(1.0 - max(dot(bumpedN, v), 0.0), 5.0);
    let wetIbl = evalStudioIbl(soilRefl, mix(0.62, 0.16, wetness), u.padExtra.w)
      * soilFresnel * wetIblMask * 0.55 * mix(0.35, 1.0, sdfAo)
      * mix(vec3<f32>(1.0), vec3<f32>(0.36, 0.48, 0.80) * 0.08, noct);
    let giWeight = mix(0.50, 1.25, noct);
    let wetGiSheen = gi * soilFresnel * wetIblMask * 0.12 * noct;

    color = albedo * ((sky + sun) * sdfAo + gi * giWeight + vec3<f32>(rimLight)) * depthAo + vec3<f32>(wetSpec + quartzGlint) + wetIbl + wetGiSheen;
  } else {
    let hemi = 0.14 + 0.10 * (bumpedN.y * 0.5 + 0.5);
    let wetSpec = pow(max(dot(bumpedN, h), 0.0), 36.0) * wetness * 0.08 * (1.0 - stoneFrac);
    color = albedo * (hemi + 0.85 * ndl) * depthAo + vec3<f32>(wetSpec) + emissive;
  }

  if (isSlice) {
    var edgeDist = 1.0;
    if (u.sliceParams.y < 0.5) {
      edgeDist = min(min(normPos.y, 1.0 - normPos.y), min(normPos.z, 1.0 - normPos.z));
    } else {
      edgeDist = min(min(normPos.y, 1.0 - normPos.y), min(normPos.x, 1.0 - normPos.x));
    }
    let rim = 1.0 - smoothstep(0.004, 0.016, edgeDist);
    color = mix(color, vec3<f32>(0.32, 0.78, 0.98), rim * 0.85);
  }

  return vec4<f32>(color, 1.0);
}
`;

// ----------------------------------------------------------------------------
// 2. DEPTH-CUED 3D MYCELIAL NETWORK SHADER (WGSL HDR)
// ----------------------------------------------------------------------------
export const MYCELIUM_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;

struct MycVertIn {
  @location(0) corner: vec2<f32>,
  @location(1) segA: vec3<f32>,
  @location(2) segB: vec3<f32>,
  @location(3) radiusTurgor: vec2<f32>,
};

struct MycVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) segCoords: vec2<f32>,
  @location(2) segMetrics: vec2<f32>,
  @location(3) turgorNorm: f32,
};

@vertex
fn vs_main(v: MycVertIn) -> MycVertOut {
  var out: MycVertOut;
  let segVec = v.segB - v.segA;
  let segLen = length(segVec);
  let segDir = segVec / max(segLen, 1e-6);
  let visRadius = clamp(v.radiusTurgor.x * 16.0 + 0.00024, 0.00028, 0.0022);
  let ext = (v.corner.x * 2.0 - 1.0) * visRadius;
  let p = mix(v.segA, v.segB, v.corner.x) + segDir * ext;
  let toCam = normalize(u.cameraPos.xyz - p);
  var side = cross(segDir, toCam);
  let sideLen = length(side);
  if (sideLen < 1e-4) {
    side = vec3<f32>(1.0, 0.0, 0.0);
  } else {
    side = side / sideLen;
  }
  let worldPos = p + side * (v.corner.y * visRadius);
  out.clipPos = u.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.segCoords = vec2<f32>(mix(-visRadius, segLen + visRadius, v.corner.x), v.corner.y * visRadius);
  out.segMetrics = vec2<f32>(segLen, visRadius);
  out.turgorNorm = clamp(v.radiusTurgor.y, 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(in: MycVertOut) -> @location(0) vec4<f32> {
  if (isClippedBySlice(in.worldPos, u.sliceParams)) {
    discard;
  }

  // Smooth rounded capsule end-caps preventing harsh square joints
  let spineT = clamp(in.segCoords.x, 0.0, in.segMetrics.x);
  let dxCap = in.segCoords.x - spineT;
  let dyCap = in.segCoords.y;
  let normR = length(vec2<f32>(dxCap, dyCap)) / max(in.segMetrics.y, 1e-6);
  if (normR > 1.0) {
    discard;
  }

  let radial = 1.0 - normR;
  let cylShade = sqrt(max(0.0, 1.0 - normR * normR));
  let xrayMode = i32(floor(u.soilBoundsMax.w + 0.5));

  // Distance inward from nearest exposed cutaway glass / surface / razor slice
  let dX = min(in.worldPos.x - u.soilBoundsMin.x, u.soilBoundsMax.x - in.worldPos.x);
  let dZ = min(in.worldPos.z - u.soilBoundsMin.z, u.soilBoundsMax.z - in.worldPos.z);
  let dY = max(0.0, -in.worldPos.y);
  let wallMin = min(dX, dZ);
  var distFromExposed = min(wallMin, dY);
  if (u.sliceParams.x > 0.5) {
    let sDist = select(u.sliceParams.z - in.worldPos.x, u.sliceParams.z - in.worldPos.z, u.sliceParams.y > 0.5);
    distFromExposed = min(distFromExposed, max(0.0, sDist));
  }
  let isSubterranean = u.sliceParams.w < 0.7;
  let depthCue = select(1.0, exp(-max(0.0, distFromExposed) * 12.0) * 0.82 + 0.18, isSubterranean);
  // Dark glass-contact crevice attenuation & rounded cylindrical rim contrast
  let glassCrevice = mix(0.68, 1.0, smoothstep(0.0, 0.010, max(0.0, wallMin)));
  let edgeCrevice = mix(0.34, 1.0, pow(cylShade, 1.25));

  var col: vec3<f32>;
  if (xrayMode == 3) {
    col = colormapViridis(in.turgorNorm) * (0.52 + 0.58 * cylShade);
  } else {
    let noct = max(clamp(u.biolumInfo.y, 0.0, 1.0), smoothstep(2.4, 2.9, u.padExtra.x));
    let dryHypha = srgbToLinear(vec3<f32>(0.40, 0.31, 0.21));
    let turgidHypha = srgbToLinear(vec3<f32>(0.97, 0.90, 0.75));
    let glowTip = srgbToLinear(vec3<f32>(1.04, 0.83, 0.40));
    col = mix(dryHypha, turgidHypha, in.turgorNorm);
    col = mix(col, glowTip, pow(in.turgorNorm, 2.2) * 0.44);
    col = col * edgeCrevice * glassCrevice;
    if (isSubterranean) {
      // Stronger subterranean depth-cued warmth toward deep amber earth tone
      let deepWarm = col * vec3<f32>(0.54, 0.30, 0.14);
      col = mix(deepWarm, col * vec3<f32>(1.04, 0.96, 0.84), depthCue);
    }
    col *= mix(vec3<f32>(1.0), vec3<f32>(0.11, 0.15, 0.22), noct);

    if (xrayMode == 0 && u.biolumInfo.x > 0.001) {
      let o2Factor = clamp(u.biolumInfo.z, 0.0, 1.0);
      let metabolic = smoothstep(0.14, 0.86, in.turgorNorm) * (0.45 + 0.55 * cylShade);
      let foxfireAmp = u.biolumInfo.x * mix(0.22, 1.85, noct) * o2Factor;
      let depthFade = select(1.0, mix(0.35, 1.0, depthCue), isSubterranean);
      col += vec3<f32>(0.045, 0.98, 0.22) * (metabolic * foxfireAmp * depthFade);
    }
  }

  let wispAlpha = mix(0.66, 1.0, smoothstep(0.00032, 0.00085, in.segMetrics.y));
  let alpha = smoothstep(0.0, 0.30, radial) * u.sliceParams.w * wispAlpha * mix(1.0, depthCue, 0.38);
  if (alpha < 0.02) {
    discard;
  }
  return vec4<f32>(col, alpha);
}
`;

// ----------------------------------------------------------------------------
// 3. BOTANICAL SPECIES PBR + WAXY CHITIN SSS + PCF SHADOW + FAST-SDF-AO/GI
// ----------------------------------------------------------------------------
export const MUSHROOM_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;
${makeWgslSdfGiHelpers(1, 'sdfSamp', 3, 2)}
${makeWgslShadowHelpers(4, 5)}
${makeWgslIblHelpers(6, 7)}

fn evalCharlieSheen(ndv: f32, roughness: f32) -> f32 {
  let r = clamp(roughness, 0.08, 1.0);
  let invR = 1.0 / r;
  let sinT = clamp(sqrt(max(1.0 - ndv * ndv, 0.0)), 1e-4, 1.0);
  return clamp((2.0 + invR) * pow(sinT, invR) * 0.15915494, 0.0, 1.6);
}

struct MushVertIn {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  @location(3) metrics: vec3<f32>, // x = stressVM (Pa), y = turgorPa (Pa), z = partId
  @location(4) botUv: vec4<f32>,   // xy = cap-local (dx/capR, dz/capR), z = relY, w = bodySeed
};

struct MushVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) metrics: vec3<f32>,
  @location(4) botUv: vec4<f32>,
};

@vertex
fn vs_main(v: MushVertIn) -> MushVertOut {
  var out: MushVertOut;
  out.clipPos = u.viewProj * vec4<f32>(v.pos, 1.0);
  out.worldPos = v.pos;
  out.worldNormal = v.normal;
  out.baseColor = v.color;
  out.metrics = v.metrics;
  out.botUv = v.botUv;
  return out;
}

@fragment
fn fs_main(in: MushVertOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4<f32> {
  if (isClippedBySlice(in.worldPos, u.sliceParams)) {
    discard;
  }

  let rawN = normalize(in.worldNormal);
  let n = select(-rawN, rawN, frontFacing);
  let l = normalize(u.lightDir.xyz);
  let v = normalize(u.cameraPos.xyz - in.worldPos);
  let h = normalize(l + v);

  let stressPa = in.metrics.x;
  let turgorPa = in.metrics.y;
  let partId = i32(floor(in.metrics.z + 0.5));
  let xrayMode = i32(floor(u.soilBoundsMax.w + 0.5));

  if (xrayMode == 4) {
    let t = clamp(stressPa / 8.5e5, 0.0, 1.0);
    var col = colormapInferno(t);
    let contour = abs(fract(t * 10.0) - 0.5);
    let line = 1.0 - smoothstep(0.02, 0.06, contour);
    let ndl = max(dot(n, l), 0.0);
    col = col * (0.58 + 0.55 * ndl) + vec3<f32>(0.15) * line;
    return vec4<f32>(col, 1.0);
  }
  if (xrayMode == 3) {
    let t = clamp(turgorPa / 7.5e5, 0.0, 1.0);
    var col = colormapViridis(t);
    let ndl = max(dot(n, l), 0.0);
    col = col * (0.55 + 0.55 * ndl);
    return vec4<f32>(col, 1.0);
  }

  let rNorm = clamp(length(in.botUv.xy), 0.0, 1.35);
  let theta = select(0.0, atan2(in.botUv.y, in.botUv.x), dot(in.botUv.xy, in.botUv.xy) > 1e-8);
  let relY = in.botUv.z;
  let bSeed = in.botUv.w;
  let speciesIdx = i32(floor(u.speciesExtra.x + 0.5));

  let micro = valueNoise3D(in.worldPos * 260.0);
  let microX = valueNoise3D((in.worldPos + vec3<f32>(0.0014, 0.0, 0.0)) * 260.0);
  let microZ = valueNoise3D((in.worldPos + vec3<f32>(0.0, 0.0, 0.0014)) * 260.0);

  var albedo = srgbToLinear(in.baseColor) * (0.94 + 0.12 * micro);
  var specPower = 36.0;
  var specStrength = select(0.14, 0.04, partId == 2);
  var sheenStrength = 0.0;
  var sheenRough = 0.45;
  var sheenColor = vec3<f32>(1.0);
  var scaleAo = 1.0;
  var pitAo = 1.0;
  var bumpVec = vec3<f32>(microX - micro, 0.0, microZ - micro) * select(0.18, 0.08, partId == 2);

  if (speciesIdx == 2 && (partId == 1 || partId == 2)) {
    // Woodland Morel: sharp 3D Voronoi cellular honeycomb pits & ridges matching sculpted silhouette
    let cellPos = in.worldPos * 165.0 + vec3<f32>(bSeed * 3.7, bSeed * 1.9, bSeed * 5.1);
    let vor = voronoi3D(cellPos);
    let ridge = 1.0 - smoothstep(0.03, 0.27, vor.y);
    let pit = smoothstep(0.18, 0.56, vor.y);
    let pitCol = srgbToLinear(vec3<f32>(0.18, 0.11, 0.06));
    let ridgeCol = srgbToLinear(vec3<f32>(0.78, 0.60, 0.40)) * (0.92 + 0.16 * micro);
    albedo = mix(pitCol, ridgeCol, ridge);
    pitAo = mix(1.0, 0.58, pit);
    specPower = 54.0;
    specStrength = mix(0.03, 0.16, ridge);
    bumpVec += vec3<f32>(microX - micro, (ridge - 0.5) * 0.12, microZ - micro) * 0.22;
  } else if (partId == 1) {
    if (speciesIdx == 0) {
      // Meadow Parasol: warm chestnut umbo + physically exfoliated concentric scales + lip AO
      let umbo = smoothstep(0.30, 0.04, rNorm);
      let bandPhase = rNorm * 26.0 + valueNoise3D(in.worldPos * 95.0) * 3.8 + bSeed;
      let scaleBand = sin(bandPhase);
      let bandSlope = cos(bandPhase);
      let patchBreak = smoothstep(-0.12, 0.52, sin(theta * 14.0 + valueNoise3D(in.worldPos * 85.0) * 6.4 + bSeed * 2.1));
      let patchZone = smoothstep(0.16, 0.42, rNorm) * (1.0 - smoothstep(0.78, 1.04, rNorm));
      let patchMask = smoothstep(0.14, 0.68, scaleBand) * patchBreak * patchZone;
      let scaleEdgeAO = smoothstep(0.24, -0.04, scaleBand) * smoothstep(-0.36, -0.02, scaleBand) * smoothstep(0.0, -0.65, bandSlope) * patchBreak * patchZone;
      let striation = sin(theta * 88.0 + valueNoise3D(in.worldPos * 120.0) * 1.4) * smoothstep(0.58, 0.96, rNorm);
      let umboCol = srgbToLinear(vec3<f32>(0.38, 0.21, 0.10));
      let scaleCol = srgbToLinear(vec3<f32>(0.50, 0.31, 0.18));
      albedo = mix(albedo, umboCol, umbo * 0.84);
      albedo = mix(albedo, scaleCol, patchMask * 0.68);
      albedo *= (1.0 - 0.10 * abs(striation)) * (1.0 - 0.52 * scaleEdgeAO);
      scaleAo = 1.0 - 0.38 * scaleEdgeAO;
      let uvLen = length(in.botUv.xy);
      let radDir = select(vec3<f32>(1.0, 0.0, 0.0), normalize(vec3<f32>(in.botUv.x, 0.0, in.botUv.y)), uvLen > 1e-4);
      bumpVec += vec3<f32>(-sin(theta), 0.0, cos(theta)) * (striation * 0.16) + radDir * (patchMask * 0.24 - scaleEdgeAO * 0.48);
      specPower = 64.0;
      specStrength = 0.16;
      sheenStrength = 0.08 * (1.0 - patchMask * 0.5);
      sheenRough = 0.52;
      sheenColor = srgbToLinear(vec3<f32>(0.96, 0.90, 0.80));
    } else if (speciesIdx == 1) {
      // Golden Chanterelle: deep apricot center -> golden rim + velvety Charlie sheen
      let centerApricot = srgbToLinear(vec3<f32>(0.84, 0.36, 0.04));
      let rimGold = srgbToLinear(vec3<f32>(0.98, 0.72, 0.12));
      albedo = mix(centerApricot, rimGold, smoothstep(0.08, 0.85, rNorm)) * (0.94 + 0.12 * micro);
      sheenStrength = 0.24;
      sheenRough = 0.42;
      sheenColor = srgbToLinear(vec3<f32>(1.0, 0.84, 0.36));
      specPower = 26.0;
      specStrength = 0.06;
    } else if (speciesIdx == 4) {
      // Mycena chlorophos: delicate translucent celadon cap striations & radial bell flutes
      let fluteWave = sin(theta * 24.0 + bSeed);
      let fluteMask = smoothstep(0.16, 0.92, rNorm);
      let umboShade = smoothstep(0.32, 0.04, rNorm);
      let celadonCenter = srgbToLinear(vec3<f32>(0.54, 0.68, 0.56));
      let celadonEdge = srgbToLinear(vec3<f32>(0.84, 0.94, 0.85));
      let ridgeCol = srgbToLinear(vec3<f32>(0.91, 0.98, 0.89));
      let grooveCol = srgbToLinear(vec3<f32>(0.62, 0.78, 0.66));
      var capCol = mix(celadonEdge, celadonCenter, umboShade * 0.70);
      capCol = mix(capCol, mix(grooveCol, ridgeCol, 0.5 + 0.5 * fluteWave), fluteMask * 0.58);
      albedo = capCol * (0.94 + 0.12 * micro);
      let tanDir = vec3<f32>(-sin(theta), 0.0, cos(theta));
      bumpVec += tanDir * (cos(theta * 24.0 + bSeed) * fluteMask * 0.20);
      specPower = 56.0;
      specStrength = 0.15;
      sheenStrength = 0.10;
      sheenRough = 0.36;
      sheenColor = srgbToLinear(vec3<f32>(0.86, 0.98, 0.88));
    } else {
      // Shaggy Ink Cap: vertical fibrous scales + deliquescing black wet margin
      let fiber = valueNoise3D(vec3<f32>(cos(theta) * 18.0, relY * 180.0, sin(theta) * 18.0) + vec3<f32>(bSeed));
      let inkMargin = smoothstep(0.68, 0.98, rNorm);
      let scaleCol = mix(albedo * 0.82, srgbToLinear(vec3<f32>(0.96, 0.94, 0.89)), smoothstep(0.42, 0.75, fiber));
      albedo = mix(scaleCol, srgbToLinear(vec3<f32>(0.06, 0.06, 0.08)), inkMargin * 0.88);
      sheenStrength = 0.20 * (1.0 - inkMargin);
      sheenRough = 0.48;
      sheenColor = srgbToLinear(vec3<f32>(0.96, 0.94, 0.90));
      specPower = mix(36.0, 110.0, inkMargin);
      specStrength = mix(0.10, 0.32, inkMargin);
    }
  } else if (partId == 2 && speciesIdx == 1) {
    // Chanterelle decurrent false-gill folds: soft golden sheen
    sheenStrength = 0.16;
    sheenRough = 0.44;
    sheenColor = srgbToLinear(vec3<f32>(1.0, 0.82, 0.34));
  } else if (partId == 2 && speciesIdx == 4) {
    // Mycena translucent celadon hymenium gills
    sheenStrength = 0.14;
    sheenRough = 0.38;
    sheenColor = srgbToLinear(vec3<f32>(0.78, 0.98, 0.82));
  } else if (partId == 0) {
    // Stipe vertical fibrous micro-striation
    let stipeFib = valueNoise3D(vec3<f32>(in.worldPos.x * 180.0, relY * 45.0, in.worldPos.z * 180.0));
    albedo *= 0.90 + 0.18 * stipeFib;
    if (speciesIdx == 4) {
      albedo = mix(albedo, srgbToLinear(vec3<f32>(0.84, 0.95, 0.86)), smoothstep(0.35, 1.0, relY) * 0.45);
    }
    sheenStrength = 0.07;
    sheenRough = 0.55;
    sheenColor = srgbToLinear(vec3<f32>(0.92, 0.88, 0.82));
  }

  if (!frontFacing) {
    let backTint = select(vec3<f32>(0.90, 0.85, 0.76), vec3<f32>(0.78, 0.94, 0.80), speciesIdx == 4);
    albedo = mix(albedo, srgbToLinear(backTint), 0.60);
  }

  let nLit = normalize(n - bumpVec);

  var structAo = scaleAo * pitAo;
  if (partId == 2 && speciesIdx != 2) {
    structAo *= mix(0.42, 0.94, clamp(0.5 + 0.5 * abs(nLit.x * 0.5 + nLit.z * 0.5), 0.0, 1.0));
    structAo *= mix(0.62, 1.0, smoothstep(-0.85, 0.25, nLit.y));
  } else if (partId == 0) {
    structAo *= mix(0.70, 1.0, smoothstep(0.0, 0.025, in.worldPos.y));
  }

  let sdfProbe = in.worldPos + nLit * 0.005;
  let aoBent = evalFastSdfAO(sdfProbe, nLit);
  let bentN = aoBent.xyz;
  let sdfAo = aoBent.w;
  let totalAo = structAo * mix(0.46, 1.0, sdfAo);

  let pcfSh = evalPcfShadow(in.worldPos, nLit);
  let sdfSh = evalSdfSoftShadow(in.worldPos + nLit * 0.008, l);
  let shadow = min(pcfSh, mix(0.34, 1.0, sdfSh));
  let gi = evalSdfGI(sdfProbe, nLit, bentN);

  let gobo = evalCanopyGobo(in.worldPos, l, u.padExtra);
  let mood = u.padExtra.x;
  let noct = max(clamp(u.biolumInfo.y, 0.0, 1.0), smoothstep(2.4, 2.9, mood));
  let isCanopyMood = smoothstep(0.5, 1.0, mood) * (1.0 - smoothstep(1.2, 1.8, mood));
  let isBacklitMood = smoothstep(1.2, 1.8, mood) * (1.0 - smoothstep(2.4, 2.9, mood));

  // 3-Point Studio Macro Botanical Lighting + IBL + Charlie Sheen + Dual-Lobe Specular + Thickness SSS + Nocturnal Moonlight
  let wrapNdl = clamp((dot(nLit, l) + 0.22) / 1.22, 0.0, 1.0);
  let sunBase = mix(vec3<f32>(0.94, 0.88, 0.76), vec3<f32>(1.06, 0.87, 0.60), isCanopyMood);
  let sunTint = mix(sunBase, vec3<f32>(0.36, 0.48, 0.78), noct) * mix(1.0, 0.07, noct);
  let sunTerm = sunTint * wrapNdl * mix(0.18, 1.0, shadow) * gobo;
  let skyScale = mix(mix(1.0, 0.32, isBacklitMood), 0.09, noct);
  let skyDay = mix(vec3<f32>(0.025, 0.030, 0.038), vec3<f32>(0.10, 0.13, 0.18), bentN.y * 0.5 + 0.5);
  let skyMoon = mix(vec3<f32>(0.015, 0.022, 0.048), vec3<f32>(0.055, 0.088, 0.168), bentN.y * 0.5 + 0.5);
  let skyFill = mix(skyDay, skyMoon, noct) * skyScale;

  let ndv = max(dot(nLit, v), 0.0);
  let fresnel = pow(1.0 - ndv, 3.0);
  let rimBack = max(0.25, dot(-v, l) * 0.5 + 0.5);
  let rimBoost = mix(1.0, 2.15, isBacklitMood);
  let rimTint = mix(vec3<f32>(0.88, 0.76, 0.60), vec3<f32>(0.36, 0.50, 0.82), noct);
  let rimLight = rimTint * fresnel * rimBack * 0.14 * u.postParams.w * rimBoost * totalAo * mix(1.0, 0.16, noct);

  // Velvety Charlie microfiber sheen
  let charlie = evalCharlieSheen(ndv, sheenRough);
  let sheen = sheenColor * charlie * sheenStrength * (0.35 + 0.65 * wrapNdl) * totalAo * mix(1.0, 0.10, noct);

  // Thickness-modulated backlit waxy chitin SSS (dramatically amplified in Dark-Field Backlit mood)
  let backDot = clamp(dot(v, -l + nLit * 0.38), 0.0, 1.0);
  let backLight = pow(backDot, 2.2) * 0.85 + pow(backDot, 5.0) * 0.65;
  let capThin = mix(0.38, 0.96, smoothstep(0.18, 0.96, rNorm));
  let thicknessFactor = select(select(capThin, 1.05, partId == 2 || partId == 3), 0.34, partId == 0);
  let sssBoost = mix(1.0, 2.85, isBacklitMood);
  let sss = srgbToLinear(u.sssColor.rgb) * backLight * 0.42 * thicknessFactor * sssBoost * (0.35 + 0.65 * shadow) * mix(0.55, 1.0, gobo) * mix(1.0, 0.08, noct);

  // Dual-lobe specular + Studio IBL environment reflections
  let ndh = max(dot(nLit, h), 0.0);
  let broadSpec = pow(ndh, specPower) * specStrength;
  let coatSpec = pow(ndh, specPower * 3.2) * (specStrength * 0.55);
  let specDim = mix(1.0, 0.08, noct);
  let specCol = vec3<f32>((broadSpec + coatSpec) * shadow * sdfAo * gobo) * mix(vec3<f32>(1.0), vec3<f32>(0.42, 0.58, 0.92), noct) * specDim;

  let cutRoughness = clamp(1.0 - sqrt(specPower) / 11.0, 0.14, 0.86);
  let fSchlick = vec3<f32>(0.035) + vec3<f32>(0.965) * pow(1.0 - ndv, 5.0);
  let iblRefl = evalStudioIbl(reflect(-v, nLit), cutRoughness, u.padExtra.w) * fSchlick * mix(0.28, 1.0, totalAo) * mix(vec3<f32>(1.0), vec3<f32>(0.35, 0.48, 0.82) * 0.08, noct);
  let iblDiff = evalStudioIbl(bentN, 0.95, u.padExtra.w) * 0.16 * skyScale * totalAo;

  let giWeight = mix(0.48, 1.25, noct);
  var color = albedo * ((skyFill + iblDiff + sunTerm) * totalAo + gi * giWeight) + sss + sheen + specCol + iblRefl + rimLight;

  // High-RH micro-water dewdrops clustered delicately near cap margins with sharp clearcoat glint
  let rhDew = smoothstep(0.80, 0.96, clamp(u.soilDims.w, 0.0, 1.0));
  let marginBand = smoothstep(0.54, 0.80, rNorm) * (1.0 - smoothstep(0.95, 1.05, rNorm));
  if (partId == 1 && rhDew > 0.01 && marginBand > 0.01) {
    let dewScale = 420.0;
    let dewSeed = vec3<f32>(bSeed * 2.1, 0.0, bSeed * 4.3);
    let dewVor = voronoi3D(in.worldPos * dewScale + dewSeed);
    let dewCluster = smoothstep(0.46, 0.72, valueNoise3D(in.worldPos * 85.0 + vec3<f32>(bSeed * 7.3)));
    let dewDrop = (1.0 - smoothstep(0.08, 0.25, dewVor.x)) * dewCluster * marginBand * rhDew;
    if (dewDrop > 0.01) {
      let dewUp = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(nLit.y) > 0.9);
      let dt1 = normalize(cross(dewUp, nLit));
      let dt2 = cross(nLit, dt1);
      let epsD = 0.00022;
      let dVorX = voronoi3D((in.worldPos + dt1 * epsD) * dewScale + dewSeed).x;
      let dVorZ = voronoi3D((in.worldPos + dt2 * epsD) * dewScale + dewSeed).x;
      let dGx = (dVorX - dewVor.x) / epsD;
      let dGz = (dVorZ - dewVor.x) / epsD;
      let dewN = normalize(nLit + (dt1 * dGx + dt2 * dGz) * 0.0020);
      let dewFresnel = 0.02 + 0.98 * pow(1.0 - max(dot(dewN, v), 0.0), 5.0);
      let dewSpec = pow(max(dot(dewN, h), 0.0), 220.0) * 1.45 * shadow * sdfAo * gobo * specDim;
      let dewIbl = (evalStudioIbl(reflect(-v, dewN), 0.03, u.padExtra.w) * mix(vec3<f32>(1.0), vec3<f32>(0.36, 0.50, 0.85) * 0.10, noct) + gi * 0.55 * noct) * dewFresnel * 1.15;
      let dewRim = smoothstep(0.18, 0.25, dewVor.x) * dewCluster * marginBand * rhDew;
      color = mix(color * (1.0 - 0.16 * dewRim), color * 0.88 + dewIbl + vec3<f32>(dewSpec), clamp(dewDrop * 1.15, 0.0, 1.0));
    }
  }

  // Real narrowband 525 nm fungal bioluminescence (Mycena gills/fluted margin/stipe; never over X-Ray modes)
  if (xrayMode == 0 && u.biolumInfo.w > 0.001) {
    let biolumCol525 = vec3<f32>(0.045, 0.98, 0.22);
    let glow = u.biolumInfo.w * mix(0.18, 1.75, noct);
    var emitMask = 0.0;
    if (partId == 2) {
      // Brightest on underside gills with subtle radial structure
      emitMask = 1.35 * mix(0.84, 1.08, 0.5 + 0.5 * sin(theta * 24.0 + bSeed));
    } else if (partId == 1) {
      // Glowing along translucent fluted thin cap margin
      let marginGlow = smoothstep(0.15, 0.95, rNorm);
      let fluteMod = select(
        0.92 + 0.08 * sin(theta * 24.0 + bSeed),
        0.72 + 0.38 * (0.5 + 0.5 * sin(theta * 24.0 + bSeed)),
        speciesIdx == 4,
      );
      emitMask = mix(0.14, 0.96, marginGlow) * fluteMod;
    } else if (partId == 0) {
      // Subtle translucent glow gradient on upper stipe beneath the gills
      emitMask = mix(0.06, 0.62, smoothstep(0.35, 1.0, relY));
    } else {
      emitMask = 0.48;
    }
    color += biolumCol525 * (emitMask * glow);
  }

  let alpha = select(1.0, 0.86, partId == 3);
  return vec4<f32>(color, alpha);
}
`;

// ----------------------------------------------------------------------------
// 4. SOLID OBSTACLES & PHYSICAL GLASS W/ CONDENSATION SHADER (WGSL HDR)
// ----------------------------------------------------------------------------
export const SOLID_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;
${makeWgslSdfGiHelpers(1, 'sdfSamp', 3, 2)}
${makeWgslShadowHelpers(4, 5)}
${makeWgslIblHelpers(6, 7)}

struct SolidVertIn {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
};

struct SolidVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(v: SolidVertIn) -> SolidVertOut {
  var out: SolidVertOut;
  out.clipPos = u.viewProj * vec4<f32>(v.pos, 1.0);
  out.worldPos = v.pos;
  out.worldNormal = v.normal;
  out.color = v.color;
  return out;
}

@fragment
fn fs_main(in: SolidVertOut) -> @location(0) vec4<f32> {
  let nLen = length(in.worldNormal);
  if (nLen < 0.05) {
    let lineFade = select(1.0, select(0.38, 0.26, u.padExtra.x > 2.5), u.padExtra.x > 1.5);
    return vec4<f32>(in.color.rgb, in.color.a * lineFade);
  }

  let n = in.worldNormal / nLen;
  let l = normalize(u.lightDir.xyz);
  let v = normalize(u.cameraPos.xyz - in.worldPos);
  let noct = max(clamp(u.biolumInfo.y, 0.0, 1.0), smoothstep(2.4, 2.9, u.padExtra.x));

  // Translucent museum glass pane + genuine studio softbox IBL reflection + live RH condensation
  if (in.color.a < 0.95) {
    let xrayMode = i32(floor(u.soilBoundsMax.w + 0.5));
    let nView = select(-n, n, dot(n, v) >= 0.0);
    let glassRefl = reflect(-v, nView);
    let glassIbl = evalStudioIbl(glassRefl, 0.06, u.padExtra.w)
      * mix(vec3<f32>(1.0), vec3<f32>(0.34, 0.48, 0.82) * 0.14, noct);
    let fresnel = 0.04 + 0.96 * pow(1.0 - abs(dot(n, v)), 3.4);
    let rh = clamp(u.soilDims.w, 0.0, 1.0);
    let rhMist = select(0.0, smoothstep(0.72, 0.98, rh), xrayMode == 0);
    let hBand = smoothstep(0.02, 0.14, in.worldPos.y) * (1.0 - smoothstep(0.24, 0.34, in.worldPos.y));
    let dropNoise = voronoi3D(in.worldPos * 280.0);
    let droplet = (1.0 - smoothstep(0.08, 0.30, dropNoise.x)) * rhMist * hBand;
    let meniscus = (1.0 - smoothstep(0.0, 0.008, abs(in.worldPos.y))) * select(0.02, 0.08, xrayMode == 0);

    let giGlass = select(vec3<f32>(0.0), evalSdfGI(in.worldPos * 0.90, nView, nView), xrayMode == 0 && noct > 0.01);
    let iblScale = select(0.16, 0.68, xrayMode == 0);
    let dropCol = (vec3<f32>(0.42, 0.56, 0.72) * mix(1.0, 0.14, noct) + giGlass * 1.10 * noct) * droplet * 0.18
      + (glassIbl + giGlass * 0.55 * noct) * droplet * 0.32;
    let col = srgbToLinear(in.color.rgb) * mix(1.0, 0.18, noct)
      + srgbToLinear(vec3<f32>(0.55, 0.72, 0.88)) * mix(vec3<f32>(1.0), vec3<f32>(0.28, 0.42, 0.76) * 0.22, noct) * fresnel * select(0.16, 0.30, xrayMode == 0)
      + glassIbl * fresnel * iblScale
      + giGlass * fresnel * 0.22 * noct
      + dropCol;
    let baseA = select(in.color.a * 0.45, in.color.a, xrayMode == 0);
    let alpha = clamp(baseA + fresnel * 0.13 + droplet * 0.055 + meniscus, 0.0, 0.42);
    return vec4<f32>(col, alpha);
  }

  let grain = fbm3D(in.worldPos * 140.0);
  let ndl = max(dot(n, l), 0.0);
  let aoBent = evalFastSdfAO(in.worldPos + n * 0.005, n);
  let shadow = min(evalPcfShadow(in.worldPos, n), mix(0.35, 1.0, evalSdfSoftShadow(in.worldPos + n * 0.006, l)));
  let gi = evalSdfGI(in.worldPos + n * 0.006, n, aoBent.xyz);
  let gobo = evalCanopyGobo(in.worldPos, l, u.padExtra);

  let hemiDay = 0.08 + 0.08 * (aoBent.y * 0.5 + 0.5);
  let hemiMoon = vec3<f32>(0.015, 0.022, 0.048) + vec3<f32>(0.035, 0.055, 0.115) * (aoBent.y * 0.5 + 0.5);
  let hemi = mix(vec3<f32>(hemiDay), hemiMoon, noct) * mix(1.0, 0.12, noct);
  let sunTint = mix(vec3<f32>(1.0), vec3<f32>(0.36, 0.48, 0.78), noct) * mix(1.0, 0.07, noct);
  let albedo = srgbToLinear(in.color.rgb);
  let iblBounce = evalStudioIbl(reflect(-v, n), 0.74, u.padExtra.w) * 0.045 * aoBent.w
    * mix(vec3<f32>(1.0), vec3<f32>(0.35, 0.48, 0.80) * 0.09, noct);
  let giWeight = mix(0.48, 1.25, noct);
  let col = albedo * (0.86 + 0.28 * grain) * ((hemi + sunTint * 0.92 * ndl * mix(0.22, 1.0, shadow) * gobo) * aoBent.w + gi * giWeight) + iblBounce;
  return vec4<f32>(col, 1.0);
}
`;

// ----------------------------------------------------------------------------
// 5. AIRBORNE SPORE CLOUD SHADER (WGSL — retained for WebGL2 fallback parity)
// ----------------------------------------------------------------------------
export const SPORE_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;

struct SporeVertIn {
  @location(0) corner: vec2<f32>,
  @location(1) posAge: vec4<f32>,
  @location(2) velLift: vec4<f32>,
};

struct SporeVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) liftM: f32,
  @location(2) activeFlag: f32,
};

@vertex
fn vs_main(v: SporeVertIn) -> SporeVertOut {
  var out: SporeVertOut;
  if (v.posAge.y < -0.5) {
    out.clipPos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = v.corner;
    out.liftM = 0.0;
    out.activeFlag = 0.0;
    return out;
  }
  let toCam = normalize(u.cameraPos.xyz - v.posAge.xyz);
  let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), toCam));
  let up = cross(toCam, right);
  let size = 0.0022;
  let worldPos = v.posAge.xyz + (right * v.corner.x + up * v.corner.y) * size;
  out.clipPos = u.viewProj * vec4<f32>(worldPos, 1.0);
  out.uv = v.corner;
  out.liftM = v.velLift.w;
  out.activeFlag = 1.0;
  return out;
}

@fragment
fn fs_main(in: SporeVertOut) -> @location(0) vec4<f32> {
  if (in.activeFlag < 0.5) {
    discard;
  }
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) {
    discard;
  }
  let soft = exp(-r2 * 2.8);
  let lifted = clamp(in.liftM / 0.03, 0.0, 1.0);
  let baseCol = mix(max(u.sporePrintColor.rgb * 1.4, vec3<f32>(0.55, 0.50, 0.44)), vec3<f32>(0.42, 0.86, 0.98), lifted * 0.72);
  let core = mix(baseCol, vec3<f32>(1.0, 0.96, 0.86), smoothstep(0.6, 1.0, soft) * 0.45);
  let noct = max(clamp(u.biolumInfo.y, 0.0, 1.0), smoothstep(2.4, 2.9, u.padExtra.x));
  return vec4<f32>(core * mix(1.0, 0.22, noct), soft * 0.86);
}
`;

// ----------------------------------------------------------------------------
// 6. BUOYANT EVAPORATIVE WIND STREAMLINES SHADER (WGSL)
// ----------------------------------------------------------------------------
export const STREAMLINE_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;

struct StreamVertIn {
  @location(0) pos: vec3<f32>,
  @location(1) color: vec4<f32>,
};

struct StreamVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(v: StreamVertIn) -> StreamVertOut {
  var out: StreamVertOut;
  out.clipPos = u.viewProj * vec4<f32>(v.pos, 1.0);
  out.color = v.color;
  return out;
}

@fragment
fn fs_main(in: StreamVertOut) -> @location(0) vec4<f32> {
  // Under Nocturnal moonlight the flow overlay stays legible but drops to a cool, dim register
  let noct = max(clamp(u.biolumInfo.y, 0.0, 1.0), smoothstep(2.4, 2.9, u.padExtra.x));
  let luma = dot(in.color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
  let moonCol = mix(in.color.rgb, vec3<f32>(luma) * vec3<f32>(0.62, 0.78, 1.0), 0.65) * 0.26;
  return vec4<f32>(mix(in.color.rgb, moonCol, noct), in.color.a * mix(1.0, 0.7, noct));
}
`;

// ============================================================================
// EQUIVALENT GLSL ES 3.00 SHADERS (WebGL2 Fallback ONLY if navigator.gpu unavailable)
// ============================================================================

const GLSL_COMMON_FS = /* glsl */ `
uniform vec4 uCameraPos;
uniform vec4 uLightDir;
uniform vec4 uSoilBoundsMin;
uniform vec4 uSoilBoundsMax;
uniform vec4 uSliceParams;
uniform vec4 uSssColor;
uniform vec4 uSporePrintColor;
uniform vec4 uSoilDims;

bool isClippedBySlice(vec3 p) {
  if (uSliceParams.x < 0.5) return false;
  if (uSliceParams.y < 0.5) return p.x > uSliceParams.z + 0.0005;
  return p.z > uSliceParams.z + 0.0005;
}

float hash31(vec3 p) {
  vec3 q = fract(p * vec3(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yxz + 33.33);
  return fract((q.x + q.y) * q.z);
}

float valueNoise3D(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  float n000 = hash31(i + vec3(0.0, 0.0, 0.0));
  float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
  float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
  float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
  float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
  float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
  float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
  float n111 = hash31(i + vec3(1.0, 1.0, 1.0));
  return mix(
    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
    u.z
  );
}

vec3 colormapViridis(float tIn) {
  float t = clamp(tIn, 0.0, 1.0);
  vec3 c0 = vec3(0.2777, 0.0054, 0.3340);
  vec3 c1 = vec3(0.1050, 1.4046, 1.3845);
  vec3 c2 = vec3(-0.3308, 0.2148, 0.0950);
  vec3 c3 = vec3(-4.6342, -5.7991, -19.3324);
  vec3 c4 = vec3(6.2282, 14.1799, 56.6905);
  vec3 c5 = vec3(4.7763, -13.7451, -65.3530);
  vec3 c6 = vec3(-5.4354, 4.6458, 26.3124);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0, 1.0);
}

vec3 colormapInferno(float tIn) {
  float t = clamp(tIn, 0.0, 1.0);
  vec3 c0 = vec3(0.0002, 0.0016, 0.0139);
  vec3 c1 = vec3(0.1065, 0.0639, 0.2297);
  vec3 c2 = vec3(0.4240, 0.0920, 0.4320);
  vec3 c3 = vec3(0.7356, 0.2159, 0.3302);
  vec3 c4 = vec3(0.9554, 0.4876, 0.1072);
  vec3 c5 = vec3(0.9883, 0.9983, 0.6449);
  float s = t * 5.0;
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  if (s < 3.0) return mix(c2, c3, s - 2.0);
  if (s < 4.0) return mix(c3, c4, s - 3.0);
  return mix(c4, c5, s - 4.0);
}

vec3 acesFilmic(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 srgbEncode(vec3 lin) {
  return pow(clamp(lin, 0.0, 1.0), vec3(1.0 / 2.2));
}
`;

export const SOIL_GLSL_VS = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aFlags;
uniform mat4 uViewProj;
out vec3 vWorldPos;
out vec3 vWorldNormal;
out float vFlags;
void main() {
  gl_Position = uViewProj * vec4(aPos, 1.0);
  vWorldPos = aPos;
  vWorldNormal = aNormal;
  vFlags = aFlags;
}
`;

export const SOIL_GLSL_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler3D;
${GLSL_COMMON_FS}
uniform sampler3D uSoilTex;
uniform sampler2D uSporePrintTex;
in vec3 vWorldPos;
in vec3 vWorldNormal;
in float vFlags;
out vec4 fragColor;

vec3 soilMaterialColor(int matId, vec3 worldPos, float wetness, float o2Norm, bool isTop) {
  float gFine = valueNoise3D(worldPos * 165.0);
  float gMed = valueNoise3D(worldPos * 55.0);
  float gStrata = valueNoise3D(vec3(worldPos.x * 16.0, worldPos.y * 85.0, worldPos.z * 16.0));
  vec3 dryCol = vec3(0.24, 0.17, 0.11);
  vec3 wetCol = vec3(0.08, 0.055, 0.035);
  if (matId == 0) {
    float speck = (gFine - 0.5) * 0.16 + (gMed - 0.5) * 0.08;
    dryCol = vec3(0.62, 0.52, 0.36) + vec3(speck);
    wetCol = vec3(0.29, 0.23, 0.15) + vec3(speck * 0.55);
  } else if (matId == 1) {
    float band = (gStrata - 0.5) * 0.09 + (gFine - 0.5) * 0.06;
    dryCol = vec3(0.27, 0.19, 0.12) + vec3(band);
    wetCol = vec3(0.095, 0.065, 0.042) + vec3(band * 0.5);
  } else if (matId == 2) {
    float slick = (gMed - 0.5) * 0.06;
    dryCol = vec3(0.39, 0.25, 0.19) + vec3(slick);
    wetCol = vec3(0.18, 0.11, 0.085) + vec3(slick * 0.5);
  } else if (matId == 3) {
    float litter = (gFine - 0.5) * 0.09;
    dryCol = vec3(0.14, 0.10, 0.07) + vec3(litter);
    wetCol = vec3(0.045, 0.032, 0.022) + vec3(litter * 0.45);
    if (isTop) {
      float moss = smoothstep(0.42, 0.76, gMed * 0.65 + gFine * 0.35);
      dryCol = mix(dryCol, vec3(0.10, 0.21, 0.075), moss * 0.68);
      wetCol = mix(wetCol, vec3(0.055, 0.14, 0.045), moss * 0.78);
    }
  } else if (matId == 4) {
    float rings = sin(length(worldPos.yz * 72.0 + vec2(3.0, 5.0)) + gMed * 4.5) * 0.5 + 0.5;
    float fiber = valueNoise3D(vec3(worldPos.x * 22.0, worldPos.y * 140.0, worldPos.z * 140.0));
    float woodVar = (rings * 0.6 + fiber * 0.4 - 0.5) * 0.15;
    dryCol = vec3(0.36, 0.22, 0.12) + vec3(woodVar);
    wetCol = vec3(0.15, 0.085, 0.045) + vec3(woodVar * 0.6);
  } else {
    float grain = (gFine - 0.5) * 0.15 + (gMed - 0.5) * 0.08;
    dryCol = vec3(0.36, 0.38, 0.40) + vec3(grain);
    wetCol = vec3(0.21, 0.23, 0.25) + vec3(grain * 0.7);
    if (worldPos.y > -0.025) {
      float moss = smoothstep(0.48, 0.78, gMed);
      dryCol = mix(dryCol, vec3(0.13, 0.24, 0.09), moss * 0.58);
    }
  }
  vec3 col = mix(dryCol, wetCol, clamp(wetness, 0.0, 1.0));
  float hypoxia = smoothstep(0.32, 0.05, o2Norm) * step(float(matId), 4.5);
  return mix(col, vec3(0.06, 0.11, 0.16), hypoxia * 0.35);
}

void main() {
  if (vFlags > 1.5) {
    vec3 n = normalize(vWorldNormal);
    float ndl = max(dot(n, normalize(uLightDir.xyz)), 0.0);
    fragColor = vec4(srgbEncode(acesFilmic(vec3(0.026, 0.034, 0.046) * (0.55 + 0.65 * ndl))), 1.0);
    return;
  }
  bool isSlice = vFlags > 0.5;
  if (!isSlice && isClippedBySlice(vWorldPos)) discard;

  vec3 span = max(uSoilBoundsMax.xyz - uSoilBoundsMin.xyz, vec3(1e-4));
  vec3 normPos = clamp((vWorldPos - uSoilBoundsMin.xyz) / span, 0.0, 0.9999);
  vec3 uvw = vec3(normPos.x, normPos.z, normPos.y);
  vec3 dimsF = vec3(uSoilDims.x, uSoilDims.z, uSoilDims.y);
  ivec3 dimsI = ivec3(int(uSoilDims.x), int(uSoilDims.z), int(uSoilDims.y));

  vec4 smoothSample = texture(uSoilTex, uvw);
  float wetness = smoothSample.g;
  float xrayVal = smoothSample.b;
  float o2Norm = smoothSample.a;

  float w1 = valueNoise3D(vWorldPos * 26.0 + vec3(1.7, 3.1, 5.3));
  float w2 = valueNoise3D(vWorldPos * 26.0 + vec3(11.2, 7.4, 2.9));
  float w3 = valueNoise3D(vWorldPos * 26.0 + vec3(4.6, 9.8, 13.1));
  vec3 warp = (vec3(w1, w2, w3) - 0.5) * (0.95 / dimsF);
  vec3 uvwMat = clamp(uvw + warp, 0.0, 0.9999);
  vec3 gridPos = uvwMat * dimsF - 0.5;
  ivec3 i0 = clamp(ivec3(floor(gridPos)), ivec3(0), dimsI - 1);
  ivec3 i1 = min(i0 + 1, dimsI - 1);
  vec3 f = smoothstep(0.0, 1.0, clamp(fract(gridPos), 0.0, 1.0));

  float wMat[6];
  for (int m = 0; m < 6; m++) wMat[m] = 0.0;
  for (int dz = 0; dz < 2; dz++) {
    float tz = dz == 1 ? f.z : 1.0 - f.z;
    int cz = dz == 1 ? i1.z : i0.z;
    for (int dy = 0; dy < 2; dy++) {
      float ty = dy == 1 ? f.y : 1.0 - f.y;
      int cy = dy == 1 ? i1.y : i0.y;
      for (int dx = 0; dx < 2; dx++) {
        float tx = dx == 1 ? f.x : 1.0 - f.x;
        int cx = dx == 1 ? i1.x : i0.x;
        int m = clamp(int(floor(texelFetch(uSoilTex, ivec3(cx, cy, cz), 0).r * 255.0 + 0.5)), 0, 5);
        wMat[m] += tx * ty * tz;
      }
    }
  }

  vec3 n = normalize(vWorldNormal);
  bool isTop = n.y > 0.5;
  vec3 albedo = vec3(0.0);
  for (int m = 0; m < 6; m++) {
    if (wMat[m] > 0.001) {
      albedo += soilMaterialColor(m, vWorldPos, wetness, o2Norm, isTop) * wMat[m];
    }
  }
  float stoneFrac = wMat[5];

  int xrayMode = int(floor(uSoilBoundsMax.w + 0.5));
  vec3 emissive = vec3(0.0);
  if (xrayMode == 1) {
    vec3 mapped = colormapViridis(xrayVal);
    albedo = mix(albedo * 0.25, mapped, isSlice ? 0.92 : 0.78);
    emissive = mapped * (isSlice ? 0.22 : 0.10);
  } else if (xrayMode == 2) {
    vec3 mapped = colormapInferno(xrayVal);
    albedo = mix(albedo * 0.22, mapped, isSlice ? 0.94 : 0.80);
    emissive = mapped * (isSlice ? 0.24 : 0.12);
  } else if (xrayMode == 3) {
    vec3 mapped = colormapViridis(xrayVal);
    albedo = mix(albedo * 0.35, mapped * 0.55, isSlice ? 0.65 : 0.45);
  } else if (xrayMode == 4) {
    albedo = mix(albedo * 0.35, vec3(0.05, 0.08, 0.12), 0.55);
  } else if (xrayMode == 5) {
    albedo *= 0.72;
  }

  if (isTop && xrayMode == 0) {
    float spVal = texture(uSporePrintTex, vec2(normPos.x, normPos.z)).r;
    albedo = mix(albedo, uSporePrintColor.rgb, clamp(spVal * 1.45, 0.0, 0.92));
  }

  vec3 l = normalize(uLightDir.xyz);
  vec3 v = normalize(uCameraPos.xyz - vWorldPos);
  vec3 h = normalize(l + v);
  float ndl = max(dot(n, l), 0.0);
  float hemi = 0.26 + 0.20 * (n.y * 0.5 + 0.5);
  float wetSpec = pow(max(dot(n, h), 0.0), 36.0) * wetness * 0.20 * (1.0 - stoneFrac);
  float depthAo = mix(0.74, 1.0, smoothstep(0.0, 0.28, normPos.y));
  vec3 color = albedo * (hemi + 0.88 * ndl) * depthAo + vec3(wetSpec) + emissive;

  if (isSlice) {
    float edgeDist = uSliceParams.y < 0.5
      ? min(min(normPos.y, 1.0 - normPos.y), min(normPos.z, 1.0 - normPos.z))
      : min(min(normPos.y, 1.0 - normPos.y), min(normPos.x, 1.0 - normPos.x));
    float rim = 1.0 - smoothstep(0.004, 0.016, edgeDist);
    color = mix(color, vec3(0.32, 0.78, 0.98), rim * 0.85);
  }

  fragColor = vec4(xrayMode == 0 ? srgbEncode(acesFilmic(color)) : srgbEncode(color), 1.0);
}
`;

export const MYCELIUM_GLSL_VS = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aSegA;
layout(location = 2) in vec3 aSegB;
layout(location = 3) in vec2 aRadiusTurgor;
uniform mat4 uViewProj;
uniform vec4 uCameraPos;
out vec3 vWorldPos;
out float vAcross;
out float vTurgorNorm;
void main() {
  vec3 p = mix(aSegA, aSegB, aCorner.x);
  vec3 segDir = normalize(aSegB - aSegA + vec3(0.0, 1e-6, 0.0));
  vec3 toCam = normalize(uCameraPos.xyz - p);
  vec3 side = cross(segDir, toCam);
  float sLen = length(side);
  side = sLen < 1e-4 ? vec3(1.0, 0.0, 0.0) : side / sLen;
  float visRadius = clamp(aRadiusTurgor.x * 4.6, 0.00072, 0.0030);
  vec3 worldPos = p + side * (aCorner.y * visRadius);
  gl_Position = uViewProj * vec4(worldPos, 1.0);
  vWorldPos = worldPos;
  vAcross = aCorner.y;
  vTurgorNorm = clamp(aRadiusTurgor.y, 0.0, 1.0);
}
`;

export const MYCELIUM_GLSL_FS = /* glsl */ `#version 300 es
precision highp float;
${GLSL_COMMON_FS}
in vec3 vWorldPos;
in float vAcross;
in float vTurgorNorm;
out vec4 fragColor;
void main() {
  if (isClippedBySlice(vWorldPos)) discard;
  float radial = 1.0 - clamp(abs(vAcross), 0.0, 1.0);
  float cylShade = sqrt(max(0.0, 1.0 - vAcross * vAcross));
  int xrayMode = int(floor(uSoilBoundsMax.w + 0.5));
  vec3 col;
  if (xrayMode == 3) {
    col = colormapViridis(vTurgorNorm) * (0.65 + 0.55 * cylShade);
  } else {
    vec3 dryHypha = vec3(0.42, 0.36, 0.28);
    vec3 turgidHypha = vec3(0.98, 0.93, 0.76);
    vec3 glowTip = vec3(1.0, 0.85, 0.45);
    col = mix(dryHypha, turgidHypha, vTurgorNorm);
    col = mix(col, glowTip, pow(vTurgorNorm, 2.2) * 0.45);
    col *= (0.55 + 0.65 * cylShade);
  }
  float alpha = smoothstep(0.0, 0.28, radial) * uSliceParams.w;
  if (alpha < 0.02) discard;
  fragColor = vec4(srgbEncode(col), alpha);
}
`;

export const MUSHROOM_GLSL_VS = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aMetrics;
uniform mat4 uViewProj;
out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec3 vBaseColor;
out vec3 vMetrics;
void main() {
  gl_Position = uViewProj * vec4(aPos, 1.0);
  vWorldPos = aPos;
  vWorldNormal = aNormal;
  vBaseColor = aColor;
  vMetrics = aMetrics;
}
`;

export const MUSHROOM_GLSL_FS = /* glsl */ `#version 300 es
precision highp float;
${GLSL_COMMON_FS}
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec3 vBaseColor;
in vec3 vMetrics;
out vec4 fragColor;
void main() {
  if (isClippedBySlice(vWorldPos)) discard;
  vec3 rawN = normalize(vWorldNormal);
  vec3 n = gl_FrontFacing ? rawN : -rawN;
  vec3 l = normalize(uLightDir.xyz);
  vec3 v = normalize(uCameraPos.xyz - vWorldPos);
  vec3 h = normalize(l + v);
  float stressPa = vMetrics.x;
  float turgorPa = vMetrics.y;
  int partId = int(floor(vMetrics.z + 0.5));
  int xrayMode = int(floor(uSoilBoundsMax.w + 0.5));

  if (xrayMode == 4) {
    float t = clamp(stressPa / 8.5e5, 0.0, 1.0);
    vec3 col = colormapInferno(t);
    float contour = abs(fract(t * 10.0) - 0.5);
    float line = 1.0 - smoothstep(0.02, 0.06, contour);
    float ndl = max(dot(n, l), 0.0);
    col = col * (0.58 + 0.55 * ndl) + vec3(0.15) * line;
    fragColor = vec4(srgbEncode(col), 1.0);
    return;
  }
  if (xrayMode == 3) {
    float t = clamp(turgorPa / 7.5e5, 0.0, 1.0);
    float ndl = max(dot(n, l), 0.0);
    fragColor = vec4(srgbEncode(colormapViridis(t) * (0.55 + 0.55 * ndl)), 1.0);
    return;
  }

  float micro = valueNoise3D(vWorldPos * 260.0);
  vec3 albedo = vBaseColor * (0.93 + 0.14 * micro);
  if (!gl_FrontFacing) {
    albedo = mix(albedo, vec3(0.93, 0.90, 0.82), 0.65);
  }

  float ao = 1.0;
  if (partId == 2) {
    ao = mix(0.45, 0.94, clamp(0.5 + 0.5 * abs(n.x * 0.5 + n.z * 0.5), 0.0, 1.0));
    ao *= mix(0.66, 1.0, smoothstep(-0.85, 0.25, n.y));
  } else if (partId == 0) {
    ao = mix(0.76, 1.0, smoothstep(0.0, 0.025, vWorldPos.y));
  }

  float wrapNdl = clamp((dot(n, l) + 0.30) / 1.30, 0.0, 1.0);
  vec3 skyFill = mix(vec3(0.08, 0.075, 0.070), vec3(0.22, 0.27, 0.33), n.y * 0.5 + 0.5);
  float backLight = pow(clamp(dot(v, -l + n * 0.38), 0.0, 1.0), 2.5);
  float thicknessFactor = (partId == 2 || partId == 3) ? 0.98 : 0.55;
  vec3 sss = uSssColor.rgb * backLight * 0.54 * thicknessFactor;
  float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.2);
  float spec = pow(max(dot(n, h), 0.0), 32.0) * (partId == 2 ? 0.07 : 0.20);
  vec3 rim = vec3(0.85, 0.88, 0.94) * fresnel * 0.14;
  vec3 color = albedo * (skyFill + vec3(1.06, 1.01, 0.92) * wrapNdl) * ao + sss + vec3(spec) + rim;
  fragColor = vec4(srgbEncode(acesFilmic(color)), partId == 3 ? 0.86 : 1.0);
}
`;

export const SOLID_GLSL_VS = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aColor;
uniform mat4 uViewProj;
out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec4 vColor;
void main() {
  gl_Position = uViewProj * vec4(aPos, 1.0);
  vWorldPos = aPos;
  vWorldNormal = aNormal;
  vColor = aColor;
}
`;

export const SOLID_GLSL_FS = /* glsl */ `#version 300 es
precision highp float;
${GLSL_COMMON_FS}
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec4 vColor;
out vec4 fragColor;
void main() {
  float nLen = length(vWorldNormal);
  if (nLen < 0.05) {
    fragColor = vColor;
    return;
  }
  vec3 n = vWorldNormal / nLen;
  vec3 l = normalize(uLightDir.xyz);
  vec3 v = normalize(uCameraPos.xyz - vWorldPos);
  if (vColor.a < 0.95) {
    float fresnel = pow(1.0 - abs(dot(n, v)), 2.8);
    vec3 col = vColor.rgb + vec3(0.55, 0.72, 0.88) * fresnel * 0.42;
    fragColor = vec4(col, clamp(vColor.a + fresnel * 0.14, 0.0, 0.45));
    return;
  }
  float grain = valueNoise3D(vWorldPos * 150.0);
  float ndl = max(dot(n, l), 0.0);
  float hemi = 0.25 + 0.20 * (n.y * 0.5 + 0.5);
  fragColor = vec4(srgbEncode(acesFilmic(vColor.rgb * (0.86 + 0.28 * grain) * (hemi + 0.85 * ndl))), 1.0);
}
`;

export const SPORE_GLSL_VS = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec4 aPosAge;
layout(location = 2) in vec4 aVelLift;
uniform mat4 uViewProj;
uniform vec4 uCameraPos;
out vec2 vUv;
out float vLiftM;
out float vActive;
void main() {
  if (aPosAge.y < -0.5) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vUv = aCorner;
    vLiftM = 0.0;
    vActive = 0.0;
    return;
  }
  vec3 toCam = normalize(uCameraPos.xyz - aPosAge.xyz);
  vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), toCam));
  vec3 up = cross(toCam, right);
  vec3 worldPos = aPosAge.xyz + (right * aCorner.x + up * aCorner.y) * 0.0022;
  gl_Position = uViewProj * vec4(worldPos, 1.0);
  vUv = aCorner;
  vLiftM = aVelLift.w;
  vActive = 1.0;
}
`;

export const SPORE_GLSL_FS = /* glsl */ `#version 300 es
precision highp float;
${GLSL_COMMON_FS}
in vec2 vUv;
in float vLiftM;
in float vActive;
out vec4 fragColor;
void main() {
  if (vActive < 0.5) discard;
  float r2 = dot(vUv, vUv);
  if (r2 > 1.0) discard;
  float soft = exp(-r2 * 2.8);
  float lifted = clamp(vLiftM / 0.03, 0.0, 1.0);
  vec3 baseCol = mix(max(uSporePrintColor.rgb * 1.4, vec3(0.55, 0.50, 0.44)), vec3(0.42, 0.86, 0.98), lifted * 0.72);
  vec3 core = mix(baseCol, vec3(1.0, 0.96, 0.86), smoothstep(0.6, 1.0, soft) * 0.45);
  fragColor = vec4(core, soft * 0.86);
}
`;

export const STREAMLINE_GLSL_VS = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uViewProj;
out vec4 vColor;
void main() {
  gl_Position = uViewProj * vec4(aPos, 1.0);
  vColor = aColor;
}
`;

export const STREAMLINE_GLSL_FS = /* glsl */ `#version 300 es
precision highp float;
in vec4 vColor;
out vec4 fragColor;
void main() {
  fragColor = vColor;
}
`;
