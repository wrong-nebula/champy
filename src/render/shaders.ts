// ============================================================================
// SPOROCARP — Hand-Crafted WGSL (WebGPU) & GLSL ES 3.00 (WebGL2) Shaders
// Implements:
//   1. Soil Cross-Section, Razor Slice Plane, Spore Print & Scientific Colormaps
//   2. 3D Branching Mycelial Network (Billboarded Filaments + Subterranean Glow)
//   3. Waxy Chitin Subsurface-Scattering (SSS) Mushroom, Gills & Veil Shader
//   4. Solid Obstacles (Twigs/Stones) & Museum Glass Terrarium Architecture
//   5. Airborne Spore Cloud (Instanced Soft Particles)
//   6. Dressaire Evaporative Buoyant Wind Streamlines
// ============================================================================

// Shared WGSL struct + colormap + noise + slice helpers
const WGSL_COMMON = /* wgsl */ `
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

// Calibrated Viridis scientific colormap (polynomial approximation)
fn colormapViridis(tIn: f32) -> vec3<f32> {
  let t = clamp(tIn, 0.0, 1.0);
  let c0 = vec3<f32>(0.2777, 0.0054, 0.3340);
  let c1 = vec3<f32>(0.1050, 1.4046, 1.3845);
  let c2 = vec3<f32>(-0.3308, 0.2148, 0.0950);
  let c3 = vec3<f32>(-4.6342, -5.7991, -19.3324);
  let c4 = vec3<f32>(6.2282, 14.1799, 56.6905);
  let c5 = vec3<f32>(4.7763, -13.7451, -65.3530);
  let c6 = vec3<f32>(-5.4354, 4.6458, 26.3124);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Calibrated Inferno scientific colormap
fn colormapInferno(tIn: f32) -> vec3<f32> {
  let t = clamp(tIn, 0.0, 1.0);
  let c0 = vec3<f32>(0.0002, 0.0016, 0.0139);
  let c1 = vec3<f32>(0.1065, 0.0639, 0.2297);
  let c2 = vec3<f32>(0.4240, 0.0920, 0.4320);
  let c3 = vec3<f32>(0.7356, 0.2159, 0.3302);
  let c4 = vec3<f32>(0.9554, 0.4876, 0.1072);
  let c5 = vec3<f32>(0.9883, 0.9983, 0.6449);
  let s = t * 5.0;
  if (s < 1.0) { return mix(c0, c1, s); }
  if (s < 2.0) { return mix(c1, c2, s - 1.0); }
  if (s < 3.0) { return mix(c2, c3, s - 2.0); }
  if (s < 4.0) { return mix(c3, c4, s - 3.0); }
  return mix(c4, c5, s - 4.0);
}

// Calibrated Coolwarm diverging colormap (0 = cool blue, 0.5 = neutral, 1 = warm amber/red)
fn colormapCoolwarm(tIn: f32) -> vec3<f32> {
  let t = clamp(tIn, 0.0, 1.0);
  let cool = vec3<f32>(0.16, 0.46, 0.88);
  let mid = vec3<f32>(0.84, 0.87, 0.90);
  let warm = vec3<f32>(0.92, 0.34, 0.16);
  if (t < 0.5) {
    return mix(cool, mid, t * 2.0);
  }
  return mix(mid, warm, (t - 0.5) * 2.0);
}

fn srgbEncode(lin: vec3<f32>) -> vec3<f32> {
  return pow(clamp(lin, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}
`;

// ----------------------------------------------------------------------------
// 1. SOIL CROSS-SECTION & RAZOR SLICE SHADER (WGSL)
// ----------------------------------------------------------------------------
export const SOIL_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var soilTex: texture_3d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var sporePrintTex: texture_2d<f32>;

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

fn soilMaterialColor(matId: i32, worldPos: vec3<f32>, wetness: f32, o2Norm: f32, isTop: bool) -> vec3<f32> {
  let gFine = valueNoise3D(worldPos * 165.0);
  let gMed = valueNoise3D(worldPos * 55.0);
  let gStrata = valueNoise3D(vec3<f32>(worldPos.x * 16.0, worldPos.y * 85.0, worldPos.z * 16.0));

  var dryCol = vec3<f32>(0.24, 0.17, 0.11);
  var wetCol = vec3<f32>(0.08, 0.055, 0.035);

  if (matId == 0) {
    // Sand: speckled warm quartz
    let speck = (gFine - 0.5) * 0.16 + (gMed - 0.5) * 0.08;
    dryCol = vec3<f32>(0.62, 0.52, 0.36) + vec3<f32>(speck);
    wetCol = vec3<f32>(0.29, 0.23, 0.15) + vec3<f32>(speck * 0.55);
  } else if (matId == 1) {
    // Forest Silt-Loam: stratified chocolate earth
    let band = (gStrata - 0.5) * 0.08 + (gFine - 0.5) * 0.06;
    dryCol = vec3<f32>(0.27, 0.19, 0.12) + vec3<f32>(band);
    wetCol = vec3<f32>(0.095, 0.065, 0.042) + vec3<f32>(band * 0.5);
  } else if (matId == 2) {
    // Heavy Clay: terracotta-slate smooth dense pockets
    let slick = (gMed - 0.5) * 0.05;
    dryCol = vec3<f32>(0.39, 0.25, 0.19) + vec3<f32>(slick);
    wetCol = vec3<f32>(0.18, 0.11, 0.085) + vec3<f32>(slick * 0.5);
  } else if (matId == 3) {
    // Forest Humus / Litter: deep rich espresso organic mulch + moss flecks on top
    let litter = (gFine - 0.5) * 0.08;
    dryCol = vec3<f32>(0.14, 0.10, 0.07) + vec3<f32>(litter);
    wetCol = vec3<f32>(0.045, 0.032, 0.022) + vec3<f32>(litter * 0.45);
    if (isTop) {
      let moss = smoothstep(0.46, 0.78, gMed * 0.65 + gFine * 0.35);
      dryCol = mix(dryCol, vec3<f32>(0.11, 0.20, 0.08), moss * 0.62);
      wetCol = mix(wetCol, vec3<f32>(0.06, 0.13, 0.045), moss * 0.72);
    }
  } else if (matId == 4) {
    // Rotting Oak Log: fibrous grain + concentric heartwood rings
    let rings = sin(length(worldPos.yz * 72.0 + vec2<f32>(3.0, 5.0)) + gMed * 4.5) * 0.5 + 0.5;
    let fiber = valueNoise3D(vec3<f32>(worldPos.x * 22.0, worldPos.y * 140.0, worldPos.z * 140.0));
    let woodVar = (rings * 0.6 + fiber * 0.4 - 0.5) * 0.14;
    dryCol = vec3<f32>(0.36, 0.22, 0.12) + vec3<f32>(woodVar);
    wetCol = vec3<f32>(0.15, 0.085, 0.045) + vec3<f32>(woodVar * 0.6);
  } else {
    // Impermeable Stone: speckled granite + moss cap near surface
    let grain = (gFine - 0.5) * 0.15 + (gMed - 0.5) * 0.08;
    dryCol = vec3<f32>(0.36, 0.38, 0.40) + vec3<f32>(grain);
    wetCol = vec3<f32>(0.21, 0.23, 0.25) + vec3<f32>(grain * 0.7);
    if (worldPos.y > -0.025) {
      let moss = smoothstep(0.5, 0.8, gMed);
      dryCol = mix(dryCol, vec3<f32>(0.14, 0.24, 0.10), moss * 0.55);
    }
  }

  var col = mix(dryCol, wetCol, clamp(wetness, 0.0, 1.0));
  // Subtle anoxic slate-blue tint when soil pores become severely hypoxic
  let hypoxia = smoothstep(0.32, 0.05, o2Norm) * step(f32(matId), 4.5);
  col = mix(col, vec3<f32>(0.06, 0.11, 0.16), hypoxia * 0.35);
  return col;
}

@fragment
fn fs_main(in: SoilVertOut) -> @location(0) vec4<f32> {
  // Obsidian laboratory plinth underneath terrarium
  if (in.flags > 1.5) {
    let n = normalize(in.worldNormal);
    let l = normalize(u.lightDir.xyz);
    let ndl = max(dot(n, l), 0.0);
    let base = vec3<f32>(0.028, 0.036, 0.048);
    let col = base * (0.55 + 0.65 * ndl);
    return vec4<f32>(srgbEncode(col), 1.0);
  }

  let isSlice = in.flags > 0.5;
  if (!isSlice && isClippedBySlice(in.worldPos, u.sliceParams)) {
    discard;
  }

  let span = max(u.soilBoundsMax.xyz - u.soilBoundsMin.xyz, vec3<f32>(1e-4));
  let normPos = clamp((in.worldPos - u.soilBoundsMin.xyz) / span, vec3<f32>(0.0), vec3<f32>(0.9999));
  // Texture layout: width = nx (x), height = nz (z), depth = ny (y)
  let uvw = vec3<f32>(normPos.x, normPos.z, normPos.y);
  let dims = vec3<i32>(i32(u.soilDims.x), i32(u.soilDims.z), i32(u.soilDims.y));
  let texel = clamp(vec3<i32>(floor(uvw * vec3<f32>(dims))), vec3<i32>(0), dims - vec3<i32>(1));

  let discrete = textureLoad(soilTex, texel, 0);
  let smoothSample = textureSampleLevel(soilTex, linearSampler, uvw, 0.0);
  let matId = i32(floor(discrete.r * 255.0 + 0.5));
  let wetness = smoothSample.g;
  let xrayVal = smoothSample.b;
  let o2Norm = smoothSample.a;

  let n = normalize(in.worldNormal);
  let isTop = n.y > 0.5;
  var albedo = soilMaterialColor(matId, in.worldPos, wetness, o2Norm, isTop);

  let xrayMode = i32(floor(u.soilBoundsMax.w + 0.5));
  var emissive = vec3<f32>(0.0);
  if (xrayMode == 1) {
    // Matric potential psi (Viridis)
    let mapped = colormapViridis(xrayVal);
    albedo = mix(albedo * 0.25, mapped, select(0.78, 0.92, isSlice));
    emissive = mapped * select(0.10, 0.22, isSlice);
  } else if (xrayMode == 2) {
    // Carbon & Enzyme (Inferno)
    let mapped = colormapInferno(xrayVal);
    albedo = mix(albedo * 0.22, mapped, select(0.80, 0.94, isSlice));
    emissive = mapped * select(0.12, 0.24, isSlice);
  } else if (xrayMode == 3) {
    // Turgor P support view: muted scientific dark soil so glowing mycelium & mushroom turgor pop
    let mapped = colormapViridis(xrayVal);
    albedo = mix(albedo * 0.35, mapped * 0.55, select(0.45, 0.65, isSlice));
  } else if (xrayMode == 4) {
    // Shell stress focus: calm blueprint slate soil
    albedo = mix(albedo * 0.35, vec3<f32>(0.05, 0.08, 0.12), 0.55);
  } else if (xrayMode == 5) {
    // Air convection focus: slightly darkened natural soil so streamlines pop
    albedo = albedo * 0.72;
  }

  // Ground spore print overlay on soil top surface
  if (isTop && xrayMode == 0) {
    let spVal = textureSampleLevel(sporePrintTex, linearSampler, vec2<f32>(normPos.x, normPos.z), 0.0).r;
    let printAlpha = clamp(spVal * 1.45, 0.0, 0.92);
    albedo = mix(albedo, u.sporePrintColor.rgb, printAlpha);
  }

  // Lighting: key sun + cool lab skylight + subtle wet specular sheen
  let l = normalize(u.lightDir.xyz);
  let v = normalize(u.cameraPos.xyz - in.worldPos);
  let h = normalize(l + v);
  let ndl = max(dot(n, l), 0.0);
  let hemi = 0.26 + 0.20 * (n.y * 0.5 + 0.5);
  let wetSpec = pow(max(dot(n, h), 0.0), 36.0) * wetness * 0.22 * step(f32(matId), 4.5);

  // Subtle depth AO near bottom of cutaway box
  let depthAo = mix(0.74, 1.0, smoothstep(0.0, 0.28, normPos.y));
  var color = albedo * (hemi + 0.88 * ndl) * depthAo + vec3<f32>(wetSpec) + emissive;

  // Razor slice optical perimeter hairline highlight
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

  return vec4<f32>(srgbEncode(color), 1.0);
}
`;

// ----------------------------------------------------------------------------
// 2. 3D MYCELIAL NETWORK SHADER (WGSL)
// ----------------------------------------------------------------------------
export const MYCELIUM_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;

struct MycVertIn {
  @location(0) corner: vec2<f32>, // x in [0,1] along segment, y in [-1,1] across width
  @location(1) segA: vec3<f32>,
  @location(2) segB: vec3<f32>,
  @location(3) radiusTurgor: vec2<f32>, // x = radiusM, y = turgorNorm (0..1)
};

struct MycVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) across: f32,
  @location(2) turgorNorm: f32,
};

@vertex
fn vs_main(v: MycVertIn) -> MycVertOut {
  var out: MycVertOut;
  let p = mix(v.segA, v.segB, v.corner.x);
  let segDir = normalize(v.segB - v.segA + vec3<f32>(0.0, 1e-6, 0.0));
  let toCam = normalize(u.cameraPos.xyz - p);
  var side = cross(segDir, toCam);
  let sideLen = length(side);
  if (sideLen < 1e-4) {
    side = vec3<f32>(1.0, 0.0, 0.0);
  } else {
    side = side / sideLen;
  }
  // Visually legible cord width while scaling proportionally with physical rhizomorph radius
  let visRadius = clamp(v.radiusTurgor.x * 5.2, 0.00085, 0.0036);
  let worldPos = p + side * (v.corner.y * visRadius);
  out.clipPos = u.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.across = v.corner.y;
  out.turgorNorm = clamp(v.radiusTurgor.y, 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(in: MycVertOut) -> @location(0) vec4<f32> {
  if (isClippedBySlice(in.worldPos, u.sliceParams)) {
    discard;
  }
  let radial = 1.0 - clamp(abs(in.across), 0.0, 1.0);
  let cylShade = sqrt(max(0.0, 1.0 - in.across * in.across));
  let xrayMode = i32(floor(u.soilBoundsMax.w + 0.5));

  var col: vec3<f32>;
  if (xrayMode == 3) {
    col = colormapViridis(in.turgorNorm) * (0.65 + 0.55 * cylShade);
  } else {
    // Low turgor = muted dry ochre; high turgor = luminous ivory-gold hyphal cords
    let dryHypha = vec3<f32>(0.42, 0.36, 0.28);
    let turgidHypha = vec3<f32>(0.98, 0.93, 0.76);
    let glowTip = vec3<f32>(1.0, 0.85, 0.45);
    col = mix(dryHypha, turgidHypha, in.turgorNorm);
    col = mix(col, glowTip, pow(in.turgorNorm, 2.2) * 0.45);
    col = col * (0.55 + 0.65 * cylShade);
  }

  let alpha = smoothstep(0.0, 0.25, radial) * u.sliceParams.w;
  if (alpha < 0.02) {
    discard;
  }
  return vec4<f32>(srgbEncode(col), alpha);
}
`;

// ----------------------------------------------------------------------------
// 3. WAXY CHITIN SUBSURFACE-SCATTERING (SSS) MUSHROOM SHADER (WGSL)
// ----------------------------------------------------------------------------
export const MUSHROOM_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;

struct MushVertIn {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  @location(3) metrics: vec3<f32>, // x = stressVM (Pa), y = turgorPa (Pa), z = partId
};

struct MushVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) metrics: vec3<f32>,
};

@vertex
fn vs_main(v: MushVertIn) -> MushVertOut {
  var out: MushVertOut;
  out.clipPos = u.viewProj * vec4<f32>(v.pos, 1.0);
  out.worldPos = v.pos;
  out.worldNormal = v.normal;
  out.baseColor = v.color;
  out.metrics = v.metrics;
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

  // Scientific X-Ray override for Shell Stress or Turgor P
  if (xrayMode == 4) {
    // Shell von Mises stress: 0 .. 850 kPa -> Inferno + subtle iso-stress bands
    let t = clamp(stressPa / 8.5e5, 0.0, 1.0);
    var col = colormapInferno(t);
    let contour = abs(fract(t * 10.0) - 0.5);
    let line = 1.0 - smoothstep(0.02, 0.06, contour);
    let ndl = max(dot(n, l), 0.0);
    col = col * (0.58 + 0.55 * ndl) + vec3<f32>(0.15) * line;
    return vec4<f32>(srgbEncode(col), 1.0);
  }
  if (xrayMode == 3) {
    // Internal Turgor Pressure P: 0 .. 750 kPa -> Viridis
    let t = clamp(turgorPa / 7.5e5, 0.0, 1.0);
    var col = colormapViridis(t);
    let ndl = max(dot(n, l), 0.0);
    col = col * (0.55 + 0.55 * ndl);
    return vec4<f32>(srgbEncode(col), 1.0);
  }

  // Organic micro-grain on cap/stipe chitin
  let micro = valueNoise3D(in.worldPos * 260.0);
  var albedo = in.baseColor * (0.94 + 0.12 * micro);

  // Interior flesh tint on sliced backfaces
  if (!frontFacing) {
    albedo = mix(albedo, vec3<f32>(0.93, 0.90, 0.82), 0.65);
  }

  // Gill cavity ambient occlusion for dramatic hymenial relief
  var ao = 1.0;
  if (partId == 2) {
    // Deep gills: attenuate light inside downward-facing crevices
    ao = mix(0.42, 0.92, clamp(0.5 + 0.5 * abs(n.x * 0.5 + n.z * 0.5), 0.0, 1.0));
    ao *= mix(0.65, 1.0, smoothstep(-0.85, 0.25, n.y));
  } else if (partId == 0) {
    // Subtle darkening right beneath cap and at soil emergence
    ao = mix(0.76, 1.0, smoothstep(0.0, 0.025, in.worldPos.y));
  }

  // Wrap-diffuse + hemisphere fill
  let wrapNdl = clamp((dot(n, l) + 0.32) / 1.32, 0.0, 1.0);
  let skyFill = mix(vec3<f32>(0.10, 0.085, 0.075), vec3<f32>(0.24, 0.28, 0.34), n.y * 0.5 + 0.5);

  // Backlit subsurface scattering (waxy chitin translucency)
  let backLight = pow(clamp(dot(v, -l + n * 0.38), 0.0, 1.0), 2.6);
  let thicknessFactor = select(0.55, 0.95, partId == 2 || partId == 3);
  let sss = u.sssColor.rgb * backLight * 0.52 * thicknessFactor;

  // Waxy chitin specular + Fresnel rim
  let fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.2);
  let spec = pow(max(dot(n, h), 0.0), 28.0) * select(0.18, 0.07, partId == 2);
  let rim = vec3<f32>(0.85, 0.88, 0.94) * fresnel * 0.14;

  let color = albedo * (skyFill + vec3<f32>(0.96, 0.93, 0.87) * wrapNdl) * ao + sss + vec3<f32>(spec) + rim;
  let alpha = select(1.0, 0.86, partId == 3);
  return vec4<f32>(srgbEncode(color), alpha);
}
`;

// ----------------------------------------------------------------------------
// 4. SOLID OBSTACLES & GLASS TERRARIUM ARCHITECTURE SHADER (WGSL)
// ----------------------------------------------------------------------------
export const SOLID_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;

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
  // Unlit hairline/glass wireframe path when normal == 0
  if (nLen < 0.05) {
    return in.color;
  }

  let n = in.worldNormal / nLen;
  let l = normalize(u.lightDir.xyz);
  let v = normalize(u.cameraPos.xyz - in.worldPos);

  // Translucent museum glass pane
  if (in.color.a < 0.95) {
    let fresnel = pow(1.0 - abs(dot(n, v)), 2.8);
    let col = in.color.rgb + vec3<f32>(0.55, 0.72, 0.88) * fresnel * 0.42;
    let alpha = clamp(in.color.a + fresnel * 0.14, 0.0, 0.45);
    return vec4<f32>(col, alpha);
  }

  // Opaque twigs & stones with organic surface grain
  let grain = valueNoise3D(in.worldPos * 150.0);
  let ndl = max(dot(n, l), 0.0);
  let hemi = 0.25 + 0.20 * (n.y * 0.5 + 0.5);
  let col = in.color.rgb * (0.86 + 0.28 * grain) * (hemi + 0.85 * ndl);
  return vec4<f32>(srgbEncode(col), 1.0);
}
`;

// ----------------------------------------------------------------------------
// 5. AIRBORNE SPORE CLOUD SHADER (WGSL)
// ----------------------------------------------------------------------------
export const SPORE_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;

struct SporeVertIn {
  @location(0) corner: vec2<f32>,  // [-1, 1] quad
  @location(1) posAge: vec4<f32>,  // xyz = pos, w = ageSec
  @location(2) velLift: vec4<f32>, // xyz = vel, w = peakLiftAboveCapM
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
  // Hide inactive spores parked below soil
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
  // Ensure even dark spores (inkcap/parasol) read cleanly against obsidian lab background
  let lifted = clamp(in.liftM / 0.03, 0.0, 1.0);
  let baseCol = mix(max(u.sporePrintColor.rgb * 1.4, vec3<f32>(0.55, 0.50, 0.44)), vec3<f32>(0.42, 0.86, 0.98), lifted * 0.72);
  let core = mix(baseCol, vec3<f32>(1.0, 0.96, 0.86), smoothstep(0.6, 1.0, soft) * 0.45);
  return vec4<f32>(core, soft * 0.86);
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
  return in.color;
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
    float band = (gStrata - 0.5) * 0.08 + (gFine - 0.5) * 0.06;
    dryCol = vec3(0.27, 0.19, 0.12) + vec3(band);
    wetCol = vec3(0.095, 0.065, 0.042) + vec3(band * 0.5);
  } else if (matId == 2) {
    float slick = (gMed - 0.5) * 0.05;
    dryCol = vec3(0.39, 0.25, 0.19) + vec3(slick);
    wetCol = vec3(0.18, 0.11, 0.085) + vec3(slick * 0.5);
  } else if (matId == 3) {
    float litter = (gFine - 0.5) * 0.08;
    dryCol = vec3(0.14, 0.10, 0.07) + vec3(litter);
    wetCol = vec3(0.045, 0.032, 0.022) + vec3(litter * 0.45);
    if (isTop) {
      float moss = smoothstep(0.46, 0.78, gMed * 0.65 + gFine * 0.35);
      dryCol = mix(dryCol, vec3(0.11, 0.20, 0.08), moss * 0.62);
      wetCol = mix(wetCol, vec3(0.06, 0.13, 0.045), moss * 0.72);
    }
  } else if (matId == 4) {
    float rings = sin(length(worldPos.yz * 72.0 + vec2(3.0, 5.0)) + gMed * 4.5) * 0.5 + 0.5;
    float fiber = valueNoise3D(vec3(worldPos.x * 22.0, worldPos.y * 140.0, worldPos.z * 140.0));
    float woodVar = (rings * 0.6 + fiber * 0.4 - 0.5) * 0.14;
    dryCol = vec3(0.36, 0.22, 0.12) + vec3(woodVar);
    wetCol = vec3(0.15, 0.085, 0.045) + vec3(woodVar * 0.6);
  } else {
    float grain = (gFine - 0.5) * 0.15 + (gMed - 0.5) * 0.08;
    dryCol = vec3(0.36, 0.38, 0.40) + vec3(grain);
    wetCol = vec3(0.21, 0.23, 0.25) + vec3(grain * 0.7);
    if (worldPos.y > -0.025) {
      float moss = smoothstep(0.5, 0.8, gMed);
      dryCol = mix(dryCol, vec3(0.14, 0.24, 0.10), moss * 0.55);
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
    fragColor = vec4(srgbEncode(vec3(0.028, 0.036, 0.048) * (0.55 + 0.65 * ndl)), 1.0);
    return;
  }
  bool isSlice = vFlags > 0.5;
  if (!isSlice && isClippedBySlice(vWorldPos)) discard;

  vec3 span = max(uSoilBoundsMax.xyz - uSoilBoundsMin.xyz, vec3(1e-4));
  vec3 normPos = clamp((vWorldPos - uSoilBoundsMin.xyz) / span, 0.0, 0.9999);
  vec3 uvw = vec3(normPos.x, normPos.z, normPos.y);
  ivec3 dims = ivec3(int(uSoilDims.x), int(uSoilDims.z), int(uSoilDims.y));
  ivec3 texel = clamp(ivec3(floor(uvw * vec3(dims))), ivec3(0), dims - ivec3(1));

  vec4 discrete = texelFetch(uSoilTex, texel, 0);
  vec4 smoothSample = texture(uSoilTex, uvw);
  int matId = int(floor(discrete.r * 255.0 + 0.5));
  float wetness = smoothSample.g;
  float xrayVal = smoothSample.b;
  float o2Norm = smoothSample.a;

  vec3 n = normalize(vWorldNormal);
  bool isTop = n.y > 0.5;
  vec3 albedo = soilMaterialColor(matId, vWorldPos, wetness, o2Norm, isTop);

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
  float wetSpec = pow(max(dot(n, h), 0.0), 36.0) * wetness * 0.22 * step(float(matId), 4.5);
  float depthAo = mix(0.74, 1.0, smoothstep(0.0, 0.28, normPos.y));
  vec3 color = albedo * (hemi + 0.88 * ndl) * depthAo + vec3(wetSpec) + emissive;

  if (isSlice) {
    float edgeDist = uSliceParams.y < 0.5
      ? min(min(normPos.y, 1.0 - normPos.y), min(normPos.z, 1.0 - normPos.z))
      : min(min(normPos.y, 1.0 - normPos.y), min(normPos.x, 1.0 - normPos.x));
    float rim = 1.0 - smoothstep(0.004, 0.016, edgeDist);
    color = mix(color, vec3(0.32, 0.78, 0.98), rim * 0.85);
  }

  fragColor = vec4(srgbEncode(color), 1.0);
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
  float visRadius = clamp(aRadiusTurgor.x * 5.2, 0.00085, 0.0036);
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
  float alpha = smoothstep(0.0, 0.25, radial) * uSliceParams.w;
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
  vec3 albedo = vBaseColor * (0.94 + 0.12 * micro);
  if (!gl_FrontFacing) {
    albedo = mix(albedo, vec3(0.93, 0.90, 0.82), 0.65);
  }

  float ao = 1.0;
  if (partId == 2) {
    ao = mix(0.42, 0.92, clamp(0.5 + 0.5 * abs(n.x * 0.5 + n.z * 0.5), 0.0, 1.0));
    ao *= mix(0.65, 1.0, smoothstep(-0.85, 0.25, n.y));
  } else if (partId == 0) {
    ao = mix(0.76, 1.0, smoothstep(0.0, 0.025, vWorldPos.y));
  }

  float wrapNdl = clamp((dot(n, l) + 0.32) / 1.32, 0.0, 1.0);
  vec3 skyFill = mix(vec3(0.10, 0.085, 0.075), vec3(0.24, 0.28, 0.34), n.y * 0.5 + 0.5);
  float backLight = pow(clamp(dot(v, -l + n * 0.38), 0.0, 1.0), 2.6);
  float thicknessFactor = (partId == 2 || partId == 3) ? 0.95 : 0.55;
  vec3 sss = uSssColor.rgb * backLight * 0.52 * thicknessFactor;
  float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.2);
  float spec = pow(max(dot(n, h), 0.0), 28.0) * (partId == 2 ? 0.07 : 0.18);
  vec3 rim = vec3(0.85, 0.88, 0.94) * fresnel * 0.14;
  vec3 color = albedo * (skyFill + vec3(0.96, 0.93, 0.87) * wrapNdl) * ao + sss + vec3(spec) + rim;
  fragColor = vec4(srgbEncode(color), partId == 3 ? 0.86 : 1.0);
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
  fragColor = vec4(srgbEncode(vColor.rgb * (0.86 + 0.28 * grain) * (hemi + 0.85 * ndl)), 1.0);
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
