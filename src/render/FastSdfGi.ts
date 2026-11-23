// ============================================================================
// SPOROCARP — Fast-SDF-AO & 2-Bounce World Radiance Volume GI (WebGPU Compute)
// Inspired by wrong-nebula/Fast-SDF-AO:
//   - Live isotropic 3D FP16 Signed Distance Field + surface albedo (64x48x64)
//   - 4-tap normal-offset SDF Ambient Occlusion + SDF-gradient Bent Normals
//   - Sphere-traced soft SDF area shadows with contact hardening
//   - GPU-relit 2-bounce 3D Radiance Volume (64x48x64 rgba16float) for real-time
//     colored indirect diffuse GI (soil->gill bounce, cap->cap color bleeding)
// ============================================================================
import type { SimulationSnapshot, ViewState } from '../types/contract.ts';
import { SPECIES_PRESETS } from '../types/contract.ts';

export const SDF_NX = 64;
export const SDF_NY = 48;
export const SDF_NZ = 64;

/** Maximum analytical scene primitives packed per frame (stipe segments + caps + obstacles) */
const MAX_PRIMS = 192;
/** Maximum 525 nm bioluminescent emitters packed per frame */
const MAX_BIOLUM_LIGHTS = 32;
/** 12 floats (48 bytes) per primitive:
 *  a: vec4(ax, ay, az, radiusA)
 *  b: vec4(bx, by, bz, kind)       kind: 0=capsule, 1=capDome, 2=capFunnel, 3=capMorel, 4=sphere
 *  col: vec4(r, g, b, extraParam)  extraParam: capThickness/height
 */
const FLOATS_PER_PRIM = 12;

export interface SdfVolumeBounds {
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
}

const BAKE_SDF_WGSL = /* wgsl */ `
struct SdfUniforms {
  volMin: vec4<f32>,       // xyz = min, w = primCount
  volMax: vec4<f32>,       // xyz = max, w = voxelSize
  lightDir: vec4<f32>,     // xyz = normalized lightDir, w = sliceEnabled
  soilCol: vec4<f32>,      // rgb = mean forest floor albedo, w = sliceAxis (0=x,1=z)
  sliceInfo: vec4<f32>,    // x = sliceWorldCoord, y = soilHalfX, z = soilHalfZ, w = ambientHumidity
  biolumParams: vec4<f32>, // x = ambientLightScale, y = emitterCount, z = biolumGain, w = nocturnalFactor
};

struct SdfPrim {
  a: vec4<f32>,   // ax, ay, az, rA
  b: vec4<f32>,   // bx, by, bz, kind
  col: vec4<f32>, // r, g, b, param
};

@group(0) @binding(0) var<uniform> u: SdfUniforms;
@group(0) @binding(1) var<storage, read> prims: array<SdfPrim>;
@group(0) @binding(2) var sdfOut: texture_storage_3d<rgba16float, write>;

fn sdCapsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

fn sdCapDome(p: vec3<f32>, center: vec3<f32>, axis: vec3<f32>, radius: f32, thickness: f32) -> f32 {
  let rel = p - center;
  let y = dot(rel, axis);
  let radialVec = rel - axis * y;
  let r = length(radialVec);
  let domeH = max(thickness * 2.8, radius * 0.38);
  let rClamped = min(r, radius);
  let rn = rClamped / max(radius, 1e-4);
  let ySurf = domeH * (1.0 - rn * rn) - domeH * 0.35;
  let dr = max(r - radius, 0.0);
  let dy = y - ySurf;
  return (sqrt(dr * dr + dy * dy) - thickness) * 0.78;
}

fn sdCapFunnel(p: vec3<f32>, center: vec3<f32>, axis: vec3<f32>, radius: f32, thickness: f32) -> f32 {
  let rel = p - center;
  let y = dot(rel, axis);
  let r = length(rel - axis * y);
  let rClamped = min(r, radius);
  let rn = rClamped / max(radius, 1e-4);
  let funnelH = radius * 0.36;
  let ySurf = funnelH * (rn * rn - 0.32);
  let dr = max(r - radius, 0.0);
  let dy = y - ySurf;
  return (sqrt(dr * dr + dy * dy) - thickness) * 0.78;
}

fn sdCapMorel(p: vec3<f32>, center: vec3<f32>, axis: vec3<f32>, radius: f32, height: f32) -> f32 {
  let rel = p - center;
  let y = dot(rel, axis);
  let r = length(rel - axis * y);
  let rx = max(radius, 0.004);
  let ry = max(height, rx * 1.45);
  let q = vec2<f32>(r / rx, y / ry);
  let k0 = length(q);
  let k1 = length(q / vec2<f32>(rx, ry));
  return select(-rx, k0 * (k0 - 1.0) / max(k1, 1e-6), k0 > 1e-5);
}

@compute @workgroup_size(4, 4, 4)
fn cs_bake_sdf(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(sdfOut);
  if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
    return;
  }
  let uvw = (vec3<f32>(gid) + 0.5) / vec3<f32>(dims);
  let p = mix(u.volMin.xyz, u.volMax.xyz, uvw);

  // Exact Euclidean box-half-space for soil terrarium (y <= 0 inside [-soilHalfX..soilHalfX] x [-soilHalfZ..soilHalfZ])
  let qXZ = abs(p.xz) - vec2<f32>(u.sliceInfo.y, u.sliceInfo.z);
  let q3 = vec3<f32>(qXZ.x, p.y, qXZ.y);
  var dBest = length(max(q3, vec3<f32>(0.0))) + min(max(q3.x, max(q3.y, q3.z)), 0.0);
  var colBest = u.soilCol.rgb;

  // Respect razor slice plane so sliced interiors receive unoccluded light & GI
  if (u.lightDir.w > 0.5) {
    let sc = u.sliceInfo.x;
    let slicePos = select(p.x, p.z, u.soilCol.w > 0.5);
    dBest = max(dBest, slicePos - sc);
  }

  let count = u32(u.volMin.w + 0.5);
  for (var i = 0u; i < count; i = i + 1u) {
    let pr = prims[i];
    if (u.lightDir.w > 0.5) {
      let sc = u.sliceInfo.x;
      let pc = select(pr.a.x, pr.a.z, u.soilCol.w > 0.5);
      if (pc > sc + pr.a.w * 0.5) {
        continue;
      }
    }
    let kind = i32(pr.b.w + 0.5);
    var d = 10.0;
    if (kind == 0) {
      d = sdCapsule(p, pr.a.xyz, pr.b.xyz, pr.a.w);
    } else if (kind == 1) {
      d = sdCapDome(p, pr.a.xyz, normalize(pr.b.xyz), pr.a.w, pr.col.w);
    } else if (kind == 2) {
      d = sdCapFunnel(p, pr.a.xyz, normalize(pr.b.xyz), pr.a.w, pr.col.w);
    } else if (kind == 3) {
      d = sdCapMorel(p, pr.a.xyz, normalize(pr.b.xyz), pr.a.w, pr.col.w);
    } else {
      d = length(p - pr.a.xyz) - pr.a.w;
    }

    if (d < dBest) {
      dBest = d;
      colBest = pr.col.rgb;
    }
  }

  textureStore(sdfOut, vec3<i32>(gid), vec4<f32>(dBest, colBest.r, colBest.g, colBest.b));
}
`;

const INJECT_BOUNCE1_WGSL = /* wgsl */ `
struct SdfUniforms {
  volMin: vec4<f32>,
  volMax: vec4<f32>,
  lightDir: vec4<f32>,
  soilCol: vec4<f32>,
  sliceInfo: vec4<f32>,
  biolumParams: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SdfUniforms;
@group(0) @binding(1) var sdfTex: texture_3d<f32>;
@group(0) @binding(2) var linearSamp: sampler;
@group(0) @binding(3) var radOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> biolumLights: array<vec4<f32>>;

fn worldToUvw(p: vec3<f32>) -> vec3<f32> {
  return (p - u.volMin.xyz) / max(u.volMax.xyz - u.volMin.xyz, vec3<f32>(1e-4));
}

fn sampleSdf(p: vec3<f32>) -> vec4<f32> {
  let uvw = worldToUvw(p);
  if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
    return vec4<f32>(0.25, u.soilCol.rgb);
  }
  return textureSampleLevel(sdfTex, linearSamp, uvw, 0.0);
}

fn sdfNormal(p: vec3<f32>, eps: f32) -> vec3<f32> {
  let dx = sampleSdf(p + vec3<f32>(eps, 0.0, 0.0)).x - sampleSdf(p - vec3<f32>(eps, 0.0, 0.0)).x;
  let dy = sampleSdf(p + vec3<f32>(0.0, eps, 0.0)).x - sampleSdf(p - vec3<f32>(0.0, eps, 0.0)).x;
  let dz = sampleSdf(p + vec3<f32>(0.0, 0.0, eps)).x - sampleSdf(p - vec3<f32>(0.0, 0.0, eps)).x;
  let g = vec3<f32>(dx, dy, dz);
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-6);
}

fn sdfSoftShadow(ro: vec3<f32>, rd: vec3<f32>, voxelSize: f32) -> f32 {
  var res = 1.0;
  var t = voxelSize * 1.3;
  for (var i = 0; i < 10; i = i + 1) {
    let h = sampleSdf(ro + rd * t).x;
    if (h < voxelSize * 0.14) {
      return 0.06;
    }
    res = min(res, clamp(8.0 * h / t, 0.06, 1.0));
    t += clamp(h, voxelSize * 0.75, 0.045);
    if (t > 0.42) {
      break;
    }
  }
  return res;
}

@compute @workgroup_size(4, 4, 4)
fn cs_inject_bounce1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(radOut);
  if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
    return;
  }
  let uvw = (vec3<f32>(gid) + 0.5) / vec3<f32>(dims);
  let p = mix(u.volMin.xyz, u.volMax.xyz, uvw);
  let voxelSize = u.volMax.w;

  let s = textureLoad(sdfTex, vec3<i32>(gid), 0);
  let d = s.x;
  let albedo = s.yzw;

  if (d < -voxelSize * 3.0) {
    textureStore(radOut, vec3<i32>(gid), vec4<f32>(0.0));
    return;
  }

  let emitterCount = min(u32(max(u.biolumParams.y + 0.5, 0.0)), 32u);
  let biolumGain = max(u.biolumParams.z, 0.0);
  let biolumColor = vec3<f32>(0.055, 0.96, 0.22);
  var biolumRad = vec3<f32>(0.0);
  if (emitterCount > 0u && biolumGain > 0.001) {
    for (var i = 0u; i < emitterCount; i = i + 1u) {
      let em = biolumLights[i];
      let flux = em.w;
      if (flux <= 0.0001) {
        continue;
      }
      let delta = em.xyz - p;
      let dist2 = dot(delta, delta);
      var vis = 1.0;
      if (dist2 > 0.0020) {
        let midD = sampleSdf(mix(p, em.xyz, 0.45)).x;
        vis = mix(0.18, 1.0, smoothstep(-voxelSize * 1.8, voxelSize * 0.6, midD));
      }
      let atten = (flux * 0.0022) / (dist2 + 0.0012);
      biolumRad += biolumColor * (atten * vis * biolumGain);
    }
  }

  let ambientScale = u.biolumParams.x;
  if (d > voxelSize * 3.0) {
    let sky = mix(vec3<f32>(0.04, 0.05, 0.065), vec3<f32>(0.11, 0.14, 0.19), clamp(uvw.y, 0.0, 1.0));
    let radAir = sky * (0.18 * ambientScale) + biolumRad * 0.65;
    textureStore(radOut, vec3<i32>(gid), vec4<f32>(radAir, 0.0));
    return;
  }

  let n = sdfNormal(p, voxelSize);
  let l = normalize(u.lightDir.xyz);
  let ndl = max(dot(n, l), 0.0);
  let sh = select(0.08, sdfSoftShadow(p + n * voxelSize * 0.65, l, voxelSize), ndl > 0.01);
  let sunCol = vec3<f32>(1.28, 1.20, 1.06);
  let skyCol = vec3<f32>(0.22, 0.27, 0.34) * (0.45 + 0.55 * max(n.y, 0.0));
  let directRad = albedo * (sunCol * ndl * sh + skyCol) * ambientScale
    + biolumRad * (vec3<f32>(0.65) + albedo * 0.55);
  let density = smoothstep(voxelSize * 3.0, 0.0, abs(d));
  textureStore(radOut, vec3<i32>(gid), vec4<f32>(directRad, density));
}
`;

const PROPAGATE_BOUNCE2_WGSL = /* wgsl */ `
struct SdfUniforms {
  volMin: vec4<f32>,
  volMax: vec4<f32>,
  lightDir: vec4<f32>,
  soilCol: vec4<f32>,
  sliceInfo: vec4<f32>,
  biolumParams: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SdfUniforms;
@group(0) @binding(1) var sdfTex: texture_3d<f32>;
@group(0) @binding(2) var linearSamp: sampler;
@group(0) @binding(3) var radIn: texture_3d<f32>;
@group(0) @binding(4) var radOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn cs_propagate_bounce2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(radOut);
  if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
    return;
  }
  let invDims = 1.0 / vec3<f32>(dims);
  let uvw = (vec3<f32>(gid) + 0.5) * invDims;
  let center = textureSampleLevel(radIn, linearSamp, uvw, 0.0);
  let s = textureLoad(sdfTex, vec3<i32>(gid), 0);

  var acc = center.rgb * 0.36;
  var wSum = 0.36;

  let dirs = array<vec3<f32>, 14>(
    vec3<f32>( 1.0,  0.0,  0.0), vec3<f32>(-1.0,  0.0,  0.0),
    vec3<f32>( 0.0,  1.0,  0.0), vec3<f32>( 0.0, -1.0,  0.0),
    vec3<f32>( 0.0,  0.0,  1.0), vec3<f32>( 0.0,  0.0, -1.0),
    vec3<f32>( 0.7,  0.7,  0.7), vec3<f32>(-0.7,  0.7,  0.7),
    vec3<f32>( 0.7, -0.7,  0.7), vec3<f32>(-0.7, -0.7,  0.7),
    vec3<f32>( 0.7,  0.7, -0.7), vec3<f32>(-0.7,  0.7, -0.7),
    vec3<f32>( 0.7, -0.7, -0.7), vec3<f32>(-0.7, -0.7, -0.7)
  );

  for (var i = 0; i < 14; i = i + 1) {
    let dir = dirs[i];
    let uv1 = clamp(uvw + dir * invDims * 1.8, vec3<f32>(0.01), vec3<f32>(0.99));
    let uv2 = clamp(uvw + dir * invDims * 4.2, vec3<f32>(0.01), vec3<f32>(0.99));
    let tap1 = textureSampleLevel(radIn, linearSamp, uv1, 0.0);
    let tap2 = textureSampleLevel(radIn, linearSamp, uv2, 0.0);
    let w1 = 0.09 * (0.4 + tap1.a * 1.6);
    let w2 = 0.05 * (0.3 + tap2.a * 1.4);
    acc += tap1.rgb * w1 + tap2.rgb * w2;
    wSum += w1 + w2;
  }

  let propagated = acc / max(wSum, 1e-4);
  let bounced = mix(
    propagated,
    propagated * (vec3<f32>(0.55) + s.yzw * 0.95) + center.rgb * 0.45,
    center.a * 0.55,
  );
  textureStore(radOut, vec3<i32>(gid), vec4<f32>(bounced, center.a));
}
`;

export class FastSdfGiSystem {
  private device: GPUDevice;
  readonly sdfTex: GPUTexture;
  readonly sdfView: GPUTextureView;
  private radTexA: GPUTexture;
  private radViewA: GPUTextureView;
  readonly radTexB: GPUTexture;
  readonly radViewB: GPUTextureView;

  private uniformBuf: GPUBuffer;
  private primBuf: GPUBuffer;
  private biolumLightsBuf: GPUBuffer;
  private linearSampler: GPUSampler;

  private bakePipe: GPUComputePipeline;
  private bounce1Pipe: GPUComputePipeline;
  private bounce2Pipe: GPUComputePipeline;

  private bakeBindGroup: GPUBindGroup;
  private bounce1BindGroup: GPUBindGroup;
  private bounce2BindGroup: GPUBindGroup;

  private uniformScratch = new Float32Array(24);
  private primScratch = new Float32Array(MAX_PRIMS * FLOATS_PER_PRIM);
  private biolumLightsScratch = new Float32Array(MAX_BIOLUM_LIGHTS * 4);
  readonly bounds: SdfVolumeBounds = {
    minX: -0.26,
    minY: -0.05,
    minZ: -0.26,
    maxX: 0.26,
    maxY: 0.27,
    maxZ: 0.26,
  };

  constructor(device: GPUDevice) {
    this.device = device;

    const texUsage =
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING;

    this.sdfTex = device.createTexture({
      size: [SDF_NX, SDF_NY, SDF_NZ],
      dimension: '3d',
      format: 'rgba16float',
      usage: texUsage,
    });
    this.sdfView = this.sdfTex.createView({ dimension: '3d' });

    this.radTexA = device.createTexture({
      size: [SDF_NX, SDF_NY, SDF_NZ],
      dimension: '3d',
      format: 'rgba16float',
      usage: texUsage,
    });
    this.radViewA = this.radTexA.createView({ dimension: '3d' });

    this.radTexB = device.createTexture({
      size: [SDF_NX, SDF_NY, SDF_NZ],
      dimension: '3d',
      format: 'rgba16float',
      usage: texUsage,
    });
    this.radViewB = this.radTexB.createView({ dimension: '3d' });

    this.uniformBuf = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.primBuf = device.createBuffer({
      size: MAX_PRIMS * FLOATS_PER_PRIM * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.biolumLightsBuf = device.createBuffer({
      size: MAX_BIOLUM_LIGHTS * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.linearSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    const bakeMod = device.createShaderModule({ code: BAKE_SDF_WGSL });
    const bakeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });
    this.bakePipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bakeLayout] }),
      compute: { module: bakeMod, entryPoint: 'cs_bake_sdf' },
    });
    this.bakeBindGroup = device.createBindGroup({
      layout: bakeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.primBuf } },
        { binding: 2, resource: this.sdfView },
      ],
    });

    const b1Mod = device.createShaderModule({ code: INJECT_BOUNCE1_WGSL });
    const b1Layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.bounce1Pipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [b1Layout] }),
      compute: { module: b1Mod, entryPoint: 'cs_inject_bounce1' },
    });
    this.bounce1BindGroup = device.createBindGroup({
      layout: b1Layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.sdfView },
        { binding: 2, resource: this.linearSampler },
        { binding: 3, resource: this.radViewA },
        { binding: 4, resource: { buffer: this.biolumLightsBuf } },
      ],
    });

    const b2Mod = device.createShaderModule({ code: PROPAGATE_BOUNCE2_WGSL });
    const b2Layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });
    this.bounce2Pipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [b2Layout] }),
      compute: { module: b2Mod, entryPoint: 'cs_propagate_bounce2' },
    });
    this.bounce2BindGroup = device.createBindGroup({
      layout: b2Layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.sdfView },
        { binding: 2, resource: this.linearSampler },
        { binding: 3, resource: this.radViewA },
        { binding: 4, resource: this.radViewB },
      ],
    });
  }

  /** Pack live fruiting bodies + solid obstacles and dispatch the 3 GPU compute passes */
  update(
    encoder: GPUCommandEncoder,
    snapshot: SimulationSnapshot,
    view: ViewState,
    lightDir: [number, number, number],
    sliceWorldCoord: number,
    lightingMoodIdx = 0,
  ): void {
    const soilHalfX = snapshot.soil.sizeX * 0.5;
    const soilHalfZ = snapshot.soil.sizeZ * 0.5;
    this.bounds.minX = -(soilHalfX + 0.02);
    this.bounds.maxX = soilHalfX + 0.02;
    this.bounds.minY = -0.05;
    this.bounds.maxY = 0.27;
    this.bounds.minZ = -(soilHalfZ + 0.02);
    this.bounds.maxZ = soilHalfZ + 0.02;

    const p = this.primScratch;
    let count = 0;
    const pushPrim = (
      ax: number, ay: number, az: number, rA: number,
      bx: number, by: number, bz: number, kind: number,
      cr: number, cg: number, cb: number, param: number,
    ): void => {
      if (count >= MAX_PRIMS) return;
      const o = count * FLOATS_PER_PRIM;
      p[o] = ax; p[o + 1] = ay; p[o + 2] = az; p[o + 3] = rA;
      p[o + 4] = bx; p[o + 5] = by; p[o + 6] = bz; p[o + 7] = kind;
      p[o + 8] = cr; p[o + 9] = cg; p[o + 10] = cb; p[o + 11] = param;
      count++;
    };

    for (const body of snapshot.bodies) {
      const sp = SPECIES_PRESETS[body.speciesId];
      const cl = body.stipeCenterline;
      const nPts = Math.floor(cl.length / 3);
      const stride = Math.max(1, Math.floor((nPts - 1) / 4));
      for (let k = 0; k < nPts - 1; k += stride) {
        const kNext = Math.min(nPts - 1, k + stride);
        const i0 = k * 3;
        const i1 = kNext * 3;
        const taper = 1.0 - 0.25 * (k / Math.max(1, nPts - 1));
        pushPrim(
          cl[i0], cl[i0 + 1], cl[i0 + 2], Math.max(0.0025, body.stipeRadiusM * taper),
          cl[i1], cl[i1 + 1], cl[i1 + 2], 0,
          sp.stipeColor[0], sp.stipeColor[1], sp.stipeColor[2], 0,
        );
      }

      let capKind = 1;
      let capParam = Math.max(0.0035, sp.gillDepthM * 0.75);
      if (sp.hymeniumType === 'false-folds') {
        capKind = 2;
        capParam = 0.004;
      } else if (sp.hymeniumType === 'honeycomb') {
        capKind = 3;
        capParam = Math.max(0.012, body.capRadiusM * 1.55);
      }
      const capR = Math.max(0.005, body.capRadiusM);
      const ax = body.stipeTipDirX;
      const ay = body.stipeTipDirY;
      const az = body.stipeTipDirZ;
      const al = Math.hypot(ax, ay, az) || 1;
      const capYShift = capKind === 3 ? capParam * 0.45 : capR * 0.12;
      pushPrim(
        body.stipeTipX + (ax / al) * capYShift,
        body.stipeTipY + (ay / al) * capYShift,
        body.stipeTipZ + (az / al) * capYShift,
        capR,
        ax / al, ay / al, az / al, capKind,
        sp.capColor[0], sp.capColor[1], sp.capColor[2], capParam,
      );
    }

    for (const obs of snapshot.obstacles) {
      if (obs.kind === 'twig') {
        pushPrim(
          obs.x, obs.y, obs.z, obs.radius,
          obs.endX, obs.endY, obs.endZ, 0,
          0.26, 0.17, 0.10, 0,
        );
      } else {
        pushPrim(
          obs.x, obs.y, obs.z, obs.radius,
          0, 1, 0, 4,
          0.34, 0.36, 0.38, 0,
        );
      }
    }

    const emitters = snapshot.biolum?.enabled ? snapshot.biolum.emitters : null;
    const emitterCount = emitters ? Math.min(MAX_BIOLUM_LIGHTS, emitters.length) : 0;
    const nocturnalFactor = lightingMoodIdx === 3 ? 1.0 : 0.0;
    const ambientLightScale =
      lightingMoodIdx === 3 ? 0.06 : lightingMoodIdx === 2 ? 0.38 : 1.0;
    const biolumGain = snapshot.biolum?.enabled ? (snapshot.species.biolumGain ?? 1.0) : 0.0;

    const lArr = this.biolumLightsScratch;
    for (let i = 0; i < emitterCount; i++) {
      const em = emitters![i];
      const o = i * 4;
      lArr[o] = em.x;
      lArr[o + 1] = em.y;
      lArr[o + 2] = em.z;
      lArr[o + 3] = Math.max(0, em.intensity);
    }

    const voxelSize = (this.bounds.maxX - this.bounds.minX) / SDF_NX;
    const u = this.uniformScratch;
    u[0] = this.bounds.minX;
    u[1] = this.bounds.minY;
    u[2] = this.bounds.minZ;
    u[3] = count;
    u[4] = this.bounds.maxX;
    u[5] = this.bounds.maxY;
    u[6] = this.bounds.maxZ;
    u[7] = voxelSize;
    u[8] = lightDir[0];
    u[9] = lightDir[1];
    u[10] = lightDir[2];
    u[11] = view.sliceEnabled ? 1.0 : 0.0;
    u[12] = 0.16;
    u[13] = 0.19;
    u[14] = 0.095;
    u[15] = view.sliceAxis === 'x' ? 0.0 : 1.0;
    u[16] = sliceWorldCoord;
    u[17] = soilHalfX;
    u[18] = soilHalfZ;
    u[19] = snapshot.env.ambientHumidity;
    u[20] = ambientLightScale;
    u[21] = emitterCount;
    u[22] = biolumGain;
    u[23] = nocturnalFactor;

    this.device.queue.writeBuffer(this.uniformBuf, 0, u.buffer, u.byteOffset, 96);
    if (count > 0) {
      this.device.queue.writeBuffer(
        this.primBuf,
        0,
        p.buffer,
        p.byteOffset,
        count * FLOATS_PER_PRIM * 4,
      );
    }
    if (emitterCount > 0) {
      this.device.queue.writeBuffer(
        this.biolumLightsBuf,
        0,
        lArr.buffer,
        lArr.byteOffset,
        emitterCount * 16,
      );
    }

    const gx = Math.ceil(SDF_NX / 4);
    const gy = Math.ceil(SDF_NY / 4);
    const gz = Math.ceil(SDF_NZ / 4);

    const passBake = encoder.beginComputePass();
    passBake.setPipeline(this.bakePipe);
    passBake.setBindGroup(0, this.bakeBindGroup);
    passBake.dispatchWorkgroups(gx, gy, gz);
    passBake.end();

    const passB1 = encoder.beginComputePass();
    passB1.setPipeline(this.bounce1Pipe);
    passB1.setBindGroup(0, this.bounce1BindGroup);
    passB1.dispatchWorkgroups(gx, gy, gz);
    passB1.end();

    const passB2 = encoder.beginComputePass();
    passB2.setPipeline(this.bounce2Pipe);
    passB2.setBindGroup(0, this.bounce2BindGroup);
    passB2.dispatchWorkgroups(gx, gy, gz);
    passB2.end();
  }

  dispose(): void {
    this.sdfTex.destroy();
    this.radTexA.destroy();
    this.radTexB.destroy();
    this.uniformBuf.destroy();
    this.primBuf.destroy();
    this.biolumLightsBuf.destroy();
  }
}
