// ============================================================================
// SPOROCARP — Hybrid WebGPU Procedural 3D Gaussian Splatting Engine
// Pipeline (100% GPU-driven sorting & projection, zero CPU sort/GC):
//   1. CPU Zero-GC Procedural Splat Builders (Soil moss/litter, Mycelial halo,
//      Species cap/basal fuzz, Velocity-stretched Dressaire spore mist)
//   2. cs_project_splats: Slice clip, Fast-SDF-AO/GI dynamic lighting,
//      Zwicker/Kerbl EWA 2D covariance projection & 16-bit depth keying
//   3. Exact 2-Pass 8-Bit GPU LSD Radix Sort (6 dispatches, 65,536 splats)
//   4. Premultiplied 3DGS Quad Rasterizer depth-tested against opaque geometry
// ============================================================================
import {
  SoilMaterialId,
  type SimulationSnapshot,
  type ViewState,
  type XRayFieldMode,
} from '../types/contract.ts';
import type { Mat4 } from './math.ts';
import { WGSL_COMMON, makeWgslSdfGiHelpers } from './shaders.ts';

export const MAX_SPLATS = 65536;
const WG_SIZE = 256;
const NUM_WG = MAX_SPLATS / WG_SIZE; // 256
const FLOATS_PER_SPLAT = 16; // 64 bytes

const MAX_SOIL_SPLATS = 24000;
const MAX_MYC_SPLATS = 12000;
const MAX_FUZZ_SPLATS = 8000;
const MAX_SPORE_SPLATS = MAX_SPLATS - MAX_SOIL_SPLATS - MAX_MYC_SPLATS - MAX_FUZZ_SPLATS; // 21536

const XRAY_INDEX: Record<XRayFieldMode, number> = {
  none: 0,
  matric_psi: 1,
  carbon_enzyme: 2,
  turgor_p: 3,
  shell_stress: 4,
  air_convection: 5,
};

function hash1(n: number): number {
  const x = Math.sin(n * 12.9898 + 78.233) * 43758.5453123;
  return x - Math.floor(x);
}

/** Write unit quaternion rotating +X axis onto normalized direction (dx, dy, dz) */
function writeQuatFromDirX(
  out: Float32Array,
  qOff: number,
  dx: number,
  dy: number,
  dz: number,
): void {
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-6) {
    out[qOff] = 0; out[qOff + 1] = 0; out[qOff + 2] = 0; out[qOff + 3] = 1;
    return;
  }
  const nx = dx / len;
  const ny = dy / len;
  const nz = dz / len;
  // Cross product (1, 0, 0) x (nx, ny, nz) = (0, -nz, ny); dot = nx
  if (nx < -0.9995) {
    out[qOff] = 0; out[qOff + 1] = 1; out[qOff + 2] = 0; out[qOff + 3] = 0;
    return;
  }
  const qw = 1 + nx;
  const qx = 0;
  const qy = -nz;
  const qz = ny;
  const inv = 1 / Math.hypot(qx, qy, qz, qw);
  out[qOff] = qx * inv;
  out[qOff + 1] = qy * inv;
  out[qOff + 2] = qz * inv;
  out[qOff + 3] = qw * inv;
}

/** Write unit quaternion for yaw-around-Y + slight tilt */
function writeQuatYawTilt(
  out: Float32Array,
  qOff: number,
  yawRad: number,
  tiltX: number,
  tiltZ: number,
): void {
  const cy = Math.cos(yawRad * 0.5);
  const sy = Math.sin(yawRad * 0.5);
  const qx = tiltX * 0.5;
  const qy = sy;
  const qz = tiltZ * 0.5;
  const qw = cy;
  const inv = 1 / Math.hypot(qx, qy, qz, qw);
  out[qOff] = qx * inv;
  out[qOff + 1] = qy * inv;
  out[qOff + 2] = qz * inv;
  out[qOff + 3] = qw * inv;
}

const PROJECT_SPLATS_WGSL = /* wgsl */ `
${WGSL_COMMON}

struct SplatParams {
  viewTiltMat: mat4x4<f32>,
  projMat: mat4x4<f32>,
  projParams: vec4<f32>, // focalX, focalY, viewportW, viewportH
  flags: vec4<f32>,      // activeSplatCount, xrayMode, showMycelium, showSpores
  depthRange: vec4<f32>, // near, far, 0, 0
};

struct SplatRaw {
  posAlpha: vec4<f32>,  // xyz = pos, w = baseAlpha
  scaleKind: vec4<f32>, // sx, sy, sz, w = kind (0=soil,1=mycelium,2=capFuzz,3=sporeMist)
  quat: vec4<f32>,      // qx, qy, qz, qw
  colorMeta: vec4<f32>, // rgb = linear albedo, w = meta (wetness/turgor/lift)
};

struct SplatProj {
  clipPos: vec4<f32>, // ndc x, y, z, activeFlag
  axes: vec4<f32>,    // axis1.xy, axis2.xy in NDC
  rgba: vec4<f32>,    // sRGB color, alpha
};

@group(0) @binding(0) var<uniform> u: SceneUniforms;
@group(0) @binding(1) var<uniform> sp: SplatParams;
@group(0) @binding(2) var<storage, read> rawSplats: array<SplatRaw>;
${makeWgslSdfGiHelpers(3, 'sdfSamp', 5, 4)}
@group(0) @binding(6) var<storage, read_write> projOut: array<SplatProj>;
@group(0) @binding(7) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(8) var<storage, read_write> valsOut: array<u32>;

fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  let xx = x * x; let yy = y * y; let zz = z * z;
  let xy = x * y; let xz = x * z; let yz = y * z;
  let wx = w * x; let wy = w * y; let wz = w * z;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
    vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
    vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy))
  );
}

fn emitCulled(idx: u32) {
  projOut[idx].clipPos = vec4<f32>(2.0, 2.0, 2.0, 0.0);
  projOut[idx].axes = vec4<f32>(0.0);
  projOut[idx].rgba = vec4<f32>(0.0);
  keysOut[idx] = 65535u;
  valsOut[idx] = idx;
}

@compute @workgroup_size(256)
fn cs_project_splats(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ${MAX_SPLATS}u) {
    return;
  }
  let activeCount = u32(sp.flags.x + 0.5);
  if (idx >= activeCount) {
    emitCulled(idx);
    return;
  }

  let s = rawSplats[idx];
  if (s.posAlpha.w <= 0.005) {
    emitCulled(idx);
    return;
  }

  let pos = s.posAlpha.xyz;
  if (isClippedBySlice(pos, u.sliceParams)) {
    emitCulled(idx);
    return;
  }

  let kind = i32(s.scaleKind.w + 0.5);
  let xrayMode = i32(sp.flags.y + 0.5);
  if ((kind == 0 || kind == 2) && xrayMode != 0) {
    emitCulled(idx);
    return;
  }
  if (kind == 1 && sp.flags.z < 0.5) {
    emitCulled(idx);
    return;
  }
  if (kind == 3 && sp.flags.w < 0.5) {
    emitCulled(idx);
    return;
  }

  // View-space position & frustum check
  let pView4 = sp.viewTiltMat * vec4<f32>(pos, 1.0);
  let pView = pView4.xyz;
  let depth = -pView.z;
  if (depth <= sp.depthRange.x || depth >= sp.depthRange.y) {
    emitCulled(idx);
    return;
  }

  let clip = sp.projMat * pView4;
  if (clip.w <= 1e-5) {
    emitCulled(idx);
    return;
  }
  let ndc = clip.xyz / clip.w;
  if (abs(ndc.x) > 1.25 || abs(ndc.y) > 1.25 || ndc.z < 0.0 || ndc.z > 1.0) {
    emitCulled(idx);
    return;
  }

  // Exact Zwicker/Kerbl 3DGS EWA 2D covariance projection
  let R = quatToMat3(s.quat);
  let sc = s.scaleKind.xyz;
  let M = mat3x3<f32>(R[0] * sc.x, R[1] * sc.y, R[2] * sc.z);
  let sigma3d = M * transpose(M);

  let W = mat3x3<f32>(
    sp.viewTiltMat[0].xyz,
    sp.viewTiltMat[1].xyz,
    sp.viewTiltMat[2].xyz
  );
  let fx = sp.projParams.x;
  let fy = sp.projParams.y;
  let invD = 1.0 / depth;
  let invD2 = invD * invD;
  let J = mat3x3<f32>(
    vec3<f32>(fx * invD, 0.0, 0.0),
    vec3<f32>(0.0, fy * invD, 0.0),
    vec3<f32>(fx * pView.x * invD2, fy * pView.y * invD2, 0.0)
  );
  let T = J * W;
  let cov = T * sigma3d * transpose(T);

  let a = cov[0][0] + 0.35;
  let b = cov[0][1];
  let c = cov[1][1] + 0.35;

  let mid = 0.5 * (a + c);
  let halfDiff = 0.5 * (a - c);
  let disc = sqrt(max(0.0, halfDiff * halfDiff + b * b));
  let l1 = max(0.25, mid + disc);
  let l2 = max(0.25, mid - disc);

  var e1 = vec2<f32>(1.0, 0.0);
  if (disc > 1e-6) {
    if (a >= c) {
      e1 = normalize(vec2<f32>(l1 - c, b));
    } else {
      e1 = normalize(vec2<f32>(b, l1 - a));
    }
  }
  let e2 = vec2<f32>(-e1.y, e1.x);

  let r1 = clamp(3.0 * sqrt(l1), 0.85, 56.0);
  let r2 = clamp(3.0 * sqrt(l2), 0.85, 56.0);
  let ndcScale = vec2<f32>(2.0 / max(sp.projParams.z, 1.0), 2.0 / max(sp.projParams.w, 1.0));
  let axis1 = e1 * r1 * ndcScale;
  let axis2 = e2 * r2 * ndcScale;

  // Dynamic Fast-SDF-AO + GI Lighting per splat
  let l = normalize(u.lightDir.xyz);
  let v = normalize(u.cameraPos.xyz - pos);
  var outCol = s.colorMeta.rgb;
  var outAlpha = s.posAlpha.w;

  if (kind == 0 || kind == 2) {
    // Organic soil/moss/litter/cap-fuzz: Fast-SDF-AO + soft shadow + 2-bounce GI
    let nSurf = normalize(select(vec3<f32>(0.0, 1.0, 0.0), R[1], kind == 0) + select(vec3<f32>(0.0), R[0], kind == 2));
    let nProbe = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), nSurf, 0.55));
    let probePos = pos + nProbe * 0.005;
    let aoBent = evalFastSdfAO(probePos, nProbe);
    let shadow = evalSdfSoftShadow(probePos, l);
    let gi = evalSdfGI(probePos, nProbe, aoBent.xyz);
    let ndl = clamp((dot(nProbe, l) + 0.35) / 1.35, 0.0, 1.0);
    let sky = mix(vec3<f32>(0.08, 0.09, 0.10), vec3<f32>(0.24, 0.29, 0.36), aoBent.y * 0.5 + 0.5);
    let sun = vec3<f32>(1.08, 1.02, 0.92) * ndl * mix(0.22, 1.0, shadow);
    let lit = s.colorMeta.rgb * ((sky + sun) * aoBent.w + gi * 0.68);
    outCol = srgbEncode(acesFilmic(lit));
  } else if (kind == 1) {
    // Cottony mycelial halo
    if (xrayMode == 3) {
      outCol = srgbEncode(colormapViridis(s.colorMeta.w));
    } else {
      outCol = srgbEncode(s.colorMeta.rgb);
    }
  } else {
    // Volumetric velocity-stretched Dressaire spore mist with Tyndall forward scatter
    let forwardScatter = pow(clamp(dot(v, -l), 0.0, 1.0), 2.2) * 0.35 + 0.65;
    let lifted = clamp(s.colorMeta.w / 0.03, 0.0, 1.0);
    let baseMist = mix(max(u.sporePrintColor.rgb * 1.45, vec3<f32>(0.58, 0.54, 0.48)), vec3<f32>(0.42, 0.86, 0.98), lifted * 0.78);
    let tint = mix(baseMist, vec3<f32>(1.0, 0.96, 0.88), 0.25 * forwardScatter);
    outCol = tint * (0.78 + 0.22 * forwardScatter);
  }

  // Quantize back-to-front 16-bit depth key in [0, 65534] so ascending sort orders far->near
  let depthNorm = clamp((depth - sp.depthRange.x) / max(sp.depthRange.y - sp.depthRange.x, 1e-4), 0.0, 1.0);
  let key = 65534u - u32(depthNorm * 65534.0);

  projOut[idx].clipPos = vec4<f32>(ndc, 1.0);
  projOut[idx].axes = vec4<f32>(axis1, axis2);
  projOut[idx].rgba = vec4<f32>(outCol, clamp(outAlpha, 0.0, 0.98));
  keysOut[idx] = key;
  valsOut[idx] = idx;
}
`;

const RADIX_SORT_WGSL = /* wgsl */ `
struct RadixUniforms {
  shift: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> uRadix: RadixUniforms;
@group(0) @binding(1) var<storage, read> keysIn: array<u32>;
@group(0) @binding(2) var<storage, read> valsIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(4) var<storage, read_write> valsOut: array<u32>;
@group(0) @binding(5) var<storage, read_write> wgHist: array<u32>;
@group(0) @binding(6) var<storage, read_write> wgOffsets: array<u32>;
@group(0) @binding(7) var<storage, read_write> globalBinOffsets: array<u32>;

var<workgroup> shHist: array<atomic<u32>, 256>;
var<workgroup> shScanA: array<u32, 256>;
var<workgroup> shScanB: array<u32, 256>;
var<workgroup> shDigits: array<u32, 256>;

@compute @workgroup_size(256)
fn cs_radix_hist(
  @builtin(local_invocation_id) lidVec: vec3<u32>,
  @builtin(workgroup_id) wgVec: vec3<u32>,
  @builtin(global_invocation_id) gidVec: vec3<u32>,
) {
  let lid = lidVec.x;
  let wgId = wgVec.x;
  let gid = gidVec.x;

  atomicStore(&shHist[lid], 0u);
  workgroupBarrier();

  let d = (keysIn[gid] >> uRadix.shift) & 255u;
  atomicAdd(&shHist[d], 1u);
  workgroupBarrier();

  wgHist[lid * ${NUM_WG}u + wgId] = atomicLoad(&shHist[lid]);
}

@compute @workgroup_size(256)
fn cs_radix_prefix(@builtin(local_invocation_id) lidVec: vec3<u32>) {
  let lid = lidVec.x;
  var sum = 0u;
  for (var w = 0u; w < ${NUM_WG}u; w = w + 1u) {
    let idx = lid * ${NUM_WG}u + w;
    let c = wgHist[idx];
    wgOffsets[idx] = sum;
    sum += c;
  }
  shScanA[lid] = sum;
  workgroupBarrier();

  for (var step = 0u; step < 8u; step = step + 1u) {
    let offset = 1u << step;
    if ((step & 1u) == 0u) {
      if (lid >= offset) {
        shScanB[lid] = shScanA[lid] + shScanA[lid - offset];
      } else {
        shScanB[lid] = shScanA[lid];
      }
    } else {
      if (lid >= offset) {
        shScanA[lid] = shScanB[lid] + shScanB[lid - offset];
      } else {
        shScanA[lid] = shScanB[lid];
      }
    }
    workgroupBarrier();
  }

  globalBinOffsets[lid] = shScanA[lid] - sum;
}

@compute @workgroup_size(256)
fn cs_radix_scatter(
  @builtin(local_invocation_id) lidVec: vec3<u32>,
  @builtin(workgroup_id) wgVec: vec3<u32>,
  @builtin(global_invocation_id) gidVec: vec3<u32>,
) {
  let lid = lidVec.x;
  let wgId = wgVec.x;
  let gid = gidVec.x;

  let key = keysIn[gid];
  let val = valsIn[gid];
  let d = (key >> uRadix.shift) & 255u;
  shDigits[lid] = d;
  workgroupBarrier();

  var rank = 0u;
  for (var j = 0u; j < lid; j = j + 1u) {
    if (shDigits[j] == d) {
      rank += 1u;
    }
  }
  let dst = globalBinOffsets[d] + wgOffsets[d * ${NUM_WG}u + wgId] + rank;
  keysOut[dst] = key;
  valsOut[dst] = val;
}
`;

const DRAW_SPLATS_WGSL = /* wgsl */ `
struct SplatProj {
  clipPos: vec4<f32>,
  axes: vec4<f32>,
  rgba: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> projSplats: array<SplatProj>;
@group(0) @binding(1) var<storage, read> sortedVals: array<u32>;

struct SplatVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) rgba: vec4<f32>,
};

@vertex
fn vs_main(
  @location(0) corner: vec2<f32>,
  @builtin(instance_index) inst: u32,
) -> SplatVertOut {
  var out: SplatVertOut;
  let splatIdx = sortedVals[inst];
  let sp = projSplats[splatIdx];
  if (sp.clipPos.w < 0.5 || sp.rgba.a < 0.008) {
    out.clipPos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.rgba = vec4<f32>(0.0);
    return out;
  }
  let ndcXY = sp.clipPos.xy + sp.axes.xy * corner.x + sp.axes.zw * corner.y;
  out.clipPos = vec4<f32>(ndcXY, sp.clipPos.z, 1.0);
  out.uv = corner * 3.0;
  out.rgba = sp.rgba;
  return out;
}

@fragment
fn fs_main(in: SplatVertOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 9.0) {
    discard;
  }
  let g = exp(-0.5 * r2);
  let a = clamp(in.rgba.a * g, 0.0, 0.98);
  if (a < 0.004) {
    discard;
  }
  return vec4<f32>(in.rgba.rgb * a, a);
}
`;

export class ProceduralSplatPass {
  private device: GPUDevice;
  private rawScratch = new Float32Array(MAX_SPLATS * FLOATS_PER_SPLAT);
  private paramsScratch = new Float32Array(48);

  private soilSplatCount = 0;
  private lastSoilSig = '';

  private activeSplatCount = 0;

  private rawBuf: GPUBuffer;
  private projBuf: GPUBuffer;
  private paramsBuf: GPUBuffer;
  private keysABuf: GPUBuffer;
  private valsABuf: GPUBuffer;
  private keysBBuf: GPUBuffer;
  private valsBBuf: GPUBuffer;
  private wgHistBuf: GPUBuffer;
  private wgOffsetsBuf: GPUBuffer;
  private globalBinOffsetsBuf: GPUBuffer;
  private shift0Buf: GPUBuffer;
  private shift8Buf: GPUBuffer;

  private projectPipe: GPUComputePipeline;
  private histPipe: GPUComputePipeline;
  private prefixPipe: GPUComputePipeline;
  private scatterPipe: GPUComputePipeline;
  private drawPipe: GPURenderPipeline;

  private projectBindGroup: GPUBindGroup;
  private sortPass0BindGroup: GPUBindGroup;
  private sortPass1BindGroup: GPUBindGroup;
  private drawBindGroup: GPUBindGroup;
  private quadBuf: GPUBuffer;

  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    sceneUniformBuf: GPUBuffer,
    sdfView: GPUTextureView,
    linearSampler: GPUSampler,
    radViewB: GPUTextureView,
    quadBuf: GPUBuffer,
  ) {
    this.device = device;
    this.quadBuf = quadBuf;

    this.rawBuf = device.createBuffer({
      size: MAX_SPLATS * FLOATS_PER_SPLAT * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.projBuf = device.createBuffer({
      size: MAX_SPLATS * 48,
      usage: GPUBufferUsage.STORAGE,
    });
    this.paramsBuf = device.createBuffer({
      size: 192,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const sortBufSize = MAX_SPLATS * 4;
    this.keysABuf = device.createBuffer({ size: sortBufSize, usage: GPUBufferUsage.STORAGE });
    this.valsABuf = device.createBuffer({ size: sortBufSize, usage: GPUBufferUsage.STORAGE });
    this.keysBBuf = device.createBuffer({ size: sortBufSize, usage: GPUBufferUsage.STORAGE });
    this.valsBBuf = device.createBuffer({ size: sortBufSize, usage: GPUBufferUsage.STORAGE });
    this.wgHistBuf = device.createBuffer({ size: NUM_WG * 256 * 4, usage: GPUBufferUsage.STORAGE });
    this.wgOffsetsBuf = device.createBuffer({ size: NUM_WG * 256 * 4, usage: GPUBufferUsage.STORAGE });
    this.globalBinOffsetsBuf = device.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.STORAGE });

    this.shift0Buf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.shift8Buf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.shift0Buf, 0, new Uint32Array([0, 0, 0, 0]));
    device.queue.writeBuffer(this.shift8Buf, 0, new Uint32Array([8, 0, 0, 0]));

    // 1. Project & light compute pipeline
    const projMod = device.createShaderModule({ code: PROJECT_SPLATS_WGSL });
    const projLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.projectPipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [projLayout] }),
      compute: { module: projMod, entryPoint: 'cs_project_splats' },
    });
    this.projectBindGroup = device.createBindGroup({
      layout: projLayout,
      entries: [
        { binding: 0, resource: { buffer: sceneUniformBuf } },
        { binding: 1, resource: { buffer: this.paramsBuf } },
        { binding: 2, resource: { buffer: this.rawBuf } },
        { binding: 3, resource: sdfView },
        { binding: 4, resource: linearSampler },
        { binding: 5, resource: radViewB },
        { binding: 6, resource: { buffer: this.projBuf } },
        { binding: 7, resource: { buffer: this.keysABuf } },
        { binding: 8, resource: { buffer: this.valsABuf } },
      ],
    });

    // 2. 2-Pass GPU LSD Radix Sort pipelines & ping-pong bind groups
    const sortMod = device.createShaderModule({ code: RADIX_SORT_WGSL });
    const sortLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const sortPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [sortLayout] });
    this.histPipe = device.createComputePipeline({
      layout: sortPipeLayout,
      compute: { module: sortMod, entryPoint: 'cs_radix_hist' },
    });
    this.prefixPipe = device.createComputePipeline({
      layout: sortPipeLayout,
      compute: { module: sortMod, entryPoint: 'cs_radix_prefix' },
    });
    this.scatterPipe = device.createComputePipeline({
      layout: sortPipeLayout,
      compute: { module: sortMod, entryPoint: 'cs_radix_scatter' },
    });

    this.sortPass0BindGroup = device.createBindGroup({
      layout: sortLayout,
      entries: [
        { binding: 0, resource: { buffer: this.shift0Buf } },
        { binding: 1, resource: { buffer: this.keysABuf } },
        { binding: 2, resource: { buffer: this.valsABuf } },
        { binding: 3, resource: { buffer: this.keysBBuf } },
        { binding: 4, resource: { buffer: this.valsBBuf } },
        { binding: 5, resource: { buffer: this.wgHistBuf } },
        { binding: 6, resource: { buffer: this.wgOffsetsBuf } },
        { binding: 7, resource: { buffer: this.globalBinOffsetsBuf } },
      ],
    });
    this.sortPass1BindGroup = device.createBindGroup({
      layout: sortLayout,
      entries: [
        { binding: 0, resource: { buffer: this.shift8Buf } },
        { binding: 1, resource: { buffer: this.keysBBuf } },
        { binding: 2, resource: { buffer: this.valsBBuf } },
        { binding: 3, resource: { buffer: this.keysABuf } },
        { binding: 4, resource: { buffer: this.valsABuf } },
        { binding: 5, resource: { buffer: this.wgHistBuf } },
        { binding: 6, resource: { buffer: this.wgOffsetsBuf } },
        { binding: 7, resource: { buffer: this.globalBinOffsetsBuf } },
      ],
    });

    // 3. Premultiplied 3DGS Render pipeline
    const drawMod = device.createShaderModule({ code: DRAW_SPLATS_WGSL });
    const drawLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.drawBindGroup = device.createBindGroup({
      layout: drawLayout,
      entries: [
        { binding: 0, resource: { buffer: this.projBuf } },
        { binding: 1, resource: { buffer: this.valsABuf } },
      ],
    });
    this.drawPipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [drawLayout] }),
      vertex: {
        module: drawMod,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 8,
          stepMode: 'vertex',
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
        }],
      },
      fragment: {
        module: drawMod,
        entryPoint: 'fs_main',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
      multisample: { count: 4 },
    });
  }

  private pushSplat(
    index: number,
    px: number, py: number, pz: number, alpha: number,
    sx: number, sy: number, sz: number, kind: number,
    cr: number, cg: number, cb: number, meta: number,
  ): number {
    const o = index * FLOATS_PER_SPLAT;
    const d = this.rawScratch;
    d[o] = px; d[o + 1] = py; d[o + 2] = pz; d[o + 3] = alpha;
    d[o + 4] = sx; d[o + 5] = sy; d[o + 6] = sz; d[o + 7] = kind;
    // Caller sets quaternion d[o+8..o+11] directly or defaults to identity
    d[o + 12] = cr; d[o + 13] = cg; d[o + 14] = cb; d[o + 15] = meta;
    return o + 8;
  }

  private buildSoilSplatsIfNeeded(snapshot: SimulationSnapshot): void {
    const soil = snapshot.soil;
    // Fast spatial hash of soil material & top moisture to rebuild only when soil changes meaningfully
    let sigNum = soil.nx * 73856093 ^ soil.ny * 19349663 ^ soil.nz * 83492791;
    const total = soil.material.length;
    const stride = Math.max(1, Math.floor(total / 256));
    for (let i = 0; i < total; i += stride) {
      sigNum = (sigNum * 31 + soil.material[i] * 17 + Math.round(soil.theta[i] * 25)) | 0;
    }
    const sig = `${sigNum}:${snapshot.species.id}`;
    if (sig === this.lastSoilSig && this.soilSplatCount > 0) {
      return;
    }
    this.lastSoilSig = sig;

    let count = 0;
    const { nx, ny, nz, dx } = soil;
    const halfX = soil.sizeX * 0.5;
    const halfZ = soil.sizeZ * 0.5;
    const d = this.rawScratch;

    // 1. Forest floor 3D moss tufts, humus mulch crumbs, leaf-litter flakes & sand aggregates
    const iyTop = ny - 1;
    for (let iz = 0; iz < nz && count < MAX_SOIL_SPLATS - 4; iz++) {
      for (let ix = 0; ix < nx && count < MAX_SOIL_SPLATS - 4; ix++) {
        const idx = soil.voxelIndex(ix, iyTop, iz);
        const mat = soil.material[idx];
        const wet = Math.max(0, Math.min(1, soil.theta[idx] / 0.55));
        const cx = -halfX + (ix + 0.5) * dx;
        const cz = -halfZ + (iz + 0.5) * dx;

        const splatsPerCell = mat === SoilMaterialId.HUMUS ? 6 : mat === SoilMaterialId.WOOD ? 4 : 3;
        for (let k = 0; k < splatsPerCell && count < MAX_SOIL_SPLATS; k++) {
          const seed = idx * 13 + k * 37;
          const jx = (hash1(seed + 1) - 0.5) * dx * 0.96;
          const jz = (hash1(seed + 2) - 0.5) * dx * 0.96;
          const jy = 0.0006 + hash1(seed + 3) * 0.0034;
          const yaw = hash1(seed + 4) * Math.PI * 2;
          const tX = (hash1(seed + 5) - 0.5) * 0.48;
          const tZ = (hash1(seed + 6) - 0.5) * 0.48;
          const varTone = (hash1(seed + 7) - 0.5) * 0.16;

          let r = 0.14, g = 0.12, b = 0.07;
          let sx = 0.0028, sy = 0.0012, sz = 0.0024, alpha = 0.78;

          if (mat === SoilMaterialId.HUMUS) {
            const mossRoll = hash1(seed + 8);
            if (mossRoll > 0.40) {
              // Crisp velvety 3D moss tuft
              r = 0.075 + varTone * 0.35;
              g = 0.21 + varTone * 0.75;
              b = 0.055 + varTone * 0.20;
              sx = 0.0024 + hash1(seed + 9) * 0.0015;
              sy = 0.0016 + hash1(seed + 10) * 0.0012;
              sz = sx * 0.86;
              alpha = 0.78;
            } else if (mossRoll > 0.16) {
              // Fallen decaying forest leaf-litter flake
              r = 0.24 + varTone * 0.5;
              g = 0.13 + varTone * 0.3;
              b = 0.062;
              sx = 0.0042;
              sy = 0.0008;
              sz = 0.0020;
              alpha = 0.80;
            } else {
              // Dark rich humus crumb
              r = 0.10 + varTone * 0.28;
              g = 0.07 + varTone * 0.18;
              b = 0.048;
              sx = 0.0026;
              sy = 0.0013;
              sz = 0.0023;
              alpha = 0.78;
            }
          } else if (mat === SoilMaterialId.WOOD) {
            // Fibrous decaying oak bark splinter
            r = 0.27 + varTone * 0.5;
            g = 0.15 + varTone * 0.3;
            b = 0.075;
            sx = 0.0046;
            sy = 0.0010;
            sz = 0.0015;
            alpha = 0.78;
          } else if (mat === SoilMaterialId.SAND) {
            r = 0.54 + varTone * 0.5;
            g = 0.45 + varTone * 0.4;
            b = 0.31 + varTone * 0.3;
            sx = 0.0021;
            sy = 0.0009;
            sz = 0.0019;
            alpha = 0.72;
          } else if (mat === SoilMaterialId.STONE) {
            // Mossy granite fleck on exposed surface stone
            r = 0.15 + varTone * 0.3;
            g = 0.23 + varTone * 0.4;
            b = 0.13 + varTone * 0.2;
            sx = 0.0026;
            sy = 0.0010;
            sz = 0.0022;
            alpha = 0.72;
          } else {
            // Loam / Clay surface crumbs
            r = (mat === SoilMaterialId.CLAY ? 0.32 : 0.22) + varTone * 0.4;
            g = (mat === SoilMaterialId.CLAY ? 0.20 : 0.15) + varTone * 0.3;
            b = (mat === SoilMaterialId.CLAY ? 0.15 : 0.095) + varTone * 0.2;
            sx = 0.0027;
            sy = 0.0011;
            sz = 0.0023;
            alpha = 0.72;
          }

          const dark = 1.0 - wet * 0.52;
          const qOff = this.pushSplat(
            count++,
            cx + jx, jy, cz + jz, alpha,
            sx, sy, sz, 0,
            r * dark, g * dark, b * dark, wet,
          );
          writeQuatYawTilt(d, qOff, yaw, tX, tZ);
        }
      }
    }

    // 2. Subtle organic relief flecks on exposed rotting oak log & humus cutaway faces
    for (let iy = 1; iy < ny - 1 && count < MAX_SOIL_SPLATS - 2; iy++) {
      const wy = -soil.sizeY + (iy + 0.5) * dx;
      for (let ix = 0; ix < nx && count < MAX_SOIL_SPLATS - 2; ix++) {
        const izFront = nz - 1;
        const idx = soil.voxelIndex(ix, iy, izFront);
        const mat = soil.material[idx];
        if (mat !== SoilMaterialId.WOOD && mat !== SoilMaterialId.HUMUS) continue;
        const wx = -halfX + (ix + 0.5) * dx + (hash1(idx * 19 + 1) - 0.5) * dx * 0.7;
        const wz = halfZ + 0.0009;
        const isWood = mat === SoilMaterialId.WOOD;
        const qOff = this.pushSplat(
          count++,
          wx, wy + (hash1(idx * 19 + 2) - 0.5) * dx * 0.7, wz, isWood ? 0.52 : 0.44,
          isWood ? 0.0075 : 0.0048, isWood ? 0.0022 : 0.0032, 0.0015, 0,
          isWood ? 0.22 : 0.09, isWood ? 0.13 : 0.065, isWood ? 0.065 : 0.042, 0.5,
        );
        writeQuatYawTilt(d, qOff, (hash1(idx * 19 + 3) - 0.5) * 0.5, 0, 0);
      }
    }

    this.soilSplatCount = count;
  }

  private buildDynamicSplats(snapshot: SimulationSnapshot, view: ViewState): number {
    this.buildSoilSplatsIfNeeded(snapshot);
    let count = this.soilSplatCount;
    const d = this.rawScratch;

    // 2. Cottony Mycelial Halo Splats (near-surface / exposed cords & hyphal tips)
    if (view.showMycelium) {
      const segs = snapshot.mycelium.segments;
      const segCount = Math.min(snapshot.mycelium.segmentCount, Math.floor(segs.length / 8));
      const maxEnd = this.soilSplatCount + MAX_MYC_SPLATS;
      const stride = segCount > MAX_MYC_SPLATS / 2 ? 2 : 1;
      for (let i = 0; i < segCount && count < maxEnd - 2; i += stride) {
        const s8 = i * 8;
        const ax = segs[s8], ay = segs[s8 + 1], az = segs[s8 + 2];
        const bx = segs[s8 + 3], by = segs[s8 + 4], bz = segs[s8 + 5];
        const rad = segs[s8 + 6];
        const turgor = Math.max(0, Math.min(1, segs[s8 + 7]));

        const dx = bx - ax, dy = by - ay, dz = bz - az;
        const segLen = Math.hypot(dx, dy, dz);
        if (segLen < 1e-5) continue;

        const mx = (ax + bx) * 0.5;
        const my = (ay + by) * 0.5;
        const mz = (az + bz) * 0.5;

        const cr = 0.46 + 0.44 * turgor;
        const cg = 0.40 + 0.40 * turgor;
        const cb = 0.28 + 0.34 * turgor;
        const rVis = Math.max(0.0007, Math.min(0.0018, rad * 4.2));

        const qOff = this.pushSplat(
          count++,
          mx, my, mz, 0.10 + 0.10 * turgor,
          segLen * 0.42, rVis, rVis, 1,
          cr, cg, cb, turgor,
        );
        writeQuatFromDirX(d, qOff, dx, dy, dz);
      }
    }

    // 3. Triangle-Bound Cap Scales, Velvet Fuzz & Basal Stipe Rhizomorphs
    const mesh = snapshot.fruitingMesh;
    if (mesh.vertexCount > 0) {
      const sp = snapshot.species;
      const isInkcap = sp.id === 'inkcap';
      const isParasol = sp.id === 'parasol';
      const isChanterelle = sp.id === 'chanterelle';

      const maxFuzzEnd = Math.min(MAX_SPLATS - MAX_SPORE_SPLATS, count + MAX_FUZZ_SPLATS);
      const vCount = mesh.vertexCount;
      const targetSamples = isInkcap ? 2600 : 1200;
      const step = Math.max(1, Math.floor(vCount / targetSamples));
      const pos = mesh.positions;
      const nrm = mesh.normals;
      const col = mesh.colors;
      const pid = mesh.partIds;

      for (let vi = 0; vi < vCount && count < maxFuzzEnd; vi += step) {
        const part = Math.round(pid[vi]);
        const v3 = vi * 3;
        const px = pos[v3], py = pos[v3 + 1], pz = pos[v3 + 2];
        const nx = nrm[v3], ny = nrm[v3 + 1], nz = nrm[v3 + 2];
        const r0 = hash1(vi * 23 + 1);
        const r1 = hash1(vi * 23 + 2);

        if (part === 0 && py < 0.013) {
          // Basal stipe rhizomorph "fuzzy feet"
          const qOff = this.pushSplat(
            count++,
            px + nx * 0.0004, py, pz + nz * 0.0004, 0.34,
            0.0018, 0.0010, 0.0010, 2,
            0.90, 0.86, 0.76, 0.5,
          );
          writeQuatFromDirX(d, qOff, nx, ny + 0.35, nz);
        } else if (part === 1 && ny > 0.25) {
          // Tangent direction orthogonal to surface normal so scales hug the cap instead of spiking out
          const tx = -nz;
          const tz = nx;
          if (isInkcap && r0 > 0.35) {
            // Curled shaggy Coprinus scales
            const qOff = this.pushSplat(
              count++,
              px + nx * 0.0006, py + ny * 0.0006, pz + nz * 0.0006, 0.55,
              0.0026, 0.0010, 0.0015, 2,
              col[v3] * 0.90, col[v3 + 1] * 0.86, col[v3 + 2] * 0.78, 0.4,
            );
            writeQuatFromDirX(d, qOff, tx, ny * 0.25, tz);
          } else if (isParasol && r0 > 0.65) {
            // Subtle flat parasol cap velum scales
            const qOff = this.pushSplat(
              count++,
              px + nx * 0.00025, py + ny * 0.00025, pz + nz * 0.00025, 0.36,
              0.0018, 0.0006, 0.0016, 2,
              0.66 + r1 * 0.14, 0.54 + r1 * 0.12, 0.42 + r1 * 0.10, 0.3,
            );
            writeQuatFromDirX(d, qOff, tx, 0, tz);
          } else if ((isChanterelle || sp.id === 'morel') && r0 > 0.78) {
            // Matte velvety micro-fuzz
            const qOff = this.pushSplat(
              count++,
              px + nx * 0.0002, py + ny * 0.0002, pz + nz * 0.0002, 0.24,
              0.0014, 0.0008, 0.0014, 2,
              col[v3] * 1.0, col[v3 + 1] * 0.98, col[v3 + 2] * 0.92, 0.3,
            );
            writeQuatFromDirX(d, qOff, tx, 0, tz);
          }
        }
      }
    }

    // 4. Volumetric Velocity-Stretched Dressaire Spore Mist
    if (view.showSpores) {
      const spBuf = snapshot.spores;
      const slots = Math.min(spBuf.maxCount, spBuf.active.length, spBuf.positions.length >> 2);
      const act = spBuf.active;
      const spos = spBuf.positions;
      const svel = spBuf.velocities;
      for (let i = 0; i < slots && count < MAX_SPLATS - 2; i++) {
        if (act[i] === 0) continue;
        const i4 = i * 4;
        const px = spos[i4], py = spos[i4 + 1], pz = spos[i4 + 2];
        if (py < -0.45) continue;
        const vx = svel[i4], vy = svel[i4 + 1], vz = svel[i4 + 2];
        const liftM = svel[i4 + 3];
        const speed = Math.hypot(vx, vy, vz);

        // Crisp glowing spore nucleus splat
        const qOffCore = this.pushSplat(
          count++,
          px, py, pz, 0.74,
          0.00105, 0.00105, 0.00105, 3,
          1, 1, 1, liftM,
        );
        d[qOffCore] = 0; d[qOffCore + 1] = 0; d[qOffCore + 2] = 0; d[qOffCore + 3] = 1;

        // Delicate velocity-stretched Tyndall volumetric mist streak (1 every 2nd spore so mist stays airy)
        if ((i & 1) === 0 && count < MAX_SPLATS) {
          const stretch = Math.min(0.011, 0.0024 + speed * 0.32);
          const qOffHaze = this.pushSplat(
            count++,
            px, py, pz, 0.042,
            stretch, 0.0022, 0.0022, 3,
            1, 1, 1, liftM,
          );
          if (speed > 1e-4) {
            writeQuatFromDirX(d, qOffHaze, vx, vy, vz);
          } else {
            d[qOffHaze] = 0; d[qOffHaze + 1] = 0; d[qOffHaze + 2] = 0; d[qOffHaze + 3] = 1;
          }
        }
      }
    }

    return count;
  }

  updateAndSort(
    encoder: GPUCommandEncoder,
    snapshot: SimulationSnapshot,
    view: ViewState,
    viewTiltMat: Mat4,
    projMat: Mat4,
    viewportW: number,
    viewportH: number,
  ): void {
    this.activeSplatCount = this.buildDynamicSplats(snapshot, view);

    const focalX = projMat[0] * viewportW * 0.5;
    const focalY = projMat[5] * viewportH * 0.5;
    const xrayIdx = XRAY_INDEX[view.xrayMode] ?? 0;

    const p = this.paramsScratch;
    p.set(viewTiltMat, 0);
    p.set(projMat, 16);
    p[32] = focalX;
    p[33] = focalY;
    p[34] = viewportW;
    p[35] = viewportH;
    p[36] = this.activeSplatCount;
    p[37] = xrayIdx;
    p[38] = view.showMycelium ? 1.0 : 0.0;
    p[39] = view.showSpores ? 1.0 : 0.0;
    p[40] = 0.02; // near
    p[41] = 4.0;  // far
    p[42] = 0.0;
    p[43] = 0.0;

    this.device.queue.writeBuffer(this.paramsBuf, 0, p.buffer, p.byteOffset, 176);
    if (this.activeSplatCount > 0) {
      this.device.queue.writeBuffer(
        this.rawBuf,
        0,
        this.rawScratch.buffer,
        this.rawScratch.byteOffset,
        this.activeSplatCount * FLOATS_PER_SPLAT * 4,
      );
    }

    // 1. Project & light all MAX_SPLATS slots (sets key=65535 on culled/inactive slots)
    const projPass = encoder.beginComputePass();
    projPass.setPipeline(this.projectPipe);
    projPass.setBindGroup(0, this.projectBindGroup);
    projPass.dispatchWorkgroups(NUM_WG);
    projPass.end();

    if (this.activeSplatCount === 0) return;

    // 2. 2-Pass 8-Bit GPU LSD Radix Sort (shift=0 A->B, shift=8 B->A)
    const dispatchSortPass = (bg: GPUBindGroup): void => {
      const pHist = encoder.beginComputePass();
      pHist.setPipeline(this.histPipe);
      pHist.setBindGroup(0, bg);
      pHist.dispatchWorkgroups(NUM_WG);
      pHist.end();

      const pPre = encoder.beginComputePass();
      pPre.setPipeline(this.prefixPipe);
      pPre.setBindGroup(0, bg);
      pPre.dispatchWorkgroups(1);
      pPre.end();

      const pScat = encoder.beginComputePass();
      pScat.setPipeline(this.scatterPipe);
      pScat.setBindGroup(0, bg);
      pScat.dispatchWorkgroups(NUM_WG);
      pScat.end();
    };

    dispatchSortPass(this.sortPass0BindGroup);
    dispatchSortPass(this.sortPass1BindGroup);
  }

  draw(pass: GPURenderPassEncoder): void {
    if (this.activeSplatCount <= 0) return;
    pass.setPipeline(this.drawPipe);
    pass.setBindGroup(0, this.drawBindGroup);
    pass.setVertexBuffer(0, this.quadBuf);
    pass.draw(6, this.activeSplatCount);
  }

  dispose(): void {
    this.rawBuf.destroy();
    this.projBuf.destroy();
    this.paramsBuf.destroy();
    this.keysABuf.destroy();
    this.valsABuf.destroy();
    this.keysBBuf.destroy();
    this.valsBBuf.destroy();
    this.wgHistBuf.destroy();
    this.wgOffsetsBuf.destroy();
    this.globalBinOffsetsBuf.destroy();
    this.shift0Buf.destroy();
    this.shift8Buf.destroy();
  }
}
