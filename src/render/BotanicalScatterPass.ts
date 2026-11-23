// ============================================================================
// SPOROCARP — Instanced Micro-Botanical Forest-Floor Scatter Pass (WebGPU)
// Renders crisp 3D silhouettes (curved fallen oak leaves, branched moss
// fronds, bark/pine needles, faceted pebbles) grounded on the displaced
// terrain heightfield, shaded with PCF shadows + Fast-SDF-AO + 2-bounce GI.
// ============================================================================
import {
  SoilMaterialId,
  type FruitingBodyState,
  type SimulationSnapshot,
} from '../types/contract.ts';
import { evalSoilSurfaceHeightM } from './terrainHeight.ts';
import { WGSL_COMMON, makeWgslSdfGiHelpers, makeWgslShadowHelpers } from './shaders.ts';

const MAX_INSTANCES = 3200;
const FLOATS_PER_INST = 12; // instPosScale(4), instQuat(4), instColWet(4)

const BOTANICAL_WGSL = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> u: SceneUniforms;
${makeWgslSdfGiHelpers(1, 'sdfSamp', 3, 2)}
${makeWgslShadowHelpers(4, 5)}

struct BotVertIn {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) instPosScale: vec4<f32>, // xyz = world pos, w = scale
  @location(3) instQuat: vec4<f32>,     // qx, qy, qz, qw
  @location(4) instColWet: vec4<f32>,   // rgb = linear albedo, w = wetness
};

struct BotVertOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) colWet: vec4<f32>,
};

fn rotateQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
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
  out.colWet = v.instColWet;
  return out;
}

@fragment
fn fs_main(in: BotVertOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4<f32> {
  if (isClippedBySlice(in.worldPos, u.sliceParams)) {
    discard;
  }
  let rawN = normalize(in.worldNormal);
  let n = select(-rawN, rawN, frontFacing);
  let l = normalize(u.lightDir.xyz);
  let v = normalize(u.cameraPos.xyz - in.worldPos);
  let h = normalize(l + v);

  let aoBent = evalFastSdfAO(in.worldPos + n * 0.004, n);
  let bentN = aoBent.xyz;
  let sdfAo = aoBent.w;
  let shadow = min(evalPcfShadow(in.worldPos, n), mix(0.35, 1.0, evalSdfSoftShadow(in.worldPos + n * 0.006, l)));
  let gi = evalSdfGI(in.worldPos + n * 0.005, n, bentN);

  // 3-point macro botanical lighting (sun key + bent sky fill + grazing rim SSS)
  let wrapNdl = clamp((dot(n, l) + 0.42) / 1.42, 0.0, 1.0);
  let sun = vec3<f32>(0.96, 0.90, 0.78) * wrapNdl * mix(0.24, 1.0, shadow);
  let sky = mix(vec3<f32>(0.032, 0.038, 0.044), vec3<f32>(0.11, 0.14, 0.18), bentN.y * 0.5 + 0.5);
  let backSss = pow(clamp(dot(v, -l + n * 0.35), 0.0, 1.0), 2.4) * 0.18 * (0.35 + 0.65 * shadow);
  let albedo = pow(max(in.colWet.rgb, vec3<f32>(0.0)), vec3<f32>(2.2));
  let sssTint = albedo * vec3<f32>(1.15, 1.25, 0.65);
  let spec = pow(max(dot(n, h), 0.0), 28.0) * (0.02 + 0.10 * in.colWet.w) * shadow * sdfAo;

  let color = albedo * ((sky + sun) * sdfAo + gi * 0.52) + sssTint * backSss + vec3<f32>(spec);
  return vec4<f32>(color, 1.0);
}
`;

function hash1(n: number): number {
  const x = Math.sin(n * 12.9898 + 78.233) * 43758.5453123;
  return x - Math.floor(x);
}

function writeQuatYawPitchRoll(
  out: Float32Array,
  off: number,
  yaw: number,
  pitch: number,
  roll: number,
): void {
  const cy = Math.cos(yaw * 0.5), sy = Math.sin(yaw * 0.5);
  const cp = Math.cos(pitch * 0.5), sp = Math.sin(pitch * 0.5);
  const cr = Math.cos(roll * 0.5), sr = Math.sin(roll * 0.5);
  out[off] = sr * cp * cy - cr * sp * sy;
  out[off + 1] = cr * sp * cy + sr * cp * sy;
  out[off + 2] = cr * cp * sy - sr * sp * cy;
  out[off + 3] = cr * cp * cy + sr * sp * sy;
}

interface SubMeshSlice {
  firstIndex: number;
  indexCount: number;
}

function buildArchetypeGeometry(): {
  verts: Float32Array;
  indices: Uint16Array;
  slices: SubMeshSlice[];
} {
  const v: number[] = [];
  const idx: number[] = [];
  const slices: SubMeshSlice[] = [];

  const pushVert = (px: number, py: number, pz: number, nx: number, ny: number, nz: number): number => {
    const l = Math.hypot(nx, ny, nz) || 1;
    const vi = v.length / 6;
    v.push(px, py, pz, nx / l, ny / l, nz / l);
    return vi;
  };
  const pushTri = (a: number, b: number, c: number) => idx.push(a, b, c);

  // 0: Lobed Fallen Oak Leaf lying nearly flat on soil contour
  {
    const firstIndex = idx.length;
    const v0 = pushVert(-0.90, 0.02, 0.0, -0.1, 1.0, 0.0);
    const v1 = pushVert(-0.35, 0.015, -0.44, 0.0, 1.0, -0.12);
    const v2 = pushVert(-0.32, 0.005, 0.0, 0.0, 1.0, 0.0);
    const v3 = pushVert(-0.35, 0.015, 0.44, 0.0, 1.0, 0.12);
    const v4 = pushVert(0.28, 0.035, -0.52, 0.06, 0.99, -0.12);
    const v5 = pushVert(0.30, 0.020, 0.0, 0.04, 1.0, 0.0);
    const v6 = pushVert(0.28, 0.035, 0.52, 0.06, 0.99, 0.12);
    const v7 = pushVert(0.86, 0.075, 0.0, 0.14, 0.99, 0.0);
    pushTri(v0, v1, v2); pushTri(v0, v2, v3);
    pushTri(v1, v4, v5); pushTri(v1, v5, v2);
    pushTri(v2, v5, v6); pushTri(v2, v6, v3);
    pushTri(v4, v7, v5); pushTri(v5, v7, v6);
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 1: Soft Pillow Moss Cushion Mound (smooth convex hemisphere — zero spiky shards)
  {
    const firstIndex = idx.length;
    const crown = pushVert(0.0, 0.42, 0.0, 0.0, 1.0, 0.0);
    const segs = 7;
    const midRing: number[] = [];
    const baseRing: number[] = [];
    for (let k = 0; k < segs; k++) {
      const a = (k / segs) * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      midRing.push(pushVert(0.58 * ca, 0.27, 0.58 * sa, 0.48 * ca, 0.88, 0.48 * sa));
      baseRing.push(pushVert(0.95 * ca, 0.0, 0.95 * sa, 0.65 * ca, 0.76, 0.65 * sa));
    }
    for (let k = 0; k < segs; k++) {
      const k1 = (k + 1) % segs;
      pushTri(crown, midRing[k], midRing[k1]);
      pushTri(midRing[k], baseRing[k], baseRing[k1]);
      pushTri(midRing[k], baseRing[k1], midRing[k1]);
    }
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 2: Slender Fallen Conifer Needle / Cedar Bark Sliver
  {
    const firstIndex = idx.length;
    const a0 = pushVert(-0.88, 0.015, -0.055, 0, 1, -0.1);
    const a1 = pushVert(-0.88, 0.015, 0.055, 0, 1, 0.1);
    const b0 = pushVert(0.0, 0.042, -0.065, 0, 1, -0.1);
    const b1 = pushVert(0.0, 0.042, 0.065, 0, 1, 0.1);
    const c0 = pushVert(0.88, 0.012, -0.032, 0, 1, -0.08);
    const c1 = pushVert(0.88, 0.012, 0.032, 0, 1, 0.08);
    pushTri(a0, b0, b1); pushTri(a0, b1, a1);
    pushTri(b0, c0, c1); pushTri(b0, c1, b1);
    slices.push({ firstIndex, indexCount: idx.length - firstIndex });
  }

  // 3: Rounded Smooth Mineral Pebble
  {
    const firstIndex = idx.length;
    const crown = pushVert(0.0, 0.44, 0.0, 0.0, 1.0, 0.0);
    const segs = 6;
    const midRing: number[] = [];
    const baseRing: number[] = [];
    for (let k = 0; k < segs; k++) {
      const a = (k / segs) * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      const rMod = 0.88 + 0.14 * Math.sin(k * 2.3);
      midRing.push(pushVert(0.62 * rMod * ca, 0.28, 0.56 * rMod * sa, 0.55 * ca, 0.82, 0.55 * sa));
      baseRing.push(pushVert(0.92 * rMod * ca, 0.0, 0.84 * rMod * sa, 0.72 * ca, 0.68, 0.72 * sa));
    }
    for (let k = 0; k < segs; k++) {
      const k1 = (k + 1) % segs;
      pushTri(crown, midRing[k], midRing[k1]);
      pushTri(midRing[k], baseRing[k], baseRing[k1]);
      pushTri(midRing[k], baseRing[k1], midRing[k1]);
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
  private counts = [0, 0, 0, 0];
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
            arrayStride: 24,
            stepMode: 'vertex',
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
            ],
          },
          {
            arrayStride: 48,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 2, offset: 0, format: 'float32x4' },
              { shaderLocation: 3, offset: 16, format: 'float32x4' },
              { shaderLocation: 4, offset: 32, format: 'float32x4' },
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
    let sigNum = soil.nx * 73856093 ^ soil.nz * 83492791 ^ snapshot.bodies.length * 19349663;
    const total = soil.material.length;
    const stride = Math.max(1, Math.floor(total / 192));
    for (let i = 0; i < total; i += stride) {
      sigNum = (sigNum * 31 + soil.material[i] * 17 + Math.round(soil.theta[i] * 20)) | 0;
    }
    const sig = `${sigNum}:${snapshot.species.id}`;
    if (sig === this.lastSig) return;
    this.lastSig = sig;

    const buckets: [number[], number[], number[], number[]] = [[], [], [], []];
    const bodies: readonly FruitingBodyState[] = snapshot.bodies;
    const halfX = soil.sizeX * 0.5;
    const halfZ = soil.sizeZ * 0.5;
    const iyTop = soil.ny - 1;
    const totalSamples = 2500;

    for (let i = 0; i < totalSamples; i++) {
      const rx = hash1(i * 17 + 1);
      const rz = hash1(i * 17 + 2);
      const wx = (rx - 0.5) * (soil.sizeX - 0.012);
      const wz = (rz - 0.5) * (soil.sizeZ - 0.012);

      // Avoid placing props inside active stipe stems
      let nearStipe = false;
      for (let b = 0; b < bodies.length; b++) {
        if (Math.hypot(wx - bodies[b].baseX, wz - bodies[b].baseZ) < 0.014) {
          nearStipe = true;
          break;
        }
      }
      if (nearStipe) continue;

      const ix = Math.max(0, Math.min(soil.nx - 1, Math.floor((wx + halfX) / soil.dx)));
      const iz = Math.max(0, Math.min(soil.nz - 1, Math.floor((wz + halfZ) / soil.dx)));
      const vIdx = soil.voxelIndex(ix, iyTop, iz);
      const mat = soil.material[vIdx];
      const wet = Math.max(0, Math.min(1, soil.theta[vIdx] / 0.55));
      const wy = evalSoilSurfaceHeightM(wx, wz, soil, bodies);

      const roll = hash1(i * 17 + 3);
      const tone = (hash1(i * 17 + 4) - 0.5) * 0.14;
      const yaw = hash1(i * 17 + 5) * Math.PI * 2;
      const pitch = (hash1(i * 17 + 6) - 0.5) * 0.16;
      const rollA = (hash1(i * 17 + 7) - 0.5) * 0.16;

      let kind = 1; // moss default
      let scale = 0.0055;
      let r = 0.09, g = 0.24, b = 0.06;

      if (mat === SoilMaterialId.HUMUS) {
        if (roll > 0.44) {
          kind = 1; // velvety moss cushion mound
          scale = 0.0046 + hash1(i * 17 + 8) * 0.0042;
          r = 0.22 + tone * 0.4;
          g = 0.42 + tone * 0.7;
          b = 0.14 + tone * 0.25;
        } else if (roll > 0.20) {
          kind = 0; // fallen curved oak leaf
          scale = 0.0055 + hash1(i * 17 + 8) * 0.0042;
          r = 0.50 + tone * 0.6;
          g = 0.27 + tone * 0.35;
          b = 0.10;
        } else if (roll > 0.07) {
          kind = 2; // conifer needle / twig flake
          scale = 0.0048 + hash1(i * 17 + 8) * 0.0038;
          r = 0.38 + tone * 0.4;
          g = 0.23 + tone * 0.3;
          b = 0.12;
        } else {
          kind = 3; // rounded river pebble
          scale = 0.0026 + hash1(i * 17 + 8) * 0.0020;
          r = 0.42 + tone * 0.3; g = 0.43 + tone * 0.3; b = 0.45 + tone * 0.3;
        }
      } else if (mat === SoilMaterialId.WOOD) {
        kind = roll > 0.42 ? 2 : 1; // bark splinter or moss cushion on log
        scale = kind === 2 ? 0.0056 : 0.0044;
        if (kind === 2) {
          r = 0.42 + tone * 0.4; g = 0.25 + tone * 0.25; b = 0.13;
        } else {
          r = 0.22; g = 0.40 + tone * 0.5; b = 0.14;
        }
      } else if (mat === SoilMaterialId.SAND) {
        kind = 3; // quartz pebble
        scale = 0.0022 + hash1(i * 17 + 8) * 0.0020;
        r = 0.62 + tone * 0.4; g = 0.54 + tone * 0.35; b = 0.40 + tone * 0.3;
      } else if (mat === SoilMaterialId.STONE) {
        kind = roll > 0.48 ? 1 : 3;
        scale = kind === 1 ? 0.0045 : 0.0030;
        if (kind === 1) {
          r = 0.22; g = 0.40 + tone * 0.4; b = 0.14;
        } else {
          r = 0.44 + tone * 0.4; g = 0.46 + tone * 0.4; b = 0.48 + tone * 0.4;
        }
      } else {
        kind = roll > 0.55 ? 0 : roll > 0.25 ? 2 : 3;
        scale = kind === 3 ? 0.0024 : 0.0050;
        r = 0.40 + tone * 0.4; g = 0.24 + tone * 0.25; b = 0.12;
      }

      const dark = 1.0 - wet * 0.28;
      const arr = buckets[kind];
      arr.push(
        wx, wy - 0.0003, wz, scale,
        yaw, pitch, rollA, 0,
        r * dark, g * dark, b * dark, wet,
      );
    }

    let cursor = 0;
    const dst = this.instScratch;
    const qTmp = new Float32Array(4);
    for (let k = 0; k < 4; k++) {
      const src = buckets[k];
      let c = 0;
      for (let i = 0; i < src.length && cursor < MAX_INSTANCES; i += 12) {
        const o = cursor * FLOATS_PER_INST;
        dst[o] = src[i]; dst[o + 1] = src[i + 1]; dst[o + 2] = src[i + 2]; dst[o + 3] = src[i + 3];
        writeQuatYawPitchRoll(qTmp, 0, src[i + 4], src[i + 5], src[i + 6]);
        dst[o + 4] = qTmp[0]; dst[o + 5] = qTmp[1]; dst[o + 6] = qTmp[2]; dst[o + 7] = qTmp[3];
        dst[o + 8] = src[i + 8]; dst[o + 9] = src[i + 9]; dst[o + 10] = src[i + 10]; dst[o + 11] = src[i + 11];
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
    const total = this.counts[0] + this.counts[1] + this.counts[2] + this.counts[3];
    if (total === 0) return;

    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, litBindGroup);
    pass.setVertexBuffer(0, this.geomVertBuf);
    pass.setVertexBuffer(1, this.instBuf);
    pass.setIndexBuffer(this.geomIdxBuf, 'uint16');

    let firstInst = 0;
    for (let k = 0; k < 4; k++) {
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
