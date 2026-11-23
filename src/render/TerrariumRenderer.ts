// ============================================================================
// SPOROCARP — 3D Terrarium Renderer (WebGPU Primary + WebGL2 Fallback)
// Zero-allocation hot path, 3D soil stratigraphy + razor slice plane,
// subterranean + exposed 3D mycelium, waxy chitin SSS mushrooms, glass
// terrarium architecture, obstacles, spore prints, airborne spores & plumes.
// ============================================================================

import {
  AMBIENT_O2_MOL_M3,
  FruitingBodyState,
  ISoilGrid,
  PickResult,
  SimulationSnapshot,
  SoilMaterialId,
  SolidObstacle,
  ViewState,
  XRayFieldMode,
} from '../types/contract.ts';
import {
  Vec3,
  computeOrbitCameraEye,
  intersectRayBox,
  intersectRayPlane,
  intersectRaySphere,
  mat4Identity,
  mat4Invert,
  mat4LookAt,
  mat4Multiply,
  mat4Perspective,
  mat4RotationZ,
  unprojectRay,
  vec3Normalize,
} from './math.ts';
import {
  BACKDROP_WGSL,
  MUSHROOM_GLSL_FS,
  MUSHROOM_GLSL_VS,
  MUSHROOM_WGSL,
  MYCELIUM_GLSL_FS,
  MYCELIUM_GLSL_VS,
  MYCELIUM_WGSL,
  SOIL_GLSL_FS,
  SOIL_GLSL_VS,
  SOIL_WGSL,
  SOLID_GLSL_FS,
  SOLID_GLSL_VS,
  SOLID_WGSL,
  SPORE_GLSL_FS,
  SPORE_GLSL_VS,
  STREAMLINE_GLSL_FS,
  STREAMLINE_GLSL_VS,
  STREAMLINE_WGSL,
} from './shaders.ts';
import { FastSdfGiSystem, SDF_NX, type SdfVolumeBounds } from './FastSdfGi.ts';
import { ProceduralSplatPass } from './ProceduralSplatPass.ts';
import { evalSoilSurfaceHeightM, evalSoilSurfaceNormal } from './terrainHeight.ts';
import { ShadowMapPass } from './ShadowMapPass.ts';
import { BotanicalScatterPass } from './BotanicalScatterPass.ts';
import { PostProcessPass } from './PostProcessPass.ts';

export type RendererBackend = 'webgpu' | 'webgl2';

// Residual & saturated water content per SoilMaterialId (from THEORY.md Table 1)
// Used only for mapping theta -> normalized wetness S_e in [0, 1] for visual shading.
const MAT_THETA_R = new Float32Array([0.045, 0.067, 0.068, 0.08, 0.1, 0.0]);
const MAT_THETA_S = new Float32Array([0.43, 0.45, 0.38, 0.65, 0.72, 0.001]);

const XRAY_MODE_INDEX: Record<XRayFieldMode, number> = {
  none: 0,
  matric_psi: 1,
  carbon_enzyme: 2,
  turgor_p: 3,
  shell_stress: 4,
  air_convection: 5,
};

const QUAD_CORNERS = new Float32Array([
  0, -1, 1, -1, 1, 1,
  0, -1, 1, 1, 0, 1,
]);

const BILLBOARD_QUAD = new Float32Array([
  -1, -1, 1, -1, 1, 1,
  -1, -1, 1, 1, -1, 1,
]);

interface SoilBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
  dx: number;
  nx: number;
  ny: number;
  nz: number;
}

function getSoilBounds(snapshot: SimulationSnapshot): SoilBounds {
  const s = snapshot.soil;
  const [x0, y0, z0] = s.voxelToWorld(0, 0, 0);
  const [x1, y1, z1] = s.voxelToWorld(s.nx - 1, s.ny - 1, s.nz - 1);
  const h = s.dx * 0.5;
  return {
    minX: Math.min(x0, x1) - h,
    maxX: Math.max(x0, x1) + h,
    minY: Math.min(y0, y1) - h,
    maxY: Math.max(y0, y1) + h,
    minZ: Math.min(z0, z1) - h,
    maxZ: Math.max(z0, z1) + h,
    dx: s.dx,
    nx: s.nx,
    ny: s.ny,
    nz: s.nz,
  };
}

function computeSliceWorldCoord(bounds: SoilBounds, view: ViewState): number {
  const t = Math.max(0.02, Math.min(0.98, (view.slicePosition + 1) * 0.5));
  if (view.sliceAxis === 'x') {
    return bounds.minX + (bounds.maxX - bounds.minX) * t;
  }
  return bounds.minZ + (bounds.maxZ - bounds.minZ) * t;
}

function computeVisualTiltRad(snapshot: SimulationSnapshot): number {
  const [gx, gy] = snapshot.env.gravityDir;
  if (Math.abs(gx) < 1e-5 && gy < 0) return 0;
  return -Math.atan2(gx, -gy);
}

// ============================================================================
// Procedural Geometry Builders (writes into reusable Float32Arrays)
// ============================================================================

function appendQuad(
  buf: Float32Array,
  offset: number,
  p0: Vec3,
  p1: Vec3,
  p2: Vec3,
  p3: Vec3,
  n: Vec3,
  flag: number,
): number {
  const verts = [p0, p1, p2, p0, p2, p3];
  let o = offset;
  for (let i = 0; i < 6; i++) {
    const p = verts[i];
    buf[o++] = p[0];
    buf[o++] = p[1];
    buf[o++] = p[2];
    buf[o++] = n[0];
    buf[o++] = n[1];
    buf[o++] = n[2];
    buf[o++] = flag;
  }
  return o;
}

const TERRAIN_GRID_N = 36;
const TERRAIN_CORNERS = TERRAIN_GRID_N + 1;
const terrainYGrid = new Float32Array(TERRAIN_CORNERS * TERRAIN_CORNERS);
const terrainNGrid = new Float32Array(TERRAIN_CORNERS * TERRAIN_CORNERS * 3);

function appendSoilTri(
  buf: Float32Array,
  o: number,
  ax: number, ay: number, az: number, nax: number, nay: number, naz: number,
  bx: number, by: number, bz: number, nbx: number, nby: number, nbz: number,
  cx: number, cy: number, cz: number, ncx: number, ncy: number, ncz: number,
  flag: number,
): number {
  buf[o++] = ax; buf[o++] = ay; buf[o++] = az; buf[o++] = nax; buf[o++] = nay; buf[o++] = naz; buf[o++] = flag;
  buf[o++] = bx; buf[o++] = by; buf[o++] = bz; buf[o++] = nbx; buf[o++] = nby; buf[o++] = nbz; buf[o++] = flag;
  buf[o++] = cx; buf[o++] = cy; buf[o++] = cz; buf[o++] = ncx; buf[o++] = ncy; buf[o++] = ncz; buf[o++] = flag;
  return o;
}

function buildSoilAndSliceMesh(
  out: Float32Array,
  b: SoilBounds,
  view: ViewState,
  soil: ISoilGrid,
  bodies: readonly FruitingBodyState[],
): number {
  let o = 0;
  const { minX, maxX, minY, minZ, maxZ } = b;
  const spanX = maxX - minX;
  const spanZ = maxZ - minZ;
  const N = TERRAIN_GRID_N;
  const C = TERRAIN_CORNERS;

  // Sample displaced forest-floor heightfield & smooth normals across 37x37 grid corners
  for (let iz = 0; iz <= N; iz++) {
    const z = minZ + (iz / N) * spanZ;
    for (let ix = 0; ix <= N; ix++) {
      const x = minX + (ix / N) * spanX;
      const idx = iz * C + ix;
      terrainYGrid[idx] = evalSoilSurfaceHeightM(x, z, soil, bodies);
      const nrm = evalSoilSurfaceNormal(x, z, soil, bodies);
      const i3 = idx * 3;
      terrainNGrid[i3] = nrm[0];
      terrainNGrid[i3 + 1] = nrm[1];
      terrainNGrid[i3 + 2] = nrm[2];
    }
  }

  // Top (+Y) heightfield quads (CCW viewed from +Y)
  for (let iz = 0; iz < N; iz++) {
    const z0 = minZ + (iz / N) * spanZ;
    const z1 = minZ + ((iz + 1) / N) * spanZ;
    for (let ix = 0; ix < N; ix++) {
      const x0 = minX + (ix / N) * spanX;
      const x1 = minX + ((ix + 1) / N) * spanX;

      const ia = (iz + 1) * C + ix;
      const ib = (iz + 1) * C + (ix + 1);
      const ic = iz * C + (ix + 1);
      const id = iz * C + ix;

      const ya = terrainYGrid[ia], yb = terrainYGrid[ib], yc = terrainYGrid[ic], yd = terrainYGrid[id];
      const ia3 = ia * 3, ib3 = ib * 3, ic3 = ic * 3, id3 = id * 3;

      o = appendSoilTri(
        out, o,
        x0, ya, z1, terrainNGrid[ia3], terrainNGrid[ia3 + 1], terrainNGrid[ia3 + 2],
        x1, yb, z1, terrainNGrid[ib3], terrainNGrid[ib3 + 1], terrainNGrid[ib3 + 2],
        x1, yc, z0, terrainNGrid[ic3], terrainNGrid[ic3 + 1], terrainNGrid[ic3 + 2],
        0,
      );
      o = appendSoilTri(
        out, o,
        x0, ya, z1, terrainNGrid[ia3], terrainNGrid[ia3 + 1], terrainNGrid[ia3 + 2],
        x1, yc, z0, terrainNGrid[ic3], terrainNGrid[ic3 + 1], terrainNGrid[ic3 + 2],
        x0, yd, z0, terrainNGrid[id3], terrainNGrid[id3 + 1], terrainNGrid[id3 + 2],
        0,
      );
    }
  }

  // Bottom (-Y)
  o = appendQuad(out, o, [minX, minY, minZ], [maxX, minY, minZ], [maxX, minY, maxZ], [minX, minY, maxZ], [0, -1, 0], 0);

  // Contoured outer cutaway walls matching displaced terrain edge
  for (let ix = 0; ix < N; ix++) {
    const x0 = minX + (ix / N) * spanX;
    const x1 = minX + ((ix + 1) / N) * spanX;
    const y0 = terrainYGrid[N * C + ix];
    const y1 = terrainYGrid[N * C + ix + 1];
    o = appendQuad(out, o, [x0, minY, maxZ], [x1, minY, maxZ], [x1, y1, maxZ], [x0, y0, maxZ], [0, 0, 1], 0);
  }
  for (let ix = 0; ix < N; ix++) {
    const x0 = minX + (ix / N) * spanX;
    const x1 = minX + ((ix + 1) / N) * spanX;
    const y0 = terrainYGrid[ix];
    const y1 = terrainYGrid[ix + 1];
    o = appendQuad(out, o, [x1, minY, minZ], [x0, minY, minZ], [x0, y0, minZ], [x1, y1, minZ], [0, 0, -1], 0);
  }
  for (let iz = 0; iz < N; iz++) {
    const z0 = minZ + (iz / N) * spanZ;
    const z1 = minZ + ((iz + 1) / N) * spanZ;
    const y0 = terrainYGrid[iz * C + N];
    const y1 = terrainYGrid[(iz + 1) * C + N];
    o = appendQuad(out, o, [maxX, minY, z1], [maxX, minY, z0], [maxX, y0, z0], [maxX, y1, z1], [1, 0, 0], 0);
  }
  for (let iz = 0; iz < N; iz++) {
    const z0 = minZ + (iz / N) * spanZ;
    const z1 = minZ + ((iz + 1) / N) * spanZ;
    const y0 = terrainYGrid[iz * C];
    const y1 = terrainYGrid[(iz + 1) * C];
    o = appendQuad(out, o, [minX, minY, z0], [minX, minY, z1], [minX, y1, z1], [minX, y0, z0], [-1, 0, 0], 0);
  }

  // Obsidian laboratory plinth underneath soil (flag = 2)
  const pad = 0.012;
  const px0 = minX - pad;
  const px1 = maxX + pad;
  const pz0 = minZ - pad;
  const pz1 = maxZ + pad;
  const py0 = minY - 0.022;
  const py1 = minY - 0.0006;
  o = appendQuad(out, o, [px0, py1, pz1], [px1, py1, pz1], [px1, py1, pz0], [px0, py1, pz0], [0, 1, 0], 2);
  o = appendQuad(out, o, [px0, py0, pz1], [px1, py0, pz1], [px1, py1, pz1], [px0, py1, pz1], [0, 0, 1], 2);
  o = appendQuad(out, o, [px1, py0, pz0], [px0, py0, pz0], [px0, py1, pz0], [px1, py1, pz0], [0, 0, -1], 2);
  o = appendQuad(out, o, [px1, py0, pz1], [px1, py0, pz0], [px1, py1, pz0], [px1, py1, pz1], [1, 0, 0], 2);
  o = appendQuad(out, o, [px0, py0, pz0], [px0, py0, pz1], [px0, py1, pz1], [px0, py1, pz0], [-1, 0, 0], 2);

  // Dynamic Razor Slice cross-section face contoured to terrain (flag = 1; single strip prevents Z-fighting)
  if (view.sliceEnabled) {
    const sc = computeSliceWorldCoord(b, view);
    if (view.sliceAxis === 'x') {
      for (let iz = 0; iz < N; iz++) {
        const z0 = minZ + (iz / N) * spanZ;
        const z1 = minZ + ((iz + 1) / N) * spanZ;
        const y0 = evalSoilSurfaceHeightM(sc, z0, soil, bodies);
        const y1 = evalSoilSurfaceHeightM(sc, z1, soil, bodies);
        o = appendQuad(out, o, [sc, minY, z1], [sc, minY, z0], [sc, y0, z0], [sc, y1, z1], [1, 0, 0], 1);
      }
    } else {
      for (let ix = 0; ix < N; ix++) {
        const x0 = minX + (ix / N) * spanX;
        const x1 = minX + ((ix + 1) / N) * spanX;
        const y0 = evalSoilSurfaceHeightM(x0, sc, soil, bodies);
        const y1 = evalSoilSurfaceHeightM(x1, sc, soil, bodies);
        o = appendQuad(out, o, [x0, minY, sc], [x1, minY, sc], [x1, y1, sc], [x0, y0, sc], [0, 0, 1], 1);
      }
    }
  }
  return o / 7;
}

const SUBDIV_EDGE_HASH_SIZE = 65536;
const subdivHashKeyLo = new Int32Array(SUBDIV_EDGE_HASH_SIZE);
const subdivHashKeyHi = new Int32Array(SUBDIV_EDGE_HASH_SIZE);
const subdivHashVal = new Int32Array(SUBDIV_EDGE_HASH_SIZE);

interface MushSubdivState {
  verts: Float32Array;
  indices: Uint32Array;
  vertCount: number;
  indexCount: number;
}

/**
 * Packs mushroom geometry into 64-byte (16-float) WebGPU vertices with botanical
 * coordinates (dxNorm, dzNorm, relY, bodySeed) and applies 1-level PN-curved
 * midpoint subdivision when triangle count <= 10,000.
 */
function packSubdividedMushroomMesh(
  mesh: SimulationSnapshot['fruitingMesh'],
  bodies: readonly FruitingBodyState[],
  state: MushSubdivState,
): void {
  const origV = mesh.vertexCount;
  const origI = mesh.indexCount;
  if (origV <= 0 || origI <= 0) {
    state.vertCount = 0;
    state.indexCount = 0;
    return;
  }

  const triCount = Math.floor(origI / 3);
  const doSubdiv = triCount > 0 && triCount <= 10000 && origV < 32768;
  const maxVertsNeeded = doSubdiv ? origV + triCount * 3 : origV;
  const maxIdxNeeded = doSubdiv ? triCount * 12 : origI;

  if (state.verts.length < maxVertsNeeded * 16) {
    state.verts = new Float32Array(Math.max(maxVertsNeeded * 16, state.verts.length * 2));
  }
  if (state.indices.length < maxIdxNeeded) {
    state.indices = new Uint32Array(Math.max(maxIdxNeeded, state.indices.length * 2));
  }

  const dst = state.verts;
  const outIndices = state.indices;
  const pos = mesh.positions;
  const nrm = mesh.normals;
  const col = mesh.colors;
  const str = mesh.stressVM;
  const tur = mesh.turgorPa;
  const pid = mesh.partIds;
  const bodyCount = bodies.length;

  for (let i = 0; i < origV; i++) {
    const i3 = i * 3;
    const o = i * 16;
    const px = pos[i3];
    const py = pos[i3 + 1];
    const pz = pos[i3 + 2];
    const partId = pid[i];

    dst[o] = px;
    dst[o + 1] = py;
    dst[o + 2] = pz;
    dst[o + 3] = nrm[i3];
    dst[o + 4] = nrm[i3 + 1];
    dst[o + 5] = nrm[i3 + 2];
    dst[o + 6] = col[i3];
    dst[o + 7] = col[i3 + 1];
    dst[o + 8] = col[i3 + 2];
    dst[o + 9] = str[i];
    dst[o + 10] = tur[i];
    dst[o + 11] = partId;

    if (bodyCount > 0) {
      let best = bodies[0];
      if (bodyCount > 1) {
        let bestD2 = Infinity;
        for (let bi = 0; bi < bodyCount; bi++) {
          const b = bodies[bi];
          const rx = partId < 0.5 ? 0.5 * (b.baseX + b.stipeTipX) : b.stipeTipX;
          const ry = partId < 0.5 ? 0.5 * (b.baseY + b.stipeTipY) : b.stipeTipY;
          const rz = partId < 0.5 ? 0.5 * (b.baseZ + b.stipeTipZ) : b.stipeTipZ;
          const d2 = (px - rx) * (px - rx) + (py - ry) * (py - ry) + (pz - rz) * (pz - rz);
          if (d2 < bestD2) {
            bestD2 = d2;
            best = b;
          }
        }
      }
      const capR = Math.max(best.capRadiusM, 0.012);
      dst[o + 12] = (px - best.stipeTipX) / capR;
      dst[o + 13] = (pz - best.stipeTipZ) / capR;
      dst[o + 14] = (py - best.baseY) / Math.max(best.stipeLengthM + 0.015, 0.02);
      dst[o + 15] = (best.id * 1.6180339) % 6.2831853;
    } else {
      dst[o + 12] = 0;
      dst[o + 13] = 0;
      dst[o + 14] = 0.5;
      dst[o + 15] = 0;
    }
  }

  if (!doSubdiv) {
    outIndices.set(mesh.indices.subarray(0, origI));
    state.vertCount = origV;
    state.indexCount = origI;
    return;
  }

  subdivHashKeyLo.fill(-1);
  let nextVert = origV;
  let outIdx = 0;
  const srcIdx = mesh.indices;

  const getOrMakeMid = (a: number, b: number): number => {
    const lo = a < b ? a : b;
    const hi = a < b ? b : a;
    let slot = ((lo * 73856093) ^ (hi * 19349663)) & (SUBDIV_EDGE_HASH_SIZE - 1);
    for (let p = 0; p < 64; p++) {
      const kLo = subdivHashKeyLo[slot];
      if (kLo === -1) {
        const mid = nextVert++;
        const ao = a * 16;
        const bo = b * 16;
        const mo = mid * 16;

        const ax = dst[ao], ay = dst[ao + 1], az = dst[ao + 2];
        const bx = dst[bo], by = dst[bo + 1], bz = dst[bo + 2];
        const nax = dst[ao + 3], nay = dst[ao + 4], naz = dst[ao + 5];
        const nbx = dst[bo + 3], nby = dst[bo + 4], nbz = dst[bo + 5];

        const nd = Math.max(-1, Math.min(1, nax * nbx + nay * nby + naz * nbz));
        const crispEdge = dst[ao + 11] >= 1.5 || dst[bo + 11] >= 1.5;
        const smoothK = Math.max(0, Math.min(1, (nd - 0.30) / 0.70)) * (crispEdge ? 0.12 : 1.0);
        const dax = ax - bx, day = ay - by, daz = az - bz;
        const wA = dax * nax + day * nay + daz * naz;
        const wB = -dax * nbx - day * nby - daz * nbz;
        const pn = 0.125 * smoothK;

        dst[mo] = 0.5 * (ax + bx) + pn * (wA * nax + wB * nbx);
        dst[mo + 1] = 0.5 * (ay + by) + pn * (wA * nay + wB * nby);
        dst[mo + 2] = 0.5 * (az + bz) + pn * (wA * naz + wB * nbz);

        const mnx = nax + nbx;
        const mny = nay + nby;
        const mnz = naz + nbz;
        const mnLen = Math.hypot(mnx, mny, mnz);
        if (mnLen > 1e-6) {
          const inv = 1 / mnLen;
          dst[mo + 3] = mnx * inv;
          dst[mo + 4] = mny * inv;
          dst[mo + 5] = mnz * inv;
        } else {
          dst[mo + 3] = 0;
          dst[mo + 4] = 1;
          dst[mo + 5] = 0;
        }

        for (let k = 6; k < 16; k++) {
          dst[mo + k] = 0.5 * (dst[ao + k] + dst[bo + k]);
        }

        subdivHashKeyLo[slot] = lo;
        subdivHashKeyHi[slot] = hi;
        subdivHashVal[slot] = mid;
        return mid;
      }
      if (kLo === lo && subdivHashKeyHi[slot] === hi) {
        return subdivHashVal[slot];
      }
      slot = (slot + 1) & (SUBDIV_EDGE_HASH_SIZE - 1);
    }
    return lo;
  };

  for (let t = 0; t < triCount; t++) {
    const t3 = t * 3;
    const i0 = srcIdx[t3];
    const i1 = srcIdx[t3 + 1];
    const i2 = srcIdx[t3 + 2];
    if (i0 >= origV || i1 >= origV || i2 >= origV) continue;

    const m01 = getOrMakeMid(i0, i1);
    const m12 = getOrMakeMid(i1, i2);
    const m20 = getOrMakeMid(i2, i0);

    outIndices[outIdx++] = i0; outIndices[outIdx++] = m01; outIndices[outIdx++] = m20;
    outIndices[outIdx++] = m01; outIndices[outIdx++] = i1; outIndices[outIdx++] = m12;
    outIndices[outIdx++] = m20; outIndices[outIdx++] = m12; outIndices[outIdx++] = i2;
    outIndices[outIdx++] = m01; outIndices[outIdx++] = m12; outIndices[outIdx++] = m20;
  }

  state.vertCount = nextVert;
  state.indexCount = outIdx;
}

function appendSolidTri(
  out: Float32Array,
  offset: number,
  a: Vec3,
  b: Vec3,
  c: Vec3,
  na: Vec3,
  nb: Vec3,
  nc: Vec3,
  rgba: [number, number, number, number],
): number {
  let o = offset;
  const pts = [a, b, c];
  const nms = [na, nb, nc];
  for (let i = 0; i < 3; i++) {
    out[o++] = pts[i][0];
    out[o++] = pts[i][1];
    out[o++] = pts[i][2];
    out[o++] = nms[i][0];
    out[o++] = nms[i][1];
    out[o++] = nms[i][2];
    out[o++] = rgba[0];
    out[o++] = rgba[1];
    out[o++] = rgba[2];
    out[o++] = rgba[3];
  }
  return o;
}

function buildObstaclesMesh(
  obstacles: SolidObstacle[],
  out: Float32Array,
): number {
  let o = 0;
  const maxFloats = out.length - 3000;

  for (let idx = 0; idx < obstacles.length; idx++) {
    if (o > maxFloats) break;
    const obs = obstacles[idx];
    if (obs.kind === 'twig') {
      const p0: Vec3 = [obs.x, obs.y, obs.z];
      const p1: Vec3 = [obs.endX, obs.endY, obs.endZ];
      const dx = p1[0] - p0[0];
      const dy = p1[1] - p0[1];
      const dz = p1[2] - p0[2];
      const len = Math.hypot(dx, dy, dz);
      const axis: Vec3 = len > 1e-5 ? [dx / len, dy / len, dz / len] : [1, 0, 0];
      const ref: Vec3 = Math.abs(axis[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      const u = vec3Normalize([
        axis[1] * ref[2] - axis[2] * ref[1],
        axis[2] * ref[0] - axis[0] * ref[2],
        axis[0] * ref[1] - axis[1] * ref[0],
      ]);
      const v = vec3Normalize([
        axis[1] * u[2] - axis[2] * u[1],
        axis[2] * u[0] - axis[0] * u[2],
        axis[0] * u[1] - axis[1] * u[0],
      ]);
      const segs = 10;
      const r0 = Math.max(0.003, obs.radius);
      const r1 = r0 * 0.82;
      const bark: [number, number, number, number] = [0.26, 0.17, 0.11, 1.0];
      for (let i = 0; i < segs; i++) {
        const a0 = (i / segs) * Math.PI * 2;
        const a1 = ((i + 1) / segs) * Math.PI * 2;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);
        const n0: Vec3 = [u[0] * c0 + v[0] * s0, u[1] * c0 + v[1] * s0, u[2] * c0 + v[2] * s0];
        const n1: Vec3 = [u[0] * c1 + v[0] * s1, u[1] * c1 + v[1] * s1, u[2] * c1 + v[2] * s1];
        const v00: Vec3 = [p0[0] + n0[0] * r0, p0[1] + n0[1] * r0, p0[2] + n0[2] * r0];
        const v10: Vec3 = [p0[0] + n1[0] * r0, p0[1] + n1[1] * r0, p0[2] + n1[2] * r0];
        const v01: Vec3 = [p1[0] + n0[0] * r1, p1[1] + n0[1] * r1, p1[2] + n0[2] * r1];
        const v11: Vec3 = [p1[0] + n1[0] * r1, p1[1] + n1[1] * r1, p1[2] + n1[2] * r1];
        o = appendSolidTri(out, o, v00, v10, v11, n0, n1, n1, bark);
        o = appendSolidTri(out, o, v00, v11, v01, n0, n1, n0, bark);
        // Caps
        const negAxis: Vec3 = [-axis[0], -axis[1], -axis[2]];
        o = appendSolidTri(out, o, p0, v10, v00, negAxis, negAxis, negAxis, bark);
        o = appendSolidTri(out, o, p1, v01, v11, axis, axis, axis, bark);
      }
    } else {
      // Stone boulder: slightly oblate UV sphere
      const rings = 6;
      const sectors = 10;
      const r = Math.max(0.008, obs.radius);
      const col: [number, number, number, number] = [0.30, 0.33, 0.36, 1.0];
      for (let i = 0; i < rings; i++) {
        const th0 = (i / rings) * Math.PI;
        const th1 = ((i + 1) / rings) * Math.PI;
        for (let j = 0; j < sectors; j++) {
          const ph0 = (j / sectors) * Math.PI * 2;
          const ph1 = ((j + 1) / sectors) * Math.PI * 2;
          const pt = (th: number, ph: number): [Vec3, Vec3] => {
            const sx = Math.sin(th) * Math.cos(ph);
            const sy = Math.cos(th);
            const sz = Math.sin(th) * Math.sin(ph);
            const rock = 1 + 0.08 * Math.sin(ph * 3 + obs.id) * Math.sin(th * 2);
            return [
              [obs.x + sx * r * rock, obs.y + sy * r * 0.78 * rock, obs.z + sz * r * rock],
              vec3Normalize([sx, sy / 0.78, sz]),
            ];
          };
          const [a, na] = pt(th0, ph0);
          const [b, nb] = pt(th1, ph0);
          const [c, nc] = pt(th1, ph1);
          const [d, nd] = pt(th0, ph1);
          o = appendSolidTri(out, o, a, b, c, na, nb, nc, col);
          o = appendSolidTri(out, o, a, c, d, na, nc, nd, col);
        }
      }
    }
  }
  return o / 10;
}

function buildGlassArchitecture(
  b: SoilBounds,
  airTopY: number,
  view: ViewState,
  glassOut: Float32Array,
  linesOut: Float32Array,
): { glassVerts: number; lineVerts: number } {
  let go = 0;
  const eps = 0.0008;
  const x0 = b.minX - eps;
  const x1 = b.maxX + eps;
  const z0 = b.minZ - eps;
  const z1 = b.maxZ + eps;
  const y0 = b.minY;
  const y1 = airTopY;
  const glassCol: [number, number, number, number] = [0.22, 0.34, 0.46, 0.045];

  // Subtle back/side museum glass panes above soil line
  const addGlassQuad = (p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, n: Vec3) => {
    go = appendSolidTri(glassOut, go, p0, p1, p2, n, n, n, glassCol);
    go = appendSolidTri(glassOut, go, p0, p2, p3, n, n, n, glassCol);
  };
  addGlassQuad([x0, 0, z0], [x1, 0, z0], [x1, y1, z0], [x0, y1, z0], [0, 0, 1]);
  addGlassQuad([x0, 0, z1], [x1, 0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, -1]);
  addGlassQuad([x0, 0, z0], [x0, 0, z1], [x0, y1, z1], [x0, y1, z0], [1, 0, 0]);
  addGlassQuad([x1, 0, z0], [x1, 0, z1], [x1, y1, z1], [x1, y1, z0], [-1, 0, 0]);

  // Wireframe edges (normal = [0,0,0] triggers unlit hairline mode in shader)
  let lo = 0;
  const pushLine = (a: Vec3, bPt: Vec3, col: [number, number, number, number]) => {
    linesOut[lo++] = a[0]; linesOut[lo++] = a[1]; linesOut[lo++] = a[2];
    linesOut[lo++] = 0; linesOut[lo++] = 0; linesOut[lo++] = 0;
    linesOut[lo++] = col[0]; linesOut[lo++] = col[1]; linesOut[lo++] = col[2]; linesOut[lo++] = col[3];
    linesOut[lo++] = bPt[0]; linesOut[lo++] = bPt[1]; linesOut[lo++] = bPt[2];
    linesOut[lo++] = 0; linesOut[lo++] = 0; linesOut[lo++] = 0;
    linesOut[lo++] = col[0]; linesOut[lo++] = col[1]; linesOut[lo++] = col[2]; linesOut[lo++] = col[3];
  };

  const frameCol: [number, number, number, number] = [0.56, 0.70, 0.84, 0.34];
  const horizonCol: [number, number, number, number] = [0.68, 0.80, 0.90, 0.48];

  // Vertical glass corners
  pushLine([x0, y0, z0], [x0, y1, z0], frameCol);
  pushLine([x1, y0, z0], [x1, y1, z0], frameCol);
  pushLine([x1, y0, z1], [x1, y1, z1], frameCol);
  pushLine([x0, y0, z1], [x0, y1, z1], frameCol);
  // Top rim
  pushLine([x0, y1, z0], [x1, y1, z0], frameCol);
  pushLine([x1, y1, z0], [x1, y1, z1], frameCol);
  pushLine([x1, y1, z1], [x0, y1, z1], frameCol);
  pushLine([x0, y1, z1], [x0, y1, z0], frameCol);
  // Soil horizon y = 0 rim
  pushLine([x0, 0, z0], [x1, 0, z0], horizonCol);
  pushLine([x1, 0, z0], [x1, 0, z1], horizonCol);
  pushLine([x1, 0, z1], [x0, 0, z1], horizonCol);
  pushLine([x0, 0, z1], [x0, 0, z0], horizonCol);
  // Bottom rim
  pushLine([x0, y0, z0], [x1, y0, z0], frameCol);
  pushLine([x1, y0, z0], [x1, y0, z1], frameCol);
  pushLine([x1, y0, z1], [x0, y0, z1], frameCol);
  pushLine([x0, y0, z1], [x0, y0, z0], frameCol);

  // Razor slice optical frame
  if (view.sliceEnabled) {
    const sc = computeSliceWorldCoord(b, view);
    const sliceCol: [number, number, number, number] = [0.28, 0.78, 0.98, 0.82];
    if (view.sliceAxis === 'x') {
      pushLine([sc, y0, z0], [sc, y1, z0], sliceCol);
      pushLine([sc, y0, z1], [sc, y1, z1], sliceCol);
      pushLine([sc, y1, z0], [sc, y1, z1], sliceCol);
      pushLine([sc, y0, z0], [sc, y0, z1], sliceCol);
    } else {
      pushLine([x0, y0, sc], [x0, y1, sc], sliceCol);
      pushLine([x1, y0, sc], [x1, y1, sc], sliceCol);
      pushLine([x0, y1, sc], [x1, y1, sc], sliceCol);
      pushLine([x0, y0, sc], [x1, y0, sc], sliceCol);
    }
  }

  return { glassVerts: go / 10, lineVerts: lo / 10 };
}

function buildWindStreamlines(
  snapshot: SimulationSnapshot,
  out: Float32Array,
): number {
  const air = snapshot.airField;
  if (!air || air.nx < 2 || air.ny < 2 || air.nz < 2) return 0;
  const { minX, maxX, minY, maxY, minZ, maxZ } = air.bounds;
  const spanX = Math.max(1e-4, maxX - minX);
  const spanY = Math.max(1e-4, maxY - minY);
  const spanZ = Math.max(1e-4, maxZ - minZ);

  const sampleAir = (px: number, py: number, pz: number): [number, number, number, number] => {
    const fx = Math.max(0, Math.min(air.nx - 1.001, ((px - minX) / spanX) * (air.nx - 1)));
    const fy = Math.max(0, Math.min(air.ny - 1.001, ((py - minY) / spanY) * (air.ny - 1)));
    const fz = Math.max(0, Math.min(air.nz - 1.001, ((pz - minZ) / spanZ) * (air.nz - 1)));
    const ix = Math.floor(fx);
    const iy = Math.floor(fy);
    const iz = Math.floor(fz);
    const tx = fx - ix;
    const ty = fy - iy;
    const tz = fz - iz;
    const idx = (x: number, y: number, z: number) => (y * air.nz + z) * air.nx + x;
    const i000 = idx(ix, iy, iz);
    const i100 = idx(ix + 1, iy, iz);
    const i010 = idx(ix, iy + 1, iz);
    const i110 = idx(ix + 1, iy + 1, iz);
    const i001 = idx(ix, iy, iz + 1);
    const i101 = idx(ix + 1, iy, iz + 1);
    const i011 = idx(ix, iy + 1, iz + 1);
    const i111 = idx(ix + 1, iy + 1, iz + 1);

    const tri = (arr: Float32Array) => {
      const c00 = arr[i000] * (1 - tx) + arr[i100] * tx;
      const c10 = arr[i010] * (1 - tx) + arr[i110] * tx;
      const c01 = arr[i001] * (1 - tx) + arr[i101] * tx;
      const c11 = arr[i011] * (1 - tx) + arr[i111] * tx;
      const c0 = c00 * (1 - ty) + c10 * ty;
      const c1 = c01 * (1 - ty) + c11 * ty;
      return c0 * (1 - tz) + c1 * tz;
    };
    return [tri(air.ux), tri(air.uy), tri(air.uz), tri(air.deltaT)];
  };

  // Seed points around mature caps + ambient convective grid
  const seeds: Vec3[] = [];
  for (let i = 0; i < snapshot.bodies.length && seeds.length < 96; i++) {
    const body = snapshot.bodies[i];
    if (body.capRadiusM < 0.005) continue;
    const rRing = Math.max(0.015, body.capRadiusM * 1.1);
    for (let k = 0; k < 12; k++) {
      const ang = (k / 12) * Math.PI * 2;
      seeds.push([
        body.stipeTipX + Math.cos(ang) * rRing * 0.65,
        Math.max(minY + 0.012, body.stipeTipY - 0.008),
        body.stipeTipZ + Math.sin(ang) * rRing * 0.65,
      ]);
      seeds.push([
        body.stipeTipX + Math.cos(ang) * rRing * 1.65,
        Math.max(minY + 0.015, body.stipeTipY * 0.45),
        body.stipeTipZ + Math.sin(ang) * rRing * 1.65,
      ]);
    }
  }
  // Background grid seeds so convective field is always visible
  for (let gx = 0; gx < 5; gx++) {
    for (let gz = 0; gz < 5; gz++) {
      seeds.push([
        minX + ((gx + 0.5) / 5) * spanX,
        minY + 0.045,
        minZ + ((gz + 0.5) / 5) * spanZ,
      ]);
    }
  }

  let o = 0;
  const maxFloats = out.length - 14;
  const stepDist = 0.011;

  for (let s = 0; s < seeds.length; s++) {
    let [px, py, pz] = seeds[s];
    for (let step = 0; step < 12; step++) {
      if (o >= maxFloats) break;
      const [ux, uy, uz, dT] = sampleAir(px, py, pz);
      const speed = Math.hypot(ux, uy, uz);
      if (speed < 1e-4) break;
      const invS = 1 / speed;
      const nx = px + ux * invS * stepDist;
      const ny = py + uy * invS * stepDist;
      const nz = pz + uz * invS * stepDist;
      if (nx < minX || nx > maxX || ny < minY || ny > maxY || nz < minZ || nz > maxZ) break;

      // Coolwarm color: negative deltaT / sinking = cool cyan-blue; rising = warm amber-orange
      const tWarm = Math.max(0, Math.min(1, 0.5 + (uy * invS) * 0.45 + dT * 0.4));
      let r: number, g: number, b: number;
      if (tWarm < 0.5) {
        const k = tWarm * 2;
        r = 0.18 + (0.76 - 0.18) * k;
        g = 0.62 + (0.85 - 0.62) * k;
        b = 0.98 + (0.92 - 0.98) * k;
      } else {
        const k = (tWarm - 0.5) * 2;
        r = 0.76 + (0.98 - 0.76) * k;
        g = 0.85 + (0.48 - 0.85) * k;
        b = 0.92 + (0.20 - 0.92) * k;
      }
      const fade = (1 - step / 12) * Math.min(1, speed * 45);
      const alpha = Math.max(0.08, Math.min(0.85, fade * 0.82));

      out[o++] = px; out[o++] = py; out[o++] = pz;
      out[o++] = r; out[o++] = g; out[o++] = b; out[o++] = alpha;
      out[o++] = nx; out[o++] = ny; out[o++] = nz;
      out[o++] = r; out[o++] = g; out[o++] = b; out[o++] = alpha * 0.65;

      px = nx; py = ny; pz = nz;
    }
  }
  return o / 7;
}

// ============================================================================
// Soil 3D Texture & Spore Print 2D Texture Packing (Zero Per-Frame Allocation)
// ============================================================================

function packSoilTexture3D(
  snapshot: SimulationSnapshot,
  view: ViewState,
  outBytes: Uint8Array,
): void {
  const soil = snapshot.soil;
  const total = soil.nx * soil.ny * soil.nz;
  const matArr = soil.material;
  const thetaArr = soil.theta;
  const psiArr = soil.psi;
  const o2Arr = soil.o2;
  const ligArr = soil.lignin;
  const csArr = soil.solubleCarbon;
  const enzArr = soil.enzyme;
  const xray = view.xrayMode;

  for (let i = 0; i < total; i++) {
    const m = matArr[i];
    const th = thetaArr[i];
    const tr = MAT_THETA_R[m] ?? 0.05;
    const ts = MAT_THETA_S[m] ?? 0.45;
    const se = m === SoilMaterialId.STONE
      ? 0
      : Math.max(0, Math.min(1, (th - tr) / Math.max(1e-4, ts - tr)));

    let xNorm = 0;
    if (xray === 'matric_psi') {
      // Log-mapped suction: -100 Pa (wet -> 1.0) to -2.0 MPa (dry -> 0.0)
      const absPsi = Math.max(100, Math.abs(psiArr[i]));
      const logV = (Math.log10(absPsi) - 2.0) / 4.3; // 2 -> 100 Pa, 6.3 -> 2 MPa
      xNorm = 1.0 - Math.max(0, Math.min(1, logV));
    } else if (xray === 'carbon_enzyme') {
      const cScore = csArr[i] * 0.25 + ligArr[i] * 0.0045 + enzArr[i] * 4.0;
      xNorm = Math.max(0, Math.min(1, Math.log1p(cScore * 3.0) / Math.log1p(5.0)));
    } else if (xray === 'turgor_p') {
      // Show combined hydraulic + oxygen turgor-support potential in soil
      const o2f = Math.max(0, Math.min(1, o2Arr[i] / AMBIENT_O2_MOL_M3));
      const absPsi = Math.max(100, Math.abs(psiArr[i]));
      const psiSupport = 1.0 - Math.max(0, Math.min(1, (Math.log10(absPsi) - 2.5) / 3.8));
      xNorm = m === SoilMaterialId.STONE ? 0 : psiSupport * o2f;
    }

    const o2Norm = Math.max(0, Math.min(1, o2Arr[i] / AMBIENT_O2_MOL_M3));
    const o4 = i * 4;
    outBytes[o4] = m;
    outBytes[o4 + 1] = Math.round(se * 255);
    outBytes[o4 + 2] = Math.round(xNorm * 255);
    outBytes[o4 + 3] = Math.round(o2Norm * 255);
  }
}

function packSporePrintTexture2D(
  snapshot: SimulationSnapshot,
  outBytes: Uint8Array,
): void {
  const sp = snapshot.sporePrint;
  const total = sp.nx * sp.nz;
  const d = sp.density;
  for (let i = 0; i < total; i++) {
    const val = Math.log1p(d[i] * 0.45) / Math.log1p(8.0);
    outBytes[i] = Math.max(0, Math.min(255, Math.round(val * 255)));
  }
}

// ============================================================================
// WebGPU Backend Implementation
// ============================================================================

class WebGPUBackend {
  readonly device: GPUDevice;
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;
  private depthTex: GPUTexture | null = null;
  private depthTexView: GPUTextureView | null = null;
  private width = 1;
  private height = 1;

  private uniformBuf: GPUBuffer;
  private uniformBufOcc: GPUBuffer;
  private linearSampler: GPUSampler;
  private sdfGi: FastSdfGiSystem;
  private splatPass: ProceduralSplatPass;
  private shadowPass: ShadowMapPass;
  private botanicalPass: BotanicalScatterPass;
  private postPass: PostProcessPass;
  private lastGiSig = '';
  private giTick = 0;

  private soilTex: GPUTexture | null = null;
  private soilTexDims: [number, number, number] = [0, 0, 0];
  private sporePrintTex: GPUTexture | null = null;
  private sporePrintDims: [number, number] = [0, 0];

  private soilBindGroupLayout: GPUBindGroupLayout;
  private soilBindGroup: GPUBindGroup | null = null;
  private litBindGroupLayout: GPUBindGroupLayout;
  private litBindGroup: GPUBindGroup;
  private simpleBindGroupLayout: GPUBindGroupLayout;
  private simpleBindGroup: GPUBindGroup;
  private occBindGroup: GPUBindGroup;

  private backdropPipe: GPURenderPipeline;
  private soilPipe: GPURenderPipeline;
  private mycPipe: GPURenderPipeline;
  private mycOccPipe: GPURenderPipeline;
  private mushPipe: GPURenderPipeline;
  private solidPipe: GPURenderPipeline;
  private glassPipe: GPURenderPipeline;
  private linePipe: GPURenderPipeline;
  private streamPipe: GPURenderPipeline;

  private quadCornerBuf: GPUBuffer;
  private billboardQuadBuf: GPUBuffer;
  private soilGeomBuf: GPUBuffer;
  private glassGeomBuf: GPUBuffer;
  private lineGeomBuf: GPUBuffer;
  private obstacleGeomBuf: GPUBuffer;
  private streamGeomBuf: GPUBuffer;
  private mushVertBuf: GPUBuffer;
  private mushVertCap = 0;
  private mushIdxBuf: GPUBuffer;
  private mushIdxCap = 0;
  private mycSegBuf: GPUBuffer;
  private mycSegCap = 0;

  constructor(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat) {
    this.device = device;
    this.context = context;
    this.format = format;

    this.uniformBuf = device.createBuffer({
      size: 384,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uniformBufOcc = device.createBuffer({
      size: 384,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.linearSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    this.postPass = new PostProcessPass(device, format);
    const hdrFormat = this.postPass.hdrFormat;

    this.sdfGi = new FastSdfGiSystem(device);

    this.simpleBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.simpleBindGroup = device.createBindGroup({
      layout: this.simpleBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });
    this.occBindGroup = device.createBindGroup({
      layout: this.simpleBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBufOcc } }],
    });

    this.shadowPass = new ShadowMapPass(device, this.simpleBindGroupLayout);

    this.soilBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });

    this.litBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });

    this.litBindGroup = device.createBindGroup({
      layout: this.litBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.sdfGi.sdfView },
        { binding: 2, resource: this.linearSampler },
        { binding: 3, resource: this.sdfGi.radViewB },
        { binding: 4, resource: this.shadowPass.shadowView },
        { binding: 5, resource: this.shadowPass.shadowSampler },
      ],
    });

    const alphaBlend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };

    const backMod = device.createShaderModule({ code: BACKDROP_WGSL });
    this.backdropPipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [] }),
      vertex: { module: backMod, entryPoint: 'vs_main' },
      fragment: { module: backMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
      multisample: { count: 4 },
    });

    const soilMod = device.createShaderModule({ code: SOIL_WGSL });
    this.soilPipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.soilBindGroupLayout] }),
      vertex: {
        module: soilMod,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 28,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32' },
          ],
        }],
      },
      fragment: { module: soilMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: 4 },
    });

    const simpleLayout = device.createPipelineLayout({ bindGroupLayouts: [this.simpleBindGroupLayout] });
    const litLayout = device.createPipelineLayout({ bindGroupLayouts: [this.litBindGroupLayout] });

    const mycMod = device.createShaderModule({ code: MYCELIUM_WGSL });
    const mycVertBuffers: GPUVertexBufferLayout[] = [
      {
        arrayStride: 8,
        stepMode: 'vertex',
        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
      },
      {
        arrayStride: 32,
        stepMode: 'instance',
        attributes: [
          { shaderLocation: 1, offset: 0, format: 'float32x3' },
          { shaderLocation: 2, offset: 12, format: 'float32x3' },
          { shaderLocation: 3, offset: 24, format: 'float32x2' },
        ],
      },
    ];
    this.mycPipe = device.createRenderPipeline({
      layout: simpleLayout,
      vertex: { module: mycMod, entryPoint: 'vs_main', buffers: mycVertBuffers },
      fragment: { module: mycMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat, blend: alphaBlend }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less-equal' },
      multisample: { count: 4 },
    });
    this.mycOccPipe = device.createRenderPipeline({
      layout: simpleLayout,
      vertex: { module: mycMod, entryPoint: 'vs_main', buffers: mycVertBuffers },
      fragment: { module: mycMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat, blend: alphaBlend }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'greater' },
      multisample: { count: 4 },
    });

    const mushMod = device.createShaderModule({ code: MUSHROOM_WGSL });
    this.mushPipe = device.createRenderPipeline({
      layout: litLayout,
      vertex: {
        module: mushMod,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 64,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x3' },
            { shaderLocation: 3, offset: 36, format: 'float32x3' },
            { shaderLocation: 4, offset: 48, format: 'float32x4' },
          ],
        }],
      },
      fragment: { module: mushMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat, blend: alphaBlend }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: 4 },
    });

    const solidMod = device.createShaderModule({ code: SOLID_WGSL });
    const solidBuffers: GPUVertexBufferLayout[] = [{
      arrayStride: 40,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x4' },
      ],
    }];
    this.solidPipe = device.createRenderPipeline({
      layout: litLayout,
      vertex: { module: solidMod, entryPoint: 'vs_main', buffers: solidBuffers },
      fragment: { module: solidMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: 4 },
    });
    this.glassPipe = device.createRenderPipeline({
      layout: litLayout,
      vertex: { module: solidMod, entryPoint: 'vs_main', buffers: solidBuffers },
      fragment: { module: solidMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat, blend: alphaBlend }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
      multisample: { count: 4 },
    });
    this.linePipe = device.createRenderPipeline({
      layout: litLayout,
      vertex: { module: solidMod, entryPoint: 'vs_main', buffers: solidBuffers },
      fragment: { module: solidMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat, blend: alphaBlend }] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: 4 },
    });

    const streamMod = device.createShaderModule({ code: STREAMLINE_WGSL });
    this.streamPipe = device.createRenderPipeline({
      layout: simpleLayout,
      vertex: {
        module: streamMod,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 28,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x4' },
          ],
        }],
      },
      fragment: { module: streamMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat, blend: alphaBlend }] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
      multisample: { count: 4 },
    });

    this.quadCornerBuf = device.createBuffer({ size: QUAD_CORNERS.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.quadCornerBuf, 0, QUAD_CORNERS);

    this.billboardQuadBuf = device.createBuffer({ size: BILLBOARD_QUAD.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.billboardQuadBuf, 0, BILLBOARD_QUAD);

    this.splatPass = new ProceduralSplatPass(
      device,
      hdrFormat,
      this.uniformBuf,
      this.sdfGi.sdfView,
      this.linearSampler,
      this.sdfGi.radViewB,
      this.billboardQuadBuf,
    );
    this.botanicalPass = new BotanicalScatterPass(device, hdrFormat, this.litBindGroupLayout);

    this.soilGeomBuf = device.createBuffer({ size: 12000 * 28, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.glassGeomBuf = device.createBuffer({ size: 1024 * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.lineGeomBuf = device.createBuffer({ size: 2048 * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.obstacleGeomBuf = device.createBuffer({ size: 24000 * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.streamGeomBuf = device.createBuffer({ size: 32000 * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });

    this.mushVertCap = 32768;
    this.mushVertBuf = device.createBuffer({ size: this.mushVertCap * 64, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.mushIdxCap = 131072;
    this.mushIdxBuf = device.createBuffer({ size: this.mushIdxCap * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    this.mycSegCap = 4096;
    this.mycSegBuf = device.createBuffer({ size: this.mycSegCap * 32, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  }

  resize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.width && h === this.height && this.depthTex && this.depthTexView) {
      this.postPass.ensureTargets(w, h, this.depthTexView);
      return;
    }
    this.width = w;
    this.height = h;
    this.depthTex?.destroy();
    this.depthTex = this.device.createTexture({
      size: [w, h, 1],
      sampleCount: 4,
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthTexView = this.depthTex.createView();
    this.postPass.ensureTargets(w, h, this.depthTexView);
  }

  private ensureTextures(nx: number, ny: number, nz: number, spNx: number, spNz: number): void {
    let rebuild = false;
    if (!this.soilTex || this.soilTexDims[0] !== nx || this.soilTexDims[1] !== ny || this.soilTexDims[2] !== nz) {
      this.soilTex?.destroy();
      this.soilTex = this.device.createTexture({
        size: [nx, nz, ny],
        dimension: '3d',
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.soilTexDims = [nx, ny, nz];
      rebuild = true;
    }
    if (!this.sporePrintTex || this.sporePrintDims[0] !== spNx || this.sporePrintDims[1] !== spNz) {
      this.sporePrintTex?.destroy();
      this.sporePrintTex = this.device.createTexture({
        size: [Math.max(1, spNx), Math.max(1, spNz), 1],
        dimension: '2d',
        format: 'r8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.sporePrintDims = [spNx, spNz];
      rebuild = true;
    }
    if (rebuild && this.soilTex && this.sporePrintTex) {
      this.soilBindGroup = this.device.createBindGroup({
        layout: this.soilBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.soilTex.createView({ dimension: '3d' }) },
          { binding: 2, resource: this.linearSampler },
          { binding: 3, resource: this.sporePrintTex.createView({ dimension: '2d' }) },
          { binding: 4, resource: this.sdfGi.sdfView },
          { binding: 5, resource: this.sdfGi.radViewB },
          { binding: 6, resource: this.shadowPass.shadowView },
          { binding: 7, resource: this.shadowPass.shadowSampler },
        ],
      });
    }
  }

  getSdfBounds(): SdfVolumeBounds {
    return this.sdfGi.bounds;
  }

  computeLightViewProj(lightDir: Readonly<Vec3>, outMat: Float32Array): void {
    this.shadowPass.computeLightViewProj(lightDir, outMat);
  }

  renderFrame(
    snapshot: SimulationSnapshot,
    view: ViewState,
    lightDir: [number, number, number],
    sliceCoord: number,
    viewTiltMat: Float32Array,
    projMat: Float32Array,
    uniformMain: Float32Array,
    uniformOcc: Float32Array,
    soilBytes: Uint8Array,
    nx: number,
    ny: number,
    nz: number,
    sporePrintBytes: Uint8Array,
    spNx: number,
    spNz: number,
    soilVertsData: Float32Array,
    soilVertCount: number,
    mushVertData: Float32Array,
    mushVertCount: number,
    mushIndices: Uint32Array,
    mushIndexCount: number,
    mycSegData: Float32Array,
    mycSegCount: number,
    showMycelium: boolean,
    obsVertData: Float32Array,
    obsVertCount: number,
    glassVertData: Float32Array,
    glassVertCount: number,
    lineVertData: Float32Array,
    lineVertCount: number,
    sporePos: Float32Array,
    sporeVel: Float32Array,
    sporeMaxCount: number,
    showSpores: boolean,
    streamVertData: Float32Array,
    streamVertCount: number,
  ): void {
    if (!this.depthTex || !this.depthTexView) return;
    this.ensureTextures(nx, ny, nz, spNx, spNz);

    this.device.queue.writeBuffer(this.uniformBuf, 0, uniformMain.buffer, uniformMain.byteOffset, 384);
    this.device.queue.writeBuffer(this.uniformBufOcc, 0, uniformOcc.buffer, uniformOcc.byteOffset, 384);

    if (this.soilTex) {
      this.device.queue.writeTexture(
        { texture: this.soilTex },
        soilBytes.buffer as ArrayBuffer,
        { offset: soilBytes.byteOffset, bytesPerRow: nx * 4, rowsPerImage: nz },
        { width: nx, height: nz, depthOrArrayLayers: ny },
      );
    }
    if (this.sporePrintTex && spNx > 0 && spNz > 0) {
      this.device.queue.writeTexture(
        { texture: this.sporePrintTex },
        sporePrintBytes.buffer as ArrayBuffer,
        { offset: sporePrintBytes.byteOffset, bytesPerRow: spNx, rowsPerImage: spNz },
        { width: spNx, height: spNz, depthOrArrayLayers: 1 },
      );
    }

    if (soilVertCount > 0) {
      this.device.queue.writeBuffer(this.soilGeomBuf, 0, soilVertsData.buffer, soilVertsData.byteOffset, soilVertCount * 28);
    }
    if (obsVertCount > 0) {
      this.device.queue.writeBuffer(this.obstacleGeomBuf, 0, obsVertData.buffer, obsVertData.byteOffset, obsVertCount * 40);
    }
    if (glassVertCount > 0) {
      this.device.queue.writeBuffer(this.glassGeomBuf, 0, glassVertData.buffer, glassVertData.byteOffset, glassVertCount * 40);
    }
    if (lineVertCount > 0) {
      this.device.queue.writeBuffer(this.lineGeomBuf, 0, lineVertData.buffer, lineVertData.byteOffset, lineVertCount * 40);
    }
    if (streamVertCount > 0) {
      this.device.queue.writeBuffer(this.streamGeomBuf, 0, streamVertData.buffer, streamVertData.byteOffset, streamVertCount * 28);
    }

    if (mushVertCount > 0 && mushIndexCount > 0) {
      if (mushVertCount > this.mushVertCap) {
        this.mushVertCap = Math.max(mushVertCount, this.mushVertCap * 2);
        this.mushVertBuf.destroy();
        this.mushVertBuf = this.device.createBuffer({ size: this.mushVertCap * 64, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      }
      if (mushIndexCount > this.mushIdxCap) {
        this.mushIdxCap = Math.max(mushIndexCount, this.mushIdxCap * 2);
        this.mushIdxBuf.destroy();
        this.mushIdxBuf = this.device.createBuffer({ size: this.mushIdxCap * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      }
      this.device.queue.writeBuffer(this.mushVertBuf, 0, mushVertData.buffer, mushVertData.byteOffset, mushVertCount * 64);
      this.device.queue.writeBuffer(this.mushIdxBuf, 0, mushIndices.buffer, mushIndices.byteOffset, mushIndexCount * 4);
    }

    if (showMycelium && mycSegCount > 0) {
      if (mycSegCount > this.mycSegCap) {
        this.mycSegCap = Math.max(mycSegCount, this.mycSegCap * 2);
        this.mycSegBuf.destroy();
        this.mycSegBuf = this.device.createBuffer({ size: this.mycSegCap * 32, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      }
      this.device.queue.writeBuffer(this.mycSegBuf, 0, mycSegData.buffer, mycSegData.byteOffset, mycSegCount * 32);
    }

    this.botanicalPass.update(snapshot);

    const encoder = this.device.createCommandEncoder();
    // Stage 1 Fast-SDF-AO & 2-Bounce World Radiance Volume GI compute pass (throttled when static)
    this.giTick++;
    const leadB = snapshot.bodies[0];
    const giSigImmediate = `${snapshot.species.id}:${snapshot.bodies.length}:${snapshot.obstacles.length}:${view.sliceEnabled ? 1 : 0}:${view.sliceAxis}:${sliceCoord.toFixed(3)}:${lightDir[0].toFixed(2)},${lightDir[1].toFixed(2)},${lightDir[2].toFixed(2)}:${leadB ? Math.round(leadB.stipeLengthM * 250) : 0}:${leadB ? Math.round(leadB.capRadiusM * 250) : 0}`;
    if (giSigImmediate !== this.lastGiSig || (this.giTick & 3) === 0) {
      this.lastGiSig = giSigImmediate;
      this.sdfGi.update(encoder, snapshot, view, lightDir, sliceCoord);
    }

    // Stage 2 Hybrid Procedural 3DGS compute projection & GPU radix sort
    void sporePos;
    void sporeVel;
    void sporeMaxCount;
    void showSpores;
    this.splatPass.updateAndSort(
      encoder,
      snapshot,
      view,
      viewTiltMat,
      projMat,
      this.width,
      this.height,
    );

    // Stage 3 2048x2048 Directional PCF Shadow Map Pass
    this.shadowPass.render(
      encoder,
      this.simpleBindGroup,
      this.mushVertBuf,
      this.mushIdxBuf,
      mushIndexCount,
      this.obstacleGeomBuf,
      obsVertCount,
      this.soilGeomBuf,
      soilVertCount,
    );

    const swapView = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.postPass.getHdrMsaaView(),
        resolveTarget: this.postPass.getHdrResolveView(),
        clearValue: { r: 0.011, g: 0.015, b: 0.023, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: {
        view: this.depthTexView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    // 0. Subtle studio gradient backdrop
    pass.setPipeline(this.backdropPipe);
    pass.draw(3);

    // 1. Displaced soil heightfield + cutaway + plinth + razor slice
    if (this.soilBindGroup && soilVertCount > 0) {
      pass.setPipeline(this.soilPipe);
      pass.setBindGroup(0, this.soilBindGroup);
      pass.setVertexBuffer(0, this.soilGeomBuf);
      pass.draw(soilVertCount);
    }

    // 2. Instanced micro-botanical forest floor (oak leaves, moss fronds, conifer needles, pebbles)
    if (view.xrayMode === 'none') {
      this.botanicalPass.draw(pass, this.litBindGroup);
    }

    // 3. Solid obstacles (twigs & stones)
    if (obsVertCount > 0) {
      pass.setPipeline(this.solidPipe);
      pass.setBindGroup(0, this.litBindGroup);
      pass.setVertexBuffer(0, this.obstacleGeomBuf);
      pass.draw(obsVertCount);
    }

    // 4. Species Macro PBR + Waxy Chitin SSS Mushrooms
    if (mushVertCount > 0 && mushIndexCount > 0) {
      pass.setPipeline(this.mushPipe);
      pass.setBindGroup(0, this.litBindGroup);
      pass.setVertexBuffer(0, this.mushVertBuf);
      pass.setIndexBuffer(this.mushIdxBuf, 'uint32');
      pass.drawIndexed(mushIndexCount);
    }

    // 5. 3D Mycelial Network (exposed pass + subterranean depth-cued pass)
    if (showMycelium && mycSegCount > 0) {
      pass.setPipeline(this.mycPipe);
      pass.setBindGroup(0, this.simpleBindGroup);
      pass.setVertexBuffer(0, this.quadCornerBuf);
      pass.setVertexBuffer(1, this.mycSegBuf);
      pass.draw(6, mycSegCount);

      pass.setPipeline(this.mycOccPipe);
      pass.setBindGroup(0, this.occBindGroup);
      pass.setVertexBuffer(0, this.quadCornerBuf);
      pass.setVertexBuffer(1, this.mycSegBuf);
      pass.draw(6, mycSegCount);
    }

    // 6. Evaporative convective wind streamlines
    if (streamVertCount > 0) {
      pass.setPipeline(this.streamPipe);
      pass.setBindGroup(0, this.simpleBindGroup);
      pass.setVertexBuffer(0, this.streamGeomBuf);
      pass.draw(streamVertCount);
    }

    // 7. Sorted Hybrid Procedural 3D Gaussian Splats (moss/litter, mycelial halo, cap fuzz, spore mist)
    this.splatPass.draw(pass);

    // 8. Physical museum glass walls w/ RH condensation & optical wireframe hairlines
    if (glassVertCount > 0) {
      pass.setPipeline(this.glassPipe);
      pass.setBindGroup(0, this.litBindGroup);
      pass.setVertexBuffer(0, this.glassGeomBuf);
      pass.draw(glassVertCount);
    }
    if (lineVertCount > 0) {
      pass.setPipeline(this.linePipe);
      pass.setBindGroup(0, this.litBindGroup);
      pass.setVertexBuffer(0, this.lineGeomBuf);
      pass.draw(lineVertCount);
    }

    pass.end();

    // Stage 4 Half-Res HDR Bloom + Macro Bokeh DoF + ACES Filmic Composite -> Swapchain
    this.postPass.renderPost(
      encoder,
      swapView,
      uniformMain[84],
      uniformMain[85],
      uniformMain[86],
      uniformMain[31],
      view.sliceEnabled,
    );

    this.device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    this.postPass.dispose();
    this.botanicalPass.dispose();
    this.shadowPass.dispose();
    this.splatPass.dispose();
    this.sdfGi.dispose();
    this.depthTex?.destroy();
    this.soilTex?.destroy();
    this.sporePrintTex?.destroy();
    this.uniformBuf.destroy();
    this.uniformBufOcc.destroy();
    this.quadCornerBuf.destroy();
    this.billboardQuadBuf.destroy();
    this.soilGeomBuf.destroy();
    this.glassGeomBuf.destroy();
    this.lineGeomBuf.destroy();
    this.obstacleGeomBuf.destroy();
    this.streamGeomBuf.destroy();
    this.mushVertBuf.destroy();
    this.mushIdxBuf.destroy();
    this.mycSegBuf.destroy();
  }
}

// ============================================================================
// WebGL2 Fallback Backend (used ONLY when WebGPU adapter is unavailable)
// ============================================================================

interface GLProgramInfo {
  prog: WebGLProgram;
  loc: Record<string, WebGLUniformLocation | null>;
}

function compileGLShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type);
  if (!s) throw new Error('Failed to create WebGL2 shader');
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s) ?? 'unknown';
    gl.deleteShader(s);
    throw new Error(`WebGL2 shader compile error: ${log}`);
  }
  return s;
}

function createGLProgram(gl: WebGL2RenderingContext, vsSrc: string, fsSrc: string): GLProgramInfo {
  const vs = compileGLShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileGLShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  if (!prog) throw new Error('Failed to create WebGL2 program');
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog) ?? 'unknown';
    gl.deleteProgram(prog);
    throw new Error(`WebGL2 program link error: ${log}`);
  }
  const names = [
    'uViewProj', 'uCameraPos', 'uLightDir', 'uSoilBoundsMin', 'uSoilBoundsMax',
    'uSliceParams', 'uSssColor', 'uSporePrintColor', 'uSoilDims', 'uSoilTex', 'uSporePrintTex',
  ];
  const loc: Record<string, WebGLUniformLocation | null> = {};
  for (const name of names) {
    loc[name] = gl.getUniformLocation(prog, name);
  }
  return { prog, loc };
}

class WebGL2Backend {
  private gl: WebGL2RenderingContext;
  private soilProg: GLProgramInfo;
  private mycProg: GLProgramInfo;
  private mushProg: GLProgramInfo;
  private solidProg: GLProgramInfo;
  private sporeProg: GLProgramInfo;
  private streamProg: GLProgramInfo;

  private soilTex: WebGLTexture;
  private sporePrintTex: WebGLTexture;
  private soilDims: [number, number, number] = [0, 0, 0];
  private spDims: [number, number] = [0, 0];

  private soilVao: WebGLVertexArrayObject;
  private soilVbo: WebGLBuffer;
  private mycVao: WebGLVertexArrayObject;
  private mycQuadVbo: WebGLBuffer;
  private mycSegVbo: WebGLBuffer;
  private mushVao: WebGLVertexArrayObject;
  private mushVbo: WebGLBuffer;
  private mushIbo: WebGLBuffer;
  private solidVao: WebGLVertexArrayObject;
  private solidVbo: WebGLBuffer;
  private sporeVao: WebGLVertexArrayObject;
  private sporeQuadVbo: WebGLBuffer;
  private sporePosVbo: WebGLBuffer;
  private sporeVelVbo: WebGLBuffer;
  private streamVao: WebGLVertexArrayObject;
  private streamVbo: WebGLBuffer;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.soilProg = createGLProgram(gl, SOIL_GLSL_VS, SOIL_GLSL_FS);
    this.mycProg = createGLProgram(gl, MYCELIUM_GLSL_VS, MYCELIUM_GLSL_FS);
    this.mushProg = createGLProgram(gl, MUSHROOM_GLSL_VS, MUSHROOM_GLSL_FS);
    this.solidProg = createGLProgram(gl, SOLID_GLSL_VS, SOLID_GLSL_FS);
    this.sporeProg = createGLProgram(gl, SPORE_GLSL_VS, SPORE_GLSL_FS);
    this.streamProg = createGLProgram(gl, STREAMLINE_GLSL_VS, STREAMLINE_GLSL_FS);

    this.soilTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_3D, this.soilTex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

    this.sporePrintTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.sporePrintTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Soil VAO
    this.soilVao = gl.createVertexArray()!;
    this.soilVbo = gl.createBuffer()!;
    gl.bindVertexArray(this.soilVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.soilVbo);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 28, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 28, 24);

    // Mycelium VAO
    this.mycVao = gl.createVertexArray()!;
    this.mycQuadVbo = gl.createBuffer()!;
    this.mycSegVbo = gl.createBuffer()!;
    gl.bindVertexArray(this.mycVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mycQuadVbo);
    gl.bufferData(gl.ARRAY_BUFFER, QUAD_CORNERS, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mycSegVbo);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 0); gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 32, 12); gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 32, 24); gl.vertexAttribDivisor(3, 1);

    // Mushroom VAO
    this.mushVao = gl.createVertexArray()!;
    this.mushVbo = gl.createBuffer()!;
    this.mushIbo = gl.createBuffer()!;
    gl.bindVertexArray(this.mushVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mushVbo);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 48, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 48, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 48, 24);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 3, gl.FLOAT, false, 48, 36);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.mushIbo);

    // Solid / Glass / Line VAO
    this.solidVao = gl.createVertexArray()!;
    this.solidVbo = gl.createBuffer()!;
    gl.bindVertexArray(this.solidVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.solidVbo);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 40, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 40, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 40, 24);

    // Spore VAO
    this.sporeVao = gl.createVertexArray()!;
    this.sporeQuadVbo = gl.createBuffer()!;
    this.sporePosVbo = gl.createBuffer()!;
    this.sporeVelVbo = gl.createBuffer()!;
    gl.bindVertexArray(this.sporeVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sporeQuadVbo);
    gl.bufferData(gl.ARRAY_BUFFER, BILLBOARD_QUAD, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sporePosVbo);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 16, 0); gl.vertexAttribDivisor(1, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sporeVelVbo);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 16, 0); gl.vertexAttribDivisor(2, 1);

    // Streamline VAO
    this.streamVao = gl.createVertexArray()!;
    this.streamVbo = gl.createBuffer()!;
    gl.bindVertexArray(this.streamVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.streamVbo);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);

    gl.bindVertexArray(null);
  }

  private bindCommonUniforms(info: GLProgramInfo, u: Float32Array): void {
    const gl = this.gl;
    gl.useProgram(info.prog);
    if (info.loc.uViewProj) gl.uniformMatrix4fv(info.loc.uViewProj, false, u.subarray(0, 16));
    if (info.loc.uCameraPos) gl.uniform4fv(info.loc.uCameraPos, u.subarray(16, 20));
    if (info.loc.uLightDir) gl.uniform4fv(info.loc.uLightDir, u.subarray(20, 24));
    if (info.loc.uSoilBoundsMin) gl.uniform4fv(info.loc.uSoilBoundsMin, u.subarray(24, 28));
    if (info.loc.uSoilBoundsMax) gl.uniform4fv(info.loc.uSoilBoundsMax, u.subarray(28, 32));
    if (info.loc.uSliceParams) gl.uniform4fv(info.loc.uSliceParams, u.subarray(32, 36));
    if (info.loc.uSssColor) gl.uniform4fv(info.loc.uSssColor, u.subarray(36, 40));
    if (info.loc.uSporePrintColor) gl.uniform4fv(info.loc.uSporePrintColor, u.subarray(40, 44));
    if (info.loc.uSoilDims) gl.uniform4fv(info.loc.uSoilDims, u.subarray(44, 48));
  }

  resize(width: number, height: number): void {
    this.gl.viewport(0, 0, Math.max(1, Math.floor(width)), Math.max(1, Math.floor(height)));
  }

  renderFrame(
    uniformMain: Float32Array,
    uniformOcc: Float32Array,
    soilBytes: Uint8Array,
    nx: number,
    ny: number,
    nz: number,
    sporePrintBytes: Uint8Array,
    spNx: number,
    spNz: number,
    soilVertsData: Float32Array,
    soilVertCount: number,
    mushVertData: Float32Array,
    mushVertCount: number,
    mushIndices: Uint32Array,
    mushIndexCount: number,
    mycSegData: Float32Array,
    mycSegCount: number,
    showMycelium: boolean,
    obsVertData: Float32Array,
    obsVertCount: number,
    glassVertData: Float32Array,
    glassVertCount: number,
    lineVertData: Float32Array,
    lineVertCount: number,
    sporePos: Float32Array,
    sporeVel: Float32Array,
    sporeMaxCount: number,
    showSpores: boolean,
    streamVertData: Float32Array,
    streamVertCount: number,
  ): void {
    const gl = this.gl;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.clearColor(0.047, 0.063, 0.082, 1.0);
    gl.clearDepth(1.0);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Upload 3D soil texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.soilTex);
    if (this.soilDims[0] !== nx || this.soilDims[1] !== ny || this.soilDims[2] !== nz) {
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, nx, nz, ny, 0, gl.RGBA, gl.UNSIGNED_BYTE, soilBytes);
      this.soilDims = [nx, ny, nz];
    } else {
      gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, nx, nz, ny, gl.RGBA, gl.UNSIGNED_BYTE, soilBytes);
    }

    // Upload 2D spore print texture
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.sporePrintTex);
    if (spNx > 0 && spNz > 0) {
      if (this.spDims[0] !== spNx || this.spDims[1] !== spNz) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, spNx, spNz, 0, gl.RED, gl.UNSIGNED_BYTE, sporePrintBytes);
        this.spDims = [spNx, spNz];
      } else {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, spNx, spNz, gl.RED, gl.UNSIGNED_BYTE, sporePrintBytes);
      }
    }

    // 1. Soil
    if (soilVertCount > 0) {
      this.bindCommonUniforms(this.soilProg, uniformMain);
      if (this.soilProg.loc.uSoilTex) gl.uniform1i(this.soilProg.loc.uSoilTex, 0);
      if (this.soilProg.loc.uSporePrintTex) gl.uniform1i(this.soilProg.loc.uSporePrintTex, 1);
      gl.bindVertexArray(this.soilVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.soilVbo);
      gl.bufferData(gl.ARRAY_BUFFER, soilVertsData.subarray(0, soilVertCount * 7), gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.TRIANGLES, 0, soilVertCount);
    }

    // 2. Obstacles
    if (obsVertCount > 0) {
      this.bindCommonUniforms(this.solidProg, uniformMain);
      gl.bindVertexArray(this.solidVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.solidVbo);
      gl.bufferData(gl.ARRAY_BUFFER, obsVertData.subarray(0, obsVertCount * 10), gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.TRIANGLES, 0, obsVertCount);
    }

    // 3. Mushrooms
    if (mushVertCount > 0 && mushIndexCount > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      this.bindCommonUniforms(this.mushProg, uniformMain);
      gl.bindVertexArray(this.mushVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.mushVbo);
      gl.bufferData(gl.ARRAY_BUFFER, mushVertData.subarray(0, mushVertCount * 12), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.mushIbo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mushIndices.subarray(0, mushIndexCount), gl.DYNAMIC_DRAW);
      gl.drawElements(gl.TRIANGLES, mushIndexCount, gl.UNSIGNED_INT, 0);
    }

    // 4. Mycelium (exposed + subterranean glow)
    if (showMycelium && mycSegCount > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(this.mycVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.mycSegVbo);
      gl.bufferData(gl.ARRAY_BUFFER, mycSegData.subarray(0, mycSegCount * 8), gl.DYNAMIC_DRAW);

      this.bindCommonUniforms(this.mycProg, uniformMain);
      gl.depthFunc(gl.LEQUAL);
      gl.depthMask(true);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, mycSegCount);

      this.bindCommonUniforms(this.mycProg, uniformOcc);
      gl.depthFunc(gl.GREATER);
      gl.depthMask(false);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, mycSegCount);
      gl.depthFunc(gl.LESS);
    }

    // 5. Streamlines
    if (streamVertCount > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      this.bindCommonUniforms(this.streamProg, uniformMain);
      gl.bindVertexArray(this.streamVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.streamVbo);
      gl.bufferData(gl.ARRAY_BUFFER, streamVertData.subarray(0, streamVertCount * 7), gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.LINES, 0, streamVertCount);
    }

    // 6. Airborne Spores
    if (showSpores && sporeMaxCount > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      this.bindCommonUniforms(this.sporeProg, uniformMain);
      gl.bindVertexArray(this.sporeVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.sporePosVbo);
      gl.bufferData(gl.ARRAY_BUFFER, sporePos.subarray(0, sporeMaxCount * 4), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.sporeVelVbo);
      gl.bufferData(gl.ARRAY_BUFFER, sporeVel.subarray(0, sporeMaxCount * 4), gl.DYNAMIC_DRAW);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, sporeMaxCount);
    }

    // 7. Glass & Wireframe lines
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    this.bindCommonUniforms(this.solidProg, uniformMain);
    gl.bindVertexArray(this.solidVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.solidVbo);
    if (glassVertCount > 0) {
      gl.bufferData(gl.ARRAY_BUFFER, glassVertData.subarray(0, glassVertCount * 10), gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.TRIANGLES, 0, glassVertCount);
    }
    if (lineVertCount > 0) {
      gl.depthFunc(gl.LEQUAL);
      gl.bufferData(gl.ARRAY_BUFFER, lineVertData.subarray(0, lineVertCount * 10), gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.LINES, 0, lineVertCount);
      gl.depthFunc(gl.LESS);
    }
    gl.depthMask(true);
    gl.bindVertexArray(null);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteTexture(this.soilTex);
    gl.deleteTexture(this.sporePrintTex);
    gl.deleteBuffer(this.soilVbo);
    gl.deleteBuffer(this.mycQuadVbo);
    gl.deleteBuffer(this.mycSegVbo);
    gl.deleteBuffer(this.mushVbo);
    gl.deleteBuffer(this.mushIbo);
    gl.deleteBuffer(this.solidVbo);
    gl.deleteBuffer(this.sporeQuadVbo);
    gl.deleteBuffer(this.sporePosVbo);
    gl.deleteBuffer(this.sporeVelVbo);
    gl.deleteBuffer(this.streamVbo);
    gl.deleteVertexArray(this.soilVao);
    gl.deleteVertexArray(this.mycVao);
    gl.deleteVertexArray(this.mushVao);
    gl.deleteVertexArray(this.solidVao);
    gl.deleteVertexArray(this.sporeVao);
    gl.deleteVertexArray(this.streamVao);
  }
}

// ============================================================================
// Public Exported Class: TerrariumRenderer
// ============================================================================

export class TerrariumRenderer {
  readonly backend: RendererBackend;
  private canvas: HTMLCanvasElement;
  private gpuBackend: WebGPUBackend | null;
  private glBackend: WebGL2Backend | null;

  private cssWidth = 800;
  private cssHeight = 600;
  private pixelWidth = 800;
  private pixelHeight = 600;

  // Reusable zero-GC scratch arrays
  private projMat = new Float32Array(16);
  private viewMat = new Float32Array(16);
  private tiltMat = new Float32Array(16);
  private viewTiltMat = new Float32Array(16);
  private vpMat = new Float32Array(16);
  private mvpMat = new Float32Array(16);
  private invMvpMat = new Float32Array(16);
  private lightViewProjMat = new Float32Array(16);

  private uniformMain = new Float32Array(96);
  private uniformOcc = new Float32Array(96);

  private soilTexScratch = new Uint8Array(48 * 32 * 48 * 4);
  private sporeTexScratch = new Uint8Array(64 * 64);
  private soilVertsScratch = new Float32Array(12000 * 7);
  private glassVertsScratch = new Float32Array(1024);
  private lineVertsScratch = new Float32Array(2048);
  private obsVertsScratch = new Float32Array(24000);
  private streamVertsScratch = new Float32Array(32000);
  private mushInterleavedScratch = new Float32Array(16384 * 12);
  private mushSubdivState: MushSubdivState = {
    verts: new Float32Array(32768 * 16),
    indices: new Uint32Array(131072),
    vertCount: 0,
    indexCount: 0,
  };
  private cachedSoilSig = '';
  private cachedSoilVertCount = 0;
  private sporePosScratch = new Float32Array(2048 * 4);
  private sporeVelScratch = new Float32Array(2048 * 4);

  private lastSnapshot: SimulationSnapshot | null = null;
  private lastView: ViewState | null = null;

  private constructor(
    canvas: HTMLCanvasElement,
    backend: RendererBackend,
    gpuBackend: WebGPUBackend | null,
    glBackend: WebGL2Backend | null,
  ) {
    this.canvas = canvas;
    this.backend = backend;
    this.gpuBackend = gpuBackend;
    this.glBackend = glBackend;
    this.resize(canvas.clientWidth || 800, canvas.clientHeight || 600);
  }

  static async create(canvas: HTMLCanvasElement): Promise<TerrariumRenderer> {
    // 1. Primary path: WebGPU
    if (typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (adapter) {
          const device = await adapter.requestDevice();
          device.addEventListener('uncapturederror', (ev) => {
            const err = (ev as GPUUncapturedErrorEvent).error;
            console.error('WebGPU uncaptured error:', err?.message ?? ev);
          });
          const ctx = canvas.getContext('webgpu');
          if (ctx) {
            const format = navigator.gpu.getPreferredCanvasFormat();
            ctx.configure({
              device,
              format,
              alphaMode: 'opaque',
            });
            const gpuBackend = new WebGPUBackend(device, ctx, format);
            return new TerrariumRenderer(canvas, 'webgpu', gpuBackend, null);
          }
        }
      } catch (err) {
        console.warn('[SPOROCARP] WebGPU initialization failed, attempting WebGL2 fallback:', err);
      }
    }

    // 2. Graceful fallback: WebGL2 (only if WebGPU unavailable)
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      depth: true,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) {
      throw new Error('SPOROCARP requires WebGPU or WebGL2 support.');
    }
    const glBackend = new WebGL2Backend(gl);
    return new TerrariumRenderer(canvas, 'webgl2', null, glBackend);
  }

  resize(width: number, height: number): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssWidth = Math.max(1, width);
    this.cssHeight = Math.max(1, height);
    this.pixelWidth = Math.max(1, Math.floor(this.cssWidth * dpr));
    this.pixelHeight = Math.max(1, Math.floor(this.cssHeight * dpr));
    if (this.canvas.width !== this.pixelWidth || this.canvas.height !== this.pixelHeight) {
      this.canvas.width = this.pixelWidth;
      this.canvas.height = this.pixelHeight;
    }
    this.gpuBackend?.resize(this.pixelWidth, this.pixelHeight);
    this.glBackend?.resize(this.pixelWidth, this.pixelHeight);
  }

  private computeCameraMatrices(snapshot: SimulationSnapshot, view: ViewState): Vec3 {
    const aspect = this.pixelWidth / Math.max(1, this.pixelHeight);
    const zeroToOne = this.backend === 'webgpu';
    mat4Perspective((38 * Math.PI) / 180, aspect, 0.02, 8.0, zeroToOne, this.projMat);
    const eye = computeOrbitCameraEye(
      view.cameraYaw,
      view.cameraPitch,
      view.cameraDistance,
      view.cameraTarget,
    );
    mat4LookAt(eye, view.cameraTarget, [0, 1, 0], this.viewMat);
    mat4Multiply(this.projMat, this.viewMat, this.vpMat);

    const visualTilt = computeVisualTiltRad(snapshot);
    if (Math.abs(visualTilt) > 1e-5) {
      mat4RotationZ(visualTilt, this.tiltMat);
      mat4Multiply(this.vpMat, this.tiltMat, this.mvpMat);
      mat4Multiply(this.viewMat, this.tiltMat, this.viewTiltMat);
    } else {
      mat4Identity(this.tiltMat);
      this.mvpMat.set(this.vpMat);
      this.viewTiltMat.set(this.viewMat);
    }
    return eye;
  }

  render(snapshot: SimulationSnapshot, view: ViewState): void {
    this.lastSnapshot = snapshot;
    this.lastView = view;

    const bounds = getSoilBounds(snapshot);
    const eye = this.computeCameraMatrices(snapshot, view);

    // Transform cameraPos into local (un-tilted) soil space so Fresnel/billboards match tilted mvpMat
    const visualTilt = computeVisualTiltRad(snapshot);
    const cT = Math.cos(-visualTilt);
    const sT = Math.sin(-visualTilt);
    const localEyeX = eye[0] * cT - eye[1] * sT;
    const localEyeY = eye[0] * sT + eye[1] * cT;
    const localEyeZ = eye[2];

    const sliceCoord = computeSliceWorldCoord(bounds, view);
    const lDir = vec3Normalize(snapshot.env.lightDir);
    const xrayIdx = XRAY_MODE_INDEX[view.xrayMode] ?? 0;

    // Pack main uniforms (256 bytes = 64 floats)
    const u = this.uniformMain;
    u.set(this.mvpMat, 0);
    u[16] = localEyeX; u[17] = localEyeY; u[18] = localEyeZ; u[19] = snapshot.simTimeSec;
    u[20] = lDir[0]; u[21] = lDir[1]; u[22] = lDir[2]; u[23] = snapshot.env.ambientTempC;
    u[24] = bounds.minX; u[25] = bounds.minY; u[26] = bounds.minZ; u[27] = bounds.dx;
    u[28] = bounds.maxX; u[29] = bounds.maxY; u[30] = bounds.maxZ; u[31] = xrayIdx;
    u[32] = view.sliceEnabled ? 1.0 : 0.0;
    u[33] = view.sliceAxis === 'x' ? 0.0 : 1.0;
    u[34] = sliceCoord;
    u[35] = 0.96; // primary mycelium alpha
    u[36] = snapshot.species.sssColor[0];
    u[37] = snapshot.species.sssColor[1];
    u[38] = snapshot.species.sssColor[2];
    u[39] = view.showMycelium ? 1.0 : 0.0;
    u[40] = snapshot.sporePrint.sporeColor[0];
    u[41] = snapshot.sporePrint.sporeColor[1];
    u[42] = snapshot.sporePrint.sporeColor[2];
    u[43] = view.selectedBodyId;
    u[44] = bounds.nx;
    u[45] = bounds.ny;
    u[46] = bounds.nz;
    u[47] = snapshot.env.ambientHumidity;

    const sdfBounds = this.gpuBackend?.getSdfBounds() ?? {
      minX: bounds.minX - 0.02,
      minY: -0.05,
      minZ: bounds.minZ - 0.02,
      maxX: bounds.maxX + 0.02,
      maxY: 0.27,
      maxZ: bounds.maxZ + 0.02,
    };
    const sdfVoxelSize = (sdfBounds.maxX - sdfBounds.minX) / SDF_NX;
    const rawRx = this.viewMat[0], rawRy = this.viewMat[4], rawRz = this.viewMat[8];
    const rawUx = this.viewMat[1], rawUy = this.viewMat[5], rawUz = this.viewMat[9];
    const localRightX = rawRx * cT - rawRy * sT;
    const localRightY = rawRx * sT + rawRy * cT;
    const localUpX = rawUx * cT - rawUy * sT;
    const localUpY = rawUx * sT + rawUy * cT;

    u[48] = sdfBounds.minX; u[49] = sdfBounds.minY; u[50] = sdfBounds.minZ; u[51] = sdfVoxelSize;
    u[52] = sdfBounds.maxX; u[53] = sdfBounds.maxY; u[54] = sdfBounds.maxZ; u[55] = 1.0;
    u[56] = localRightX; u[57] = localRightY; u[58] = rawRz; u[59] = this.pixelWidth;
    u[60] = localUpX; u[61] = localUpY; u[62] = rawUz; u[63] = this.pixelHeight;

    // Pack extended studio-macro uniforms (floats 64..95 -> bytes 256..384)
    if (this.gpuBackend) {
      this.gpuBackend.computeLightViewProj(lDir, this.lightViewProjMat);
    } else {
      mat4Identity(this.lightViewProjMat);
    }
    u.set(this.lightViewProjMat, 64);
    const speciesIdx =
      snapshot.species.id === 'parasol'
        ? 0
        : snapshot.species.id === 'chanterelle'
          ? 1
          : snapshot.species.id === 'morel'
            ? 2
            : 3;
    u[80] = speciesIdx;
    u[81] = 1.0;
    u[82] = 0.02;
    u[83] = 8.0;
    const focusDistM = Math.hypot(
      eye[0] - view.cameraTarget[0],
      eye[1] - view.cameraTarget[1],
      eye[2] - view.cameraTarget[2],
    );
    u[84] = focusDistM;
    u[85] = 0.28;
    u[86] = 0.14;
    u[87] = 1.0;
    u[88] = 0;
    u[89] = 0;
    u[90] = 0;
    u[91] = 0;
    u[92] = 0;
    u[93] = 0;
    u[94] = 0;
    u[95] = 0;

    // Occluded subterranean mycelium uniform copy (soft translucent window through dark earth)
    this.uniformOcc.set(u);
    this.uniformOcc[35] = view.xrayMode === 'turgor_p' ? 0.62 : 0.34;

    // Pack 3D soil voxel texture
    const soilVoxelBytes = bounds.nx * bounds.ny * bounds.nz * 4;
    if (this.soilTexScratch.length < soilVoxelBytes) {
      this.soilTexScratch = new Uint8Array(soilVoxelBytes);
    }
    packSoilTexture3D(snapshot, view, this.soilTexScratch);

    // Pack 2D spore print texture
    const sp = snapshot.sporePrint;
    const spBytes = Math.max(1, sp.nx * sp.nz);
    if (this.sporeTexScratch.length < spBytes) {
      this.sporeTexScratch = new Uint8Array(spBytes);
    }
    if (sp.nx > 0 && sp.nz > 0) {
      packSporePrintTexture2D(snapshot, this.sporeTexScratch);
    }

    // Build dynamic scene geometry into scratch buffers (cached displaced heightfield terrain)
    let soilSig = `${snapshot.species.id}:${snapshot.obstacles.length}:${bounds.minX.toFixed(4)},${bounds.maxX.toFixed(4)},${bounds.minY.toFixed(4)},${bounds.maxY.toFixed(4)},${bounds.minZ.toFixed(4)},${bounds.maxZ.toFixed(4)}:${view.sliceEnabled ? 1 : 0}:${view.sliceAxis}:${sliceCoord.toFixed(4)}:${snapshot.bodies.length}`;
    for (let i = 0; i < snapshot.bodies.length; i++) {
      const b = snapshot.bodies[i];
      soilSig += `|${b.id},${b.baseX.toFixed(4)},${b.baseZ.toFixed(4)}`;
    }
    if (soilSig !== this.cachedSoilSig) {
      this.cachedSoilSig = soilSig;
      this.cachedSoilVertCount = buildSoilAndSliceMesh(
        this.soilVertsScratch,
        bounds,
        view,
        snapshot.soil,
        snapshot.bodies,
      );
    }
    const soilVertCount = this.cachedSoilVertCount;
    const obsVertCount = buildObstaclesMesh(snapshot.obstacles, this.obsVertsScratch);
    const airTopY = snapshot.airField?.bounds?.maxY ?? 0.30;
    const { glassVerts, lineVerts } = buildGlassArchitecture(
      bounds,
      airTopY,
      view,
      this.glassVertsScratch,
      this.lineVertsScratch,
    );

    const streamVertCount =
      view.showStreamlines || view.xrayMode === 'air_convection'
        ? buildWindStreamlines(snapshot, this.streamVertsScratch)
        : 0;

    const mesh = snapshot.fruitingMesh;
    const mushVertCount = mesh.vertexCount;
    const mushIndexCount = mesh.indexCount;
    if (this.gpuBackend) {
      packSubdividedMushroomMesh(mesh, snapshot.bodies, this.mushSubdivState);
    } else if (mushVertCount > 0 && mushIndexCount > 0) {
      // Pack WebGL2 interleaved vertices (12 floats per vertex)
      const needed = mushVertCount * 12;
      if (this.mushInterleavedScratch.length < needed) {
        this.mushInterleavedScratch = new Float32Array(Math.max(needed, this.mushInterleavedScratch.length * 2));
      }
      const dst = this.mushInterleavedScratch;
      const pos = mesh.positions;
      const nrm = mesh.normals;
      const col = mesh.colors;
      const str = mesh.stressVM;
      const tur = mesh.turgorPa;
      const pid = mesh.partIds;
      for (let i = 0; i < mushVertCount; i++) {
        const i3 = i * 3;
        const o = i * 12;
        dst[o] = pos[i3];
        dst[o + 1] = pos[i3 + 1];
        dst[o + 2] = pos[i3 + 2];
        dst[o + 3] = nrm[i3];
        dst[o + 4] = nrm[i3 + 1];
        dst[o + 5] = nrm[i3 + 2];
        dst[o + 6] = col[i3];
        dst[o + 7] = col[i3 + 1];
        dst[o + 8] = col[i3 + 2];
        dst[o + 9] = str[i];
        dst[o + 10] = tur[i];
        dst[o + 11] = pid[i];
      }
    }

    // Pack only active airborne spores into zero-GC scratch arrays
    const spBuf = snapshot.spores;
    const totalSlots = Math.min(spBuf.maxCount, spBuf.active.length, spBuf.positions.length >> 2);
    if (this.sporePosScratch.length < totalSlots * 4) {
      const nextCap = Math.max(totalSlots, this.sporePosScratch.length >> 1);
      this.sporePosScratch = new Float32Array(nextCap * 4);
      this.sporeVelScratch = new Float32Array(nextCap * 4);
    }
    let activeSporeCount = 0;
    if (view.showSpores && totalSlots > 0) {
      const srcPos = spBuf.positions;
      const srcVel = spBuf.velocities;
      const act = spBuf.active;
      const dstPos = this.sporePosScratch;
      const dstVel = this.sporeVelScratch;
      for (let i = 0; i < totalSlots; i++) {
        if (act[i] === 0) continue;
        const s4 = i * 4;
        const d4 = activeSporeCount * 4;
        dstPos[d4] = srcPos[s4];
        dstPos[d4 + 1] = srcPos[s4 + 1];
        dstPos[d4 + 2] = srcPos[s4 + 2];
        dstPos[d4 + 3] = srcPos[s4 + 3];
        dstVel[d4] = srcVel[s4];
        dstVel[d4 + 1] = srcVel[s4 + 1];
        dstVel[d4 + 2] = srcVel[s4 + 2];
        dstVel[d4 + 3] = srcVel[s4 + 3];
        activeSporeCount++;
      }
    }

    if (this.gpuBackend) {
      this.gpuBackend.renderFrame(
        snapshot,
        view,
        lDir,
        sliceCoord,
        this.viewTiltMat,
        this.projMat,
        this.uniformMain,
        this.uniformOcc,
        this.soilTexScratch,
        bounds.nx,
        bounds.ny,
        bounds.nz,
        this.sporeTexScratch,
        sp.nx,
        sp.nz,
        this.soilVertsScratch,
        soilVertCount,
        this.mushSubdivState.verts,
        this.mushSubdivState.vertCount,
        this.mushSubdivState.indices,
        this.mushSubdivState.indexCount,
        snapshot.mycelium.segments,
        snapshot.mycelium.segmentCount,
        view.showMycelium,
        this.obsVertsScratch,
        obsVertCount,
        this.glassVertsScratch,
        glassVerts,
        this.lineVertsScratch,
        lineVerts,
        this.sporePosScratch,
        this.sporeVelScratch,
        activeSporeCount,
        view.showSpores,
        this.streamVertsScratch,
        streamVertCount,
      );
    } else if (this.glBackend) {
      this.glBackend.renderFrame(
        this.uniformMain,
        this.uniformOcc,
        this.soilTexScratch,
        bounds.nx,
        bounds.ny,
        bounds.nz,
        this.sporeTexScratch,
        sp.nx,
        sp.nz,
        this.soilVertsScratch,
        soilVertCount,
        this.mushInterleavedScratch,
        mushVertCount,
        mesh.indices,
        mushIndexCount,
        snapshot.mycelium.segments,
        snapshot.mycelium.segmentCount,
        view.showMycelium,
        this.obsVertsScratch,
        obsVertCount,
        this.glassVertsScratch,
        glassVerts,
        this.lineVertsScratch,
        lineVerts,
        this.sporePosScratch,
        this.sporeVelScratch,
        activeSporeCount,
        view.showSpores,
        this.streamVertsScratch,
        streamVertCount,
      );
    }
  }

  /**
   * Pick a world-space point on a mushroom cap, razor slice plane, soil cutaway face, or top surface.
   * Accepts optional snapshot & view (falls back to last rendered frame for CONTRACT compatibility).
   */
  pickWorldPoint(
    clientX: number,
    clientY: number,
    snapshot?: SimulationSnapshot,
    view?: ViewState,
  ): PickResult {
    const snap = snapshot ?? this.lastSnapshot;
    const vw = view ?? this.lastView;
    const miss: PickResult = { hitKind: 'none', worldX: 0, worldY: 0, worldZ: 0, bodyId: -1 };
    if (!snap || !vw) return miss;

    const rect = this.canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return miss;
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;

    this.computeCameraMatrices(snap, vw);
    if (!mat4Invert(this.mvpMat, this.invMvpMat)) return miss;

    const ray = unprojectRay(ndcX, ndcY, this.invMvpMat, this.backend === 'webgpu');

    let bestT = Infinity;
    let best: PickResult = miss;

    // 1. Check mature/growing mushroom caps first
    for (let i = 0; i < snap.bodies.length; i++) {
      const body = snap.bodies[i];
      const capR = Math.max(0.016, body.capRadiusM * 1.15);
      const tHit = intersectRaySphere(ray, [body.stipeTipX, body.stipeTipY, body.stipeTipZ], capR);
      if (tHit !== null && tHit < bestT) {
        bestT = tHit;
        best = {
          hitKind: 'mushroom',
          worldX: ray.origin[0] + ray.dir[0] * tHit,
          worldY: ray.origin[1] + ray.dir[1] * tHit,
          worldZ: ray.origin[2] + ray.dir[2] * tHit,
          bodyId: body.id,
        };
      }
    }

    // 2. Check clipped soil bounding box & razor slice plane
    const bounds = getSoilBounds(snap);
    const clippedMax: Vec3 = [bounds.maxX, bounds.maxY, bounds.maxZ];
    if (vw.sliceEnabled) {
      const sc = computeSliceWorldCoord(bounds, vw);
      if (vw.sliceAxis === 'x') clippedMax[0] = Math.min(clippedMax[0], sc);
      else clippedMax[2] = Math.min(clippedMax[2], sc);

      const planeNormal: Vec3 = vw.sliceAxis === 'x' ? [1, 0, 0] : [0, 0, 1];
      const sliceHit = intersectRayPlane(ray, planeNormal, sc);
      if (sliceHit && sliceHit.t < bestT) {
        const p = sliceHit.point;
        const inBounds =
          p[0] >= bounds.minX - 0.002 && p[0] <= bounds.maxX + 0.002 &&
          p[1] >= bounds.minY - 0.002 && p[1] <= bounds.maxY + 0.002 &&
          p[2] >= bounds.minZ - 0.002 && p[2] <= bounds.maxZ + 0.002;
        if (inBounds) {
          bestT = sliceHit.t;
          best = {
            hitKind: 'soil',
            worldX: p[0],
            worldY: p[1],
            worldZ: p[2],
            bodyId: -1,
          };
        }
      }
    }

    const boxHit = intersectRayBox(ray, [bounds.minX, bounds.minY, bounds.minZ], clippedMax);
    if (boxHit && boxHit.tNear < bestT) {
      bestT = boxHit.tNear;
      best = {
        hitKind: boxHit.faceNormal[1] > 0.5 ? 'surface' : 'soil',
        worldX: boxHit.point[0],
        worldY: boxHit.point[1],
        worldZ: boxHit.point[2],
        bodyId: -1,
      };
    }

    return best;
  }

  dispose(): void {
    this.gpuBackend?.dispose();
    this.glBackend?.dispose();
    this.gpuBackend = null;
    this.glBackend = null;
  }
}
