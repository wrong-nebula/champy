// ============================================================================
// SPOROCARP — 3D Sculpted Mushroom Geometry & PN-Curved Subdivision (Zero-GC)
// Performs 1-level open-addressed PN-curved midpoint subdivision plus
// species-specific 3D silhouette vertex displacement and smooth area-weighted
// vertex normal recomputation into 64-byte (16-float) WebGPU vertices.
// ============================================================================

import type {
  FruitingBodyState,
  FruitingRenderMesh,
  SpeciesId,
} from '../types/contract.ts';

export interface MushSubdivState {
  verts: Float32Array;
  indices: Uint32Array;
  vertCount: number;
  indexCount: number;
}

const SUBDIV_EDGE_HASH_SIZE = 131072;
const subdivHashKeyLo = new Int32Array(SUBDIV_EDGE_HASH_SIZE);
const subdivHashKeyHi = new Int32Array(SUBDIV_EDGE_HASH_SIZE);
const subdivHashVal = new Int32Array(SUBDIV_EDGE_HASH_SIZE);

// Module-level zero-GC scratch arrays (grow only on demand)
let vertCapR = new Float32Array(32768);
let vertMaturity = new Float32Array(32768);
let vertTipY = new Float32Array(32768);
let vertDisplaced = new Uint8Array(32768);
let origNormals = new Float32Array(32768 * 3);
let accumNormals = new Float32Array(32768 * 3);

function ensureScratchCapacity(vertCap: number): void {
  if (vertCapR.length >= vertCap) return;
  const next = Math.max(vertCap, vertCapR.length * 2);
  vertCapR = new Float32Array(next);
  vertMaturity = new Float32Array(next);
  vertTipY = new Float32Array(next);
  vertDisplaced = new Uint8Array(next);
  origNormals = new Float32Array(next * 3);
  accumNormals = new Float32Array(next * 3);
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

function fract(x: number): number {
  return x - Math.floor(x);
}

/** Exact WGSL-matched hash31 */
function hash31(px: number, py: number, pz: number): number {
  let qx = fract(px * 0.1031);
  let qy = fract(py * 0.1030);
  let qz = fract(pz * 0.0973);
  const d = qx * (qy + 33.33) + qy * (qx + 33.33) + qz * (qz + 33.33);
  qx += d;
  qy += d;
  qz += d;
  return fract((qx + qy) * qz);
}

/** Exact WGSL-matched 3D valueNoise3D */
function valueNoise3D(px: number, py: number, pz: number): number {
  const ix = Math.floor(px);
  const iy = Math.floor(py);
  const iz = Math.floor(pz);
  const fx = px - ix;
  const fy = py - iy;
  const fz = pz - iz;
  const ux = fx * fx * (3.0 - 2.0 * fx);
  const uy = fy * fy * (3.0 - 2.0 * fy);
  const uz = fz * fz * (3.0 - 2.0 * fz);

  const n000 = hash31(ix, iy, iz);
  const n100 = hash31(ix + 1, iy, iz);
  const n010 = hash31(ix, iy + 1, iz);
  const n110 = hash31(ix + 1, iy + 1, iz);
  const n001 = hash31(ix, iy, iz + 1);
  const n101 = hash31(ix + 1, iy, iz + 1);
  const n011 = hash31(ix, iy + 1, iz + 1);
  const n111 = hash31(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;

  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

/** Exact WGSL-matched Voronoi 3D returning F2 - F1 for honeycomb ridge sculpting */
function evalVoronoiDiff(px: number, py: number, pz: number): number {
  const nx = Math.floor(px);
  const ny = Math.floor(py);
  const nz = Math.floor(pz);
  const fx = px - nx;
  const fy = py - ny;
  const fz = pz - nz;

  let f1 = 8.0;
  let f2 = 8.0;

  for (let dz = -1; dz <= 1; dz++) {
    const gz = nz + dz;
    for (let dy = -1; dy <= 1; dy++) {
      const gy = ny + dy;
      for (let dx = -1; dx <= 1; dx++) {
        const gx = nx + dx;
        // Exact WGSL hash33(n + g)
        let qx = fract(gx * 0.1031);
        let qy = fract(gy * 0.1030);
        let qz = fract(gz * 0.0973);
        const dotQ = qx * (qy + 33.33) + qy * (qx + 33.33) + qz * (qz + 33.33);
        qx += dotQ;
        qy += dotQ;
        qz += dotQ;
        const ox = fract((qx + qy) * qz);
        const oy = fract((qx + qx) * qy);
        const oz = fract((qy + qx) * qx);

        const rx = dx + ox - fx;
        const ry = dy + oy - fy;
        const rz = dz + oz - fz;
        const d2 = rx * rx + ry * ry + rz * rz;
        if (d2 < f1) {
          f2 = f1;
          f1 = d2;
        } else if (d2 < f2) {
          f2 = d2;
        }
      }
    }
  }

  return Math.sqrt(f2) - Math.sqrt(f1);
}

function applySpeciesSculpting(
  verts: Float32Array,
  indices: Uint32Array,
  vertCount: number,
  indexCount: number,
  speciesId: SpeciesId,
): void {
  if (vertCount <= 0 || indexCount <= 0) return;

  const isMorel = speciesId === 'morel';
  const isChanterelle = speciesId === 'chanterelle';
  const isParasol = speciesId === 'parasol';
  const isMycena = speciesId === 'mycena';

  // Save pre-displacement normals & apply species-specific 3D surface sculpting
  for (let i = 0; i < vertCount; i++) {
    const o = i * 16;
    const i3 = i * 3;
    const nx = verts[o + 3];
    const ny = verts[o + 4];
    const nz = verts[o + 5];
    origNormals[i3] = nx;
    origNormals[i3 + 1] = ny;
    origNormals[i3 + 2] = nz;
    accumNormals[i3] = 0;
    accumNormals[i3 + 1] = 0;
    accumNormals[i3 + 2] = 0;

    const partId = verts[o + 11];
    const eligible =
      isMorel || isChanterelle
        ? partId > 0.5 && partId < 2.5
        : partId > 0.5 && partId < 1.5;

    if (!eligible) {
      vertDisplaced[i] = 0;
      continue;
    }
    vertDisplaced[i] = 1;

    const px = verts[o];
    const py = verts[o + 1];
    const pz = verts[o + 2];
    const dxNorm = verts[o + 12];
    const dzNorm = verts[o + 13];
    const relY = verts[o + 14];
    const bSeed = verts[o + 15];

    const capR = Math.max(vertCapR[i], 0.0045);
    const maturity = clamp(vertMaturity[i], 0.0, 1.0);
    const rNorm = Math.hypot(dxNorm, dzNorm);
    const theta = Math.atan2(dzNorm, dxNorm);

    let disp = 0.0;

    if (isMorel) {
      // Exact GPU-matched 3D Voronoi cellular honeycomb displacement
      const cx = px * 165.0 + bSeed * 3.7;
      const cy = py * 165.0 + bSeed * 1.9;
      const cz = pz * 165.0 + bSeed * 5.1;
      const vorDiff = evalVoronoiDiff(cx, cy, cz);

      const ridge = 1.0 - smoothstep(0.03, 0.26, vorDiff);
      const pit = smoothstep(0.18, 0.56, vorDiff);
      const ridgeVar = 0.5 + 0.5 * Math.sin(px * 310.0 + pz * 270.0 + bSeed);
      const ridgeAmp = 0.0011 + 0.0003 * ridgeVar; // +1.1 .. +1.4 mm
      const pitAmp = 0.0006; // -0.6 mm

      // Taper cleanly at tucked lower collar near stipe junction to prevent stipe intersection
      const collarFade =
        smoothstep(-0.78, -0.22, ny) *
        (py < vertTipY[i] - 0.14 * capR && rNorm < 0.56 ? smoothstep(0.32, 0.58, rNorm) : 1.0);

      const morelScale = clamp(capR / 0.018, 0.35, 1.0);
      disp = (ridge * ridgeAmp - pit * pitAmp) * collarFade * morelScale;

      // Subtle cohesive tint reinforcement on pits vs ridge crests
      const shadeMod = clamp(0.86 + 0.24 * ridge - 0.12 * pit, 0.72, 1.15);
      verts[o + 6] = clamp(verts[o + 6] * shadeMod, 0.0, 1.0);
      verts[o + 7] = clamp(verts[o + 7] * shadeMod, 0.0, 1.0);
      verts[o + 8] = clamp(verts[o + 8] * shadeMod, 0.0, 1.0);
    } else if (isChanterelle) {
      const matScale = clamp(capR / 0.032, 0.25, 1.15) * (0.40 + 0.60 * maturity);
      // Harmonious low-frequency fluted margin wave shared across cap top & underside rim
      const marginScallop =
        (Math.sin(theta * 7.0 + bSeed) * 0.52 +
          Math.sin(theta * 13.0 + bSeed * 1.7) * 0.32 +
          Math.sin(theta * 21.0 - bSeed * 0.9) * 0.16) *
        smoothstep(0.54, 0.98, rNorm);

      if (partId > 1.5) {
        // Decurrent false-gill radial ridge pleats with natural dichotomous forking
        const wobble = Math.sin(theta * 5.0 + bSeed * 1.3) * 0.55 + rNorm * 3.1;
        const primary = Math.sin(theta * 22.0 + wobble + bSeed);
        const secondary = Math.sin(theta * 44.0 + wobble * 1.6 + bSeed * 2.3);
        const forkMix = smoothstep(0.38, 0.78, rNorm);
        const pleatWave = primary * (1.0 - 0.42 * forkMix) + secondary * (0.48 * forkMix);
        // Blunt, rounded ridge profile
        const bluntRidge = Math.pow(0.5 + 0.5 * clamp(pleatWave, -1.0, 1.0), 0.75) - 0.42;
        const rootTaper = smoothstep(0.16, 0.34, rNorm);
        disp = (bluntRidge * 0.00105 * rootTaper - marginScallop * 0.00055) * matScale;
      } else {
        // Upper cap margin undulation + faint radial rib echo near rim
        const ribEcho =
          Math.sin(theta * 22.0 + Math.sin(theta * 5.0 + bSeed * 1.3) * 0.55 + rNorm * 3.1 + bSeed) *
          smoothstep(0.68, 0.98, rNorm);
        disp = (marginScallop * 0.00105 + ribEcho * 0.00028) * matScale;
      }
    } else if (isParasol) {
      // Exact GPU-matched concentric scale-patch relief + clean umbo peak + margin scallop
      const pScale = clamp(capR / 0.036, 0.25, 1.15) * (0.40 + 0.60 * maturity);
      const umbo = smoothstep(0.28, 0.01, rNorm);
      const n95 = valueNoise3D(px * 95.0, py * 95.0, pz * 95.0);
      const n85 = valueNoise3D(px * 85.0, py * 85.0, pz * 85.0);
      const scaleBand = Math.sin(rNorm * 26.0 + n95 * 3.8 + bSeed);
      const patchBreak = smoothstep(-0.12, 0.52, Math.sin(theta * 14.0 + n85 * 6.4 + bSeed * 2.1));
      const patchMask =
        smoothstep(0.14, 0.68, scaleBand) *
        patchBreak *
        smoothstep(0.16, 0.42, rNorm) *
        (1.0 - smoothstep(0.78, 1.04, rNorm));
      const marginScallop =
        (Math.sin(theta * 9.0 + bSeed) * 0.50 +
          Math.sin(theta * 17.0 + bSeed * 1.3) * 0.35 +
          Math.sin(theta * 29.0 - bSeed * 0.7) * 0.15) *
        smoothstep(0.62, 0.98, rNorm);

      disp = (umbo * 0.00055 + patchMask * 0.00048 + marginScallop * 0.00042) * pScale;
    } else if (isMycena) {
      // Mycena chlorophos: delicate translucent campanulate radial bell fluting + smooth scalloped edge bevel
      const mycScale = clamp(capR / 0.022, 0.30, 1.15) * (0.40 + 0.60 * maturity);
      const radialMask = smoothstep(0.18, 0.92, rNorm);
      const flute = Math.sin(theta * 24.0 + bSeed) * radialMask * 0.00045;
      const rimScallop =
        (Math.sin(theta * 12.0 + bSeed) * 0.52 +
          Math.sin(theta * 24.0 + bSeed) * 0.48) *
        smoothstep(0.70, 0.99, rNorm) *
        0.00024;
      const edgeBevel = -smoothstep(0.84, 1.02, rNorm) * 0.00022;
      disp = (flute + rimScallop + edgeBevel) * mycScale;
    } else {
      // Inkcap: longitudinal fibrous shaggy ridges + delicate deliquescent rim curl
      const inkScale = clamp(capR / 0.022, 0.30, 1.12) * (0.45 + 0.55 * maturity);
      const fibNoise = valueNoise3D(
        Math.cos(theta) * 18.0 + bSeed,
        relY * 180.0 + bSeed,
        Math.sin(theta) * 18.0 + bSeed,
      );
      const longRib =
        (Math.sin(theta * 18.0 + fibNoise * 2.8 + bSeed) * 0.6 +
          Math.sin(theta * 34.0 - relY * 14.0 + bSeed * 1.7) * 0.4) *
        smoothstep(0.16, 0.38, rNorm) *
        (1.0 - smoothstep(0.84, 0.99, rNorm));
      const shaggyBoost = smoothstep(0.40, 0.76, fibNoise);
      const shaggyRidge = (0.35 + 0.65 * shaggyBoost) * longRib * 0.00052;

      const rimFade = smoothstep(0.72, 0.99, rNorm);
      const dripWave =
        Math.sin(theta * 11.0 + bSeed) * 0.50 +
        Math.sin(theta * 23.0 + bSeed * 2.1) * 0.35 +
        Math.sin(theta * 37.0 - bSeed) * 0.15;
      const rimCurl = rimFade * (0.00048 + dripWave * 0.00042);

      disp = (shaggyRidge + rimCurl) * inkScale;
    }

    // Clamp displacement safely away from tearing or self-intersection
    const maxOut = Math.min(0.0022, capR * 0.16);
    const maxIn = Math.min(0.0014, capR * 0.10);
    disp = clamp(disp, -maxIn, maxOut);

    verts[o] = px + nx * disp;
    verts[o + 1] = py + ny * disp;
    verts[o + 2] = pz + nz * disp;
  }

  // Recompute smooth triangle-weighted vertex normals across displaced surfaces
  for (let t = 0; t + 2 < indexCount; t += 3) {
    const i0 = indices[t];
    const i1 = indices[t + 1];
    const i2 = indices[t + 2];
    if (i0 >= vertCount || i1 >= vertCount || i2 >= vertCount) continue;
    if (vertDisplaced[i0] === 0 && vertDisplaced[i1] === 0 && vertDisplaced[i2] === 0) continue;

    const o0 = i0 * 16;
    const o1 = i1 * 16;
    const o2 = i2 * 16;

    const ax = verts[o1] - verts[o0];
    const ay = verts[o1 + 1] - verts[o0 + 1];
    const az = verts[o1 + 2] - verts[o0 + 2];
    const bx = verts[o2] - verts[o0];
    const by = verts[o2 + 1] - verts[o0 + 1];
    const bz = verts[o2 + 2] - verts[o0 + 2];

    // Triangle area-weighted face normal (cross product magnitude = 2 * area)
    const fx = ay * bz - az * by;
    const fy = az * bx - ax * bz;
    const fz = ax * by - ay * bx;

    if (vertDisplaced[i0] !== 0) {
      const k = i0 * 3;
      accumNormals[k] += fx;
      accumNormals[k + 1] += fy;
      accumNormals[k + 2] += fz;
    }
    if (vertDisplaced[i1] !== 0) {
      const k = i1 * 3;
      accumNormals[k] += fx;
      accumNormals[k + 1] += fy;
      accumNormals[k + 2] += fz;
    }
    if (vertDisplaced[i2] !== 0) {
      const k = i2 * 3;
      accumNormals[k] += fx;
      accumNormals[k + 1] += fy;
      accumNormals[k + 2] += fz;
    }
  }

  for (let i = 0; i < vertCount; i++) {
    if (vertDisplaced[i] === 0) continue;
    const i3 = i * 3;
    const o = i * 16;
    const ax = accumNormals[i3];
    const ay = accumNormals[i3 + 1];
    const az = accumNormals[i3 + 2];
    const len = Math.hypot(ax, ay, az);
    if (len > 1e-12) {
      const inv = 1.0 / len;
      let nx = ax * inv;
      let ny = ay * inv;
      let nz = az * inv;
      // Guard against inverted winding on degenerate boundary triangles
      const ox = origNormals[i3];
      const oy = origNormals[i3 + 1];
      const oz = origNormals[i3 + 2];
      if (nx * ox + ny * oy + nz * oz < -0.15) {
        nx = ox;
        ny = oy;
        nz = oz;
      }
      verts[o + 3] = nx;
      verts[o + 4] = ny;
      verts[o + 5] = nz;
    } else {
      verts[o + 3] = origNormals[i3];
      verts[o + 4] = origNormals[i3 + 1];
      verts[o + 5] = origNormals[i3 + 2];
    }
  }
}

/**
 * Packs mushroom geometry into 64-byte (16-float) WebGPU vertices with botanical
 * coordinates (dxNorm, dzNorm, relY, bodySeed), applies 1-level PN-curved midpoint
 * subdivision when triangle count <= 10,000, sculpts species-specific 3D silhouette
 * relief, and recomputes smooth area-weighted vertex normals.
 */
export function packSculptedMushroomMesh(
  mesh: FruitingRenderMesh,
  bodies: readonly FruitingBodyState[],
  speciesId: SpeciesId,
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
  if (triCount <= 0) {
    state.vertCount = 0;
    state.indexCount = 0;
    return;
  }

  const doSubdiv = triCount <= 10000 && origV < 32768;
  const maxVertsNeeded = doSubdiv ? origV + triCount * 3 : origV;
  const maxIdxNeeded = doSubdiv ? triCount * 12 : origI;

  if (state.verts.length < maxVertsNeeded * 16) {
    state.verts = new Float32Array(Math.max(maxVertsNeeded * 16, state.verts.length * 2));
  }
  if (state.indices.length < maxIdxNeeded) {
    state.indices = new Uint32Array(Math.max(maxIdxNeeded, state.indices.length * 2));
  }
  ensureScratchCapacity(maxVertsNeeded);

  const dst = state.verts;
  const outIndices = state.indices;
  const pos = mesh.positions;
  const nrm = mesh.normals;
  const col = mesh.colors;
  const str = mesh.stressVM;
  const tur = mesh.turgorPa;
  const pid = mesh.partIds;
  const bodyCount = bodies.length;
  const isMorel = speciesId === 'morel';

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

    const rnx = nrm[i3];
    const rny = nrm[i3 + 1];
    const rnz = nrm[i3 + 2];
    const nLen = Math.hypot(rnx, rny, rnz);
    if (nLen > 1e-6) {
      const invN = 1.0 / nLen;
      dst[o + 3] = rnx * invN;
      dst[o + 4] = rny * invN;
      dst[o + 5] = rnz * invN;
    } else {
      dst[o + 3] = 0;
      dst[o + 4] = 1;
      dst[o + 5] = 0;
    }

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
      vertCapR[i] = best.capRadiusM;
      vertMaturity[i] = best.maturity;
      vertTipY[i] = best.stipeTipY;
    } else {
      dst[o + 12] = 0;
      dst[o + 13] = 0;
      dst[o + 14] = 0.5;
      dst[o + 15] = 0;
      vertCapR[i] = 0.025;
      vertMaturity[i] = 0.8;
      vertTipY[i] = 0.06;
    }
  }

  if (!doSubdiv) {
    outIndices.set(mesh.indices.subarray(0, origI));
    state.vertCount = origV;
    state.indexCount = origI;
    if (origV > 0) {
      applySpeciesSculpting(dst, outIndices, origV, origI, speciesId);
    }
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
    for (let p = 0; p < 512; p++) {
      const kLo = subdivHashKeyLo[slot];
      if (kLo === -1) {
        const mid = nextVert++;
        const ao = a * 16;
        const bo = b * 16;
        const mo = mid * 16;

        const ax = dst[ao];
        const ay = dst[ao + 1];
        const az = dst[ao + 2];
        const bx = dst[bo];
        const by = dst[bo + 1];
        const bz = dst[bo + 2];
        const nax = dst[ao + 3];
        const nay = dst[ao + 4];
        const naz = dst[ao + 5];
        const nbx = dst[bo + 3];
        const nby = dst[bo + 4];
        const nbz = dst[bo + 5];

        const nd = Math.max(-1, Math.min(1, nax * nbx + nay * nby + naz * nbz));
        const crispThresh = isMorel ? 2.5 : 1.5;
        const crispEdge = dst[ao + 11] >= crispThresh || dst[bo + 11] >= crispThresh;
        const smoothK = Math.max(0, Math.min(1, (nd - 0.30) / 0.70)) * (crispEdge ? 0.12 : 1.0);
        const dax = ax - bx;
        const day = ay - by;
        const daz = az - bz;
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
          const inv = 1.0 / mnLen;
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

        vertCapR[mid] = 0.5 * (vertCapR[a] + vertCapR[b]);
        vertMaturity[mid] = 0.5 * (vertMaturity[a] + vertMaturity[b]);
        vertTipY[mid] = 0.5 * (vertTipY[a] + vertTipY[b]);

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

  applySpeciesSculpting(dst, outIndices, nextVert, outIdx, speciesId);
}
