// ============================================================================
// SPOROCARP — Curved Organic Rhizomorph & Hyphal Wisp Geometry (Zero-GC)
// Converts straight 8-float mycelium graph segments into smooth 2- or 3-part
// quadratic Bezier organic curves with tip tapering plus sparse fine wispy
// translucent hyphal side filaments up to a hard cap of 12,000 segments.
// ============================================================================

export const MAX_CURVED_MYCELIUM_SEGMENTS = 12000;

export interface CurvedMyceliumState {
  segments: Float32Array;
  count: number;
}

function hashInt(x: number, y: number, z: number, salt: number): number {
  let h = Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791) ^ Math.imul(salt, 2654435761);
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967296.0;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function writeSeg(
  dst: Float32Array,
  idx: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  r: number,
  turgorNorm: number,
): void {
  const o = idx * 8;
  dst[o] = ax;
  dst[o + 1] = ay;
  dst[o + 2] = az;
  dst[o + 3] = bx;
  dst[o + 4] = by;
  dst[o + 5] = bz;
  dst[o + 6] = r;
  dst[o + 7] = turgorNorm;
}

// Module-level zero-GC scratch for shared curve geometry helper (u, v, controlPt)
const _curveScratch = new Float64Array(9);

function computeSegmentBasisAndControl(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  dirX: number,
  dirY: number,
  dirZ: number,
  len: number,
  qx: number,
  qy: number,
  qz: number,
): void {
  const refX = Math.abs(dirY) < 0.88 ? 0.0 : 1.0;
  const refY = Math.abs(dirY) < 0.88 ? 1.0 : 0.0;
  let ux = -dirZ * refY;
  let uy = dirZ * refX;
  let uz = dirX * refY - dirY * refX;
  const uLen = Math.hypot(ux, uy, uz) || 1.0;
  ux /= uLen;
  uy /= uLen;
  uz /= uLen;
  const vx = dirY * uz - dirZ * uy;
  const vy = dirZ * ux - dirX * uz;
  const vz = dirX * uy - dirY * ux;

  const hAng = hashInt(qx, qy, qz, 17);
  const hAmp = hashInt(qx, qy, qz, 53);
  const ang = hAng * Math.PI * 2.0;
  const ca = Math.cos(ang);
  const sa = Math.sin(ang);

  const nrmX = ux * ca + vx * sa;
  const nrmY = uy * ca + vy * sa;
  const nrmZ = uz * ca + vz * sa;

  const bendAmp = Math.min(0.0046, len * (0.12 + 0.14 * hAmp));
  const mx = 0.5 * (ax + bx);
  const my = 0.5 * (ay + by);
  const mz = 0.5 * (az + bz);
  let cy = my + nrmY * bendAmp;
  if (ay < -0.0005 && by < -0.0005 && cy > -0.0004) {
    cy = -0.0004;
  }

  _curveScratch[0] = ux;
  _curveScratch[1] = uy;
  _curveScratch[2] = uz;
  _curveScratch[3] = vx;
  _curveScratch[4] = vy;
  _curveScratch[5] = vz;
  _curveScratch[6] = mx + nrmX * bendAmp;
  _curveScratch[7] = cy;
  _curveScratch[8] = mz + nrmZ * bendAmp;
}

/**
 * Converts each 8-float input segment [ax, ay, az, bx, by, bz, r, turgorNorm]
 * into smooth 2- or 3-part quadratic organic curved segments with tapered tips,
 * plus sparse fine wispy translucent hyphal side filaments up to 12,000 segments.
 */
export function buildCurvedMyceliumSegments(
  rawSegments: Float32Array,
  rawCount: number,
  state: CurvedMyceliumState,
): void {
  const maxRaw = Math.floor(rawSegments.length / 8);
  const inCount = Math.min(rawCount, maxRaw);
  if (inCount <= 0) {
    state.count = 0;
    return;
  }

  if (state.segments.length < MAX_CURVED_MYCELIUM_SEGMENTS * 8) {
    state.segments = new Float32Array(MAX_CURVED_MYCELIUM_SEGMENTS * 8);
  }
  const dst = state.segments;

  const partsPerSeg = inCount <= 3000 ? 3 : inCount <= 5500 ? 2 : 1;
  let outCount = 0;

  // Pass 1: Convert primary rhizomorph/hyphal edges into quadratic Bezier curves
  for (let i = 0; i < inCount && outCount < MAX_CURVED_MYCELIUM_SEGMENTS; i++) {
    const s8 = i * 8;
    const ax = rawSegments[s8];
    const ay = rawSegments[s8 + 1];
    const az = rawSegments[s8 + 2];
    const bx = rawSegments[s8 + 3];
    const by = rawSegments[s8 + 4];
    const bz = rawSegments[s8 + 5];
    const r = rawSegments[s8 + 6];
    const turgor = clamp(rawSegments[s8 + 7], 0.0, 1.0);

    const dx = bx - ax;
    const dy = by - ay;
    const dz = bz - az;
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-6) continue;

    if (partsPerSeg === 1 || outCount + partsPerSeg > MAX_CURVED_MYCELIUM_SEGMENTS) {
      writeSeg(dst, outCount++, ax, ay, az, bx, by, bz, r, turgor);
      continue;
    }

    const invL = 1.0 / len;
    const qx = Math.round(0.5 * (ax + bx) * 10000.0);
    const qy = Math.round(0.5 * (ay + by) * 10000.0);
    const qz = Math.round(0.5 * (az + bz) * 10000.0);

    computeSegmentBasisAndControl(
      ax, ay, az, bx, by, bz,
      dx * invL, dy * invL, dz * invL, len,
      qx, qy, qz,
    );
    const cx = _curveScratch[6];
    const cy = _curveScratch[7];
    const cz = _curveScratch[8];

    if (partsPerSeg === 3) {
      // Quadratic Bezier evaluated at t = 0, 1/3, 2/3, 1
      const t1 = 1.0 / 3.0;
      const o1 = 1.0 - t1;
      const p1x = o1 * o1 * ax + 2.0 * o1 * t1 * cx + t1 * t1 * bx;
      const p1y = o1 * o1 * ay + 2.0 * o1 * t1 * cy + t1 * t1 * by;
      const p1z = o1 * o1 * az + 2.0 * o1 * t1 * cz + t1 * t1 * bz;

      const t2 = 2.0 / 3.0;
      const o2 = 1.0 - t2;
      const p2x = o2 * o2 * ax + 2.0 * o2 * t2 * cx + t2 * t2 * bx;
      const p2y = o2 * o2 * ay + 2.0 * o2 * t2 * cy + t2 * t2 * by;
      const p2z = o2 * o2 * az + 2.0 * o2 * t2 * cz + t2 * t2 * bz;

      writeSeg(dst, outCount++, ax, ay, az, p1x, p1y, p1z, r * 1.04, clamp(turgor * 0.96, 0.0, 1.0));
      writeSeg(dst, outCount++, p1x, p1y, p1z, p2x, p2y, p2z, r * 0.93, turgor);
      writeSeg(dst, outCount++, p2x, p2y, p2z, bx, by, bz, r * 0.78, clamp(turgor * 1.04, 0.0, 1.0));
    } else {
      // 2-part quadratic Bezier at t = 0, 0.5, 1
      const px = 0.25 * ax + 0.5 * cx + 0.25 * bx;
      const py = 0.25 * ay + 0.5 * cy + 0.25 * by;
      const pz = 0.25 * az + 0.5 * cz + 0.25 * bz;

      writeSeg(dst, outCount++, ax, ay, az, px, py, pz, r * 1.02, turgor);
      writeSeg(dst, outCount++, px, py, pz, bx, by, bz, r * 0.80, clamp(turgor * 1.03, 0.0, 1.0));
    }
  }

  // Pass 2: Sparse fine wispy translucent hyphal side filaments branching into pores
  for (let i = 0; i < inCount && outCount < MAX_CURVED_MYCELIUM_SEGMENTS; i++) {
    const s8 = i * 8;
    const ax = rawSegments[s8];
    const ay = rawSegments[s8 + 1];
    const az = rawSegments[s8 + 2];
    const bx = rawSegments[s8 + 3];
    const by = rawSegments[s8 + 4];
    const bz = rawSegments[s8 + 5];
    const r = rawSegments[s8 + 6];
    const turgor = clamp(rawSegments[s8 + 7], 0.0, 1.0);

    if (turgor < 0.12) continue;
    const dx = bx - ax;
    const dy = by - ay;
    const dz = bz - az;
    const len = Math.hypot(dx, dy, dz);
    if (len < 0.0025) continue;

    const mx = 0.5 * (ax + bx);
    const my = 0.5 * (ay + by);
    const mz = 0.5 * (az + bz);
    const qx = Math.round(mx * 10000.0);
    const qy = Math.round(my * 10000.0);
    const qz = Math.round(mz * 10000.0);

    if (hashInt(qx, qy, qz, 101) < 0.58) continue;

    const invL = 1.0 / len;
    const dirX = dx * invL;
    const dirY = dy * invL;
    const dirZ = dz * invL;

    computeSegmentBasisAndControl(
      ax, ay, az, bx, by, bz,
      dirX, dirY, dirZ, len,
      qx, qy, qz,
    );
    const ux = _curveScratch[0];
    const uy = _curveScratch[1];
    const uz = _curveScratch[2];
    const vx = _curveScratch[3];
    const vy = _curveScratch[4];
    const vz = _curveScratch[5];
    const cx = _curveScratch[6];
    const cy = _curveScratch[7];
    const cz = _curveScratch[8];

    const tAnc = 0.28 + 0.44 * hashInt(qx, qy, qz, 137);
    const oAnc = 1.0 - tAnc;
    const w0x = oAnc * oAnc * ax + 2.0 * oAnc * tAnc * cx + tAnc * tAnc * bx;
    const w0y = oAnc * oAnc * ay + 2.0 * oAnc * tAnc * cy + tAnc * tAnc * by;
    const w0z = oAnc * oAnc * az + 2.0 * oAnc * tAnc * cz + tAnc * tAnc * bz;

    const sideAng = hashInt(qx, qy, qz, 193) * Math.PI * 2.0;
    const latX = ux * Math.cos(sideAng) + vx * Math.sin(sideAng);
    const latY = uy * Math.cos(sideAng) + vy * Math.sin(sideAng);
    const latZ = uz * Math.cos(sideAng) + vz * Math.sin(sideAng);

    let wx = dirX * 0.56 + latX * 0.83;
    let wy = dirY * 0.56 + latY * 0.83;
    let wz = dirZ * 0.56 + latZ * 0.83;
    const wNorm = Math.hypot(wx, wy, wz) || 1.0;
    wx /= wNorm;
    wy /= wNorm;
    wz /= wNorm;

    const nrmX = cx - mx;
    const nrmY = cy - my;
    const nrmZ = cz - mz;
    let curlX = wy * nrmZ - wz * nrmY;
    let curlY = wz * nrmX - wx * nrmZ;
    let curlZ = wx * nrmY - wy * nrmX;
    const cLen = Math.hypot(curlX, curlY, curlZ) || 1.0;
    curlX /= cLen;
    curlY /= cLen;
    curlZ /= cLen;

    const wLen = Math.min(0.011, Math.max(0.0032, len * (0.42 + 0.36 * hashInt(qx, qy, qz, 241))));
    const wRad = r * 0.45;
    const wTurg = turgor * 0.66;

    if (outCount + 2 <= MAX_CURVED_MYCELIUM_SEGMENTS) {
      let w1x = w0x + wx * (0.52 * wLen) + curlX * (0.14 * wLen);
      let w1y = w0y + wy * (0.52 * wLen) + curlY * (0.14 * wLen);
      let w1z = w0z + wz * (0.52 * wLen) + curlZ * (0.14 * wLen);

      let w2x = w0x + wx * wLen + curlX * (0.34 * wLen);
      let w2y = w0y + wy * wLen + curlY * (0.34 * wLen);
      let w2z = w0z + wz * wLen + curlZ * (0.34 * wLen);

      if (w0y < -0.0005) {
        if (w1y > -0.0004) w1y = -0.0004;
        if (w2y > -0.0004) w2y = -0.0004;
      }

      writeSeg(dst, outCount++, w0x, w0y, w0z, w1x, w1y, w1z, wRad, wTurg);
      writeSeg(dst, outCount++, w1x, w1y, w1z, w2x, w2y, w2z, r * 0.34, wTurg * 0.78);
    } else {
      let w2x = w0x + wx * wLen;
      let w2y = w0y + wy * wLen;
      let w2z = w0z + wz * wLen;
      if (w0y < -0.0005 && w2y > -0.0004) {
        w2y = -0.0004;
      }
      writeSeg(dst, outCount++, w0x, w0y, w0z, w2x, w2y, w2z, wRad, wTurg);
    }
  }

  state.count = outCount;
}
