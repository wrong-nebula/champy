// ============================================================================
// SPOROCARP — Minimal Zero-Dependency 3D Math, Camera & Ray-Picking Helpers
// Column-major 4x4 matrices (WebGPU / WebGL2 standard).
// ============================================================================

export type Vec3 = [number, number, number];
export type Mat4 = Float32Array; // length 16, column-major

export interface Ray {
  origin: Vec3;
  dir: Vec3;
}

export function vec3(x = 0, y = 0, z = 0): Vec3 {
  return [x, y, z];
}

export function vec3Sub(a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vec3Add(a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function vec3Scale(a: Readonly<Vec3>, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function vec3Dot(a: Readonly<Vec3>, b: Readonly<Vec3>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vec3Cross(a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function vec3Length(a: Readonly<Vec3>): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function vec3Normalize(a: Readonly<Vec3>): Vec3 {
  const len = Math.hypot(a[0], a[1], a[2]);
  if (len < 1e-12) return [0, 1, 0];
  const inv = 1 / len;
  return [a[0] * inv, a[1] * inv, a[2] * inv];
}

export function mat4Identity(out: Mat4 = new Float32Array(16)): Mat4 {
  out.fill(0);
  out[0] = 1;
  out[5] = 1;
  out[10] = 1;
  out[15] = 1;
  return out;
}

/**
 * Multiply two column-major 4x4 matrices: out = a * b
 */
export function mat4Multiply(a: Mat4, b: Mat4, out: Mat4 = new Float32Array(16)): Mat4 {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  for (let col = 0; col < 4; col++) {
    const i = col * 4;
    const b0 = b[i];
    const b1 = b[i + 1];
    const b2 = b[i + 2];
    const b3 = b[i + 3];
    out[i] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
    out[i + 1] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
    out[i + 2] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
    out[i + 3] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;
  }
  return out;
}

/**
 * Perspective projection matrix.
 * zeroToOne = true for WebGPU NDC z in [0, 1], false for WebGL2 NDC z in [-1, 1].
 */
export function mat4Perspective(
  fovYRad: number,
  aspect: number,
  near: number,
  far: number,
  zeroToOne: boolean,
  out: Mat4 = new Float32Array(16),
): Mat4 {
  const f = 1.0 / Math.tan(fovYRad * 0.5);
  out.fill(0);
  out[0] = f / Math.max(aspect, 1e-4);
  out[5] = f;
  out[11] = -1;
  if (zeroToOne) {
    const nf = 1 / (near - far);
    out[10] = far * nf;
    out[14] = far * near * nf;
  } else {
    const nf = 1 / (near - far);
    out[10] = (far + near) * nf;
    out[14] = 2 * far * near * nf;
  }
  return out;
}

/**
 * Right-handed lookAt view matrix.
 */
export function mat4LookAt(
  eye: Readonly<Vec3>,
  target: Readonly<Vec3>,
  up: Readonly<Vec3>,
  out: Mat4 = new Float32Array(16),
): Mat4 {
  const z = vec3Normalize(vec3Sub(eye, target));
  const x = vec3Normalize(vec3Cross(up, z));
  const y = vec3Cross(z, x);

  out[0] = x[0];
  out[1] = y[0];
  out[2] = z[0];
  out[3] = 0;

  out[4] = x[1];
  out[5] = y[1];
  out[6] = z[1];
  out[7] = 0;

  out[8] = x[2];
  out[9] = y[2];
  out[10] = z[2];
  out[11] = 0;

  out[12] = -vec3Dot(x, eye);
  out[13] = -vec3Dot(y, eye);
  out[14] = -vec3Dot(z, eye);
  out[15] = 1;
  return out;
}

/**
 * Compute eye world position from spherical orbit coordinates.
 */
export function computeOrbitCameraEye(
  yaw: number,
  pitch: number,
  distance: number,
  target: Readonly<Vec3>,
): Vec3 {
  const clampedPitch = Math.max(-Math.PI * 0.48, Math.min(Math.PI * 0.48, pitch));
  const cosP = Math.cos(clampedPitch);
  const sinP = Math.sin(clampedPitch);
  return [
    target[0] + distance * cosP * Math.sin(yaw),
    target[1] + distance * sinP,
    target[2] + distance * cosP * Math.cos(yaw),
  ];
}

/**
 * Invert a general 4x4 matrix. Returns null if singular.
 */
export function mat4Invert(m: Mat4, out: Mat4 = new Float32Array(16)): Mat4 | null {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) return null;
  det = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

export function mat4TransformPoint(m: Mat4, p: Readonly<Vec3>): Vec3 {
  const x = p[0], y = p[1], z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  const invW = Math.abs(w) > 1e-12 ? 1 / w : 1;
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) * invW,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) * invW,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) * invW,
  ];
}

/**
 * Unproject normalized device coordinates (ndcX in [-1,1], ndcY in [-1,1])
 * using an inverse view-projection matrix.
 */
export function unprojectRay(
  ndcX: number,
  ndcY: number,
  invViewProj: Mat4,
  zeroToOne: boolean,
): Ray {
  const nearZ = zeroToOne ? 0.0 : -1.0;
  const farZ = 1.0;
  const pNear = mat4TransformPoint(invViewProj, [ndcX, ndcY, nearZ]);
  const pFar = mat4TransformPoint(invViewProj, [ndcX, ndcY, farZ]);
  const dir = vec3Normalize(vec3Sub(pFar, pNear));
  return { origin: pNear, dir };
}

/**
 * Ray-AABB intersection (slab method). Returns t >= 0 or null if no hit.
 */
export function intersectRayBox(
  ray: Ray,
  minB: Readonly<Vec3>,
  maxB: Readonly<Vec3>,
): { tNear: number; tFar: number; point: Vec3; faceNormal: Vec3 } | null {
  let tMin = -Infinity;
  let tMax = Infinity;
  let hitAxis = 0;
  let hitSign = 1;

  for (let axis = 0; axis < 3; axis++) {
    const o = ray.origin[axis];
    const d = ray.dir[axis];
    if (Math.abs(d) < 1e-9) {
      if (o < minB[axis] || o > maxB[axis]) return null;
    } else {
      const invD = 1 / d;
      let t1 = (minB[axis] - o) * invD;
      let t2 = (maxB[axis] - o) * invD;
      let sign = -1;
      if (t1 > t2) {
        const tmp = t1;
        t1 = t2;
        t2 = tmp;
        sign = 1;
      }
      if (t1 > tMin) {
        tMin = t1;
        hitAxis = axis;
        hitSign = sign;
      }
      if (t2 < tMax) {
        tMax = t2;
      }
      if (tMin > tMax) return null;
    }
  }

  if (tMax < 0) return null;
  const t = tMin >= 0 ? tMin : tMax;
  const point = vec3Add(ray.origin, vec3Scale(ray.dir, t));
  const faceNormal: Vec3 = [0, 0, 0];
  faceNormal[hitAxis] = hitSign;
  return { tNear: t, tFar: tMax, point, faceNormal };
}

/**
 * Ray-sphere intersection. Returns closest positive t or null.
 */
export function intersectRaySphere(
  ray: Ray,
  center: Readonly<Vec3>,
  radius: number,
): number | null {
  const oc = vec3Sub(ray.origin, center);
  const b = vec3Dot(oc, ray.dir);
  const c = vec3Dot(oc, oc) - radius * radius;
  const disc = b * b - c;
  if (disc < 0) return null;
  const sqrtD = Math.sqrt(disc);
  const t1 = -b - sqrtD;
  if (t1 >= 0) return t1;
  const t2 = -b + sqrtD;
  return t2 >= 0 ? t2 : null;
}

/**
 * Ray-plane intersection for a plane `dot(p, normal) = d`.
 */
export function intersectRayPlane(
  ray: Ray,
  normal: Readonly<Vec3>,
  d: number,
): { t: number; point: Vec3 } | null {
  const denom = vec3Dot(ray.dir, normal);
  if (Math.abs(denom) < 1e-8) return null;
  const t = (d - vec3Dot(ray.origin, normal)) / denom;
  if (t < 0) return null;
  return { t, point: vec3Add(ray.origin, vec3Scale(ray.dir, t)) };
}

/**
 * Create a rotation matrix around the Z axis (for terrain tilt visualization).
 */
export function mat4RotationZ(angleRad: number, out: Mat4 = new Float32Array(16)): Mat4 {
  mat4Identity(out);
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  out[0] = c;
  out[1] = s;
  out[4] = -s;
  out[5] = c;
  return out;
}
