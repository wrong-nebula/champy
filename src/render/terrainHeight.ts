// ============================================================================
// SPOROCARP — Deterministic Forest-Floor Heightfield & Surface Normal Field
// Shared by the displaced soil top mesh, contoured cutaway walls, razor
// slice plane, BotanicalScatterPass, and ProceduralSplatPass so all props
// and hyphae seat cleanly with zero floating or clipping.
// ============================================================================
import {
  SoilMaterialId,
  type FruitingBodyState,
  type ISoilGrid,
} from '../types/contract.ts';

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

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * Evaluate forest-floor elevation above y = 0 at world (x, z).
 * Smoothly swells above shallow rotting oak logs & stones while tapering to
 * zero at active fruiting-body bases so stipes remain grounded at y = 0.
 */
export function evalSoilSurfaceHeightM(
  x: number,
  z: number,
  soil: ISoilGrid,
  bodies: readonly FruitingBodyState[],
): number {
  const halfX = soil.sizeX * 0.5;
  const halfZ = soil.sizeZ * 0.5;
  const fx = ((x + halfX) / soil.dx) - 0.5;
  const fz = ((z + halfZ) / soil.dx) - 0.5;
  const ix0 = Math.max(0, Math.min(soil.nx - 1, Math.floor(fx)));
  const iz0 = Math.max(0, Math.min(soil.nz - 1, Math.floor(fz)));
  const ix1 = Math.min(soil.nx - 1, ix0 + 1);
  const iz1 = Math.min(soil.nz - 1, iz0 + 1);
  const tx = Math.max(0, Math.min(1, fx - ix0));
  const tz = Math.max(0, Math.min(1, fz - iz0));

  const corners: [number, number][] = [
    [ix0, iz0],
    [ix1, iz0],
    [ix0, iz1],
    [ix1, iz1],
  ];
  const weights = [
    (1 - tx) * (1 - tz),
    tx * (1 - tz),
    (1 - tx) * tz,
    tx * tz,
  ];

  let woodScore = 0;
  let stoneScore = 0;
  let humusScore = 0;
  const iyTop = soil.ny - 1;
  const iyMin = Math.max(0, soil.ny - 5);

  for (let c = 0; c < 4; c++) {
    const [cx, cz] = corners[c];
    const w = weights[c];
    for (let iy = iyTop; iy >= iyMin; iy--) {
      const m = soil.material[soil.voxelIndex(cx, iy, cz)];
      const depthWeight = 1 - (iyTop - iy) * 0.18;
      if (m === SoilMaterialId.WOOD) {
        woodScore += w * depthWeight * 0.36;
      } else if (m === SoilMaterialId.STONE) {
        stoneScore += w * depthWeight * 0.34;
      } else if (iy === iyTop && m === SoilMaterialId.HUMUS) {
        humusScore += w;
      }
    }
  }

  woodScore = Math.min(1, woodScore);
  stoneScore = Math.min(1, stoneScore);

  const macroMound = fbm2(x * 10.5 + 3.1, z * 10.5 + 7.4) * 0.0085;
  const microTuft = fbm2(x * 34.0 - 5.2, z * 34.0 + 1.9) * 0.0024 * (0.4 + 0.6 * humusScore);
  const woodArch = woodScore * (0.011 + 0.004 * Math.sin(z * 32 + x * 18));
  const stoneBump = stoneScore * 0.0095;

  let h = macroMound + microTuft + woodArch + stoneBump;
  // Keep vertical relief within natural bounds [-4 mm, +15 mm]
  h = Math.max(-0.004, Math.min(0.015, h));

  // Gentle organic edge recession (1-2 mm soft bevel fading cleanly within 1.5 cm of perimeter)
  const edgeDistX = Math.max(0, halfX - Math.abs(x));
  const edgeDistZ = Math.max(0, halfZ - Math.abs(z));
  const bevelWidth = Math.max(0.011, Math.min(0.015, 0.0135 + fbm2(x * 24.0 + 1.3, z * 24.0 - 2.7) * 0.003));
  const edgeFadeX = smoothstep(0.0, bevelWidth, edgeDistX);
  const edgeFadeZ = smoothstep(0.0, bevelWidth, edgeDistZ);
  const edgeFade = edgeFadeX * edgeFadeZ;
  const edgeRecession = Math.max(
    0.0011,
    Math.min(0.0019, 0.0015 + fbm2(x * 28.0 + 7.1, z * 28.0 - 3.9) * 0.0008),
  );
  h = h * edgeFade - edgeRecession * (1 - edgeFade);

  // Taper cleanly to y = 0 at fruiting body bases so stipes never float or bury
  for (let i = 0; i < bodies.length; i++) {
    const b = bodies[i];
    const dist = Math.hypot(x - b.baseX, z - b.baseZ);
    if (dist < 0.026) {
      h *= smoothstep(0.005, 0.026, dist);
    }
  }
  return h;
}

export function evalSoilSurfaceNormal(
  x: number,
  z: number,
  soil: ISoilGrid,
  bodies: readonly FruitingBodyState[],
): [number, number, number] {
  const eps = 0.004;
  const hx0 = evalSoilSurfaceHeightM(x - eps, z, soil, bodies);
  const hx1 = evalSoilSurfaceHeightM(x + eps, z, soil, bodies);
  const hz0 = evalSoilSurfaceHeightM(x, z - eps, soil, bodies);
  const hz1 = evalSoilSurfaceHeightM(x, z + eps, soil, bodies);
  const dx = (hx1 - hx0) / (2 * eps);
  const dz = (hz1 - hz0) / (2 * eps);
  const nx = -dx;
  const ny = 1;
  const nz = -dz;
  const inv = 1 / Math.hypot(nx, ny, nz);
  return [nx * inv, ny * inv, nz * inv];
}
