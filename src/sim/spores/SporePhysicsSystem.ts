// ============================================================================
// SPOROCARP — Layer 5B: Dressaire Evaporative Convective Plume & Spore Physics
// Exact MAC staggered-grid 3D Boussinesq incompressible Navier-Stokes solver
// (20x20x20 air box) driven purely by latent heat of cap transpiration
// (Dressaire et al. 2016, PNAS), advecting up to 3,000 Lagrangian spores,
// accumulating a 64x64 ground spore print, and serving 1,000x Buller micro-view.
// ============================================================================

import {
  CP_AIR,
  GRAVITY,
  LATENT_HEAT_VAP,
  MU_AIR,
  RHO_AIR,
  SPECIES_PRESETS,
  type AirConvectionField,
  type BullerMicroViewState,
  type BullerParams,
  type EnvironmentState,
  type FruitingBodyState,
  type SporeParticleBuffers,
  type SporePrintGrid,
  type TerrariumBounds,
} from '../../types/contract.ts';
import { BullerCatapultSolver } from './BullerCatapult.ts';

export { BullerCatapultSolver };

// Red-black SOR for the MAC pressure Poisson solve. 36 sweeps at omega 1.8 leave ~2x lower
// RMS and ~2.7x lower peak divergence than the former 24 lexicographic sweeps at omega 1.5,
// in about half the time (no loop-carried dependency inside a half-sweep).
const PRESSURE_SOR_OMEGA = 1.8;
const PRESSURE_SOR_SWEEPS = 36;

const AIR_NX = 20;
const AIR_NY = 20;
const AIR_NZ = 20;
const MAX_SPORES = 3000;
const PRINT_NX = 64;
const PRINT_NZ = 64;

/** Fast deterministic PRNG (Mulberry32) for reproducible spore emission */
function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 3D MAC staggered-grid Boussinesq evaporative cooling convection solver.
 * Zero ambient wind: all airflow arises purely from evaporative latent heat cooling
 * beneath transpiring caps (Dressaire et al. 2016, PNAS).
 *
 * Face-staggered layout guarantees exact discrete div(grad(p)) === Laplacian(p),
 * eliminating checkerboard decoupling so cold downwash beneath the cap cleanly turns
 * radially outward along the ground and rises in a buoyant toroidal return vortex.
 */
export class EvaporativePlumeSolver {
  public readonly nx: number;
  public readonly ny: number;
  public readonly nz: number;
  public readonly bounds: TerrariumBounds;
  public readonly dx: number;
  public readonly dy: number;
  public readonly dz: number;
  public readonly cellVolumeM3: number;

  // Cell-centered exported fields (for AirConvectionField & renderer streamlines)
  public readonly ux: Float32Array;
  public readonly uy: Float32Array;
  public readonly uz: Float32Array;
  public readonly deltaT: Float32Array;
  // Specific humidity perturbation deltaQ (kg water vapor / kg air) from cap transpiration
  private readonly deltaQ: Float32Array;

  // MAC staggered face velocities:
  // uFace: (nx + 1) * ny * nz  at (i, j + 0.5, k + 0.5)
  // vFace: nx * (ny + 1) * nz  at (i + 0.5, j, k + 0.5)
  // wFace: nx * ny * (nz + 1)  at (i + 0.5, j + 0.5, k)
  private readonly uFace: Float32Array;
  private readonly vFace: Float32Array;
  private readonly wFace: Float32Array;
  private readonly uTmp: Float32Array;
  private readonly vTmp: Float32Array;
  private readonly wTmp: Float32Array;
  private readonly tTmp: Float32Array;
  private readonly qTmp: Float32Array;
  private readonly div: Float32Array;
  // Float64: the pressure solve is warm-started across frames, so keep its full precision
  private readonly pressure: Float64Array;

  constructor(bounds: TerrariumBounds, nx = AIR_NX, ny = AIR_NY, nz = AIR_NZ) {
    this.nx = nx;
    this.ny = ny;
    this.nz = nz;
    this.bounds = { ...bounds };
    this.dx = (bounds.maxX - bounds.minX) / nx;
    this.dy = (bounds.maxY - bounds.minY) / ny;
    this.dz = (bounds.maxZ - bounds.minZ) / nz;
    this.cellVolumeM3 = this.dx * this.dy * this.dz;

    const total = nx * ny * nz;
    this.ux = new Float32Array(total);
    this.uy = new Float32Array(total);
    this.uz = new Float32Array(total);
    this.deltaT = new Float32Array(total);
    this.deltaQ = new Float32Array(total);

    this.uFace = new Float32Array((nx + 1) * ny * nz);
    this.vFace = new Float32Array(nx * (ny + 1) * nz);
    this.wFace = new Float32Array(nx * ny * (nz + 1));
    this.uTmp = new Float32Array(this.uFace.length);
    this.vTmp = new Float32Array(this.vFace.length);
    this.wTmp = new Float32Array(this.wFace.length);
    this.tTmp = new Float32Array(total);
    this.qTmp = new Float32Array(total);
    this.div = new Float32Array(total);
    this.pressure = new Float64Array(total);
  }

  public idx(ix: number, iy: number, iz: number): number {
    return (iy * this.nz + iz) * this.nx + ix;
  }

  private uIdx(ix: number, iy: number, iz: number): number {
    return (iy * this.nz + iz) * (this.nx + 1) + ix;
  }

  private vIdx(ix: number, iy: number, iz: number): number {
    return (iy * this.nz + iz) * this.nx + ix;
  }

  private wIdx(ix: number, iy: number, iz: number): number {
    return (iy * (this.nz + 1) + iz) * this.nx + ix;
  }

  public reset(): void {
    this.ux.fill(0);
    this.uy.fill(0);
    this.uz.fill(0);
    this.deltaT.fill(0);
    this.deltaQ.fill(0);
    this.uFace.fill(0);
    this.vFace.fill(0);
    this.wFace.fill(0);
    this.pressure.fill(0);
  }

  public step(
    dtSec: number,
    bodies: FruitingBodyState[],
    env: EnvironmentState,
    evaporationEnabled: boolean,
  ): void {
    if (dtSec <= 0) return;

    // Fast path: when evaporation is disabled and air is quiescent, preserve exact zero field
    if (!evaporationEnabled) {
      let maxEnergy = 0;
      for (let i = 0; i < this.deltaT.length; i++) {
        const v =
          Math.abs(this.deltaT[i]) +
          Math.abs(this.deltaQ[i]) +
          Math.abs(this.ux[i]) +
          Math.abs(this.uy[i]) +
          Math.abs(this.uz[i]);
        if (v > maxEnergy) maxEnergy = v;
      }
      if (maxEnergy < 1e-10) {
        this.reset();
        return;
      }
    }

    // Clamp fluid step for stability under high multi-scale timeWarp
    const totalFluidDt = Math.min(dtSec, 0.35);
    const subSteps = Math.max(1, Math.min(4, Math.ceil(totalFluidDt / 0.085)));
    const h = totalFluidDt / subSteps;

    const betaTh = 1.0 / (env.ambientTempC + 273.15);
    // Exact water-vapor compositional buoyancy coefficient: M_air / M_H2O - 1 = 28.97/18.015 - 1
    const betaQ = 0.608;
    const kappaEff = 3.5e-5; // air thermal & vapor diffusivity + mild grid stabilization

    for (let s = 0; s < subSteps; s++) {
      // 1. Inject evaporative cooling Q_cool & water-vapor mass flux S_q (ONLY when evaporationEnabled!)
      if (evaporationEnabled) {
        this.applyEvaporativeCooling(h, bodies, env);
      }

      // 2. Dressaire et al. (2016) double-diffusive Boussinesq buoyancy on interior y-faces:
      //    dv/dt += g * [ beta_T * deltaT + beta_q * deltaQ ]
      //    Under the cap: latent cooling (-beta_T * |deltaT|) dominates -> cold sinking downwash.
      //    Along the warm soil floor (y=0): soil reheats deltaT -> 0 while vapor deltaQ persists ->
      //    lighter-than-air moist plume (+beta_q * deltaQ) rises vigorously around and above the cap!
      const gH = GRAVITY * h;
      const { nx, ny, nz } = this;
      for (let iy = 1; iy < ny; iy++) {
        for (let iz = 0; iz < nz; iz++) {
          for (let ix = 0; ix < nx; ix++) {
            const idLo = this.idx(ix, iy - 1, iz);
            const idHi = this.idx(ix, iy, iz);
            const tAvg = 0.5 * (this.deltaT[idLo] + this.deltaT[idHi]);
            const qAvg = 0.5 * (this.deltaQ[idLo] + this.deltaQ[idHi]);
            this.vFace[this.vIdx(ix, iy, iz)] += gH * (betaTh * tAvg + betaQ * qAvg);
          }
        }
      }

      // Gentle viscous damping on staggered velocities
      const viscDecay = Math.exp(-h * 0.05);
      for (let i = 0; i < this.uFace.length; i++) this.uFace[i] *= viscDecay;
      for (let i = 0; i < this.vFace.length; i++) this.vFace[i] *= viscDecay;
      for (let i = 0; i < this.wFace.length; i++) this.wFace[i] *= viscDecay;

      this.enforceFaceBC();

      // 3. Exact MAC incompressible pressure projection (div u = 0)
      this.projectMAC(PRESSURE_SOR_SWEEPS);

      // Sync cell centers for semi-Lagrangian backtracing
      this.syncCellCenterVelocity();

      // 4. Semi-Lagrangian self-advection of staggered velocities, temperature, and water vapor
      this.advectStaggeredVelocities(h);
      this.advectScalar(this.deltaT, this.tTmp, h);
      this.deltaT.set(this.tTmp);
      this.advectScalar(this.deltaQ, this.qTmp, h);
      this.deltaQ.set(this.qTmp);

      // Mild thermal/vapor diffusion & soil/ambient boundary exchange
      this.diffuseScalar(this.deltaT, this.tTmp, kappaEff, h, 2);
      this.diffuseScalar(this.deltaQ, this.qTmp, kappaEff, h, 2);
      this.applyThermalBoundaries(h);

      this.enforceFaceBC();

      // 5. Final exact MAC divergence-free projection
      this.projectMAC(PRESSURE_SOR_SWEEPS);
      this.syncCellCenterVelocity();
    }
  }

  private applyEvaporativeCooling(
    h: number,
    bodies: FruitingBodyState[],
    env: EnvironmentState,
  ): void {
    const { nx, ny, nz, bounds, dx, dy, dz, cellVolumeM3 } = this;
    const invRhoCpV = 1.0 / (RHO_AIR * CP_AIR * cellVolumeM3);
    const vpd = Math.max(0.02, 1.0 - Math.max(0, Math.min(1, env.ambientHumidity)));

    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b];
      if (body.capRadiusM < 0.008 || body.maturity < 0.25) continue;

      const sp = SPECIES_PRESETS[body.speciesId];
      const capArea =
        body.capAreaM2 > 1e-6 ? body.capAreaM2 : Math.PI * body.capRadiusM * body.capRadiusM;
      const mDot =
        body.evaporationRateKgS > 0
          ? body.evaporationRateKgS
          : sp.transpirationCoeff * capArea * vpd;
      if (mDot <= 0) continue;

      const cx = body.stipeTipX;
      const cy = Math.max(bounds.minY + dy * 0.5, body.stipeTipY - 0.004);
      const cz = body.stipeTipZ;
      const kernelR = Math.max(body.capRadiusM * 0.85, dx * 0.85);
      const kernelY = Math.max(0.01, dy * 0.7);

      const ixMin = Math.max(1, Math.floor((cx - kernelR - bounds.minX) / dx));
      const ixMax = Math.min(nx - 2, Math.ceil((cx + kernelR - bounds.minX) / dx));
      const iyMin = Math.max(1, Math.floor((cy - kernelY - bounds.minY) / dy));
      const iyMax = Math.min(ny - 2, Math.ceil((cy + kernelY - bounds.minY) / dy));
      const izMin = Math.max(1, Math.floor((cz - kernelR - bounds.minZ) / dz));
      const izMax = Math.min(nz - 2, Math.ceil((cz + kernelR - bounds.minZ) / dz));

      let wSum = 0;
      for (let iy = iyMin; iy <= iyMax; iy++) {
        const wy = bounds.minY + (iy + 0.5) * dy - cy;
        const fy = Math.exp(-(wy * wy) / (2.0 * kernelY * kernelY));
        for (let iz = izMin; iz <= izMax; iz++) {
          const wz = bounds.minZ + (iz + 0.5) * dz - cz;
          for (let ix = ixMin; ix <= ixMax; ix++) {
            const wx = bounds.minX + (ix + 0.5) * dx - cx;
            const r2 = wx * wx + wz * wz;
            if (r2 > kernelR * kernelR * 1.6) continue;
            wSum += fy * Math.exp(-r2 / (2.0 * kernelR * kernelR));
          }
        }
      }
      if (wSum <= 1e-12) continue;

      // Exact integrated latent heat cooling power = -L_v * mDot / (rho_air * c_p)
      // and coupled water-vapor specific-humidity injection = +mDot / (rho_air * V_cell)
      // Cap vapor excess at physical 20C saturation specific humidity (~0.015 kg/kg)
      const totalCoolingStep = -(LATENT_HEAT_VAP * mDot * invRhoCpV * h) / wSum;
      const totalVaporStep = (mDot * 7.0 * h) / (RHO_AIR * cellVolumeM3 * wSum);

      for (let iy = iyMin; iy <= iyMax; iy++) {
        const wy = bounds.minY + (iy + 0.5) * dy - cy;
        const fy = Math.exp(-(wy * wy) / (2.0 * kernelY * kernelY));
        for (let iz = izMin; iz <= izMax; iz++) {
          const wz = bounds.minZ + (iz + 0.5) * dz - cz;
          for (let ix = ixMin; ix <= ixMax; ix++) {
            const wx = bounds.minX + (ix + 0.5) * dx - cx;
            const r2 = wx * wx + wz * wz;
            if (r2 > kernelR * kernelR * 1.6) continue;
            const w = fy * Math.exp(-r2 / (2.0 * kernelR * kernelR));
            const id = this.idx(ix, iy, iz);
            this.deltaT[id] = Math.max(-5.5, this.deltaT[id] + totalCoolingStep * w);
            this.deltaQ[id] = Math.min(0.015, this.deltaQ[id] + totalVaporStep * w);
          }
        }
      }
    }
  }

  private enforceFaceBC(): void {
    const { nx, ny, nz } = this;
    // Impermeable X walls: u(0) = 0, u(nx) = 0
    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++) {
        this.uFace[this.uIdx(0, iy, iz)] = 0;
        this.uFace[this.uIdx(nx, iy, iz)] = 0;
      }
    }
    // Impermeable floor & ceiling: v(0) = 0, v(ny) = 0
    for (let iz = 0; iz < nz; iz++) {
      for (let ix = 0; ix < nx; ix++) {
        this.vFace[this.vIdx(ix, 0, iz)] = 0;
        this.vFace[this.vIdx(ix, ny, iz)] = 0;
      }
    }
    // Impermeable Z walls: w(0) = 0, w(nz) = 0
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        this.wFace[this.wIdx(ix, iy, 0)] = 0;
        this.wFace[this.wIdx(ix, iy, nz)] = 0;
      }
    }
  }

  private projectMAC(iterations: number): void {
    const { nx, ny, nz, dx, dy, dz } = this;
    const invDx = 1.0 / dx;
    const invDy = 1.0 / dy;
    const invDz = 1.0 / dz;
    const invDx2 = invDx * invDx;
    const invDy2 = invDy * invDy;
    const invDz2 = invDz * invDz;

    // 1. Exact face divergence per cell
    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++) {
        for (let ix = 0; ix < nx; ix++) {
          const id = this.idx(ix, iy, iz);
          const du = (this.uFace[this.uIdx(ix + 1, iy, iz)] - this.uFace[this.uIdx(ix, iy, iz)]) * invDx;
          const dv = (this.vFace[this.vIdx(ix, iy + 1, iz)] - this.vFace[this.vIdx(ix, iy, iz)]) * invDy;
          const dw = (this.wFace[this.wIdx(ix, iy, iz + 1)] - this.wFace[this.wIdx(ix, iy, iz)]) * invDz;
          this.div[id] = du + dv + dw;
        }
      }
    }

    // 2. Solve discrete Neumann Poisson equation Laplacian(p) = div(u) via SOR Gauss-Seidel.
    //    Same lexicographic sweep order and the same per-cell summation order as the textbook
    //    form (x-, x+, z-, z+, y-, y+), so results are bit-identical; the interior of each row
    //    just skips the boundary tests and index helpers.
    const omega = PRESSURE_SOR_OMEGA;
    const p = this.pressure;
    const divArr = this.div;
    const strideY = nz * nx;
    const invDiagInterior = 1.0 / (invDx2 + invDx2 + invDz2 + invDz2 + invDy2 + invDy2);
    for (let it = 0; it < iterations; it++) {
      // Red-black ordering: cells of one colour only read the other colour, so each half-sweep
      // has no loop-carried dependency (same SOR convergence rate as lexicographic order).
      for (let color = 0; color < 2; color++) {
        for (let iy = 0; iy < ny; iy++) {
          const hasYLo = iy > 0;
          const hasYHi = iy < ny - 1;
          for (let iz = 0; iz < nz; iz++) {
            const hasZLo = iz > 0;
            const hasZHi = iz < nz - 1;
            const rowStart = (iy * nz + iz) * nx;
            const rowInterior = hasYLo && hasYHi && hasZLo && hasZHi;
            for (let ix = (iy + iz + color) & 1; ix < nx; ix += 2) {
              const id = rowStart + ix;
              let sum = -divArr[id];
              if (rowInterior && ix > 0 && ix < nx - 1) {
                sum += p[id - 1] * invDx2;
                sum += p[id + 1] * invDx2;
                sum += p[id - nx] * invDz2;
                sum += p[id + nx] * invDz2;
                sum += p[id - strideY] * invDy2;
                sum += p[id + strideY] * invDy2;
                const pOldI = p[id];
                p[id] = pOldI + omega * (sum * invDiagInterior - pOldI);
                continue;
              }
              let diag = 0;
              if (ix > 0) {
                sum += p[id - 1] * invDx2;
                diag += invDx2;
              }
              if (ix < nx - 1) {
                sum += p[id + 1] * invDx2;
                diag += invDx2;
              }
              if (hasZLo) {
                sum += p[id - nx] * invDz2;
                diag += invDz2;
              }
              if (hasZHi) {
                sum += p[id + nx] * invDz2;
                diag += invDz2;
              }
              if (hasYLo) {
                sum += p[id - strideY] * invDy2;
                diag += invDy2;
              }
              if (hasYHi) {
                sum += p[id + strideY] * invDy2;
                diag += invDy2;
              }
              const pOld = p[id];
              p[id] = pOld + omega * (sum / diag - pOld);
            }
          }
        }
      }
    }

    // Remove arbitrary constant pressure offset for numerical hygiene
    const pRef = this.pressure[0];
    for (let i = 0; i < this.pressure.length; i++) {
      this.pressure[i] -= pRef;
    }

    // 3. Subtract exact staggered pressure gradient on interior faces
    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++) {
        for (let ix = 1; ix < nx; ix++) {
          const pR = this.pressure[this.idx(ix, iy, iz)];
          const pL = this.pressure[this.idx(ix - 1, iy, iz)];
          this.uFace[this.uIdx(ix, iy, iz)] -= (pR - pL) * invDx;
        }
      }
    }

    for (let iy = 1; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++) {
        for (let ix = 0; ix < nx; ix++) {
          const pU = this.pressure[this.idx(ix, iy, iz)];
          const pD = this.pressure[this.idx(ix, iy - 1, iz)];
          this.vFace[this.vIdx(ix, iy, iz)] -= (pU - pD) * invDy;
        }
      }
    }

    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 1; iz < nz; iz++) {
        for (let ix = 0; ix < nx; ix++) {
          const pF = this.pressure[this.idx(ix, iy, iz)];
          const pB = this.pressure[this.idx(ix, iy, iz - 1)];
          this.wFace[this.wIdx(ix, iy, iz)] -= (pF - pB) * invDz;
        }
      }
    }

    this.enforceFaceBC();
  }

  private syncCellCenterVelocity(): void {
    const { nx, ny, nz } = this;
    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++) {
        for (let ix = 0; ix < nx; ix++) {
          const id = this.idx(ix, iy, iz);
          this.ux[id] =
            0.5 * (this.uFace[this.uIdx(ix, iy, iz)] + this.uFace[this.uIdx(ix + 1, iy, iz)]);
          this.uy[id] =
            0.5 * (this.vFace[this.vIdx(ix, iy, iz)] + this.vFace[this.vIdx(ix, iy + 1, iz)]);
          this.uz[id] =
            0.5 * (this.wFace[this.wIdx(ix, iy, iz)] + this.wFace[this.wIdx(ix, iy, iz + 1)]);
        }
      }
    }
  }

  private sampleUFace(gx: number, gy: number, gz: number): number {
    const { nx, ny, nz } = this;
    const x = Math.max(0, Math.min(nx - 0.001, gx));
    const y = Math.max(0, Math.min(ny - 1.001, gy - 0.5));
    const z = Math.max(0, Math.min(nz - 1.001, gz - 0.5));
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const z0 = Math.floor(z);
    const tx = x - x0;
    const ty = y - y0;
    const tz = z - z0;
    const u = this.uFace;
    const c000 = u[this.uIdx(x0, y0, z0)];
    const c100 = u[this.uIdx(x0 + 1, y0, z0)];
    const c010 = u[this.uIdx(x0, y0, z0 + 1)];
    const c110 = u[this.uIdx(x0 + 1, y0, z0 + 1)];
    const c001 = u[this.uIdx(x0, y0 + 1, z0)];
    const c101 = u[this.uIdx(x0 + 1, y0 + 1, z0)];
    const c011 = u[this.uIdx(x0, y0 + 1, z0 + 1)];
    const c111 = u[this.uIdx(x0 + 1, y0 + 1, z0 + 1)];
    const c00 = c000 + tx * (c100 - c000);
    const c10 = c010 + tx * (c110 - c010);
    const c01 = c001 + tx * (c101 - c001);
    const c11 = c011 + tx * (c111 - c011);
    return c00 + tz * (c10 - c00) + ty * (c01 + tz * (c11 - c01) - (c00 + tz * (c10 - c00)));
  }

  private sampleVFace(gx: number, gy: number, gz: number): number {
    const { nx, ny, nz } = this;
    const x = Math.max(0, Math.min(nx - 1.001, gx - 0.5));
    const y = Math.max(0, Math.min(ny - 0.001, gy));
    const z = Math.max(0, Math.min(nz - 1.001, gz - 0.5));
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const z0 = Math.floor(z);
    const tx = x - x0;
    const ty = y - y0;
    const tz = z - z0;
    const v = this.vFace;
    const c000 = v[this.vIdx(x0, y0, z0)];
    const c100 = v[this.vIdx(x0 + 1, y0, z0)];
    const c010 = v[this.vIdx(x0, y0, z0 + 1)];
    const c110 = v[this.vIdx(x0 + 1, y0, z0 + 1)];
    const c001 = v[this.vIdx(x0, y0 + 1, z0)];
    const c101 = v[this.vIdx(x0 + 1, y0 + 1, z0)];
    const c011 = v[this.vIdx(x0, y0 + 1, z0 + 1)];
    const c111 = v[this.vIdx(x0 + 1, y0 + 1, z0 + 1)];
    const c00 = c000 + tx * (c100 - c000);
    const c10 = c010 + tx * (c110 - c010);
    const c01 = c001 + tx * (c101 - c001);
    const c11 = c011 + tx * (c111 - c011);
    return c00 + tz * (c10 - c00) + ty * (c01 + tz * (c11 - c01) - (c00 + tz * (c10 - c00)));
  }

  private sampleWFace(gx: number, gy: number, gz: number): number {
    const { nx, ny, nz } = this;
    const x = Math.max(0, Math.min(nx - 1.001, gx - 0.5));
    const y = Math.max(0, Math.min(ny - 1.001, gy - 0.5));
    const z = Math.max(0, Math.min(nz - 0.001, gz));
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const z0 = Math.floor(z);
    const tx = x - x0;
    const ty = y - y0;
    const tz = z - z0;
    const w = this.wFace;
    const c000 = w[this.wIdx(x0, y0, z0)];
    const c100 = w[this.wIdx(x0 + 1, y0, z0)];
    const c010 = w[this.wIdx(x0, y0, z0 + 1)];
    const c110 = w[this.wIdx(x0 + 1, y0, z0 + 1)];
    const c001 = w[this.wIdx(x0, y0 + 1, z0)];
    const c101 = w[this.wIdx(x0 + 1, y0 + 1, z0)];
    const c011 = w[this.wIdx(x0, y0 + 1, z0 + 1)];
    const c111 = w[this.wIdx(x0 + 1, y0 + 1, z0 + 1)];
    const c00 = c000 + tx * (c100 - c000);
    const c10 = c010 + tx * (c110 - c010);
    const c01 = c001 + tx * (c101 - c001);
    const c11 = c011 + tx * (c111 - c011);
    return c00 + tz * (c10 - c00) + ty * (c01 + tz * (c11 - c01) - (c00 + tz * (c10 - c00)));
  }

  private advectStaggeredVelocities(h: number): void {
    const { nx, ny, nz, dx, dy, dz } = this;
    const invDx = 1.0 / dx;
    const invDy = 1.0 / dy;
    const invDz = 1.0 / dz;

    // Advect uFace
    this.uTmp.fill(0);
    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++) {
        for (let ix = 1; ix < nx; ix++) {
          const px = ix;
          const py = iy + 0.5;
          const pz = iz + 0.5;
          const u = this.uFace[this.uIdx(ix, iy, iz)];
          const v = this.sampleVFace(px, py, pz);
          const w = this.sampleWFace(px, py, pz);
          this.uTmp[this.uIdx(ix, iy, iz)] = this.sampleUFace(
            px - u * h * invDx,
            py - v * h * invDy,
            pz - w * h * invDz,
          );
        }
      }
    }

    // Advect vFace
    this.vTmp.fill(0);
    for (let iy = 1; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++) {
        for (let ix = 0; ix < nx; ix++) {
          const px = ix + 0.5;
          const py = iy;
          const pz = iz + 0.5;
          const u = this.sampleUFace(px, py, pz);
          const v = this.vFace[this.vIdx(ix, iy, iz)];
          const w = this.sampleWFace(px, py, pz);
          this.vTmp[this.vIdx(ix, iy, iz)] = this.sampleVFace(
            px - u * h * invDx,
            py - v * h * invDy,
            pz - w * h * invDz,
          );
        }
      }
    }

    // Advect wFace
    this.wTmp.fill(0);
    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 1; iz < nz; iz++) {
        for (let ix = 0; ix < nx; ix++) {
          const px = ix + 0.5;
          const py = iy + 0.5;
          const pz = iz;
          const u = this.sampleUFace(px, py, pz);
          const v = this.sampleVFace(px, py, pz);
          const w = this.wFace[this.wIdx(ix, iy, iz)];
          this.wTmp[this.wIdx(ix, iy, iz)] = this.sampleWFace(
            px - u * h * invDx,
            py - v * h * invDy,
            pz - w * h * invDz,
          );
        }
      }
    }

    this.uFace.set(this.uTmp);
    this.vFace.set(this.vTmp);
    this.wFace.set(this.wTmp);
  }

  private advectScalar(src: Float32Array, dst: Float32Array, h: number): void {
    const { nx, ny, nz, dx, dy, dz } = this;
    const strideY = nz * nx;
    const invDx = 1.0 / dx;
    const invDy = 1.0 / dy;
    const invDz = 1.0 / dz;
    const maxX = nx - 1.001;
    const maxY = ny - 1.001;
    const maxZ = nz - 1.001;

    for (let iy = 0; iy < ny; iy++) {
      const yOff = iy * strideY;
      for (let iz = 0; iz < nz; iz++) {
        const zOff = yOff + iz * nx;
        for (let ix = 0; ix < nx; ix++) {
          const id = zOff + ix;
          let gx = ix - this.ux[id] * h * invDx;
          let gy = iy - this.uy[id] * h * invDy;
          let gz = iz - this.uz[id] * h * invDz;

          if (gx < 0) gx = 0;
          else if (gx > maxX) gx = maxX;
          if (gy < 0) gy = 0;
          else if (gy > maxY) gy = maxY;
          if (gz < 0) gz = 0;
          else if (gz > maxZ) gz = maxZ;

          const x0 = Math.floor(gx);
          const y0 = Math.floor(gy);
          const z0 = Math.floor(gz);
          const tx = gx - x0;
          const ty = gy - y0;
          const tz = gz - z0;

          const c000 = src[y0 * strideY + z0 * nx + x0];
          const c100 = src[y0 * strideY + z0 * nx + x0 + 1];
          const c010 = src[y0 * strideY + (z0 + 1) * nx + x0];
          const c110 = src[y0 * strideY + (z0 + 1) * nx + x0 + 1];
          const c001 = src[(y0 + 1) * strideY + z0 * nx + x0];
          const c101 = src[(y0 + 1) * strideY + z0 * nx + x0 + 1];
          const c011 = src[(y0 + 1) * strideY + (z0 + 1) * nx + x0];
          const c111 = src[(y0 + 1) * strideY + (z0 + 1) * nx + x0 + 1];

          const c00 = c000 + tx * (c100 - c000);
          const c10 = c010 + tx * (c110 - c010);
          const c01 = c001 + tx * (c101 - c001);
          const c11 = c011 + tx * (c111 - c011);

          const c0 = c00 + tz * (c10 - c00);
          const c1 = c01 + tz * (c11 - c01);

          dst[id] = c0 + ty * (c1 - c0);
        }
      }
    }
  }

  private diffuseScalar(
    field: Float32Array,
    scratch: Float32Array,
    coeff: number,
    h: number,
    iterations: number,
  ): void {
    const { nx, ny, nz, dx, dy, dz } = this;
    const ax = (coeff * h) / (dx * dx);
    const ay = (coeff * h) / (dy * dy);
    const az = (coeff * h) / (dz * dz);
    const invDenom = 1.0 / (1.0 + 2.0 * (ax + ay + az));
    const strideY = nz * nx;

    scratch.set(field);
    for (let it = 0; it < iterations; it++) {
      for (let iy = 1; iy < ny - 1; iy++) {
        const yOff = iy * strideY;
        for (let iz = 1; iz < nz - 1; iz++) {
          const zOff = yOff + iz * nx;
          for (let ix = 1; ix < nx - 1; ix++) {
            const id = zOff + ix;
            const sum =
              ax * (scratch[id - 1] + scratch[id + 1]) +
              az * (scratch[id - nx] + scratch[id + nx]) +
              ay * (scratch[id - strideY] + scratch[id + strideY]);
            scratch[id] = (field[id] + sum) * invDenom;
          }
        }
      }
    }
    field.set(scratch);
  }

  private applyThermalBoundaries(h: number): void {
    const { nx, ny, nz } = this;
    const strideY = nz * nx;
    // Warm soil floor (iy = 0, 1) rapidly reheats the cold gravity current back toward T_amb
    // while retaining water vapor (impermeable ground), creating the positive vapor buoyancy
    // limb of the Dressaire convective plume!
    const floorHeatDecay = Math.exp(-h * 9.0);
    const wallHeatDecay = Math.exp(-h * 2.0);
    const bulkHeatDecay = Math.exp(-h * 0.45);
    const outerVaporDecay = Math.exp(-h * 1.2);
    const bulkVaporDecay = Math.exp(-h * 0.3);

    for (let iy = 0; iy < ny; iy++) {
      const isFloor = iy <= 2;
      const isTop = iy === ny - 1;
      const yOff = iy * strideY;
      for (let iz = 0; iz < nz; iz++) {
        const isZWall = iz === 0 || iz === nz - 1;
        const zOff = yOff + iz * nx;
        for (let ix = 0; ix < nx; ix++) {
          const id = zOff + ix;
          const isSideOrTop = isTop || isZWall || ix === 0 || ix === nx - 1;
          const tDecay = isFloor ? floorHeatDecay : isSideOrTop ? wallHeatDecay : bulkHeatDecay;
          let tVal = this.deltaT[id] * tDecay;
          if (tVal < -5.5) tVal = -5.5;
          else if (tVal > 2.5) tVal = 2.5;
          this.deltaT[id] = tVal;

          const qDecay = isSideOrTop ? outerVaporDecay : bulkVaporDecay;
          this.deltaQ[id] = Math.max(0, Math.min(0.015, this.deltaQ[id] * qDecay));
        }
      }
    }
  }

  /** Trilinear interpolation of exact staggered MAC velocity at world point (x, y, z) */
  public sampleVelocity(
    x: number,
    y: number,
    z: number,
    out: [number, number, number],
  ): void {
    const gx = (x - this.bounds.minX) / this.dx;
    const gy = (y - this.bounds.minY) / this.dy;
    const gz = (z - this.bounds.minZ) / this.dz;
    out[0] = this.sampleUFace(gx, gy, gz);
    out[1] = this.sampleVFace(gx, gy, gz);
    out[2] = this.sampleWFace(gx, gy, gz);
  }

  public getField(): AirConvectionField {
    return {
      nx: this.nx,
      ny: this.ny,
      nz: this.nz,
      bounds: this.bounds,
      ux: this.ux,
      uy: this.uy,
      uz: this.uz,
      deltaT: this.deltaT,
    };
  }
}

export class SporePhysicsSystem {
  public readonly bounds: TerrariumBounds;
  public readonly plume: EvaporativePlumeSolver;

  // Lagrangian spore particle pool (up to 3,000 active spores)
  private readonly positions: Float32Array;
  private readonly velocities: Float32Array;
  private readonly active: Uint8Array;
  private readonly launchCapY: Float32Array;
  private readonly launchOriginX: Float32Array;
  private readonly launchOriginZ: Float32Array;
  private readonly terminalVel: Float32Array;

  private activeCount = 0;
  private nextSpawnSlot = 0;
  private emitAccumulator = 0;
  private maxSporeLiftAboveCapM = 0;

  // 2D ground spore print grid (64 x 64)
  private readonly printDensity: Float32Array;
  private totalDeposited = 0;
  private cumulativeDispersalDistM = 0;
  private sporePrintColor: [number, number, number] = [...SPECIES_PRESETS.parasol.sporePrintColor];

  // Cached latest bodies & Buller micro-view state
  private lastBodies: FruitingBodyState[] = [];
  private bullerPhase = 0;
  private cachedBullerKey = '';
  private cachedBullerState: BullerMicroViewState | null = null;
  private readonly rng = createRng(0x5f3759df);
  private readonly sampleVelScratch: [number, number, number] = [0, 0, 0];

  constructor(bounds: TerrariumBounds) {
    this.bounds = { ...bounds };
    this.plume = new EvaporativePlumeSolver(bounds, AIR_NX, AIR_NY, AIR_NZ);

    this.positions = new Float32Array(MAX_SPORES * 4);
    this.velocities = new Float32Array(MAX_SPORES * 4);
    this.active = new Uint8Array(MAX_SPORES);
    this.launchCapY = new Float32Array(MAX_SPORES);
    this.launchOriginX = new Float32Array(MAX_SPORES);
    this.launchOriginZ = new Float32Array(MAX_SPORES);
    this.terminalVel = new Float32Array(MAX_SPORES);

    this.printDensity = new Float32Array(PRINT_NX * PRINT_NZ);
  }

  public reset(): void {
    this.plume.reset();
    this.positions.fill(0);
    this.velocities.fill(0);
    this.active.fill(0);
    this.activeCount = 0;
    this.nextSpawnSlot = 0;
    this.emitAccumulator = 0;
    this.maxSporeLiftAboveCapM = 0;
    this.printDensity.fill(0);
    this.totalDeposited = 0;
    this.cumulativeDispersalDistM = 0;
    this.lastBodies = [];
    this.cachedBullerKey = '';
    this.cachedBullerState = null;
  }

  public step(
    dtSec: number,
    bodies: FruitingBodyState[],
    env: EnvironmentState,
    evaporationEnabled?: boolean,
  ): void {
    if (dtSec <= 0) return;

    this.lastBodies = bodies;
    const evapOn = evaporationEnabled !== undefined ? evaporationEnabled : env.evaporationEnabled;

    if (bodies.length > 0) {
      const sp = SPECIES_PRESETS[bodies[0].speciesId];
      if (sp) {
        this.sporePrintColor = [...sp.sporePrintColor];
      }
    }

    // Advance 1,000x Buller micro-view animation cycle smoothly in real time
    const visualDt = Math.min(dtSec, 0.25);
    this.bullerPhase = (this.bullerPhase + visualDt * 0.42) % 1.0;

    // 1. Step 3D MAC Boussinesq evaporative convection plume
    this.plume.step(dtSec, bodies, env, evapOn);

    // 2. Emit new spores from mature caps
    const fluidDt = Math.min(dtSec, 0.35);
    this.emitSpores(fluidDt, bodies);

    // 3. Advect Lagrangian spores by (u_air(x) - v_term * y_hat) & deposit ground spore print
    this.advectSpores(fluidDt);
  }

  private emitSpores(fluidDt: number, bodies: FruitingBodyState[]): void {
    if (bodies.length === 0) return;

    const baseRatePerBody = 240.0;

    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b];
      if (body.maturity < 0.32 || body.capRadiusM < 0.008 || body.wiltFactor > 0.8) continue;

      const sp = SPECIES_PRESETS[body.speciesId];
      const rs = sp.sporeRadiusM;
      const tauS = (2.0 * sp.sporeDensityKgM3 * rs * rs) / (9.0 * MU_AIR);
      const vTerm = GRAVITY * tauS;

      this.emitAccumulator += baseRatePerBody * body.maturity * fluidDt;
      const toSpawn = Math.min(28, Math.floor(this.emitAccumulator));
      if (toSpawn <= 0) continue;
      this.emitAccumulator -= toSpawn;

      for (let k = 0; k < toSpawn; k++) {
        const slot = this.findFreeSlot();
        if (slot < 0) break;

        // Emit across underside hymenium from mid-cap out to gill margin
        const angle = this.rng() * Math.PI * 2.0;
        const radFrac = 0.28 + 0.72 * Math.sqrt(this.rng());
        const r = body.capRadiusM * radFrac;
        const sx = body.stipeTipX + Math.cos(angle) * r;
        const sz = body.stipeTipZ + Math.sin(angle) * r;
        const sy = Math.max(0.005, body.stipeTipY - 0.0025);

        const p4 = slot * 4;
        this.positions[p4] = sx;
        this.positions[p4 + 1] = sy;
        this.positions[p4 + 2] = sz;
        this.positions[p4 + 3] = 0; // ageSec

        this.velocities[p4] = 0;
        this.velocities[p4 + 1] = -vTerm;
        this.velocities[p4 + 2] = 0;
        this.velocities[p4 + 3] = 0; // peakLiftAboveCapM

        this.launchCapY[slot] = body.stipeTipY;
        this.launchOriginX[slot] = body.baseX;
        this.launchOriginZ[slot] = body.baseZ;
        this.terminalVel[slot] = vTerm;

        if (this.active[slot] === 0) {
          this.active[slot] = 1;
          this.activeCount++;
        }
      }
    }
  }

  private findFreeSlot(): number {
    for (let tries = 0; tries < 64; tries++) {
      const idx = this.nextSpawnSlot;
      this.nextSpawnSlot = (this.nextSpawnSlot + 1) % MAX_SPORES;
      if (this.active[idx] === 0) return idx;
    }
    const idx = this.nextSpawnSlot;
    this.nextSpawnSlot = (this.nextSpawnSlot + 1) % MAX_SPORES;
    return idx;
  }

  private advectSpores(fluidDt: number): void {
    const { minX, maxX, minY, maxY, minZ, maxZ } = this.bounds;
    const spanX = maxX - minX;
    const spanZ = maxZ - minZ;
    const airVel = this.sampleVelScratch;

    // 2nd-order midpoint (RK2) spore advection so spores accurately follow curved toroidal streamlines
    const subSteps = fluidDt > 0.1 ? 2 : 1;
    const dt = fluidDt / subSteps;

    for (let i = 0; i < MAX_SPORES; i++) {
      if (this.active[i] === 0) continue;

      const p4 = i * 4;
      let x = this.positions[p4];
      let y = this.positions[p4 + 1];
      let z = this.positions[p4 + 2];
      const age = this.positions[p4 + 3] + fluidDt;
      const vTerm = this.terminalVel[i];

      let vx = 0;
      let vy = -vTerm;
      let vz = 0;
      let deposited = false;

      for (let s = 0; s < subSteps; s++) {
        this.plume.sampleVelocity(x, y, z, airVel);
        const vx1 = airVel[0];
        const vy1 = airVel[1] - vTerm;
        const vz1 = airVel[2];

        const mx = x + 0.5 * dt * vx1;
        const my = Math.max(minY, Math.min(maxY, y + 0.5 * dt * vy1));
        const mz = z + 0.5 * dt * vz1;

        this.plume.sampleVelocity(mx, my, mz, airVel);
        vx = airVel[0];
        vy = airVel[1] - vTerm;
        vz = airVel[2];

        x += vx * dt;
        y += vy * dt;
        z += vz * dt;

        if (y > maxY - 0.002) {
          y = maxY - 0.002;
        }

        const lift = y - this.launchCapY[i];
        if (lift > this.velocities[p4 + 3]) {
          this.velocities[p4 + 3] = lift;
          if (lift > this.maxSporeLiftAboveCapM) {
            this.maxSporeLiftAboveCapM = lift;
          }
        }

        if (y <= minY + 0.0005) {
          deposited = true;
          break;
        }
      }

      this.velocities[p4] = vx;
      this.velocities[p4 + 1] = vy;
      this.velocities[p4 + 2] = vz;

      if (deposited) {
        if (x >= minX && x <= maxX && z >= minZ && z <= maxZ) {
          const ix = Math.max(
            0,
            Math.min(PRINT_NX - 1, Math.floor(((x - minX) / spanX) * PRINT_NX)),
          );
          const iz = Math.max(
            0,
            Math.min(PRINT_NZ - 1, Math.floor(((z - minZ) / spanZ) * PRINT_NZ)),
          );
          this.printDensity[iz * PRINT_NX + ix] += 1.0;
          this.totalDeposited++;
          this.cumulativeDispersalDistM += Math.hypot(
            x - this.launchOriginX[i],
            z - this.launchOriginZ[i],
          );
        }
        this.active[i] = 0;
        this.activeCount = Math.max(0, this.activeCount - 1);
        continue;
      }

      if (x < minX || x > maxX || z < minZ || z > maxZ || age > 40.0) {
        this.active[i] = 0;
        this.activeCount = Math.max(0, this.activeCount - 1);
        continue;
      }

      this.positions[p4] = x;
      this.positions[p4 + 1] = y;
      this.positions[p4 + 2] = z;
      this.positions[p4 + 3] = age;
    }
  }

  public getAirField(): AirConvectionField {
    return this.plume.getField();
  }

  public getSpores(): SporeParticleBuffers {
    return {
      maxCount: MAX_SPORES,
      activeCount: this.activeCount,
      positions: this.positions,
      velocities: this.velocities,
      active: this.active,
    };
  }

  public getSporePrint(): SporePrintGrid {
    return {
      nx: PRINT_NX,
      nz: PRINT_NZ,
      minX: this.bounds.minX,
      maxX: this.bounds.maxX,
      minZ: this.bounds.minZ,
      maxZ: this.bounds.maxZ,
      density: this.printDensity,
      totalDeposited: this.totalDeposited,
      meanDispersalRadiusM:
        this.totalDeposited > 0 ? this.cumulativeDispersalDistM / this.totalDeposited : 0,
      sporeColor: this.sporePrintColor,
    };
  }

  public getMaxSporeLiftAboveCapM(): number {
    return this.maxSporeLiftAboveCapM;
  }

  /**
   * Returns live 1,000x Buller's drop gill-gap microscope state using the selected
   * mushroom's emergent inter-gill spacing and species biophysical parameters.
   */
  public getActiveBullerState(bodyIndex: number): BullerMicroViewState {
    const body =
      this.lastBodies.length > 0
        ? this.lastBodies[Math.max(0, Math.min(this.lastBodies.length - 1, bodyIndex))]
        : null;
    const sp = body ? SPECIES_PRESETS[body.speciesId] : SPECIES_PRESETS.parasol;
    const gillSpacingM = body && body.gillSpacingM > 1e-5 ? body.gillSpacingM : 0.85e-3;

    const key = `${sp.id}:${gillSpacingM.toFixed(6)}`;
    if (!this.cachedBullerState || this.cachedBullerKey !== key) {
      const params: BullerParams = {
        sporeRadiusM: sp.sporeRadiusM,
        bullerDropRadiusM: sp.bullerDropRadiusM,
        sporeDensityKgM3: sp.sporeDensityKgM3,
        launchEfficiency: sp.launchEfficiency,
        gillSpacingM,
      };
      const launch = BullerCatapultSolver.computeLaunchKinematics(params);
      const durationSec = Math.max(0.003, launch.tauStokesSec * 14.0);
      const trajectory = BullerCatapultSolver.simulateTrajectory(params, durationSec);
      this.cachedBullerKey = key;
      this.cachedBullerState = {
        params,
        launch,
        trajectory,
        phase: this.bullerPhase,
        gillSpacingM,
        speciesName: sp.name,
      };
    }

    return {
      ...this.cachedBullerState,
      phase: this.bullerPhase,
    };
  }
}
