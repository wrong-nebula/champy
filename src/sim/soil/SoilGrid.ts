// ============================================================================
// SPOROCARP — Layer 1: 3D Unsaturated Forest Soil Grid & PDE Solvers
// - Mass-conservative mixed-form finite-volume 3D Richards equation
// - Millington–Quirk O2 / CO2 pore gas diffusion (implicit Red-Black GS)
// - Michaelis–Menten extracellular enzyme E + lignocellulose L -> sugar C_s
// - Soil thermal diffusion
// ============================================================================

import {
  AMBIENT_O2_MOL_M3,
  MOLAR_DENSITY_AIR,
  PA_PER_M_HEAD,
  SoilMaterialId,
  type EnvironmentState,
  type ISoilGrid,
  type SoilConfig,
  type TerrariumScenePreset,
} from '../../types/contract.ts';
import {
  D_CO2_AIR,
  D_O2_AIR,
  SOIL_MATERIAL_LIST,
  SOIL_MATERIALS,
  millingtonQuirkDiffusivity,
  vgConductivity,
  vgHeadFromTheta,
  vgPsiFromTheta,
  vgThetaFromHead,
} from './vanGenuchten.ts';

const THETA_EPS = 1.0e-5;
const K_EVAP_SOIL = 2.5e-8; // m / s at VPD = 1 (forest floor evaporation)
const K_CAT_ENZYME = 1.2e-4; // 1 / s (Michaelis-Menten k_cat)
const KM_LIGNIN = 15.0; // kg / m^3
const LAMBDA_ENZYME = 2.0e-6; // 1 / s enzyme decay
const D_ENZYME_0 = 2.0e-9; // m^2 / s
const D_CARBON_0 = 6.0e-9; // m^2 / s (effective soluble sugar dispersion)
const HENRY_O2_WATER = 0.008; // effective aqueous dissolved O2 storage scaling when waterlogged
const MICROBE_RESP_RATE = 1.2e-5; // mol / (m^3_bulk * s) forest soil microbial respiration
const CO2_TO_O2_DIFF_RATIO = D_CO2_AIR / D_O2_AIR;

// Pre-unboxed material parameter tables (indices 0..5)
const MAT_THETA_R = Float64Array.from(SOIL_MATERIAL_LIST, (m) => m.thetaR);
const MAT_THETA_S = Float64Array.from(SOIL_MATERIAL_LIST, (m) => m.thetaS);
const MAT_SPAN = Float64Array.from(SOIL_MATERIAL_LIST, (m) => m.thetaS - m.thetaR);
const MAT_INV_ALPHA = Float64Array.from(SOIL_MATERIAL_LIST, (m) => 1.0 / m.alpha);
const MAT_INV_N = Float64Array.from(SOIL_MATERIAL_LIST, (m) => 1.0 / m.n);
const MAT_INV_M = Float64Array.from(SOIL_MATERIAL_LIST, (m) => 1.0 / m.m);
const MAT_TH_DRY = Float64Array.from(SOIL_MATERIAL_LIST, (m) => m.thermalDiffusivityDry);
const MAT_TH_DELTA = Float64Array.from(
  SOIL_MATERIAL_LIST,
  (m) => m.thermalDiffusivityWet - m.thermalDiffusivityDry,
);

// Fast 2048-entry LUTs per material for K(Se), invC(Se), and Millington-Quirk D_O2(Se)
// (Head h(theta) stays exact analytical for exact hydrostatic equilibrium.)
const LUT_SIZE = 2048;
const LUT_MAX = LUT_SIZE - 1;
const MAT_COUNT = SOIL_MATERIAL_LIST.length;
const K_LUT = new Float64Array(MAT_COUNT * LUT_SIZE);
const INVC_LUT = new Float64Array(MAT_COUNT * LUT_SIZE);
const DO2_LUT = new Float32Array(MAT_COUNT * LUT_SIZE);

for (let mId = 0; mId < MAT_COUNT; mId++) {
  const mat = SOIL_MATERIAL_LIST[mId];
  const base = mId * LUT_SIZE;
  const span = mat.thetaS - mat.thetaR;
  for (let i = 0; i < LUT_SIZE; i++) {
    const se = i / LUT_MAX;
    const th = mat.thetaR + se * span;
    if (mat.id === SoilMaterialId.STONE) {
      K_LUT[base + i] = 0.0;
      INVC_LUT[base + i] = 0.0;
      DO2_LUT[base + i] = 0.0;
    } else {
      K_LUT[base + i] = vgConductivity(th, mat);
      DO2_LUT[base + i] = millingtonQuirkDiffusivity(th, mat.thetaS, D_O2_AIR);

      const seClamped = se <= 1e-5 ? 1e-5 : se >= 1.0 - 1e-6 ? 1.0 - 1e-6 : se;
      const h = vgHeadFromTheta(mat.thetaR + seClamped * span, mat);
      const oneMinus = 1.0 - Math.pow(seClamped, 1.0 / mat.m);
      const denom = mat.m * mat.n * span * seClamped * (oneMinus > 1e-12 ? oneMinus : 1e-12);
      const ic = Math.abs(h) / denom;
      INVC_LUT[base + i] = ic > 250.0 ? 250.0 : ic;
    }
  }
}

const DEFAULT_SOIL_CONFIG: SoilConfig = {
  nx: 48,
  ny: 32,
  nz: 48,
  dx: 0.01,
  ambientTempC: 15.0,
  ambientHumidity: 0.85,
  ambientCo2Ppm: 420.0,
  freeDrainBottom: false,
};

export class SoilGrid implements ISoilGrid {
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly dx: number;
  readonly sizeX: number;
  readonly sizeY: number;
  readonly sizeZ: number;
  readonly totalVoxels: number;

  readonly material: Uint8Array;
  readonly theta: Float32Array;
  readonly psi: Float32Array;
  readonly o2: Float32Array;
  readonly co2: Float32Array;
  readonly lignin: Float32Array;
  readonly solubleCarbon: Float32Array;
  readonly enzyme: Float32Array;
  readonly temperature: Float32Array;

  freeDrainBottom: boolean;

  private readonly theta64: Float64Array;
  private readonly cachedTheta: Float64Array;
  private readonly cachedMat: Uint8Array;
  private readonly head64: Float64Array;
  private readonly cond64: Float64Array;
  private readonly invC64: Float64Array;
  private readonly dO2Cache: Float32Array;
  private readonly fluxX: Float64Array;
  private readonly fluxY: Float64Array;
  private readonly fluxZ: Float64Array;
  private readonly outSum: Float64Array;
  private readonly inSum: Float64Array;
  private readonly scaleOut: Float64Array;
  private readonly scaleIn: Float64Array;
  private readonly diffScratch: Float32Array;

  constructor(config?: Partial<SoilConfig>) {
    const cfg: SoilConfig = { ...DEFAULT_SOIL_CONFIG, ...config };
    this.nx = cfg.nx;
    this.ny = cfg.ny;
    this.nz = cfg.nz;
    this.dx = cfg.dx;
    this.sizeX = this.nx * this.dx;
    this.sizeY = this.ny * this.dx;
    this.sizeZ = this.nz * this.dx;
    this.totalVoxels = this.nx * this.ny * this.nz;
    this.freeDrainBottom = cfg.freeDrainBottom;

    const n = this.totalVoxels;
    this.material = new Uint8Array(n);
    this.theta = new Float32Array(n);
    this.psi = new Float32Array(n);
    this.o2 = new Float32Array(n);
    this.co2 = new Float32Array(n);
    this.lignin = new Float32Array(n);
    this.solubleCarbon = new Float32Array(n);
    this.enzyme = new Float32Array(n);
    this.temperature = new Float32Array(n);

    this.theta64 = new Float64Array(n);
    this.cachedTheta = new Float64Array(n);
    this.cachedTheta.fill(-1.0);
    this.cachedMat = new Uint8Array(n);
    this.cachedMat.fill(255);
    this.head64 = new Float64Array(n);
    this.cond64 = new Float64Array(n);
    this.invC64 = new Float64Array(n);
    this.dO2Cache = new Float32Array(n);
    this.fluxX = new Float64Array(n);
    this.fluxY = new Float64Array(n);
    this.fluxZ = new Float64Array(n);
    this.outSum = new Float64Array(n);
    this.inSum = new Float64Array(n);
    this.scaleOut = new Float64Array(n);
    this.scaleIn = new Float64Array(n);
    this.diffScratch = new Float32Array(n);

    this.setupPresetTerrarium('forest-cross-section');
    const co2Init = MOLAR_DENSITY_AIR * cfg.ambientCo2Ppm * 1.0e-6;
    this.temperature.fill(cfg.ambientTempC);
    for (let i = 0; i < n; i++) {
      if (this.material[i] !== SoilMaterialId.STONE) {
        this.co2[i] = co2Init;
      }
    }
  }

  voxelIndex(ix: number, iy: number, iz: number): number {
    return (iy * this.nz + iz) * this.nx + ix;
  }

  worldToVoxel(x: number, y: number, z: number): [number, number, number] {
    const fx = Math.floor((x + this.sizeX * 0.5) / this.dx);
    const fy = Math.floor((y + this.sizeY) / this.dx);
    const fz = Math.floor((z + this.sizeZ * 0.5) / this.dx);
    const ix = fx < 0 ? 0 : fx >= this.nx ? this.nx - 1 : fx;
    const iy = fy < 0 ? 0 : fy >= this.ny ? this.ny - 1 : fy;
    const iz = fz < 0 ? 0 : fz >= this.nz ? this.nz - 1 : fz;
    return [ix, iy, iz];
  }

  voxelToWorld(ix: number, iy: number, iz: number): [number, number, number] {
    return [
      (ix + 0.5) * this.dx - this.sizeX * 0.5,
      -this.sizeY + (iy + 0.5) * this.dx,
      (iz + 0.5) * this.dx - this.sizeZ * 0.5,
    ];
  }

  totalWaterVolumeM3(): number {
    const n = this.totalVoxels;
    const th = this.theta;
    let sum = 0.0;
    for (let i = 0; i < n; i++) {
      sum += th[i];
    }
    return sum * this.dx * this.dx * this.dx;
  }

  setUniformHydrostaticHead(totalHeadMeters: number): void {
    const { nx, ny, nz, dx, sizeY } = this;
    for (let iy = 0; iy < ny; iy++) {
      const y = -sizeY + (iy + 0.5) * dx;
      const h = totalHeadMeters - y;
      const yOffset = iy * nz;
      for (let iz = 0; iz < nz; iz++) {
        const rowOffset = (yOffset + iz) * nx;
        for (let ix = 0; ix < nx; ix++) {
          const idx = rowOffset + ix;
          const matId = this.material[idx] as SoilMaterialId;
          if (matId === SoilMaterialId.STONE) {
            this.theta64[idx] = 0.0;
            this.theta[idx] = 0.0;
            this.psi[idx] = 0.0;
            continue;
          }
          const mat = SOIL_MATERIAL_LIST[matId];
          let th = vgThetaFromHead(h, mat);
          const lo = mat.thetaR + THETA_EPS;
          const hi = mat.thetaS - THETA_EPS;
          if (th < lo) th = lo;
          else if (th > hi) th = hi;
          this.theta64[idx] = th;
          this.theta[idx] = th;
          this.psi[idx] = vgPsiFromTheta(th, mat);
        }
      }
    }
    this.cachedTheta.fill(-1.0);
  }

  pourRain(worldX: number, worldZ: number, radiusM: number, depthM: number): void {
    if (depthM <= 0 || radiusM <= 0) return;
    this.syncExternalThetaEdits();
    const { nx, ny, nz, dx, sizeX, sizeZ } = this;
    const r2 = radiusM * radiusM;
    const halfX = sizeX * 0.5;
    const halfZ = sizeZ * 0.5;

    for (let iz = 0; iz < nz; iz++) {
      const wz = (iz + 0.5) * dx - halfZ;
      const dz = wz - worldZ;
      const dz2 = dz * dz;
      if (dz2 > r2) continue;

      for (let ix = 0; ix < nx; ix++) {
        const wx = (ix + 0.5) * dx - halfX;
        const rx = wx - worldX;
        if (rx * rx + dz2 > r2) continue;

        let remDepth = depthM;
        for (let iy = ny - 1; iy >= 0 && remDepth > 1e-12; iy--) {
          const idx = (iy * nz + iz) * nx + ix;
          const matId = this.material[idx] as SoilMaterialId;
          if (matId === SoilMaterialId.STONE) break;
          const mat = SOIL_MATERIAL_LIST[matId];
          const room = mat.thetaS - THETA_EPS - this.theta64[idx];
          if (room <= 0) continue;
          const wantTheta = remDepth / dx;
          const add = wantTheta < room ? wantTheta : room;
          this.theta64[idx] += add;
          this.theta[idx] = this.theta64[idx];
          this.psi[idx] = vgPsiFromTheta(this.theta64[idx], mat);
          this.cachedTheta[idx] = -1.0;
          remDepth -= add * dx;
        }
      }
    }
  }

  paintMaterial(
    worldX: number,
    worldY: number,
    worldZ: number,
    radiusM: number,
    mat: SoilMaterialId,
  ): void {
    if (radiusM <= 0) return;
    this.syncExternalThetaEdits();
    const { nx, ny, nz, dx, sizeX, sizeY, sizeZ } = this;
    const matParams = SOIL_MATERIALS[mat];
    const r2 = radiusM * radiusM;
    const halfX = sizeX * 0.5;
    const halfZ = sizeZ * 0.5;

    const minIx = Math.max(0, Math.floor((worldX - radiusM + halfX) / dx));
    const maxIx = Math.min(nx - 1, Math.floor((worldX + radiusM + halfX) / dx));
    const minIy = Math.max(0, Math.floor((worldY - radiusM + sizeY) / dx));
    const maxIy = Math.min(ny - 1, Math.floor((worldY + radiusM + sizeY) / dx));
    const minIz = Math.max(0, Math.floor((worldZ - radiusM + halfZ) / dx));
    const maxIz = Math.min(nz - 1, Math.floor((worldZ + radiusM + halfZ) / dx));

    for (let iy = minIy; iy <= maxIy; iy++) {
      const wy = -sizeY + (iy + 0.5) * dx;
      const dy = wy - worldY;
      const dy2 = dy * dy;
      for (let iz = minIz; iz <= maxIz; iz++) {
        const wz = (iz + 0.5) * dx - halfZ;
        const dz = wz - worldZ;
        const dyz2 = dy2 + dz * dz;
        if (dyz2 > r2) continue;
        for (let ix = minIx; ix <= maxIx; ix++) {
          const wx = (ix + 0.5) * dx - halfX;
          const rx = wx - worldX;
          if (rx * rx + dyz2 > r2) continue;
          const idx = (iy * nz + iz) * nx + ix;
          const prevMat = this.material[idx] as SoilMaterialId;
          const prevH =
            prevMat === SoilMaterialId.STONE
              ? -0.95
              : vgHeadFromTheta(this.theta64[idx], SOIL_MATERIAL_LIST[prevMat]);
          this.setVoxelMaterial(idx, mat, mat === SoilMaterialId.STONE ? 0 : prevH);
          this.lignin[idx] = matParams.initialLignin;
          this.solubleCarbon[idx] = matParams.initialSolubleCarbon;
          this.cachedTheta[idx] = -1.0;
          if (mat === SoilMaterialId.STONE) {
            this.o2[idx] = 0.0;
            this.co2[idx] = 0.0;
            this.enzyme[idx] = 0.0;
          }
        }
      }
    }
  }

  setupPresetTerrarium(preset: TerrariumScenePreset): void {
    const { nx, ny, nz, dx, sizeX, sizeY, sizeZ } = this;
    const halfX = sizeX * 0.5;
    const halfZ = sizeZ * 0.5;
    const co2Atm = MOLAR_DENSITY_AIR * 420.0e-6;

    this.enzyme.fill(0.0);
    this.temperature.fill(15.0);

    let initTotalHead = -0.95;

    for (let iy = 0; iy < ny; iy++) {
      const wy = -sizeY + (iy + 0.5) * dx;
      const yOffset = iy * nz;
      for (let iz = 0; iz < nz; iz++) {
        const wz = (iz + 0.5) * dx - halfZ;
        const rowOffset = (yOffset + iz) * nx;
        for (let ix = 0; ix < nx; ix++) {
          const wx = (ix + 0.5) * dx - halfX;
          const idx = rowOffset + ix;
          let matId: SoilMaterialId = SoilMaterialId.LOAM;

          if (preset === 'forest-cross-section') {
            initTotalHead = -0.95;
            const humusThickness =
              0.038 + 0.008 * Math.sin(wx * 18.0 + 0.4) * Math.cos(wz * 15.0 - 0.3);
            if (wy >= -humusThickness) {
              matId = SoilMaterialId.HUMUS;
            } else {
              matId = SoilMaterialId.LOAM;
            }

            // Left sandy drainage lens
            const sandCenterX = -0.12;
            const sandCenterY = -0.15 - 0.15 * wx;
            const dSand =
              ((wx - sandCenterX) / 0.11) ** 2 + ((wy - sandCenterY) / 0.065) ** 2;
            if (dSand < 1.0 && wy < -0.045) {
              matId = SoilMaterialId.SAND;
            }

            // Right heavy clay water-perching pocket
            const clayCenterX = 0.12;
            const clayCenterY = -0.18;
            const dClay =
              ((wx - clayCenterX) / 0.105) ** 2 + ((wy - clayCenterY) / 0.065) ** 2;
            if (dClay < 1.0 && wy < -0.07) {
              matId = SoilMaterialId.CLAY;
            }

            // Buried rotting oak log cylinder (rich lignin L = 240)
            const logX = -0.025 + 0.12 * wz;
            const logY = -0.078;
            const distLogR2 = (wx - logX) ** 2 + (wy - logY) ** 2;
            if (distLogR2 <= 0.044 * 0.044 && Math.abs(wz) <= 0.145) {
              matId = SoilMaterialId.WOOD;
            }

            // Granite stone obstacle
            const stDx = wx - 0.115;
            const stDy = wy - -0.075;
            const stDz = wz - 0.055;
            if (
              (stDx / 0.044) ** 2 + (stDy / 0.036) ** 2 + (stDz / 0.042) ** 2 <=
              1.0
            ) {
              matId = SoilMaterialId.STONE;
            }
          } else if (preset === 'fairy-ring-meadow') {
            initTotalHead = -1.1;
            if (iy >= ny - 1) {
              matId = SoilMaterialId.HUMUS;
            } else {
              matId = SoilMaterialId.LOAM;
            }
          } else if (preset === 'layered-column') {
            initTotalHead = -0.75;
            const yThird1 = Math.floor(ny / 3);
            const yThird2 = Math.floor((ny * 2) / 3);
            if (iy >= yThird2) {
              matId = SoilMaterialId.SAND;
            } else if (iy >= yThird1) {
              matId = SoilMaterialId.LOAM;
            } else {
              matId = SoilMaterialId.CLAY;
            }
          } else if (preset === 'rotting-log') {
            initTotalHead = -0.9;
            if (iy >= ny - 2) {
              matId = SoilMaterialId.HUMUS;
            } else {
              matId = SoilMaterialId.LOAM;
            }
            const distR2 = wx * wx + (wy - -0.068) ** 2;
            if (distR2 <= 0.058 * 0.058 && Math.abs(wz) <= 0.185) {
              matId = SoilMaterialId.WOOD;
            }
          }

          const matParams = SOIL_MATERIAL_LIST[matId];
          this.material[idx] = matId;
          this.lignin[idx] = matParams.initialLignin;
          this.solubleCarbon[idx] = matParams.initialSolubleCarbon;
          if (matId === SoilMaterialId.STONE) {
            this.o2[idx] = 0.0;
            this.co2[idx] = 0.0;
          } else {
            this.o2[idx] = AMBIENT_O2_MOL_M3;
            this.co2[idx] = co2Atm;
          }
        }
      }
    }

    this.setUniformHydrostaticHead(initTotalHead);
  }

  applyHyphalWaterSink(voxelIdx: number, deltaVolumeM3: number): number {
    if (voxelIdx < 0 || voxelIdx >= this.totalVoxels) return 0.0;
    const matId = this.material[voxelIdx] as SoilMaterialId;
    if (matId === SoilMaterialId.STONE) return 0.0;
    this.syncVoxelTheta(voxelIdx);

    const mat = SOIL_MATERIAL_LIST[matId];
    const voxVol = this.dx * this.dx * this.dx;
    const reqDeltaTheta = deltaVolumeM3 / voxVol;
    const curTheta = this.theta64[voxelIdx];

    let actualDeltaTheta: number;
    if (reqDeltaTheta >= 0.0) {
      const avail = curTheta - (mat.thetaR + THETA_EPS);
      if (avail <= 0.0) return 0.0;
      actualDeltaTheta = reqDeltaTheta < avail ? reqDeltaTheta : avail;
      this.theta64[voxelIdx] = curTheta - actualDeltaTheta;
    } else {
      const room = mat.thetaS - THETA_EPS - curTheta;
      if (room <= 0.0) return 0.0;
      const add = -reqDeltaTheta < room ? -reqDeltaTheta : room;
      actualDeltaTheta = -add;
      this.theta64[voxelIdx] = curTheta + add;
    }

    this.theta[voxelIdx] = this.theta64[voxelIdx];
    this.psi[voxelIdx] = vgPsiFromTheta(this.theta64[voxelIdx], mat);
    this.cachedTheta[voxelIdx] = -1.0;
    return actualDeltaTheta * voxVol;
  }

  consumeSolubleCarbon(voxelIdx: number, massKg: number): number {
    if (massKg <= 0 || voxelIdx < 0 || voxelIdx >= this.totalVoxels) return 0.0;
    if (this.material[voxelIdx] === SoilMaterialId.STONE) return 0.0;
    const voxVol = this.dx * this.dx * this.dx;
    const reqConc = massKg / voxVol;
    const availConc = this.solubleCarbon[voxelIdx];
    if (availConc <= 0.0) return 0.0;
    const takenConc = reqConc < availConc ? reqConc : availConc;
    this.solubleCarbon[voxelIdx] = availConc - takenConc;
    return takenConc * voxVol;
  }

  secreteEnzyme(voxelIdx: number, rateKgPerM3: number): void {
    if (rateKgPerM3 <= 0 || voxelIdx < 0 || voxelIdx >= this.totalVoxels) return;
    if (this.material[voxelIdx] === SoilMaterialId.STONE) return;
    const next = this.enzyme[voxelIdx] + rateKgPerM3;
    this.enzyme[voxelIdx] = next > 2.0 ? 2.0 : next;
  }

  applyRespiration(voxelIdx: number, o2ConsumedMol: number, co2ProducedMol: number): void {
    if (voxelIdx < 0 || voxelIdx >= this.totalVoxels) return;
    const matId = this.material[voxelIdx] as SoilMaterialId;
    if (matId === SoilMaterialId.STONE) return;

    const th = this.theta[voxelIdx];
    const epsA = Math.max(0.0, MAT_THETA_S[matId] - th);
    const epsO2 = Math.max(0.003, epsA + HENRY_O2_WATER * th);
    const gasVol = this.dx * this.dx * this.dx * epsO2;

    if (o2ConsumedMol > 0) {
      const nextO2 = this.o2[voxelIdx] - o2ConsumedMol / gasVol;
      this.o2[voxelIdx] = nextO2 > 0.0 ? nextO2 : 0.0;
    }
    if (co2ProducedMol > 0) {
      const nextCo2 = this.co2[voxelIdx] + co2ProducedMol / gasVol;
      this.co2[voxelIdx] = nextCo2 > 15.0 ? 15.0 : nextCo2;
    }
  }

  step(dtSec: number, env: EnvironmentState): void {
    if (dtSec <= 0 || !Number.isFinite(dtSec)) return;

    this.syncExternalThetaEdits();
    this.stepRichards(dtSec, env);
    this.stepGasDiffusion(dtSec, env);
    this.stepEnzymeAndCarbon(dtSec);
    this.stepTemperature(dtSec, env);
  }

  // --------------------------------------------------------------------------
  // Internal helpers
  // --------------------------------------------------------------------------

  private setVoxelMaterial(idx: number, matId: SoilMaterialId, headMeters: number): void {
    this.material[idx] = matId;
    if (matId === SoilMaterialId.STONE) {
      this.theta64[idx] = 0.0;
      this.theta[idx] = 0.0;
      this.psi[idx] = 0.0;
      return;
    }
    const mat = SOIL_MATERIAL_LIST[matId];
    let th = vgThetaFromHead(headMeters, mat);
    const lo = mat.thetaR + THETA_EPS;
    const hi = mat.thetaS - THETA_EPS;
    if (th < lo) th = lo;
    else if (th > hi) th = hi;
    this.theta64[idx] = th;
    this.theta[idx] = th;
    this.psi[idx] = vgPsiFromTheta(th, mat);
  }

  private syncVoxelTheta(idx: number): void {
    if (Math.fround(this.theta64[idx]) !== this.theta[idx]) {
      this.theta64[idx] = this.theta[idx];
      this.cachedTheta[idx] = -1.0;
    }
  }

  private syncExternalThetaEdits(): void {
    const n = this.totalVoxels;
    const th32 = this.theta;
    const th64 = this.theta64;
    const cTh = this.cachedTheta;
    for (let i = 0; i < n; i++) {
      if (Math.fround(th64[i]) !== th32[i]) {
        th64[i] = th32[i];
        cTh[i] = -1.0;
      }
    }
  }

  private updateConstitutiveCache(): void {
    const n = this.totalVoxels;
    const matArr = this.material;
    const th64 = this.theta64;
    const cTh = this.cachedTheta;
    const cMat = this.cachedMat;
    const h64 = this.head64;
    const k64 = this.cond64;
    const invC64 = this.invC64;
    const dO2 = this.dO2Cache;

    for (let i = 0; i < n; i++) {
      const matId = matArr[i];
      const th = th64[i];
      if (th === cTh[i] && matId === cMat[i]) continue;

      cTh[i] = th;
      cMat[i] = matId;

      if (matId === SoilMaterialId.STONE) {
        h64[i] = 0.0;
        k64[i] = 0.0;
        invC64[i] = 0.0;
        dO2[i] = 0.0;
        continue;
      }

      const seRaw = (th - MAT_THETA_R[matId]) / MAT_SPAN[matId];
      const se = seRaw <= 1e-6 ? 1e-6 : seRaw >= 1.0 - 1e-9 ? 1.0 - 1e-9 : seRaw;

      // Exact analytical head h(theta) so hydrostatic equilibrium holds to machine precision
      const logTerm = -Math.log(se) * MAT_INV_M[matId];
      let h: number;
      if (logTerm > 40.0) {
        const expArg = logTerm * MAT_INV_N[matId];
        h = expArg > 12.0 ? -1.0e4 : -Math.exp(expArg) * MAT_INV_ALPHA[matId];
      } else {
        const inner = Math.exp(logTerm) - 1.0;
        h = inner <= 0.0 ? 0.0 : -Math.pow(inner, MAT_INV_N[matId]) * MAT_INV_ALPHA[matId];
      }
      h64[i] = h < -1.0e4 ? -1.0e4 : h;

      // Fast linear-interpolated LUTs for K(Se), invC(Se), and Millington-Quirk D_O2(Se)
      const fIdx = se * LUT_MAX;
      const i0 = fIdx | 0;
      const frac = fIdx - i0;
      const base = matId * LUT_SIZE + i0;
      const base1 = i0 < LUT_MAX ? base + 1 : base;

      k64[i] = K_LUT[base] + (K_LUT[base1] - K_LUT[base]) * frac;
      invC64[i] = INVC_LUT[base] + (INVC_LUT[base1] - INVC_LUT[base]) * frac;
      dO2[i] = DO2_LUT[base] + (DO2_LUT[base1] - DO2_LUT[base]) * frac;
    }
  }

  private stepRichards(dtSec: number, env: EnvironmentState): void {
    const { nx, ny, nz, dx, totalVoxels } = this;
    const invDx = 1.0 / dx;
    const invDx2 = invDx * invDx;
    const matArr = this.material;
    const th64 = this.theta64;
    const h64 = this.head64;
    const k64 = this.cond64;
    const invC64 = this.invC64;
    const fx = this.fluxX;
    const fy = this.fluxY;
    const fz = this.fluxZ;
    const outSum = this.outSum;
    const inSum = this.inSum;
    const scaleOut = this.scaleOut;
    const scaleIn = this.scaleIn;

    let nSub: number;
    if (dtSec <= 10.0) {
      nSub = 1;
    } else if (dtSec <= 120.0) {
      nSub = 2;
    } else {
      nSub = 3;
    }
    const subDt = dtSec / nSub;
    const subDtOverDx = subDt * invDx;
    const subDtOverDx2 = subDt * invDx2;
    const strideY = nz * nx;

    const hasRain = env.rainRateMs > 0.0;
    const hasEvap = env.evaporationEnabled && env.ambientHumidity < 0.999;
    const evapFactor = hasEvap ? K_EVAP_SOIL * (1.0 - env.ambientHumidity) * subDtOverDx : 0.0;
    const rainDTheta = hasRain ? env.rainRateMs * subDtOverDx : 0.0;

    for (let sub = 0; sub < nSub; sub++) {
      this.updateConstitutiveCache();
      outSum.fill(0.0);
      inSum.fill(0.0);

      let anyFlux = false;

      // 1. Interior face transfers (pairwise stabilized finite-volume Richards)
      for (let iy = 0; iy < ny; iy++) {
        const yOff = iy * strideY;
        const hasYPlus = iy + 1 < ny;
        for (let iz = 0; iz < nz; iz++) {
          const rowOff = yOff + iz * nx;
          const hasZPlus = iz + 1 < nz;
          for (let ix = 0; ix < nx; ix++) {
            const idx = rowOff + ix;
            const matI = matArr[idx];
            if (matI === SoilMaterialId.STONE) {
              fx[idx] = 0.0;
              fy[idx] = 0.0;
              fz[idx] = 0.0;
              continue;
            }
            const hI = h64[idx];
            const kI = k64[idx];
            const thI = th64[idx];
            const icI = invC64[idx];

            // +X face
            if (ix + 1 < nx) {
              const j = idx + 1;
              const matJ = matArr[j];
              if (matJ !== SoilMaterialId.STONE && kI > 0.0 && k64[j] > 0.0) {
                const dh = h64[j] - hI;
                if (dh !== 0.0) {
                  const kij = Math.sqrt(kI * k64[j]);
                  let pairInvC: number;
                  if (matI === matJ) {
                    const dthAbs = Math.abs(th64[j] - thI);
                    pairInvC =
                      dthAbs > 1e-4 ? (2.0 * Math.abs(dh)) / dthAbs : icI + invC64[j];
                  } else {
                    pairInvC = icI + invC64[j];
                  }
                  const omega = 2.0 * subDtOverDx2 * kij * pairInvC;
                  const d = (-subDtOverDx2 * kij * dh) / (1.0 + omega);
                  fx[idx] = d;
                  anyFlux = true;
                  if (d > 0.0) {
                    outSum[idx] += d;
                    inSum[j] += d;
                  } else {
                    outSum[j] -= d;
                    inSum[idx] -= d;
                  }
                } else {
                  fx[idx] = 0.0;
                }
              } else {
                fx[idx] = 0.0;
              }
            } else {
              fx[idx] = 0.0;
            }

            // +Z face
            if (hasZPlus) {
              const j = idx + nx;
              const matJ = matArr[j];
              if (matJ !== SoilMaterialId.STONE && kI > 0.0 && k64[j] > 0.0) {
                const dh = h64[j] - hI;
                if (dh !== 0.0) {
                  const kij = Math.sqrt(kI * k64[j]);
                  let pairInvC: number;
                  if (matI === matJ) {
                    const dthAbs = Math.abs(th64[j] - thI);
                    pairInvC =
                      dthAbs > 1e-4 ? (2.0 * Math.abs(dh)) / dthAbs : icI + invC64[j];
                  } else {
                    pairInvC = icI + invC64[j];
                  }
                  const omega = 2.0 * subDtOverDx2 * kij * pairInvC;
                  const d = (-subDtOverDx2 * kij * dh) / (1.0 + omega);
                  fz[idx] = d;
                  anyFlux = true;
                  if (d > 0.0) {
                    outSum[idx] += d;
                    inSum[j] += d;
                  } else {
                    outSum[j] -= d;
                    inSum[idx] -= d;
                  }
                } else {
                  fz[idx] = 0.0;
                }
              } else {
                fz[idx] = 0.0;
              }
            } else {
              fz[idx] = 0.0;
            }

            // +Y face (exact gravity term +dx)
            if (hasYPlus) {
              const j = idx + strideY;
              const matJ = matArr[j];
              if (matJ !== SoilMaterialId.STONE && kI > 0.0 && k64[j] > 0.0) {
                const dH = h64[j] - hI + dx;
                if (Math.abs(dH) > 1e-14) {
                  const kij = Math.sqrt(kI * k64[j]);
                  let pairInvC: number;
                  if (matI === matJ) {
                    const dthAbs = Math.abs(th64[j] - thI);
                    pairInvC =
                      dthAbs > 1e-4
                        ? (2.0 * Math.abs(h64[j] - hI)) / dthAbs
                        : icI + invC64[j];
                  } else {
                    pairInvC = icI + invC64[j];
                  }
                  const omega = 2.0 * subDtOverDx2 * kij * pairInvC;
                  const d = (-subDtOverDx2 * kij * dH) / (1.0 + omega);
                  fy[idx] = d;
                  anyFlux = true;
                  if (d > 0.0) {
                    outSum[idx] += d;
                    inSum[j] += d;
                  } else {
                    outSum[j] -= d;
                    inSum[idx] -= d;
                  }
                } else {
                  fy[idx] = 0.0;
                }
              } else {
                fy[idx] = 0.0;
              }
            } else {
              fy[idx] = 0.0;
            }
          }
        }
      }

      if (anyFlux) {
        // 2. Conservative donor/receiver scaling
        for (let i = 0; i < totalVoxels; i++) {
          const matId = matArr[i];
          if (matId === SoilMaterialId.STONE) {
            scaleOut[i] = 0.0;
            scaleIn[i] = 0.0;
            continue;
          }
          const avail = th64[i] - (MAT_THETA_R[matId] + THETA_EPS);
          const room = MAT_THETA_S[matId] - THETA_EPS - th64[i];
          const outS = outSum[i];
          const inS = inSum[i];
          scaleOut[i] = outS > avail ? (avail > 0.0 ? avail / outS : 0.0) : 1.0;
          scaleIn[i] = inS > room ? (room > 0.0 ? room / inS : 0.0) : 1.0;
        }

        // 3. Exact symmetric conservative face updates
        for (let iy = 0; iy < ny; iy++) {
          const yOff = iy * strideY;
          const hasYPlus = iy + 1 < ny;
          for (let iz = 0; iz < nz; iz++) {
            const rowOff = yOff + iz * nx;
            const hasZPlus = iz + 1 < nz;
            for (let ix = 0; ix < nx; ix++) {
              const idx = rowOff + ix;

              const dxFlux = fx[idx];
              if (dxFlux !== 0.0) {
                const j = idx + 1;
                const s =
                  dxFlux > 0.0
                    ? Math.min(scaleOut[idx], scaleIn[j])
                    : Math.min(scaleOut[j], scaleIn[idx]);
                const dEff = dxFlux * s;
                th64[idx] -= dEff;
                th64[j] += dEff;
              }

              if (hasZPlus) {
                const dzFlux = fz[idx];
                if (dzFlux !== 0.0) {
                  const j = idx + nx;
                  const s =
                    dzFlux > 0.0
                      ? Math.min(scaleOut[idx], scaleIn[j])
                      : Math.min(scaleOut[j], scaleIn[idx]);
                  const dEff = dzFlux * s;
                  th64[idx] -= dEff;
                  th64[j] += dEff;
                }
              }

              if (hasYPlus) {
                const dyFlux = fy[idx];
                if (dyFlux !== 0.0) {
                  const j = idx + strideY;
                  const s =
                    dyFlux > 0.0
                      ? Math.min(scaleOut[idx], scaleIn[j])
                      : Math.min(scaleOut[j], scaleIn[idx]);
                  const dEff = dyFlux * s;
                  th64[idx] -= dEff;
                  th64[j] += dEff;
                }
              }
            }
          }
        }
      }

      // 4. Boundary fluxes: rain, surface evaporation, optional bottom free drainage
      if (hasRain || hasEvap) {
        const topOff = (ny - 1) * strideY;
        for (let iz = 0; iz < nz; iz++) {
          const rowOff = topOff + iz * nx;
          for (let ix = 0; ix < nx; ix++) {
            const idx = rowOff + ix;
            const matId = matArr[idx];
            if (matId === SoilMaterialId.STONE) continue;
            if (hasRain) {
              const room = MAT_THETA_S[matId] - THETA_EPS - th64[idx];
              if (room > 0.0) {
                th64[idx] += rainDTheta < room ? rainDTheta : room;
              }
            }
            if (hasEvap) {
              const se = (th64[idx] - MAT_THETA_R[matId]) / MAT_SPAN[matId];
              if (se > 0.05) {
                const dEvap = evapFactor * (se - 0.05);
                const avail = th64[idx] - (MAT_THETA_R[matId] + THETA_EPS);
                if (avail > 0.0) {
                  th64[idx] -= dEvap < avail ? dEvap : avail;
                }
              }
            }
          }
        }
      }

      if (this.freeDrainBottom) {
        for (let iz = 0; iz < nz; iz++) {
          const rowOff = iz * nx;
          for (let ix = 0; ix < nx; ix++) {
            const idx = rowOff + ix;
            const matId = matArr[idx];
            if (matId === SoilMaterialId.STONE) continue;
            const kBot = k64[idx];
            const dDrain = kBot * subDtOverDx;
            const avail = th64[idx] - (MAT_THETA_R[matId] + THETA_EPS);
            if (avail > 0.0 && dDrain > 0.0) {
              th64[idx] -= dDrain < avail ? dDrain : avail;
            }
          }
        }
      }
    }

    // 5. Refresh constitutive cache for changed voxels and write back theta & psi
    this.updateConstitutiveCache();
    for (let i = 0; i < totalVoxels; i++) {
      if (matArr[i] === SoilMaterialId.STONE) {
        this.theta[i] = 0.0;
        this.psi[i] = 0.0;
      } else {
        this.theta[i] = th64[i];
        this.psi[i] = PA_PER_M_HEAD * h64[i];
      }
    }
  }

  private stepGasDiffusion(dtSec: number, env: EnvironmentState): void {
    const { nx, ny, nz, dx, totalVoxels } = this;
    const strideY = nz * nx;
    const invDx2 = 1.0 / (dx * dx);
    const matArr = this.material;
    const thArr = this.theta;
    const o2 = this.o2;
    const co2 = this.co2;
    const csArr = this.solubleCarbon;
    const dO2Arr = this.dO2Cache;

    const o2Atm = AMBIENT_O2_MOL_M3;
    const co2Atm = MOLAR_DENSITY_AIR * env.ambientCo2Ppm * 1.0e-6;

    // Background forest soil microbial aerobic respiration
    for (let i = 0; i < totalVoxels; i++) {
      const matId = matArr[i];
      if (matId === SoilMaterialId.STONE) continue;
      const cs = csArr[i];
      const curO2 = o2[i];
      if (cs > 1e-4 && curO2 > 1e-4) {
        const th = thArr[i];
        const epsA = Math.max(0.0, MAT_THETA_S[matId] - th);
        const epsO2 = Math.max(0.003, epsA + HENRY_O2_WATER * th);
        const rateBulk =
          MICROBE_RESP_RATE * (cs / (0.2 + cs)) * (curO2 / (0.5 + curO2));
        const dConc = (rateBulk * dtSec) / epsO2;
        const consumed = dConc < curO2 ? dConc : curO2;
        o2[i] = curO2 - consumed;
        const nextCo2 = co2[i] + consumed;
        co2[i] = nextCo2 > 15.0 ? 15.0 : nextCo2;
      }
    }

    // Unconditionally stable implicit Red-Black Gauss-Seidel for O2 & CO2
    const sweeps = dtSec > 300.0 ? 2 : 1;
    const invDt = 1.0 / dtSec;
    for (let s = 0; s < sweeps; s++) {
      for (let color = 0; color < 2; color++) {
        for (let iy = 0; iy < ny; iy++) {
          const yOff = iy * strideY;
          const isTop = iy === ny - 1;
          for (let iz = 0; iz < nz; iz++) {
            const rowOff = yOff + iz * nx;
            const startIx = (iy + iz + color) & 1;
            for (let ix = startIx; ix < nx; ix += 2) {
              const idx = rowOff + ix;
              const matId = matArr[idx];
              if (matId === SoilMaterialId.STONE) continue;

              const dO2I = dO2Arr[idx];
              const th = thArr[idx];
              const epsA = Math.max(0.0, MAT_THETA_S[matId] - th);
              const epsO2 = Math.max(0.003, epsA + HENRY_O2_WATER * th);
              const capOverDt = epsO2 * invDt;

              let diagO2 = capOverDt;
              let rhsO2 = capOverDt * o2[idx];
              let diagCo2 = capOverDt;
              let rhsCo2 = capOverDt * co2[idx];

              if (dO2I > 0.0) {
                if (ix > 0) {
                  const j = idx - 1;
                  const dJ = dO2Arr[j];
                  if (dJ > 0.0) {
                    const gO2 = ((2.0 * dO2I * dJ) / (dO2I + dJ)) * invDx2;
                    diagO2 += gO2;
                    rhsO2 += gO2 * o2[j];
                    const gCo2 = gO2 * CO2_TO_O2_DIFF_RATIO;
                    diagCo2 += gCo2;
                    rhsCo2 += gCo2 * co2[j];
                  }
                }
                if (ix + 1 < nx) {
                  const j = idx + 1;
                  const dJ = dO2Arr[j];
                  if (dJ > 0.0) {
                    const gO2 = ((2.0 * dO2I * dJ) / (dO2I + dJ)) * invDx2;
                    diagO2 += gO2;
                    rhsO2 += gO2 * o2[j];
                    const gCo2 = gO2 * CO2_TO_O2_DIFF_RATIO;
                    diagCo2 += gCo2;
                    rhsCo2 += gCo2 * co2[j];
                  }
                }
                if (iz > 0) {
                  const j = idx - nx;
                  const dJ = dO2Arr[j];
                  if (dJ > 0.0) {
                    const gO2 = ((2.0 * dO2I * dJ) / (dO2I + dJ)) * invDx2;
                    diagO2 += gO2;
                    rhsO2 += gO2 * o2[j];
                    const gCo2 = gO2 * CO2_TO_O2_DIFF_RATIO;
                    diagCo2 += gCo2;
                    rhsCo2 += gCo2 * co2[j];
                  }
                }
                if (iz + 1 < nz) {
                  const j = idx + nx;
                  const dJ = dO2Arr[j];
                  if (dJ > 0.0) {
                    const gO2 = ((2.0 * dO2I * dJ) / (dO2I + dJ)) * invDx2;
                    diagO2 += gO2;
                    rhsO2 += gO2 * o2[j];
                    const gCo2 = gO2 * CO2_TO_O2_DIFF_RATIO;
                    diagCo2 += gCo2;
                    rhsCo2 += gCo2 * co2[j];
                  }
                }
                if (iy > 0) {
                  const j = idx - strideY;
                  const dJ = dO2Arr[j];
                  if (dJ > 0.0) {
                    const gO2 = ((2.0 * dO2I * dJ) / (dO2I + dJ)) * invDx2;
                    diagO2 += gO2;
                    rhsO2 += gO2 * o2[j];
                    const gCo2 = gO2 * CO2_TO_O2_DIFF_RATIO;
                    diagCo2 += gCo2;
                    rhsCo2 += gCo2 * co2[j];
                  }
                }
                if (!isTop) {
                  const j = idx + strideY;
                  const dJ = dO2Arr[j];
                  if (dJ > 0.0) {
                    const gO2 = ((2.0 * dO2I * dJ) / (dO2I + dJ)) * invDx2;
                    diagO2 += gO2;
                    rhsO2 += gO2 * o2[j];
                    const gCo2 = gO2 * CO2_TO_O2_DIFF_RATIO;
                    diagCo2 += gCo2;
                    rhsCo2 += gCo2 * co2[j];
                  }
                } else {
                  const gTopO2 = 2.0 * dO2I * invDx2;
                  diagO2 += gTopO2;
                  rhsO2 += gTopO2 * o2Atm;

                  const gTopCo2 = gTopO2 * CO2_TO_O2_DIFF_RATIO;
                  diagCo2 += gTopCo2;
                  rhsCo2 += gTopCo2 * co2Atm;
                }
              }

              o2[idx] = rhsO2 / diagO2;
              co2[idx] = rhsCo2 / diagCo2;
            }
          }
        }
      }
    }
  }

  private stepEnzymeAndCarbon(dtSec: number): void {
    const { nx, ny, nz, dx, totalVoxels } = this;
    const strideY = nz * nx;
    const invDx2 = 1.0 / (dx * dx);
    const matArr = this.material;
    const thArr = this.theta;
    const enz = this.enzyme;
    const lig = this.lignin;
    const cs = this.solubleCarbon;
    const temp = this.temperature;
    const se2Arr = this.diffScratch;

    const decayFactor = Math.exp(-LAMBDA_ENZYME * dtSec);
    let anyEnzyme = false;

    for (let i = 0; i < totalVoxels; i++) {
      const matId = matArr[i];
      if (matId === SoilMaterialId.STONE) {
        se2Arr[i] = 0.0;
        continue;
      }
      const seRaw = (thArr[i] - MAT_THETA_R[matId]) / MAT_SPAN[matId];
      const se = seRaw < 0.0 ? 0.0 : seRaw > 1.0 ? 1.0 : seRaw;
      se2Arr[i] = se * se;

      const e = enz[i];
      if (e > 1e-8) {
        anyEnzyme = true;
        const l = lig[i];
        if (l > 1e-6 && se > 0.05) {
          const tFactor = Math.max(0.2, Math.min(2.2, 1.0 + 0.045 * (temp[i] - 15.0)));
          const waterFactor = se > 0.25 ? 1.0 : se * 4.0;
          const rate = K_CAT_ENZYME * e * (l / (KM_LIGNIN + l)) * tFactor * waterFactor;
          const dL = rate * dtSec;
          const actualDL = dL < l ? dL : l;
          lig[i] = l - actualDL;
          cs[i] += actualDL;
        }
        enz[i] = e * decayFactor;
      }
    }

    const invDt = 1.0 / dtSec;
    const cCoeff = D_CARBON_0 * invDx2;
    const eCoeff = D_ENZYME_0 * invDx2;

    for (let color = 0; color < 2; color++) {
      for (let iy = 0; iy < ny; iy++) {
        const yOff = iy * strideY;
        for (let iz = 0; iz < nz; iz++) {
          const rowOff = yOff + iz * nx;
          const startIx = (iy + iz + color) & 1;
          for (let ix = startIx; ix < nx; ix += 2) {
            const idx = rowOff + ix;
            const sI = se2Arr[idx];
            if (sI <= 0.0) continue;

            let diagC = invDt;
            let rhsC = invDt * cs[idx];
            let diagE = invDt;
            let rhsE = invDt * enz[idx];

            if (ix > 0) {
              const j = idx - 1;
              const sJ = se2Arr[j];
              if (sJ > 0.0) {
                const sFace = 0.5 * (sI + sJ);
                const gC = cCoeff * sFace;
                diagC += gC;
                rhsC += gC * cs[j];
                if (anyEnzyme) {
                  const gE = eCoeff * sFace;
                  diagE += gE;
                  rhsE += gE * enz[j];
                }
              }
            }
            if (ix + 1 < nx) {
              const j = idx + 1;
              const sJ = se2Arr[j];
              if (sJ > 0.0) {
                const sFace = 0.5 * (sI + sJ);
                const gC = cCoeff * sFace;
                diagC += gC;
                rhsC += gC * cs[j];
                if (anyEnzyme) {
                  const gE = eCoeff * sFace;
                  diagE += gE;
                  rhsE += gE * enz[j];
                }
              }
            }
            if (iz > 0) {
              const j = idx - nx;
              const sJ = se2Arr[j];
              if (sJ > 0.0) {
                const sFace = 0.5 * (sI + sJ);
                const gC = cCoeff * sFace;
                diagC += gC;
                rhsC += gC * cs[j];
                if (anyEnzyme) {
                  const gE = eCoeff * sFace;
                  diagE += gE;
                  rhsE += gE * enz[j];
                }
              }
            }
            if (iz + 1 < nz) {
              const j = idx + nx;
              const sJ = se2Arr[j];
              if (sJ > 0.0) {
                const sFace = 0.5 * (sI + sJ);
                const gC = cCoeff * sFace;
                diagC += gC;
                rhsC += gC * cs[j];
                if (anyEnzyme) {
                  const gE = eCoeff * sFace;
                  diagE += gE;
                  rhsE += gE * enz[j];
                }
              }
            }
            if (iy > 0) {
              const j = idx - strideY;
              const sJ = se2Arr[j];
              if (sJ > 0.0) {
                const sFace = 0.5 * (sI + sJ);
                const gC = cCoeff * sFace;
                diagC += gC;
                rhsC += gC * cs[j];
                if (anyEnzyme) {
                  const gE = eCoeff * sFace;
                  diagE += gE;
                  rhsE += gE * enz[j];
                }
              }
            }
            if (iy + 1 < ny) {
              const j = idx + strideY;
              const sJ = se2Arr[j];
              if (sJ > 0.0) {
                const sFace = 0.5 * (sI + sJ);
                const gC = cCoeff * sFace;
                diagC += gC;
                rhsC += gC * cs[j];
                if (anyEnzyme) {
                  const gE = eCoeff * sFace;
                  diagE += gE;
                  rhsE += gE * enz[j];
                }
              }
            }

            cs[idx] = rhsC / diagC;
            if (anyEnzyme) {
              enz[idx] = rhsE / diagE;
            }
          }
        }
      }
    }
  }

  private stepTemperature(dtSec: number, env: EnvironmentState): void {
    const { nx, ny, nz, dx, totalVoxels } = this;
    const temp = this.temperature;
    const tAmb = env.ambientTempC;

    let maxDiff = 0.0;
    for (let i = 0; i < totalVoxels; i++) {
      const d = Math.abs(temp[i] - tAmb);
      if (d > maxDiff) maxDiff = d;
    }
    if (maxDiff < 1e-3) return;

    const strideY = nz * nx;
    const invDx2 = 1.0 / (dx * dx);
    const invDt = 1.0 / dtSec;
    const matArr = this.material;
    const thArr = this.theta;
    const alphaArr = this.diffScratch;

    for (let i = 0; i < totalVoxels; i++) {
      const matId = matArr[i];
      if (matId === SoilMaterialId.STONE) {
        alphaArr[i] = MAT_TH_DRY[matId];
      } else {
        const seRaw = (thArr[i] - MAT_THETA_R[matId]) / MAT_SPAN[matId];
        const se = seRaw < 0.0 ? 0.0 : seRaw > 1.0 ? 1.0 : seRaw;
        alphaArr[i] = MAT_TH_DRY[matId] + MAT_TH_DELTA[matId] * se;
      }
    }

    for (let color = 0; color < 2; color++) {
      for (let iy = 0; iy < ny; iy++) {
        const yOff = iy * strideY;
        const isTop = iy === ny - 1;
        for (let iz = 0; iz < nz; iz++) {
          const rowOff = yOff + iz * nx;
          const startIx = (iy + iz + color) & 1;
          for (let ix = startIx; ix < nx; ix += 2) {
            const idx = rowOff + ix;
            const aI = alphaArr[idx];
            let diag = invDt;
            let rhs = invDt * temp[idx];

            if (ix > 0) {
              const g = 0.5 * (aI + alphaArr[idx - 1]) * invDx2;
              diag += g;
              rhs += g * temp[idx - 1];
            }
            if (ix + 1 < nx) {
              const g = 0.5 * (aI + alphaArr[idx + 1]) * invDx2;
              diag += g;
              rhs += g * temp[idx + 1];
            }
            if (iz > 0) {
              const g = 0.5 * (aI + alphaArr[idx - nx]) * invDx2;
              diag += g;
              rhs += g * temp[idx - nx];
            }
            if (iz + 1 < nz) {
              const g = 0.5 * (aI + alphaArr[idx + nx]) * invDx2;
              diag += g;
              rhs += g * temp[idx + nx];
            }
            if (iy > 0) {
              const g = 0.5 * (aI + alphaArr[idx - strideY]) * invDx2;
              diag += g;
              rhs += g * temp[idx - strideY];
            }
            if (!isTop) {
              const g = 0.5 * (aI + alphaArr[idx + strideY]) * invDx2;
              diag += g;
              rhs += g * temp[idx + strideY];
            } else {
              const gTop = 2.0 * aI * invDx2;
              diag += gTop;
              rhs += gTop * tAmb;
            }

            temp[idx] = rhs / diag;
          }
        }
      }
    }
  }
}
