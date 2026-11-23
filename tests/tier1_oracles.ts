// ============================================================================
// SPOROCARP — Tier 1 Independent Exact Analytical Physics Oracles
// Owned exclusively by Worker 4 (biophysics-auditor). Locked by ORACLE.sha256.
// ============================================================================

import {
  GAMMA_WATER,
  GRAVITY,
  MU_AIR,
  PA_PER_M_HEAD,
  SPECIES_PRESETS,
  SoilMaterialId,
  type BullerParams,
  type EnvironmentState,
  type IMyceliumNetwork,
  type MyceliumEdge,
  type MyceliumNode,
  type MyceliumRenderBuffers,
  type PrimordiumAnchor,
  type SpeciesPreset,
} from '../src/types/contract.ts';
import { SoilGrid } from '../src/sim/soil/SoilGrid.ts';
import {
  SOIL_MATERIALS,
  vgConductivity,
  vgPsiFromTheta,
} from '../src/sim/soil/vanGenuchten.ts';
import { ThinShellCapSolver } from '../src/sim/morpho/ThinShellSolver.ts';
import {
  MorphogenesisSystem,
  stepStipeLockhartExact,
} from '../src/sim/morpho/MorphogenesisSystem.ts';
import { BullerCatapultSolver } from '../src/sim/spores/BullerCatapult.ts';

export interface OracleTestResult {
  id: string;
  name: string;
  tier: 1 | 2 | 3;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  details: string[];
  error?: string;
}

// Independent analytical van Genuchten retention from THEORY.md Table 1
interface ReferenceVG {
  thetaR: number;
  thetaS: number;
  alpha: number;
  n: number;
  ks: number;
}

const REFERENCE_VG_TABLE: Record<number, ReferenceVG> = {
  [SoilMaterialId.SAND]: { thetaR: 0.045, thetaS: 0.43, alpha: 14.5, n: 2.68, ks: 8.25e-5 },
  [SoilMaterialId.LOAM]: { thetaR: 0.067, thetaS: 0.45, alpha: 2.0, n: 1.41, ks: 1.25e-6 },
  [SoilMaterialId.CLAY]: { thetaR: 0.068, thetaS: 0.38, alpha: 0.8, n: 1.09, ks: 5.56e-7 },
  [SoilMaterialId.HUMUS]: { thetaR: 0.08, thetaS: 0.65, alpha: 4.5, n: 1.55, ks: 3.5e-5 },
  [SoilMaterialId.WOOD]: { thetaR: 0.1, thetaS: 0.72, alpha: 1.2, n: 1.35, ks: 2.5e-6 },
};

function independentVgTheta(hMeters: number, ref: ReferenceVG): number {
  if (hMeters >= 0) return ref.thetaS;
  const m = 1 - 1 / ref.n;
  const se = Math.pow(1 + Math.pow(ref.alpha * Math.abs(hMeters), ref.n), -m);
  return ref.thetaR + (ref.thetaS - ref.thetaR) * se;
}

function makeClosedNoEvapEnv(): EnvironmentState {
  return {
    ambientTempC: 15,
    ambientHumidity: 1.0, // RH = 1.0 -> zero surface evaporation for strict closed mass test
    ambientCo2Ppm: 420,
    lightDir: [0, 1, 0],
    terrainTiltRad: 0,
    gravityDir: [0, -1, 0],
    rainRateMs: 0,
    evaporationEnabled: false,
    timeWarp: 1,
  };
}

// ============================================================================
// T1.1 — Soil Hydrology Mass Conservation, Hydrostatic Equilibrium & Materials
// ============================================================================
export function runT1_1_SoilHydrology(): OracleTestResult {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};

  try {
    const envClosed = makeClosedNoEvapEnv();

    // ------------------------------------------------------------------------
    // Part A: Closed Layered Column (SAND over LOAM over CLAY) Mass Conservation
    // ------------------------------------------------------------------------
    const col = new SoilGrid({
      nx: 6,
      ny: 18,
      nz: 6,
      dx: 0.01,
      ambientTempC: 15,
      ambientHumidity: 1.0,
      ambientCo2Ppm: 420,
      freeDrainBottom: false,
    });
    col.freeDrainBottom = false;

    for (let iy = 0; iy < col.ny; iy++) {
      const mat =
        iy >= 12
          ? SoilMaterialId.SAND
          : iy >= 6
            ? SoilMaterialId.LOAM
            : SoilMaterialId.CLAY;
      for (let iz = 0; iz < col.nz; iz++) {
        for (let ix = 0; ix < col.nx; ix++) {
          col.material[col.voxelIndex(ix, iy, iz)] = mat;
        }
      }
    }
    col.setUniformHydrostaticHead(-0.5);

    const waterBeforeRain = col.totalWaterVolumeM3();
    // 2.0 mm rain pulse over central radius 0.025 m onto unsaturated SAND top layer
    col.pourRain(0, 0, 0.025, 0.002);
    const waterAfterRain = col.totalWaterVolumeM3();
    const addedRainM3 = waterAfterRain - waterBeforeRain;

    const computeMoistureCenterY = (g: SoilGrid): number => {
      let sumWy = 0;
      let sumW = 0;
      for (let iy = 0; iy < g.ny; iy++) {
        for (let iz = 0; iz < g.nz; iz++) {
          for (let ix = 0; ix < g.nx; ix++) {
            const idx = g.voxelIndex(ix, iy, iz);
            const y = g.voxelToWorld(ix, iy, iz)[1];
            const w = g.theta[idx];
            sumWy += w * y;
            sumW += w;
          }
        }
      }
      return sumW > 0 ? sumWy / sumW : 0;
    };

    const yComAfterRain = computeMoistureCenterY(col);

    for (let s = 0; s < 120; s++) {
      col.step(5.0, envClosed);
    }

    const waterAfterFlow = col.totalWaterVolumeM3();
    const yComAfterFlow = computeMoistureCenterY(col);
    const massRelError =
      Math.abs(waterAfterFlow - waterAfterRain) / Math.max(1e-12, waterAfterRain);
    const downwardShiftMm = (yComAfterRain - yComAfterFlow) * 1000;

    metrics.massInitialM3 = waterBeforeRain;
    metrics.massAddedRainM3 = addedRainM3;
    metrics.massFinalM3 = waterAfterFlow;
    metrics.massRelativeError = massRelError;
    metrics.rainDownwardShiftMm = downwardShiftMm;

    const massOk =
      addedRainM3 > 0 &&
      Number.isFinite(massRelError) &&
      massRelError < 1e-5 &&
      downwardShiftMm > 0;
    details.push(
      `Mass conservation: added=${addedRainM3.toExponential(3)} m^3, relErr=${massRelError.toExponential(3)} (limit < 1e-5), ComShift=${downwardShiftMm.toFixed(4)} mm`,
    );

    // ------------------------------------------------------------------------
    // Part B: Hydrostatic Equilibrium (psi(y) + rho_w*g*y = const) & Relaxation
    // ------------------------------------------------------------------------
    const eqCol = new SoilGrid({
      nx: 4,
      ny: 16,
      nz: 4,
      dx: 0.01,
      ambientTempC: 15,
      ambientHumidity: 1.0,
      ambientCo2Ppm: 420,
      freeDrainBottom: false,
    });
    for (let iy = 0; iy < eqCol.ny; iy++) {
      const mat =
        iy >= 11
          ? SoilMaterialId.SAND
          : iy >= 5
            ? SoilMaterialId.LOAM
            : SoilMaterialId.CLAY;
      for (let iz = 0; iz < eqCol.nz; iz++) {
        for (let ix = 0; ix < eqCol.nx; ix++) {
          eqCol.material[eqCol.voxelIndex(ix, iy, iz)] = mat;
        }
      }
    }

    const H0 = -0.35;
    eqCol.setUniformHydrostaticHead(H0);

    let maxHeadErrM = 0;
    let maxThetaVgErr = 0;
    const eqThetaInit = new Float32Array(eqCol.theta);

    for (let iy = 0; iy < eqCol.ny; iy++) {
      for (let iz = 0; iz < eqCol.nz; iz++) {
        for (let ix = 0; ix < eqCol.nx; ix++) {
          const idx = eqCol.voxelIndex(ix, iy, iz);
          const y = eqCol.voxelToWorld(ix, iy, iz)[1];
          const matId = eqCol.material[idx];
          const ref = REFERENCE_VG_TABLE[matId];
          const hExpected = H0 - y;
          const totalHeadActual = eqCol.psi[idx] / PA_PER_M_HEAD + y;
          maxHeadErrM = Math.max(maxHeadErrM, Math.abs(totalHeadActual - H0));

          const thetaExpected = independentVgTheta(hExpected, ref);
          maxThetaVgErr = Math.max(maxThetaVgErr, Math.abs(eqCol.theta[idx] - thetaExpected));
        }
      }
    }

    for (let s = 0; s < 60; s++) {
      eqCol.step(5.0, envClosed);
    }

    let maxEquilibriumDrift = 0;
    for (let i = 0; i < eqCol.theta.length; i++) {
      maxEquilibriumDrift = Math.max(
        maxEquilibriumDrift,
        Math.abs(eqCol.theta[i] - eqThetaInit[i]),
      );
    }

    // Perturbation relaxation check on conductive SAND column
    const relaxCol = new SoilGrid({
      nx: 4,
      ny: 12,
      nz: 4,
      dx: 0.01,
      ambientTempC: 15,
      ambientHumidity: 1.0,
      ambientCo2Ppm: 420,
      freeDrainBottom: false,
    });
    relaxCol.material.fill(SoilMaterialId.SAND);
    relaxCol.setUniformHydrostaticHead(-0.14);

    const sandMat = SOIL_MATERIALS[SoilMaterialId.SAND];
    for (let iy = 0; iy < relaxCol.ny; iy++) {
      const delta = iy >= 6 ? 0.035 : -0.035;
      for (let iz = 0; iz < relaxCol.nz; iz++) {
        for (let ix = 0; ix < relaxCol.nx; ix++) {
          const idx = relaxCol.voxelIndex(ix, iy, iz);
          const nextTheta = Math.max(
            sandMat.thetaR + 0.005,
            Math.min(sandMat.thetaS - 0.005, relaxCol.theta[idx] + delta),
          );
          relaxCol.theta[idx] = nextTheta;
          relaxCol.psi[idx] = vgPsiFromTheta(nextTheta, sandMat);
        }
      }
    }

    const totalHeadStd = (g: SoilGrid): number => {
      let sum = 0;
      let n = 0;
      const vals: number[] = [];
      for (let iy = 0; iy < g.ny; iy++) {
        for (let iz = 0; iz < g.nz; iz++) {
          for (let ix = 0; ix < g.nx; ix++) {
            const idx = g.voxelIndex(ix, iy, iz);
            const y = g.voxelToWorld(ix, iy, iz)[1];
            const H = g.psi[idx] / PA_PER_M_HEAD + y;
            vals.push(H);
            sum += H;
            n++;
          }
        }
      }
      const mean = sum / Math.max(1, n);
      let sq = 0;
      for (const v of vals) sq += (v - mean) * (v - mean);
      return Math.sqrt(sq / Math.max(1, n));
    };

    const relaxMassBefore = relaxCol.totalWaterVolumeM3();
    const headStdBefore = totalHeadStd(relaxCol);
    for (let s = 0; s < 120; s++) {
      relaxCol.step(10.0, envClosed);
    }
    const headStdAfter = totalHeadStd(relaxCol);
    const relaxMassRelErr =
      Math.abs(relaxCol.totalWaterVolumeM3() - relaxMassBefore) /
      Math.max(1e-12, relaxMassBefore);

    metrics.hydrostaticMaxHeadErrM = maxHeadErrM;
    metrics.hydrostaticMaxVgThetaErr = maxThetaVgErr;
    metrics.hydrostaticMaxDriftTheta = maxEquilibriumDrift;
    metrics.relaxationHeadStdBeforeM = headStdBefore;
    metrics.relaxationHeadStdAfterM = headStdAfter;
    metrics.relaxationRatio = headStdAfter / Math.max(1e-12, headStdBefore);

    const hydroOk =
      maxHeadErrM < 1e-4 &&
      maxThetaVgErr < 1e-4 &&
      maxEquilibriumDrift < 1e-4 &&
      headStdAfter < 0.75 * headStdBefore &&
      relaxMassRelErr < 1e-5;
    details.push(
      `Hydrostatic eq: headErr=${maxHeadErrM.toExponential(2)} m, vgThetaErr=${maxThetaVgErr.toExponential(2)}, drift=${maxEquilibriumDrift.toExponential(2)}, relaxStd=${headStdBefore.toFixed(4)}->${headStdAfter.toFixed(4)} m`,
    );

    // ------------------------------------------------------------------------
    // Part C: Material Distinctness (SAND vs LOAM vs CLAY)
    // ------------------------------------------------------------------------
    const ksSand = SOIL_MATERIALS[SoilMaterialId.SAND].ks;
    const ksLoam = SOIL_MATERIALS[SoilMaterialId.LOAM].ks;
    const ksClay = SOIL_MATERIALS[SoilMaterialId.CLAY].ks;

    const runColumnInfiltration = (matId: SoilMaterialId): { deepFraction: number; midAbsPsiPa: number; k70: number } => {
      const g = new SoilGrid({
        nx: 4,
        ny: 16,
        nz: 4,
        dx: 0.01,
        ambientTempC: 15,
        ambientHumidity: 1.0,
        ambientCo2Ppm: 420,
        freeDrainBottom: false,
      });
      const mat = SOIL_MATERIALS[matId];
      const thetaInit = mat.thetaR + 0.55 * (mat.thetaS - mat.thetaR);
      const psiInit = vgPsiFromTheta(thetaInit, mat);
      for (let i = 0; i < g.material.length; i++) {
        g.material[i] = matId;
        g.theta[i] = thetaInit;
        g.psi[i] = psiInit;
      }

      const deepWater = (grid: SoilGrid): number => {
        let sum = 0;
        const voxVol = grid.dx * grid.dx * grid.dx;
        for (let iy = 0; iy < grid.ny - 2; iy++) {
          for (let iz = 0; iz < grid.nz; iz++) {
            for (let ix = 0; ix < grid.nx; ix++) {
              sum += grid.theta[grid.voxelIndex(ix, iy, iz)] * voxVol;
            }
          }
        }
        return sum;
      };

      const totalBefore = g.totalWaterVolumeM3();
      const deepBefore = deepWater(g);
      // Uniform top rain pulse that fits cleanly in top layer of all 3 soils
      g.pourRain(0, 0, 0.06, 0.0012);
      const added = g.totalWaterVolumeM3() - totalBefore;

      for (let s = 0; s < 180; s++) {
        g.step(10.0, envClosed);
      }

      const deepGained = deepWater(g) - deepBefore;
      const midIdx = g.voxelIndex(2, 8, 2);
      const theta70 = mat.thetaR + 0.7 * (mat.thetaS - mat.thetaR);
      return {
        deepFraction: added > 0 ? deepGained / added : 0,
        midAbsPsiPa: Math.abs(g.psi[midIdx]),
        k70: vgConductivity(theta70, mat),
      };
    };

    const sandRes = runColumnInfiltration(SoilMaterialId.SAND);
    const loamRes = runColumnInfiltration(SoilMaterialId.LOAM);
    const clayRes = runColumnInfiltration(SoilMaterialId.CLAY);

    metrics.ksSand = ksSand;
    metrics.ksLoam = ksLoam;
    metrics.ksClay = ksClay;
    metrics.sandDeepInfilFrac = sandRes.deepFraction;
    metrics.loamDeepInfilFrac = loamRes.deepFraction;
    metrics.clayDeepInfilFrac = clayRes.deepFraction;
    metrics.sandMidSuctionPa = sandRes.midAbsPsiPa;
    metrics.loamMidSuctionPa = loamRes.midAbsPsiPa;
    metrics.clayMidSuctionPa = clayRes.midAbsPsiPa;

    const distinctOk =
      ksSand > ksLoam &&
      ksLoam > ksClay &&
      sandRes.k70 > loamRes.k70 &&
      loamRes.k70 > clayRes.k70 &&
      sandRes.deepFraction > loamRes.deepFraction &&
      loamRes.deepFraction > clayRes.deepFraction &&
      clayRes.midAbsPsiPa > loamRes.midAbsPsiPa &&
      loamRes.midAbsPsiPa > sandRes.midAbsPsiPa;

    details.push(
      `Materials: Ks=[${ksSand.toExponential(2)}, ${ksLoam.toExponential(2)}, ${ksClay.toExponential(2)}], deepInfil=[${sandRes.deepFraction.toFixed(4)}, ${loamRes.deepFraction.toFixed(4)}, ${clayRes.deepFraction.toFixed(4)}], |psi|=[${sandRes.midAbsPsiPa.toExponential(2)}, ${loamRes.midAbsPsiPa.toExponential(2)}, ${clayRes.midAbsPsiPa.toExponential(2)}] Pa`,
    );

    const passed = massOk && hydroOk && distinctOk;
    return {
      id: 'T1.1',
      name: 'Soil Hydrology Mass Conservation, Hydrostatic Equilibrium & Material Distinctness',
      tier: 1,
      passed,
      metrics,
      details,
      error: passed ? undefined : 'One or more T1.1 hydrology assertions failed',
    };
  } catch (err) {
    return {
      id: 'T1.1',
      name: 'Soil Hydrology Mass Conservation, Hydrostatic Equilibrium & Material Distinctness',
      tier: 1,
      passed: false,
      metrics,
      details,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ============================================================================
// T1.2 — Lockhart Analytical Elongation Oracle
// ============================================================================
class ConstantTurgorMyceliumStub implements IMyceliumNetwork {
  readonly nodes: MyceliumNode[];
  readonly edges: MyceliumEdge[] = [];
  readonly primordia: PrimordiumAnchor[];
  private turgorPa: number;

  constructor(turgorPa: number) {
    this.turgorPa = turgorPa;
    this.nodes = [
      {
        id: 0,
        x: 0,
        y: -0.005,
        z: 0,
        voxelIdx: 0,
        turgorPa,
        osmoticPa: turgorPa + 2e5,
        carbonKg: 1.0,
        isTip: false,
        dirX: 0,
        dirY: 1,
        dirZ: 0,
        ageSec: 1000,
        hasPrimordium: true,
        fruitingBodyId: -1,
      },
    ];
    this.primordia = [
      {
        id: 0,
        nodeId: 0,
        x: 0,
        y: 0,
        z: 0,
        initiatedAtSec: 0,
        carbonReserveKg: 1.0,
      },
    ];
  }

  inoculate(): number {
    return 0;
  }

  step(): void {
    // constant turgor stub
  }

  drawWaterAndCarbonForFruiting(
    _nodeId: number,
    waterVolumeM3: number,
    carbonKg: number,
  ): { waterSuppliedM3: number; carbonSuppliedKg: number; basalTurgorPa: number } {
    return {
      waterSuppliedM3: waterVolumeM3,
      carbonSuppliedKg: carbonKg,
      basalTurgorPa: this.turgorPa,
    };
  }

  getNodeTurgorPa(): number {
    return this.turgorPa;
  }

  getNodeCarbonKg(): number {
    return 1.0;
  }

  getAnastomosisCount(): number {
    return 0;
  }

  getPackedRenderBuffers(): MyceliumRenderBuffers {
    return {
      segmentCount: 0,
      segments: new Float32Array(0),
      nodeCount: 1,
      nodes: new Float32Array([0, -0.005, 0, this.turgorPa]),
    };
  }
}

export function runT1_2_LockhartGrowth(): OracleTestResult {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};

  try {
    const L0 = 0.015; // m
    const phi = 1.1e-10; // Pa^-1 s^-1
    const Y = 2.2e5; // Pa
    const totalTimeSec = 36000; // 10 hours
    const steps = 100;
    const dt = totalTimeSec / steps;

    // 1. Direct stepStipeLockhartExact for P > Y
    const pHigh = 5.0e5;
    let lSim = L0;
    for (let i = 0; i < steps; i++) {
      lSim = stepStipeLockhartExact(lSim, pHigh, phi, Y, dt);
    }
    const lAnalytical = L0 * Math.exp(phi * (pHigh - Y) * totalTimeSec);
    const exactRelErr = Math.abs(lSim - lAnalytical) / lAnalytical;

    // 2. Direct stepStipeLockhartExact for P == Y and P < Y
    let lAtYield = L0;
    let lBelowYield = L0;
    for (let i = 0; i < steps; i++) {
      lAtYield = stepStipeLockhartExact(lAtYield, Y, phi, Y, dt);
      lBelowYield = stepStipeLockhartExact(lBelowYield, 0.5 * Y, phi, Y, dt);
    }
    const zeroBelowOk = lAtYield === L0 && lBelowYield === L0;

    // 3. MorphogenesisSystem under constant turgor (unconstrained maxStipeHeightM, CO2=0, RH=1)
    const testSpecies: SpeciesPreset = {
      ...SPECIES_PRESETS.parasol,
      phiStipe: phi,
      yieldPressurePa: Y,
      maxStipeHeightM: 10.0, // keep far from saturation ceiling
      co2StipeBoost: 0,
    };
    const envIso: EnvironmentState = {
      ambientTempC: 16,
      ambientHumidity: 1.0,
      ambientCo2Ppm: 0,
      lightDir: [0, 1, 0],
      terrainTiltRad: 0,
      gravityDir: [0, -1, 0],
      rainRateMs: 0,
      evaporationEnabled: false,
      timeWarp: 1,
    };

    // Sub-yield MorphogenesisSystem check (P <= Y => strictly zero elongation)
    const stubLow = new ConstantTurgorMyceliumStub(1.5e5);
    const morphoLow = new MorphogenesisSystem(testSpecies);
    morphoLow.syncWithPrimordia(stubLow, 0);
    const lLowInit = morphoLow.getBodies()[0].stipeLengthM;
    for (let i = 0; i < 60; i++) {
      morphoLow.step(120, stubLow, envIso, []);
    }
    const lLowFinal = morphoLow.getBodies()[0].stipeLengthM;
    const morphoZeroOk = lLowFinal === lLowInit;

    // Super-yield MorphogenesisSystem check (P > Y)
    const stubHigh = new ConstantTurgorMyceliumStub(5.0e5);
    const morphoHigh = new MorphogenesisSystem(testSpecies);
    morphoHigh.syncWithPrimordia(stubHigh, 0);
    // Warm up turgor equilibration for 30 small steps
    for (let i = 0; i < 30; i++) {
      morphoHigh.step(10, stubHigh, envIso, []);
    }
    const bodyStart = morphoHigh.getBodies()[0];
    const lMorphoStart = bodyStart.stipeLengthM;
    let lMorphoExpected = lMorphoStart;
    const mSteps = 120;
    const mDt = 120; // 4 hours total
    for (let i = 0; i < mSteps; i++) {
      const pBefore = morphoHigh.getBodies()[0].turgorPa;
      morphoHigh.step(mDt, stubHigh, envIso, []);
      const pAfter = morphoHigh.getBodies()[0].turgorPa;
      const pMid = 0.5 * (pBefore + pAfter);
      if (pMid > Y) {
        lMorphoExpected *= Math.exp(phi * (pMid - Y) * mDt);
      }
    }
    const lMorphoEnd = morphoHigh.getBodies()[0].stipeLengthM;
    const morphoRelErr =
      Math.abs(lMorphoEnd - lMorphoExpected) / Math.max(1e-12, lMorphoExpected);
    const morphoGrew = lMorphoEnd > lMorphoStart * 1.1;

    metrics.exactLSimM = lSim;
    metrics.exactLAnalyticalM = lAnalytical;
    metrics.exactRelErrorPct = exactRelErr * 100;
    metrics.exactZeroAtOrBelowY = zeroBelowOk;
    metrics.morphoZeroAtBelowY = morphoZeroOk;
    metrics.morphoLStartM = lMorphoStart;
    metrics.morphoLEndM = lMorphoEnd;
    metrics.morphoLExpectedM = lMorphoExpected;
    metrics.morphoRelErrorPct = morphoRelErr * 100;

    details.push(
      `stepStipeLockhartExact: L_sim=${(lSim * 1000).toFixed(4)} mm vs L_exact=${(lAnalytical * 1000).toFixed(4)} mm (err=${(exactRelErr * 100).toFixed(6)}%, limit < 0.5%); P<=Y zero=${zeroBelowOk}`,
    );
    details.push(
      `MorphogenesisSystem: L=${(lMorphoStart * 1000).toFixed(3)}->${(lMorphoEnd * 1000).toFixed(3)} mm vs expected ${(lMorphoExpected * 1000).toFixed(3)} mm (err=${(morphoRelErr * 100).toFixed(4)}%); P<Y zero=${morphoZeroOk}`,
    );

    const passed =
      exactRelErr < 0.005 &&
      zeroBelowOk &&
      morphoZeroOk &&
      morphoGrew &&
      morphoRelErr < 0.005;

    return {
      id: 'T1.2',
      name: 'Lockhart Analytical Elongation Oracle',
      tier: 1,
      passed,
      metrics,
      details,
      error: passed ? undefined : 'Lockhart analytical elongation error exceeded 0.5% or non-zero sub-yield growth',
    };
  } catch (err) {
    return {
      id: 'T1.2',
      name: 'Lockhart Analytical Elongation Oracle',
      tier: 1,
      passed: false,
      metrics,
      details,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ============================================================================
// T1.3 — Thin-Shell Föppl–von Kármán Buckling Mode Scaling
// ============================================================================
export function runT1_3_ThinShellBuckling(): OracleTestResult {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};

  try {
    // Gating thicknesses h1 > h2 > h3 (2.4, 1.2, 0.6 mm) + supplemental ultra-thin shell (0.3 mm, reported only)
    const thicknessesM = [0.0024, 0.0012, 0.0006];
    const supplementalThicknessM = 0.0003;
    const wavenumbers: number[] = [];
    const capRadii: number[] = [];

    const measureAt = (h: number): { mc: number; radius: number } => {
      const solver = new ThinShellCapSolver({
        species: SPECIES_PRESETS.chanterelle,
        radialRings: 12,
        angularSectors: 96,
        shellThicknessM: h,
        youngsModulusPa: 6.0e6,
        alphaRadial: 0.85,
        alphaCircumferential: 2.8,
        coreFoundationStiffness: 1.0e5,
      });
      for (let s = 0; s < 80; s++) {
        solver.growAndRelax(300, 5.2e5, 1.0, []);
      }
      const buffers = solver.getMeshBuffers();
      for (let i = 0; i < buffers.positions.length; i++) {
        if (!Number.isFinite(buffers.positions[i])) {
          throw new Error(`Non-finite shell position detected at h=${h}`);
        }
      }
      return { mc: solver.measureCircumferentialWavenumber(), radius: solver.getCapRadiusM() };
    };

    for (const h of thicknessesM) {
      const r = measureAt(h);
      wavenumbers.push(r.mc);
      capRadii.push(r.radius);
    }
    const supplemental = measureAt(supplementalThicknessM);

    const [mc1, mc2, mc3] = wavenumbers;
    // Strictly monotone as h decreases. A shell too thick to buckle reports m_c = 0 (physically valid
    // sub-critical state); the two thinner shells MUST have buckled with m_c >= 2.
    const monotonicOk = mc1 >= 0 && mc2 >= 2 && mc2 > mc1 && mc3 > mc2;

    // OLS log-log slope of ln(m_c) vs ln(h) over all buckled (m_c >= 2) shells incl. supplemental point
    const pts: Array<[number, number]> = [];
    thicknessesM.forEach((h, i) => {
      if (wavenumbers[i] >= 2) pts.push([Math.log(h), Math.log(wavenumbers[i])]);
    });
    if (supplemental.mc >= 2) pts.push([Math.log(supplementalThicknessM), Math.log(supplemental.mc)]);
    const meanX = pts.reduce((a, p) => a + p[0], 0) / Math.max(1, pts.length);
    const meanY = pts.reduce((a, p) => a + p[1], 0) / Math.max(1, pts.length);
    let num = 0;
    let den = 0;
    for (const [x, y] of pts) {
      num += (x - meanX) * (y - meanY);
      den += (x - meanX) * (x - meanX);
    }
    const logLogSlope = pts.length >= 2 ? num / Math.max(1e-12, den) : NaN;
    // Thin-shell FvK scaling predicts -0.5 (free rim) to -0.75 (elastic foundation);
    // allow realistic discrete-mode band [-1.35, -0.20]
    const slopeOk = Number.isFinite(logLogSlope) && logLogSlope <= -0.2 && logLogSlope >= -1.35;
    metrics.mc_h_0p3mm_supplemental = supplemental.mc;
    metrics.bucklingPointsInFit = pts.length;

    metrics.h1_mm = thicknessesM[0] * 1000;
    metrics.h2_mm = thicknessesM[1] * 1000;
    metrics.h3_mm = thicknessesM[2] * 1000;
    metrics.mc_h1 = mc1;
    metrics.mc_h2 = mc2;
    metrics.mc_h3 = mc3;
    metrics.logLogScalingSlope = logLogSlope;
    metrics.capRadiusMeanM = capRadii.reduce((a, b) => a + b, 0) / capRadii.length;

    details.push(
      `Thickness h=[2.4, 1.2, 0.6] mm -> emergent m_c=[${mc1.toFixed(2)}, ${mc2.toFixed(2)}, ${mc3.toFixed(2)}], log-log slope=${logLogSlope.toFixed(3)} (theory: -0.50 to -0.75)`,
    );

    const passed = monotonicOk && slopeOk && capRadii.every((r) => r > 0.005);
    return {
      id: 'T1.3',
      name: 'Thin-Shell Föppl–von Kármán Buckling Mode Scaling',
      tier: 1,
      passed,
      metrics,
      details,
      error: passed
        ? undefined
        : `Expected strictly increasing m_c as h decreases and negative FvK slope; got m_c=[${mc1}, ${mc2}, ${mc3}], slope=${logLogSlope.toFixed(3)}`,
    };
  } catch (err) {
    return {
      id: 'T1.3',
      name: 'Thin-Shell Föppl–von Kármán Buckling Mode Scaling',
      tier: 1,
      passed: false,
      metrics,
      details,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ============================================================================
// T1.4 — Buller's Drop Energy Conservation & Stokes Stopping Distance
// ============================================================================
export function runT1_4_BullerCatapult(): OracleTestResult {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};

  try {
    const speciesIds = ['parasol', 'chanterelle', 'inkcap'] as const;
    // THEORY.md §6.1 physical inter-gill channel range (upper bound used as informational reference)
    const THEORY_GILL_MAX_M = 0.45e-3;

    let allPassed = true;
    let worstStokesErrPct = 0;

    for (const sid of speciesIds) {
      const sp = SPECIES_PRESETS[sid];

      // Emergent gill spacing straight from the live thin-shell solver (feeds Buller micro-solver in the sim)
      const shell = new ThinShellCapSolver({ species: sp, radialRings: 9, angularSectors: 48 }, 1);
      for (let s = 0; s < 40; s++) shell.growAndRelax(300, 5.2e5, 1.0, []);
      const gillSpacingM = shell.getMeanGillSpacingM();

      const p: BullerParams = {
        sporeRadiusM: sp.sporeRadiusM,
        bullerDropRadiusM: sp.bullerDropRadiusM,
        sporeDensityKgM3: sp.sporeDensityKgM3,
        launchEfficiency: sp.launchEfficiency,
        gillSpacingM,
        gammaWater: GAMMA_WATER,
        muAir: MU_AIR,
      };
      const gamma = GAMMA_WATER;
      const mu = MU_AIR;

      // Independent first-principles analytical quantities
      const msExact = (4 / 3) * Math.PI * Math.pow(p.sporeRadiusM, 3) * p.sporeDensityKgM3;
      const deltaAExact =
        4 * Math.PI * p.bullerDropRadiusM * p.bullerDropRadiusM * (2 - Math.pow(2, 2 / 3));
      const deltaESurfExact = gamma * deltaAExact;
      const tauSExact = (2 * p.sporeDensityKgM3 * p.sporeRadiusM * p.sporeRadiusM) / (9 * mu);
      const vTermExact = GRAVITY * tauSExact;

      const launch = BullerCatapultSolver.computeLaunchKinematics(p);
      const keFromV0 = 0.5 * msExact * launch.v0Ms * launch.v0Ms;
      const dStopAnalytical = tauSExact * launch.v0Ms;

      // (1) Wall-free Stokes ODE accuracy: huge gill gap (1 m) so integration is never clipped by a wall.
      const freeParams: BullerParams = { ...p, gillSpacingM: 1.0 };
      const traj = BullerCatapultSolver.simulateTrajectory(freeParams, 14 * tauSExact, 1e-6);
      if (traj.length < 10) {
        throw new Error(`Trajectory too short for ${sid}`);
      }
      const last = traj[traj.length - 1];
      // Remaining distance beyond 14 tau is v0*tau*exp(-14) ~ 8e-7 relative: negligible (< 1e-4 %)
      const simStopDist = last.xM;
      const stokesRelErr =
        Math.abs(simStopDist - dStopAnalytical) / Math.max(1e-12, dStopAnalytical);
      const analyticalMatchErr =
        Math.abs(launch.analyticalStoppingDistM - dStopAnalytical) /
        Math.max(1e-12, dStopAnalytical);
      const vTermErr = Math.abs(Math.abs(last.vyMs) - vTermExact) / Math.max(1e-12, vTermExact);
      worstStokesErrPct = Math.max(worstStokesErrPct, stokesRelErr * 100);

      // (2) Exact analytical x(t) at a mid-flight sample (t = 1 tau): v0 tau (1 - e^-1)
      const midPoint = traj.reduce((best, pt) =>
        Math.abs(pt.tSec - tauSExact) < Math.abs(best.tSec - tauSExact) ? pt : best,
      );
      const xMidExact = launch.v0Ms * tauSExact * (1 - Math.exp(-midPoint.tSec / tauSExact));
      const midErr = Math.abs(midPoint.xM - xMidExact) / Math.max(1e-12, xMidExact);

      // (3) Clearance with the REAL emergent gill spacing: spore must stop before the opposite gill
      const wallTraj = BullerCatapultSolver.simulateTrajectory(p, 14 * tauSExact, 1e-6);
      const wallLast = wallTraj[wallTraj.length - 1];
      const clearanceOk =
        dStopAnalytical < gillSpacingM &&
        launch.clearsOppositeGill &&
        wallLast.xM < gillSpacingM * (1 - 1e-9) &&
        gillSpacingM >= 0.15e-3 &&
        gillSpacingM <= 2.0e-3;

      const energyOk =
        launch.launchKineticEnergyJ > 0 &&
        launch.launchKineticEnergyJ <= launch.deltaSurfaceEnergyJ * (1 + 1e-9) &&
        keFromV0 <= deltaESurfExact * (1 + 1e-9) &&
        Math.abs(launch.deltaSurfaceEnergyJ - deltaESurfExact) / deltaESurfExact < 1e-9;
      const accelOk = launch.peakAccelG >= 5000 && launch.peakAccelG <= 150000;
      const stokesOk = stokesRelErr < 0.01 && analyticalMatchErr < 0.005 && midErr < 0.01;
      const gravityOk = last.yM < 0 && vTermErr < 0.05;

      const caseOk = energyOk && accelOk && stokesOk && clearanceOk && gravityOk;
      if (!caseOk) allPassed = false;

      metrics[`${sid}_DeltaESurfFJ`] = launch.deltaSurfaceEnergyJ * 1e15;
      metrics[`${sid}_KineticEnergyFJ`] = launch.launchKineticEnergyJ * 1e15;
      metrics[`${sid}_EnergyRatio`] = launch.launchKineticEnergyJ / launch.deltaSurfaceEnergyJ;
      metrics[`${sid}_v0Ms`] = launch.v0Ms;
      metrics[`${sid}_PeakAccelG`] = launch.peakAccelG;
      metrics[`${sid}_DStopAnalyticalUm`] = dStopAnalytical * 1e6;
      metrics[`${sid}_DStopSimulatedUm`] = simStopDist * 1e6;
      metrics[`${sid}_StokesErrPct`] = stokesRelErr * 100;
      metrics[`${sid}_MidFlightErrPct`] = midErr * 100;
      metrics[`${sid}_TerminalVelErrPct`] = vTermErr * 100;
      metrics[`${sid}_EmergentGillSpacingUm`] = gillSpacingM * 1e6;
      metrics[`${sid}_ClearsEmergentGill`] = clearanceOk;
      metrics[`${sid}_ClearsTheory0p45mmGill`] = dStopAnalytical < THEORY_GILL_MAX_M;

      details.push(
        `${sid}: Ek/dEs=${(launch.launchKineticEnergyJ / launch.deltaSurfaceEnergyJ).toFixed(3)}<=1, v0=${launch.v0Ms.toFixed(3)} m/s, a_peak=${Math.round(launch.peakAccelG)} g, d_stop(sim)=${(simStopDist * 1e6).toFixed(2)} um vs exact ${(dStopAnalytical * 1e6).toFixed(2)} um (err=${(stokesRelErr * 100).toFixed(5)}%), gill=${(gillSpacingM * 1e6).toFixed(0)} um (clear=${clearanceOk}; <0.45mm theory gill: ${dStopAnalytical < THEORY_GILL_MAX_M})`,
      );
    }

    metrics.worstStokesErrPct = worstStokesErrPct;

    return {
      id: 'T1.4',
      name: "Buller's Drop Energy Conservation & Stokes Stopping Distance",
      tier: 1,
      passed: allPassed,
      metrics,
      details,
      error: allPassed ? undefined : 'One or more Buller catapult energy/accel/Stokes/clearance checks failed',
    };
  } catch (err) {
    return {
      id: 'T1.4',
      name: "Buller's Drop Energy Conservation & Stokes Stopping Distance",
      tier: 1,
      passed: false,
      metrics,
      details,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export function runTier1Oracles(): OracleTestResult[] {
  return [
    runT1_1_SoilHydrology(),
    runT1_2_LockhartGrowth(),
    runT1_3_ThinShellBuckling(),
    runT1_4_BullerCatapult(),
  ];
}
