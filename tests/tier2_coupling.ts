// ============================================================================
// SPOROCARP — Tier 2 Emergent Coupling & Falsification Tests
// Owned exclusively by Worker 4 (biophysics-auditor). Locked by ORACLE.sha256.
// ============================================================================

import {
  SPECIES_PRESETS,
  SoilMaterialId,
  type EnvironmentState,
  type FruitingBodyState,
  type PrimordiumAnchor,
  type SpeciesPreset,
  type TerrariumBounds,
} from '../src/types/contract.ts';
import { SoilGrid } from '../src/sim/soil/SoilGrid.ts';
import { SOIL_MATERIALS, vgPsiFromTheta } from '../src/sim/soil/vanGenuchten.ts';
import { MyceliumNetwork } from '../src/sim/mycelium/MyceliumNetwork.ts';
import { MorphogenesisSystem } from '../src/sim/morpho/MorphogenesisSystem.ts';
import { SporePhysicsSystem } from '../src/sim/spores/SporePhysicsSystem.ts';
import type { OracleTestResult } from './tier1_oracles.ts';

function makeBaseEnv(overrides: Partial<EnvironmentState> = {}): EnvironmentState {
  return {
    ambientTempC: 16,
    ambientHumidity: 0.88,
    ambientCo2Ppm: 420,
    lightDir: [0, 1, 0],
    terrainTiltRad: 0,
    gravityDir: [0, -1, 0],
    rainRateMs: 0,
    evaporationEnabled: true,
    timeWarp: 1,
    ...overrides,
  };
}

// Tilted-terrain gravity / anti-gravity helpers (tilt about the Z axis)
function tiltedGravity(tiltRad: number): [number, number, number] {
  return [Math.sin(tiltRad), -Math.cos(tiltRad), 0];
}

interface CoupledHarness {
  soil: SoilGrid;
  mycelium: MyceliumNetwork;
  morpho: MorphogenesisSystem;
  env: EnvironmentState;
  species: SpeciesPreset;
  simTimeSec: number;
}

const HARNESS_DT_SEC = 600; // 10 min coupled step

/**
 * Standard coupled terrarium: homogeneous 'fairy-ring-meadow' soil (36 x 12 x 36 voxels, 36 cm wide),
 * a single exact-center inoculation, no hand-placed primordia: fruiting bodies emerge only
 * from the mycelium's own pinning rule.
 */
function createColony(
  species: SpeciesPreset,
  envOverrides: Partial<EnvironmentState> = {},
): CoupledHarness {
  const env = makeBaseEnv(envOverrides);
  const soil = new SoilGrid({
    nx: 36,
    ny: 12,
    nz: 36,
    dx: 0.01,
    ambientTempC: env.ambientTempC,
    ambientHumidity: env.ambientHumidity,
    ambientCo2Ppm: env.ambientCo2Ppm,
    freeDrainBottom: false,
  });
  soil.setupPresetTerrarium('fairy-ring-meadow');
  const mycelium = new MyceliumNetwork(soil, species);
  mycelium.inoculate(0, -0.02, 0);
  const morpho = new MorphogenesisSystem(species);
  return { soil, mycelium, morpho, env, species, simTimeSec: 0 };
}

function stepCoupled(h: CoupledHarness, dtSec: number = HARNESS_DT_SEC): void {
  h.soil.step(dtSec, h.env);
  h.mycelium.step(dtSec, h.env, h.species);
  h.simTimeSec += dtSec;
  h.morpho.syncWithPrimordia(h.mycelium, h.simTimeSec);
  h.morpho.step(dtSec, h.mycelium, h.env, []);
}

function leadBody(h: CoupledHarness): FruitingBodyState | undefined {
  return h.morpho.getBodies()[0];
}

/** Step until the lead body exists and has elongated to at least `minLengthM` (or the step cap is hit). */
function growUntilLeadLength(h: CoupledHarness, minLengthM: number, maxSteps = 1800): boolean {
  for (let s = 0; s < maxSteps; s++) {
    stepCoupled(h);
    const b = leadBody(h);
    if (b && b.stipeLengthM >= minLengthM) return true;
  }
  return false;
}

function meanNodeO2(h: CoupledHarness): number {
  const nodes = h.mycelium.nodes;
  if (nodes.length === 0) return 0;
  let sum = 0;
  for (const n of nodes) sum += h.soil.o2[n.voxelIdx];
  return sum / nodes.length;
}

function setUniformSaturation(h: CoupledHarness, mode: 'dry' | 'flood'): void {
  for (let i = 0; i < h.soil.material.length; i++) {
    const matId = h.soil.material[i] as SoilMaterialId;
    if (matId === SoilMaterialId.STONE) continue;
    const mat = SOIL_MATERIALS[matId];
    const theta = mode === 'dry' ? mat.thetaR + 0.01 : mat.thetaS - 1e-4;
    h.soil.theta[i] = theta;
    h.soil.psi[i] = vgPsiFromTheta(theta, mat);
  }
}

interface StressTrace {
  preO2: number;
  preLengthM: number;
  lengthM: number[]; // per-step lead length
  turgorPa: number[];
  wilt: number[];
  o2: number[];
}

/** Grow a colony to an elongating lead body, optionally apply a stress, then record `steps` steps. */
function runStressBranch(
  species: SpeciesPreset,
  stress: 'none' | 'dry' | 'flood',
  steps: number,
): StressTrace {
  const h = createColony(species);
  const ok = growUntilLeadLength(h, 0.014);
  if (!ok) {
    throw new Error('Colony never produced an elongating lead fruiting body (>= 14 mm) in healthy soil');
  }
  const pre = leadBody(h)!;
  const trace: StressTrace = {
    preO2: meanNodeO2(h),
    preLengthM: pre.stipeLengthM,
    lengthM: [],
    turgorPa: [],
    wilt: [],
    o2: [],
  };
  if (stress !== 'none') setUniformSaturation(h, stress);
  for (let s = 0; s < steps; s++) {
    stepCoupled(h);
    const b = leadBody(h)!;
    trace.lengthM.push(b.stipeLengthM);
    trace.turgorPa.push(b.turgorPa);
    trace.wilt.push(b.wiltFactor);
    trace.o2.push(meanNodeO2(h));
  }
  return trace;
}

const mean = (a: number[]): number => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);

// ============================================================================
// T2.1 — Soil-Drought & Hypoxia Wilt Coupling
// ============================================================================
export function runT2_1_DroughtHypoxiaWilt(): OracleTestResult {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};

  try {
    const species = SPECIES_PRESETS.parasol;
    const Y = species.yieldPressurePa;
    const WINDOW_STEPS = 72; // 12 h post-stress observation window
    const EARLY_STEPS = 30; // first 5 h (hypoxia is transient: aqueous O2 slowly refills)

    // Identical deterministic colony in three branches: control / drought / flood
    const ctrl = runStressBranch(species, 'none', WINDOW_STEPS);
    const drought = runStressBranch(species, 'dry', WINDOW_STEPS);
    const flood = runStressBranch(species, 'flood', WINDOW_STEPS);

    const ctrlGrowthMm = (ctrl.lengthM[WINDOW_STEPS - 1] - ctrl.preLengthM) * 1000;
    const ctrlEarlyGrowthMm = (ctrl.lengthM[EARLY_STEPS - 1] - ctrl.preLengthM) * 1000;
    const ctrlPeakP = Math.max(...ctrl.turgorPa);
    const ctrlMaxWiltEarly = Math.max(...ctrl.wilt.slice(0, EARLY_STEPS));

    // Branch A: Drought
    const droughtGrowthMm = (drought.lengthM[WINDOW_STEPS - 1] - drought.preLengthM) * 1000;
    const droughtPeakPLate = Math.max(...drought.turgorPa.slice(WINDOW_STEPS - 36));
    const droughtMeanWilt = mean(drought.wilt.slice(WINDOW_STEPS - 36));
    const ctrlMeanWilt = mean(ctrl.wilt.slice(WINDOW_STEPS - 36));
    const droughtOk =
      droughtPeakPLate < 0.5 * Y &&
      droughtMeanWilt > ctrlMeanWilt + 0.3 &&
      droughtGrowthMm < 0.1 * ctrlGrowthMm;

    // Branch B: Hypoxic flood
    const floodMinO2 = Math.min(...flood.o2);
    const floodO2Ratio = floodMinO2 / Math.max(1e-9, flood.preO2);
    const floodMinP = Math.min(...flood.turgorPa.slice(0, EARLY_STEPS));
    const floodMaxPEarly = Math.max(...flood.turgorPa.slice(0, EARLY_STEPS));
    const floodEarlyGrowthMm = (flood.lengthM[EARLY_STEPS - 1] - flood.preLengthM) * 1000;
    const floodMaxWiltEarly = Math.max(...flood.wilt.slice(0, EARLY_STEPS));
    const hypoxiaOk =
      floodO2Ratio < 0.5 &&
      floodMinP < 0.6 * Y &&
      floodMaxWiltEarly > ctrlMaxWiltEarly + 0.1 &&
      floodEarlyGrowthMm < 0.4 * ctrlEarlyGrowthMm;

    metrics.ctrlPeakTurgorKPa = ctrlPeakP / 1e3;
    metrics.ctrlGrowthMm12h = ctrlGrowthMm;
    metrics.ctrlEarlyGrowthMm5h = ctrlEarlyGrowthMm;
    metrics.ctrlMeanWiltLate = ctrlMeanWilt;
    metrics.ctrlMaxWiltEarly = ctrlMaxWiltEarly;
    metrics.droughtPeakTurgorLateKPa = droughtPeakPLate / 1e3;
    metrics.droughtMeanWiltLate = droughtMeanWilt;
    metrics.droughtGrowthMm12h = droughtGrowthMm;
    metrics.preFloodO2MolM3 = flood.preO2;
    metrics.floodMinO2MolM3 = floodMinO2;
    metrics.floodO2RatioMin = floodO2Ratio;
    metrics.floodEarlyMinTurgorKPa = floodMinP / 1e3;
    metrics.floodEarlyPeakTurgorKPa = floodMaxPEarly / 1e3;
    metrics.floodEarlyMaxWilt = floodMaxWiltEarly;
    metrics.floodEarlyGrowthMm5h = floodEarlyGrowthMm;
    metrics.yieldKPa = Y / 1e3;

    details.push(
      `Control: peak P=${(ctrlPeakP / 1e3).toFixed(0)} kPa (Y=${(Y / 1e3).toFixed(0)}), growth 12h=${ctrlGrowthMm.toFixed(2)} mm, 5h=${ctrlEarlyGrowthMm.toFixed(2)} mm`,
    );
    details.push(
      `Drought (theta=thetaR+0.01): late peak P=${(droughtPeakPLate / 1e3).toFixed(1)} kPa (<0.5Y), wilt=${droughtMeanWilt.toFixed(2)} vs ctrl ${ctrlMeanWilt.toFixed(2)}, growth=${droughtGrowthMm.toFixed(3)} mm (<10% of ctrl)`,
    );
    details.push(
      `Hypoxia (theta=thetaS-1e-4): node O2 ${flood.preO2.toFixed(2)} -> min ${floodMinO2.toFixed(2)} mol/m3 (ratio ${floodO2Ratio.toFixed(2)} < 0.5), early min P=${(floodMinP / 1e3).toFixed(0)} kPa (<0.6Y), max wilt=${floodMaxWiltEarly.toFixed(2)} (ctrl ${ctrlMaxWiltEarly.toFixed(2)}, need +0.1), early growth=${floodEarlyGrowthMm.toFixed(3)} mm vs ctrl ${ctrlEarlyGrowthMm.toFixed(3)} mm`,
    );

    const passed = ctrlGrowthMm > 0.5 && ctrlPeakP > Y && droughtOk && hypoxiaOk;
    return {
      id: 'T2.1',
      name: 'Soil-Drought & Hypoxia Wilt Coupling',
      tier: 2,
      passed,
      metrics,
      details,
      error: passed
        ? undefined
        : `Coupling failure: ctrlGrew=${ctrlGrowthMm > 0.5}, ctrlTurgid=${ctrlPeakP > Y}, droughtOk=${droughtOk}, hypoxiaOk=${hypoxiaOk}`,
    };
  } catch (err) {
    return {
      id: 'T2.1',
      name: 'Soil-Drought & Hypoxia Wilt Coupling',
      tier: 2,
      passed: false,
      metrics,
      details,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ============================================================================
// T2.2 — Gravitropism Test (45 deg Substrate Tilt -> Curves within 10 deg)
// ============================================================================
export function runT2_2_Gravitropism(): OracleTestResult {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};

  try {
    const tiltRad = Math.PI / 4; // 45 degrees
    const g = tiltedGravity(tiltRad);
    const up: [number, number, number] = [-g[0], -g[1], -g[2]];

    // Isolate pure gravitropism (zero phototropism bias)
    const species: SpeciesPreset = { ...SPECIES_PRESETS.parasol, phototropismGain: 0 };

    const h = createColony(species, {
      terrainTiltRad: tiltRad,
      gravityDir: g,
      lightDir: up,
    });

    const angleFromUpDeg = (b: FruitingBodyState): number => {
      const norm = Math.hypot(b.stipeTipDirX, b.stipeTipDirY, b.stipeTipDirZ) || 1;
      const dot = Math.max(
        -1,
        Math.min(
          1,
          (b.stipeTipDirX * up[0] + b.stipeTipDirY * up[1] + b.stipeTipDirZ * up[2]) / norm,
        ),
      );
      return (Math.acos(dot) * 180) / Math.PI;
    };

    // Phase 1: grow until the lead stipe has elongated >= 2x its 8 mm pin (>= 16 mm)
    const reached = growUntilLeadLength(h, 0.016);
    if (!reached) {
      throw new Error('Lead stipe never elongated to 16 mm under 45 deg tilt');
    }

    // Phase 2: keep growing for 24 h, sampling the tip tangent every step
    let maxAngleTurgid = 0;
    let maxAngleAll = 0;
    let turgidSamples = 0;
    let finalLengthM = 0;
    let finalMaturity = 0;
    let finalAngleDeg = 0;
    for (let s = 0; s < 144; s++) {
      stepCoupled(h);
      const b = leadBody(h);
      if (!b) continue;
      const ang = angleFromUpDeg(b);
      maxAngleAll = Math.max(maxAngleAll, ang);
      // Wilting is a separate physical phenomenon (T2.1): the gravitropic set-point is judged
      // only while the stipe is turgid (wiltFactor < 0.1).
      if (b.wiltFactor < 0.1) {
        turgidSamples++;
        maxAngleTurgid = Math.max(maxAngleTurgid, ang);
      }
      finalLengthM = b.stipeLengthM;
      finalMaturity = b.maturity;
      finalAngleDeg = ang;
    }

    metrics.terrainTiltDeg = 45;
    metrics.untiltedAngleFromMinusGDeg = 45;
    metrics.maxTurgidTipAngleDeg = maxAngleTurgid;
    metrics.maxAnyTipAngleDeg = maxAngleAll;
    metrics.finalTipAngleDeg = finalAngleDeg;
    metrics.turgidSamples = turgidSamples;
    metrics.finalStipeLengthM = finalLengthM;
    metrics.finalMaturity = finalMaturity;

    details.push(
      `45 deg tilt: turgid stipe tip angle to -g: max=${maxAngleTurgid.toFixed(2)} deg over ${turgidSamples} turgid samples (limit < 10 deg); worst incl. wilt droop=${maxAngleAll.toFixed(1)} deg; L=${(finalLengthM * 100).toFixed(2)} cm, maturity=${finalMaturity.toFixed(2)}`,
    );

    const passed = turgidSamples >= 24 && maxAngleTurgid < 10.0 && finalLengthM >= 0.016;
    return {
      id: 'T2.2',
      name: 'Gravitropism Under 45° Substrate Tilt',
      tier: 2,
      passed,
      metrics,
      details,
      error: passed
        ? undefined
        : `Turgid stipe tip angle max=${maxAngleTurgid.toFixed(2)} deg (limit 10) with ${turgidSamples} turgid samples`,
    };
  } catch (err) {
    return {
      id: 'T2.2',
      name: 'Gravitropism Under 45° Substrate Tilt',
      tier: 2,
      passed: false,
      metrics,
      details,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ============================================================================
// T2.3 — CO2 Etiolation Test (420 ppm vs 2100 ppm)
// ============================================================================
export function runT2_3_Co2Etiolation(): OracleTestResult {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};

  try {
    const species = SPECIES_PRESETS.parasol;

    // Compare the lead body at an identical physiological age (36 h after first pinning/emergence)
    const AGE_AFTER_EMERGENCE_STEPS = 216;
    const runAtCo2 = (co2Ppm: number): { body: FruitingBodyState; emergedAtH: number } => {
      const h = createColony(species, { ambientCo2Ppm: co2Ppm });
      let emergedAtStep = -1;
      for (let step = 0; step < 2400; step++) {
        stepCoupled(h);
        if (leadBody(h) && emergedAtStep < 0) emergedAtStep = step;
        if (emergedAtStep >= 0 && step - emergedAtStep >= AGE_AFTER_EMERGENCE_STEPS) break;
      }
      if (emergedAtStep < 0) {
        throw new Error(`No fruiting body emerged at CO2=${co2Ppm} ppm`);
      }
      return { body: leadBody(h)!, emergedAtH: (emergedAtStep * HARNESS_DT_SEC) / 3600 };
    };

    const low = runAtCo2(420);
    const high = runAtCo2(2100);
    const bodyLow = low.body;
    const bodyHigh = high.body;

    const aspectLow = bodyLow.stipeLengthM / Math.max(1e-6, bodyLow.capRadiusM);
    const aspectHigh = bodyHigh.stipeLengthM / Math.max(1e-6, bodyHigh.capRadiusM);
    const pctIncrease = ((aspectHigh - aspectLow) / Math.max(1e-6, aspectLow)) * 100;

    metrics.lowCo2Ppm = 420;
    metrics.highCo2Ppm = 2100;
    metrics.lowEmergedAtH = low.emergedAtH;
    metrics.highEmergedAtH = high.emergedAtH;
    metrics.lowStipeM = bodyLow.stipeLengthM;
    metrics.lowCapRadiusM = bodyLow.capRadiusM;
    metrics.lowAspectRatio = aspectLow;
    metrics.highStipeM = bodyHigh.stipeLengthM;
    metrics.highCapRadiusM = bodyHigh.capRadiusM;
    metrics.highAspectRatio = aspectHigh;
    metrics.aspectIncreasePct = pctIncrease;

    details.push(
      `420 ppm: L=${(bodyLow.stipeLengthM * 100).toFixed(2)} cm, R=${(bodyLow.capRadiusM * 100).toFixed(2)} cm, L/R=${aspectLow.toFixed(2)}`,
    );
    details.push(
      `2100 ppm: L=${(bodyHigh.stipeLengthM * 100).toFixed(2)} cm, R=${(bodyHigh.capRadiusM * 100).toFixed(2)} cm, L/R=${aspectHigh.toFixed(2)} (+${pctIncrease.toFixed(1)}%, limit >= 25%)`,
    );

    const passed =
      bodyLow.capRadiusM > 0.005 &&
      bodyHigh.capRadiusM > 0.003 &&
      bodyHigh.stipeLengthM > 0.012 &&
      pctIncrease >= 25.0;

    return {
      id: 'T2.3',
      name: 'CO2 Etiolation Aspect-Ratio Shift (420 vs 2100 ppm)',
      tier: 2,
      passed,
      metrics,
      details,
      error: passed
        ? undefined
        : `Expected >= 25% higher stipe/cap aspect ratio at 2100 ppm; got ${pctIncrease.toFixed(1)}%`,
    };
  } catch (err) {
    return {
      id: 'T2.3',
      name: 'CO2 Etiolation Aspect-Ratio Shift (420 vs 2100 ppm)',
      tier: 2,
      passed: false,
      metrics,
      details,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ============================================================================
// T2.4 — Emergent Fairy Ring Test (Center Inoculation -> Annular Primordia)
// ============================================================================
export function runT2_4_FairyRing(): OracleTestResult {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};

  try {
    const species = SPECIES_PRESETS.parasol;
    const env = makeBaseEnv({ ambientHumidity: 0.88, ambientTempC: 16 });
    const soil = new SoilGrid({
      nx: 36,
      ny: 10,
      nz: 36,
      dx: 0.01,
      ambientTempC: env.ambientTempC,
      ambientHumidity: env.ambientHumidity,
      ambientCo2Ppm: env.ambientCo2Ppm,
      freeDrainBottom: false,
    });
    soil.setupPresetTerrarium('fairy-ring-meadow');

    const mycelium = new MyceliumNetwork(soil, species);
    // Exact center inoculation
    mycelium.inoculate(0, -0.02, 0);

    const dt = 1800; // 30 min per step
    for (let step = 0; step < 1200; step++) {
      soil.step(dt, env);
      mycelium.step(dt, env, species);
      if (mycelium.primordia.length >= 12) break;
    }

    // Measure central vs frontier soluble carbon in upper bioactive zone (top 4 cm)
    let centerCSum = 0;
    let centerCount = 0;
    let frontierCSum = 0;
    let frontierCount = 0;

    for (let iy = Math.max(0, soil.ny - 4); iy < soil.ny; iy++) {
      for (let iz = 0; iz < soil.nz; iz++) {
        for (let ix = 0; ix < soil.nx; ix++) {
          const idx = soil.voxelIndex(ix, iy, iz);
          if (soil.material[idx] === SoilMaterialId.STONE) continue;
          const [wx, , wz] = soil.voxelToWorld(ix, iy, iz);
          const r = Math.hypot(wx, wz);
          if (r < 0.025) {
            centerCSum += soil.solubleCarbon[idx];
            centerCount++;
          } else if (r >= 0.06 && r <= 0.13) {
            frontierCSum += soil.solubleCarbon[idx];
            frontierCount++;
          }
        }
      }
    }

    const centerCarbon = centerCount > 0 ? centerCSum / centerCount : 0;
    const frontierCarbon = frontierCount > 0 ? frontierCSum / frontierCount : 0;

    const radii: number[] = mycelium.primordia.map((p: PrimordiumAnchor) =>
      Math.hypot(p.x, p.z),
    );
    const primCount = radii.length;
    const meanRadius =
      primCount > 0 ? radii.reduce((acc: number, r: number) => acc + r, 0) / primCount : 0;
    const maxRadius = primCount > 0 ? Math.max(...radii) : 0;

    const coreR = 0.035;
    const outerR = Math.max(0.08, maxRadius + 0.015);
    const innerCount = radii.filter((r: number) => r < coreR).length;
    const outerCount = radii.filter((r: number) => r >= coreR).length;
    const innerDensity = innerCount / (Math.PI * coreR * coreR);
    const outerDensity = outerCount / (Math.PI * (outerR * outerR - coreR * coreR));

    metrics.myceliumNodes = mycelium.nodes.length;
    metrics.anastomosisCount = mycelium.getAnastomosisCount();
    metrics.primordiaCount = primCount;
    metrics.centerSolubleCarbonKgM3 = centerCarbon;
    metrics.frontierSolubleCarbonKgM3 = frontierCarbon;
    metrics.meanPrimordiaRadiusM = meanRadius;
    metrics.innerCoreDensityPerM2 = innerDensity;
    metrics.outerRingDensityPerM2 = outerDensity;

    details.push(
      `Soil C_s: center=${centerCarbon.toFixed(4)} < frontier=${frontierCarbon.toFixed(4)} kg/m^3; nodes=${mycelium.nodes.length}, anastomoses=${mycelium.getAnastomosisCount()}`,
    );
    details.push(
      `Primordia n=${primCount}, meanR=${(meanRadius * 100).toFixed(2)} cm (limit > 3.5 cm), density inner=${innerDensity.toFixed(1)} vs outer=${outerDensity.toFixed(1)} /m^2`,
    );

    const passed =
      centerCarbon < frontierCarbon &&
      primCount >= 3 &&
      meanRadius > 0.035 &&
      outerDensity > innerDensity;

    return {
      id: 'T2.4',
      name: 'Emergent Fairy Ring via Central Carbon Depletion',
      tier: 2,
      passed,
      metrics,
      details,
      error: passed
        ? undefined
        : `Fairy ring failed: centerC<frontierC=${centerCarbon < frontierCarbon}, n=${primCount}, meanR=${meanRadius.toFixed(4)} m, outer>inner=${outerDensity > innerDensity}`,
    };
  } catch (err) {
    return {
      id: 'T2.4',
      name: 'Emergent Fairy Ring via Central Carbon Depletion',
      tier: 2,
      passed: false,
      metrics,
      details,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ============================================================================
// T2.5 — Dressaire Evaporative Convective Lift Test (ON vs OFF, Zero Wind)
// ============================================================================
export function runT2_5_DressaireEvaporativeLift(): OracleTestResult {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};

  try {
    const bounds: TerrariumBounds = {
      minX: -0.24,
      maxX: 0.24,
      minY: 0.0,
      maxY: 0.36,
      minZ: -0.24,
      maxZ: 0.24,
    };

    const species = SPECIES_PRESETS.parasol;
    const capRadiusM = 0.05;
    const capAreaM2 = Math.PI * capRadiusM * capRadiusM * 1.4;
    const rh = 0.65;
    const evapRateKgS = species.transpirationCoeff * capAreaM2 * (1 - rh);

    const makeMatureBody = (evap: number): FruitingBodyState => ({
      id: 0,
      primordiumId: 0,
      myceliumNodeId: 0,
      speciesId: 'parasol',
      baseX: 0,
      baseY: 0,
      baseZ: 0,
      stipeLengthM: 0.09,
      stipeRadiusM: 0.007,
      stipeTipX: 0,
      stipeTipY: 0.09,
      stipeTipZ: 0,
      stipeTipDirX: 0,
      stipeTipDirY: 1,
      stipeTipDirZ: 0,
      turgorPa: 5.0e5,
      wiltFactor: 0,
      maturity: 1.0,
      capRadiusM,
      capAreaM2,
      gillWavenumber: 24,
      lamellulaeCount: 24,
      gillSpacingM: 0.00035,
      veilIntactFraction: 0.1,
      evaporationRateKgS: evap,
      stipeCenterline: new Float32Array([0, 0, 0, 0, 0.045, 0, 0, 0.09, 0]),
    });

    // Case A: Evaporation DISABLED -> strictly zero lift above cap in calm air
    const envOff = makeBaseEnv({ ambientHumidity: rh, evaporationEnabled: false });
    const sysOff = new SporePhysicsSystem(bounds);
    const bodiesOff = [makeMatureBody(0)];
    for (let s = 0; s < 140; s++) {
      sysOff.step(0.25, bodiesOff, envOff, false);
    }
    const maxLiftOffM = sysOff.getMaxSporeLiftAboveCapM();
    const airOff = sysOff.getAirField();
    let maxAbsUyOff = 0;
    for (let i = 0; i < airOff.uy.length; i++) {
      maxAbsUyOff = Math.max(maxAbsUyOff, Math.abs(airOff.uy[i]));
    }

    // Case B: Evaporation ENABLED -> genuine buoyant convective lift above cap
    const envOn = makeBaseEnv({ ambientHumidity: rh, evaporationEnabled: true });
    const sysOn = new SporePhysicsSystem(bounds);
    const bodiesOn = [makeMatureBody(evapRateKgS)];
    for (let s = 0; s < 220; s++) {
      sysOn.step(0.25, bodiesOn, envOn, true);
    }
    const maxLiftOnM = sysOn.getMaxSporeLiftAboveCapM();
    const airOn = sysOn.getAirField();
    let minDeltaT = 0;
    let maxUpdraftMs = 0;
    for (let i = 0; i < airOn.deltaT.length; i++) {
      minDeltaT = Math.min(minDeltaT, airOn.deltaT[i]);
      maxUpdraftMs = Math.max(maxUpdraftMs, airOn.uy[i]);
    }

    metrics.evapOffMaxLiftM = maxLiftOffM;
    metrics.evapOffMaxAbsUyMs = maxAbsUyOff;
    metrics.evapOnMaxLiftM = maxLiftOnM;
    metrics.evapOnMinDeltaTK = minDeltaT;
    metrics.evapOnMaxUpdraftMs = maxUpdraftMs;
    metrics.evapOnDepositedSpores = sysOn.getSporePrint().totalDeposited;

    details.push(
      `Evap OFF: maxLift=${maxLiftOffM.toFixed(6)} m (strict === 0), max|uy|=${maxAbsUyOff.toExponential(2)} m/s`,
    );
    details.push(
      `Evap ON: maxLift=${(maxLiftOnM * 1000).toFixed(2)} mm (limit > 5 mm), minDeltaT=${minDeltaT.toFixed(3)} K, maxUpdraft=${(maxUpdraftMs * 1000).toFixed(2)} mm/s`,
    );

    const passed = maxLiftOffM === 0 && maxLiftOnM > 0.005 && minDeltaT < -1e-4;
    return {
      id: 'T2.5',
      name: 'Dressaire Evaporative Convective Spore Lift (ON vs OFF)',
      tier: 2,
      passed,
      metrics,
      details,
      error: passed
        ? undefined
        : `Dressaire lift failed: liftOff=${maxLiftOffM} (need 0), liftOn=${maxLiftOnM} m (need >0.005), minDeltaT=${minDeltaT} K`,
    };
  } catch (err) {
    return {
      id: 'T2.5',
      name: 'Dressaire Evaporative Convective Spore Lift (ON vs OFF)',
      tier: 2,
      passed: false,
      metrics,
      details,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export function runTier2Coupling(): OracleTestResult[] {
  return [
    runT2_1_DroughtHypoxiaWilt(),
    runT2_2_Gravitropism(),
    runT2_3_Co2Etiolation(),
    runT2_4_FairyRing(),
    runT2_5_DressaireEvaporativeLift(),
  ];
}
