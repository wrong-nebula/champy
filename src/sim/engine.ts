// ============================================================================
// SPOROCARP — Master simulation engine (Coordinator-owned integration layer)
// Orchestrates Soil -> Mycelium -> Morphogenesis -> Spores/Air on a
// deterministic, clamped, fixed-timestep multi-rate schedule.
// No physics lives here: only scheduling, coupling order, and snapshots.
// ============================================================================
import {
  SPECIES_PRESETS,
  type EnvironmentState,
  type SimulationSnapshot,
  type SolidObstacle,
  type SpeciesId,
  type SpeciesPreset,
  type TelemetrySample,
  type TerrariumBounds,
  type TerrariumScenePreset,
} from '../types/contract.ts';
import { SoilGrid } from './soil/SoilGrid.ts';
import { MyceliumNetwork } from './mycelium/MyceliumNetwork.ts';
import { MorphogenesisSystem } from './morpho/MorphogenesisSystem.ts';
import { SporePhysicsSystem } from './spores/SporePhysicsSystem.ts';
import { BioluminescenceSystem } from './biolum/BioluminescenceSystem.ts';

/** Maximum macro (soil / mycelium / morphogenesis) step — conservative vs. tested 300 s. */
const MACRO_DT_MAX_S = 600;
/** Macro layers are only advanced once this much simulated time has accrued. */
const MACRO_DT_MIN_S = 1.0;
/** Micro (air convection / spore) steps are real-time physics, never time-warped. */
const MICRO_DT_MAX_S = 1 / 30;
/** Wall-clock budget (ms) per frame for macro stepping. */
const MACRO_BUDGET_MS = 14;
const HISTORY_MAX = 600;
const HISTORY_PERIOD_S = 600; // one telemetry history sample per 10 simulated minutes

export function gravityFromTilt(tiltRad: number): [number, number, number] {
  return [Math.sin(tiltRad), -Math.cos(tiltRad), 0];
}

export function defaultEnvironment(): EnvironmentState {
  const l: [number, number, number] = [0.35, 0.88, 0.32];
  const n = Math.hypot(l[0], l[1], l[2]);
  return {
    ambientTempC: 15,
    ambientHumidity: 0.88,
    ambientCo2Ppm: 420,
    lightDir: [l[0] / n, l[1] / n, l[2] / n],
    terrainTiltRad: 0,
    gravityDir: [0, -1, 0],
    rainRateMs: 0,
    evaporationEnabled: true,
    timeWarp: 10000,
  };
}

export class SimulationEngine {
  soil: SoilGrid;
  mycelium: MyceliumNetwork;
  morpho: MorphogenesisSystem;
  spores: SporePhysicsSystem;
  public foxfireEnabled = false;
  public readonly biolum = new BioluminescenceSystem();
  species: SpeciesPreset;
  scene: TerrariumScenePreset;
  env: EnvironmentState = defaultEnvironment();
  obstacles: SolidObstacle[] = [];
  simTimeSec = 0;
  paused = false;
  selectedBodyId = -1;
  fps = 60;

  private macroAccumS = 0;
  private historyAccumS = 0;
  private history: TelemetrySample[] = [];
  private nextObstacleId = 1;
  private readonly bounds: TerrariumBounds;

  constructor(
    speciesId: SpeciesId = 'parasol',
    scene: TerrariumScenePreset = 'forest-cross-section',
  ) {
    this.species = SPECIES_PRESETS[speciesId];
    this.scene = scene;
    // 32x22x32 @ 1.5 cm preserves the exact 0.48 x 0.33 x 0.48 m terrarium volume
    // while running 4.6x faster than 48x32x48 to sustain 60 FPS live on Apple Silicon.
    this.soil = new SoilGrid({ nx: 32, ny: 22, nz: 32, dx: 0.015 });
    this.bounds = {
      minX: -this.soil.sizeX / 2,
      maxX: this.soil.sizeX / 2,
      minY: 0,
      maxY: 0.36,
      minZ: -this.soil.sizeZ / 2,
      maxZ: this.soil.sizeZ / 2,
    };
    this.soil.setupPresetTerrarium(scene);
    this.mycelium = new MyceliumNetwork(this.soil, this.species);
    this.morpho = new MorphogenesisSystem(this.species);
    this.spores = new SporePhysicsSystem(this.bounds);
    this.syncSoilAmbient();
    this.inoculate();
    this.refreshBiolum();
  }

  private refreshBiolum(): void {
    this.biolum.step(
      0,
      this.simTimeSec,
      this.env,
      this.species,
      this.soil,
      this.mycelium,
      this.morpho.getBodies(),
      this.foxfireEnabled,
    );
  }

  // --- configuration ------------------------------------------------------
  private inoculate(): void {
    switch (this.scene) {
      case 'forest-cross-section':
        this.mycelium.inoculate(-0.02, -0.05, 0, 8e-5);
        this.mycelium.inoculate(0.06, -0.03, 0.05, 5e-5);
        break;
      case 'rotting-log':
        this.mycelium.inoculate(0, -0.04, 0, 8e-5);
        break;
      default:
        this.mycelium.inoculate(0, -0.02, 0, 8e-5);
    }
  }

  private syncSoilAmbient(): void {
    // Ambient boundary conditions read by the soil each step via env; nothing else to push.
  }

  setScene(scene: TerrariumScenePreset): void {
    this.scene = scene;
    this.soil.setupPresetTerrarium(scene);
    this.resetBiology();
    this.obstacles = [];
  }

  setSpecies(id: SpeciesId): void {
    this.species = SPECIES_PRESETS[id];
    this.resetBiology();
  }

  /** Re-seed mycelium + fruiting bodies + spores (soil state is preserved). */
  resetBiology(): void {
    this.mycelium = new MyceliumNetwork(this.soil, this.species);
    this.morpho.reset(this.species);
    this.spores.reset();
    this.biolum.reset();
    this.selectedBodyId = -1;
    this.macroAccumS = 0;
    this.historyAccumS = 0;
    this.history = [];
    this.simTimeSec = 0;
    this.inoculate();
    if ((this.species.biolumGain ?? 0) > 0 || this.foxfireEnabled) {
      this.refreshBiolum();
    }
  }

  reset(): void {
    this.soil.setupPresetTerrarium(this.scene);
    this.obstacles = [];
    this.resetBiology();
  }

  setFoxfireEnabled(enabled: boolean): void {
    this.foxfireEnabled = enabled;
    this.refreshBiolum();
  }

  setEnv(patch: Partial<EnvironmentState>): void {
    Object.assign(this.env, patch);
    if (patch.terrainTiltRad !== undefined && patch.gravityDir === undefined) {
      this.env.gravityDir = gravityFromTilt(this.env.terrainTiltRad);
    }
    this.env.timeWarp = Math.max(1, Math.min(100000, this.env.timeWarp));
    if (patch.ambientTempC !== undefined) {
      this.refreshBiolum();
    }
  }

  drainSoil(): void {
    this.soil.setUniformHydrostaticHead(-6.0);
  }

  floodSoil(): void {
    this.soil.setUniformHydrostaticHead(0.2);
  }

  addTwig(x: number, z: number): void {
    const y = 0.0;
    this.obstacles.push({
      id: this.nextObstacleId++,
      kind: 'twig',
      x: x - 0.02,
      y: y + 0.012,
      z,
      radius: 0.004,
      endX: x + 0.02,
      endY: y + 0.012,
      endZ: z,
    });
  }

  // --- stepping -----------------------------------------------------------
  /** Advance by `dtRealSec` of wall time (scaled by time warp). */
  advanceFrame(dtRealSec: number): void {
    const dtReal = Math.min(Math.max(dtRealSec, 0), 0.1);
    if (this.paused) return;
    this.advanceSim(dtReal * this.env.timeWarp, dtReal);
  }

  /**
   * Advance simulated time by `dtSimSec`; micro (spore/air) layers advance by
   * `dtMicroSec` of *real* physics time (default: same as dtSim, clamped).
   */
  advanceSim(dtSimSec: number, dtMicroSec?: number, budgetMs = MACRO_BUDGET_MS): void {
    this.macroAccumS += dtSimSec;
    const t0 = performance.now();
    while (this.macroAccumS >= MACRO_DT_MIN_S) {
      const dt = Math.min(this.macroAccumS, MACRO_DT_MAX_S);
      this.macroStep(dt);
      this.macroAccumS -= dt;
      if (performance.now() - t0 > budgetMs) {
        // Wall-clock budget exhausted: drop the remaining debt (time-warp saturates)
        this.macroAccumS = 0;
        break;
      }
    }
    let micro = Math.min(dtMicroSec ?? dtSimSec, 0.1);
    while (micro > 1e-9) {
      const dt = Math.min(micro, MICRO_DT_MAX_S);
      this.spores.step(dt, this.morpho.getBodies(), this.env, this.env.evaporationEnabled);
      micro -= dt;
    }
  }

  /** Deterministic stepping used by tests and the debug handle (no wall-clock budget). */
  stepFixed(dtSec: number, steps = 1): void {
    for (let i = 0; i < steps; i++) {
      this.macroStep(Math.min(dtSec, MACRO_DT_MAX_S));
      this.spores.step(Math.min(dtSec, MICRO_DT_MAX_S), this.morpho.getBodies(), this.env, this.env.evaporationEnabled);
    }
  }

  fastForward(simSeconds: number): void {
    // Multi-rate operator splitting: soil Richards/gas diffusion is unconditionally
    // stable at 1800 s steps, while mycelium + morphogenesis sub-step at 300 s
    // (still calling live soil water/carbon sinks on every 300 s sub-step).
    let remaining = simSeconds;
    const bioDtMax = 300;
    const soilStride = 6; // 6 * 300 s = 1800 s per soil diffusion pass
    let soilAccum = 0;
    let stepIdx = 0;
    while (remaining > 1e-9) {
      const dt = Math.min(remaining, bioDtMax);
      soilAccum += dt;
      stepIdx++;
      const runSoil = stepIdx % soilStride === 0 || remaining <= bioDtMax + 1e-9;
      if (runSoil) {
        if (this.env.rainRateMs > 0) {
          this.soil.pourRain(0, 0, this.soil.sizeX, this.env.rainRateMs * soilAccum);
        }
        this.soil.step(soilAccum, this.env);
        soilAccum = 0;
      }
      this.mycelium.step(dt, this.env, this.species);
      this.simTimeSec += dt;
      this.morpho.syncWithPrimordia(this.mycelium, this.simTimeSec);
      this.morpho.step(dt, this.mycelium, this.env, this.obstacles);
      this.historyAccumS += dt;
      if (this.historyAccumS >= HISTORY_PERIOD_S) {
        this.historyAccumS = 0;
        this.history.push(this.buildTelemetry());
        if (this.history.length > HISTORY_MAX) this.history.shift();
      }
      remaining -= dt;
    }
    // Let the Dressaire air convection + spore plume develop over a 4-second real-time window
    for (let i = 0; i < 90; i++) {
      this.spores.step(1 / 24, this.morpho.getBodies(), this.env, this.env.evaporationEnabled);
    }
    this.refreshBiolum();
  }

  private macroStep(dt: number): void {
    // Rain is a surface source: convert continuous rate to a pour over the terrarium.
    if (this.env.rainRateMs > 0) {
      this.soil.pourRain(0, 0, this.soil.sizeX, this.env.rainRateMs * dt);
    }
    this.soil.step(dt, this.env);
    this.mycelium.step(dt, this.env, this.species);
    this.simTimeSec += dt;
    this.morpho.syncWithPrimordia(this.mycelium, this.simTimeSec);
    this.morpho.step(dt, this.mycelium, this.env, this.obstacles);
    this.biolum.step(
      dt,
      this.simTimeSec,
      this.env,
      this.species,
      this.soil,
      this.mycelium,
      this.morpho.getBodies(),
      this.foxfireEnabled,
    );

    this.historyAccumS += dt;
    if (this.historyAccumS >= HISTORY_PERIOD_S) {
      this.historyAccumS = 0;
      this.history.push(this.buildTelemetry());
      if (this.history.length > HISTORY_MAX) this.history.shift();
    }
  }

  pourRain(x: number, z: number, radius: number, depth: number): void {
    this.soil.pourRain(x, z, radius, depth);
  }

  // --- snapshot -----------------------------------------------------------
  private buildTelemetry(): TelemetrySample {
    const s = this.soil;
    let sumTheta = 0;
    let sumPsi = 0;
    let sumO2 = 0;
    let n = 0;
    // Sample the root zone (top 12 voxels) excluding stone
    const iyMin = Math.max(0, s.ny - 12);
    for (let iy = iyMin; iy < s.ny; iy++) {
      for (let iz = 0; iz < s.nz; iz += 2) {
        for (let ix = 0; ix < s.nx; ix += 2) {
          const i = s.voxelIndex(ix, iy, iz);
          if (s.material[i] === 5) continue;
          sumTheta += s.theta[i];
          sumPsi += s.psi[i];
          sumO2 += s.o2[i];
          n++;
        }
      }
    }
    const inv = n > 0 ? 1 / n : 0;
    const bodies = this.morpho.getBodies();
    let lead = bodies[0];
    for (const b of bodies) if (b.stipeLengthM > (lead?.stipeLengthM ?? -1)) lead = b;
    if (this.selectedBodyId >= 0) {
      const sel = bodies.find((b) => b.id === this.selectedBodyId);
      if (sel) lead = sel;
    }
    const print = this.spores.getSporePrint();
    return {
      simTimeSec: this.simTimeSec,
      meanSoilTheta: sumTheta * inv,
      meanSoilPsiPa: sumPsi * inv,
      meanSoilO2MolM3: sumO2 * inv,
      myceliumNodeCount: this.mycelium.nodes.length,
      anastomosisCount: this.mycelium.getAnastomosisCount(),
      fruitingCount: bodies.length,
      leadTurgorPa: lead ? lead.turgorPa : 0,
      leadYieldPa: this.species.yieldPressurePa,
      leadHeightM: lead ? lead.stipeLengthM : 0,
      leadCapRadiusM: lead ? lead.capRadiusM : 0,
      leadGillWavenumber: lead ? lead.gillWavenumber : 0,
      leadVeilIntact: lead ? lead.veilIntactFraction : 1,
      maxSporeLiftM: this.spores.getMaxSporeLiftAboveCapM(),
      meanSporeDispersalM: print.meanDispersalRadiusM,
      fps: this.fps,
    };
  }

  getSnapshot(): SimulationSnapshot {
    const bodies = this.morpho.getBodies();
    let microIdx = 0;
    if (this.selectedBodyId >= 0) {
      const i = bodies.findIndex((b) => b.id === this.selectedBodyId);
      if (i >= 0) microIdx = i;
    }
    const telemetry = this.buildTelemetry();
    return {
      simTimeSec: this.simTimeSec,
      species: this.species,
      env: this.env,
      soil: this.soil,
      mycelium: this.mycelium.getPackedRenderBuffers(),
      fruitingMesh: this.morpho.getCombinedRenderMesh(),
      bodies,
      obstacles: this.obstacles,
      airField: this.spores.getAirField(),
      spores: this.spores.getSpores(),
      sporePrint: this.spores.getSporePrint(),
      bullerMicro: this.spores.getActiveBullerState(microIdx),
      biolum: this.biolum.getState(),
      telemetry,
      telemetryHistory: [...this.history, telemetry],
    };
  }
}
