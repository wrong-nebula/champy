// ============================================================================
// SPOROCARP — Layer 5: 3-Hydroxyhispidin (LH2) Bioluminescence Biophysics
// - 525 nm fungal luciferin-luciferase enzymology (Mycena chlorophos & foxfire)
// - 24 h circadian oscillator (peaking around nocturnal t ~ 160_000..180_000 s)
// - Sharpe-DeMichele / Arrhenius thermal activation & >29 C denaturation
// - Coupled Michaelis-Menten O2 & soluble carbon luciferin pool ODE (tau ~ 1800 s)
// - Spatial cap emitters ('body') & >= 2.2 cm spaced oak/hyphal 'foxfire' emitters
// - Strictly read-only on SoilGrid, MyceliumNetwork, and MorphogenesisSystem
// ============================================================================

import {
  SoilMaterialId,
  type BiolumEmitter,
  type BioluminescenceState,
  type EnvironmentState,
  type FruitingBodyState,
  type IMyceliumNetwork,
  type ISoilGrid,
  type MyceliumRenderBuffers,
  type SpeciesPreset,
} from '../../types/contract.ts';

const MAX_EMITTERS = 32;
const MAX_BODY_GLOW = 32;
const FOXFIRE_MIN_SPACING_M = 0.022;
const FOXFIRE_MIN_SPACING_M2 = FOXFIRE_MIN_SPACING_M * FOXFIRE_MIN_SPACING_M;

const PERIOD_24H_SEC = 86400;
const CIRCADIAN_OFFSET_SEC = 27000;

const KM_O2_MOL_M3 = 1.6;
const KM_CARBON_KG_M3 = 0.25;

// Luciferin pool ODE rates (tau ~ 1800 s at warm aerobic night)
const K_SYN_LUCIFERIN = 6.9e-4; // 1 / s
const K_OX_LUCIFERIN = 5.0e-4; // 1 / s
const K_DEG_LUCIFERIN = 1.2e-4; // 1 / s

const PEAK_WAVELENGTH_NM = 525;
const PEAK_PHOTON_FLUX_CM2_S = 3.2e9;

const INIT_CAND_CAPACITY = 512;

/**
 * Arrhenius enzymatic rise (Q10 ~= 2 from 5 C) with steep high-temperature
 * luciferase denaturation above 29 C, peaking at 23 C and clamped to [0.06, 1.05].
 */
function computeTempFactor(ambientTempC: number): number {
  const tK = Math.max(240.0, Math.min(360.0, ambientTempC + 273.15));
  const tRefK = 296.15; // 23.0 C
  const eaOverR = 5750.0;
  const edOverR = 36000.0;
  const thK = 300.25; // df/dT = 0 at 23.0 C
  const arrh = Math.exp(eaOverR * (1.0 / tRefK - 1.0 / tK));
  const denat = 1.0 + Math.exp(edOverR * (1.0 / thK - 1.0 / tK));
  const val = (1.19 * arrh) / denat;
  return val < 0.06 ? 0.06 : val > 1.05 ? 1.05 : val;
}

function isMyceliumBuffers(
  m: IMyceliumNetwork | MyceliumRenderBuffers,
): m is MyceliumRenderBuffers {
  return 'segments' in m && !('edges' in m);
}

export class BioluminescenceSystem {
  private luciferinPool = 0.85;
  private readonly bodyGlow = new Float32Array(MAX_BODY_GLOW);
  private readonly emitterPool: BiolumEmitter[];
  private readonly emitters: BiolumEmitter[] = [];
  private readonly state: BioluminescenceState;

  // Reusable scratch buffers for zero-allocation foxfire candidate sorting
  private candCapacity = INIT_CAND_CAPACITY;
  private candX = new Float32Array(INIT_CAND_CAPACITY);
  private candY = new Float32Array(INIT_CAND_CAPACITY);
  private candZ = new Float32Array(INIT_CAND_CAPACITY);
  private candIntensity = new Float32Array(INIT_CAND_CAPACITY);
  private candRadius = new Float32Array(INIT_CAND_CAPACITY);
  private candOrder = new Int32Array(INIT_CAND_CAPACITY);

  constructor() {
    this.emitterPool = Array.from({ length: MAX_EMITTERS }, () => ({
      x: 0,
      y: 0,
      z: 0,
      intensity: 0,
      radiusM: 0,
      kind: 'foxfire' as const,
    }));
    this.state = {
      enabled: false,
      circadianPhase01: 0,
      circadianFactor: 0,
      tempFactor: 0,
      meanO2Factor: 0,
      luciferinPoolNorm: 0,
      photonFluxCm2S: 0,
      peakWavelengthNm: PEAK_WAVELENGTH_NM,
      bodyGlow: this.bodyGlow,
      emitters: this.emitters,
    };
  }

  reset(): void {
    this.luciferinPool = 0.85;
    this.bodyGlow.fill(0);
    this.emitters.length = 0;
    this.state.enabled = false;
    this.state.circadianPhase01 = 0;
    this.state.circadianFactor = 0;
    this.state.tempFactor = 0;
    this.state.meanO2Factor = 0;
    this.state.luciferinPoolNorm = 0;
    this.state.photonFluxCm2S = 0;
    this.state.peakWavelengthNm = PEAK_WAVELENGTH_NM;
  }

  getState(): BioluminescenceState {
    return this.state;
  }

  step(
    dtSec: number,
    simTimeSec: number,
    env: EnvironmentState,
    species: SpeciesPreset,
    soil: ISoilGrid,
    mycelium: IMyceliumNetwork | MyceliumRenderBuffers,
    bodies: readonly FruitingBodyState[],
    foxfireEnabled: boolean,
  ): void {
    const speciesGain = species.biolumGain ?? 0;
    const active = speciesGain > 0 || foxfireEnabled;

    if (!active) {
      if (this.state.enabled) {
        this.state.enabled = false;
        this.state.photonFluxCm2S = 0;
        this.state.meanO2Factor = 0;
        this.state.luciferinPoolNorm = 0;
        this.bodyGlow.fill(0);
        this.emitters.length = 0;
      }
      return;
    }

    // 1. Circadian oscillator (24 h period, nocturnal peak around t ~ 160_000..180_000 s)
    const safeTime = Number.isFinite(simTimeSec) ? simTimeSec : 0.0;
    const shifted = safeTime - CIRCADIAN_OFFSET_SEC;
    const phaseAngle = (2.0 * Math.PI * shifted) / PERIOD_24H_SEC;
    const circadianPhase01 =
      (((shifted % PERIOD_24H_SEC) + PERIOD_24H_SEC) % PERIOD_24H_SEC) / PERIOD_24H_SEC;
    const cosWave = Math.max(0.0, 0.5 * (1.0 - Math.cos(phaseAngle)));
    const circadianFactor = 0.22 + 0.78 * Math.pow(cosWave, 1.4);

    // 2. Arrhenius + denaturation thermal factor
    const safeTempC = Number.isFinite(env.ambientTempC) ? env.ambientTempC : 15.0;
    const tempFactor = computeTempFactor(safeTempC);

    // 3. Read-only sampling of top aerobic soil layers + WOOD voxels
    const { nx, ny, nz, dx } = soil;
    const topLayerCount = Math.max(3, Math.round(0.11 / dx));
    const iyMax = ny > 2 ? ny - 2 : ny - 1;
    const iyMin = Math.max(0, iyMax - topLayerCount + 1);
    const strideY = nx * nz;
    const matArr = soil.material;
    const o2Arr = soil.o2;
    const csArr = soil.solubleCarbon;

    let sumO2 = 0.0;
    let sumCs = 0.0;
    let sampleCount = 0;

    for (let iy = 0; iy < ny; iy++) {
      const inTopZone = iy >= iyMin && iy <= iyMax;
      const yOff = iy * strideY;
      for (let iz = 0; iz < nz; iz++) {
        const rowOff = yOff + iz * nx;
        for (let ix = 0; ix < nx; ix++) {
          const idx = rowOff + ix;
          const m = matArr[idx];
          if (m === SoilMaterialId.STONE) continue;
          if (inTopZone || m === SoilMaterialId.WOOD) {
            sumO2 += o2Arr[idx];
            sumCs += csArr[idx];
            sampleCount++;
          }
        }
      }
    }

    const meanO2 = sampleCount > 0 ? sumO2 / sampleCount : 0.0;
    const meanCs = sampleCount > 0 ? sumCs / sampleCount : 0.0;
    const meanO2Factor = meanO2 <= 0.0 ? 0.0 : meanO2 / (KM_O2_MOL_M3 + meanO2);
    const carbonFactor = meanCs <= 0.0 ? 0.0 : meanCs / (KM_CARBON_KG_M3 + meanCs);

    // 4. Coupled luciferin pool ODE (sub-stepped safely if dtSec > 300 s)
    if (dtSec > 0.0 && Number.isFinite(dtSec)) {
      const prod = K_SYN_LUCIFERIN * carbonFactor * tempFactor * circadianFactor;
      const loss = K_OX_LUCIFERIN * meanO2Factor * tempFactor + K_DEG_LUCIFERIN;
      const subSteps = dtSec > 300.0 ? Math.min(240, Math.ceil(dtSec / 300.0)) : 1;
      const subDt = dtSec / subSteps;
      for (let s = 0; s < subSteps; s++) {
        if (loss > 1e-12) {
          const lInf = prod / loss;
          this.luciferinPool = lInf + (this.luciferinPool - lInf) * Math.exp(-loss * subDt);
        } else {
          this.luciferinPool += prod * subDt;
        }
        if (this.luciferinPool < 0.0) this.luciferinPool = 0.0;
        else if (this.luciferinPool > 1.5) this.luciferinPool = 1.5;
      }
    }

    // 5. Per-body bioluminescence & 'body' emitters
    const effectiveGain = speciesGain;
    const nBodies = Math.min(bodies.length, MAX_BODY_GLOW);
    let outCount = 0;
    let peakGlow = 0.0;

    for (let i = 0; i < nBodies; i++) {
      const body = bodies[i];
      if (effectiveGain <= 0.0 || meanO2 <= 0.0) {
        this.bodyGlow[i] = 0.0;
        continue;
      }

      const [ix, iy, iz] = soil.worldToVoxel(body.baseX, -0.02, body.baseZ);
      const bIdx = soil.voxelIndex(ix, iy, iz);
      const isStone = matArr[bIdx] === SoilMaterialId.STONE;
      const localO2 = isStone ? 0.0 : o2Arr[bIdx];
      if (localO2 <= 0.0) {
        this.bodyGlow[i] = 0.0;
        continue;
      }

      const localCs = isStone ? 0.0 : csArr[bIdx];
      const o2F = localO2 / (KM_O2_MOL_M3 + localO2);
      const cF =
        0.4 + 0.6 * (localCs <= 0.0 ? 0.0 : localCs / (KM_CARBON_KG_M3 + localCs));

      const rawTurgor = body.turgorPa / 2.2e5;
      const turgorTerm = rawTurgor < 0.0 ? 0.0 : rawTurgor > 1.2 ? 1.2 : rawTurgor;
      const wiltTerm = Math.max(0.0, 1.0 - 0.85 * body.wiltFactor);
      const maturityTerm = 0.35 + 0.65 * Math.min(1.0, body.maturity + 0.25);

      const glow =
        turgorTerm *
        wiltTerm *
        maturityTerm *
        o2F *
        cF *
        tempFactor *
        circadianFactor *
        this.luciferinPool *
        effectiveGain;

      this.bodyGlow[i] = glow;
      if (glow > peakGlow) peakGlow = glow;

      if (glow > 1e-5 && outCount < MAX_EMITTERS) {
        const em = this.emitterPool[outCount];
        em.x = body.stipeTipX;
        em.y = body.stipeTipY;
        em.z = body.stipeTipZ;
        em.intensity = glow;
        em.radiusM = Math.max(0.03, body.capRadiusM * 2.2);
        em.kind = 'body';
        this.emitters[outCount] = em;
        outCount++;
      }
    }
    for (let i = nBodies; i < MAX_BODY_GLOW; i++) {
      this.bodyGlow[i] = 0.0;
    }

    // 6. Foxfire hyphal emitters (2.2x boost in SoilMaterialId.WOOD, >= 2.2 cm spacing)
    const foxfireScale = foxfireEnabled ? 1.0 : 0.35 * effectiveGain;
    if (outCount < MAX_EMITTERS && foxfireScale > 0.0 && meanO2 > 0.0) {
      const candCount = this.collectFoxfireCandidates(
        soil,
        mycelium,
        foxfireScale * tempFactor * circadianFactor * this.luciferinPool,
      );

      if (candCount > 0) {
        for (let i = 0; i < candCount; i++) {
          this.candOrder[i] = i;
        }
        this.sortCandidatesDesc(0, candCount - 1);

        const foxfireStart = outCount;
        for (let k = 0; k < candCount && outCount < MAX_EMITTERS; k++) {
          const ci = this.candOrder[k];
          const cx = this.candX[ci];
          const cy = this.candY[ci];
          const cz = this.candZ[ci];

          let tooClose = false;
          for (let j = foxfireStart; j < outCount; j++) {
            const existing = this.emitters[j];
            const dxE = cx - existing.x;
            const dyE = cy - existing.y;
            const dzE = cz - existing.z;
            if (dxE * dxE + dyE * dyE + dzE * dzE < FOXFIRE_MIN_SPACING_M2) {
              tooClose = true;
              break;
            }
          }
          if (tooClose) continue;

          const intensity = this.candIntensity[ci];
          if (intensity > peakGlow) peakGlow = intensity;

          const em = this.emitterPool[outCount];
          em.x = cx;
          em.y = cy;
          em.z = cz;
          em.intensity = intensity;
          em.radiusM = this.candRadius[ci];
          em.kind = 'foxfire';
          this.emitters[outCount] = em;
          outCount++;
        }
      }
    }

    this.emitters.length = outCount;
    this.state.enabled = true;
    this.state.circadianPhase01 = circadianPhase01;
    this.state.circadianFactor = circadianFactor;
    this.state.tempFactor = tempFactor;
    this.state.meanO2Factor = meanO2Factor;
    this.state.luciferinPoolNorm = this.luciferinPool;
    this.state.peakWavelengthNm = PEAK_WAVELENGTH_NM;
    this.state.photonFluxCm2S =
      meanO2Factor <= 0.0 || peakGlow <= 0.0 ? 0.0 : peakGlow * PEAK_PHOTON_FLUX_CM2_S;
  }

  private ensureCandCapacity(needed: number): void {
    if (needed <= this.candCapacity) return;
    let nextCap = this.candCapacity;
    while (nextCap < needed) nextCap *= 2;
    this.candCapacity = nextCap;
    this.candX = new Float32Array(nextCap);
    this.candY = new Float32Array(nextCap);
    this.candZ = new Float32Array(nextCap);
    this.candIntensity = new Float32Array(nextCap);
    this.candRadius = new Float32Array(nextCap);
    this.candOrder = new Int32Array(nextCap);
  }

  private collectFoxfireCandidates(
    soil: ISoilGrid,
    mycelium: IMyceliumNetwork | MyceliumRenderBuffers,
    globalScale: number,
  ): number {
    const matArr = soil.material;
    const o2Arr = soil.o2;
    let candCount = 0;

    if (isMyceliumBuffers(mycelium)) {
      const segCount = mycelium.segmentCount;
      const segs = mycelium.segments;
      this.ensureCandCapacity(segCount);

      for (let i = 0; i < segCount; i++) {
        const off = i * 8;
        const turgorNorm = segs[off + 7];
        if (!Number.isFinite(turgorNorm) || turgorNorm <= 0.01) continue;

        const mx = 0.5 * (segs[off] + segs[off + 3]);
        const my = 0.5 * (segs[off + 1] + segs[off + 4]);
        const mz = 0.5 * (segs[off + 2] + segs[off + 5]);

        const [ix, iy, iz] = soil.worldToVoxel(mx, my, mz);
        const vIdx = soil.voxelIndex(ix, iy, iz);
        const matId = matArr[vIdx];
        if (matId === SoilMaterialId.STONE) continue;

        const localO2 = o2Arr[vIdx];
        if (!Number.isFinite(localO2) || localO2 <= 0.0) continue;

        const o2F = localO2 / (KM_O2_MOL_M3 + localO2);
        const isWood = matId === SoilMaterialId.WOOD;
        const intensity = (isWood ? 2.2 : 1.0) * o2F * turgorNorm * globalScale;
        if (!Number.isFinite(intensity) || intensity <= 1e-4) continue;

        this.candX[candCount] = mx;
        this.candY[candCount] = my;
        this.candZ[candCount] = mz;
        this.candIntensity[candCount] = intensity;
        this.candRadius[candCount] = isWood ? 0.045 : 0.032;
        candCount++;
      }
      return candCount;
    }

    const edges = mycelium.edges;
    const nodes = mycelium.nodes;
    const segCount = edges.length;
    this.ensureCandCapacity(segCount);

    for (let i = 0; i < segCount; i++) {
      const e = edges[i];
      const na = nodes[e.a];
      const nb = nodes[e.b];
      const rawTurgor = (0.5 * (na.turgorPa + nb.turgorPa)) / 6.5e5;
      if (!Number.isFinite(rawTurgor)) continue;
      const turgorNorm = rawTurgor < 0.0 ? 0.0 : rawTurgor > 1.0 ? 1.0 : rawTurgor;
      if (turgorNorm <= 0.01) continue;

      const mx = 0.5 * (na.x + nb.x);
      const my = 0.5 * (na.y + nb.y);
      const mz = 0.5 * (na.z + nb.z);

      const [ix, iy, iz] = soil.worldToVoxel(mx, my, mz);
      const vIdx = soil.voxelIndex(ix, iy, iz);
      const matId = matArr[vIdx];
      if (matId === SoilMaterialId.STONE) continue;

      const localO2 = o2Arr[vIdx];
      if (!Number.isFinite(localO2) || localO2 <= 0.0) continue;

      const o2F = localO2 / (KM_O2_MOL_M3 + localO2);
      const isWood =
        matId === SoilMaterialId.WOOD ||
        matArr[na.voxelIdx] === SoilMaterialId.WOOD ||
        matArr[nb.voxelIdx] === SoilMaterialId.WOOD;

      const intensity = (isWood ? 2.2 : 1.0) * o2F * turgorNorm * globalScale;
      if (!Number.isFinite(intensity) || intensity <= 1e-4) continue;

      this.candX[candCount] = mx;
      this.candY[candCount] = my;
      this.candZ[candCount] = mz;
      this.candIntensity[candCount] = intensity;
      this.candRadius[candCount] = isWood ? 0.045 : 0.032;
      candCount++;
    }

    return candCount;
  }

  private sortCandidatesDesc(lo: number, hi: number): void {
    if (lo >= hi) return;
    const order = this.candOrder;
    const inten = this.candIntensity;

    // Insertion sort for small sub-arrays
    if (hi - lo < 16) {
      for (let i = lo + 1; i <= hi; i++) {
        const key = order[i];
        const keyVal = inten[key];
        let j = i - 1;
        while (j >= lo && inten[order[j]] < keyVal) {
          order[j + 1] = order[j];
          j--;
        }
        order[j + 1] = key;
      }
      return;
    }

    const mid = (lo + hi) >>> 1;
    const pivotVal = inten[order[mid]];
    let i = lo;
    let j = hi;

    while (i <= j) {
      while (inten[order[i]] > pivotVal) i++;
      while (inten[order[j]] < pivotVal) j--;
      if (i <= j) {
        const tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
        i++;
        j--;
      }
    }

    if (lo < j) this.sortCandidatesDesc(lo, j);
    if (i < hi) this.sortCandidatesDesc(i, hi);
  }
}
