// ============================================================================
// SPOROCARP — Layer 3: Turgor Hydraulic Inflation, Tropisms & CO2 Morphogenesis
// Exact Lockhart stipe elongation, coupled mycelial hydraulic draw & wilting,
// 16-segment gravitropism/phototropism, CO2 etiolation, and world-space mesh
// packing for caps, gills, stipes, and fractured partial veils.
// ============================================================================

import {
  MeshPartId,
  RHO_WATER,
  type EnvironmentState,
  type FruitingBodyState,
  type FruitingRenderMesh,
  type IMyceliumNetwork,
  type PrimordiumAnchor,
  type SolidObstacle,
  type SpeciesPreset,
} from '../../types/contract.ts';
import { ThinShellCapSolver } from './ThinShellSolver.ts';

const NUM_STIPE_SEGMENTS = 16;
const STIPE_RADIAL_SIDES = 14;
const MAX_ACTIVE_BODIES = 16;

/**
 * Exact analytical Lockhart stipe elongation step.
 * For P > Y: L(t + dt) = L(t) * exp(phi * (P - Y) * dt).
 * For P <= Y: strictly returns lengthM (zero growth).
 */
export function stepStipeLockhartExact(
  lengthM: number,
  turgorPa: number,
  phi: number,
  yieldPa: number,
  dtSec: number,
): number {
  if (turgorPa <= yieldPa || dtSec <= 0 || phi <= 0 || lengthM <= 0) {
    return lengthM;
  }
  return lengthM * Math.exp(phi * (turgorPa - yieldPa) * dtSec);
}

/** Computes CO2 etiolation scaling factors from THEORY.md §4.3 */
export function computeCo2EtiolationFactors(
  co2Ppm: number,
  species: SpeciesPreset,
): { etaCap: number; etaStipe: number } {
  const c = Math.max(0, co2Ppm);
  const km = Math.max(1, species.co2KmPpm);
  const ratio = c / km;
  const etaCap = 1.0 / (1.0 + ratio * ratio);
  const etaStipe = 1.0 + species.co2StipeBoost * (c / (km + c));
  return { etaCap, etaStipe };
}

function normalize3(x: number, y: number, z: number): [number, number, number] {
  const len = Math.hypot(x, y, z);
  if (len < 1e-12) return [0, 1, 0];
  return [x / len, y / len, z / len];
}

/** Build a right-handed orthonormal frame (right, up, forward) given unit up vector */
function buildFrameFromUp(
  ux: number,
  uy: number,
  uz: number,
): {
  rx: number;
  ry: number;
  rz: number;
  fx: number;
  fy: number;
  fz: number;
} {
  // Choose a reference axis not parallel to up
  const refX = Math.abs(ux) < 0.85 ? 1 : 0;
  const refY = Math.abs(ux) < 0.85 ? 0 : 0;
  const refZ = Math.abs(ux) < 0.85 ? 0 : 1;

  // forward = normalize(ref x up)
  let fx = refY * uz - refZ * uy;
  let fy = refZ * ux - refX * uz;
  let fz = refX * uy - refY * ux;
  const fl = Math.hypot(fx, fy, fz);
  if (fl > 1e-12) {
    fx /= fl;
    fy /= fl;
    fz /= fl;
  } else {
    fx = 0;
    fy = 0;
    fz = 1;
  }

  // right = up x forward
  const rx = uy * fz - uz * fy;
  const ry = uz * fx - ux * fz;
  const rz = ux * fy - uy * fx;
  return { rx, ry, rz, fx, fy, fz };
}

export class FruitingBody {
  public readonly id: number;
  public readonly primordiumId: number;
  public readonly myceliumNodeId: number;
  public readonly species: SpeciesPreset;
  public readonly baseX: number;
  public readonly baseY: number;
  public readonly baseZ: number;
  public readonly shell: ThinShellCapSolver;

  public stipeLengthM: number;
  public stipeRadiusM: number;
  public turgorPa: number;
  public wiltFactor: number;
  public maturity: number;
  public evaporationRateKgS: number;

  // 16 articulated stipe segments: tangent unit vectors (length NUM_STIPE_SEGMENTS * 3)
  private readonly segTangents: Float64Array;
  // 17 centerline vertices (base = 0 .. tip = 16)
  public readonly stipeCenterline: Float32Array;
  public stipeTipX: number;
  public stipeTipY: number;
  public stipeTipZ: number;
  public stipeTipDirX: number;
  public stipeTipDirY: number;
  public stipeTipDirZ: number;

  private readonly wiltLeanAngle: number;
  private ageSec = 0;
  private supplyRatioSmooth = 1.0;

  constructor(
    id: number,
    anchor: PrimordiumAnchor,
    species: SpeciesPreset,
    initialTurgorPa: number,
  ) {
    this.id = id;
    this.primordiumId = anchor.id;
    this.myceliumNodeId = anchor.nodeId;
    this.species = species;
    this.baseX = anchor.x;
    this.baseY = Math.max(-0.004, anchor.y); // emerge at soil surface
    this.baseZ = anchor.z;

    this.stipeLengthM = 0.008; // 8 mm initial button pin
    this.stipeRadiusM = species.stipeBaseRadiusM * 0.45;
    this.turgorPa = Math.max(species.yieldPressurePa, initialTurgorPa);
    this.wiltFactor = 0;
    this.maturity = 0;
    this.evaporationRateKgS = 0;

    // Deterministic azimuth for slight natural droop direction if gravity is purely vertical
    this.wiltLeanAngle = ((id * 2.399963229728653) % (Math.PI * 2));

    // Compact live thin-shell resolution for crisp 60 FPS across up to 16 mushrooms
    this.shell = new ThinShellCapSolver(
      {
        species,
        radialRings: 9,
        angularSectors: 48,
      },
      id + 1,
    );

    // Initialize all 16 segment tangents perpendicular to the local substrate surface (+y)
    this.segTangents = new Float64Array(NUM_STIPE_SEGMENTS * 3);
    for (let k = 0; k < NUM_STIPE_SEGMENTS; k++) {
      this.segTangents[k * 3] = 0;
      this.segTangents[k * 3 + 1] = 1;
      this.segTangents[k * 3 + 2] = 0;
    }

    this.stipeCenterline = new Float32Array((NUM_STIPE_SEGMENTS + 1) * 3);
    this.stipeTipX = this.baseX;
    this.stipeTipY = this.baseY + this.stipeLengthM;
    this.stipeTipZ = this.baseZ;
    this.stipeTipDirX = 0;
    this.stipeTipDirY = 1;
    this.stipeTipDirZ = 0;

    this.rebuildCenterline([0, -1, 0]);
  }

  public step(
    dtSec: number,
    mycelium: IMyceliumNetwork,
    env: EnvironmentState,
    obstacles: SolidObstacle[],
  ): void {
    if (dtSec <= 0) return;
    this.ageSec += dtSec;

    const sp = this.species;
    const { etaCap, etaStipe } = computeCo2EtiolationFactors(env.ambientCo2Ppm, sp);

    // 1. Transpiration / evaporative mass flux from cap area
    const capArea = this.shell.getCapAreaM2();
    const vpdRel = Math.max(0, 1.0 - Math.max(0, Math.min(1, env.ambientHumidity)));
    const turgorFactor = Math.max(0.05, Math.min(1.25, this.turgorPa / 4.0e5));
    this.evaporationRateKgS = env.evaporationEnabled
      ? sp.transpirationCoeff * capArea * vpdRel * turgorFactor
      : 0;

    // 2. Estimate hydraulic & carbon demand and draw from basal mycelium node
    const effectiveMaxStipe = sp.maxStipeHeightM * (0.82 + 0.36 * etaStipe);
    const stipeSat = Math.max(0, 1.0 - this.stipeLengthM / Math.max(1e-4, effectiveMaxStipe));
    const potOverP = Math.max(0, this.turgorPa - sp.yieldPressurePa);
    const estElongRate = sp.phiStipe * etaStipe * stipeSat * potOverP * this.stipeLengthM;
    const stipeVolGrowRate = Math.PI * this.stipeRadiusM * this.stipeRadiusM * estElongRate;
    const capVolGrowRate = capArea * sp.shellThicknessM * sp.phiStipe * etaCap * potOverP;
    const evapVolRate = this.evaporationRateKgS / RHO_WATER;

    const waterDemandM3 = Math.max(0, (evapVolRate + stipeVolGrowRate + capVolGrowRate) * dtSec);
    // Dry biomass ~ 8% of fresh water mass, scaled realistically to mycelial carbon reserves
    const carbonDemandKg = Math.max(0, (stipeVolGrowRate + capVolGrowRate) * RHO_WATER * 0.015 * dtSec);

    const supply = mycelium.drawWaterAndCarbonForFruiting(
      this.myceliumNodeId,
      waterDemandM3,
      carbonDemandKg,
    );

    // 3. Couple internal fruiting-body turgor P to basal mycelial turgor minus transpiration draw
    const waterSupplyRatio =
      waterDemandM3 > 1e-15 ? Math.max(0, Math.min(1, supply.waterSuppliedM3 / waterDemandM3)) : 1.0;
    const carbonSupplyRatio =
      carbonDemandKg > 1e-15 ? Math.max(0, Math.min(1, supply.carbonSuppliedKg / carbonDemandKg)) : 1.0;

    // Hydraulic resistance along stipe increases gently with length
    const stipeHydResistance = 6.0e13 * (0.4 + this.stipeLengthM / Math.max(0.02, sp.maxStipeHeightM));
    const transDropPa = stipeHydResistance * evapVolRate;
    // Smooth the supplied/demanded ratio over ~10 min so a single depleted basal reservoir draw
    // cannot trigger pump-and-dump limit cycles; sustained shortfall still lowers turgor.
    const alphaSupply = 1.0 - Math.exp(-dtSec / 600.0);
    this.supplyRatioSmooth += (waterSupplyRatio - this.supplyRatioSmooth) * alphaSupply;
    const targetTurgorPa = Math.max(
      0,
      supply.basalTurgorPa * (0.7 + 0.3 * this.supplyRatioSmooth) - transDropPa,
    );

    // Unconditionally stable exponential hydraulic equilibration (tau ~ 240 s)
    const tauHyd = 240.0;
    const alphaHyd = 1.0 - Math.exp(-dtSec / tauHyd);
    this.turgorPa += (targetTurgorPa - this.turgorPa) * alphaHyd;

    // 4. Drought / hypoxia wilting state when turgor drops below wiltPressurePa
    const targetWilt =
      this.turgorPa < sp.wiltPressurePa
        ? Math.max(0, Math.min(1, (sp.wiltPressurePa - this.turgorPa) / Math.max(1e3, 0.5 * sp.wiltPressurePa)))
        : 0;
    const tauWilt = 300.0;
    const alphaWilt = 1.0 - Math.exp(-dtSec / tauWilt);
    this.wiltFactor += (targetWilt - this.wiltFactor) * alphaWilt;

    // 5. Exact Lockhart stipe elongation
    const carbonFactor = 0.25 + 0.75 * carbonSupplyRatio;
    const phiEff = sp.phiStipe * etaStipe * stipeSat * carbonFactor;
    const prevLen = this.stipeLengthM;
    const nextLen = stepStipeLockhartExact(
      prevLen,
      this.turgorPa,
      phiEff,
      sp.yieldPressurePa,
      dtSec,
    );
    this.stipeLengthM = Math.min(effectiveMaxStipe, nextLen);

    const growthStrain =
      prevLen > 1e-9 && this.stipeLengthM > prevLen ? Math.log(this.stipeLengthM / prevLen) : 0;

    // Radial stipe thickening tracks stipe maturity (slenderer under high CO2 etiolation)
    const lenProg = Math.min(1, this.stipeLengthM / Math.max(1e-4, effectiveMaxStipe));
    const co2Slender = 0.72 + 0.28 * etaCap;
    this.stipeRadiusM = sp.stipeBaseRadiusM * co2Slender * (0.45 + 0.55 * Math.sqrt(lenProg));

    // 6. Gravitropism & Phototropism (strictly gated by growthStrain > 0!)
    if (growthStrain > 0) {
      const [gx, gy, gz] = normalize3(env.gravityDir[0], env.gravityDir[1], env.gravityDir[2]);
      // Anti-gravity upward target -g_hat
      const ugx = -gx;
      const ugy = -gy;
      const ugz = -gz;
      const [lx, ly, lz] = normalize3(env.lightDir[0], env.lightDir[1], env.lightDir[2]);

      const subSteps = Math.min(12, Math.max(1, Math.ceil(growthStrain / 0.08)));
      const dEps = growthStrain / subSteps;

      for (let sub = 0; sub < subSteps; sub++) {
        for (let k = 0; k < NUM_STIPE_SEGMENTS; k++) {
          const sFrac = (k + 0.5) / NUM_STIPE_SEGMENTS;
          // Basal segments stay anchored near substrate normal; upper elongation zone curves strongly
          const zoneWeight = 0.12 + 1.38 * Math.pow(sFrac, 0.85);
          const k3 = k * 3;
          const tx = this.segTangents[k3];
          const ty = this.segTangents[k3 + 1];
          const tz = this.segTangents[k3 + 2];

          const dotG = ugx * tx + ugy * ty + ugz * tz;
          const dotL = lx * tx + ly * ty + lz * tz;

          // d(t_k)/dEps = zoneWeight * [ gamma_g * (u_g - (u_g.t)t) + gamma_L * (L - (L.t)t) ]
          const gg = sp.gravitropismGain * zoneWeight;
          const gl = sp.phototropismGain * zoneWeight;

          const dtx = dEps * (gg * (ugx - dotG * tx) + gl * (lx - dotL * tx));
          const dty = dEps * (gg * (ugy - dotG * ty) + gl * (ly - dotL * ty));
          const dtz = dEps * (gg * (ugz - dotG * tz) + gl * (lz - dotL * tz));

          const [ntx, nty, ntz] = normalize3(tx + dtx, ty + dty, tz + dtz);
          this.segTangents[k3] = ntx;
          this.segTangents[k3 + 1] = nty;
          this.segTangents[k3 + 2] = ntz;
        }
      }
    }

    // 7. Reconstruct articulated stipe centerline (with elastic wilt droop if turgor < wiltPa)
    this.rebuildCenterline(env.gravityDir);

    // 8. Transform solid obstacles into cap-local space and grow/relax thin-shell cap
    const localObstacles = this.transformObstaclesToCapLocal(obstacles);
    this.shell.setWiltFactor(this.wiltFactor);
    this.shell.growAndRelax(dtSec, this.turgorPa, etaCap, localObstacles);

    // 9. Update maturity metric
    const capProg = this.shell.getMaturity();
    const stipeProg = Math.min(1, this.stipeLengthM / Math.max(1e-4, effectiveMaxStipe));
    this.maturity = Math.max(0, Math.min(1, 0.45 * stipeProg + 0.55 * capProg));
  }

  private rebuildCenterline(gravityDir: [number, number, number]): void {
    const [gx, gy, gz] = normalize3(gravityDir[0], gravityDir[1], gravityDir[2]);
    const segLen = this.stipeLengthM / NUM_STIPE_SEGMENTS;

    // Slight lateral horizontal direction for natural cantilever wilt droop
    const leanX = Math.cos(this.wiltLeanAngle);
    const leanZ = Math.sin(this.wiltLeanAngle);

    let px = this.baseX;
    let py = this.baseY;
    let pz = this.baseZ;

    this.stipeCenterline[0] = px;
    this.stipeCenterline[1] = py;
    this.stipeCenterline[2] = pz;

    let lastDx = 0;
    let lastDy = 1;
    let lastDz = 0;

    for (let k = 0; k < NUM_STIPE_SEGMENTS; k++) {
      const k3 = k * 3;
      const sFrac = (k + 1) / NUM_STIPE_SEGMENTS;
      let tx = this.segTangents[k3];
      let ty = this.segTangents[k3 + 1];
      let tz = this.segTangents[k3 + 2];

      // Elastic wilt droop: loss of turgor bending stiffness EI(P) lets gravity droop upper stipe
      if (this.wiltFactor > 1e-4) {
        const droop = this.wiltFactor * 1.15 * sFrac * sFrac;
        tx += droop * (gx * 0.85 + leanX * 0.55);
        ty += droop * gy * 0.95;
        tz += droop * (gz * 0.85 + leanZ * 0.55);
        const [ntx, nty, ntz] = normalize3(tx, ty, tz);
        tx = ntx;
        ty = nty;
        tz = ntz;
      }

      px += tx * segLen;
      py += ty * segLen;
      pz += tz * segLen;

      // Keep stipe tip above soil floor even under severe wilting
      if (py < 0.002) {
        py = 0.002;
      }

      const pIdx = (k + 1) * 3;
      this.stipeCenterline[pIdx] = px;
      this.stipeCenterline[pIdx + 1] = py;
      this.stipeCenterline[pIdx + 2] = pz;

      lastDx = tx;
      lastDy = ty;
      lastDz = tz;
    }

    this.stipeTipX = px;
    this.stipeTipY = py;
    this.stipeTipZ = pz;
    this.stipeTipDirX = lastDx;
    this.stipeTipDirY = lastDy;
    this.stipeTipDirZ = lastDz;
  }

  private transformObstaclesToCapLocal(obstacles: SolidObstacle[]): SolidObstacle[] {
    if (obstacles.length === 0) return obstacles;
    const capR = this.shell.getCapRadiusM();
    const cullDist = capR * 2.5 + 0.05;
    const frame = buildFrameFromUp(this.stipeTipDirX, this.stipeTipDirY, this.stipeTipDirZ);
    const local: SolidObstacle[] = [];

    const toLocal = (wx: number, wy: number, wz: number): [number, number, number] => {
      const dx = wx - this.stipeTipX;
      const dy = wy - this.stipeTipY;
      const dz = wz - this.stipeTipZ;
      return [
        dx * frame.rx + dy * frame.ry + dz * frame.rz,
        dx * this.stipeTipDirX + dy * this.stipeTipDirY + dz * this.stipeTipDirZ,
        dx * frame.fx + dy * frame.fy + dz * frame.fz,
      ];
    };

    for (let i = 0; i < obstacles.length; i++) {
      const o = obstacles[i];
      const mx = 0.5 * (o.x + o.endX) - this.stipeTipX;
      const my = 0.5 * (o.y + o.endY) - this.stipeTipY;
      const mz = 0.5 * (o.z + o.endZ) - this.stipeTipZ;
      const halfSeg = 0.5 * Math.hypot(o.endX - o.x, o.endY - o.y, o.endZ - o.z);
      if (Math.hypot(mx, my, mz) > cullDist + o.radius + halfSeg) continue;

      const [lx, ly, lz] = toLocal(o.x, o.y, o.z);
      const [lex, ley, lez] = toLocal(o.endX, o.endY, o.endZ);
      local.push({
        id: o.id,
        kind: o.kind,
        x: lx,
        y: ly,
        z: lz,
        radius: o.radius,
        endX: lex,
        endY: ley,
        endZ: lez,
      });
    }
    return local;
  }

  public getState(): FruitingBodyState {
    return {
      id: this.id,
      primordiumId: this.primordiumId,
      myceliumNodeId: this.myceliumNodeId,
      speciesId: this.species.id,
      baseX: this.baseX,
      baseY: this.baseY,
      baseZ: this.baseZ,
      stipeLengthM: this.stipeLengthM,
      stipeRadiusM: this.stipeRadiusM,
      stipeTipX: this.stipeTipX,
      stipeTipY: this.stipeTipY,
      stipeTipZ: this.stipeTipZ,
      stipeTipDirX: this.stipeTipDirX,
      stipeTipDirY: this.stipeTipDirY,
      stipeTipDirZ: this.stipeTipDirZ,
      turgorPa: this.turgorPa,
      wiltFactor: this.wiltFactor,
      maturity: this.maturity,
      capRadiusM: this.shell.getCapRadiusM(),
      capAreaM2: this.shell.getCapAreaM2(),
      gillWavenumber: this.shell.measureCircumferentialWavenumber(),
      lamellulaeCount: this.shell.getLamellulaeCount(),
      gillSpacingM: this.shell.getMeanGillSpacingM(),
      veilIntactFraction: this.shell.getVeilIntactFraction(),
      evaporationRateKgS: this.evaporationRateKgS,
      stipeCenterline: this.stipeCenterline,
    };
  }

  public getAgeSec(): number {
    return this.ageSec;
  }
}

export class MorphogenesisSystem {
  public species: SpeciesPreset;
  private readonly bodies: FruitingBody[] = [];
  private readonly spawnedPrimordiaIds = new Set<number>();
  private nextBodyId = 1;

  // Reusable packed render mesh buffers
  private posBuf = new Float32Array(0);
  private normBuf = new Float32Array(0);
  private colBuf = new Float32Array(0);
  private stressBuf = new Float32Array(0);
  private turgorBuf = new Float32Array(0);
  private partBuf = new Float32Array(0);
  private idxBuf = new Uint32Array(0);

  constructor(species: SpeciesPreset) {
    this.species = species;
  }

  public static stepStipeLockhartExact(
    lengthM: number,
    turgorPa: number,
    phi: number,
    yieldPa: number,
    dtSec: number,
  ): number {
    return stepStipeLockhartExact(lengthM, turgorPa, phi, yieldPa, dtSec);
  }

  public stepStipeLockhartExact(
    lengthM: number,
    turgorPa: number,
    phi: number,
    yieldPa: number,
    dtSec: number,
  ): number {
    return stepStipeLockhartExact(lengthM, turgorPa, phi, yieldPa, dtSec);
  }

  public setSpecies(species: SpeciesPreset): void {
    this.species = species;
  }

  public reset(species?: SpeciesPreset): void {
    if (species) this.species = species;
    this.bodies.length = 0;
    this.spawnedPrimordiaIds.clear();
    this.nextBodyId = 1;
  }

  /**
   * Synchronizes active fruiting bodies with newly pinned primordia in the mycelium network.
   * Caps simultaneous active bodies at MAX_ACTIVE_BODIES (~16) for crisp 60 FPS.
   */
  public syncWithPrimordia(mycelium: IMyceliumNetwork, _simTimeSec: number): void {
    const primordia = mycelium.primordia;
    for (let i = 0; i < primordia.length; i++) {
      if (this.bodies.length >= MAX_ACTIVE_BODIES) break;
      const p = primordia[i];
      if (this.spawnedPrimordiaIds.has(p.id)) continue;

      const basalTurgor = mycelium.getNodeTurgorPa(p.nodeId);
      const bodyId = this.nextBodyId++;
      const body = new FruitingBody(
        bodyId,
        p,
        this.species,
        basalTurgor > 0 ? basalTurgor : this.species.pinTurgorThresholdPa,
      );
      this.bodies.push(body);
      this.spawnedPrimordiaIds.add(p.id);

      // Link back on mycelium node if accessible
      if (p.nodeId >= 0 && p.nodeId < mycelium.nodes.length) {
        mycelium.nodes[p.nodeId].fruitingBodyId = bodyId;
      }
    }
  }

  public step(
    dtSec: number,
    mycelium: IMyceliumNetwork,
    env: EnvironmentState,
    obstacles: SolidObstacle[],
  ): void {
    if (dtSec <= 0) return;
    for (let i = 0; i < this.bodies.length; i++) {
      this.bodies[i].step(dtSec, mycelium, env, obstacles);
    }
  }

  public getBodies(): FruitingBodyState[] {
    const out: FruitingBodyState[] = new Array(this.bodies.length);
    for (let i = 0; i < this.bodies.length; i++) {
      out[i] = this.bodies[i].getState();
    }
    return out;
  }

  public getBodyInstances(): readonly FruitingBody[] {
    return this.bodies;
  }

  private ensureRenderCapacity(vertCount: number, indexCount: number): void {
    if (this.posBuf.length < vertCount * 3) {
      const cap = Math.max(vertCount, 2048);
      this.posBuf = new Float32Array(cap * 3);
      this.normBuf = new Float32Array(cap * 3);
      this.colBuf = new Float32Array(cap * 3);
      this.stressBuf = new Float32Array(cap);
      this.turgorBuf = new Float32Array(cap);
      this.partBuf = new Float32Array(cap);
    }
    if (this.idxBuf.length < indexCount) {
      this.idxBuf = new Uint32Array(Math.max(indexCount, 4096));
    }
  }

  /**
   * Packs all active fruiting bodies (articulated stipe tube + cap + gills + fractured veil)
   * into a single world-space indexed triangle mesh buffer for the GPU renderer.
   */
  public getCombinedRenderMesh(): FruitingRenderMesh {
    const stipeRings = NUM_STIPE_SEGMENTS + 1;
    const stipeVertsPerBody = stipeRings * STIPE_RADIAL_SIDES;
    const stipeIndicesPerBody = NUM_STIPE_SEGMENTS * STIPE_RADIAL_SIDES * 6;

    let totalVerts = 0;
    let totalIndices = 0;
    const shellMeshes = new Array(this.bodies.length);

    for (let b = 0; b < this.bodies.length; b++) {
      const sm = this.bodies[b].shell.getMeshBuffers();
      shellMeshes[b] = sm;
      totalVerts += stipeVertsPerBody + sm.vertexCount;
      totalIndices += stipeIndicesPerBody + sm.indices.length;
    }

    this.ensureRenderCapacity(totalVerts, totalIndices);

    let vOffset = 0;
    let iOffset = 0;

    for (let b = 0; b < this.bodies.length; b++) {
      const body = this.bodies[b];
      const sp = body.species;
      const cl = body.stipeCenterline;

      // ----------------------------------------------------------------
      // 1. Pack articulated stipe tube using parallel-transport frames
      // ----------------------------------------------------------------
      const stipeStartV = vOffset;
      // Longitudinal wall stress sigma_parallel = P * R / (2 * h_w)
      const hWall = Math.max(0.0005, sp.shellThicknessM * 0.6);
      const stipeStressPa = (body.turgorPa * body.stipeRadiusM) / (2.0 * hWall);

      let prevTx = 0;
      let prevTy = 1;
      let prevTz = 0;
      let frameRx = 1;
      let frameRy = 0;
      let frameRz = 0;
      let frameFx = 0;
      let frameFy = 0;
      let frameFz = 1;

      for (let r = 0; r < stipeRings; r++) {
        const p3 = r * 3;
        const cx = cl[p3];
        const cy = cl[p3 + 1];
        const cz = cl[p3 + 2];

        let tx: number;
        let ty: number;
        let tz: number;
        if (r < NUM_STIPE_SEGMENTS) {
          tx = cl[p3 + 3] - cx;
          ty = cl[p3 + 4] - cy;
          tz = cl[p3 + 5] - cz;
        } else {
          tx = cx - cl[p3 - 3];
          ty = cy - cl[p3 - 2];
          tz = cz - cl[p3 - 1];
        }
        [tx, ty, tz] = normalize3(tx, ty, tz);

        if (r === 0) {
          const initFrame = buildFrameFromUp(tx, ty, tz);
          frameRx = initFrame.rx;
          frameRy = initFrame.ry;
          frameRz = initFrame.rz;
          frameFx = initFrame.fx;
          frameFy = initFrame.fy;
          frameFz = initFrame.fz;
        } else {
          // Parallel-transport rotation from prevT to T (eliminates twisting!)
          const ax = prevTy * tz - prevTz * ty;
          const ay = prevTz * tx - prevTx * tz;
          const az = prevTx * ty - prevTy * tx;
          const sinA = Math.hypot(ax, ay, az);
          if (sinA > 1e-6) {
            const cosA = Math.max(-1, Math.min(1, prevTx * tx + prevTy * ty + prevTz * tz));
            const kx = ax / sinA;
            const ky = ay / sinA;
            const kz = az / sinA;
            const oneMinusC = 1.0 - cosA;

            // Rodrigues rotation on frameR
            const kDotR = kx * frameRx + ky * frameRy + kz * frameRz;
            const crx = ky * frameRz - kz * frameRy;
            const cry = kz * frameRx - kx * frameRz;
            const crz = kx * frameRy - ky * frameRx;
            frameRx = frameRx * cosA + crx * sinA + kx * kDotR * oneMinusC;
            frameRy = frameRy * cosA + cry * sinA + ky * kDotR * oneMinusC;
            frameRz = frameRz * cosA + crz * sinA + kz * kDotR * oneMinusC;
          }
          // Re-orthogonalize
          const rDotT = frameRx * tx + frameRy * ty + frameRz * tz;
          [frameRx, frameRy, frameRz] = normalize3(
            frameRx - rDotT * tx,
            frameRy - rDotT * ty,
            frameRz - rDotT * tz,
          );
          frameFx = ty * frameRz - tz * frameRy;
          frameFy = tz * frameRx - tx * frameRz;
          frameFz = tx * frameRy - ty * frameRx;
        }
        prevTx = tx;
        prevTy = ty;
        prevTz = tz;

        const frac = r / NUM_STIPE_SEGMENTS;
        // Bulbous mycelial base tapering smoothly toward apex (chanterelle flares into funnel)
        let taper = 1.24 - 0.36 * frac + 0.14 * Math.exp(-frac * 9.0);
        if (sp.hymeniumType === 'false-folds' && frac > 0.65) {
          taper += 0.65 * Math.pow((frac - 0.65) / 0.35, 1.5);
        }
        const rad = body.stipeRadiusM * taper;

        // Subtle basal earth darkening + wilt desaturation
        const baseShade = 0.84 + 0.16 * Math.min(1, frac * 3.0);
        const wiltShade = 1.0 - 0.18 * body.wiltFactor;

        for (let s = 0; s < STIPE_RADIAL_SIDES; s++) {
          const theta = (2.0 * Math.PI * s) / STIPE_RADIAL_SIDES;
          const ct = Math.cos(theta);
          const st = Math.sin(theta);

          const nx = frameRx * ct + frameFx * st;
          const ny = frameRy * ct + frameFy * st;
          const nz = frameRz * ct + frameFz * st;

          const v3 = vOffset * 3;
          this.posBuf[v3] = cx + nx * rad;
          this.posBuf[v3 + 1] = cy + ny * rad;
          this.posBuf[v3 + 2] = cz + nz * rad;

          this.normBuf[v3] = nx;
          this.normBuf[v3 + 1] = ny;
          this.normBuf[v3 + 2] = nz;

          const fiber = 0.96 + 0.04 * Math.sin(theta * 6.0);
          this.colBuf[v3] = sp.stipeColor[0] * baseShade * wiltShade * fiber;
          this.colBuf[v3 + 1] = sp.stipeColor[1] * baseShade * wiltShade * fiber;
          this.colBuf[v3 + 2] = sp.stipeColor[2] * baseShade * wiltShade * fiber;

          this.stressBuf[vOffset] = stipeStressPa;
          this.turgorBuf[vOffset] = body.turgorPa;
          this.partBuf[vOffset] = MeshPartId.STIPE;
          vOffset++;
        }
      }

      // Stipe tube triangles
      for (let r = 0; r < NUM_STIPE_SEGMENTS; r++) {
        const rowA = stipeStartV + r * STIPE_RADIAL_SIDES;
        const rowB = stipeStartV + (r + 1) * STIPE_RADIAL_SIDES;
        for (let s = 0; s < STIPE_RADIAL_SIDES; s++) {
          const sNext = (s + 1) % STIPE_RADIAL_SIDES;
          const a = rowA + s;
          const bIdx = rowA + sNext;
          const c = rowB + s;
          const d = rowB + sNext;

          this.idxBuf[iOffset++] = a;
          this.idxBuf[iOffset++] = bIdx;
          this.idxBuf[iOffset++] = d;

          this.idxBuf[iOffset++] = a;
          this.idxBuf[iOffset++] = d;
          this.idxBuf[iOffset++] = c;
        }
      }

      // ----------------------------------------------------------------
      // 2. Pack thin-shell cap + gills + fractured veil in world space
      // ----------------------------------------------------------------
      const sm = shellMeshes[b];
      const capStartV = vOffset;
      const capFrame = buildFrameFromUp(body.stipeTipDirX, body.stipeTipDirY, body.stipeTipDirZ);

      // Anchor cap slightly tucked over stipe apex for seamless biological junction
      const tuck =
        sp.hymeniumType === 'honeycomb'
          ? body.shell.getCapRadiusM() * 0.25
          : sp.hymeniumType === 'false-folds'
            ? body.shell.getCapRadiusM() * 0.14
            : body.shell.getCapRadiusM() * 0.32;
      const ox = body.stipeTipX - body.stipeTipDirX * tuck;
      const oy = body.stipeTipY - body.stipeTipDirY * tuck;
      const oz = body.stipeTipZ - body.stipeTipDirZ * tuck;

      const capRadius = Math.max(1e-4, body.shell.getCapRadiusM());
      const wiltShade = 1.0 - 0.16 * body.wiltFactor;

      for (let v = 0; v < sm.vertexCount; v++) {
        const s3 = v * 3;
        const lx = sm.positions[s3];
        const ly = sm.positions[s3 + 1];
        const lz = sm.positions[s3 + 2];

        const wx = ox + capFrame.rx * lx + body.stipeTipDirX * ly + capFrame.fx * lz;
        const wy = oy + capFrame.ry * lx + body.stipeTipDirY * ly + capFrame.fy * lz;
        const wz = oz + capFrame.rz * lx + body.stipeTipDirZ * ly + capFrame.fz * lz;

        const lnx = sm.normals[s3];
        const lny = sm.normals[s3 + 1];
        const lnz = sm.normals[s3 + 2];

        const wnx = capFrame.rx * lnx + body.stipeTipDirX * lny + capFrame.fx * lnz;
        const wny = capFrame.ry * lnx + body.stipeTipDirY * lny + capFrame.fy * lnz;
        const wnz = capFrame.rz * lnx + body.stipeTipDirZ * lny + capFrame.fz * lnz;

        const v3 = vOffset * 3;
        this.posBuf[v3] = wx;
        this.posBuf[v3 + 1] = wy;
        this.posBuf[v3 + 2] = wz;

        this.normBuf[v3] = wnx;
        this.normBuf[v3 + 1] = wny;
        this.normBuf[v3 + 2] = wnz;

        const partId = sm.partIds[v];
        if (partId === MeshPartId.CAP_TOP) {
          // Subtle umbo / margin organic color gradient
          const rNorm = Math.min(1, Math.hypot(lx, lz) / capRadius);
          const umboTint = sp.id === 'parasol' ? 0.78 + 0.22 * Math.min(1, rNorm * 2.2) : 0.92 + 0.08 * rNorm;
          this.colBuf[v3] = sp.capColor[0] * umboTint * wiltShade;
          this.colBuf[v3 + 1] = sp.capColor[1] * umboTint * wiltShade;
          this.colBuf[v3 + 2] = sp.capColor[2] * umboTint * wiltShade;
        } else if (partId === MeshPartId.GILLS) {
          this.colBuf[v3] = sp.gillColor[0] * wiltShade;
          this.colBuf[v3 + 1] = sp.gillColor[1] * wiltShade;
          this.colBuf[v3 + 2] = sp.gillColor[2] * wiltShade;
        } else {
          // Partial veil / annulus skirt
          this.colBuf[v3] = Math.min(1, sp.stipeColor[0] * 1.03);
          this.colBuf[v3 + 1] = Math.min(1, sp.stipeColor[1] * 1.02);
          this.colBuf[v3 + 2] = Math.min(1, sp.stipeColor[2] * 0.98);
        }

        this.stressBuf[vOffset] = sm.stressVM[v];
        this.turgorBuf[vOffset] = body.turgorPa;
        this.partBuf[vOffset] = partId;
        vOffset++;
      }

      for (let i = 0; i < sm.indices.length; i++) {
        this.idxBuf[iOffset++] = capStartV + sm.indices[i];
      }
    }

    return {
      vertexCount: totalVerts,
      indexCount: totalIndices,
      positions: this.posBuf.subarray(0, totalVerts * 3),
      normals: this.normBuf.subarray(0, totalVerts * 3),
      colors: this.colBuf.subarray(0, totalVerts * 3),
      stressVM: this.stressBuf.subarray(0, totalVerts),
      turgorPa: this.turgorBuf.subarray(0, totalVerts),
      partIds: this.partBuf.subarray(0, totalVerts),
      indices: this.idxBuf.subarray(0, totalIndices),
    };
  }
}
