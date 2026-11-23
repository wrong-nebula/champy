// ============================================================================
// SPOROCARP — Layer 2: Subterranean 3D Mycelial Network
// - Explicit 3D hyphal graph with Lockhart chemotropic tip extension
// - Lateral branching and topological-distance >= 3 anastomosis (loop fusion)
// - Coupled osmotic water uptake + implicit Jacobi Hagen–Poiseuille hydraulics
// - Boswell et al. (2002) nutrient uptake, active translocation & Fairy Rings
// - Five-condition primordium pinning trigger & fruiting hydraulic supply
// ============================================================================

import {
  SoilMaterialId,
  type EnvironmentState,
  type IMyceliumNetwork,
  type ISoilGrid,
  type MyceliumEdge,
  type MyceliumNode,
  type MyceliumRenderBuffers,
  type PrimordiumAnchor,
  type SpeciesPreset,
} from '../../types/contract.ts';
import { SOIL_MATERIAL_LIST, vgThetaFromHead } from '../soil/vanGenuchten.ts';

const MAX_NODES = 2200;
const MAX_EDGES = 4800;
const MAX_PRIMORDIA = 18;
const CATCHMENT_MAX_NODES = 48;
const CATCHMENT_MAX_HOPS = 7;
const CATCHMENT_VOXEL_FRACTION = 0.10; // max fraction of extractable voxel water per fruiting call
// Plant-available water floor per material: theta at h = -50 m (~ -0.5 MPa)
const THETA_FLOOR_BY_MAT = Float64Array.from(SOIL_MATERIAL_LIST, (m) =>
  m.id === SoilMaterialId.STONE ? 0.0 : vgThetaFromHead(-50.0, m),
);

const ETA_CYTOPLASM = 1.5e-3; // Pa * s
const LP_A_NODE = 1.0e-16; // m^3 / (Pa * s) effective transmembrane conductance per node
const G_LEAK_BASE = 0.22 * LP_A_NODE; // baseline wall/maintenance water loss conductance
const C_WALL_NODE = 5.5e-14; // m^3 / Pa elastic turgor capacitance per node
const PI_BASE_PA = 4.6e5; // Pa baseline cellular osmotic pressure
const PI_CARBON_GAIN = 8.5e9; // Pa / kg osmotic solute contribution from internal carbon
const PI_MAX_PA = 1.15e6; // Pa upper osmotic pressure cap
const K_O2_OSM = 2.0; // mol / m^3 half-saturation for aerobic osmotic pumping
const K_O2_TIP = 1.2; // mol / m^3 half-saturation for tip Lockhart extension

const PHI_TIP = 2.4e-12; // m / (Pa * s) hyphal tip wall extensibility (~30 mm/day in rich moist soil)
const SEGMENT_LENGTH_M = 0.013; // m between structural hyphal nodes
const ANASTOMOSIS_RADIUS_M = 0.0105; // m capture radius for loop fusion
const HASH_CELL_M = 0.016; // spatial hash cell size

const K_CARBON_UPTAKE = 5.0e-5; // kg / (m^3 * s) max soluble sugar uptake rate
const KM_SOLUBLE_CARBON = 0.12; // kg / m^3
const CARBON_NODE_CAP_KG = 8.0e-5; // kg max stored carbon per node
const K_CARBON_MAINT = 4.5e-7; // 1 / s carbon maintenance respiration
const D_INTERNAL_CARBON = 2.2e-7; // m / s passive internal diffusion coeff
const V_ACTIVE_TRANSLOC = 5.5e-5; // 1 / s active Boswell forward translocation rate
const ENZYME_SECRETE_RATE = 4.0e-6; // kg / (m^3 * s) extracellular enzyme secretion per active node

const INIT_CORD_RADIUS_M = 1.4e-5; // 14 um young hyphal bundle
const MAX_CORD_RADIUS_M = 7.5e-5; // 75 um mature rhizomorph cord

export class MyceliumNetwork implements IMyceliumNetwork {
  readonly nodes: MyceliumNode[] = [];
  readonly edges: MyceliumEdge[] = [];
  readonly primordia: PrimordiumAnchor[] = [];

  private readonly soil: ISoilGrid;
  private species: SpeciesPreset;
  private simTimeSec = 0.0;
  private anastomosisCount = 0;
  private nextPrimordiumId = 1;

  // Adjacency list: nodeId -> array of edge indices
  private readonly adj: number[][] = [];
  // Spatial hash: packed cell key -> array of nodeIds
  private readonly spatialHash = new Map<number, number[]>();
  private readonly nodeHashKey: number[] = [];
  // Tip segment elongation tracker (distance grown from parent node)
  private readonly tipParentId: number[] = [];
  // Pending fruiting hydraulic draw rate (m^3 / s) per node for next pressure solve
  private readonly fruitingFlowM3s: Float64Array = new Float64Array(MAX_NODES);
  // Fruiting catchment scratch (BFS over hyphal graph)
  private readonly catchScratch: Int32Array = new Int32Array(CATCHMENT_MAX_NODES);
  private readonly catchCaps: Float64Array = new Float64Array(CATCHMENT_MAX_NODES);
  private readonly catchSeen: Int32Array = new Int32Array(MAX_NODES);
  private readonly catchHops: Int32Array = new Int32Array(MAX_NODES);
  private readonly voxSeen = new Map<number, number>();
  private catchStamp = 0;
  private voxStampCounter = 0;
  // Reusable scratch arrays for implicit Jacobi pressure & carbon solvers
  private readonly pScratch: Float64Array = new Float64Array(MAX_NODES);
  private readonly gUpScratch: Float64Array = new Float64Array(MAX_NODES);
  private readonly gLeakScratch: Float64Array = new Float64Array(MAX_NODES);
  private readonly psiPlusPiScratch: Float64Array = new Float64Array(MAX_NODES);
  private readonly cDeltaScratch: Float64Array = new Float64Array(MAX_NODES);
  // Pre-allocated render buffers
  private readonly segBuffer = new Float32Array(MAX_EDGES * 8);
  private readonly nodeBuffer = new Float32Array(MAX_NODES * 4);

  // Deterministic PRNG state (Mulberry32)
  private rngState = 0x5f3759df;

  constructor(soil: ISoilGrid, species: SpeciesPreset) {
    this.soil = soil;
    this.species = species;
  }

  inoculate(
    worldX: number,
    worldY: number,
    worldZ: number,
    initialCarbonKg = 3.0e-6,
  ): number {
    if (this.nodes.length >= MAX_NODES) return -1;

    // Clamp inside valid non-STONE soil region
    const pos = this.clampToValidSoil(worldX, worldY, worldZ);
    const rootCarbon = Math.min(CARBON_NODE_CAP_KG, Math.max(2.0e-6, initialCarbonKg * 0.28));
    const rootId = this.createNode(pos[0], pos[1], pos[2], rootCarbon, false, 0, 0, 1, -1);

    // Radiate 6 symmetric initial hyphal tips in the horizontal/slightly-angled plane
    const nTips = 6;
    const tipCarbon = Math.min(
      CARBON_NODE_CAP_KG,
      Math.max(1.5e-6, (initialCarbonKg * 0.72) / nTips),
    );
    const initLen = 0.0065;
    for (let k = 0; k < nTips; k++) {
      if (this.nodes.length >= MAX_NODES) break;
      const angle = (k / nTips) * Math.PI * 2.0 + 0.17;
      const dx = Math.cos(angle);
      const dz = Math.sin(angle);
      const dy = (k % 2 === 0 ? 0.12 : -0.1) * 0.5;
      const invLen = 1.0 / Math.hypot(dx, dy, dz);
      const ux = dx * invLen;
      const uy = dy * invLen;
      const uz = dz * invLen;

      const tx = pos[0] + ux * initLen;
      const ty = pos[1] + uy * initLen;
      const tz = pos[2] + uz * initLen;
      const clamped = this.clampToValidSoil(tx, ty, tz);
      const tipId = this.createNode(
        clamped[0],
        clamped[1],
        clamped[2],
        tipCarbon,
        true,
        ux,
        uy,
        uz,
        rootId,
      );
      this.createEdge(rootId, tipId, INIT_CORD_RADIUS_M * 1.15, false);
    }

    return rootId;
  }

  step(dtSec: number, env: EnvironmentState, species: SpeciesPreset): void {
    this.species = species;
    if (dtSec <= 0 || !Number.isFinite(dtSec) || this.nodes.length === 0) return;

    // Sub-step large timeWarp increments cleanly so tips & anastomosis stay accurate
    const nSub = dtSec <= 600.0 ? 1 : Math.min(12, Math.ceil(dtSec / 600.0));
    const subDt = dtSec / nSub;

    for (let s = 0; s < nSub; s++) {
      this.simTimeSec += subDt;
      this.stepCarbonAndRespiration(subDt, env);
      this.stepHydraulics(subDt, env);
      this.stepGrowthAndAnastomosis(subDt);
      this.stepPinning(env, species);
    }

    // Decay fruiting flow demand between external fruiting updates
    const n = this.nodes.length;
    for (let i = 0; i < n; i++) {
      this.fruitingFlowM3s[i] *= 0.5;
    }
  }

  drawWaterAndCarbonForFruiting(
    nodeId: number,
    waterVolumeM3: number,
    carbonKg: number,
  ): { waterSuppliedM3: number; carbonSuppliedKg: number; basalTurgorPa: number } {
    if (nodeId < 0 || nodeId >= this.nodes.length) {
      return { waterSuppliedM3: 0, carbonSuppliedKg: 0, basalTurgorPa: 0 };
    }
    const node = this.nodes[nodeId];
    const soil = this.soil;
    const voxVol = soil.dx * soil.dx * soil.dx;

    // Hydraulic catchment: breadth-first walk through the (anastomosed) hyphal network. The
    // mycelium is one connected pressure vessel, so a fruiting body is fed by the whole nearby
    // network rather than a single basal voxel (prevents pump-and-dump limit cycles).
    const cat = this.catchScratch;
    const seen = this.catchSeen;
    let nCat = 0;
    cat[nCat++] = nodeId;
    const stamp = ++this.catchStamp;
    seen[nodeId] = stamp;
    for (let head = 0; head < nCat && nCat < CATCHMENT_MAX_NODES; head++) {
      const cur = cat[head];
      if (this.catchHops[cur] >= CATCHMENT_MAX_HOPS && cur !== nodeId) continue;
      const incident = this.adj[cur];
      for (let k = 0; k < incident.length && nCat < CATCHMENT_MAX_NODES; k++) {
        const e = this.edges[incident[k]];
        const nb = e.a === cur ? e.b : e.a;
        if (seen[nb] === stamp) continue;
        seen[nb] = stamp;
        this.catchHops[nb] = (cur === nodeId ? 0 : this.catchHops[cur]) + 1;
        cat[nCat++] = nb;
      }
    }
    this.catchHops[nodeId] = 0;

    // Mean catchment turgor (hydraulically connected network equilibrates pressure)
    let meanP = 0.0;
    for (let k = 0; k < nCat; k++) meanP += this.nodes[cat[k]].turgorPa;
    meanP /= nCat;

    // Water supply gated by catchment turgor (collapses under drought or hypoxia)
    let waterSuppliedM3 = 0.0;
    if (waterVolumeM3 > 0 && meanP > 1.5e4) {
      const turgorGate = Math.min(1.0, meanP / 2.2e5);
      const desired = waterVolumeM3 * turgorGate;

      // Per-voxel extractable water above the plant-available floor, deduplicated by voxel
      const caps = this.catchCaps;
      const voxStamp = ++this.voxStampCounter;
      let capTotal = 0.0;
      for (let k = 0; k < nCat; k++) {
        const nd = this.nodes[cat[k]];
        const v = nd.voxelIdx;
        caps[k] = 0.0;
        if (this.voxSeen.get(v) === voxStamp) continue;
        this.voxSeen.set(v, voxStamp);
        const matId = soil.material[v];
        if (matId === SoilMaterialId.STONE) continue;
        const floor = THETA_FLOOR_BY_MAT[matId];
        const avail = soil.theta[v] - floor;
        if (avail > 0) {
          const c = avail * voxVol * CATCHMENT_VOXEL_FRACTION * Math.min(1.0, nd.turgorPa / 1.0e5);
          caps[k] = c;
          capTotal += c;
        }
      }
      if (capTotal > 0) {
        const target = Math.min(desired, capTotal);
        for (let k = 0; k < nCat; k++) {
          if (caps[k] <= 0) continue;
          waterSuppliedM3 += soil.applyHyphalWaterSink(
            this.nodes[cat[k]].voxelIdx,
            (target * caps[k]) / capTotal,
          );
        }
      }

      // Smooth, bounded elastic drawdown spread over the whole catchment
      if (waterSuppliedM3 > 0) {
        const frac = Math.min(1.0, waterSuppliedM3 / 6.0e-8);
        const drop = 0.06 * frac;
        for (let k = 0; k < nCat; k++) {
          const nd = this.nodes[cat[k]];
          nd.turgorPa -= nd.turgorPa * drop;
        }
        this.fruitingFlowM3s[nodeId] += Math.min(2.0e-11, waterSuppliedM3 / 120.0);
        meanP *= 1.0 - drop;
      }
    }

    // Carbon supply drawn proportionally from carbon-rich catchment nodes (keeps a reserve)
    let carbonSuppliedKg = 0.0;
    if (carbonKg > 0) {
      let avTotal = 0.0;
      const caps = this.catchCaps;
      for (let k = 0; k < nCat; k++) {
        const nd = this.nodes[cat[k]];
        const av = Math.max(0.0, nd.carbonKg - 3.0e-6) * (k === 0 ? 0.6 : 0.25);
        caps[k] = av;
        avTotal += av;
      }
      if (avTotal > 0) {
        const take = Math.min(carbonKg, avTotal);
        for (let k = 0; k < nCat; k++) {
          const t = (take * caps[k]) / avTotal;
          this.nodes[cat[k]].carbonKg -= t;
        }
        carbonSuppliedKg = take;
      }
    }

    // Sync primordium reserve display if present
    if (node.hasPrimordium) {
      for (let p = 0; p < this.primordia.length; p++) {
        if (this.primordia[p].nodeId === nodeId) {
          this.primordia[p].carbonReserveKg = node.carbonKg;
          break;
        }
      }
    }

    return {
      waterSuppliedM3,
      carbonSuppliedKg,
      basalTurgorPa: meanP,
    };
  }

  getNodeTurgorPa(nodeId: number): number {
    if (nodeId < 0 || nodeId >= this.nodes.length) return 0.0;
    return this.nodes[nodeId].turgorPa;
  }

  getNodeCarbonKg(nodeId: number): number {
    if (nodeId < 0 || nodeId >= this.nodes.length) return 0.0;
    return this.nodes[nodeId].carbonKg;
  }

  getAnastomosisCount(): number {
    return this.anastomosisCount;
  }

  getPackedRenderBuffers(): MyceliumRenderBuffers {
    const segCount = this.edges.length;
    const nodeCount = this.nodes.length;
    const segs = this.segBuffer;
    const nds = this.nodeBuffer;

    for (let i = 0; i < segCount; i++) {
      const e = this.edges[i];
      const na = this.nodes[e.a];
      const nb = this.nodes[e.b];
      const off = i * 8;
      segs[off] = na.x;
      segs[off + 1] = na.y;
      segs[off + 2] = na.z;
      segs[off + 3] = nb.x;
      segs[off + 4] = nb.y;
      segs[off + 5] = nb.z;
      segs[off + 6] = e.radiusM;
      const turgNorm = (0.5 * (na.turgorPa + nb.turgorPa)) / 6.5e5;
      segs[off + 7] = turgNorm < 0.0 ? 0.0 : turgNorm > 1.0 ? 1.0 : turgNorm;
    }

    for (let i = 0; i < nodeCount; i++) {
      const n = this.nodes[i];
      const off = i * 4;
      nds[off] = n.x;
      nds[off + 1] = n.y;
      nds[off + 2] = n.z;
      nds[off + 3] = n.turgorPa;
    }

    return {
      segmentCount: segCount,
      segments: segs.subarray(0, segCount * 8),
      nodeCount,
      nodes: nds.subarray(0, nodeCount * 4),
    };
  }

  // --------------------------------------------------------------------------
  // Step 1: Coupled Osmotic Uptake + Implicit Jacobi Hagen–Poiseuille Solver
  // --------------------------------------------------------------------------

  private stepHydraulics(dtSec: number, env: EnvironmentState): void {
    const n = this.nodes.length;
    const soil = this.soil;
    const pOld = this.pScratch;
    const gUp = this.gUpScratch;
    const gLeak = this.gLeakScratch;
    const psiPlusPi = this.psiPlusPiScratch;

    const dryAirPenalty = env.ambientHumidity < 0.55 ? 1.0 + (0.55 - env.ambientHumidity) * 1.4 : 1.0;

    for (let i = 0; i < n; i++) {
      const node = this.nodes[i];
      const v = node.voxelIdx;
      const matId = soil.material[v] as SoilMaterialId;
      const mat = SOIL_MATERIAL_LIST[matId];
      const th = soil.theta[v];
      const psiSoil = soil.psi[v];
      const o2 = soil.o2[v];

      // Aerobic osmotic pumping factor g_O2: collapses under flooded hypoxia!
      const o2Sq = o2 * o2;
      let gO2 = o2Sq / (K_O2_OSM * K_O2_OSM + o2Sq);
      if (o2 < 0.35) {
        gO2 *= o2 / 0.35;
      }

      const piRaw = Math.min(PI_MAX_PA, PI_BASE_PA + PI_CARBON_GAIN * node.carbonKg);
      const pi = piRaw * gO2;
      node.osmoticPa = pi;

      // Rhizosphere water-film contact factor: drops when soil drains dry
      const span = mat.thetaS - mat.thetaR;
      const se = span > 0 ? Math.max(0.0, Math.min(1.0, (th - mat.thetaR) / span)) : 0.0;
      const contact = se <= 0.06 ? 0.0 : se >= 0.28 ? 1.0 : (se - 0.06) / 0.22;
      const contact2 = contact * contact;

      const driving = psiSoil + pi;
      psiPlusPi[i] = driving;

      // Inflow requires water-film contact & aerobic pumping; dehydration/hypoxia can always drain
      if (driving >= node.turgorPa) {
        gUp[i] = LP_A_NODE * contact2 * Math.max(0.05, gO2);
      } else {
        gUp[i] = LP_A_NODE * (0.25 + 0.75 * contact2);
      }

      const surfaceMult = node.y > -0.025 ? dryAirPenalty : 1.0;
      gLeak[i] = G_LEAK_BASE * surfaceMult;
      pOld[i] = node.turgorPa;
    }

    // Unconditionally stable implicit Gauss-Seidel / Jacobi relaxation for P_i
    const capOverDt = C_WALL_NODE / dtSec;
    const iters = 8;
    for (let it = 0; it < iters; it++) {
      for (let i = 0; i < n; i++) {
        const incident = this.adj[i];
        let sumG = 0.0;
        let sumGP = 0.0;
        for (let k = 0; k < incident.length; k++) {
          const e = this.edges[incident[k]];
          const j = e.a === i ? e.b : e.a;
          const g = e.conductance;
          sumG += g;
          sumGP += g * this.nodes[j].turgorPa;
        }
        const diag = capOverDt + gUp[i] + gLeak[i] + sumG;
        const rhs =
          capOverDt * pOld[i] +
          gUp[i] * psiPlusPi[i] +
          sumGP -
          this.fruitingFlowM3s[i];
        const pNext = rhs / diag;
        this.nodes[i].turgorPa = pNext > 0.0 ? pNext : 0.0;
      }
    }

    // Update Hagen-Poiseuille edge fluxes & gentle Murray's-law cord thickening
    const m = this.edges.length;
    for (let eIdx = 0; eIdx < m; eIdx++) {
      const e = this.edges[eIdx];
      const pa = this.nodes[e.a].turgorPa;
      const pb = this.nodes[e.b].turgorPa;
      const q = e.conductance * (pa - pb);
      e.fluxM3s = q;

      const absQ = Math.abs(q);
      if (absQ > 2.0e-11 && e.radiusM < MAX_CORD_RADIUS_M) {
        const dr = Math.min(2.0e-6, absQ * 1.2e-1 * dtSec);
        e.radiusM = Math.min(MAX_CORD_RADIUS_M, e.radiusM + dr);
        e.conductance = this.computeConductance(e.radiusM, e.lengthM);
      }
    }

    // Apply actual soil water sink / hydraulic redistribution exchange (capped per voxel per step)
    const maxNodeExchangeM3 = 0.015 * soil.dx * soil.dx * soil.dx;
    for (let i = 0; i < n; i++) {
      const node = this.nodes[i];
      const jw = gUp[i] * (psiPlusPi[i] - node.turgorPa);
      if (Math.abs(jw) > 1e-19) {
        let reqVol = jw * dtSec;
        if (reqVol > maxNodeExchangeM3) reqVol = maxNodeExchangeM3;
        else if (reqVol < -maxNodeExchangeM3) reqVol = -maxNodeExchangeM3;
        const actualVol = soil.applyHyphalWaterSink(node.voxelIdx, reqVol);
        // If soil voxel ran dry (could not supply requested inflow), damp turgor realistically
        if (reqVol > 1e-16 && actualVol < reqVol * 0.5) {
          const ratio = actualVol / reqVol;
          node.turgorPa *= 0.75 + 0.25 * ratio;
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // Step 2: Boswell Nutrient Uptake, Enzyme Secretion, Respiration & Transport
  // --------------------------------------------------------------------------

  private stepCarbonAndRespiration(dtSec: number, env: EnvironmentState): void {
    const n = this.nodes.length;
    const soil = this.soil;
    const voxVol = soil.dx * soil.dx * soil.dx;
    const { nx, ny, nz } = soil;
    const strideY = nz * nx;
    const cDelta = this.cDeltaScratch;

    const tFactor = Math.max(
      0.25,
      Math.min(2.0, Math.pow(1.6, (env.ambientTempC - 15.0) / 10.0)),
    );

    // 1. Soil sugar uptake (voxel + 6-neighbor fine absorptive hyphal halo), enzyme secretion & O2 respiration
    for (let i = 0; i < n; i++) {
      const node = this.nodes[i];
      node.ageSec += dtSec;
      cDelta[i] = 0.0;

      const v = node.voxelIdx;
      const o2 = soil.o2[v];
      const fO2 = o2 / (0.6 + o2);
      const matId = soil.material[v] as SoilMaterialId;
      const mat = SOIL_MATERIAL_LIST[matId];
      const se =
        mat.thetaS > mat.thetaR
          ? Math.max(0.0, Math.min(1.0, (soil.theta[v] - mat.thetaR) / (mat.thetaS - mat.thetaR)))
          : 0.0;
      const waterGate = se > 0.22 ? 1.0 : se / 0.22;

      let totalUptake = 0.0;
      if (fO2 > 0.02 && waterGate > 0.05 && node.carbonKg < CARBON_NODE_CAP_KG) {
        const uptakeScale = K_CARBON_UPTAKE * fO2 * waterGate * tFactor * voxVol * dtSec;
        // Center voxel (full weight)
        const cs0 = soil.solubleCarbon[v];
        if (cs0 > 1e-6) {
          const demand0 = uptakeScale * (cs0 / (KM_SOLUBLE_CARBON + cs0));
          totalUptake += soil.consumeSolubleCarbon(v, demand0);
        }

        // 6 face-neighbor voxels (absorptive hyphal halo ~1 cm)
        const iy = Math.floor(v / strideY);
        const rem = v - iy * strideY;
        const iz = Math.floor(rem / nx);
        const ix = rem - iz * nx;
        const haloWeight = 0.45;

        if (ix > 0) {
          const j = v - 1;
          const cs = soil.solubleCarbon[j];
          if (cs > 1e-6) {
            totalUptake += soil.consumeSolubleCarbon(
              j,
              uptakeScale * haloWeight * (cs / (KM_SOLUBLE_CARBON + cs)),
            );
          }
        }
        if (ix + 1 < nx) {
          const j = v + 1;
          const cs = soil.solubleCarbon[j];
          if (cs > 1e-6) {
            totalUptake += soil.consumeSolubleCarbon(
              j,
              uptakeScale * haloWeight * (cs / (KM_SOLUBLE_CARBON + cs)),
            );
          }
        }
        if (iz > 0) {
          const j = v - nx;
          const cs = soil.solubleCarbon[j];
          if (cs > 1e-6) {
            totalUptake += soil.consumeSolubleCarbon(
              j,
              uptakeScale * haloWeight * (cs / (KM_SOLUBLE_CARBON + cs)),
            );
          }
        }
        if (iz + 1 < nz) {
          const j = v + nx;
          const cs = soil.solubleCarbon[j];
          if (cs > 1e-6) {
            totalUptake += soil.consumeSolubleCarbon(
              j,
              uptakeScale * haloWeight * (cs / (KM_SOLUBLE_CARBON + cs)),
            );
          }
        }
        if (iy > 0) {
          const j = v - strideY;
          const cs = soil.solubleCarbon[j];
          if (cs > 1e-6) {
            totalUptake += soil.consumeSolubleCarbon(
              j,
              uptakeScale * haloWeight * (cs / (KM_SOLUBLE_CARBON + cs)),
            );
          }
        }
        if (iy + 1 < ny) {
          const j = v + strideY;
          const cs = soil.solubleCarbon[j];
          if (cs > 1e-6) {
            totalUptake += soil.consumeSolubleCarbon(
              j,
              uptakeScale * haloWeight * (cs / (KM_SOLUBLE_CARBON + cs)),
            );
          }
        }

        // Secrete extracellular cellulase/laccase when lignin is present
        if (soil.lignin[v] > 0.1) {
          soil.secreteEnzyme(v, ENZYME_SECRETE_RATE * fO2 * waterGate * dtSec);
        }
      }

      // Maintenance carbon cost
      const maintCost = Math.min(
        node.carbonKg * 0.5,
        (K_CARBON_MAINT * node.carbonKg + 2.5e-12) * dtSec,
      );
      node.carbonKg = Math.min(
        CARBON_NODE_CAP_KG,
        Math.max(0.0, node.carbonKg + totalUptake - maintCost),
      );

      // Aerobic respiration coupling to soil O2 / CO2 (safe in aerated loam, drives hypoxia when flooded!)
      if (o2 > 1e-5) {
        const o2Consumed = (2.6e-11 + 3.0e-6 * (totalUptake / dtSec)) * fO2 * dtSec;
        soil.applyRespiration(v, o2Consumed, o2Consumed);
      }
    }

    // 2. Internal translocation along edges: passive diffusion + bulk advection + Boswell active forward transport
    const m = this.edges.length;
    for (let eIdx = 0; eIdx < m; eIdx++) {
      const e = this.edges[eIdx];
      const na = this.nodes[e.a];
      const nb = this.nodes[e.b];
      const invLen = 1.0 / Math.max(0.003, e.lengthM);

      // Passive diffusion (from high C to low C)
      let rateAtoB = D_INTERNAL_CARBON * (na.carbonKg - nb.carbonKg) * invLen;

      // Bulk Hagen-Poiseuille advection (fluxM3s > 0 means a -> b)
      const q = e.fluxM3s;
      if (q > 0.0) {
        rateAtoB += q * (na.carbonKg / 1.0e-8) * 0.05;
      } else if (q < 0.0) {
        rateAtoB += q * (nb.carbonKg / 1.0e-8) * 0.05;
      }

      // Boswell active translocation (strictly local graph & substrate physics, zero radial hard-coding):
      // Older hyphae in substrate-depleted zones export internal carbon forward toward
      // younger hyphae foraging undepleted soil. Undepleted nodes do not drain each other.
      const csA = soil.solubleCarbon[na.voxelIdx];
      const csB = soil.solubleCarbon[nb.voxelIdx];
      const ageDiff = na.ageSec - nb.ageSec;

      if (ageDiff > 600.0 && csA < 0.28 && na.carbonKg > 2.0e-6) {
        // a is older and in depleted soil -> export forward toward younger node b
        const urgency = 1.0 + (0.28 - csA) / 0.14;
        rateAtoB += V_ACTIVE_TRANSLOC * urgency * (na.carbonKg - 1.5e-6);
      } else if (ageDiff < -600.0 && csB < 0.28 && nb.carbonKg > 2.0e-6) {
        // b is older and in depleted soil -> export forward toward younger node a
        const urgency = 1.0 + (0.28 - csB) / 0.14;
        rateAtoB -= V_ACTIVE_TRANSLOC * urgency * (nb.carbonKg - 1.5e-6);
      }

      // Fruiting sink pull (moderate pull so one primordium doesn't starve the entire frontier)
      if (nb.hasPrimordium && !na.hasPrimordium && na.carbonKg > nb.carbonKg * 0.85) {
        rateAtoB += V_ACTIVE_TRANSLOC * 0.35 * na.carbonKg;
      } else if (na.hasPrimordium && !nb.hasPrimordium && nb.carbonKg > na.carbonKg * 0.85) {
        rateAtoB -= V_ACTIVE_TRANSLOC * 0.35 * nb.carbonKg;
      }

      // Clamp transfer per edge so no node goes negative
      let transfer = rateAtoB * dtSec;
      const maxFromA = na.carbonKg * 0.22;
      const maxFromB = nb.carbonKg * 0.22;
      if (transfer > maxFromA) transfer = maxFromA;
      else if (transfer < -maxFromB) transfer = -maxFromB;

      cDelta[e.a] -= transfer;
      cDelta[e.b] += transfer;
    }

    for (let i = 0; i < n; i++) {
      const next = this.nodes[i].carbonKg + cDelta[i];
      this.nodes[i].carbonKg =
        next < 0.0 ? 0.0 : next > CARBON_NODE_CAP_KG ? CARBON_NODE_CAP_KG : next;
    }
  }

  // --------------------------------------------------------------------------
  // Step 3: Lockhart Chemotropic Tip Extension, Branching & Anastomosis
  // --------------------------------------------------------------------------

  private stepGrowthAndAnastomosis(dtSec: number): void {
    const soil = this.soil;
    const yieldTipPa = this.species.yieldPressurePa * 0.78;
    const currentCount = this.nodes.length;

    const halfX = soil.sizeX * 0.5 - 0.014;
    const halfZ = soil.sizeZ * 0.5 - 0.014;
    const minY = -soil.sizeY + 0.014;
    const maxY = -0.005;

    // 1. Advance active tips along Lockhart + chemotropic gradient
    for (let i = 0; i < currentCount; i++) {
      const tip = this.nodes[i];
      if (!tip.isTip) continue;

      const excessP = tip.turgorPa - yieldTipPa;
      if (excessP <= 0.0) continue;

      const v = tip.voxelIdx;
      const o2 = soil.o2[v];
      // Halt growth in hypoxic soil pockets
      if (o2 < 0.4) continue;

      const fO2 = o2 / (K_O2_TIP + o2);
      const fC = tip.carbonKg / (3.0e-6 + tip.carbonKg);
      const speed = PHI_TIP * excessP * fO2 * fC;
      const stepDist = Math.min(0.009, speed * dtSec);
      if (stepDist < 1.0e-6) continue;

      // Compute local 3D chemotropic gradient: grad(Cs) + w_O2 * grad(O2)
      const [ix, iy, iz] = soil.worldToVoxel(tip.x, tip.y, tip.z);
      const gradC = this.sampleGradient(soil.solubleCarbon, ix, iy, iz);
      const gradO = this.sampleGradient(soil.o2, ix, iy, iz);

      // Gentle upward horizon bias when deep, slight downward bias right under surface
      let horizonBiasY = 0.0;
      if (tip.y < -0.032) {
        horizonBiasY = 0.25;
      } else if (tip.y > -0.012) {
        horizonBiasY = -0.25;
      }

      const jitterX = (this.nextRand() - 0.5) * 0.22;
      const jitterY = (this.nextRand() - 0.5) * 0.08;
      const jitterZ = (this.nextRand() - 0.5) * 0.22;

      let nxDir = tip.dirX + (gradC[0] * 0.08 + gradO[0] * 0.03 + jitterX) * 0.45;
      let nyDir =
        tip.dirY + (gradC[1] * 0.06 + gradO[1] * 0.03 + horizonBiasY + jitterY) * 0.35;
      let nzDir = tip.dirZ + (gradC[2] * 0.08 + gradO[2] * 0.03 + jitterZ) * 0.45;

      const norm = Math.hypot(nxDir, nyDir, nzDir);
      if (norm > 1e-8) {
        nxDir /= norm;
        nyDir /= norm;
        nzDir /= norm;
      }

      let candX = tip.x + nxDir * stepDist;
      let candY = tip.y + nyDir * stepDist;
      let candZ = tip.z + nzDir * stepDist;

      // Wall reflections
      if (candX < -halfX) {
        candX = -halfX;
        nxDir = Math.abs(nxDir);
      } else if (candX > halfX) {
        candX = halfX;
        nxDir = -Math.abs(nxDir);
      }
      if (candZ < -halfZ) {
        candZ = -halfZ;
        nzDir = Math.abs(nzDir);
      } else if (candZ > halfZ) {
        candZ = halfZ;
        nzDir = -Math.abs(nzDir);
      }
      if (candY < minY) {
        candY = minY;
        nyDir = Math.abs(nyDir);
      } else if (candY > maxY) {
        candY = maxY;
        nyDir = -Math.abs(nyDir);
      }

      // Check STONE and severe hypoxia at candidate voxel
      const [cix, ciy, ciz] = soil.worldToVoxel(candX, candY, candZ);
      const candIdx = soil.voxelIndex(cix, ciy, ciz);
      if (soil.material[candIdx] === SoilMaterialId.STONE || soil.o2[candIdx] < 0.35) {
        // Reflect tangent around obstacle
        const tmpX = -nzDir;
        const tmpZ = nxDir;
        tip.dirX = tmpX;
        tip.dirY = nyDir * 0.5;
        tip.dirZ = tmpZ;
        continue;
      }

      // Move tip node and update spatial hash & parent edge length
      tip.x = candX;
      tip.y = candY;
      tip.z = candZ;
      tip.dirX = nxDir;
      tip.dirY = nyDir;
      tip.dirZ = nzDir;
      tip.voxelIdx = candIdx;
      this.updateSpatialHash(i);

      // Update connected edge length & conductance
      const parentId = this.tipParentId[i];
      let parentDist = 0.0;
      const incident = this.adj[i];
      for (let k = 0; k < incident.length; k++) {
        const e = this.edges[incident[k]];
        const other = this.nodes[e.a === i ? e.b : e.a];
        const len = Math.hypot(tip.x - other.x, tip.y - other.y, tip.z - other.z);
        e.lengthM = Math.max(0.002, len);
        e.conductance = this.computeConductance(e.radiusM, e.lengthM);
        if (other.id === parentId) {
          parentDist = len;
        }
      }

      // Spend structural chitin wall synthesis carbon proportional to tip extension
      const wallCost = stepDist * 1.5e-5;
      tip.carbonKg = Math.max(0.0, tip.carbonKg - Math.min(tip.carbonKg * 0.15, wallCost));

      // Check ANASTOMOSIS (fusion with non-adjacent node at graph distance >= 3)
      const fuseTarget = this.findAnastomosisTarget(i);
      if (fuseTarget >= 0 && this.edges.length < MAX_EDGES) {
        this.createEdge(i, fuseTarget, INIT_CORD_RADIUS_M * 1.1, true);
        this.anastomosisCount++;
        tip.isTip = false;
        continue;
      }

      // Split into new structural segment when tip exceeds SEGMENT_LENGTH_M
      if (parentDist >= SEGMENT_LENGTH_M && this.nodes.length < MAX_NODES) {
        // Convert current tip into an interior node and spawn continuing tip just ahead
        tip.isTip = false;
        const shareC = tip.carbonKg * 0.46;
        tip.carbonKg -= shareC;

        const nextX = Math.max(-halfX, Math.min(halfX, tip.x + nxDir * 0.002));
        const nextY = Math.max(minY, Math.min(maxY, tip.y + nyDir * 0.002));
        const nextZ = Math.max(-halfZ, Math.min(halfZ, tip.z + nzDir * 0.002));
        const newTipId = this.createNode(
          nextX,
          nextY,
          nextZ,
          shareC,
          true,
          nxDir,
          nyDir,
          nzDir,
          i,
        );
        this.nodes[newTipId].turgorPa = tip.turgorPa;
        this.createEdge(i, newTipId, INIT_CORD_RADIUS_M, false);

        // Occasional apical bifurcation / lateral branch at 45-70 deg when well-fed
        if (
          this.nodes.length < MAX_NODES &&
          tip.turgorPa > 2.8e5 &&
          tip.carbonKg > 3.2e-6 &&
          this.adj[i].length < 4 &&
          this.nextRand() < 0.56
        ) {
          this.spawnLateralBranch(i, nxDir, nyDir, nzDir);
        }
      }
    }

    // 2. Sub-apical lateral branching from carbon-rich interior nodes
    if (this.nodes.length < MAX_NODES - 8) {
      const branchProb = Math.min(0.14, dtSec / 3.8e4);
      for (let i = 0; i < currentCount; i++) {
        if (this.nodes.length >= MAX_NODES) break;
        const node = this.nodes[i];
        if (node.isTip) continue;
        if (this.adj[i].length >= 4) continue;
        if (node.ageSec < 12000.0) continue;
        if (node.turgorPa < 2.9e5 || node.carbonKg < 4.5e-6) continue;
        if (soil.o2[node.voxelIdx] < 1.5) continue;

        if (this.nextRand() < branchProb) {
          this.spawnLateralBranch(i, node.dirX, node.dirY, node.dirZ);
        }
      }
    }
  }

  private spawnLateralBranch(
    parentId: number,
    baseDx: number,
    baseDy: number,
    baseDz: number,
  ): void {
    if (this.nodes.length >= MAX_NODES || this.edges.length >= MAX_EDGES) return;
    const parent = this.nodes[parentId];
    // Rotate horizontally by 48..68 degrees left or right
    const sign = this.nextRand() < 0.5 ? 1.0 : -1.0;
    const theta = sign * (0.84 + this.nextRand() * 0.35);
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);

    let bx = baseDx * cosT - baseDz * sinT;
    let bz = baseDx * sinT + baseDz * cosT;
    let by = baseDy * 0.5 + (this.nextRand() - 0.45) * 0.22;
    const norm = Math.hypot(bx, by, bz);
    if (norm < 1e-6) {
      bx = 1.0;
      by = 0.0;
      bz = 0.0;
    } else {
      bx /= norm;
      by /= norm;
      bz /= norm;
    }

    const pos = this.clampToValidSoil(
      parent.x + bx * 0.004,
      parent.y + by * 0.004,
      parent.z + bz * 0.004,
    );
    const shareC = parent.carbonKg * 0.32;
    parent.carbonKg -= shareC;
    const branchId = this.createNode(pos[0], pos[1], pos[2], shareC, true, bx, by, bz, parentId);
    this.nodes[branchId].turgorPa = parent.turgorPa;
    this.createEdge(parentId, branchId, INIT_CORD_RADIUS_M * 0.9, false);
  }

  private findAnastomosisTarget(tipId: number): number {
    const tip = this.nodes[tipId];
    const cx = Math.floor(tip.x / HASH_CELL_M);
    const cy = Math.floor(tip.y / HASH_CELL_M);
    const cz = Math.floor(tip.z / HASH_CELL_M);
    const r2 = ANASTOMOSIS_RADIUS_M * ANASTOMOSIS_RADIUS_M;

    let bestId = -1;
    let bestDist2 = r2;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dz = -1; dz <= 1; dz++) {
        for (let dx = -1; dx <= 1; dx++) {
          const key = this.packHashKey(cx + dx, cy + dy, cz + dz);
          const bucket = this.spatialHash.get(key);
          if (!bucket) continue;
          for (let b = 0; b < bucket.length; b++) {
            const candId = bucket[b];
            if (candId === tipId) continue;
            const cand = this.nodes[candId];
            const rx = cand.x - tip.x;
            const ry = cand.y - tip.y;
            const rz = cand.z - tip.z;
            const d2 = rx * rx + ry * ry + rz * rz;
            if (d2 < bestDist2 && d2 > 1e-8) {
              // Avoid two young parallel frontier tips prematurely terminating each other;
              // fuse when approaching an established cord or a crossing/converging branch.
              const dirDot =
                tip.dirX * cand.dirX + tip.dirY * cand.dirY + tip.dirZ * cand.dirZ;
              if (cand.isTip && cand.ageSec < 18000.0 && dirDot > 0.25) continue;
              // Enforce topological graph distance >= 3 (no self-loops or triangles)
              if (this.isAtLeastThreeHopsAway(tipId, candId)) {
                bestDist2 = d2;
                bestId = candId;
              }
            }
          }
        }
      }
    }

    return bestId;
  }

  private isAtLeastThreeHopsAway(a: number, b: number): boolean {
    if (a === b) return false;
    const adjA = this.adj[a];
    for (let i = 0; i < adjA.length; i++) {
      const e1 = this.edges[adjA[i]];
      const n1 = e1.a === a ? e1.b : e1.a;
      if (n1 === b) return false; // distance 1
      const adj1 = this.adj[n1];
      for (let j = 0; j < adj1.length; j++) {
        const e2 = this.edges[adj1[j]];
        const n2 = e2.a === n1 ? e2.b : e2.a;
        if (n2 === b) return false; // distance 2
      }
    }
    return true;
  }

  // --------------------------------------------------------------------------
  // Step 4: Five-Condition Primordium Pinning Trigger (Zero Radial Hard-Coding)
  // --------------------------------------------------------------------------

  private stepPinning(env: EnvironmentState, species: SpeciesPreset): void {
    if (this.primordia.length >= MAX_PRIMORDIA) return;
    if (env.ambientHumidity < species.pinHumidityMin) return;

    const soil = this.soil;
    const inhibR2 = species.pinInhibitionRadiusM * species.pinInhibitionRadiusM;
    const n = this.nodes.length;

    // Collect eligible near-surface nodes and prioritize highest carbon + turgor
    let bestNodeId = -1;
    let bestScore = -1.0;

    for (let i = 0; i < n; i++) {
      const node = this.nodes[i];
      if (node.hasPrimordium) continue;
      // 1. Near-surface depth window
      if (node.y < -0.028) continue;
      // 2. Carbon reserve threshold
      if (node.carbonKg < species.pinCarbonThresholdKg) continue;
      // 3. Turgor pressure threshold
      if (node.turgorPa < species.pinTurgorThresholdPa) continue;
      // 4. Soil temperature window
      const localT = soil.temperature[node.voxelIdx];
      if (localT < species.pinTempMinC || localT > species.pinTempMaxC) continue;
      // Also require adequate O2 for aerobic fruiting differentiation
      if (soil.o2[node.voxelIdx] < 2.0) continue;

      // 5. Lateral inhibition spacing from existing primordia
      let inhibited = false;
      for (let p = 0; p < this.primordia.length; p++) {
        const prim = this.primordia[p];
        const dx = node.x - prim.x;
        const dz = node.z - prim.z;
        if (dx * dx + dz * dz < inhibR2) {
          inhibited = true;
          break;
        }
      }
      if (inhibited) continue;

      const score = node.carbonKg * (node.turgorPa / 4.0e5);
      if (score > bestScore) {
        bestScore = score;
        bestNodeId = i;
      }
    }

    if (bestNodeId >= 0) {
      const node = this.nodes[bestNodeId];
      node.hasPrimordium = true;
      this.primordia.push({
        id: this.nextPrimordiumId++,
        nodeId: bestNodeId,
        x: node.x,
        y: 0.0, // Emerges at soil-air interface y = 0 directly above near-surface knot
        z: node.z,
        initiatedAtSec: this.simTimeSec,
        carbonReserveKg: node.carbonKg,
      });
    }
  }

  // --------------------------------------------------------------------------
  // Graph & Spatial Utilities
  // --------------------------------------------------------------------------

  private createNode(
    x: number,
    y: number,
    z: number,
    carbonKg: number,
    isTip: boolean,
    dirX: number,
    dirY: number,
    dirZ: number,
    parentId: number,
  ): number {
    const id = this.nodes.length;
    const [ix, iy, iz] = this.soil.worldToVoxel(x, y, z);
    const voxelIdx = this.soil.voxelIndex(ix, iy, iz);
    const o2 = this.soil.o2[voxelIdx];
    const psiSoil = this.soil.psi[voxelIdx];
    const gO2 = (o2 * o2) / (K_O2_OSM * K_O2_OSM + o2 * o2);
    const pi = Math.min(PI_MAX_PA, PI_BASE_PA + PI_CARBON_GAIN * carbonKg) * gO2;
    const initP = Math.max(0.0, (psiSoil + pi) * 0.78);

    this.nodes.push({
      id,
      x,
      y,
      z,
      voxelIdx,
      turgorPa: initP,
      osmoticPa: pi,
      carbonKg,
      isTip,
      dirX,
      dirY,
      dirZ,
      ageSec: 0.0,
      hasPrimordium: false,
      fruitingBodyId: -1,
    });
    this.adj.push([]);
    this.tipParentId.push(parentId);
    this.nodeHashKey.push(-1);
    this.updateSpatialHash(id);
    return id;
  }

  private createEdge(
    a: number,
    b: number,
    radiusM: number,
    isAnastomosis: boolean,
  ): number {
    const id = this.edges.length;
    const na = this.nodes[a];
    const nb = this.nodes[b];
    const lengthM = Math.max(
      0.002,
      Math.hypot(nb.x - na.x, nb.y - na.y, nb.z - na.z),
    );
    const conductance = this.computeConductance(radiusM, lengthM);
    this.edges.push({
      id,
      a,
      b,
      lengthM,
      radiusM,
      conductance,
      fluxM3s: 0.0,
      isAnastomosis,
    });
    this.adj[a].push(id);
    this.adj[b].push(id);
    return id;
  }

  private computeConductance(radiusM: number, lengthM: number): number {
    const r2 = radiusM * radiusM;
    return (Math.PI * r2 * r2) / (8.0 * ETA_CYTOPLASM * Math.max(0.002, lengthM));
  }

  private packHashKey(cx: number, cy: number, cz: number): number {
    return ((cx + 128) << 16) | ((cy + 128) << 8) | (cz + 128);
  }

  private updateSpatialHash(nodeId: number): void {
    const node = this.nodes[nodeId];
    const newKey = this.packHashKey(
      Math.floor(node.x / HASH_CELL_M),
      Math.floor(node.y / HASH_CELL_M),
      Math.floor(node.z / HASH_CELL_M),
    );
    const oldKey = this.nodeHashKey[nodeId];
    if (oldKey === newKey) return;

    if (oldKey !== -1) {
      const oldBucket = this.spatialHash.get(oldKey);
      if (oldBucket) {
        const pos = oldBucket.indexOf(nodeId);
        if (pos >= 0) {
          oldBucket[pos] = oldBucket[oldBucket.length - 1];
          oldBucket.pop();
        }
      }
    }

    let bucket = this.spatialHash.get(newKey);
    if (!bucket) {
      bucket = [];
      this.spatialHash.set(newKey, bucket);
    }
    bucket.push(nodeId);
    this.nodeHashKey[nodeId] = newKey;
  }

  private sampleGradient(
    field: Float32Array,
    ix: number,
    iy: number,
    iz: number,
  ): [number, number, number] {
    const { nx, ny, nz, dx } = this.soil;
    const inv2Dx = 0.5 / dx;
    const xm = ix > 0 ? ix - 1 : ix;
    const xp = ix + 1 < nx ? ix + 1 : ix;
    const ym = iy > 0 ? iy - 1 : iy;
    const yp = iy + 1 < ny ? iy + 1 : iy;
    const zm = iz > 0 ? iz - 1 : iz;
    const zp = iz + 1 < nz ? iz + 1 : iz;

    const gx =
      (field[this.soil.voxelIndex(xp, iy, iz)] - field[this.soil.voxelIndex(xm, iy, iz)]) *
      inv2Dx;
    const gy =
      (field[this.soil.voxelIndex(ix, yp, iz)] - field[this.soil.voxelIndex(ix, ym, iz)]) *
      inv2Dx;
    const gz =
      (field[this.soil.voxelIndex(ix, iy, zp)] - field[this.soil.voxelIndex(ix, iy, zm)]) *
      inv2Dx;
    return [gx, gy, gz];
  }

  private clampToValidSoil(x: number, y: number, z: number): [number, number, number] {
    const halfX = this.soil.sizeX * 0.5 - 0.012;
    const halfZ = this.soil.sizeZ * 0.5 - 0.012;
    const cx = Math.max(-halfX, Math.min(halfX, x));
    const cy = Math.max(-this.soil.sizeY + 0.012, Math.min(-0.005, y));
    const cz = Math.max(-halfZ, Math.min(halfZ, z));

    const [ix, iy, iz] = this.soil.worldToVoxel(cx, cy, cz);
    if (this.soil.material[this.soil.voxelIndex(ix, iy, iz)] !== SoilMaterialId.STONE) {
      return [cx, cy, cz];
    }

    // Nudge upward out of STONE if inoculum was placed inside a boulder
    for (let tryIy = iy + 1; tryIy < this.soil.ny; tryIy++) {
      if (this.soil.material[this.soil.voxelIndex(ix, tryIy, iz)] !== SoilMaterialId.STONE) {
        return this.soil.voxelToWorld(ix, tryIy, iz);
      }
    }
    return [cx, cy, cz];
  }

  private nextRand(): number {
    let t = (this.rngState += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  }
}
