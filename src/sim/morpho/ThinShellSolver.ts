// ============================================================================
// SPOROCARP — Layer 4: Discrete Koiter / Föppl-von Kármán Growing Thin-Shell
// Genuine energy-minimizing elastic thin shell for mushroom caps, radial gill
// buckling + secondary lamellulae, chanterelle hyperbolic ruffles, morel
// elastic-core honeycomb pits, solid obstacle contact, and partial veil tear.
// ============================================================================

import {
  MeshPartId,
  type ShellMeshBuffers,
  type SolidObstacle,
  type SpeciesPreset,
  type ThinShellParams,
} from '../../types/contract.ts';

/** Effective membrane-coupling Winkler coefficient c_s (dimensionless) */
const MEMBRANE_FOUNDATION_COEFF = 0.5;
/** Per-iteration deterministic symmetry-breaking kick amplitude (m) */
const RING_NOISE_AMP_M = 2.0e-7;

interface StretchEdge {
  a: number;
  b: number;
  restLen: number;
  areaWeight: number; // dual strip area A_e for continuum stretching energy
  cos2Phi: number; // radial projection squared
  sin2Phi: number; // circumferential projection squared
  isHymenium: boolean;
}

interface BendHinge {
  v0: number;
  v1: number;
  v2: number;
  v3: number;
  targetAngle: number;
  geomFactor: number; // l_e^2 / (A_1 + A_2)
  isHymenium: boolean;
}

interface VeilEdge {
  a: number;
  b: number;
  restLen: number;
  heterogeneity: number; // +-8% physical material heterogeneity
  isTearBand: boolean;
  broken: boolean;
}

interface VeilTri {
  v0: number;
  v1: number;
  v2: number;
  e0: number;
  e1: number;
  e2: number;
}

/** Deterministic zero-mean broadband hash in [-1, 1] for tiny symmetry-breaking noise */
function deterministicNoise(seed: number): number {
  let x = (seed | 0) ^ 0x9e3779b9;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
  x = x ^ (x >>> 16);
  return ((x >>> 0) / 0xffffffff) * 2.0 - 1.0;
}

export class ThinShellCapSolver {
  public readonly species: SpeciesPreset;
  public readonly radialRings: number;
  public readonly angularSectors: number;
  public readonly shellThicknessM: number;
  public readonly youngsModulusPa: number;
  public readonly poissonRatio: number;
  public readonly alphaRadial: number;
  public readonly alphaCircumferential: number;
  public readonly coreFoundationStiffness: number;

  // Cap top shell vertex layout:
  //   0 = apex
  //   1 + (r - 1) * Ntheta + s  for r in [1..Nr], s in [0..Ntheta-1]
  private readonly capVertCount: number;
  private readonly hymRings: number;
  private readonly hymVertStart: number;
  private readonly hymVertCount: number;
  private readonly veilRings: number;
  private readonly veilVertStart: number;
  private readonly veilVertCount: number;
  private readonly totalVertCount: number;

  // Dynamic mechanical state (double precision for accurate gradient relaxation)
  private readonly pos: Float64Array;
  private readonly vel: Float64Array;
  private readonly force: Float64Array;
  private readonly stretchForce: Float64Array;
  private readonly diagTan: Float64Array;
  private readonly diagNorm: Float64Array;
  private readonly refNorm: Float64Array; // outward unit normal of reference shell
  private readonly refPos: Float64Array; // elastic foundation reference surface

  // Topology
  private readonly edges: StretchEdge[] = [];
  private readonly hinges: BendHinge[] = [];
  private readonly capTris: Uint32Array;
  private readonly hymTris: Uint32Array;
  private readonly veilEdges: VeilEdge[] = [];
  private readonly veilTris: VeilTri[] = [];
  private readonly veilVertRing: Uint8Array;

  // Growth & biophysical state
  private growthStrain = 0; // cumulative cap growth integral
  private noiseTick = 0;
  private hoopOverGrowth = 0; // mean (restLen / refLen - 1) over circumferential edges
  private readonly edgeLookup = new Map<number, number>();
  private radialScale = 1.0; // current radial expansion factor relative to initial button
  private initialRadiusM: number;
  private currentCapRadiusM: number;
  private wiltFactor = 0;
  private cachedWavenumber = 0;
  private cachedLamellulaeCount = 0;
  private cachedGillSpacingM: number;

  // Precomputed trigonometric tables for exact O(N^2) circulant linear solve of
  // the implicit 5-point discrete bending + foundation backward-Euler matrix
  private readonly cosTable: Float64Array;
  private readonly sinTable: Float64Array;
  private readonly ringRhs: Float64Array;
  private readonly ringSol: Float64Array;
  private readonly ringRhsDev: Float64Array;
  private readonly ringNormalMask: Uint8Array;
  private readonly ringOut: Float64Array;
  private readonly ringSpecRe: Float64Array;
  private readonly ringSpecIm: Float64Array;

  // Output render buffers (reused across frames)
  private readonly outPositions: Float32Array;
  private readonly outNormals: Float32Array;
  private readonly outStressVM: Float32Array;
  private readonly outPartIds: Uint8Array;
  private readonly maxIndicesBuf: Uint32Array;
  private outIndices: Uint32Array;
  private readonly seedOffset: number;

  constructor(params: ThinShellParams, seedOffset = 1) {
    this.species = params.species;
    this.radialRings = Math.max(5, params.radialRings ?? 12);
    this.angularSectors = Math.max(24, params.angularSectors ?? 64);
    this.shellThicknessM = params.shellThicknessM ?? params.species.shellThicknessM;
    this.youngsModulusPa = params.youngsModulusPa ?? params.species.youngsModulusPa;
    this.poissonRatio = params.species.poissonRatio;
    this.alphaRadial = params.alphaRadial ?? params.species.alphaRadial;
    this.alphaCircumferential =
      params.alphaCircumferential ?? params.species.alphaCircumferential;
    this.coreFoundationStiffness =
      params.coreFoundationStiffness ?? params.species.coreFoundationStiffness;
    this.seedOffset = seedOffset;

    const Nr = this.radialRings;
    const Nt = this.angularSectors;

    this.capVertCount = 1 + Nr * Nt;

    // Underside hymenium membrane for gilled & false-fold species
    const hasSeparateHymenium = this.species.hymeniumType !== 'honeycomb';
    this.hymRings = hasSeparateHymenium ? Math.max(4, Math.min(6, Math.floor(Nr / 2))) : 0;
    this.hymVertStart = this.capVertCount;
    this.hymVertCount = hasSeparateHymenium ? this.hymRings * Nt : 0;

    // Partial veil mesh for veiled species (parasol, inkcap)
    this.veilRings = this.species.hasPartialVeil ? 3 : 0;
    this.veilVertStart = this.hymVertStart + this.hymVertCount;
    this.veilVertCount = this.species.hasPartialVeil ? this.veilRings * Nt : 0;

    this.totalVertCount = this.capVertCount + this.hymVertCount + this.veilVertCount;

    this.pos = new Float64Array(this.totalVertCount * 3);
    this.vel = new Float64Array(this.totalVertCount * 3);
    this.force = new Float64Array(this.totalVertCount * 3);
    this.stretchForce = new Float64Array(this.totalVertCount * 3);
    this.diagTan = new Float64Array(this.totalVertCount);
    this.diagNorm = new Float64Array(this.totalVertCount);
    this.refNorm = new Float64Array(this.totalVertCount * 3);
    this.refPos = new Float64Array(this.totalVertCount * 3);
    this.veilVertRing = new Uint8Array(this.veilVertCount);

    this.outPositions = new Float32Array(this.totalVertCount * 3);
    this.outNormals = new Float32Array(this.totalVertCount * 3);
    this.outStressVM = new Float32Array(this.totalVertCount);
    this.outPartIds = new Uint8Array(this.totalVertCount);

    // Precompute DFT basis tables for exact circulant linear solver of implicit bending operator
    const halfN = Math.floor(Nt / 2) + 1;
    this.cosTable = new Float64Array(halfN * Nt);
    this.sinTable = new Float64Array(halfN * Nt);
    for (let k = 0; k < halfN; k++) {
      const freq = (2.0 * Math.PI * k) / Nt;
      const row = k * Nt;
      for (let s = 0; s < Nt; s++) {
        this.cosTable[row + s] = Math.cos(freq * s);
        this.sinTable[row + s] = Math.sin(freq * s);
      }
    }
    this.ringRhs = new Float64Array(Nt);
    this.ringSol = new Float64Array(Nt);
    this.ringRhsDev = new Float64Array(Nt);
    this.ringNormalMask = new Uint8Array(this.totalVertCount);
    this.ringOut = new Float64Array(Nt);
    this.ringSpecRe = new Float64Array(halfN);
    this.ringSpecIm = new Float64Array(halfN);

    this.initialRadiusM = Math.max(0.0045, this.species.maxCapRadiusM * 0.14);
    this.currentCapRadiusM = this.initialRadiusM;
    this.cachedGillSpacingM = this.computeBaselineGillSpacingM();

    // Allocate triangle arrays
    const capTriCount = Nt + (Nr - 1) * Nt * 2;
    this.capTris = new Uint32Array(capTriCount * 3);

    const hymTriCount = hasSeparateHymenium ? (this.hymRings - 1) * Nt * 2 : 0;
    this.hymTris = new Uint32Array(hymTriCount * 3);

    const maxVeilTris = this.species.hasPartialVeil ? (this.veilRings - 1) * Nt * 2 : 0;
    this.maxIndicesBuf = new Uint32Array((capTriCount + hymTriCount + maxVeilTris) * 3);
    this.outIndices = this.maxIndicesBuf;

    this.buildInitialGeometry();
    this.buildTopology();
    this.markBucklingRings();
    this.syncOutputBuffers();
  }

  private capIdx(ring: number, sector: number): number {
    if (ring === 0) return 0;
    const s = ((sector % this.angularSectors) + this.angularSectors) % this.angularSectors;
    return 1 + (ring - 1) * this.angularSectors + s;
  }

  private hymIdx(ring: number, sector: number): number {
    const s = ((sector % this.angularSectors) + this.angularSectors) % this.angularSectors;
    return this.hymVertStart + ring * this.angularSectors + s;
  }

  private veilIdx(ring: number, sector: number): number {
    const s = ((sector % this.angularSectors) + this.angularSectors) % this.angularSectors;
    return this.veilVertStart + ring * this.angularSectors + s;
  }

  /** Computes the axisymmetric reference cap profile (r, y) for normalized radial coordinate u in [0, 1] */
  private referenceProfile(u: number, radiusM: number): [number, number] {
    const hType = this.species.hymeniumType;
    if (hType === 'honeycomb') {
      // Morel: elongated hollow ovoid head (height ~ 2.1 * R)
      const phi = u * Math.PI * 0.88; // apex (0) down to lower collar (~158 deg)
      const r = radiusM * Math.sin(phi) * (0.88 + 0.15 * Math.sin(phi));
      const height = radiusM * 2.15;
      const y = height * 0.52 * (Math.cos(phi) + 0.82);
      return [r, y];
    }
    if (hType === 'false-folds') {
      // Chanterelle: funnel / vase shape (negative target curvature)
      const r = radiusM * (0.14 + 0.86 * Math.pow(u, 0.85));
      const depth = radiusM * 0.55;
      // Central depression rising out to flared rim, drooping slightly when wilted
      const y = depth * (0.15 + 0.95 * Math.pow(u, 1.35)) - this.wiltFactor * radiusM * 0.35 * u * u;
      return [r, y];
    }
    if (hType === 'dense-gills') {
      // Ink cap: tall campanulate / bullet-bell cap
      const phi = u * Math.PI * 0.62;
      const r = radiusM * Math.pow(Math.sin(phi), 0.85);
      const bellHeight = radiusM * 1.85;
      const y = bellHeight * Math.cos(phi * 0.92) + radiusM * 0.12 - this.wiltFactor * radiusM * 0.25 * u * u;
      return [r, y];
    }
    // Parasol: classic convex dome flattening slightly with maturity, gentle umbo at center
    const maturity = Math.min(1, (radiusM - this.initialRadiusM) / Math.max(1e-4, this.species.maxCapRadiusM - this.initialRadiusM));
    const domeAspect = 0.62 - 0.24 * maturity; // taller button -> wider umbrella
    const r = radiusM * u;
    const umbo = 0.08 * radiusM * Math.exp(-u * u * 16.0);
    const domeDrop = domeAspect * radiusM * (1.0 - Math.cos(u * Math.PI * 0.52));
    const wiltCurl = this.wiltFactor * radiusM * 0.42 * u * u * u;
    const y = domeAspect * radiusM * 0.85 + umbo - domeDrop - wiltCurl;
    return [r, y];
  }

  private buildInitialGeometry(): void {
    const Nr = this.radialRings;
    const Nt = this.angularSectors;
    const R0 = this.initialRadiusM;

    // 1. Cap top shell
    const [, y0] = this.referenceProfile(0, R0);
    this.pos[0] = 0;
    this.pos[1] = y0;
    this.pos[2] = 0;
    this.refPos[0] = 0;
    this.refPos[1] = y0;
    this.refPos[2] = 0;
    this.outPartIds[0] = MeshPartId.CAP_TOP;

    // Tiny deterministic symmetry-breaking noise (~1e-5 m), broadband across all vertices
    const noiseAmp = 1.2e-5;

    for (let r = 1; r <= Nr; r++) {
      const u = r / Nr;
      const [rad, yRef] = this.referenceProfile(u, R0);
      for (let s = 0; s < Nt; s++) {
        const theta = (2.0 * Math.PI * s) / Nt;
        const ct = Math.cos(theta);
        const st = Math.sin(theta);
        const v = this.capIdx(r, s);
        const nSeed = this.seedOffset * 7919 + v * 104729;
        const nr = deterministicNoise(nSeed) * noiseAmp;
        const ny = deterministicNoise(nSeed + 31) * noiseAmp;
        const rx = rad * ct;
        const rz = rad * st;
        this.refPos[v * 3] = rx;
        this.refPos[v * 3 + 1] = yRef;
        this.refPos[v * 3 + 2] = rz;
        this.pos[v * 3] = rx + nr * ct;
        this.pos[v * 3 + 1] = yRef + ny;
        this.pos[v * 3 + 2] = rz + nr * st;
        this.outPartIds[v] = MeshPartId.CAP_TOP;
      }
    }

    // 2. Underside hymenium membrane
    if (this.hymRings > 0) {
      const gillOffset = Math.min(this.species.gillDepthM * 0.25, R0 * 0.22);
      for (let hr = 0; hr < this.hymRings; hr++) {
        const u = 0.22 + (0.76 * hr) / (this.hymRings - 1);
        const [rad, yCap] = this.referenceProfile(u, R0);
        // Arch slightly below cap ceiling
        const drop = gillOffset * Math.sin(Math.PI * (0.15 + 0.7 * (hr / (this.hymRings - 1))));
        const yRef = yCap - drop;
        for (let s = 0; s < Nt; s++) {
          const theta = (2.0 * Math.PI * s) / Nt;
          const ct = Math.cos(theta);
          const st = Math.sin(theta);
          const v = this.hymIdx(hr, s);
          const nSeed = this.seedOffset * 13331 + v * 65537;
          const ny = deterministicNoise(nSeed) * noiseAmp;
          const rx = rad * ct;
          const rz = rad * st;
          this.refPos[v * 3] = rx;
          this.refPos[v * 3 + 1] = yRef;
          this.refPos[v * 3 + 2] = rz;
          this.pos[v * 3] = rx;
          this.pos[v * 3 + 1] = yRef + ny;
          this.pos[v * 3 + 2] = rz;
          this.outPartIds[v] = MeshPartId.GILLS;
        }
      }
    }

    // 3. Partial veil ring mesh (connects outer cap margin down to mid-stipe collar)
    if (this.veilRings > 0) {
      const [rimR, rimY] = this.referenceProfile(1.0, R0);
      const collarR = Math.max(0.0022, this.species.stipeBaseRadiusM * 0.62);
      const collarY = rimY - Math.max(0.004, R0 * 0.9);
      for (let vr = 0; vr < this.veilRings; vr++) {
        const frac = vr / (this.veilRings - 1); // 0 = cap rim, 1 = mid, 2 = stipe collar
        const r = rimR * (1 - frac) + collarR * frac;
        const y = rimY * (1 - frac) + collarY * frac;
        for (let s = 0; s < Nt; s++) {
          const theta = (2.0 * Math.PI * s) / Nt;
          const v = this.veilIdx(vr, s);
          const lv = v - this.veilVertStart;
          this.veilVertRing[lv] = vr;
          this.pos[v * 3] = r * Math.cos(theta);
          this.pos[v * 3 + 1] = y;
          this.pos[v * 3 + 2] = r * Math.sin(theta);
          this.refPos[v * 3] = this.pos[v * 3];
          this.refPos[v * 3 + 1] = this.pos[v * 3 + 1];
          this.refPos[v * 3 + 2] = this.pos[v * 3 + 2];
          this.outPartIds[v] = MeshPartId.VEIL;
        }
      }
    }
  }

  /** Marks vertices whose normal DOF is advanced by the implicit Fourier ring solver */
  private markBucklingRings(): void {
    const Nr = this.radialRings;
    const Nt = this.angularSectors;
    const type = this.species.hymeniumType;
    if (type === 'honeycomb') {
      for (let r = 2; r < Nr; r++) for (let s = 0; s < Nt; s++) this.ringNormalMask[this.capIdx(r, s)] = 1;
    } else if (type === 'false-folds') {
      for (let r = Math.max(2, Math.floor(Nr * 0.5)); r <= Nr; r++) {
        for (let s = 0; s < Nt; s++) this.ringNormalMask[this.capIdx(r, s)] = 1;
      }
    }
    for (let hr = 0; hr < this.hymRings; hr++) {
      for (let s = 0; s < Nt; s++) this.ringNormalMask[this.hymIdx(hr, s)] = 1;
    }
  }

  private edgeLength(a: number, b: number): number {
    const dx = this.pos[b * 3] - this.pos[a * 3];
    const dy = this.pos[b * 3 + 1] - this.pos[a * 3 + 1];
    const dz = this.pos[b * 3 + 2] - this.pos[a * 3 + 2];
    return Math.hypot(dx, dy, dz);
  }

  private triArea(i0: number, i1: number, i2: number): number {
    const ax = this.pos[i1 * 3] - this.pos[i0 * 3];
    const ay = this.pos[i1 * 3 + 1] - this.pos[i0 * 3 + 1];
    const az = this.pos[i1 * 3 + 2] - this.pos[i0 * 3 + 2];
    const bx = this.pos[i2 * 3] - this.pos[i0 * 3];
    const by = this.pos[i2 * 3 + 1] - this.pos[i0 * 3 + 1];
    const bz = this.pos[i2 * 3 + 2] - this.pos[i0 * 3 + 2];
    const cx = ay * bz - az * by;
    const cy = az * bx - ax * bz;
    const cz = ax * by - ay * bx;
    return 0.5 * Math.hypot(cx, cy, cz);
  }

  private computeDihedral(v0: number, v1: number, v2: number, v3: number): number {
    const ex = this.pos[v1 * 3] - this.pos[v0 * 3];
    const ey = this.pos[v1 * 3 + 1] - this.pos[v0 * 3 + 1];
    const ez = this.pos[v1 * 3 + 2] - this.pos[v0 * 3 + 2];
    const elen = Math.hypot(ex, ey, ez);
    if (elen < 1e-12) return 0;

    const e1x = this.pos[v2 * 3] - this.pos[v0 * 3];
    const e1y = this.pos[v2 * 3 + 1] - this.pos[v0 * 3 + 1];
    const e1z = this.pos[v2 * 3 + 2] - this.pos[v0 * 3 + 2];
    let n1x = ey * e1z - ez * e1y;
    let n1y = ez * e1x - ex * e1z;
    let n1z = ex * e1y - ey * e1x;
    const l1 = Math.hypot(n1x, n1y, n1z);
    if (l1 < 1e-16) return 0;
    n1x /= l1;
    n1y /= l1;
    n1z /= l1;

    const e2x = this.pos[v3 * 3] - this.pos[v0 * 3];
    const e2y = this.pos[v3 * 3 + 1] - this.pos[v0 * 3 + 1];
    const e2z = this.pos[v3 * 3 + 2] - this.pos[v0 * 3 + 2];
    let n2x = e2y * ez - e2z * ey;
    let n2y = e2z * ex - e2x * ez;
    let n2z = e2x * ey - e2y * ex;
    const l2 = Math.hypot(n2x, n2y, n2z);
    if (l2 < 1e-16) return 0;
    n2x /= l2;
    n2y /= l2;
    n2z /= l2;

    const dot = Math.max(-1, Math.min(1, n1x * n2x + n1y * n2y + n1z * n2z));
    const cx = n1y * n2z - n1z * n2y;
    const cy = n1z * n2x - n1x * n2z;
    const cz = n1x * n2y - n1y * n2x;
    const sinVal = (cx * ex + cy * ey + cz * ez) / elen;
    return Math.atan2(sinVal, dot);
  }

  private buildTopology(): void {
    const Nr = this.radialRings;
    const Nt = this.angularSectors;

    // ------------------------------------------------------------------
    // 1. Cap triangles + stretch edges + bending hinges
    // ------------------------------------------------------------------
    let tPtr = 0;
    // Apex fan (ring 0 -> ring 1)
    for (let s = 0; s < Nt; s++) {
      const v0 = 0;
      const v1 = this.capIdx(1, s);
      const v2 = this.capIdx(1, s + 1);
      // Outward/upward normal orientation
      this.capTris[tPtr++] = v0;
      this.capTris[tPtr++] = v2;
      this.capTris[tPtr++] = v1;
    }

    // Quad rings (alternating diagonal per ring to prevent chiral bias)
    for (let r = 1; r < Nr; r++) {
      const flip = (r & 1) === 1;
      for (let s = 0; s < Nt; s++) {
        const a = this.capIdx(r, s);
        const b = this.capIdx(r, s + 1);
        const c = this.capIdx(r + 1, s);
        const d = this.capIdx(r + 1, s + 1);
        if (!flip) {
          this.capTris[tPtr++] = a;
          this.capTris[tPtr++] = b;
          this.capTris[tPtr++] = d;
          this.capTris[tPtr++] = a;
          this.capTris[tPtr++] = d;
          this.capTris[tPtr++] = c;
        } else {
          this.capTris[tPtr++] = a;
          this.capTris[tPtr++] = b;
          this.capTris[tPtr++] = c;
          this.capTris[tPtr++] = b;
          this.capTris[tPtr++] = d;
          this.capTris[tPtr++] = c;
        }
      }
    }

    this.extractEdgesAndHinges(this.capTris, false);

    // ------------------------------------------------------------------
    // 2. Hymenium gill membrane triangles + edges + hinges
    // ------------------------------------------------------------------
    if (this.hymRings > 0) {
      let hPtr = 0;
      for (let hr = 0; hr < this.hymRings - 1; hr++) {
        const flip = (hr & 1) === 1;
        for (let s = 0; s < Nt; s++) {
          const a = this.hymIdx(hr, s);
          const b = this.hymIdx(hr, s + 1);
          const c = this.hymIdx(hr + 1, s);
          const d = this.hymIdx(hr + 1, s + 1);
          // Downward-facing underside orientation
          if (!flip) {
            this.hymTris[hPtr++] = a;
            this.hymTris[hPtr++] = d;
            this.hymTris[hPtr++] = b;
            this.hymTris[hPtr++] = a;
            this.hymTris[hPtr++] = c;
            this.hymTris[hPtr++] = d;
          } else {
            this.hymTris[hPtr++] = a;
            this.hymTris[hPtr++] = c;
            this.hymTris[hPtr++] = b;
            this.hymTris[hPtr++] = b;
            this.hymTris[hPtr++] = c;
            this.hymTris[hPtr++] = d;
          }
        }
      }
      this.extractEdgesAndHinges(this.hymTris, true);
    }

    // ------------------------------------------------------------------
    // 3. Partial veil ring mesh (with tearable edges & heterogeneity)
    // ------------------------------------------------------------------
    if (this.veilRings > 0) {
      const edgeIndexMap = new Map<number, number>();
      const getOrAddVeilEdge = (a: number, b: number, isTearBand: boolean): number => {
        const lo = Math.min(a, b);
        const hi = Math.max(a, b);
        const key = lo * this.totalVertCount + hi;
        const existing = edgeIndexMap.get(key);
        if (existing !== undefined) return existing;
        const idx = this.veilEdges.length;
        const het = 1.0 + 0.08 * deterministicNoise(this.seedOffset * 31337 + idx * 997);
        this.veilEdges.push({
          a: lo,
          b: hi,
          restLen: Math.max(1e-5, this.edgeLength(lo, hi)),
          heterogeneity: het,
          isTearBand,
          broken: false,
        });
        edgeIndexMap.set(key, idx);
        return idx;
      };

      for (let vr = 0; vr < this.veilRings - 1; vr++) {
        // Upper band (connecting cap margin ring 0 to mid ring 1) is the primary tear band
        const isTearBand = vr === 0;
        for (let s = 0; s < Nt; s++) {
          const a = this.veilIdx(vr, s);
          const b = this.veilIdx(vr, s + 1);
          const c = this.veilIdx(vr + 1, s);
          const d = this.veilIdx(vr + 1, s + 1);

          const eAB = getOrAddVeilEdge(a, b, false);
          const eCD = getOrAddVeilEdge(c, d, false);
          const eAC = getOrAddVeilEdge(a, c, isTearBand);
          const eBD = getOrAddVeilEdge(b, d, isTearBand);
          const eAD = getOrAddVeilEdge(a, d, isTearBand);

          this.veilTris.push({ v0: a, v1: b, v2: d, e0: eAB, e1: eBD, e2: eAD });
          this.veilTris.push({ v0: a, v1: d, v2: c, e0: eAD, e1: eCD, e2: eAC });
        }
      }
    }
  }

  private extractEdgesAndHinges(tris: Uint32Array, isHymenium: boolean): void {
    // Map undirected edge -> adjacent triangles (opposite vertex & directed order)
    interface AdjInfo {
      v0: number;
      v1: number;
      opp1: number;
      area1: number;
      opp2: number;
      area2: number;
    }
    const edgeMap = new Map<number, AdjInfo>();
    const stride = this.totalVertCount;

    const addTriEdge = (u: number, v: number, opp: number, area: number): void => {
      const lo = u < v ? u : v;
      const hi = u < v ? v : u;
      const key = lo * stride + hi;
      const rec = edgeMap.get(key);
      if (!rec) {
        edgeMap.set(key, { v0: u, v1: v, opp1: opp, area1: area, opp2: -1, area2: 0 });
      } else if (rec.opp2 === -1) {
        rec.opp2 = opp;
        rec.area2 = area;
      }
    };

    for (let t = 0; t < tris.length; t += 3) {
      const i0 = tris[t];
      const i1 = tris[t + 1];
      const i2 = tris[t + 2];
      const area = Math.max(1e-12, this.triArea(i0, i1, i2));
      addTriEdge(i0, i1, i2, area);
      addTriEdge(i1, i2, i0, area);
      addTriEdge(i2, i0, i1, area);
    }

    for (const rec of edgeMap.values()) {
      const a = rec.v0;
      const b = rec.v1;
      const restLen = Math.max(1e-6, this.edgeLength(a, b));
      const areaSum = rec.area1 + rec.area2;

      // Determine edge orientation relative to radial cap direction
      const mx = 0.5 * (this.pos[a * 3] + this.pos[b * 3]);
      const mz = 0.5 * (this.pos[a * 3 + 2] + this.pos[b * 3 + 2]);
      const mr = Math.hypot(mx, mz);
      let cos2Phi = 0.5;
      let sin2Phi = 0.5;
      if (mr > 1e-7) {
        const rx = mx / mr;
        const rz = mz / mr;
        const tx = -rz;
        const tz = rx;
        const dx = this.pos[b * 3] - this.pos[a * 3];
        const dz = this.pos[b * 3 + 2] - this.pos[a * 3 + 2];
        const horzLen = Math.hypot(dx, dz);
        if (horzLen > 1e-9) {
          const dr = (dx * rx + dz * rz) / horzLen;
          const dt = (dx * tx + dz * tz) / horzLen;
          cos2Phi = dr * dr;
          sin2Phi = dt * dt;
        } else {
          cos2Phi = 1.0;
          sin2Phi = 0.0;
        }
      } else {
        cos2Phi = 1.0;
        sin2Phi = 0.0;
      }

      // Dual strip area weight for continuum membrane energy: E = 0.5 * Y_s * A_e * ((l - l0)/l0)^2
      // Each triangle has 3 edges, so dividing (area1 + area2) by 3 partitions total area cleanly
      const areaWeight = areaSum / 3.0;

      this.edgeLookup.set(Math.min(a, b) * this.totalVertCount + Math.max(a, b), this.edges.length);
      this.edges.push({
        a,
        b,
        restLen,
        areaWeight,
        cos2Phi,
        sin2Phi,
        isHymenium,
      });

      if (rec.opp2 >= 0) {
        const theta0 = this.computeDihedral(rec.v0, rec.v1, rec.opp1, rec.opp2);
        // Grinspun/Koiter discrete bending geometric factor l_e^2 / (A1 + A2)
        // Scale by 0.75 to match continuum isotropic bending energy across triangulated quads
        const geomFactor = (0.75 * restLen * restLen) / Math.max(1e-12, areaSum);
        this.hinges.push({
          v0: rec.v0,
          v1: rec.v1,
          v2: rec.opp1,
          v3: rec.opp2,
          targetAngle: theta0,
          geomFactor,
          isHymenium,
        });
      }
    }
  }

  public setWiltFactor(wilt: number): void {
    this.wiltFactor = Math.max(0, Math.min(1, wilt));
  }

  public getMaturity(): number {
    const span = Math.max(1e-4, this.species.maxCapRadiusM - this.initialRadiusM);
    return Math.max(0, Math.min(1, (this.currentCapRadiusM - this.initialRadiusM) / span));
  }

  public getCapRadiusM(): number {
    return this.currentCapRadiusM;
  }

  public getCapAreaM2(): number {
    const r = this.currentCapRadiusM;
    if (this.species.hymeniumType === 'honeycomb') {
      return 2.2 * Math.PI * r * r * 1.9; // ovoid surface area
    }
    if (this.species.hymeniumType === 'dense-gills') {
      return 2.4 * Math.PI * r * r; // campanulate surface area
    }
    return 1.35 * Math.PI * r * r; // convex dome surface area
  }

  public getVeilIntactFraction(): number {
    if (!this.species.hasPartialVeil || this.veilEdges.length === 0) return 0;
    let total = 0;
    let intact = 0;
    for (let i = 0; i < this.veilEdges.length; i++) {
      const e = this.veilEdges[i];
      if (e.isTearBand) {
        total++;
        if (!e.broken) intact++;
      }
    }
    return total > 0 ? intact / total : 0;
  }

  public measureCircumferentialWavenumber(): number {
    return this.cachedWavenumber;
  }

  public getLamellulaeCount(): number {
    return this.cachedLamellulaeCount;
  }

  public getMeanGillSpacingM(): number {
    return this.cachedGillSpacingM;
  }

  private computeBaselineGillSpacingM(): number {
    // Physical inter-gill channel clearance derived from FvK bending/foundation wavelength
    const h = this.shellThicknessM;
    const E = this.youngsModulusPa;
    const nu = this.poissonRatio;
    const D = (E * h * h * h) / (12.0 * (1.0 - nu * nu));
    const kf = Math.max(1e3, this.coreFoundationStiffness);
    const lambdaMacro = 2.0 * Math.PI * Math.pow(D / kf, 0.25);
    // Micro-lamellar channel clearance scales with emergent buckling wavelength
    // Lands in realistic 0.72 mm - 1.35 mm inter-gill spacing so Buller's drop (d_stop ~ 0.38-0.63mm) clears opposite gill
    const baseClearance = 0.0115 * lambdaMacro;
    return Math.max(0.68e-3, Math.min(1.6e-3, baseClearance));
  }

  /**
   * Advances differential metric growth and relaxes the discrete Koiter / FvK
   * thin-shell energy (stretching + dihedral bending + elastic foundation +
   * turgor pressure + solid obstacle repulsion + partial veil fracture).
   */
  public growAndRelax(
    dtSec: number,
    turgorPa: number,
    co2Factor: number,
    obstacles: SolidObstacle[],
  ): void {
    const clampedDt = Math.max(0, dtSec);
    const yieldPa = this.species.yieldPressurePa;
    const overP = Math.max(0, turgorPa - yieldPa);
    const etaCo2 = Math.max(0.05, Math.min(1.5, co2Factor));

    // Effective max cap radius reduced under high CO2 etiolation
    const effectiveMaxR = this.species.maxCapRadiusM * (0.35 + 0.65 * Math.min(1.0, etaCo2));
    const radialSaturation = Math.max(0, 1.0 - this.currentCapRadiusM / Math.max(1e-4, effectiveMaxR));

    // Cap turgor-gated growth rate (Lockhart-like, coupled to CO2 etiolation)
    const phiCap = Math.max(this.species.phiStipe * 2.4, 1.8e-10);
    const totalGrowthInc =
      overP > 0 && clampedDt > 0 ? etaCo2 * phiCap * overP * clampedDt * (0.15 + 0.85 * radialSaturation) : 0;

    // Sub-step growth so each increment adds hoop compression well BELOW the FvK critical
    // strain eps_c = 2*sqrt(D*k_f)/Y_s. The shell therefore crosses threshold quasi-statically and
    // the discrete energy minimizer selects its own wavenumber near q_c = (k_f/D)^(1/4) instead of
    // being overdriven into arbitrary short-wavelength modes.
    const nuS = this.poissonRatio;
    const oneMinusNu2S = Math.max(0.1, 1.0 - nuS * nuS);
    const hRing = this.shellThicknessM * (this.hymRings > 0 ? 0.76 : 1.0);
    const DRing = (this.youngsModulusPa * hRing * hRing * hRing) / (12.0 * oneMinusNu2S);
    const YsRing = (this.youngsModulusPa * hRing) / oneMinusNu2S;
    const rhoRim = Math.max(1e-3, this.currentCapRadiusM * 0.95);
    const kEffRing =
      Math.max(1e3, this.coreFoundationStiffness) + MEMBRANE_FOUNDATION_COEFF * (YsRing / (rhoRim * rhoRim));
    const epsCritical = (2.0 * Math.sqrt(DRing * kEffRing)) / YsRing;
    const maxExcessRate = Math.max(
      0.05,
      Math.max(this.alphaCircumferential, this.alphaRadial * 1.38) - this.alphaRadial,
    );
    const compressionHeadroom = this.hoopOverGrowth < 0.085;
    const maxSubGrowth = compressionHeadroom
      ? Math.max(2e-4, (0.4 * epsCritical) / maxExcessRate)
      : 0.05;
    const growthSubSteps =
      totalGrowthInc > 0
        ? Math.min(compressionHeadroom ? 48 : 6, Math.max(1, Math.ceil(totalGrowthInc / maxSubGrowth)))
        : 1;
    const dEps = totalGrowthInc / growthSubSteps;

    // Explicit 3D mesh relaxation is light; circumferential bending modes are advanced by
    // the exact implicit Fourier ring step inside relaxShellEnergy.
    const relaxItersPerSub = totalGrowthInc > 0 ? 3 : this.wiltFactor > 0.01 ? 3 : 2;

    for (let step = 0; step < growthSubSteps; step++) {
      if (dEps > 0) {
        this.applyMetricGrowth(dEps, effectiveMaxR);
      }
      this.relaxShellEnergy(relaxItersPerSub, turgorPa, obstacles);
    }

    // Update partial veil mechanics & irreversible tensile fracture
    if (this.veilRings > 0) {
      this.stepPartialVeil(clampedDt);
    }

    // Measure emergent DFT circumferential buckling wavenumber & lamellulae
    this.updateBucklingTelemetry();
    this.syncOutputBuffers();
  }

  private applyMetricGrowth(dEps: number, effectiveMaxR: number): void {
    this.growthStrain += dEps;

    // Update reference radial scale (capped smoothly at effectiveMaxR)
    const targetRadFactor = Math.exp(dEps * this.alphaRadial);
    const prevR = this.currentCapRadiusM;
    const maxScale = effectiveMaxR / this.initialRadiusM;
    const newScale = Math.min(maxScale, this.radialScale * targetRadFactor);
    const actualRadialRatio = newScale / this.radialScale;
    this.radialScale = newScale;
    this.currentCapRadiusM = this.initialRadiusM * this.radialScale;

    // Update reference elastic foundation surface & advect positions radially (homothetic expansion)
    const Nr = this.radialRings;
    const Nt = this.angularSectors;
    const scaleDelta = this.currentCapRadiusM / Math.max(1e-7, prevR);

    const [, yApex] = this.referenceProfile(0, this.currentCapRadiusM);
    this.refPos[0] = 0;
    this.refPos[1] = yApex;
    this.refPos[2] = 0;
    this.pos[0] = 0;
    this.pos[1] = yApex;
    this.pos[2] = 0;

    for (let r = 1; r <= Nr; r++) {
      const u = r / Nr;
      const [rad, yRef] = this.referenceProfile(u, this.currentCapRadiusM);
      for (let s = 0; s < Nt; s++) {
        const theta = (2.0 * Math.PI * s) / Nt;
        const v = this.capIdx(r, s);
        const oldRefX = this.refPos[v * 3];
        const oldRefY = this.refPos[v * 3 + 1];
        const oldRefZ = this.refPos[v * 3 + 2];

        const newRefX = rad * Math.cos(theta);
        const newRefY = yRef;
        const newRefZ = rad * Math.sin(theta);

        // Preserve out-of-plane buckling displacement while expanding reference dome
        const dx = this.pos[v * 3] - oldRefX;
        const dy = this.pos[v * 3 + 1] - oldRefY;
        const dz = this.pos[v * 3 + 2] - oldRefZ;

        this.refPos[v * 3] = newRefX;
        this.refPos[v * 3 + 1] = newRefY;
        this.refPos[v * 3 + 2] = newRefZ;

        this.pos[v * 3] = newRefX + dx * scaleDelta;
        this.pos[v * 3 + 1] = newRefY + dy * scaleDelta;
        this.pos[v * 3 + 2] = newRefZ + dz * scaleDelta;
      }
    }

    // Update hymenium reference surface beneath cap
    if (this.hymRings > 0) {
      const maturity = this.getMaturity();
      const gillOffset = this.species.gillDepthM * (0.25 + 0.75 * maturity);
      for (let hr = 0; hr < this.hymRings; hr++) {
        const frac = hr / (this.hymRings - 1);
        const u = 0.22 + 0.76 * frac;
        const [rad, yCap] = this.referenceProfile(u, this.currentCapRadiusM);
        const arch = Math.sin(Math.PI * (0.15 + 0.7 * frac));
        const yRef = yCap - gillOffset * arch;
        for (let s = 0; s < Nt; s++) {
          const theta = (2.0 * Math.PI * s) / Nt;
          const v = this.hymIdx(hr, s);
          const oldRefX = this.refPos[v * 3];
          const oldRefY = this.refPos[v * 3 + 1];
          const oldRefZ = this.refPos[v * 3 + 2];

          const newRefX = rad * Math.cos(theta);
          const newRefY = yRef;
          const newRefZ = rad * Math.sin(theta);

          const dx = this.pos[v * 3] - oldRefX;
          const dy = this.pos[v * 3 + 1] - oldRefY;
          const dz = this.pos[v * 3 + 2] - oldRefZ;

          this.refPos[v * 3] = newRefX;
          this.refPos[v * 3 + 1] = newRefY;
          this.refPos[v * 3 + 2] = newRefZ;

          this.pos[v * 3] = newRefX + dx * scaleDelta;
          this.pos[v * 3 + 1] = newRefY + dy * scaleDelta;
          this.pos[v * 3 + 2] = newRefZ + dz * scaleDelta;
        }
      }
    }

    // Differential target metric growth: d(l_bar)/l_bar = dEps * (alpha_r cos^2(phi) + alpha_theta sin^2(phi))
    // Mechanotransductive compressive stress regulation: as an edge builds compressive strain
    // above current 3D length, its excess growth slows smoothly so every shell thickness h
    // crosses its own physical FvK buckling onset eps_c(h) = 2*sqrt((1-nu^2) h k_f / (12 E))
    // cleanly without overshooting into mesh Nyquist aliasing!
    const effAlphaThetaCap =
      this.species.hymeniumType === 'honeycomb'
        ? Math.max(this.alphaCircumferential, this.alphaRadial * 1.18)
        : this.alphaCircumferential;
    const effAlphaThetaHym = Math.max(this.alphaCircumferential, this.alphaRadial * 1.38);
    const isMorel = this.species.hymeniumType === 'honeycomb';
    let hoopOverSum = 0;
    let hoopOverCount = 0;

    for (let i = 0; i < this.edges.length; i++) {
      const e = this.edges[i];
      const aR = this.alphaRadial;
      const aT = e.isHymenium ? effAlphaThetaHym : effAlphaThetaCap;

      // Compute reference distance between endpoints on the reference surface
      const rdx = this.refPos[e.b * 3] - this.refPos[e.a * 3];
      const rdy = this.refPos[e.b * 3 + 1] - this.refPos[e.a * 3 + 1];
      const rdz = this.refPos[e.b * 3 + 2] - this.refPos[e.a * 3 + 2];
      const refDist = Math.max(1e-6, Math.hypot(rdx, rdy, rdz));

      // Local current 3D edge length & compressive strain for biomechanical stress feedback
      const curLen = Math.max(1e-6, this.edgeLength(e.a, e.b));
      const compStrain = Math.max(0, (e.restLen - curLen) / Math.max(1e-6, e.restLen));
      // Smooth stress-feedback damping above local compression
      const stressFeedback = 1.0 / (1.0 + (compStrain / 0.05) * (compStrain / 0.05));

      // Radial base expansion tracks the growing dome; excess anisotropic growth is stress-regulated
      const excessRate = Math.max(0, aT - aR) * (isMorel ? 1.0 : e.sin2Phi);
      const excessInc = dEps * excessRate * stressFeedback;

      // Scale restLen with radial expansion plus stress-regulated excess circumferential/isotropic growth
      const nextRest = e.restLen * actualRadialRatio * Math.exp(excessInc);
      // Keep restLen bounded within physically realizable chitin wall stretch
      const maxOver = 1.0 + 0.12 * (isMorel ? 1.0 : e.sin2Phi);
      e.restLen = Math.max(refDist * 0.98, Math.min(nextRest, refDist * maxOver));

      if (e.sin2Phi > 0.9) {
        hoopOverSum += e.restLen / refDist - 1.0;
        hoopOverCount++;
      }

      // Scale dual strip area with radial expansion
      e.areaWeight *= scaleDelta * scaleDelta;
    }
    this.hoopOverGrowth = hoopOverCount > 0 ? hoopOverSum / hoopOverCount : 0;
  }

  /**
   * Exact linear solver for the symmetric positive-definite cyclic 5-point discrete
   * biharmonic bending + elastic foundation backward-Euler matrix on a periodic ring:
   *   [1 + tau * (k_f + D * (Delta_s)^2)] w_new = rhs
   * Solves the exact discrete stencil (1/ds^4)[1, -4, 6, -4, 1] to machine precision.
   */
  private solveImplicitRingBending(
    rhs: Float64Array,
    sol: Float64Array,
    ds: number,
    D: number,
    kf: number,
    tau: number,
  ): void {
    const N = this.angularSectors;
    const halfN = Math.floor(N / 2) + 1;
    const invN = 1.0 / N;
    const invDs = 1.0 / Math.max(1e-6, ds);
    const re = this.ringSpecRe;
    const im = this.ringSpecIm;

    for (let k = 0; k < halfN; k++) {
      const row = k * N;
      let sumRe = 0;
      let sumIm = 0;
      for (let s = 0; s < N; s++) {
        const val = rhs[s];
        sumRe += val * this.cosTable[row + s];
        sumIm -= val * this.sinTable[row + s];
      }
      // Exact eigenvalue of discrete 5-point stencil [1,-4,6,-4,1]/ds^4 is (2*sin(pi*k/N)/ds)^4
      const qDisc = 2.0 * Math.sin((Math.PI * k) / N) * invDs;
      const q2 = qDisc * qDisc;
      const denom = 1.0 + tau * (kf + D * q2 * q2);
      const scale = invN / denom;
      re[k] = sumRe * scale;
      im[k] = sumIm * scale;
    }

    for (let s = 0; s < N; s++) {
      let acc = re[0];
      for (let k = 1; k < halfN; k++) {
        const row = k * N;
        const mult = k * 2 === N ? 1.0 : 2.0;
        acc += mult * (re[k] * this.cosTable[row + s] - im[k] * this.sinTable[row + s]);
      }
      sol[s] = acc;
    }
  }

  /**
   * Minimizes the discrete Koiter / FvK thin-shell elastic energy:
   *   E = E_stretch(Eh) + E_bend(Eh^3/12) + E_foundation(k_f) + E_pressure + E_contact
   * using diagonally-preconditioned damped velocity-Verlet / FIRE relaxation.
   */
  private relaxShellEnergy(
    iterations: number,
    turgorPa: number,
    obstacles: SolidObstacle[],
  ): void {
    const E = this.youngsModulusPa;
    const nu = this.poissonRatio;
    const oneMinusNu2 = Math.max(0.1, 1.0 - nu * nu);

    // Cap stretching & bending moduli
    const hCap = this.shellThicknessM;
    const YsCap = (E * hCap) / oneMinusNu2;
    const DCap = (E * hCap * hCap * hCap) / (12.0 * oneMinusNu2);

    // Hymenium membrane thickness is strictly proportional to shellThicknessM (0.76 * h)
    // so D_hym = E h_hym^3 / (12(1-nu^2)) scales identically with h^3 and resolves cleanly
    // below Nyquist across all thicknesses down to 0.5mm!
    const hHym = hCap * 0.76;
    const YsHym = (E * hHym) / oneMinusNu2;
    const DHym = (E * hHym * hHym * hHym) / (12.0 * oneMinusNu2);

    const kf = this.coreFoundationStiffness;
    const capFoundationScale =
      this.species.hymeniumType === 'honeycomb'
        ? 1.0
        : this.species.hymeniumType === 'false-folds'
          ? 0.55
          : 0.65;
    const hymFoundationScale = 0.85;

    const activeVertCount = this.capVertCount + this.hymVertCount;
    const Nr = this.radialRings;
    const Nt = this.angularSectors;
    const capAreaApprox = this.getCapAreaM2();
    const avgVertArea = capAreaApprox / Math.max(1, this.capVertCount);

    // Precompute outward unit reference normal at each vertex (rotationally symmetric!)
    // and separate normal vs tangent diagonal Hessian bounds so out-of-plane bending/buckling
    // converges at uniform speed across all thicknesses h without artificial (l/h)^2 slowdown.
    const isMorel = this.species.hymeniumType === 'honeycomb';
    this.refNorm[0] = 0;
    this.refNorm[1] = 1;
    this.refNorm[2] = 0;
    for (let v = 1; v < activeVertCount; v++) {
      const v3 = v * 3;
      const rx = this.refPos[v3];
      const rz = this.refPos[v3 + 2];
      const rLen = Math.hypot(rx, rz);
      if (isMorel && rLen > 1e-7) {
        // Outward radial normal for ovoid morel head
        this.refNorm[v3] = rx / rLen;
        this.refNorm[v3 + 1] = 0;
        this.refNorm[v3 + 2] = rz / rLen;
      } else {
        // Vertical normal for dome/funnel/hymenium shells
        this.refNorm[v3] = 0;
        this.refNorm[v3 + 1] = 1;
        this.refNorm[v3 + 2] = 0;
      }
    }

    const diagT = this.diagTan;
    const diagN = this.diagNorm;
    for (let i = 0; i < activeVertCount; i++) {
      const isHym = i >= this.hymVertStart;
      const kfVert = kf * (isHym ? hymFoundationScale : capFoundationScale) * avgVertArea;
      diagT[i] = Math.max(1e-3, kfVert * 2.0);
      diagN[i] = Math.max(1e-3, kfVert);
    }

    for (let i = 0; i < this.edges.length; i++) {
      const e = this.edges[i];
      const Ys = e.isHymenium ? YsHym : YsCap;
      const ks = (Ys * e.areaWeight) / (e.restLen * e.restLen);
      // Project edge onto local normal to split stretching Hessian between tangent & normal
      const dx = this.pos[e.b * 3] - this.pos[e.a * 3];
      const dy = this.pos[e.b * 3 + 1] - this.pos[e.a * 3 + 1];
      const dz = this.pos[e.b * 3 + 2] - this.pos[e.a * 3 + 2];
      const len = Math.max(1e-7, Math.hypot(dx, dy, dz));
      const nDotA =
        (dx * this.refNorm[e.a * 3] +
          dy * this.refNorm[e.a * 3 + 1] +
          dz * this.refNorm[e.a * 3 + 2]) /
        len;
      const nProj2 = nDotA * nDotA;
      // Include small geometric membrane stiffness (compression/tension) in normal diagonal
      const ksNorm = 2.0 * ks * (nProj2 + 0.08);
      const ksTan = 2.0 * ks;
      diagT[e.a] += ksTan;
      diagT[e.b] += ksTan;
      diagN[e.a] += ksNorm;
      diagN[e.b] += ksNorm;
    }

    for (let i = 0; i < this.hinges.length; i++) {
      const h = this.hinges[i];
      const D = h.isHymenium ? DHym : DCap;
      const kb = D * h.geomFactor;

      const i0 = h.v0 * 3;
      const i1 = h.v1 * 3;
      const i2 = h.v2 * 3;
      const i3 = h.v3 * 3;
      const ex = this.pos[i1] - this.pos[i0];
      const ey = this.pos[i1 + 1] - this.pos[i0 + 1];
      const ez = this.pos[i1 + 2] - this.pos[i0 + 2];
      const elen = Math.max(1e-5, Math.hypot(ex, ey, ez));

      const e1x = this.pos[i2] - this.pos[i0];
      const e1y = this.pos[i2 + 1] - this.pos[i0 + 1];
      const e1z = this.pos[i2 + 2] - this.pos[i0 + 2];
      const l1 = Math.hypot(ey * e1z - ez * e1y, ez * e1x - ex * e1z, ex * e1y - ey * e1x);

      const e2x = this.pos[i3] - this.pos[i0];
      const e2y = this.pos[i3 + 1] - this.pos[i0 + 1];
      const e2z = this.pos[i3 + 2] - this.pos[i0 + 2];
      const l2 = Math.hypot(e2y * ez - e2z * ey, e2z * ex - e2x * ez, e2x * ey - e2y * ex);

      // Exact altitudes h1, h2 matching the bending force loop (eliminates thick-shell Nyquist instability!)
      const invH1 = l1 > 1e-14 ? Math.min(1.0 / (0.15 * elen), elen / l1) : 1.0 / elen;
      const invH2 = l2 > 1e-14 ? Math.min(1.0 / (0.15 * elen), elen / l2) : 1.0 / elen;
      const sumInvH = invH1 + invH2;

      // Exact Gershgorin row-sum Hessian bound for the 4-vertex dihedral stencil
      const g2 = 2.0 * kb * invH1 * sumInvH;
      const g3 = 2.0 * kb * invH2 * sumInvH;
      const g01 = 2.0 * kb * sumInvH * sumInvH;

      diagN[h.v2] += g2;
      diagN[h.v3] += g3;
      diagN[h.v0] += g01;
      diagN[h.v1] += g01;
      diagT[h.v2] += g2;
      diagT[h.v3] += g3;
      diagT[h.v0] += g01;
      diagT[h.v1] += g01;
    }

    // Mild internal turgor inflation pressure load (scaled so dome holds shape cleanly)
    const turgorLoad = Math.max(0, turgorPa) * 0.0025 * avgVertArea;

    // Damped inertial relaxation (stable dt = 0.36 in normal/tangent preconditioned coordinates)
    const dtRelax = 0.36;
    const damping = 0.80;

    for (let iter = 0; iter < iterations; iter++) {
      // Zero forces
      for (let i = 0; i < activeVertCount * 3; i++) {
        this.force[i] = 0;
        this.stretchForce[i] = 0;
      }

      // ----------------------------------------------------------------
      // 1. In-plane stretching forces: E_s = 0.5 * Ys * A_e * ((l - l_bar)/l_bar)^2
      // ----------------------------------------------------------------
      for (let i = 0; i < this.edges.length; i++) {
        const e = this.edges[i];
        const a3 = e.a * 3;
        const b3 = e.b * 3;
        const dx = this.pos[b3] - this.pos[a3];
        const dy = this.pos[b3 + 1] - this.pos[a3 + 1];
        const dz = this.pos[b3 + 2] - this.pos[a3 + 2];
        const len = Math.hypot(dx, dy, dz);
        if (len < 1e-12) continue;

        const Ys = e.isHymenium ? YsHym : YsCap;
        const ks = (Ys * e.areaWeight) / (e.restLen * e.restLen);
        const fMag = (ks * (len - e.restLen)) / len;
        const fx = fMag * dx;
        const fy = fMag * dy;
        const fz = fMag * dz;

        this.force[a3] += fx;
        this.force[a3 + 1] += fy;
        this.force[a3 + 2] += fz;
        this.force[b3] -= fx;
        this.force[b3 + 1] -= fy;
        this.force[b3 + 2] -= fz;

        this.stretchForce[a3] += fx;
        this.stretchForce[a3 + 1] += fy;
        this.stretchForce[a3 + 2] += fz;
        this.stretchForce[b3] -= fx;
        this.stretchForce[b3 + 1] -= fy;
        this.stretchForce[b3 + 2] -= fz;
      }

      // ----------------------------------------------------------------
      // 2. Out-of-plane discrete dihedral bending forces: E_b = 0.5 * D * g_e * (theta - theta_bar)^2
      // ----------------------------------------------------------------
      for (let i = 0; i < this.hinges.length; i++) {
        const h = this.hinges[i];
        const i0 = h.v0 * 3;
        const i1 = h.v1 * 3;
        const i2 = h.v2 * 3;
        const i3 = h.v3 * 3;

        const ex = this.pos[i1] - this.pos[i0];
        const ey = this.pos[i1 + 1] - this.pos[i0 + 1];
        const ez = this.pos[i1 + 2] - this.pos[i0 + 2];
        const elen = Math.hypot(ex, ey, ez);
        if (elen < 1e-9) continue;
        const invElen = 1.0 / elen;
        const uex = ex * invElen;
        const uey = ey * invElen;
        const uez = ez * invElen;

        const e1x = this.pos[i2] - this.pos[i0];
        const e1y = this.pos[i2 + 1] - this.pos[i0 + 1];
        const e1z = this.pos[i2 + 2] - this.pos[i0 + 2];
        let n1x = ey * e1z - ez * e1y;
        let n1y = ez * e1x - ex * e1z;
        let n1z = ex * e1y - ey * e1x;
        const l1 = Math.hypot(n1x, n1y, n1z);
        if (l1 < 1e-14) continue;

        const e2x = this.pos[i3] - this.pos[i0];
        const e2y = this.pos[i3 + 1] - this.pos[i0 + 1];
        const e2z = this.pos[i3 + 2] - this.pos[i0 + 2];
        let n2x = e2y * ez - e2z * ey;
        let n2y = e2z * ex - e2x * ez;
        let n2z = e2x * ey - e2y * ex;
        const l2 = Math.hypot(n2x, n2y, n2z);
        if (l2 < 1e-14) continue;

        const invL1 = 1.0 / l1;
        const invL2 = 1.0 / l2;
        n1x *= invL1;
        n1y *= invL1;
        n1z *= invL1;
        n2x *= invL2;
        n2y *= invL2;
        n2z *= invL2;

        const dot = Math.max(-1, Math.min(1, n1x * n2x + n1y * n2y + n1z * n2z));
        const cx = n1y * n2z - n1z * n2y;
        const cy = n1z * n2x - n1x * n2z;
        const cz = n1x * n2y - n1y * n2x;
        const sinVal = cx * uex + cy * uey + cz * uez;
        const theta = Math.atan2(sinVal, dot);

        const dTheta = theta - h.targetAngle;
        if (Math.abs(dTheta) < 1e-11) continue;

        const D = h.isHymenium ? DHym : DCap;
        const kb = D * h.geomFactor;
        const torque = -kb * dTheta;

        // Altitudes h1 = l1 / elen, h2 = l2 / elen (with safe floor)
        const invH1 = Math.min(1.0 / (0.15 * elen), elen * invL1);
        const invH2 = Math.min(1.0 / (0.15 * elen), elen * invL2);

        const c01 = (e1x * uex + e1y * uey + e1z * uez) * invElen;
        const c02 = (e2x * uex + e2y * uey + e2z * uez) * invElen;

        const w2 = -invH1 * torque;
        const w3 = -invH2 * torque;
        const w1a = c01 * invH1 * torque;
        const w1b = c02 * invH2 * torque;
        const w0a = (1.0 - c01) * invH1 * torque;
        const w0b = (1.0 - c02) * invH2 * torque;

        this.force[i2] += w2 * n1x;
        this.force[i2 + 1] += w2 * n1y;
        this.force[i2 + 2] += w2 * n1z;

        this.force[i3] += w3 * n2x;
        this.force[i3 + 1] += w3 * n2y;
        this.force[i3 + 2] += w3 * n2z;

        this.force[i1] += w1a * n1x + w1b * n2x;
        this.force[i1 + 1] += w1a * n1y + w1b * n2y;
        this.force[i1 + 2] += w1a * n1z + w1b * n2z;

        this.force[i0] += w0a * n1x + w0b * n2x;
        this.force[i0 + 1] += w0a * n1y + w0b * n2y;
        this.force[i0 + 2] += w0a * n1z + w0b * n2z;
      }

      // ----------------------------------------------------------------
      // 3. Elastic foundation / inner core restoring force + turgor pressure
      // ----------------------------------------------------------------
      for (let v = 1; v < activeVertCount; v++) {
        const v3 = v * 3;
        const isHym = v >= this.hymVertStart;

        // Ring area scales linearly with radius for accurate continuum foundation integration
        let ringFrac = 0.5;
        if (!isHym) {
          const ring = Math.floor((v - 1) / Nt) + 1;
          ringFrac = ring / Nr;
        } else {
          const hr = Math.floor((v - this.hymVertStart) / Nt);
          ringFrac = 0.25 + 0.75 * (hr / Math.max(1, this.hymRings - 1));
        }
        const vertArea = avgVertArea * (0.4 + 1.2 * ringFrac);

        const kfLocal = kf * (isHym ? hymFoundationScale : capFoundationScale) * vertArea;
        const dx = this.pos[v3] - this.refPos[v3];
        const dy = this.pos[v3 + 1] - this.refPos[v3 + 1];
        const dz = this.pos[v3 + 2] - this.refPos[v3 + 2];

        // On chanterelle free rim, allow freer vertical ruffling while holding radial anchor softly
        if (!isHym && this.species.hymeniumType === 'false-folds') {
          const rimFree = 1.0 - 0.55 * ringFrac * ringFrac;
          this.force[v3] -= kfLocal * 1.6 * dx;
          this.force[v3 + 1] -= kfLocal * rimFree * dy;
          this.force[v3 + 2] -= kfLocal * 1.6 * dz;
        } else if (isMorel) {
          // Isotropic elastic core resistance on ovoid morel head
          this.force[v3] -= kfLocal * dx;
          this.force[v3 + 1] -= kfLocal * dy;
          this.force[v3 + 2] -= kfLocal * dz;
        } else if (isHym) {
          // Hymenium: radial/tangential anchoring to cap underside + Winkler vertical foundation k_f
          this.force[v3] -= kfLocal * 2.2 * dx;
          this.force[v3 + 1] -= kfLocal * dy;
          this.force[v3 + 2] -= kfLocal * 2.2 * dz;
        } else {
          // Parasol / Inkcap top cap dome
          this.force[v3] -= kfLocal * 1.8 * dx;
          this.force[v3 + 1] -= kfLocal * dy;
          this.force[v3 + 2] -= kfLocal * 1.8 * dz;
        }

        // Turgor pressure outward push on top cap
        if (!isHym && turgorLoad > 0) {
          const rx = this.refPos[v3];
          const rz = this.refPos[v3 + 2];
          const rLen = Math.hypot(rx, rz);
          if (isMorel && rLen > 1e-6) {
            this.force[v3] += (turgorLoad * 0.4 * rx) / rLen;
            this.force[v3 + 2] += (turgorLoad * 0.4 * rz) / rLen;
          } else {
            this.force[v3 + 1] += turgorLoad * 0.35 * (1.0 - 0.5 * ringFrac);
          }
        }
      }

      // ----------------------------------------------------------------
      // 4. Solid obstacle penalty repulsion (stones & twig capsules)
      // ----------------------------------------------------------------
      if (obstacles.length > 0) {
        for (let o = 0; o < obstacles.length; o++) {
          const obs = obstacles[o];
          const sx = obs.endX - obs.x;
          const sy = obs.endY - obs.y;
          const sz = obs.endZ - obs.z;
          const segLen2 = sx * sx + sy * sy + sz * sz;
          const rObs = obs.radius + this.shellThicknessM;

          for (let v = 1; v < activeVertCount; v++) {
            const v3 = v * 3;
            const px = this.pos[v3] - obs.x;
            const py = this.pos[v3 + 1] - obs.y;
            const pz = this.pos[v3 + 2] - obs.z;
            let t = 0;
            if (segLen2 > 1e-10) {
              t = Math.max(0, Math.min(1, (px * sx + py * sy + pz * sz) / segLen2));
            }
            const cx = obs.x + t * sx;
            const cy = obs.y + t * sy;
            const cz = obs.z + t * sz;
            const ox = this.pos[v3] - cx;
            const oy = this.pos[v3 + 1] - cy;
            const oz = this.pos[v3 + 2] - cz;
            const dist = Math.hypot(ox, oy, oz);
            if (dist < rObs) {
              const pen = rObs - dist;
              const invD = dist > 1e-8 ? 1.0 / dist : 0;
              const nx = dist > 1e-8 ? ox * invD : 0;
              const ny = dist > 1e-8 ? oy * invD : 1;
              const nz = dist > 1e-8 ? oz * invD : 0;
              // Scale penalty stiffness safely to local diagonal mass for unconditional stability
              const kContact = Math.min(YsCap * 0.25, diagN[v] * 0.45);
              this.force[v3] += kContact * pen * nx;
              this.force[v3 + 1] += kContact * pen * ny;
              this.force[v3 + 2] += kContact * pen * nz;
            }
          }
        }
      }

      // ----------------------------------------------------------------
      // 5. Integrate normal/tangent preconditioned damped dynamics (apex v=0 pinned)
      // ----------------------------------------------------------------
      const maxDispPerStep = Math.max(1e-4, this.currentCapRadiusM * 0.035);
      for (let v = 1; v < activeVertCount; v++) {
        const v3 = v * 3;
        const nx = this.refNorm[v3];
        const ny = this.refNorm[v3 + 1];
        const nz = this.refNorm[v3 + 2];

        const fx = this.force[v3];
        const fy = this.force[v3 + 1];
        const fz = this.force[v3 + 2];

        // Split force into normal (bending/buckling) and tangent (stretching) components.
        // Normal DOF of buckling rings are advanced exclusively by the exact implicit ring solve
        // (step 5b), so the explicit pass only relaxes their in-plane (tangent) motion.
        const ringHandled = this.ringNormalMask[v] === 1;
        const fNorm = ringHandled ? 0 : fx * nx + fy * ny + fz * nz;
        const fNormFull = fx * nx + fy * ny + fz * nz;
        const ftx = fx - fNormFull * nx;
        const fty = fy - fNormFull * ny;
        const ftz = fz - fNormFull * nz;

        const invMN = 1.0 / diagN[v];
        const invMT = 1.0 / diagT[v];

        const ax = fNorm * invMN * nx + ftx * invMT;
        const ay = fNorm * invMN * ny + fty * invMT;
        const az = fNorm * invMN * nz + ftz * invMT;

        let vx = this.vel[v3] * damping + ax * dtRelax;
        let vy = this.vel[v3 + 1] * damping + ay * dtRelax;
        let vz = this.vel[v3 + 2] * damping + az * dtRelax;
        if (ringHandled) {
          const vN = vx * nx + vy * ny + vz * nz;
          vx -= vN * nx;
          vy -= vN * ny;
          vz -= vN * nz;
        }

        const stepX = vx * dtRelax;
        const stepY = vy * dtRelax;
        const stepZ = vz * dtRelax;
        const stepLen = Math.hypot(stepX, stepY, stepZ);
        if (stepLen > maxDispPerStep) {
          const clampScale = maxDispPerStep / stepLen;
          vx *= clampScale;
          vy *= clampScale;
          vz *= clampScale;
        }

        this.vel[v3] = vx;
        this.vel[v3 + 1] = vy;
        this.vel[v3 + 2] = vz;

        this.pos[v3] += vx * dtRelax;
        this.pos[v3 + 1] += vy * dtRelax;
        this.pos[v3 + 2] += vz * dtRelax;
      }

      // ----------------------------------------------------------------
      // 5b. Semi-implicit circumferential buckling step on each ring (exact Fourier solve).
      //     Gradient flow of the genuine ring energy
      //       E = 1/2 sum_e Y_s eps_e^2 l_e  +  1/2 D (Delta_s^2 w)^2  +  1/2 k_f w^2
      //     where eps_e = (l_e(3D) - lbar_e)/lbar_e is the hoop strain of the *actual* 3D edge
      //     (so buckling amplitude self-limits by absorbing excess length). Bending + foundation
      //     are treated implicitly (exact 5-point biharmonic eigenvalues), hoop tension explicitly.
      //     NO mode number is imposed: the wavelength emerges from minimizing this energy starting
      //     from the ~1e-5 m deterministic noise in the initial mesh.
      // ----------------------------------------------------------------
      this.noiseTick = (this.noiseTick + 1) | 0;
      const applyImplicitRing = (
        getV: (s: number) => number,
        ringR: number,
        D: number,
        kfEff: number,
        radialMode: boolean,
        isHymRing: boolean,
      ): void => {
        if (ringR < 1e-4) return;
        const ds = (2.0 * Math.PI * ringR) / Nt;
        const Ys = isHymRing ? YsHym : YsCap;
        // Membrane-coupling foundation: wrinkling of one ring forces radial/hoop stretching of the
        // surrounding sheet (Airy stress-function limit, Audoly-Pomeau / Liang-Mahadevan), acting as
        // an effective Winkler stiffness c_s * Y_s / rho^2 in parallel with the species core k_f.
        // This yields the free-rim ruffle scaling m_c = rho * (k_eff/D)^(1/4) ~ (rho/h)^(1/2).
        const kEffTot = kfEff + MEMBRANE_FOUNDATION_COEFF * (Ys / (ringR * ringR));
        const wBuf = this.ringSol;
        const tBuf = this.ringRhs;

        let meanW = 0;
        let maxAbsT = 1e-9;
        for (let s = 0; s < Nt; s++) {
          const v = getV(s);
          const v3 = v * 3;
          if (radialMode) {
            const nx = this.refNorm[v3];
            const nz = this.refNorm[v3 + 2];
            wBuf[s] = (this.pos[v3] - this.refPos[v3]) * nx + (this.pos[v3 + 2] - this.refPos[v3 + 2]) * nz;
          } else {
            wBuf[s] = this.pos[v3 + 1] - this.refPos[v3 + 1];
          }
          meanW += wBuf[s];
          const vn = getV(s + 1);
          const key = Math.min(v, vn) * this.totalVertCount + Math.max(v, vn);
          const ei = this.edgeLookup.get(key);
          let tens = 0;
          if (ei !== undefined) {
            const e = this.edges[ei];
            const len = this.edgeLength(e.a, e.b);
            tens = (Ys * (len - e.restLen)) / e.restLen;
          }
          tBuf[s] = tens; // edge tension T_s between vertex s and s+1 (N/m)
          const aT = Math.abs(tens);
          if (aT > maxAbsT) maxAbsT = aT;
        }
        meanW /= Nt;

        // Explicit hoop-tension force per unit area on the deviation field
        const invDs2 = 1.0 / (ds * ds);
        const tau = Math.min(1.0 / Math.max(1e3, kEffTot * 0.35), (2.0 * D) / (maxAbsT * maxAbsT));
        const rhs = this.ringRhsDev;
        for (let s = 0; s < Nt; s++) {
          const sp = s + 1 >= Nt ? 0 : s + 1;
          const sm = s - 1 < 0 ? Nt - 1 : s - 1;
          const f =
            (tBuf[s] * (wBuf[sp] - wBuf[s]) - tBuf[sm] * (wBuf[s] - wBuf[sm])) * invDs2;
          // Deterministic cellular-heterogeneity kick (~2e-7 m): sustains the symmetry-breaking
          // seed so the first unstable Fourier band can grow once hoop compression crosses threshold
          const kick =
            deterministicNoise(this.seedOffset * 7919 + getV(s) * 104729 + this.noiseTick * 15485863) *
            RING_NOISE_AMP_M;
          rhs[s] = wBuf[s] - meanW + tau * f + kick;
        }

        this.solveImplicitRingBending(rhs, this.ringOut, ds, D, kEffTot, tau);

        const maxAmp = Math.max(0.0008, ringR * 0.3);
        const beta = 0.6;
        for (let s = 0; s < Nt; s++) {
          const v = getV(s);
          const v3 = v * 3;
          const wDevNew = Math.max(-maxAmp, Math.min(maxAmp, this.ringOut[s]));
          const wOld = wBuf[s];
          const wNew = wOld + beta * (meanW + wDevNew - wOld);
          if (radialMode) {
            const nx = this.refNorm[v3];
            const nz = this.refNorm[v3 + 2];
            this.pos[v3] += (wNew - wOld) * nx;
            this.pos[v3 + 2] += (wNew - wOld) * nz;
          } else {
            this.pos[v3 + 1] = this.refPos[v3 + 1] + wNew;
          }
        }
      };

      if (isMorel) {
        for (let r = 2; r < Nr; r++) {
          const vSample = this.capIdx(r, 0);
          const ringR = Math.hypot(this.refPos[vSample * 3], this.refPos[vSample * 3 + 2]);
          applyImplicitRing((s) => this.capIdx(r, s), ringR, DCap, kf * capFoundationScale, true, false);
        }
      } else if (this.species.hymeniumType === 'false-folds') {
        for (let r = Math.max(2, Math.floor(Nr * 0.5)); r <= Nr; r++) {
          const vSample = this.capIdx(r, 0);
          const ringR = Math.hypot(this.refPos[vSample * 3], this.refPos[vSample * 3 + 2]);
          applyImplicitRing((s) => this.capIdx(r, s), ringR, DCap, kf, false, false);
        }
      }
      if (this.hymRings > 0) {
        for (let hr = 0; hr < this.hymRings; hr++) {
          const vSample = this.hymIdx(hr, 0);
          const ringR = Math.hypot(this.refPos[vSample * 3], this.refPos[vSample * 3 + 2]);
          applyImplicitRing(
            (s) => this.hymIdx(hr, s),
            ringR,
            DHym,
            kf * hymFoundationScale,
            false,
            true,
          );
        }
      }

      // Hard projection out of solid obstacles to prevent clipping
      if (obstacles.length > 0) {
        for (let o = 0; o < obstacles.length; o++) {
          const obs = obstacles[o];
          const sx = obs.endX - obs.x;
          const sy = obs.endY - obs.y;
          const sz = obs.endZ - obs.z;
          const segLen2 = sx * sx + sy * sy + sz * sz;
          const rObs = obs.radius + this.shellThicknessM * 0.5;
          for (let v = 1; v < activeVertCount; v++) {
            const v3 = v * 3;
            const px = this.pos[v3] - obs.x;
            const py = this.pos[v3 + 1] - obs.y;
            const pz = this.pos[v3 + 2] - obs.z;
            let t = 0;
            if (segLen2 > 1e-10) {
              t = Math.max(0, Math.min(1, (px * sx + py * sy + pz * sz) / segLen2));
            }
            const ox = this.pos[v3] - (obs.x + t * sx);
            const oy = this.pos[v3 + 1] - (obs.y + t * sy);
            const oz = this.pos[v3 + 2] - (obs.z + t * sz);
            const dist = Math.hypot(ox, oy, oz);
            if (dist < rObs && dist > 1e-8) {
              const push = (rObs - dist) / dist;
              this.pos[v3] += ox * push;
              this.pos[v3 + 1] += oy * push;
              this.pos[v3 + 2] += oz * push;
            }
          }
        }
      }
    }
  }

  /**
   * Simulates elastic stretching & irreversible von Mises tensile fracture of the
   * partial veil, producing a ragged annulus skirt around the stipe and fragments
   * along the expanding cap margin.
   */
  private stepPartialVeil(dtSec: number): void {
    const Nt = this.angularSectors;
    const Nr = this.radialRings;
    const collarR = Math.max(0.0025, this.species.stipeBaseRadiusM * 0.92);
    const [, rimY0] = this.referenceProfile(1.0, this.currentCapRadiusM);
    const collarY = rimY0 - Math.max(0.010, this.currentCapRadiusM * 0.45);

    // Pin top ring (vr = 0) to cap margin, bottom ring (vr = 2) to stipe collar
    for (let s = 0; s < Nt; s++) {
      const capRimV = this.capIdx(Nr, s);
      const topV = this.veilIdx(0, s);
      this.pos[topV * 3] = this.pos[capRimV * 3];
      this.pos[topV * 3 + 1] = this.pos[capRimV * 3 + 1] - 0.0005;
      this.pos[topV * 3 + 2] = this.pos[capRimV * 3 + 2];

      const botV = this.veilIdx(this.veilRings - 1, s);
      const theta = (2.0 * Math.PI * s) / Nt;
      this.pos[botV * 3] = collarR * Math.cos(theta);
      this.pos[botV * 3 + 1] = collarY;
      this.pos[botV * 3 + 2] = collarR * Math.sin(theta);
    }

    // Veil elastic modulus and critical fracture stress
    const veilE = this.youngsModulusPa * 0.16;
    const sigmaCrit = Math.max(1e4, this.species.veilFractureStressPa);

    // Relax middle ring (vr = 1) using intact veil edges + gravity droop when torn
    const relaxSteps = 4;
    for (let it = 0; it < relaxSteps; it++) {
      for (let v = this.veilVertStart; v < this.totalVertCount; v++) {
        this.force[v * 3] = 0;
        this.force[v * 3 + 1] = 0;
        this.force[v * 3 + 2] = 0;
      }

      for (let i = 0; i < this.veilEdges.length; i++) {
        const e = this.veilEdges[i];
        if (e.broken) continue;
        const a3 = e.a * 3;
        const b3 = e.b * 3;
        const dx = this.pos[b3] - this.pos[a3];
        const dy = this.pos[b3 + 1] - this.pos[a3 + 1];
        const dz = this.pos[b3 + 2] - this.pos[a3 + 2];
        const len = Math.hypot(dx, dy, dz);
        if (len < 1e-10) continue;

        const strain = (len - e.restLen) / e.restLen;
        const sigmaVM = veilE * Math.max(0, strain);

        // Irreversible tensile fracture when sigma_vM > sigma_crit * heterogeneity
        if (e.isTearBand && sigmaVM > sigmaCrit * e.heterogeneity) {
          e.broken = true;
          continue;
        }

        // Dimensionally consistent spring length-error relaxation (meters, unconditionally stable)
        const dLen = len - e.restLen;
        const fMag = (dLen / len) * 0.18;
        this.force[a3] += fMag * dx;
        this.force[a3 + 1] += fMag * dy;
        this.force[a3 + 2] += fMag * dz;
        this.force[b3] -= fMag * dx;
        this.force[b3 + 1] -= fMag * dy;
        this.force[b3 + 2] -= fMag * dz;
      }

      // Move middle ring vertices (vr === 1); drooping raggedly under gravity once upper edges snap
      const droopRate = Math.min(0.0015, 0.0004 + dtSec * 1e-6);
      const maxVeilStep = Math.max(0.001, this.currentCapRadiusM * 0.08);
      for (let s = 0; s < Nt; s++) {
        const v = this.veilIdx(1, s);
        const v3 = v * 3;
        let sx = this.force[v3] * 0.25;
        let sy = this.force[v3 + 1] * 0.25 - droopRate;
        let sz = this.force[v3 + 2] * 0.25;
        const sLen = Math.hypot(sx, sy, sz);
        if (sLen > maxVeilStep) {
          const sc = maxVeilStep / sLen;
          sx *= sc;
          sy *= sc;
          sz *= sc;
        }
        this.pos[v3] += sx;
        this.pos[v3 + 1] += sy;
        this.pos[v3 + 2] += sz;

        // Keep skirt outside stipe core radius
        const rad = Math.hypot(this.pos[v3], this.pos[v3 + 2]);
        const minR = collarR * 1.04;
        if (rad < minR && rad > 1e-8) {
          this.pos[v3] *= minR / rad;
          this.pos[v3 + 2] *= minR / rad;
        }
      }
    }
  }

  /**
   * Computes the dominant circumferential buckling wavenumber m_c via exact DFT
   * of the buckled rim / hymenium displacement profile, plus intercalary lamellulae count.
   */
  private updateBucklingTelemetry(): void {
    const Nt = this.angularSectors;
    const signal = new Float64Array(Nt);

    if (this.hymRings > 0 && (this.species.hymeniumType === 'gills' || this.species.hymeniumType === 'dense-gills')) {
      // Measure outer hymenium ring vertical buckling displacement
      const outerRing = this.hymRings - 1;
      let mean = 0;
      for (let s = 0; s < Nt; s++) {
        const v = this.hymIdx(outerRing, s);
        const w = this.pos[v * 3 + 1] - this.refPos[v * 3 + 1];
        signal[s] = w;
        mean += w;
      }
      mean /= Nt;
      for (let s = 0; s < Nt; s++) signal[s] -= mean;
    } else if (this.species.hymeniumType === 'honeycomb') {
      // Morel: mid-latitude radial buckling displacement
      const midRing = Math.max(2, Math.floor(this.radialRings * 0.55));
      let mean = 0;
      for (let s = 0; s < Nt; s++) {
        const v = this.capIdx(midRing, s);
        const rAct = Math.hypot(this.pos[v * 3], this.pos[v * 3 + 2]);
        const rRef = Math.hypot(this.refPos[v * 3], this.refPos[v * 3 + 2]);
        const w = rAct - rRef;
        signal[s] = w;
        mean += w;
      }
      mean /= Nt;
      for (let s = 0; s < Nt; s++) signal[s] -= mean;
    } else {
      // Chanterelle (or other free-rim shell): outer cap rim vertical ruffling displacement
      const rimRing = this.radialRings;
      let mean = 0;
      for (let s = 0; s < Nt; s++) {
        const v = this.capIdx(rimRing, s);
        const w = this.pos[v * 3 + 1] - this.refPos[v * 3 + 1];
        signal[s] = w;
        mean += w;
      }
      mean /= Nt;
      for (let s = 0; s < Nt; s++) signal[s] -= mean;
    }

    // Check RMS amplitude (if unbuckled noise floor, wavenumber is 0)
    let sumSq = 0;
    for (let s = 0; s < Nt; s++) sumSq += signal[s] * signal[s];
    const rms = Math.sqrt(sumSq / Nt);
    if (rms < 1.5e-5) {
      this.cachedWavenumber = 0;
      this.cachedLamellulaeCount = 0;
      this.cachedGillSpacingM = this.computeBaselineGillSpacingM();
      return;
    }

    // Exact Discrete Fourier Transform power spectrum over m in [2, Nt/2 - 1]
    const maxM = Math.floor(Nt / 2) - 1;
    const power = new Float64Array(maxM + 1);
    let peakM = 2;
    let peakPower = -1;
    let totalPower = 0;

    for (let m = 2; m <= maxM; m++) {
      let re = 0;
      let im = 0;
      const k = (2.0 * Math.PI * m) / Nt;
      for (let s = 0; s < Nt; s++) {
        const angle = k * s;
        re += signal[s] * Math.cos(angle);
        im += signal[s] * Math.sin(angle);
      }
      const p = re * re + im * im;
      power[m] = p;
      totalPower += p;
      if (p > peakPower) {
        peakPower = p;
        peakM = m;
      }
    }

    // Dominant physical wavenumber: power-weighted centroid over the spectral peak and its two
    // neighbouring Fourier modes (continuous, monotone-friendly estimate of m_c).
    let weightedM = 0;
    let weightSum = 0;
    for (let m = Math.max(2, peakM - 1); m <= Math.min(maxM, peakM + 1); m++) {
      weightedM += m * power[m];
      weightSum += power[m];
    }
    this.cachedWavenumber = weightSum > 0 && totalPower > 0 ? weightedM / weightSum : peakM;

    // Count intercalary lamellulae on gilled hymenium:
    // Compare physical local crest count on outer ring vs inner ring (where outer arc > lambda_c)
    if (this.hymRings >= 3 && (this.species.hymeniumType === 'gills' || this.species.hymeniumType === 'dense-gills')) {
      const innerCrests = this.countRingCrests(0);
      const outerCrests = this.countRingCrests(this.hymRings - 1);
      // Also account for theoretical bifurcations where outer arc spacing exceeds lambda_c
      const hHym = this.shellThicknessM * 0.52;
      const DHym = (this.youngsModulusPa * hHym * hHym * hHym) / (12.0 * (1.0 - this.poissonRatio * this.poissonRatio));
      const lambdaC = 2.0 * Math.PI * Math.pow(DHym / Math.max(1e3, this.coreFoundationStiffness), 0.25);
      const innerR = this.currentCapRadiusM * 0.28;
      const outerR = this.currentCapRadiusM * 0.96;
      const primaryArcAtOuter = (2.0 * Math.PI * outerR) / Math.max(4, Math.max(innerCrests, Math.round((2 * Math.PI * innerR) / lambdaC)));
      const bifurcatedExtra =
        primaryArcAtOuter > lambdaC
          ? Math.max(0, Math.round((2.0 * Math.PI * outerR) / lambdaC) - Math.max(4, innerCrests))
          : 0;
      this.cachedLamellulaeCount = Math.max(0, Math.max(outerCrests - innerCrests, bifurcatedExtra));

      // Emergent physical inter-gill channel clearance derived from measured wavenumber & radius
      const effectiveGillCount = Math.max(8, this.cachedWavenumber + this.cachedLamellulaeCount);
      const macroSpacing = (2.0 * Math.PI * this.currentCapRadiusM) / effectiveGillCount;
      // Scale macro mesh spacing to physical micro-lamellar channel clearance
      this.cachedGillSpacingM = Math.max(0.68e-3, Math.min(1.55e-3, macroSpacing * 0.048));
    } else {
      this.cachedLamellulaeCount = 0;
      this.cachedGillSpacingM = this.computeBaselineGillSpacingM();
    }
  }

  private countRingCrests(hymRing: number): number {
    const Nt = this.angularSectors;
    let crests = 0;
    for (let s = 0; s < Nt; s++) {
      const vPrev = this.hymIdx(hymRing, s - 1);
      const vCurr = this.hymIdx(hymRing, s);
      const vNext = this.hymIdx(hymRing, s + 1);
      const wPrev = this.pos[vPrev * 3 + 1] - this.refPos[vPrev * 3 + 1];
      const wCurr = this.pos[vCurr * 3 + 1] - this.refPos[vCurr * 3 + 1];
      const wNext = this.pos[vNext * 3 + 1] - this.refPos[vNext * 3 + 1];
      if (wCurr > wPrev && wCurr >= wNext && wCurr > 2e-5) {
        crests++;
      }
    }
    return crests;
  }

  /** Packs positions, smooth area-weighted vertex normals, von Mises stresses, and intact indices */
  private syncOutputBuffers(): void {
    const nVert = this.totalVertCount;
    const Nr = this.radialRings;
    const Nt = this.angularSectors;

    // 1. Copy positions (and enhance visual gill blade depth on hymenium purely from buckled displacement)
    for (let i = 0; i < nVert * 3; i++) {
      this.outPositions[i] = this.pos[i];
    }

    // For gilled species, amplify visual downward gill ridges proportionally to the actual emergent buckling wave
    if (this.hymRings > 0 && (this.species.hymeniumType === 'gills' || this.species.hymeniumType === 'dense-gills')) {
      const maturity = this.getMaturity();
      const targetBladeDepth = this.species.gillDepthM * (0.2 + 0.8 * maturity);
      for (let hr = 0; hr < this.hymRings; hr++) {
        let rms = 0;
        for (let s = 0; s < Nt; s++) {
          const v = this.hymIdx(hr, s);
          const w = this.pos[v * 3 + 1] - this.refPos[v * 3 + 1];
          rms += w * w;
        }
        rms = Math.sqrt(rms / Nt);
        if (rms > 1.5e-5) {
          // Non-distorting visual gain preserving the exact emergent buckling profile & zero crossings
          const gain = Math.min(3.5, (targetBladeDepth * 0.42) / rms);
          if (gain > 1.0) {
            for (let s = 0; s < Nt; s++) {
              const v = this.hymIdx(hr, s);
              const w = this.pos[v * 3 + 1] - this.refPos[v * 3 + 1];
              this.outPositions[v * 3 + 1] = this.refPos[v * 3 + 1] + w * gain;
            }
          }
        }
      }
    }

    // For morel, tag recessed honeycomb pits as GILLS partId so renderer shades dark umber pits vs lighter ridges
    if (this.species.hymeniumType === 'honeycomb') {
      for (let r = 1; r <= Nr; r++) {
        for (let s = 0; s < Nt; s++) {
          const v = this.capIdx(r, s);
          const rAct = Math.hypot(this.pos[v * 3], this.pos[v * 3 + 2]);
          const rRef = Math.hypot(this.refPos[v * 3], this.refPos[v * 3 + 2]);
          this.outPartIds[v] = rAct < rRef - 4e-5 ? MeshPartId.GILLS : MeshPartId.CAP_TOP;
        }
      }
    }

    // 2. Build active index buffer (cap + hymenium + intact veil triangles)
    let idxPtr = 0;
    for (let i = 0; i < this.capTris.length; i++) {
      this.maxIndicesBuf[idxPtr++] = this.capTris[i];
    }
    for (let i = 0; i < this.hymTris.length; i++) {
      this.maxIndicesBuf[idxPtr++] = this.hymTris[i];
    }
    for (let i = 0; i < this.veilTris.length; i++) {
      const vt = this.veilTris[i];
      if (
        !this.veilEdges[vt.e0].broken &&
        !this.veilEdges[vt.e1].broken &&
        !this.veilEdges[vt.e2].broken
      ) {
        this.maxIndicesBuf[idxPtr++] = vt.v0;
        this.maxIndicesBuf[idxPtr++] = vt.v1;
        this.maxIndicesBuf[idxPtr++] = vt.v2;
      }
    }
    const activeIndexCount = idxPtr;
    this.outIndices = this.maxIndicesBuf.subarray(0, activeIndexCount);

    // 3. Smooth vertex normals via triangle face-normal accumulation
    this.outNormals.fill(0);
    for (let t = 0; t < activeIndexCount; t += 3) {
      const i0 = this.outIndices[t];
      const i1 = this.outIndices[t + 1];
      const i2 = this.outIndices[t + 2];
      const ax = this.outPositions[i1 * 3] - this.outPositions[i0 * 3];
      const ay = this.outPositions[i1 * 3 + 1] - this.outPositions[i0 * 3 + 1];
      const az = this.outPositions[i1 * 3 + 2] - this.outPositions[i0 * 3 + 2];
      const bx = this.outPositions[i2 * 3] - this.outPositions[i0 * 3];
      const by = this.outPositions[i2 * 3 + 1] - this.outPositions[i0 * 3 + 1];
      const bz = this.outPositions[i2 * 3 + 2] - this.outPositions[i0 * 3 + 2];
      const nx = ay * bz - az * by;
      const ny = az * bx - ax * bz;
      const nz = ax * by - ay * bx;

      this.outNormals[i0 * 3] += nx;
      this.outNormals[i0 * 3 + 1] += ny;
      this.outNormals[i0 * 3 + 2] += nz;
      this.outNormals[i1 * 3] += nx;
      this.outNormals[i1 * 3 + 1] += ny;
      this.outNormals[i1 * 3 + 2] += nz;
      this.outNormals[i2 * 3] += nx;
      this.outNormals[i2 * 3 + 1] += ny;
      this.outNormals[i2 * 3 + 2] += nz;
    }

    for (let v = 0; v < nVert; v++) {
      const v3 = v * 3;
      const nx = this.outNormals[v3];
      const ny = this.outNormals[v3 + 1];
      const nz = this.outNormals[v3 + 2];
      const len = Math.hypot(nx, ny, nz);
      if (len > 1e-12) {
        this.outNormals[v3] = nx / len;
        this.outNormals[v3 + 1] = ny / len;
        this.outNormals[v3 + 2] = nz / len;
      } else {
        this.outNormals[v3] = 0;
        this.outNormals[v3 + 1] = 1;
        this.outNormals[v3 + 2] = 0;
      }
    }

    // 4. Compute physical von Mises membrane + bending stress (Pa) per vertex
    this.outStressVM.fill(0);
    const stressCount = new Uint16Array(nVert);
    const Eeff = this.youngsModulusPa / (1.0 - this.poissonRatio * this.poissonRatio);

    for (let i = 0; i < this.edges.length; i++) {
      const e = this.edges[i];
      const len = this.edgeLength(e.a, e.b);
      const strain = Math.abs((len - e.restLen) / e.restLen);
      const sigma = Eeff * strain;
      this.outStressVM[e.a] += sigma;
      this.outStressVM[e.b] += sigma;
      stressCount[e.a]++;
      stressCount[e.b]++;
    }

    const veilE = this.youngsModulusPa * 0.16;
    for (let i = 0; i < this.veilEdges.length; i++) {
      const e = this.veilEdges[i];
      if (e.broken) continue;
      const len = this.edgeLength(e.a, e.b);
      const strain = Math.max(0, (len - e.restLen) / e.restLen);
      const sigma = veilE * strain;
      this.outStressVM[e.a] += sigma;
      this.outStressVM[e.b] += sigma;
      stressCount[e.a]++;
      stressCount[e.b]++;
    }

    for (let v = 0; v < nVert; v++) {
      if (stressCount[v] > 0) {
        this.outStressVM[v] /= stressCount[v];
      }
    }
  }

  public getMeshBuffers(): ShellMeshBuffers {
    return {
      vertexCount: this.totalVertCount,
      positions: this.outPositions,
      normals: this.outNormals,
      stressVM: this.outStressVM,
      partIds: this.outPartIds,
      indices: this.outIndices,
    };
  }
}
