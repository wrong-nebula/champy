// ============================================================================
// SPOROCARP — Frozen Shared Biophysical Types & Buffer Contract
// All modules (soil, mycelium, morpho, spores, render, ui, verify) import here.
// Strict SI units throughout unless explicitly suffixed (e.g. Ppm, C, Deg).
// ============================================================================

export const RHO_WATER = 1000.0; // kg / m^3
export const GRAVITY = 9.80665; // m / s^2
export const PA_PER_M_HEAD = RHO_WATER * GRAVITY; // 9806.65 Pa / m
export const GAMMA_WATER = 0.0728; // N / m (surface tension at 20 C)
export const MU_AIR = 1.81e-5; // Pa * s (air dynamic viscosity at 20 C)
export const RHO_AIR = 1.204; // kg / m^3
export const CP_AIR = 1005.0; // J / (kg * K)
export const LATENT_HEAT_VAP = 2.45e6; // J / kg
export const MOLAR_DENSITY_AIR = 41.57; // mol / m^3 at 20 C, 1 atm
export const AMBIENT_O2_MOL_M3 = 8.71; // 20.95% O2

export const enum SoilMaterialId {
  SAND = 0,
  LOAM = 1,
  CLAY = 2,
  HUMUS = 3,
  WOOD = 4,
  STONE = 5,
}

export interface SoilMaterialParams {
  id: SoilMaterialId;
  name: string;
  thetaR: number; // residual volumetric water content (m^3/m^3)
  thetaS: number; // saturated volumetric water content (m^3/m^3)
  alpha: number; // van Genuchten alpha (1 / m)
  n: number; // van Genuchten n (> 1)
  m: number; // 1 - 1/n
  ks: number; // saturated hydraulic conductivity (m / s)
  initialLignin: number; // insoluble lignocellulose (kg / m^3)
  initialSolubleCarbon: number; // soluble carbon/sugar (kg / m^3)
  thermalDiffusivityDry: number; // m^2 / s
  thermalDiffusivityWet: number; // m^2 / s
  colorDry: [number, number, number]; // linear RGB 0..1
  colorWet: [number, number, number]; // linear RGB 0..1
}

export interface SoilConfig {
  nx: number;
  ny: number;
  nz: number;
  dx: number; // meters per voxel
  ambientTempC: number;
  ambientHumidity: number; // 0..1
  ambientCo2Ppm: number;
  freeDrainBottom: boolean;
}

export type TerrariumScenePreset =
  | 'forest-cross-section'
  | 'fairy-ring-meadow'
  | 'layered-column'
  | 'rotting-log';

export interface EnvironmentState {
  ambientTempC: number; // degrees C (default 15)
  ambientHumidity: number; // 0..1 (default 0.88)
  ambientCo2Ppm: number; // ppm (default 420)
  lightDir: [number, number, number]; // unit vector pointing TOWARD light source
  terrainTiltRad: number; // radians tilt around Z axis (default 0)
  gravityDir: [number, number, number]; // unit gravity vector (default [0, -1, 0], rotated by tilt)
  rainRateMs: number; // continuous rain intensity m/s (default 0)
  evaporationEnabled: boolean; // toggles Dressaire cap evaporative cooling
  timeWarp: number; // 1 .. 100000
}

export interface SolidObstacle {
  id: number;
  kind: 'twig' | 'stone';
  x: number;
  y: number;
  z: number;
  radius: number;
  // For twig capsules (if endX != x):
  endX: number;
  endY: number;
  endZ: number;
}

export type SpeciesId = 'parasol' | 'chanterelle' | 'morel' | 'inkcap' | 'mycena';
export type HymeniumType = 'gills' | 'false-folds' | 'honeycomb' | 'dense-gills';

export interface SpeciesPreset {
  id: SpeciesId;
  name: string;
  scientificName: string;
  description: string;
  // Lockhart stipe & hydraulic parameters
  phiStipe: number; // wall extensibility (1 / (Pa * s))
  yieldPressurePa: number; // turgor yield threshold Y (Pa)
  wiltPressurePa: number; // below this, stipe droops (Pa)
  maxStipeHeightM: number;
  stipeBaseRadiusM: number;
  // Tropisms & CO2 sensitivity
  gravitropismGain: number;
  phototropismGain: number;
  co2KmPpm: number;
  co2StipeBoost: number;
  // Growing thin-shell cap parameters (Koiter / FvK)
  maxCapRadiusM: number;
  alphaRadial: number;
  alphaCircumferential: number;
  shellThicknessM: number;
  youngsModulusPa: number;
  poissonRatio: number;
  coreFoundationStiffness: number; // elastic foundation k_f for gills/morel
  targetCurvature: number; // intrinsic dome curvature (1/m)
  hymeniumType: HymeniumType;
  gillDepthM: number;
  // Partial veil
  hasPartialVeil: boolean;
  veilFractureStressPa: number;
  // Pinning & fruiting thresholds
  pinCarbonThresholdKg: number;
  pinTurgorThresholdPa: number;
  pinHumidityMin: number;
  pinTempMinC: number;
  pinTempMaxC: number;
  pinInhibitionRadiusM: number;
  // Spore & Buller's drop parameters
  sporeRadiusM: number;
  bullerDropRadiusM: number;
  sporeDensityKgM3: number;
  launchEfficiency: number;
  transpirationCoeff: number; // kg / (m^2 * s) at VPD = 1
  // Visual chitin PBR parameters
  capColor: [number, number, number];
  gillColor: [number, number, number];
  stipeColor: [number, number, number];
  sssColor: [number, number, number];
  sporePrintColor: [number, number, number];
  biolumGain?: number;
}

export const SPECIES_PRESETS: Record<SpeciesId, SpeciesPreset> = {
  parasol: {
    id: 'parasol',
    name: 'Meadow Parasol',
    scientificName: 'Agaricus / Macrolepiota',
    description: 'Domed cap with radial gill buckling, secondary lamellulae, ragged partial veil ring, and emergent fairy rings.',
    phiStipe: 1.1e-10,
    yieldPressurePa: 2.2e5,
    wiltPressurePa: 1.5e5,
    maxStipeHeightM: 0.11,
    stipeBaseRadiusM: 0.0075,
    gravitropismGain: 2.4,
    phototropismGain: 0.45,
    co2KmPpm: 1800,
    co2StipeBoost: 0.65,
    maxCapRadiusM: 0.055,
    alphaRadial: 1.0,
    alphaCircumferential: 1.18,
    shellThicknessM: 0.0016,
    youngsModulusPa: 8.0e6,
    poissonRatio: 0.35,
    coreFoundationStiffness: 1.8e5,
    targetCurvature: 16.0,
    hymeniumType: 'gills',
    gillDepthM: 0.0065,
    hasPartialVeil: true,
    veilFractureStressPa: 4.2e5,
    pinCarbonThresholdKg: 1.6e-5,
    pinTurgorThresholdPa: 3.0e5,
    pinHumidityMin: 0.76,
    pinTempMinC: 6,
    pinTempMaxC: 26,
    pinInhibitionRadiusM: 0.055,
    sporeRadiusM: 4.5e-6,
    bullerDropRadiusM: 2.6e-6,
    sporeDensityKgM3: 1150,
    launchEfficiency: 0.22,
    transpirationCoeff: 1.6e-4,
    capColor: [0.88, 0.82, 0.72],
    gillColor: [0.58, 0.42, 0.34],
    stipeColor: [0.92, 0.89, 0.83],
    sssColor: [0.95, 0.68, 0.48],
    sporePrintColor: [0.23, 0.15, 0.11],
  },
  chanterelle: {
    id: 'chanterelle',
    name: 'Golden Chanterelle',
    scientificName: 'Cantharellus cibarius',
    description: 'High circumferential excess growth producing hyperbolic Pringle ruffles and blunt decurrent false-gill ridges.',
    phiStipe: 6.5e-11,
    yieldPressurePa: 2.4e5,
    wiltPressurePa: 1.5e5,
    maxStipeHeightM: 0.068,
    stipeBaseRadiusM: 0.0085,
    gravitropismGain: 1.9,
    phototropismGain: 0.35,
    co2KmPpm: 2000,
    co2StipeBoost: 0.55,
    maxCapRadiusM: 0.048,
    alphaRadial: 0.85,
    alphaCircumferential: 2.8,
    shellThicknessM: 0.0011,
    youngsModulusPa: 5.5e6,
    poissonRatio: 0.38,
    coreFoundationStiffness: 6.0e4,
    targetCurvature: -10.0, // funnel/vase shape!
    hymeniumType: 'false-folds',
    gillDepthM: 0.003,
    hasPartialVeil: false,
    veilFractureStressPa: 0,
    pinCarbonThresholdKg: 1.8e-5,
    pinTurgorThresholdPa: 3.1e5,
    pinHumidityMin: 0.78,
    pinTempMinC: 8,
    pinTempMaxC: 25,
    pinInhibitionRadiusM: 0.05,
    sporeRadiusM: 4.2e-6,
    bullerDropRadiusM: 2.4e-6,
    sporeDensityKgM3: 1150,
    launchEfficiency: 0.20,
    transpirationCoeff: 1.4e-4,
    capColor: [0.92, 0.66, 0.15],
    gillColor: [0.90, 0.62, 0.14],
    stipeColor: [0.93, 0.74, 0.28],
    sssColor: [1.0, 0.72, 0.18],
    sporePrintColor: [0.91, 0.84, 0.64],
  },
  morel: {
    id: 'morel',
    name: 'Woodland Morel',
    scientificName: 'Morchella esculenta',
    description: 'Isotropic compressive thin-shell buckling against an elastic core forming cellular honeycomb ridges and pits.',
    phiStipe: 7.0e-11,
    yieldPressurePa: 2.3e5,
    wiltPressurePa: 1.5e5,
    maxStipeHeightM: 0.082,
    stipeBaseRadiusM: 0.009,
    gravitropismGain: 2.1,
    phototropismGain: 0.3,
    co2KmPpm: 1900,
    co2StipeBoost: 0.5,
    maxCapRadiusM: 0.026,
    alphaRadial: 1.38,
    alphaCircumferential: 1.42,
    shellThicknessM: 0.0009,
    youngsModulusPa: 11.0e6,
    poissonRatio: 0.32,
    coreFoundationStiffness: 5.5e5,
    targetCurvature: 34.0, // ovoid elongated head
    hymeniumType: 'honeycomb',
    gillDepthM: 0.0045,
    hasPartialVeil: false,
    veilFractureStressPa: 0,
    pinCarbonThresholdKg: 1.7e-5,
    pinTurgorThresholdPa: 3.0e5,
    pinHumidityMin: 0.75,
    pinTempMinC: 5,
    pinTempMaxC: 22,
    pinInhibitionRadiusM: 0.048,
    sporeRadiusM: 5.2e-6,
    bullerDropRadiusM: 3.0e-6,
    sporeDensityKgM3: 1150,
    launchEfficiency: 0.25,
    transpirationCoeff: 1.2e-4,
    capColor: [0.46, 0.34, 0.23],
    gillColor: [0.30, 0.21, 0.14],
    stipeColor: [0.89, 0.84, 0.74],
    sssColor: [0.78, 0.54, 0.35],
    sporePrintColor: [0.85, 0.78, 0.62],
  },
  inkcap: {
    id: 'inkcap',
    name: 'Shaggy Ink Cap',
    scientificName: 'Coprinus comatus',
    description: 'Rapid overnight hydraulic elongation, tall campanulate cap with densely packed radial gills, fragile veil, and jet-black spores.',
    phiStipe: 1.8e-10,
    yieldPressurePa: 2.0e5,
    wiltPressurePa: 1.4e5,
    maxStipeHeightM: 0.145,
    stipeBaseRadiusM: 0.0068,
    gravitropismGain: 2.8,
    phototropismGain: 0.55,
    co2KmPpm: 1600,
    co2StipeBoost: 0.75,
    maxCapRadiusM: 0.029,
    alphaRadial: 0.72,
    alphaCircumferential: 1.08,
    shellThicknessM: 0.0012,
    youngsModulusPa: 6.5e6,
    poissonRatio: 0.34,
    coreFoundationStiffness: 3.2e5,
    targetCurvature: 28.0,
    hymeniumType: 'dense-gills',
    gillDepthM: 0.0075,
    hasPartialVeil: true,
    veilFractureStressPa: 2.8e5,
    pinCarbonThresholdKg: 1.5e-5,
    pinTurgorThresholdPa: 2.9e5,
    pinHumidityMin: 0.74,
    pinTempMinC: 6,
    pinTempMaxC: 26,
    pinInhibitionRadiusM: 0.045,
    sporeRadiusM: 5.5e-6,
    bullerDropRadiusM: 3.1e-6,
    sporeDensityKgM3: 1150,
    launchEfficiency: 0.24,
    transpirationCoeff: 1.9e-4,
    capColor: [0.95, 0.93, 0.89],
    gillColor: [0.22, 0.20, 0.22],
    stipeColor: [0.96, 0.95, 0.92],
    sssColor: [0.96, 0.85, 0.76],
    sporePrintColor: [0.08, 0.08, 0.10],
  },
  mycena: {
    id: 'mycena',
    name: 'Glowing Bonnet',
    scientificName: 'Mycena chlorophos',
    description: 'Bioluminescent celadon-fluted bonnet emitting 525 nm 3-hydroxyhispidin photons fueled by oxygen, metabolic carbon, and a 24 h circadian rhythm.',
    phiStipe: 1.35e-10,
    yieldPressurePa: 2.0e5,
    wiltPressurePa: 1.35e5,
    maxStipeHeightM: 0.075,
    stipeBaseRadiusM: 0.0032,
    gravitropismGain: 2.6,
    phototropismGain: 0.55,
    co2KmPpm: 1600,
    co2StipeBoost: 0.55,
    maxCapRadiusM: 0.028,
    alphaRadial: 1.0,
    alphaCircumferential: 1.12,
    shellThicknessM: 0.0010,
    youngsModulusPa: 6.5e6,
    poissonRatio: 0.35,
    coreFoundationStiffness: 1.6e5,
    targetCurvature: 24.0,
    hymeniumType: 'gills',
    gillDepthM: 0.0038,
    hasPartialVeil: false,
    veilFractureStressPa: 4.0e5,
    pinCarbonThresholdKg: 1.1e-5,
    pinTurgorThresholdPa: 2.7e5,
    pinHumidityMin: 0.78,
    pinTempMinC: 8,
    pinTempMaxC: 28,
    pinInhibitionRadiusM: 0.036,
    sporeRadiusM: 4.0e-6,
    bullerDropRadiusM: 2.3e-6,
    sporeDensityKgM3: 1120,
    launchEfficiency: 0.21,
    transpirationCoeff: 1.5e-4,
    capColor: [0.84, 0.91, 0.84],
    gillColor: [0.76, 0.96, 0.80],
    stipeColor: [0.86, 0.92, 0.87],
    sssColor: [0.42, 0.96, 0.56],
    sporePrintColor: [0.88, 0.94, 0.89],
    biolumGain: 1.0,
  },
};

// ----------------------------------------------------------------------------
// Soil Interface
// ----------------------------------------------------------------------------
export interface ISoilGrid {
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly dx: number;
  readonly sizeX: number;
  readonly sizeY: number;
  readonly sizeZ: number;
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

  voxelIndex(ix: number, iy: number, iz: number): number;
  worldToVoxel(x: number, y: number, z: number): [number, number, number];
  voxelToWorld(ix: number, iy: number, iz: number): [number, number, number];
  step(dtSec: number, env: EnvironmentState): void;
  totalWaterVolumeM3(): number;
  pourRain(worldX: number, worldZ: number, radiusM: number, depthM: number): void;
  paintMaterial(
    worldX: number,
    worldY: number,
    worldZ: number,
    radiusM: number,
    mat: SoilMaterialId,
  ): void;
  setupPresetTerrarium(preset: TerrariumScenePreset): void;
  applyHyphalWaterSink(voxelIdx: number, deltaVolumeM3: number): number;
  consumeSolubleCarbon(voxelIdx: number, massKg: number): number;
  secreteEnzyme(voxelIdx: number, rateKgPerM3: number): void;
  applyRespiration(voxelIdx: number, o2ConsumedMol: number, co2ProducedMol: number): void;
  setUniformHydrostaticHead(totalHeadMeters: number): void;
}

// ----------------------------------------------------------------------------
// Mycelium Interface
// ----------------------------------------------------------------------------
export interface MyceliumNode {
  id: number;
  x: number;
  y: number;
  z: number;
  voxelIdx: number;
  turgorPa: number;
  osmoticPa: number;
  carbonKg: number;
  isTip: boolean;
  dirX: number;
  dirY: number;
  dirZ: number;
  ageSec: number;
  hasPrimordium: boolean;
  fruitingBodyId: number; // -1 if none
}

export interface MyceliumEdge {
  id: number;
  a: number;
  b: number;
  lengthM: number;
  radiusM: number;
  conductance: number; // m^3 / (Pa * s)
  fluxM3s: number; // positive a -> b
  isAnastomosis: boolean;
}

export interface PrimordiumAnchor {
  id: number;
  nodeId: number;
  x: number;
  y: number;
  z: number;
  initiatedAtSec: number;
  carbonReserveKg: number;
}

export interface MyceliumRenderBuffers {
  segmentCount: number;
  /** Packed [ax, ay, az, bx, by, bz, radius, turgorNorm] per segment (8 floats) */
  segments: Float32Array;
  nodeCount: number;
  /** Packed [x, y, z, turgorPa] per node (4 floats) */
  nodes: Float32Array;
}

export interface IMyceliumNetwork {
  readonly nodes: MyceliumNode[];
  readonly edges: MyceliumEdge[];
  readonly primordia: PrimordiumAnchor[];
  inoculate(worldX: number, worldY: number, worldZ: number, initialCarbonKg?: number): number;
  step(dtSec: number, env: EnvironmentState, species: SpeciesPreset): void;
  drawWaterAndCarbonForFruiting(
    nodeId: number,
    waterVolumeM3: number,
    carbonKg: number,
  ): { waterSuppliedM3: number; carbonSuppliedKg: number; basalTurgorPa: number };
  getNodeTurgorPa(nodeId: number): number;
  getNodeCarbonKg(nodeId: number): number;
  getAnastomosisCount(): number;
  getPackedRenderBuffers(): MyceliumRenderBuffers;
}

// ----------------------------------------------------------------------------
// Morphogenesis & Thin-Shell Interface
// ----------------------------------------------------------------------------
export interface ThinShellParams {
  species: SpeciesPreset;
  radialRings?: number;
  angularSectors?: number;
  shellThicknessM?: number;
  youngsModulusPa?: number;
  alphaRadial?: number;
  alphaCircumferential?: number;
  coreFoundationStiffness?: number;
}

export const enum MeshPartId {
  STIPE = 0,
  CAP_TOP = 1,
  GILLS = 2,
  VEIL = 3,
}

export interface ShellMeshBuffers {
  vertexCount: number;
  /** Interleaved or separate arrays in local cap/body space */
  positions: Float32Array; // 3 * vertexCount
  normals: Float32Array; // 3 * vertexCount
  stressVM: Float32Array; // von Mises stress (Pa), vertexCount
  partIds: Uint8Array; // MeshPartId per vertex
  indices: Uint32Array;
}

export interface FruitingBodyState {
  id: number;
  primordiumId: number;
  myceliumNodeId: number;
  speciesId: SpeciesId;
  baseX: number;
  baseY: number;
  baseZ: number;
  stipeLengthM: number;
  stipeRadiusM: number;
  stipeTipX: number;
  stipeTipY: number;
  stipeTipZ: number;
  stipeTipDirX: number;
  stipeTipDirY: number;
  stipeTipDirZ: number;
  turgorPa: number;
  wiltFactor: number; // 0 = turgid, 1 = severely wilted
  maturity: number; // 0 = button, 1 = mature sporulating
  capRadiusM: number;
  capAreaM2: number;
  gillWavenumber: number;
  lamellulaeCount: number;
  gillSpacingM: number;
  veilIntactFraction: number; // 1 = intact, 0 = fully torn
  evaporationRateKgS: number; // current transpiration rate
  stipeCenterline: Float32Array; // 3 * numSegments points
}

export interface FruitingRenderMesh {
  vertexCount: number;
  indexCount: number;
  /** World-space positions (3 floats/vert) */
  positions: Float32Array;
  /** World-space normals (3 floats/vert) */
  normals: Float32Array;
  /** Base RGB color (3 floats/vert) */
  colors: Float32Array;
  /** Scalar von Mises stress Pa (1 float/vert) */
  stressVM: Float32Array;
  /** Turgor pressure Pa at vertex (1 float/vert) */
  turgorPa: Float32Array;
  /** Part ID (0=stipe, 1=cap, 2=gills, 3=veil) */
  partIds: Float32Array;
  indices: Uint32Array;
}

// ----------------------------------------------------------------------------
// Spore Physics & Buller's Drop Interface
// ----------------------------------------------------------------------------
export interface BullerParams {
  sporeRadiusM: number;
  bullerDropRadiusM: number;
  sporeDensityKgM3: number;
  launchEfficiency: number;
  gillSpacingM: number;
  gammaWater?: number;
  muAir?: number;
}

export interface BullerLaunchResult {
  deltaSurfaceEnergyJ: number;
  launchKineticEnergyJ: number;
  v0Ms: number;
  peakAccelG: number;
  tauStokesSec: number;
  analyticalStoppingDistM: number;
  terminalVelMs: number;
  clearsOppositeGill: boolean;
}

export interface BullerTrajectoryPoint {
  tSec: number;
  xM: number; // horizontal distance from gill surface
  yM: number; // vertical drop (negative = downward)
  vxMs: number;
  vyMs: number;
}

export interface BullerMicroViewState {
  params: BullerParams;
  launch: BullerLaunchResult;
  trajectory: BullerTrajectoryPoint[];
  phase: number; // 0..1 animation cycle (drop growth -> coalescence -> flight)
  gillSpacingM: number;
  speciesName: string;
}

export interface TerrariumBounds {
  minX: number;
  maxX: number;
  minY: number; // 0 (soil surface)
  maxY: number; // air box ceiling (e.g. 0.36 m)
  minZ: number;
  maxZ: number;
}

export interface AirConvectionField {
  nx: number;
  ny: number;
  nz: number;
  bounds: TerrariumBounds;
  ux: Float32Array;
  uy: Float32Array;
  uz: Float32Array;
  deltaT: Float32Array; // K perturbation (negative = evaporative cooling)
}

export interface SporeParticleBuffers {
  maxCount: number;
  activeCount: number;
  /** [x, y, z, ageSec] per particle (4 floats) */
  positions: Float32Array;
  /** [vx, vy, vz, peakLiftAboveCapM] per particle (4 floats) */
  velocities: Float32Array;
  active: Uint8Array;
}

export interface SporePrintGrid {
  nx: number;
  nz: number;
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  density: Float32Array;
  totalDeposited: number;
  meanDispersalRadiusM: number;
  sporeColor: [number, number, number];
}

// ----------------------------------------------------------------------------
// Renderer & Seeing-Space View State
// ----------------------------------------------------------------------------
export type XRayFieldMode =
  | 'none'
  | 'matric_psi'
  | 'carbon_enzyme'
  | 'turgor_p'
  | 'shell_stress'
  | 'air_convection';

export type BrushTool =
  | 'orbit'
  | 'rain'
  | 'sand'
  | 'loam'
  | 'clay'
  | 'humus'
  | 'wood'
  | 'stone'
  | 'twig'
  | 'inspect_gill';

export interface ViewState {
  cameraYaw: number;
  cameraPitch: number;
  cameraDistance: number;
  cameraTarget: [number, number, number];
  sliceEnabled: boolean;
  sliceAxis: 'x' | 'z';
  slicePosition: number; // -1..1 normalized across terrarium
  xrayMode: XRayFieldMode;
  showStreamlines: boolean;
  showSpores: boolean;
  showMycelium: boolean;
  selectedBodyId: number; // -1 if none
}

export interface TelemetrySample {
  simTimeSec: number;
  meanSoilTheta: number;
  meanSoilPsiPa: number;
  meanSoilO2MolM3: number;
  myceliumNodeCount: number;
  anastomosisCount: number;
  fruitingCount: number;
  leadTurgorPa: number;
  leadYieldPa: number;
  leadHeightM: number;
  leadCapRadiusM: number;
  leadGillWavenumber: number;
  leadVeilIntact: number;
  maxSporeLiftM: number;
  meanSporeDispersalM: number;
  fps: number;
}

export interface BiolumEmitter {
  x: number;
  y: number;
  z: number;
  intensity: number;
  radiusM: number;
  kind: 'body' | 'foxfire';
}

export interface BioluminescenceState {
  enabled: boolean;
  circadianPhase01: number;
  circadianFactor: number;
  tempFactor: number;
  meanO2Factor: number;
  luciferinPoolNorm: number;
  photonFluxCm2S: number;
  peakWavelengthNm: number;
  bodyGlow: Float32Array;
  emitters: BiolumEmitter[];
}

export interface SimulationSnapshot {
  simTimeSec: number;
  species: SpeciesPreset;
  env: EnvironmentState;
  soil: ISoilGrid;
  mycelium: MyceliumRenderBuffers;
  fruitingMesh: FruitingRenderMesh;
  bodies: FruitingBodyState[];
  obstacles: SolidObstacle[];
  airField: AirConvectionField;
  spores: SporeParticleBuffers;
  sporePrint: SporePrintGrid;
  bullerMicro: BullerMicroViewState;
  biolum?: BioluminescenceState;
  telemetry: TelemetrySample;
  telemetryHistory: TelemetrySample[];
}

export interface PickResult {
  hitKind: 'soil' | 'surface' | 'mushroom' | 'none';
  worldX: number;
  worldY: number;
  worldZ: number;
  bodyId: number;
}
