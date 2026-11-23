# SPOROCARP — Frozen Module & Buffer Contract (`CONTRACT.md`)

> **FROZEN AT PHASE 1.** All 4 swarm workers build strictly against this contract and `src/types/contract.ts`. Any schema change requires Coordinator sign-off before editing.

---

## 1. Directory Ownership (Strict Separation of Powers)

| Owner | Paths | Responsibilities |
|---|---|---|
| **Coordinator** | `package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `THEORY.md`, `CONTRACT.md`, `src/types/contract.ts`, `src/main.ts`, `src/sim/engine.ts` | Shared types, master simulation loop, integration |
| **Worker 1 (`soil-mycelium`)** | `src/sim/soil/*`, `src/sim/mycelium/*` | van Genuchten–Richards 3D soil grid, Millington–Quirk gas diffusion, Michaelis–Menten enzyme/carbon decay, thermal diffusion, 3D Lockhart mycelial network, anastomosis, Hagen–Poiseuille hydraulics, Boswell translocation, primordium pinning |
| **Worker 2 (`morpho-dispersal`)** | `src/sim/morpho/*`, `src/sim/spores/*` | Coupled hydraulic stipe inflation, gravitropism/phototropism, $\text{CO}_2$ etiolation, Koiter/FvK growing thin-shell cap + gills + lamellulae + chanterelle ruffles + morel core + obstacle contact + partial veil fracture, Buller's drop 1,000× catapult + Stokes drag, Dressaire Boussinesq evaporative spore-wind plume + spore print |
| **Worker 3 (`seeing-space-terrarium`)** | `src/render/*`, `src/ui/*` | WebGPU/WGSL 3D cutaway terrarium renderer (with graceful WebGL2 fallback only if `navigator.gpu` unavailable), soil stratigraphy shader, mycelial network renderer, waxy chitin SSS mushroom & gill renderer, razor slice plane + quantitative colormaps, interactive lab HUD, 1,000× Buller's drop microscope inset canvas, live biophysical telemetry graphs |
| **Worker 4 (`biophysics-auditor`)** | `verify_sporocarp.ts`, `ORACLE.sha256`, `tests/*` | Independent analytical Tier-1 oracles, emergent Tier-2 falsification suite, Playwright Tier-3 live browser visual audit, SHA-256 integrity lock |

---

## 2. Canonical Shared Types (`src/types/contract.ts`)

All data structures, enums, parameter tables, and interfaces are exported from `src/types/contract.ts`. Every worker MUST import types from `../../types/contract.ts` (or `../types/contract.ts`) — never duplicate type definitions.

### 2.1 Soil Layer Contract (`src/sim/soil/SoilGrid.ts`)
Exported class `SoilGrid` implementing `ISoilGrid`:
- Constructor: `new SoilGrid(config?: Partial<SoilConfig>)`
  - Default `SoilConfig`: `{ nx: 48, ny: 32, nz: 48, dx: 0.01, ambientTempC: 15, ambientHumidity: 0.85, ambientCo2Ppm: 420, freeDrainBottom: false }`
- Required flat TypedArrays (length `nx * ny * nz`, indexed `idx = (iy * nz + iz) * nx + ix`):
  - `material: Uint8Array` (`SoilMaterialId`: `0=SAND, 1=LOAM, 2=CLAY, 3=HUMUS, 4=WOOD, 5=STONE`)
  - `theta: Float32Array` (volumetric water content $\theta$, $\text{m}^3/\text{m}^3$)
  - `psi: Float32Array` (matric water potential $\psi \le 0$, $\text{Pa}$)
  - `o2: Float32Array` (pore oxygen concentration, $\text{mol}/\text{m}^3$)
  - `co2: Float32Array` (pore carbon dioxide concentration, $\text{mol}/\text{m}^3$)
  - `lignin: Float32Array` (insoluble lignocellulose $L$, $\text{kg}/\text{m}^3$)
  - `solubleCarbon: Float32Array` (soluble sugar $C_s$, $\text{kg}/\text{m}^3$)
  - `enzyme: Float32Array` (extracellular cellulase/laccase $E$, $\text{kg}/\text{m}^3$)
  - `temperature: Float32Array` (soil temperature $T$, $^\circ\text{C}$)
- Required pure physics helper exports (from `src/sim/soil/vanGenuchten.ts`):
  - `SOIL_MATERIALS: Record<SoilMaterialId, SoilMaterialParams>`
  - `vgEffectiveSaturation(hMeters: number, mat: SoilMaterialParams): number`
  - `vgThetaFromHead(hMeters: number, mat: SoilMaterialParams): number`
  - `vgHeadFromTheta(theta: number, mat: SoilMaterialParams): number`
  - `vgPsiFromTheta(theta: number, mat: SoilMaterialParams): number`
  - `vgConductivity(theta: number, mat: SoilMaterialParams): number`
  - `millingtonQuirkDiffusivity(theta: number, thetaS: number, d0: number): number`
- Required methods on `SoilGrid`:
  - `step(dtSec: number, env: EnvironmentState): void`
  - `worldToVoxel(x: number, y: number, z: number): [number, number, number]`
  - `voxelToWorld(ix: number, iy: number, iz: number): [number, number, number]`
  - `voxelIndex(ix: number, iy: number, iz: number): number`
  - `totalWaterVolumeM3(): number` (exact $\sum \theta_i \Delta x^3$ for Tier-1 mass conservation!)
  - `pourRain(worldX: number, worldZ: number, radiusM: number, depthM: number): void`
  - `paintMaterial(worldX: number, worldY: number, worldZ: number, radiusM: number, mat: SoilMaterialId): void`
  - `setupPresetTerrarium(preset: TerrariumScenePreset): void` (`'forest-cross-section' | 'fairy-ring-meadow' | 'layered-column' | 'rotting-log'`)
  - `applyHyphalWaterSink(voxelIdx: number, deltaVolumeM3: number): number` (returns actual volume extracted, respecting $\theta_r + \varepsilon$)
  - `consumeSolubleCarbon(voxelIdx: number, massKg: number): number`
  - `secreteEnzyme(voxelIdx: number, rateKgPerM3: number): void`
  - `applyRespiration(voxelIdx: number, o2ConsumedMol: number, co2ProducedMol: number): void`
  - `setUniformHydrostaticHead(totalHeadMeters: number): void` (sets $h(y) = H_0 - y$ and corresponding $\theta(h)$)

### 2.2 Mycelium Layer Contract (`src/sim/mycelium/MyceliumNetwork.ts`)
Exported class `MyceliumNetwork` implementing `IMyceliumNetwork`:
- Constructor: `new MyceliumNetwork(soil: ISoilGrid, species: SpeciesPreset)`
- Required state:
  - `nodes: MyceliumNode[]` (each has `id, x, y, z, voxelIdx, turgorPa, osmoticPa, carbonKg, isTip, dirX, dirY, dirZ, ageSec, hasPrimordium, fruitingBodyId`)
  - `edges: MyceliumEdge[]` (each has `id, a, b, lengthM, radiusM, conductance, fluxM3s, isAnastomosis`)
  - `primordia: PrimordiumAnchor[]` (ready/active surface knots with `id, nodeId, x, y, z, initiatedAtSec, carbonReserveKg`)
- Required methods:
  - `inoculate(worldX: number, worldY: number, worldZ: number, initialCarbonKg?: number): number`
  - `step(dtSec: number, env: EnvironmentState, species: SpeciesPreset): void`
  - `drawWaterAndCarbonForFruiting(nodeId: number, waterVolumeM3: number, carbonKg: number): { waterSuppliedM3: number; carbonSuppliedKg: number; basalTurgorPa: number }`
  - `getNodeTurgorPa(nodeId: number): number`
  - `getNodeCarbonKg(nodeId: number): number`
  - `getAnastomosisCount(): number`
  - `getPackedRenderBuffers(): MyceliumRenderBuffers` (`Float32Array` segment endpoints, radii, turgor, carbon)

### 2.3 Morphogenesis Layer Contract (`src/sim/morpho/MorphogenesisSystem.ts` & `ThinShellSolver.ts`)
Exported classes:
1. `ThinShellCapSolver` (standalone testable solver for Tier-1 buckling scaling + live caps):
   - Constructor: `new ThinShellCapSolver(params: ThinShellParams)`
   - `growAndRelax(dtSec: number, turgorPa: number, co2Factor: number, obstacles: SolidObstacle[]): void`
   - `measureCircumferentialWavenumber(): number` (FFT dominant angular buckling mode $m_c$ of rim/hymenium displacement — used by both telemetry and Tier-1 Oracle #3!)
   - `getMeanGillSpacingM(): number` (feeds exact $d_{\text{gill}}$ to Buller's drop micro-solver!)
   - `getLamellulaeCount(): number`
   - `getCapRadiusM(): number`
   - `getVeilIntactFraction(): number`
   - `getMeshBuffers(): ShellMeshBuffers`
2. `FruitingBody` & `MorphogenesisSystem`:
   - `new MorphogenesisSystem(species: SpeciesPreset)`
   - `syncWithPrimordia(mycelium: IMyceliumNetwork, simTimeSec: number): void`
   - `step(dtSec: number, mycelium: IMyceliumNetwork, env: EnvironmentState, obstacles: SolidObstacle[]): void`
   - `stepStipeLockhartExact(lengthM: number, turgorPa: number, phi: number, yieldPa: number, dtSec: number): number` (exported pure helper / method used internally so Tier-1 Oracle #2 tests the exact same integrator!)
   - `getBodies(): FruitingBodyState[]`
   - `getCombinedRenderMesh(): FruitingRenderMesh` (packed cap + gills + stipe + veil vertices, normals, stress $\sigma_{\text{vM}}$, partId, colors, indices)

### 2.4 Spore Physics Layer Contract (`src/sim/spores/SporePhysicsSystem.ts`)
Exported classes & functions:
1. `BullerCatapultSolver` (1,000× micro-scale gill-gap physics):
   - `computeLaunchKinematics(params: BullerParams): BullerLaunchResult` (returns `deltaSurfaceEnergyJ`, `launchKineticEnergyJ`, `v0Ms`, `peakAccelG`, `tauStokesSec`, `analyticalStoppingDistM`, `terminalVelMs`, `clearsOppositeGill`)
   - `simulateTrajectory(params: BullerParams, durationSec: number, dtSec?: number): BullerTrajectoryPoint[]` (exact Stokes drag + gravity trajectory for micro-view and Tier-1 Oracle #4!)
2. `EvaporativePlumeSolver` & `SporePhysicsSystem`:
   - Constructor: `new SporePhysicsSystem(bounds: TerrariumBounds)`
   - `step(dtSec: number, bodies: FruitingBodyState[], env: EnvironmentState, evaporationEnabled?: boolean): void`
   - `getAirField(): AirConvectionField` (3D/2D velocity `(ux, uy, uz)`, temperature perturbation `deltaT`, grid dimensions)
   - `getSpores(): SporeParticleBuffers` (positions, velocities, active flags, peakHeightReached)
   - `getSporePrint(): SporePrintGrid` (`nx × nz` accumulated ground spore density + radial profile)
   - `getMaxSporeLiftAboveCapM(): number` (for Tier-2 Dressaire lift test!)
   - `getActiveBullerState(bodyIndex: number): BullerMicroViewState` (for 1,000× microscope inset!)

### 2.5 Renderer & UI Contract (`src/render/` & `src/ui/`)
- `TerrariumRenderer` (`src/render/TerrariumRenderer.ts`):
  - `static async create(canvas: HTMLCanvasElement): Promise<TerrariumRenderer>` (uses WebGPU/WGSL primary; clean WebGL2 fallback only if `navigator.gpu` is unavailable in headless environments, clearly reporting `backend: 'webgpu' | 'webgl2'`)
  - `render(snapshot: SimulationSnapshot, view: ViewState): void`
  - `pickWorldPoint(clientX: number, clientY: number): PickResult | null` (supports painting soil, pouring rain, placing twigs, and clicking a mature cap to inspect its 1,000× gill gap!)
  - `resize(width: number, height: number): void`
  - `dispose(): void`
- `LabUI` (`src/ui/LabUI.ts`):
  - Mounts full interactive Bret Victor *Seeing Space* bench:
    - Terrarium viewport + orbit/pan/zoom camera + brush cursor
    - Soil/Substrate painter & tool selector (*Rain*, *Sand*, *Loam*, *Clay*, *Oak Log*, *Stone*, *Twig Obstacle*, *Inspect Gill Gap*)
    - Species preset switcher (`parasol`, `chanterelle`, `morel`, `inkcap`)
    - Environment sliders (`Humidity`, `CO2 ppm`, `Temperature`, `Light Angle`, `Terrain Tilt`, `Time Warp 1×..100,000×`, `Evaporation Toggle`)
    - Razor Slice Plane controls (`Slice Axis/Position`, `X-Ray Field`: `none | matric_psi | carbon_enzyme | turgor_p | shell_stress | air_convection`)
    - 1,000× Buller's Drop Gill-Gap Microscope Inset (`MicroscopeCanvas.ts`)
    - Live Biophysical Telemetry Plots (`TelemetryPlots.ts`): $\theta(\psi)$ retention curve, $P(t)$ vs $Y$, $L(t)$, gill wavenumber $m_c$, and spore dispersal radius.
  - Exposes `window.__SPOROCARP__` debug/inspection handle for Playwright Tier-3 automated audit (camera presets, scene switching, slice mode toggling, telemetry readouts, WebGPU backend check).
