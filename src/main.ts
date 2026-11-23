// ============================================================================
// SPOROCARP — Browser Entrypoint (Coordinator Integration Layer)
// Wires SimulationEngine <-> TerrariumRenderer <-> LabUI + window.__SPOROCARP__
// ============================================================================
import {
  SoilMaterialId,
  type BrushTool,
  type EnvironmentState,
  type PickResult,
  type SimulationSnapshot,
  type SpeciesId,
  type TerrariumScenePreset,
  type ViewState,
} from './types/contract.ts';
import { SimulationEngine } from './sim/engine.ts';
import { TerrariumRenderer } from './render/TerrariumRenderer.ts';
import { LabUI, type LabUICallbacks, type LightingMood } from './ui/LabUI.ts';

interface SporocarpApi {
  getBackend(): 'webgpu' | 'webgl2';
  getSnapshot(): SimulationSnapshot;
  setSpecies(id: SpeciesId): void;
  setScene(preset: TerrariumScenePreset): void;
  setView(partial: Partial<ViewState>): void;
  setEnv(partial: Partial<EnvironmentState>): void;
  setLightingMood(mood: LightingMood): void;
  setFoxfire(enabled: boolean): void;
  stepSim(dtSec: number, steps?: number): void;
  pourRain(x: number, z: number, radius: number, depth: number): void;
  triggerFastForward(simSeconds: number): void;
}

const BRUSH_TO_MATERIAL: Partial<Record<BrushTool, SoilMaterialId>> = {
  sand: SoilMaterialId.SAND,
  loam: SoilMaterialId.LOAM,
  clay: SoilMaterialId.CLAY,
  humus: SoilMaterialId.HUMUS,
  wood: SoilMaterialId.WOOD,
  stone: SoilMaterialId.STONE,
};

async function bootstrap(): Promise<void> {
  const appEl = document.getElementById('app');
  if (!appEl) {
    throw new Error('Missing #app container in index.html');
  }

  const engine = new SimulationEngine('parasol', 'forest-cross-section');
  let renderer: TerrariumRenderer | null = null;
  let lastSnapshot: SimulationSnapshot = engine.getSnapshot();

  const syncUiFromEngine = (): void => {
    const foxfire = (engine as unknown as { foxfireEnabled?: boolean }).foxfireEnabled;
    ui.syncState(engine.species.id, engine.scene, engine.env, engine.paused, foxfire);
  };

  const setEngineFoxfire = (enabled: boolean): void => {
    (engine as unknown as { setFoxfireEnabled?: (v: boolean) => void }).setFoxfireEnabled?.(enabled);
  };

  const callbacks: LabUICallbacks = {
    onSpeciesChange(speciesId: SpeciesId): void {
      engine.setSpecies(speciesId);
      syncUiFromEngine();
      lastSnapshot = engine.getSnapshot();
      ui.update(lastSnapshot, true);
    },
    onSceneChange(preset: TerrariumScenePreset): void {
      engine.setScene(preset);
      syncUiFromEngine();
      lastSnapshot = engine.getSnapshot();
      ui.update(lastSnapshot, true);
    },
    onTimeWarpChange(timeWarp: number): void {
      engine.setEnv({ timeWarp });
    },
    onTogglePause(paused: boolean): void {
      engine.paused = paused;
    },
    onReset(): void {
      engine.reset();
      syncUiFromEngine();
      lastSnapshot = engine.getSnapshot();
      ui.update(lastSnapshot, true);
    },
    onEnvChange(patch: Partial<EnvironmentState>): void {
      engine.setEnv(patch);
    },
    onViewChange(patch: Partial<ViewState>): void {
      if (patch.selectedBodyId !== undefined) {
        engine.selectedBodyId = patch.selectedBodyId;
      }
    },
    onLightingMoodChange(mood: LightingMood): void {
      renderer?.setLightingMood(mood);
    },
    onFoxfireToggle(enabled: boolean): void {
      setEngineFoxfire(enabled);
      lastSnapshot = engine.getSnapshot();
      ui.update(lastSnapshot, true);
    },
    onDrainSoil(): void {
      engine.drainSoil();
    },
    onFloodSoil(): void {
      engine.floodSoil();
    },
    onCanvasPick(clientX: number, clientY: number): PickResult {
      if (!renderer) {
        return { hitKind: 'none', worldX: 0, worldY: 0, worldZ: 0, bodyId: -1 };
      }
      return renderer.pickWorldPoint(clientX, clientY, lastSnapshot, ui.getViewState());
    },
    onBrushApply(tool: BrushTool, pick: PickResult): void {
      if (pick.hitKind === 'none') return;
      if (tool === 'rain') {
        engine.pourRain(pick.worldX, pick.worldZ, 0.065, 0.006);
        return;
      }
      if (tool === 'twig') {
        engine.addTwig(pick.worldX, pick.worldZ);
        return;
      }
      if (tool === 'inspect_gill') {
        if (pick.bodyId >= 0) {
          engine.selectedBodyId = pick.bodyId;
          ui.setViewState({ selectedBodyId: pick.bodyId });
        }
        return;
      }
      const mat = BRUSH_TO_MATERIAL[tool];
      if (mat !== undefined) {
        const wy = Math.min(-0.01, pick.worldY);
        engine.soil.paintMaterial(pick.worldX, wy, pick.worldZ, 0.045, mat);
      }
    },
  };

  const ui = new LabUI(appEl, callbacks);
  syncUiFromEngine();

  renderer = await TerrariumRenderer.create(ui.canvas);
  ui.setRendererBackend(renderer.backend);
  renderer.setLightingMood(ui.getLightingMood());

  const syncCanvasSize = (): void => {
    if (!renderer) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = ui.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width * dpr));
    const h = Math.max(1, Math.floor(rect.height * dpr));
    renderer.resize(w, h);
  };
  syncCanvasSize();
  const resizeObserver = new ResizeObserver(syncCanvasSize);
  resizeObserver.observe(ui.canvas);

  // Brief initial warm-up so the terrarium opens alive with colonizing hyphae & budding primordia
  engine.fastForward(160_000);
  lastSnapshot = engine.getSnapshot();

  // Expose structured inspection & control handle for Tier-3 Playwright verification & console use
  const debugApi: SporocarpApi = {
    getBackend: () => renderer!.backend,
    getSnapshot: () => {
      lastSnapshot = engine.getSnapshot();
      return lastSnapshot;
    },
    setSpecies: (id: SpeciesId) => {
      engine.setSpecies(id);
      syncUiFromEngine();
      lastSnapshot = engine.getSnapshot();
      ui.update(lastSnapshot, true);
    },
    setScene: (preset: TerrariumScenePreset) => {
      engine.setScene(preset);
      syncUiFromEngine();
      lastSnapshot = engine.getSnapshot();
      ui.update(lastSnapshot, true);
    },
    setView: (partial: Partial<ViewState>) => {
      ui.setViewState(partial);
      if (partial.selectedBodyId !== undefined) {
        engine.selectedBodyId = partial.selectedBodyId;
      }
    },
    setEnv: (patch: Partial<EnvironmentState>) => {
      engine.setEnv(patch);
      syncUiFromEngine();
    },
    setLightingMood: (mood: LightingMood) => {
      ui.setLightingMood(mood);
    },
    setFoxfire: (enabled: boolean) => {
      setEngineFoxfire(enabled);
      ui.setFoxfire(enabled);
      lastSnapshot = engine.getSnapshot();
      ui.update(lastSnapshot, true);
    },
    stepSim: (dtSec: number, steps = 1) => {
      engine.stepFixed(dtSec, steps);
      lastSnapshot = engine.getSnapshot();
    },
    pourRain: (x: number, z: number, radius: number, depth: number) => {
      engine.pourRain(x, z, radius, depth);
    },
    triggerFastForward: (simSeconds: number) => {
      engine.fastForward(simSeconds);
      lastSnapshot = engine.getSnapshot();
      renderer?.render(lastSnapshot, ui.getViewState());
      ui.update(lastSnapshot);
    },
  };
  (window as unknown as { __SPOROCARP__?: SporocarpApi }).__SPOROCARP__ = debugApi;

  let prevTime = performance.now();
  let smoothedFps = 60;

  const frame = (now: number): void => {
    const dtReal = Math.min(0.05, Math.max(0.001, (now - prevTime) / 1000));
    prevTime = now;
    const instFps = 1 / dtReal;
    smoothedFps = smoothedFps * 0.9 + instFps * 0.1;
    engine.fps = smoothedFps;

    engine.advanceFrame(dtReal);
    lastSnapshot = engine.getSnapshot();
    renderer!.render(lastSnapshot, ui.getViewState());
    ui.update(lastSnapshot);
    requestAnimationFrame(frame);
  };

  // Render immediate first frame before rAF loop
  renderer.render(lastSnapshot, ui.getViewState());
  ui.update(lastSnapshot);
  requestAnimationFrame(frame);
}

bootstrap().catch((err) => {
  console.error('SPOROCARP initialization failed:', err);
  const appEl = document.getElementById('app');
  if (appEl) {
    appEl.innerHTML = `<pre style="color:#ff6b6b;padding:24px;font-family:monospace">SPOROCARP boot error:\n${String(err)}</pre>`;
  }
});
