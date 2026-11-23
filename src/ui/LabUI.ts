// ============================================================================
// SPOROCARP — Bret Victor "Seeing Space" Interactive Laboratory Bench UI
// Mounts full-bleed 3D WebGPU viewport + top instrument bar + left tool/env
// dock + right Seeing Space panel (razor slice, X-ray colorbar, 1,000x
// Buller's drop microscope inset, and live biophysical telemetry plots).
// ============================================================================

import {
  BrushTool,
  EnvironmentState,
  PickResult,
  SPECIES_PRESETS,
  SimulationSnapshot,
  SpeciesId,
  TerrariumScenePreset,
  ViewState,
  XRayFieldMode,
} from '../types/contract.ts';
import { MicroscopeCanvas } from './MicroscopeCanvas.ts';
import { TelemetryPlots } from './TelemetryPlots.ts';

export type LightingMood = 'studio' | 'canopy' | 'backlit' | 'nocturnal';

export interface LabUICallbacks {
  onSpeciesChange(speciesId: SpeciesId): void;
  onSceneChange(preset: TerrariumScenePreset): void;
  onTimeWarpChange(timeWarp: number): void;
  onTogglePause(paused: boolean): void;
  onReset(): void;
  onEnvChange(patch: Partial<EnvironmentState>): void;
  onViewChange(patch: Partial<ViewState>): void;
  onLightingMoodChange?(mood: LightingMood): void;
  onFoxfireToggle?(enabled: boolean): void;
  onDrainSoil(): void;
  onFloodSoil(): void;
  onCanvasPick(clientX: number, clientY: number): PickResult;
  onBrushApply(tool: BrushTool, pick: PickResult): void;
}

interface ColorbarSpec {
  title: string;
  unit: string;
  gradient: string;
  ticks: [string, string, string];
}

const COLORBAR_SPECS: Record<Exclude<XRayFieldMode, 'none'>, ColorbarSpec> = {
  matric_psi: {
    title: 'Matric Water Potential ψ',
    unit: 'Viridis · kPa',
    gradient: 'linear-gradient(90deg, #440154 0%, #3b528b 25%, #21918c 55%, #5ec962 80%, #fde725 100%)',
    ticks: ['-2000 kPa (Dry)', '-20 kPa', '-0.1 kPa (Sat)'],
  },
  carbon_enzyme: {
    title: 'Soluble Carbon & Enzyme E',
    unit: 'Inferno · kg/m³',
    gradient: 'linear-gradient(90deg, #000004 0%, #420a68 25%, #932667 50%, #dd513a 75%, #fcffa4 100%)',
    ticks: ['0.0 (Depleted)', '1.5', '5.0+ (Rich)'],
  },
  turgor_p: {
    title: 'Internal Turgor Pressure P',
    unit: 'Viridis · kPa',
    gradient: 'linear-gradient(90deg, #440154 0%, #3b528b 25%, #21918c 55%, #5ec962 80%, #fde725 100%)',
    ticks: ['0 kPa (Wilt)', '375 kPa', '750 kPa'],
  },
  shell_stress: {
    title: 'Thin-Shell von Mises Stress σ',
    unit: 'Inferno · kPa',
    gradient: 'linear-gradient(90deg, #000004 0%, #420a68 25%, #932667 50%, #dd513a 75%, #fcffa4 100%)',
    ticks: ['0 kPa', '425 kPa', '850 kPa'],
  },
  air_convection: {
    title: 'Evaporative Air Plume ΔT / u_y',
    unit: 'Coolwarm · K',
    gradient: 'linear-gradient(90deg, #2975e0 0%, #8bc6f8 28%, #d6dee6 50%, #f59e42 75%, #ea5729 100%)',
    ticks: ['-1.2 K (Sink)', '0.0 K', '+0.6 K (Lift)'],
  },
};

const BRUSH_TOOL_DEFS: { id: BrushTool; label: string; swatch: string }[] = [
  { id: 'orbit', label: 'Orbit', swatch: '#38bdf8' },
  { id: 'rain', label: 'Pour Rain', swatch: '#0ea5e9' },
  { id: 'sand', label: 'Sand', swatch: '#d9b578' },
  { id: 'loam', label: 'Loam', swatch: '#7c583b' },
  { id: 'clay', label: 'Clay', swatch: '#a66248' },
  { id: 'humus', label: 'Humus', swatch: '#3f2e21' },
  { id: 'wood', label: 'Oak Log', swatch: '#8b4e27' },
  { id: 'stone', label: 'Stone', swatch: '#7b848e' },
  { id: 'twig', label: 'Place Twig', swatch: '#65442b' },
  { id: 'inspect_gill', label: 'Inspect Gill', swatch: '#34d399' },
];

const XRAY_DEFS: { id: XRayFieldMode; label: string }[] = [
  { id: 'none', label: 'Natural' },
  { id: 'matric_psi', label: 'Matric ψ' },
  { id: 'carbon_enzyme', label: 'Carbon / E' },
  { id: 'turgor_p', label: 'Turgor P' },
  { id: 'shell_stress', label: 'Stress σ' },
  { id: 'air_convection', label: 'Air ΔT' },
];

const LIGHTING_MOOD_DEFS: { id: LightingMood; label: string; title: string }[] = [
  { id: 'studio', label: 'Studio', title: 'Neutral botanical softbox & studio IBL' },
  { id: 'canopy', label: 'Canopy', title: 'Warm dappled woodland sun & drifting foliage gobo' },
  { id: 'backlit', label: 'Backlit', title: 'Dark-field obsidian spotlight & strong subsurface glow' },
  { id: 'nocturnal', label: 'Night', title: 'Nocturnal Bioluminescence Moonlight' },
];

function formatPhotonFlux(flux: number): string {
  if (!Number.isFinite(flux) || flux <= 0) return '0';
  if (flux < 1000) return `${flux.toFixed(0)}`;
  const exp = Math.floor(Math.log10(flux));
  const mant = flux / Math.pow(10, exp);
  return `${mant.toFixed(1)}e${exp}`;
}

function formatSimClock(simSec: number): string {
  if (simSec < 120) return `${simSec.toFixed(1)} s`;
  if (simSec < 7200) {
    const m = Math.floor(simSec / 60);
    const s = Math.floor(simSec % 60);
    return `${m}m ${String(s).padStart(2, '0')}s`;
  }
  const days = Math.floor(simSec / 86400);
  const hours = Math.floor((simSec % 86400) / 3600);
  const mins = Math.floor((simSec % 3600) / 60);
  return `${days}d ${String(hours).padStart(2, '0')}h ${String(mins).padStart(2, '0')}m`;
}

function formatTimeWarp(tw: number): string {
  if (tw >= 1000) return `${(tw / 1000).toFixed(tw >= 10000 ? 0 : 1)}k×`;
  return `${Math.round(tw)}×`;
}

export class LabUI {
  readonly container: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  readonly microscope: MicroscopeCanvas;
  readonly telemetryPlots: TelemetryPlots;

  private callbacks: LabUICallbacks;
  private rootEl: HTMLDivElement;

  private activeTool: BrushTool = 'orbit';
  private paused = false;
  private currentMood: LightingMood = 'studio';
  private currentSpeciesId: SpeciesId = 'parasol';
  private foxfireEnabled = false;
  private viewState: ViewState = {
    cameraYaw: 0.62,
    cameraPitch: 0.42,
    cameraDistance: 0.78,
    cameraTarget: [0, -0.05, 0],
    sliceEnabled: false,
    sliceAxis: 'z',
    slicePosition: 0.0,
    xrayMode: 'none',
    showStreamlines: true,
    showSpores: true,
    showMycelium: true,
    selectedBodyId: -1,
  };

  // Pointer interaction state
  private pointerDown = false;
  private pointerMode: 'orbit' | 'brush' = 'orbit';
  private lastPointerX = 0;
  private lastPointerY = 0;
  private lastBrushTime = 0;

  // DOM handles for crisp zero-rebuilding live updates
  private speciesBtns = new Map<SpeciesId, HTMLButtonElement>();
  private sceneSelect!: HTMLSelectElement;
  private timeWarpSlider!: HTMLInputElement;
  private timeWarpValEl!: HTMLSpanElement;
  private timeWarpPills = new Map<number, HTMLButtonElement>();
  private pauseBtn!: HTMLButtonElement;
  private backendBadgeEl!: HTMLSpanElement;
  private clockEl!: HTMLSpanElement;
  private fpsEl!: HTMLSpanElement;

  private toolBtns = new Map<BrushTool, HTMLButtonElement>();
  private humiditySlider!: HTMLInputElement;
  private humidityValEl!: HTMLSpanElement;
  private co2Slider!: HTMLInputElement;
  private co2ValEl!: HTMLSpanElement;
  private tempSlider!: HTMLInputElement;
  private tempValEl!: HTMLSpanElement;
  private lightAzSlider!: HTMLInputElement;
  private lightAzValEl!: HTMLSpanElement;
  private tiltSlider!: HTMLInputElement;
  private tiltValEl!: HTMLSpanElement;
  private evapCheck!: HTMLInputElement;

  private layerMycBtn!: HTMLButtonElement;
  private layerSporeBtn!: HTMLButtonElement;
  private layerWindBtn!: HTMLButtonElement;
  private layerFoxfireBtn!: HTMLButtonElement;
  private moodBtns = new Map<LightingMood, HTMLButtonElement>();

  private sliceCheck!: HTMLInputElement;
  private sliceAxisXBtn!: HTMLButtonElement;
  private sliceAxisZBtn!: HTMLButtonElement;
  private slicePosSlider!: HTMLInputElement;
  private slicePosValEl!: HTMLSpanElement;

  private xrayBtns = new Map<XRayFieldMode, HTMLButtonElement>();
  private colorbarWrapEl!: HTMLDivElement;
  private colorbarTitleEl!: HTMLSpanElement;
  private colorbarUnitEl!: HTMLSpanElement;
  private colorbarGradEl!: HTMLDivElement;
  private colorbarTickEls: [HTMLSpanElement, HTMLSpanElement, HTMLSpanElement] = [
    document.createElement('span'),
    document.createElement('span'),
    document.createElement('span'),
  ];

  private microHeaderEl!: HTMLSpanElement;
  private kpiNodesEl!: HTMLDivElement;
  private kpiLoopsEl!: HTMLDivElement;
  private kpiCapsEl!: HTMLDivElement;
  private kpiLiftEl!: HTMLDivElement;
  private biolumBarEl!: HTMLDivElement;
  private biolumFluxValEl!: HTMLSpanElement;
  private biolumO2ValEl!: HTMLSpanElement;
  private biolumCircValEl!: HTMLSpanElement;
  private captionEl!: HTMLDivElement;

  private boundListeners: Array<() => void> = [];
  private lastFrameUiTime = 0;

  constructor(container: HTMLElement, callbacks: LabUICallbacks) {
    this.container = container;
    this.callbacks = callbacks;

    this.rootEl = document.createElement('div');
    this.rootEl.className = 'sp-root';
    container.innerHTML = '';
    container.appendChild(this.rootEl);

    // 1. Full-bleed 3D WebGPU viewport canvas
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'sp-viewport';
    this.canvas.id = 'sporocarp-canvas';
    this.rootEl.appendChild(this.canvas);

    // 2. Build instrument panels
    this.buildTopBar();
    this.buildLeftDock();
    const { microCanvas, retCanvas, dynCanvas } = this.buildRightPanel();
    this.buildCaptionPill();

    // 3. Instantiate 2D scientific canvas widgets
    this.microscope = new MicroscopeCanvas(microCanvas);
    this.telemetryPlots = new TelemetryPlots(retCanvas, dynCanvas);

    // 4. Attach pointer/wheel controls
    this.attachPointerControls();
    this.syncViewDOM();
  }

  private addListener<K extends keyof HTMLElementEventMap>(
    el: EventTarget,
    type: K | string,
    fn: EventListenerOrEventListenerObject,
    options?: AddEventListenerOptions,
  ): void {
    el.addEventListener(type, fn, options);
    this.boundListeners.push(() => el.removeEventListener(type, fn, options));
  }

  private buildTopBar(): void {
    const bar = document.createElement('header');
    bar.className = 'sp-topbar sp-glass';

    const brand = document.createElement('div');
    brand.className = 'sp-brand';
    brand.innerHTML = `<span class="sp-brand-title">SPOROCARP</span><span class="sp-brand-sub">Mycology Bench</span>`;
    bar.appendChild(brand);

    // Species preset buttons
    const spGroup = document.createElement('div');
    spGroup.className = 'sp-segment-group';
    const speciesOrder: SpeciesId[] = ['parasol', 'chanterelle', 'morel', 'inkcap', 'mycena'];
    const shortNames: Record<SpeciesId, string> = {
      parasol: 'Parasol',
      chanterelle: 'Chanterelle',
      morel: 'Morel',
      inkcap: 'Ink Cap',
      mycena: 'Mycena',
    };
    for (const sid of speciesOrder) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'sp-btn' + (sid === 'parasol' ? ' active' : '');
      btn.textContent = shortNames[sid];
      btn.title = SPECIES_PRESETS[sid].name;
      btn.dataset.species = sid;
      this.addListener(btn, 'click', () => {
        this.setActiveSpeciesButton(sid);
        this.callbacks.onSpeciesChange(sid);
      });
      this.speciesBtns.set(sid, btn);
      spGroup.appendChild(btn);
    }
    bar.appendChild(spGroup);

    // Scene selector
    this.sceneSelect = document.createElement('select');
    this.sceneSelect.className = 'sp-select';
    this.sceneSelect.title = 'Terrarium Soil Stratigraphy Preset';
    const scenes: Array<[TerrariumScenePreset, string]> = [
      ['forest-cross-section', 'Forest Cross-Section'],
      ['fairy-ring-meadow', 'Fairy Ring Meadow'],
      ['layered-column', 'Layered Soil Column'],
      ['rotting-log', 'Rotting Oak Log'],
    ];
    for (const [val, label] of scenes) {
      const opt = document.createElement('option');
      opt.value = val;
      opt.textContent = label;
      this.sceneSelect.appendChild(opt);
    }
    this.addListener(this.sceneSelect, 'change', () => {
      this.callbacks.onSceneChange(this.sceneSelect.value as TerrariumScenePreset);
    });
    bar.appendChild(this.sceneSelect);

    // Logarithmic Time Warp Scrubber + quick pills
    const twWrap = document.createElement('div');
    twWrap.className = 'sp-timewarp';

    this.timeWarpSlider = document.createElement('input');
    this.timeWarpSlider.type = 'range';
    this.timeWarpSlider.className = 'sp-range';
    this.timeWarpSlider.style.width = '88px';
    this.timeWarpSlider.min = '0';
    this.timeWarpSlider.max = '5';
    this.timeWarpSlider.step = '0.02';
    this.timeWarpSlider.value = '4'; // default 10,000x
    this.timeWarpSlider.title = 'Logarithmic Time Warp (1x .. 100,000x)';

    this.timeWarpValEl = document.createElement('span');
    this.timeWarpValEl.className = 'sp-timewarp-label';
    this.timeWarpValEl.textContent = '10k×';

    const applyWarp = (warp: number, emit = true) => {
      const clamped = Math.max(1, Math.min(100000, warp));
      this.timeWarpSlider.value = String(Math.log10(clamped));
      this.timeWarpValEl.textContent = formatTimeWarp(clamped);
      for (const [v, pill] of this.timeWarpPills) {
        pill.classList.toggle('active', Math.abs(Math.log10(v) - Math.log10(clamped)) < 0.08);
      }
      if (emit) this.callbacks.onTimeWarpChange(clamped);
    };

    this.addListener(this.timeWarpSlider, 'input', () => {
      const val = Math.pow(10, parseFloat(this.timeWarpSlider.value));
      applyWarp(val, true);
    });

    twWrap.appendChild(this.timeWarpValEl);
    twWrap.appendChild(this.timeWarpSlider);

    const pillValues: Array<[number, string]> = [
      [1, '1×'],
      [10000, '10k×'],
      [100000, '100k×'],
    ];
    for (const [val, label] of pillValues) {
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = 'sp-pill' + (val === 10000 ? ' active' : '');
      pill.textContent = label;
      this.addListener(pill, 'click', () => applyWarp(val, true));
      this.timeWarpPills.set(val, pill);
      twWrap.appendChild(pill);
    }
    bar.appendChild(twWrap);

    // Play/Pause & Reset
    this.pauseBtn = document.createElement('button');
    this.pauseBtn.type = 'button';
    this.pauseBtn.className = 'sp-btn sp-btn-outline';
    this.pauseBtn.textContent = 'Pause';
    this.addListener(this.pauseBtn, 'click', () => {
      this.paused = !this.paused;
      this.pauseBtn.textContent = this.paused ? 'Resume' : 'Pause';
      this.pauseBtn.classList.toggle('active', this.paused);
      this.callbacks.onTogglePause(this.paused);
    });
    bar.appendChild(this.pauseBtn);

    const resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'sp-btn sp-btn-outline';
    resetBtn.textContent = 'Reset';
    this.addListener(resetBtn, 'click', () => this.callbacks.onReset());
    bar.appendChild(resetBtn);

    const spacer = document.createElement('div');
    spacer.className = 'sp-spacer';
    bar.appendChild(spacer);

    // Status cluster: Backend badge, Sim Clock, FPS
    const status = document.createElement('div');
    status.className = 'sp-status-cluster';
    this.backendBadgeEl = document.createElement('span');
    this.backendBadgeEl.className = 'sp-badge sp-badge-webgpu';
    this.backendBadgeEl.textContent = 'WebGPU';
    this.clockEl = document.createElement('span');
    this.clockEl.className = 'sp-clock';
    this.clockEl.textContent = '0.0 s';
    this.fpsEl = document.createElement('span');
    this.fpsEl.className = 'sp-fps';
    this.fpsEl.textContent = '60 fps';
    status.appendChild(this.backendBadgeEl);
    status.appendChild(this.clockEl);
    status.appendChild(this.fpsEl);
    bar.appendChild(status);

    this.rootEl.appendChild(bar);
  }

  private buildLeftDock(): void {
    const dock = document.createElement('aside');
    dock.className = 'sp-left-dock sp-glass';

    // Section 1: Brush Tools
    const toolSec = document.createElement('div');
    toolSec.className = 'sp-section';
    toolSec.innerHTML = `<div class="sp-section-header"><span>Bench Tools</span><span>Click / Drag</span></div>`;
    const toolGrid = document.createElement('div');
    toolGrid.className = 'sp-tool-grid';
    for (const def of BRUSH_TOOL_DEFS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'sp-tool-btn' + (def.id === this.activeTool ? ' active' : '');
      btn.dataset.tool = def.id;
      btn.innerHTML = `<span class="sp-swatch" style="background:${def.swatch}"></span><span>${def.label}</span>`;
      this.addListener(btn, 'click', () => this.setBrushTool(def.id));
      this.toolBtns.set(def.id, btn);
      toolGrid.appendChild(btn);
    }
    toolSec.appendChild(toolGrid);
    dock.appendChild(toolSec);

    // Section 2: Environment Controls
    const envSec = document.createElement('div');
    envSec.className = 'sp-section';
    envSec.innerHTML = `<div class="sp-section-header"><span>Environment</span><span>SI Physics</span></div>`;

    const mkSlider = (
      label: string,
      min: number,
      max: number,
      step: number,
      val: number,
      fmt: (v: number) => string,
      onChange: (v: number) => void,
    ): [HTMLInputElement, HTMLSpanElement] => {
      const row = document.createElement('div');
      row.className = 'sp-slider-row';
      const meta = document.createElement('div');
      meta.className = 'sp-slider-meta';
      const nameEl = document.createElement('span');
      nameEl.className = 'sp-slider-name';
      nameEl.textContent = label;
      const valEl = document.createElement('span');
      valEl.className = 'sp-slider-val';
      valEl.textContent = fmt(val);
      meta.appendChild(nameEl);
      meta.appendChild(valEl);

      const input = document.createElement('input');
      input.type = 'range';
      input.className = 'sp-range';
      input.min = String(min);
      input.max = String(max);
      input.step = String(step);
      input.value = String(val);
      this.addListener(input, 'input', () => {
        const num = parseFloat(input.value);
        valEl.textContent = fmt(num);
        onChange(num);
      });

      row.appendChild(meta);
      row.appendChild(input);
      envSec.appendChild(row);
      return [input, valEl];
    };

    [this.humiditySlider, this.humidityValEl] = mkSlider(
      'Humidity (RH)', 30, 100, 1, 88, (v) => `${Math.round(v)}%`,
      (v) => this.callbacks.onEnvChange({ ambientHumidity: v / 100 }),
    );
    [this.co2Slider, this.co2ValEl] = mkSlider(
      'Ambient CO₂', 300, 3500, 20, 420, (v) => `${Math.round(v)} ppm`,
      (v) => this.callbacks.onEnvChange({ ambientCo2Ppm: v }),
    );
    [this.tempSlider, this.tempValEl] = mkSlider(
      'Temperature', 2, 30, 0.5, 15, (v) => `${v.toFixed(1)} °C`,
      (v) => this.callbacks.onEnvChange({ ambientTempC: v }),
    );
    [this.lightAzSlider, this.lightAzValEl] = mkSlider(
      'Light Azimuth', -180, 180, 2, 35, (v) => `${Math.round(v)}°`,
      (v) => {
        const az = (v * Math.PI) / 180;
        const el = (52 * Math.PI) / 180;
        const cosE = Math.cos(el);
        this.callbacks.onEnvChange({
          lightDir: [cosE * Math.sin(az), Math.sin(el), cosE * Math.cos(az)],
        });
      },
    );
    [this.tiltSlider, this.tiltValEl] = mkSlider(
      'Terrain Tilt', -45, 45, 1, 0, (v) => `${Math.round(v)}°`,
      (v) => {
        const rad = (v * Math.PI) / 180;
        this.callbacks.onEnvChange({
          terrainTiltRad: rad,
          gravityDir: [Math.sin(rad), -Math.cos(rad), 0],
        });
      },
    );

    // Evaporative Wind Toggle
    const evapRow = document.createElement('label');
    evapRow.className = 'sp-toggle-row';
    evapRow.innerHTML = `<span>Evaporative Wind (Dressaire)</span>`;
    this.evapCheck = document.createElement('input');
    this.evapCheck.type = 'checkbox';
    this.evapCheck.className = 'sp-checkbox';
    this.evapCheck.checked = true;
    this.addListener(this.evapCheck, 'change', () => {
      this.callbacks.onEnvChange({ evaporationEnabled: this.evapCheck.checked });
    });
    evapRow.appendChild(this.evapCheck);
    envSec.appendChild(evapRow);

    // Quick Hydrology Perturbation Buttons (Drought / Hypoxia tests)
    const hydroActions = document.createElement('div');
    hydroActions.className = 'sp-action-row';
    const drainBtn = document.createElement('button');
    drainBtn.type = 'button';
    drainBtn.className = 'sp-btn sp-btn-outline sp-btn-danger';
    drainBtn.textContent = 'Drain (Drought)';
    this.addListener(drainBtn, 'click', () => this.callbacks.onDrainSoil());
    const floodBtn = document.createElement('button');
    floodBtn.type = 'button';
    floodBtn.className = 'sp-btn sp-btn-outline sp-btn-water';
    floodBtn.textContent = 'Flood (Hypoxia)';
    this.addListener(floodBtn, 'click', () => this.callbacks.onFloodSoil());
    hydroActions.appendChild(drainBtn);
    hydroActions.appendChild(floodBtn);
    envSec.appendChild(hydroActions);
    dock.appendChild(envSec);

    // Section 3: Visual Layers
    const layerSec = document.createElement('div');
    layerSec.className = 'sp-section';
    layerSec.innerHTML = `<div class="sp-section-header"><span>Layers</span><span>Toggle</span></div>`;
    const pills = document.createElement('div');
    pills.className = 'sp-layer-pills';

    const mkLayer = (label: string, layerId: string, initial: boolean, onToggle: (next: boolean) => void) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'sp-btn' + (initial ? ' active' : '');
      b.textContent = label;
      b.dataset.layer = layerId;
      b.setAttribute('aria-pressed', String(initial));
      this.addListener(b, 'click', () => {
        const next = !b.classList.contains('active');
        b.classList.toggle('active', next);
        b.setAttribute('aria-pressed', String(next));
        onToggle(next);
      });
      pills.appendChild(b);
      return b;
    };

    this.layerMycBtn = mkLayer('Mycelium', 'mycelium', true, (v) => this.emitViewPatch({ showMycelium: v }));
    this.layerSporeBtn = mkLayer('Spores', 'spores', true, (v) => this.emitViewPatch({ showSpores: v }));
    this.layerWindBtn = mkLayer('Plume', 'plume', true, (v) => this.emitViewPatch({ showStreamlines: v }));
    this.layerFoxfireBtn = mkLayer('Foxfire', 'foxfire', false, (v) => {
      this.foxfireEnabled = v;
      this.syncBiolumToggleDOM();
      this.callbacks.onFoxfireToggle?.(v);
    });
    this.layerFoxfireBtn.title = 'Toggle 525 nm Foxfire Bioluminescence';
    layerSec.appendChild(pills);
    dock.appendChild(layerSec);

    // Section 4: Lighting Mood Presets
    const moodSec = document.createElement('div');
    moodSec.className = 'sp-section';
    moodSec.innerHTML = `<div class="sp-section-header"><span>Lighting Mood</span><span>Preset</span></div>`;
    const moodGroup = document.createElement('div');
    moodGroup.className = 'sp-mood-segment';
    moodGroup.setAttribute('role', 'group');
    moodGroup.setAttribute('aria-label', 'Lighting Mood');
    for (const def of LIGHTING_MOOD_DEFS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      const active = def.id === this.currentMood;
      btn.className = 'sp-btn sp-mood-btn' + (active ? ' active' : '');
      btn.textContent = def.label;
      btn.title = def.title;
      btn.dataset.mood = def.id;
      btn.setAttribute('aria-pressed', String(active));
      this.addListener(btn, 'click', () => this.setLightingMood(def.id));
      this.moodBtns.set(def.id, btn);
      moodGroup.appendChild(btn);
    }
    moodSec.appendChild(moodGroup);
    dock.appendChild(moodSec);

    this.rootEl.appendChild(dock);
  }

  private buildRightPanel(): {
    microCanvas: HTMLCanvasElement;
    retCanvas: HTMLCanvasElement;
    dynCanvas: HTMLCanvasElement;
  } {
    const panel = document.createElement('aside');
    panel.className = 'sp-right-panel sp-glass';

    // 1. Razor Slice Plane Controls
    const sliceSec = document.createElement('div');
    sliceSec.className = 'sp-section';
    sliceSec.innerHTML = `<div class="sp-section-header"><span>Razor Slice &amp; X-Ray</span><span>Seeing Space</span></div>`;

    const sliceTopRow = document.createElement('div');
    sliceTopRow.className = 'sp-toggle-row';
    const sliceLbl = document.createElement('label');
    sliceLbl.style.display = 'flex';
    sliceLbl.style.alignItems = 'center';
    sliceLbl.style.gap = '6px';
    sliceLbl.style.cursor = 'pointer';
    this.sliceCheck = document.createElement('input');
    this.sliceCheck.type = 'checkbox';
    this.sliceCheck.className = 'sp-checkbox';
    this.addListener(this.sliceCheck, 'change', () => {
      this.emitViewPatch({ sliceEnabled: this.sliceCheck.checked });
    });
    sliceLbl.appendChild(this.sliceCheck);
    const sliceTxt = document.createElement('span');
    sliceTxt.textContent = 'Razor Cutaway Plane';
    sliceLbl.appendChild(sliceTxt);
    sliceTopRow.appendChild(sliceLbl);

    const axisGroup = document.createElement('div');
    axisGroup.className = 'sp-segment-group';
    this.sliceAxisXBtn = document.createElement('button');
    this.sliceAxisXBtn.type = 'button';
    this.sliceAxisXBtn.className = 'sp-btn';
    this.sliceAxisXBtn.textContent = 'X';
    this.sliceAxisZBtn = document.createElement('button');
    this.sliceAxisZBtn.type = 'button';
    this.sliceAxisZBtn.className = 'sp-btn active';
    this.sliceAxisZBtn.textContent = 'Z';
    this.addListener(this.sliceAxisXBtn, 'click', () => this.emitViewPatch({ sliceAxis: 'x', sliceEnabled: true }));
    this.addListener(this.sliceAxisZBtn, 'click', () => this.emitViewPatch({ sliceAxis: 'z', sliceEnabled: true }));
    axisGroup.appendChild(this.sliceAxisXBtn);
    axisGroup.appendChild(this.sliceAxisZBtn);
    sliceTopRow.appendChild(axisGroup);
    sliceSec.appendChild(sliceTopRow);

    // Slice Position Slider
    const sRow = document.createElement('div');
    sRow.className = 'sp-slider-row';
    const sMeta = document.createElement('div');
    sMeta.className = 'sp-slider-meta';
    const sName = document.createElement('span');
    sName.className = 'sp-slider-name';
    sName.textContent = 'Slice Position';
    this.slicePosValEl = document.createElement('span');
    this.slicePosValEl.className = 'sp-slider-val';
    this.slicePosValEl.textContent = '0 mm';
    sMeta.appendChild(sName);
    sMeta.appendChild(this.slicePosValEl);
    this.slicePosSlider = document.createElement('input');
    this.slicePosSlider.type = 'range';
    this.slicePosSlider.className = 'sp-range';
    this.slicePosSlider.min = '-0.95';
    this.slicePosSlider.max = '0.95';
    this.slicePosSlider.step = '0.01';
    this.slicePosSlider.value = '0';
    this.addListener(this.slicePosSlider, 'input', () => {
      const pos = parseFloat(this.slicePosSlider.value);
      this.emitViewPatch({ slicePosition: pos, sliceEnabled: true });
    });
    sRow.appendChild(sMeta);
    sRow.appendChild(this.slicePosSlider);
    sliceSec.appendChild(sRow);

    // X-Ray Field Grid
    const xrayGrid = document.createElement('div');
    xrayGrid.className = 'sp-xray-grid';
    for (const def of XRAY_DEFS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'sp-btn' + (def.id === 'none' ? ' active' : '');
      btn.textContent = def.label;
      btn.dataset.xray = def.id;
      this.addListener(btn, 'click', () => this.emitViewPatch({ xrayMode: def.id }));
      this.xrayBtns.set(def.id, btn);
      xrayGrid.appendChild(btn);
    }
    sliceSec.appendChild(xrayGrid);

    // Calibrated SI Colorbar Legend
    this.colorbarWrapEl = document.createElement('div');
    this.colorbarWrapEl.className = 'sp-colorbar-wrap hidden';
    const cbHead = document.createElement('div');
    cbHead.className = 'sp-colorbar-title';
    this.colorbarTitleEl = document.createElement('span');
    this.colorbarUnitEl = document.createElement('span');
    cbHead.appendChild(this.colorbarTitleEl);
    cbHead.appendChild(this.colorbarUnitEl);
    this.colorbarGradEl = document.createElement('div');
    this.colorbarGradEl.className = 'sp-colorbar-grad';
    const cbTicks = document.createElement('div');
    cbTicks.className = 'sp-colorbar-ticks';
    this.colorbarTickEls = [
      document.createElement('span'),
      document.createElement('span'),
      document.createElement('span'),
    ];
    for (const t of this.colorbarTickEls) cbTicks.appendChild(t);
    this.colorbarWrapEl.appendChild(cbHead);
    this.colorbarWrapEl.appendChild(this.colorbarGradEl);
    this.colorbarWrapEl.appendChild(cbTicks);
    sliceSec.appendChild(this.colorbarWrapEl);
    panel.appendChild(sliceSec);

    // 2. 1,000x Buller's Drop Gill-Gap Microscope Inset
    const microSec = document.createElement('div');
    microSec.className = 'sp-section';
    const microHead = document.createElement('div');
    microHead.className = 'sp-section-header';
    const microTitle = document.createElement('span');
    microTitle.textContent = '1,000× Gill-Gap Microscope';
    this.microHeaderEl = document.createElement('span');
    this.microHeaderEl.textContent = "Buller's Drop";
    microHead.appendChild(microTitle);
    microHead.appendChild(this.microHeaderEl);
    microSec.appendChild(microHead);

    const microCard = document.createElement('div');
    microCard.className = 'sp-canvas-card';
    const microCanvas = document.createElement('canvas');
    microCanvas.className = 'sp-microscope-canvas';
    microCanvas.id = 'sporocarp-microscope';
    microCard.appendChild(microCanvas);
    microSec.appendChild(microCard);
    panel.appendChild(microSec);

    // 3. Live Biophysical Telemetry Plots & KPIs
    const telSec = document.createElement('div');
    telSec.className = 'sp-section';
    telSec.innerHTML = `<div class="sp-section-header"><span>Live Biophysical Telemetry</span><span>SI</span></div>`;

    const kpiGrid = document.createElement('div');
    kpiGrid.className = 'sp-kpi-grid';
    const mkKpi = (label: string): HTMLDivElement => {
      const box = document.createElement('div');
      box.className = 'sp-kpi';
      const l = document.createElement('div');
      l.className = 'sp-kpi-label';
      l.textContent = label;
      const v = document.createElement('div');
      v.className = 'sp-kpi-val';
      v.textContent = '0';
      box.appendChild(l);
      box.appendChild(v);
      kpiGrid.appendChild(box);
      return v;
    };
    this.kpiNodesEl = mkKpi('Hyphae');
    this.kpiLoopsEl = mkKpi('Loops');
    this.kpiCapsEl = mkKpi('Bodies');
    this.kpiLiftEl = mkKpi('Lift');
    telSec.appendChild(kpiGrid);

    // 525 nm Bioluminescence telemetry readout strip (revealed when active)
    this.biolumBarEl = document.createElement('div');
    this.biolumBarEl.className = 'sp-biolum-bar hidden';
    const bioHead = document.createElement('div');
    bioHead.className = 'sp-biolum-head';
    bioHead.innerHTML = `<span class="sp-biolum-title"><span class="sp-biolum-dot"></span>525 nm Bioluminescence</span><span>LH₂ · O₂</span>`;
    const bioRow = document.createElement('div');
    bioRow.className = 'sp-biolum-row';

    const mkBioMetric = (prefix: string, init: string): HTMLSpanElement => {
      const wrap = document.createElement('span');
      wrap.className = 'sp-biolum-metric';
      const lbl = document.createElement('b');
      lbl.textContent = prefix + ' ';
      const val = document.createElement('span');
      val.textContent = init;
      wrap.appendChild(lbl);
      wrap.appendChild(val);
      bioRow.appendChild(wrap);
      return val;
    };
    this.biolumFluxValEl = mkBioMetric('FLUX', '0 ph/cm²·s');
    this.biolumO2ValEl = mkBioMetric('O₂', '0%');
    this.biolumCircValEl = mkBioMetric('CIRCADIAN', '0% · Night');
    this.biolumBarEl.appendChild(bioHead);
    this.biolumBarEl.appendChild(bioRow);
    telSec.appendChild(this.biolumBarEl);

    const plotCard = document.createElement('div');
    plotCard.className = 'sp-canvas-card';
    const retCanvas = document.createElement('canvas');
    retCanvas.className = 'sp-retention-canvas';
    const dynCanvas = document.createElement('canvas');
    dynCanvas.className = 'sp-dynamics-canvas';
    plotCard.appendChild(retCanvas);
    plotCard.appendChild(dynCanvas);
    telSec.appendChild(plotCard);
    panel.appendChild(telSec);

    this.rootEl.appendChild(panel);
    return { microCanvas, retCanvas, dynCanvas };
  }

  private buildCaptionPill(): void {
    this.captionEl = document.createElement('div');
    this.captionEl.className = 'sp-species-caption sp-glass';
    const def = SPECIES_PRESETS.parasol;
    this.captionEl.innerHTML = `<strong>${def.name}</strong> <em>${def.scientificName}</em>`;
    this.rootEl.appendChild(this.captionEl);
  }

  private attachPointerControls(): void {
    this.addListener(this.canvas, 'contextmenu', (e) => e.preventDefault());

    this.addListener(this.canvas, 'pointerdown', (e) => {
      const pe = e as PointerEvent;
      this.canvas.setPointerCapture(pe.pointerId);
      this.pointerDown = true;
      this.lastPointerX = pe.clientX;
      this.lastPointerY = pe.clientY;

      const forceOrbit = pe.button === 2 || pe.shiftKey || this.activeTool === 'orbit';
      if (forceOrbit) {
        this.pointerMode = 'orbit';
        this.canvas.classList.add('dragging');
      } else {
        this.pointerMode = 'brush';
        this.applyBrushAt(pe.clientX, pe.clientY);
      }
    });

    this.addListener(this.canvas, 'pointermove', (e) => {
      const pe = e as PointerEvent;
      if (!this.pointerDown) return;
      const dx = pe.clientX - this.lastPointerX;
      const dy = pe.clientY - this.lastPointerY;
      this.lastPointerX = pe.clientX;
      this.lastPointerY = pe.clientY;

      if (this.pointerMode === 'orbit') {
        const nextYaw = this.viewState.cameraYaw - dx * 0.0075;
        const nextPitch = Math.max(-0.55, Math.min(1.42, this.viewState.cameraPitch + dy * 0.0075));
        this.emitViewPatch({ cameraYaw: nextYaw, cameraPitch: nextPitch });
      } else if (this.activeTool !== 'twig' && this.activeTool !== 'inspect_gill') {
        const now = performance.now();
        if (now - this.lastBrushTime > 40) {
          this.lastBrushTime = now;
          this.applyBrushAt(pe.clientX, pe.clientY);
        }
      }
    });

    const endPointer = (e: Event) => {
      const pe = e as PointerEvent;
      if (this.pointerDown) {
        this.pointerDown = false;
        this.canvas.classList.remove('dragging');
        if (this.canvas.hasPointerCapture(pe.pointerId)) {
          this.canvas.releasePointerCapture(pe.pointerId);
        }
      }
    };
    this.addListener(this.canvas, 'pointerup', endPointer);
    this.addListener(this.canvas, 'pointercancel', endPointer);

    this.addListener(
      this.canvas,
      'wheel',
      (e) => {
        const we = e as WheelEvent;
        we.preventDefault();
        const factor = Math.exp(we.deltaY * 0.0012);
        const nextDist = Math.max(0.28, Math.min(1.65, this.viewState.cameraDistance * factor));
        this.emitViewPatch({ cameraDistance: nextDist });
      },
      { passive: false },
    );
  }

  private applyBrushAt(clientX: number, clientY: number): void {
    const pick = this.callbacks.onCanvasPick(clientX, clientY);
    if (pick.hitKind === 'none') return;
    if (this.activeTool === 'inspect_gill' || pick.hitKind === 'mushroom') {
      if (pick.bodyId >= 0) {
        this.emitViewPatch({ selectedBodyId: pick.bodyId });
      }
      if (this.activeTool === 'inspect_gill') {
        this.callbacks.onBrushApply('inspect_gill', pick);
        return;
      }
    }
    this.callbacks.onBrushApply(this.activeTool, pick);
  }

  private emitViewPatch(patch: Partial<ViewState>): void {
    this.viewState = { ...this.viewState, ...patch };
    this.syncViewDOM();
    this.callbacks.onViewChange(patch);
  }

  private setActiveSpeciesButton(speciesId: SpeciesId): void {
    this.currentSpeciesId = speciesId;
    for (const [id, btn] of this.speciesBtns) {
      btn.classList.toggle('active', id === speciesId);
    }
    const preset = SPECIES_PRESETS[speciesId];
    if (preset && this.captionEl) {
      this.captionEl.innerHTML = `<strong>${preset.name}</strong> <em>${preset.scientificName}</em>`;
    }
    this.syncBiolumToggleDOM();
  }

  setBrushTool(tool: BrushTool): void {
    this.activeTool = tool;
    for (const [id, btn] of this.toolBtns) {
      btn.classList.toggle('active', id === tool);
    }
    this.canvas.classList.toggle('tool-active', tool !== 'orbit');
  }

  setRendererBackend(backend: 'webgpu' | 'webgl2'): void {
    this.backendBadgeEl.textContent = backend === 'webgpu' ? 'WebGPU' : 'WebGL2';
    this.backendBadgeEl.className =
      backend === 'webgpu' ? 'sp-badge sp-badge-webgpu' : 'sp-badge sp-badge-webgl2';
  }

  getViewState(): ViewState {
    return { ...this.viewState, cameraTarget: [...this.viewState.cameraTarget] };
  }

  setViewState(patch: Partial<ViewState>): void {
    this.viewState = { ...this.viewState, ...patch };
    this.syncViewDOM();
  }

  getLightingMood(): LightingMood {
    return this.currentMood;
  }

  setLightingMood(mood: LightingMood): void {
    if (mood !== 'studio' && mood !== 'canopy' && mood !== 'backlit' && mood !== 'nocturnal') {
      return;
    }
    this.currentMood = mood;
    this.syncMoodDOM();
    this.callbacks.onLightingMoodChange?.(mood);
  }

  setFoxfire(enabled: boolean): void {
    this.foxfireEnabled = enabled;
    this.syncBiolumToggleDOM();
  }

  private syncBiolumToggleDOM(snapshotActive?: boolean): void {
    if (!this.layerFoxfireBtn) return;
    const active =
      typeof snapshotActive === 'boolean'
        ? snapshotActive
        : this.foxfireEnabled || this.currentSpeciesId === 'mycena';
    this.layerFoxfireBtn.classList.toggle('active', active);
    this.layerFoxfireBtn.setAttribute('aria-pressed', String(active));
  }

  private syncMoodDOM(): void {
    for (const [id, btn] of this.moodBtns) {
      const active = id === this.currentMood;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', String(active));
    }
  }

  syncState(
    speciesId: SpeciesId,
    scene: TerrariumScenePreset,
    env: EnvironmentState,
    paused?: boolean,
    foxfireEnabled?: boolean,
  ): void {
    if (typeof foxfireEnabled === 'boolean') {
      this.foxfireEnabled = foxfireEnabled;
    }
    this.setActiveSpeciesButton(speciesId);
    this.sceneSelect.value = scene;

    const tw = Math.max(1, Math.min(100000, env.timeWarp));
    this.timeWarpSlider.value = String(Math.log10(tw));
    this.timeWarpValEl.textContent = formatTimeWarp(tw);
    for (const [v, pill] of this.timeWarpPills) {
      pill.classList.toggle('active', Math.abs(Math.log10(v) - Math.log10(tw)) < 0.08);
    }

    const rhPct = Math.round(env.ambientHumidity * 100);
    this.humiditySlider.value = String(rhPct);
    this.humidityValEl.textContent = `${rhPct}%`;

    this.co2Slider.value = String(Math.round(env.ambientCo2Ppm));
    this.co2ValEl.textContent = `${Math.round(env.ambientCo2Ppm)} ppm`;

    this.tempSlider.value = String(env.ambientTempC);
    this.tempValEl.textContent = `${env.ambientTempC.toFixed(1)} °C`;

    const tiltDeg = Math.round((env.terrainTiltRad * 180) / Math.PI);
    this.tiltSlider.value = String(tiltDeg);
    this.tiltValEl.textContent = `${tiltDeg}°`;

    const azDeg = Math.round((Math.atan2(env.lightDir[0], env.lightDir[2]) * 180) / Math.PI);
    this.lightAzSlider.value = String(azDeg);
    this.lightAzValEl.textContent = `${azDeg}°`;

    this.evapCheck.checked = env.evaporationEnabled;

    if (typeof paused === 'boolean') {
      this.paused = paused;
      this.pauseBtn.textContent = paused ? 'Resume' : 'Pause';
      this.pauseBtn.classList.toggle('active', paused);
    }
  }

  private syncViewDOM(): void {
    const v = this.viewState;
    this.sliceCheck.checked = v.sliceEnabled;
    this.sliceAxisXBtn.classList.toggle('active', v.sliceAxis === 'x');
    this.sliceAxisZBtn.classList.toggle('active', v.sliceAxis === 'z');
    this.slicePosSlider.value = String(v.slicePosition);
    this.slicePosValEl.textContent = `${Math.round(v.slicePosition * 240)} mm`;

    this.layerMycBtn.classList.toggle('active', v.showMycelium);
    this.layerSporeBtn.classList.toggle('active', v.showSpores);
    this.layerWindBtn.classList.toggle('active', v.showStreamlines);
    this.syncBiolumToggleDOM();
    this.syncMoodDOM();

    for (const [id, btn] of this.xrayBtns) {
      btn.classList.toggle('active', id === v.xrayMode);
    }

    if (v.xrayMode === 'none') {
      this.colorbarWrapEl.classList.add('hidden');
    } else {
      const spec = COLORBAR_SPECS[v.xrayMode];
      this.colorbarWrapEl.classList.remove('hidden');
      this.colorbarTitleEl.textContent = spec.title;
      this.colorbarUnitEl.textContent = spec.unit;
      this.colorbarGradEl.style.background = spec.gradient;
      this.colorbarTickEls[0].textContent = spec.ticks[0];
      this.colorbarTickEls[1].textContent = spec.ticks[1];
      this.colorbarTickEls[2].textContent = spec.ticks[2];
    }
  }

  update(snapshot: SimulationSnapshot, force = false): void {
    // Always animate the 1,000x Buller's Drop microscope smoothly at 60 FPS
    this.microscope.render(snapshot.bullerMicro);

    // Update live text/plots at ~15 Hz to keep DOM layout cost negligible while remaining crisp
    const now = performance.now();
    if (!force && now - this.lastFrameUiTime < 60 && this.lastFrameUiTime > 0) {
      return;
    }
    this.lastFrameUiTime = now;

    const tel = snapshot.telemetry;
    this.clockEl.textContent = formatSimClock(snapshot.simTimeSec);
    this.fpsEl.textContent = `${Math.round(tel.fps)} fps`;

    this.kpiNodesEl.textContent = String(tel.myceliumNodeCount);
    this.kpiLoopsEl.textContent = String(tel.anastomosisCount);
    this.kpiCapsEl.textContent = String(tel.fruitingCount);
    this.kpiLiftEl.textContent = `${(tel.maxSporeLiftM * 1000).toFixed(1)}mm`;

    const b = snapshot.biolum;
    const biolumActive = b
      ? b.enabled || snapshot.species.id === 'mycena'
      : this.foxfireEnabled || snapshot.species.id === 'mycena';
    this.syncBiolumToggleDOM(biolumActive);

    if (!biolumActive) {
      this.biolumBarEl.classList.add('hidden');
    } else {
      this.biolumBarEl.classList.remove('hidden');
      const o2Ratio = b
        ? b.meanO2Factor
        : Math.max(0, Math.min(1, tel.meanSoilO2MolM3 / (1.6 + tel.meanSoilO2MolM3)));
      const phase01 = b
        ? b.circadianPhase01
        : ((((snapshot.simTimeSec - 27000) % 86400) + 86400) % 86400) / 86400;
      const circFactor = b
        ? b.circadianFactor
        : 0.22 + 0.78 * Math.pow(0.5 * (1 - Math.cos(2 * Math.PI * phase01)), 1.4);
      const phaseLabel = circFactor >= 0.52 ? 'Night' : 'Day';
      const flux = b ? b.photonFluxCm2S : 0;

      this.biolumFluxValEl.textContent = b ? `${formatPhotonFlux(flux)} ph/cm²·s` : '— ph/cm²·s';
      this.biolumO2ValEl.textContent = `${Math.round(o2Ratio * 100)}%`;
      this.biolumCircValEl.textContent = `${Math.round(circFactor * 100)}% · ${phaseLabel}`;
      this.biolumBarEl.classList.toggle('quenched', Boolean(b && flux <= 0));
    }

    this.microHeaderEl.textContent = snapshot.bullerMicro.speciesName || snapshot.species.name;
    this.telemetryPlots.render(snapshot);
  }

  dispose(): void {
    for (const off of this.boundListeners) off();
    this.boundListeners = [];
    this.rootEl.remove();
  }
}
