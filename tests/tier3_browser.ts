// ============================================================================
// SPOROCARP — Tier 3 Live Browser Visual & Interactive Audit (Playwright)
// Owned exclusively by Worker 4 (biophysics-auditor). Locked by ORACLE.sha256.
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { chromium, type Browser, type Page } from 'playwright-core';
import type {
  EnvironmentState,
  SimulationSnapshot,
  SpeciesId,
  TerrariumScenePreset,
  ViewState,
} from '../src/types/contract.ts';
import type { OracleTestResult } from './tier1_oracles.ts';

interface SporocarpDebugHandle {
  getBackend(): 'webgpu' | 'webgl2' | string;
  getSnapshot(): SimulationSnapshot;
  setSpecies(id: SpeciesId): void;
  setScene(preset: TerrariumScenePreset): void;
  setView(partial: Partial<ViewState>): void;
  setEnv(partial: Partial<EnvironmentState>): void;
  stepSim(dtSec: number, steps?: number): void;
  pourRain(x: number, z: number, radius: number, depth: number): void;
  triggerFastForward(simSeconds: number): void;
}

declare global {
  interface Window {
    __SPOROCARP__?: SporocarpDebugHandle;
  }
}

export interface CapturedScreenshot {
  filename: string;
  absPath: string;
  description: string;
  bytes: number;
}

const DEFAULT_CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEFAULT_ARTIFACT_DIR = path.resolve(process.cwd(), 'artifacts');

async function launchChrome(): Promise<Browser> {
  const executablePath = process.env.CHROME_PATH || DEFAULT_CHROME_PATH;
  const commonArgs = [
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    '--window-size=1440,900',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
  ];

  try {
    return await chromium.launch({
      executablePath,
      headless: true,
      args: ['--headless=new', ...commonArgs],
    });
  } catch {
    // Fallback to visible window if --headless=new fails on host GPU
    return await chromium.launch({
      executablePath,
      headless: false,
      args: commonArgs,
    });
  }
}

async function waitFrames(page: Page, count = 4): Promise<void> {
  // String expression (no named inner functions) so tsx/esbuild never injects __name helpers into the page
  await page.evaluate(
    `new Promise((resolve) => { let rem = ${count}; const step = () => { rem -= 1; if (rem <= 0) resolve(undefined); else requestAnimationFrame(step); }; requestAnimationFrame(step); })`,
  );
}

async function captureAndVerify(
  page: Page,
  outDir: string,
  filename: string,
  description: string,
  captures: CapturedScreenshot[],
): Promise<void> {
  await waitFrames(page, 6);
  const absPath = path.join(outDir, filename);
  await page.screenshot({ path: absPath, fullPage: false });
  const stat = fs.statSync(absPath);
  if (stat.size < 15_000) {
    throw new Error(
      `Captured screenshot ${filename} looks suspiciously blank (${stat.size} bytes)`,
    );
  }
  captures.push({ filename, absPath, description, bytes: stat.size });
  console.log(`Screenshot: ${absPath} | ${description}`);
}

async function runTier3Attempt(): Promise<OracleTestResult> {
  const details: string[] = [];
  const metrics: Record<string, number | string | boolean> = {};
  const captures: CapturedScreenshot[] = [];

  const port = process.env.PORT || '5173';
  const url = process.env.SPOROCARP_URL || `http://localhost:${port}`;
  const outDir = process.env.SPOROCARP_SCREENSHOT_DIR || DEFAULT_ARTIFACT_DIR;
  fs.mkdirSync(outDir, { recursive: true });

  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const httpFailures: string[] = [];

  let browser: Browser | null = null;
  try {
    browser = await launchChrome();
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        const loc = msg.location();
        consoleErrors.push(`${msg.text()}${loc && loc.url ? ` @ ${loc.url}` : ''}`);
      }
    });
    page.on('response', (resp) => {
      if (resp.status() >= 400) {
        httpFailures.push(`${resp.status()} ${resp.url()}`);
      }
    });
    page.on('pageerror', (err) => {
      pageErrors.push(err.message || String(err));
    });

    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30_000 });
    await page.waitForFunction(() => Boolean(window.__SPOROCARP__), { timeout: 120_000 });
    await waitFrames(page, 10);

    // Check renderer backend and canvas geometry
    const initCheck = await page.evaluate(() => {
      const api = window.__SPOROCARP__!;
      const backend = api.getBackend();
      const snap = api.getSnapshot();
      const canvases = Array.from(document.querySelectorAll('canvas'));
      const mainCanvas = canvases[0];
      const rect = mainCanvas ? mainCanvas.getBoundingClientRect() : null;
      return {
        backend,
        canvasCount: canvases.length,
        mainWidth: rect ? Math.round(rect.width) : 0,
        mainHeight: rect ? Math.round(rect.height) : 0,
        soilVoxels: snap.soil.theta.length,
        myceliumNodes: snap.telemetry.myceliumNodeCount,
      };
    });

    metrics.backend = initCheck.backend;
    metrics.canvasCount = initCheck.canvasCount;
    metrics.mainCanvasWidth = initCheck.mainWidth;
    metrics.mainCanvasHeight = initCheck.mainHeight;

    if (initCheck.mainWidth < 400 || initCheck.mainHeight < 300) {
      throw new Error(
        `Main renderer canvas too small: ${initCheck.mainWidth}x${initCheck.mainHeight}`,
      );
    }

    details.push(
      `Loaded ${url} with backend=${initCheck.backend}, canvases=${initCheck.canvasCount} (main ${initCheck.mainWidth}x${initCheck.mainHeight})`,
    );

    // ------------------------------------------------------------------------
    // (a) Soil cutaway + glowing underground mycelium
    // ------------------------------------------------------------------------
    await page.evaluate(() => {
      const api = window.__SPOROCARP__!;
      api.setScene('forest-cross-section');
      api.setSpecies('parasol');
      api.setEnv({ ambientHumidity: 0.9, ambientCo2Ppm: 420, evaporationEnabled: true });
      api.pourRain(0.02, 0.0, 0.14, 0.008);
      api.triggerFastForward(700_000);
      api.setView({
        cameraYaw: 0.36,
        cameraPitch: 0.2,
        cameraDistance: 0.6,
        cameraTarget: [0, -0.07, 0],
        sliceEnabled: true,
        sliceAxis: 'z',
        slicePosition: 0.0,
        xrayMode: 'none',
        showMycelium: true,
        showSpores: false,
        showStreamlines: false,
      });
    });
    const soilStats = await page.evaluate(() => {
      const s = window.__SPOROCARP__!.getSnapshot();
      return {
        myceliumNodes: s.telemetry.myceliumNodeCount,
        anastomoses: s.telemetry.anastomosisCount,
        fruitingCount: s.telemetry.fruitingCount,
      };
    });
    metrics.cutawayMyceliumNodes = soilStats.myceliumNodes;
    metrics.cutawayAnastomoses = soilStats.anastomoses;
    await captureAndVerify(
      page,
      outDir,
      'soil_mycelium_cutaway.png',
      'Forest soil cross-section cutaway with glowing subterranean mycelial network',
      captures,
    );

    // ------------------------------------------------------------------------
    // (b1) Species: Meadow Parasol (domed cap, radial gills, torn veil annulus)
    // ------------------------------------------------------------------------
    await page.evaluate(() => {
      const api = window.__SPOROCARP__!;
      api.setScene('forest-cross-section');
      api.setSpecies('parasol');
      api.setEnv({ ambientHumidity: 0.88, ambientCo2Ppm: 420, evaporationEnabled: true });
      api.triggerFastForward(260_000);
      api.setView({
        cameraYaw: 0.55,
        cameraPitch: 0.24,
        cameraDistance: 0.56,
        cameraTarget: [0, 0.03, 0],
        sliceEnabled: false,
        xrayMode: 'none',
        showMycelium: true,
        showSpores: true,
        showStreamlines: false,
      });
    });
    const parasolStats = await page.evaluate(() => {
      const t = window.__SPOROCARP__!.getSnapshot().telemetry;
      return {
        fruitingCount: t.fruitingCount,
        heightM: t.leadHeightM,
        capRadiusM: t.leadCapRadiusM,
        gillWavenumber: t.leadGillWavenumber,
      };
    });
    metrics.parasolFruitingCount = parasolStats.fruitingCount;
    metrics.parasolLeadHeightM = parasolStats.heightM;
    await captureAndVerify(
      page,
      outDir,
      'species_parasol.png',
      'Meadow Parasol morphology — domed cap, radial gills, torn partial veil annulus',
      captures,
    );

    // ------------------------------------------------------------------------
    // (b2) Species: Golden Chanterelle (hyperbolic ruffled funnel cap)
    // ------------------------------------------------------------------------
    await page.evaluate(() => {
      const api = window.__SPOROCARP__!;
      api.setScene('forest-cross-section');
      api.setSpecies('chanterelle');
      api.setEnv({ ambientHumidity: 0.88, ambientCo2Ppm: 420, evaporationEnabled: true });
      api.triggerFastForward(420_000);
      api.setView({
        cameraYaw: 0.72,
        cameraPitch: 0.34,
        cameraDistance: 0.52,
        cameraTarget: [0, 0.02, 0],
        sliceEnabled: false,
        xrayMode: 'none',
        showMycelium: true,
        showSpores: true,
        showStreamlines: false,
      });
    });
    await captureAndVerify(
      page,
      outDir,
      'species_chanterelle.png',
      'Golden Chanterelle morphology — hyperbolic ruffled funnel cap and decurrent folds',
      captures,
    );

    // ------------------------------------------------------------------------
    // (b3) Species: Woodland Morel (cellular honeycomb ovoid head)
    // ------------------------------------------------------------------------
    await page.evaluate(() => {
      const api = window.__SPOROCARP__!;
      api.setScene('rotting-log');
      api.setSpecies('morel');
      api.setEnv({ ambientHumidity: 0.88, ambientCo2Ppm: 420, evaporationEnabled: true });
      api.triggerFastForward(600_000);
      api.setView({
        cameraYaw: 0.65,
        cameraPitch: 0.28,
        cameraDistance: 0.54,
        cameraTarget: [0, 0.02, 0],
        sliceEnabled: false,
        xrayMode: 'none',
        showMycelium: true,
        showSpores: true,
        showStreamlines: false,
      });
    });
    await captureAndVerify(
      page,
      outDir,
      'species_morel.png',
      'Woodland Morel morphology — cellular honeycomb ridges & pits on rotting-log substrate',
      captures,
    );

    // ------------------------------------------------------------------------
    // (c) Razor Slice Cutaway + Quantitative X-Ray Stress / Turgor Colormap
    // ------------------------------------------------------------------------
    await page.evaluate(() => {
      const api = window.__SPOROCARP__!;
      api.setScene('forest-cross-section');
      api.setSpecies('parasol');
      api.setEnv({ ambientHumidity: 0.88, ambientCo2Ppm: 420, evaporationEnabled: true });
      api.triggerFastForward(260_000);
      api.setView({
        cameraYaw: 0.42,
        cameraPitch: 0.26,
        cameraDistance: 0.64,
        cameraTarget: [0, 0.0, 0],
        sliceEnabled: true,
        sliceAxis: 'z',
        slicePosition: 0.0,
        xrayMode: 'shell_stress',
        showMycelium: true,
        showSpores: false,
        showStreamlines: false,
      });
    });
    await captureAndVerify(
      page,
      outDir,
      'xray_stress_slice.png',
      'Razor slice plane cutaway showing quantitative cap shell stress & soil X-ray colormap',
      captures,
    );

    // ------------------------------------------------------------------------
    // (d) 1,000x Buller's Drop Microscope Inset + Evaporative Spore Plume & Print
    // ------------------------------------------------------------------------
    await page.evaluate(() => {
      const api = window.__SPOROCARP__!;
      api.setScene('forest-cross-section');
      api.setSpecies('parasol');
      api.setEnv({ ambientHumidity: 0.8, ambientCo2Ppm: 420, evaporationEnabled: true });
      api.triggerFastForward(400_000);
      // Step real-time evaporative plume & spore cloud for 45 s
      api.stepSim(0.25, 180);
      api.setView({
        cameraYaw: 0.58,
        cameraPitch: 0.24,
        cameraDistance: 0.68,
        cameraTarget: [0, 0.05, 0],
        sliceEnabled: false,
        xrayMode: 'none',
        showMycelium: true,
        showSpores: true,
        showStreamlines: true,
        selectedBodyId: 0,
      });
    });
    const plumeStats = await page.evaluate(() => {
      const s = window.__SPOROCARP__!.getSnapshot();
      return {
        activeSpores: s.spores.activeCount,
        depositedSpores: s.sporePrint.totalDeposited,
        maxSporeLiftM: s.telemetry.maxSporeLiftM,
        bullerV0Ms: s.bullerMicro.launch.v0Ms,
        bullerPeakG: s.bullerMicro.launch.peakAccelG,
        bullerClearsGill: s.bullerMicro.launch.clearsOppositeGill,
      };
    });
    metrics.plumeActiveSpores = plumeStats.activeSpores;
    metrics.plumeDepositedSpores = plumeStats.depositedSpores;
    metrics.plumeMaxLiftM = plumeStats.maxSporeLiftM;
    metrics.bullerV0Ms = plumeStats.bullerV0Ms;
    metrics.bullerPeakG = plumeStats.bullerPeakG;
    await captureAndVerify(
      page,
      outDir,
      'buller_microscope_and_plume.png',
      "1,000x Buller's drop microscope inset + Dressaire evaporative spore plume & spore print",
      captures,
    );

    metrics.consoleErrorCount = consoleErrors.length;
    metrics.pageErrorCount = pageErrors.length;
    metrics.screenshotCount = captures.length;

    details.push(
      `Captured ${captures.length} full-window 1440x900 screenshots; consoleErrors=${consoleErrors.length}, pageErrors=${pageErrors.length}`,
    );
    details.push(
      `Plume & Buller live: activeSpores=${plumeStats.activeSpores}, deposited=${plumeStats.depositedSpores}, maxLift=${(plumeStats.maxSporeLiftM * 1000).toFixed(2)} mm, v0=${plumeStats.bullerV0Ms.toFixed(2)} m/s, a=${Math.round(plumeStats.bullerPeakG)} g`,
    );

    if (consoleErrors.length > 0) {
      details.push(`Console errors: ${consoleErrors.join(' | ')}`);
    }
    if (httpFailures.length > 0) {
      details.push(`HTTP failures: ${httpFailures.join(' | ')}`);
    }
    if (pageErrors.length > 0) {
      details.push(`Page errors: ${pageErrors.join(' | ')}`);
    }

    const passed =
      consoleErrors.length === 0 &&
      pageErrors.length === 0 &&
      captures.length === 6 &&
      soilStats.myceliumNodes > 0 &&
      parasolStats.fruitingCount > 0 &&
      plumeStats.bullerClearsGill &&
      plumeStats.maxSporeLiftM > 0 &&
      plumeStats.activeSpores + plumeStats.depositedSpores > 0;

    return {
      id: 'T3.1',
      name: 'Live Browser Visual, WebGPU & Interactive Bench Audit',
      tier: 3,
      passed,
      metrics,
      details,
      error: passed
        ? undefined
        : `Tier 3 audit failed: consoleErrors=${consoleErrors.length}, pageErrors=${pageErrors.length}, mycelium=${soilStats.myceliumNodes}, fruiting=${parasolStats.fruitingCount}`,
    };
  } catch (err) {
    let bodySnippet = '';
    try {
      if (browser) {
        const pages = browser.contexts().flatMap((c) => c.pages());
        if (pages[0]) {
          bodySnippet = (await pages[0].evaluate('document.body ? document.body.innerText.slice(0, 300) : ""')) as string;
        }
      }
    } catch {
      bodySnippet = '(page unavailable)';
    }
    if (bodySnippet) details.push(`Page body snippet: ${bodySnippet.replace(/\s+/g, ' ')}`);
    return {
      id: 'T3.1',
      name: 'Live Browser Visual, WebGPU & Interactive Bench Audit',
      tier: 3,
      passed: false,
      metrics: {
        ...metrics,
        consoleErrorCount: consoleErrors.length,
        pageErrorCount: pageErrors.length,
      },
      details: [
        ...details,
        ...(consoleErrors.length ? [`Console errors: ${consoleErrors.join(' | ')}`] : []),
        ...(pageErrors.length ? [`Page errors: ${pageErrors.join(' | ')}`] : []),
        ...(httpFailures.length ? [`HTTP failures: ${httpFailures.join(' | ')}`] : []),
      ],
      error: err instanceof Error ? err.message : String(err),
    };
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

/**
 * Vite dev servers hot-reload the page whenever a source file changes (other workers edit src/ live).
 * A mid-audit full reload is an environmental navigation, not an application fault: retry the whole
 * audit (up to 3 attempts). Console errors / page errors are NEVER retried away: any attempt that
 * finishes with an error is reported as-is.
 */
export async function runTier3BrowserAudit(): Promise<OracleTestResult> {
  let last: OracleTestResult | null = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    last = await runTier3Attempt();
    const navigationRace =
      !last.passed && typeof last.error === 'string' && last.error.includes('Execution context was destroyed');
    if (!navigationRace) {
      last.metrics.auditAttempts = attempt;
      return last;
    }
    console.log(`Tier 3 attempt ${attempt} interrupted by a dev-server page reload; retrying...`);
  }
  last!.metrics.auditAttempts = 3;
  return last!;
}
