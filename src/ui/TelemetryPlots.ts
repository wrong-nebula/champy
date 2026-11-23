// ============================================================================
// SPOROCARP — High-DPI Bret Victor Live Biophysical Telemetry Plots
// Renders:
//   1. van Genuchten Soil Water Retention Curves theta(psi) + Live Operating Dot
//   2. Stipe Turgor P(t) vs Yield Threshold Y & Wilt Pressure P_wilt
//   3. Stipe Height L(t), Cap Radius R(t), Gill Wavenumber m_c & Spore Dispersal
// ============================================================================

import { PA_PER_M_HEAD, SimulationSnapshot, TelemetrySample } from '../types/contract.ts';

interface SoilCurveSpec {
  label: string;
  color: string;
  thetaR: number;
  thetaS: number;
  alpha: number;
  n: number;
}

// Frozen Table 1 van Genuchten parameters (THEORY.md §2.1)
const SOIL_CURVES: SoilCurveSpec[] = [
  { label: 'Sand', color: '#e5b869', thetaR: 0.045, thetaS: 0.43, alpha: 14.5, n: 2.68 },
  { label: 'Loam', color: '#38bdf8', thetaR: 0.067, thetaS: 0.45, alpha: 2.0, n: 1.41 },
  { label: 'Clay', color: '#f87171', thetaR: 0.068, thetaS: 0.38, alpha: 0.8, n: 1.09 },
  { label: 'Humus', color: '#34d399', thetaR: 0.08, thetaS: 0.65, alpha: 4.5, n: 1.55 },
  { label: 'Oak', color: '#c084fc', thetaR: 0.1, thetaS: 0.72, alpha: 1.2, n: 1.35 },
];

function vgTheta(absPsiPa: number, spec: SoilCurveSpec): number {
  const hAbs = Math.max(0, absPsiPa / PA_PER_M_HEAD);
  const m = 1 - 1 / spec.n;
  const se = Math.pow(1 + Math.pow(spec.alpha * hAbs, spec.n), -m);
  return spec.thetaR + (spec.thetaS - spec.thetaR) * se;
}

export class TelemetryPlots {
  readonly retentionCanvas: HTMLCanvasElement;
  readonly dynamicsCanvas: HTMLCanvasElement;
  private retCtx: CanvasRenderingContext2D;
  private dynCtx: CanvasRenderingContext2D;

  constructor(retentionCanvas: HTMLCanvasElement, dynamicsCanvas: HTMLCanvasElement) {
    this.retentionCanvas = retentionCanvas;
    this.dynamicsCanvas = dynamicsCanvas;
    const rCtx = retentionCanvas.getContext('2d');
    const dCtx = dynamicsCanvas.getContext('2d');
    if (!rCtx || !dCtx) {
      throw new Error('Could not acquire 2D contexts for TelemetryPlots');
    }
    this.retCtx = rCtx;
    this.dynCtx = dCtx;
  }

  private prepareCanvas(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, defaultW: number, defaultH: number): [number, number] {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(200, Math.round(rect.width || defaultW));
    const h = Math.max(100, Math.round(rect.height || defaultH));
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [w, h];
  }

  render(snapshot: SimulationSnapshot): void {
    this.renderRetentionCurve(snapshot);
    this.renderDynamicsCharts(snapshot);
  }

  private renderRetentionCurve(snapshot: SimulationSnapshot): void {
    const ctx = this.retCtx;
    const [w, h] = this.prepareCanvas(this.retentionCanvas, ctx, 300, 128);

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#090e15';
    ctx.fillRect(0, 0, w, h);

    const padL = 32;
    const padR = 10;
    const padT = 18;
    const padB = 22;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    // Log10 |psi| in Pa: 10^2 (0.1 kPa) .. 10^6.5 (~3.16 MPa)
    const logMin = 2.0;
    const logMax = 6.5;
    const thetaMax = 0.75;

    const xOfPsi = (absPsiPa: number) => {
      const lp = Math.max(logMin, Math.min(logMax, Math.log10(Math.max(100, absPsiPa))));
      return padL + ((lp - logMin) / (logMax - logMin)) * pw;
    };
    const yOfTheta = (theta: number) => {
      const t = Math.max(0, Math.min(thetaMax, theta));
      return padT + (1 - t / thetaMax) * ph;
    };

    // Gridlines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.10)';
    ctx.lineWidth = 1;
    for (const lp of [2, 3, 4, 5, 6]) {
      const gx = padL + ((lp - logMin) / (logMax - logMin)) * pw;
      ctx.beginPath();
      ctx.moveTo(gx, padT);
      ctx.lineTo(gx, padT + ph);
      ctx.stroke();
    }
    for (const th of [0.2, 0.4, 0.6]) {
      const gy = yOfTheta(th);
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + pw, gy);
      ctx.stroke();
    }

    // Axes border
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.24)';
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + ph);
    ctx.lineTo(padL + pw, padT + ph);
    ctx.stroke();

    // Plot van Genuchten retention curves for each substrate
    for (const spec of SOIL_CURVES) {
      ctx.beginPath();
      const steps = 48;
      for (let i = 0; i <= steps; i++) {
        const lp = logMin + (i / steps) * (logMax - logMin);
        const absPsi = Math.pow(10, lp);
        const th = vgTheta(absPsi, spec);
        const x = padL + (i / steps) * pw;
        const y = yOfTheta(th);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = spec.color;
      ctx.globalAlpha = 0.72;
      ctx.lineWidth = 1.35;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Live operating dot from current soil state
    const tel = snapshot.telemetry;
    const opX = xOfPsi(Math.abs(tel.meanSoilPsiPa));
    const opY = yOfTheta(tel.meanSoilTheta);

    // Crosshair lines to axes
    ctx.save();
    ctx.strokeStyle = 'rgba(248, 250, 252, 0.25)';
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(opX, padT + ph);
    ctx.lineTo(opX, opY);
    ctx.lineTo(padL, opY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Pulsing glow halo
    ctx.fillStyle = 'rgba(56, 189, 248, 0.28)';
    ctx.beginPath();
    ctx.arc(opX, opY, 7.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#f8fafc';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(opX, opY, 3.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // Axis ticks & legend
    ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';
    ctx.font = '500 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('0.6', padL - 4, yOfTheta(0.6) + 3);
    ctx.fillText('0.4', padL - 4, yOfTheta(0.4) + 3);
    ctx.fillText('0.2', padL - 4, yOfTheta(0.2) + 3);

    ctx.textAlign = 'center';
    ctx.fillText('0.1k', xOfPsi(100), h - 6);
    ctx.fillText('10k', xOfPsi(1e4), h - 6);
    ctx.fillText('100k', xOfPsi(1e5), h - 6);
    ctx.fillText('1M Pa', xOfPsi(1e6), h - 6);

    // Top mini-legend
    let lx = padL + 2;
    ctx.textAlign = 'left';
    ctx.font = '600 8px ui-sans-serif, system-ui, sans-serif';
    for (const spec of SOIL_CURVES) {
      ctx.fillStyle = spec.color;
      ctx.fillRect(lx, 6, 6, 3);
      ctx.fillStyle = 'rgba(203, 213, 225, 0.8)';
      ctx.fillText(spec.label, lx + 8, 10.5);
      lx += spec.label.length * 5.3 + 15;
    }

    // Live operating readout top-right inside plot
    const psiKpa = tel.meanSoilPsiPa / 1000;
    ctx.textAlign = 'right';
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(`θ=${tel.meanSoilTheta.toFixed(2)}  ψ=${psiKpa.toFixed(1)}kPa`, w - 8, padT + 10);
  }

  private renderDynamicsCharts(snapshot: SimulationSnapshot): void {
    const ctx = this.dynCtx;
    const [w, h] = this.prepareCanvas(this.dynamicsCanvas, ctx, 300, 174);

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#090e15';
    ctx.fillRect(0, 0, w, h);

    const history = snapshot.telemetryHistory;
    const tel = snapshot.telemetry;
    const species = snapshot.species;

    // Split vertically into two clean stacked mini-charts:
    // Top half: Turgor P(t) vs Yield Y & Wilt P_wilt
    // Bottom half: Stipe Height L(t) & Cap Radius R(t)
    const padL = 34;
    const padR = 10;
    const topT = 16;
    const topH = 62;
    const botT = topT + topH + 24;
    const botH = 56;
    const pw = w - padL - padR;

    // ---- Chart A: Stipe Turgor P(t) vs Y & Wilt ----
    const maxP = 750; // kPa
    const yOfP = (kPa: number) => topT + (1 - Math.max(0, Math.min(maxP, kPa)) / maxP) * topH;

    const yieldKpa = species.yieldPressurePa / 1000;
    const wiltKpa = species.wiltPressurePa / 1000;

    // Shaded growth (P > Y) and wilt (P < P_wilt) bands
    ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
    ctx.fillRect(padL, topT, pw, Math.max(0, yOfP(yieldKpa) - topT));
    ctx.fillStyle = 'rgba(239, 68, 68, 0.09)';
    ctx.fillRect(padL, yOfP(wiltKpa), pw, Math.max(0, topT + topH - yOfP(wiltKpa)));

    // Reference dashed lines for Y and Wilt
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.65)';
    ctx.beginPath();
    ctx.moveTo(padL, yOfP(yieldKpa));
    ctx.lineTo(padL + pw, yOfP(yieldKpa));
    ctx.stroke();

    ctx.strokeStyle = 'rgba(248, 113, 113, 0.65)';
    ctx.beginPath();
    ctx.moveTo(padL, yOfP(wiltKpa));
    ctx.lineTo(padL + pw, yOfP(wiltKpa));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Axes
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
    ctx.strokeRect(padL, topT, pw, topH);

    // Plot P(t)
    this.plotSeries(ctx, history, padL, pw, (s) => yOfP(s.leadTurgorPa / 1000), '#38bdf8', 1.8);

    // Labels for Chart A
    ctx.font = '600 8.5px ui-sans-serif, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.fillText('TURGOR P(t) vs Y & WILT', padL, topT - 5);
    ctx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    const curPkpa = tel.leadTurgorPa / 1000;
    ctx.fillStyle = curPkpa >= yieldKpa ? '#34d399' : curPkpa <= wiltKpa ? '#f87171' : '#38bdf8';
    ctx.fillText(`${curPkpa.toFixed(0)} kPa`, w - 8, topT - 5);

    ctx.fillStyle = 'rgba(148, 163, 184, 0.78)';
    ctx.font = '500 8px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('750', padL - 4, topT + 6);
    ctx.fillStyle = '#34d399';
    ctx.fillText('Y', padL - 4, yOfP(yieldKpa) + 3);
    ctx.fillStyle = '#f87171';
    ctx.fillText('W', padL - 4, yOfP(wiltKpa) + 3);

    // ---- Chart B: Stipe Height L(t) & Cap Radius R(t) ----
    const maxMm = Math.max(80, species.maxStipeHeightM * 1050);
    const yOfMm = (mm: number) => botT + (1 - Math.max(0, Math.min(maxMm, mm)) / maxMm) * botH;

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
    ctx.strokeRect(padL, botT, pw, botH);

    // Subtle horizontal grid
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
    ctx.beginPath();
    ctx.moveTo(padL, botT + botH * 0.5);
    ctx.lineTo(padL + pw, botT + botH * 0.5);
    ctx.stroke();

    this.plotSeries(ctx, history, padL, pw, (s) => yOfMm(s.leadHeightM * 1000), '#fbbf24', 1.7);
    this.plotSeries(ctx, history, padL, pw, (s) => yOfMm(s.leadCapRadiusM * 1000), '#c084fc', 1.6);

    // Labels for Chart B
    ctx.font = '600 8.5px ui-sans-serif, system-ui, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.fillText('HEIGHT L(t) & CAP R(t)', padL, botT - 5);
    ctx.font = '600 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#fbbf24';
    const lMm = (tel.leadHeightM * 1000).toFixed(1);
    const rMm = (tel.leadCapRadiusM * 1000).toFixed(1);
    ctx.fillText(`L=${lMm}mm`, w - 66, botT - 5);
    ctx.fillStyle = '#c084fc';
    ctx.fillText(`R=${rMm}mm`, w - 8, botT - 5);

    ctx.fillStyle = 'rgba(148, 163, 184, 0.78)';
    ctx.font = '500 8px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(`${Math.round(maxMm)}`, padL - 4, botT + 6);
    ctx.fillText('0', padL - 4, botT + botH);

    // Bottom compact biophysical metric pills (m_c, Dispersal Radius, O2)
    const footY = h - 4;
    ctx.textAlign = 'left';
    ctx.font = '600 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(
      `m_c=${tel.leadGillWavenumber.toFixed(0)}   r_spore=${(tel.meanSporeDispersalM * 1000).toFixed(1)}mm   O₂=${tel.meanSoilO2MolM3.toFixed(1)}mol/m³`,
      padL,
      footY,
    );
  }

  private plotSeries(
    ctx: CanvasRenderingContext2D,
    history: TelemetrySample[],
    padL: number,
    pw: number,
    getY: (sample: TelemetrySample) => number,
    color: string,
    lineWidth: number,
  ): void {
    if (history.length === 0) return;
    ctx.save();
    ctx.beginPath();
    const count = history.length;
    if (count === 1) {
      const y = getY(history[0]);
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + pw, y);
    } else {
      for (let i = 0; i < count; i++) {
        const x = padL + (i / (count - 1)) * pw;
        const y = getY(history[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();

    // Terminal live dot
    const lastY = getY(history[count - 1]);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(padL + pw, lastY, 2.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}
