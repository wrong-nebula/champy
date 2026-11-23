// ============================================================================
// SPOROCARP — 1,000x Buller's Drop Gill-Gap Microscope Inset
// High-DPI 2D scientific canvas rendering opposing hymenial gill walls to
// scale, animated Buller's drop condensation/coalescence/launch, Stokes drag
// braking arc, exact d_stop caliper, and CLEAR/COLLIDE clearance readout.
// ============================================================================

import { BullerMicroViewState } from '../types/contract.ts';

export class MicroscopeCanvas {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private cssWidth = 300;
  private cssHeight = 196;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not acquire 2D context for MicroscopeCanvas');
    this.ctx = ctx;
    this.syncSize();
  }

  syncSize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(220, Math.round(rect.width || this.cssWidth));
    const h = Math.max(150, Math.round(rect.height || this.cssHeight));
    this.cssWidth = w;
    this.cssHeight = h;
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (this.canvas.width !== pw || this.canvas.height !== ph) {
      this.canvas.width = pw;
      this.canvas.height = ph;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  render(state: BullerMicroViewState): void {
    this.syncSize();
    const ctx = this.ctx;
    const w = this.cssWidth;
    const h = this.cssHeight;

    // 1. Optical dark-field microscope background
    ctx.clearRect(0, 0, w, h);
    const bgGrad = ctx.createRadialGradient(w * 0.5, h * 0.44, 10, w * 0.5, h * 0.46, Math.max(w, h) * 0.65);
    bgGrad.addColorStop(0, '#101824');
    bgGrad.addColorStop(1, '#070b10');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Physical domain mapped to canvas plot region
    const padL = 34;
    const padR = 34;
    const padT = 26;
    const padB = 42;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const gillSpacingUm = Math.max(60, state.gillSpacingM * 1e6);
    const dStopUm = Math.max(1, state.launch.analyticalStoppingDistM * 1e6);
    const domainUm = Math.max(gillSpacingUm * 1.12, dStopUm * 1.15, 220);
    const pxPerUm = plotW / domainUm;

    const leftWallX = padL;
    const rightWallX = Math.min(w - 14, leftWallX + gillSpacingUm * pxPerUm);
    const launchY = padT + plotH * 0.34;
    const vertScalePxPerUm = pxPerUm;

    // 2. Calibrated 50 µm optical reticle grid
    ctx.save();
    ctx.strokeStyle = 'rgba(120, 155, 190, 0.09)';
    ctx.lineWidth = 1;
    const gridStepUm = domainUm > 420 ? 100 : 50;
    for (let um = 0; um <= domainUm; um += gridStepUm) {
      const gx = leftWallX + um * pxPerUm;
      ctx.beginPath();
      ctx.moveTo(gx, padT - 4);
      ctx.lineTo(gx, h - padB + 4);
      ctx.stroke();
    }
    for (let gy = padT; gy <= h - padB; gy += gridStepUm * pxPerUm) {
      ctx.beginPath();
      ctx.moveTo(leftWallX, gy);
      ctx.lineTo(w - 10, gy);
      ctx.stroke();
    }
    ctx.restore();

    // 3. Opposing curved hymenial gill walls
    const drawGillWall = (xWall: number, isLeft: boolean) => {
      ctx.save();
      const sign = isLeft ? -1 : 1;
      ctx.beginPath();
      if (isLeft) {
        ctx.moveTo(0, padT - 8);
        ctx.lineTo(xWall - 2, padT - 8);
        ctx.bezierCurveTo(xWall + 4, padT + plotH * 0.35, xWall + 3, padT + plotH * 0.72, xWall - 1, h - padB + 6);
        ctx.lineTo(0, h - padB + 6);
      } else {
        ctx.moveTo(w, padT - 8);
        ctx.lineTo(xWall + 2, padT - 8);
        ctx.bezierCurveTo(xWall - 4, padT + plotH * 0.35, xWall - 3, padT + plotH * 0.72, xWall + 1, h - padB + 6);
        ctx.lineTo(w, h - padB + 6);
      }
      ctx.closePath();

      const grad = ctx.createLinearGradient(xWall + sign * 28, 0, xWall, 0);
      grad.addColorStop(0, 'rgba(72, 54, 42, 0.92)');
      grad.addColorStop(1, 'rgba(138, 108, 86, 0.85)');
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.strokeStyle = 'rgba(218, 188, 158, 0.55)';
      ctx.lineWidth = 1.4;
      ctx.stroke();

      // Subtle palisade hymenial basidia bumps along wall
      ctx.fillStyle = 'rgba(182, 148, 118, 0.32)';
      for (let by = padT + 6; by < h - padB - 4; by += 11) {
        if (isLeft && Math.abs(by - launchY) < 9) continue;
        ctx.beginPath();
        ctx.ellipse(xWall - sign * 1.5, by, 4.5, 3.2, 0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    };
    drawGillWall(leftWallX, true);
    drawGillWall(rightWallX, false);

    // 4. Active Basidium + Sterigma on left wall
    ctx.save();
    ctx.fillStyle = 'rgba(215, 192, 162, 0.88)';
    ctx.strokeStyle = 'rgba(245, 228, 200, 0.7)';
    ctx.lineWidth = 1;
    // Club-shaped basidium cell
    ctx.beginPath();
    ctx.ellipse(leftWallX + 4, launchY, 8.5, 4.8, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    // Curved sterigma prong
    ctx.beginPath();
    ctx.moveTo(leftWallX + 11, launchY - 1.5);
    ctx.quadraticCurveTo(leftWallX + 16, launchY - 3, leftWallX + 19, launchY - 0.5);
    ctx.strokeStyle = 'rgba(245, 232, 208, 0.85)';
    ctx.lineWidth = 1.3;
    ctx.stroke();
    ctx.restore();

    const originX = leftWallX + 20;
    const traj = state.trajectory;

    // 5. Full analytical / simulated ballistic + Stokes braking curve
    if (traj.length > 1) {
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < traj.length; i++) {
        const pt = traj[i];
        const px = leftWallX + Math.max(0, pt.xM * 1e6) * pxPerUm;
        const py = launchY - pt.yM * 1e6 * vertScalePxPerUm;
        const cx = Math.min(rightWallX, px);
        const cy = Math.min(h - padB + 2, Math.max(padT, py));
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
        if (px >= rightWallX) break;
      }
      ctx.strokeStyle = state.launch.clearsOppositeGill
        ? 'rgba(56, 189, 248, 0.62)'
        : 'rgba(248, 113, 113, 0.72)';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // 6. Exact d_stop vertical dashed asymptote & horizontal caliper
    const stopX = Math.min(w - 10, leftWallX + dStopUm * pxPerUm);
    ctx.save();
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.65)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(stopX, padT + 10);
    ctx.lineTo(stopX, h - padB);
    ctx.stroke();
    ctx.setLineDash([]);

    // Top gill-spacing caliper: d_gill
    const calY1 = padT - 10;
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.68)';
    ctx.fillStyle = 'rgba(203, 213, 225, 0.9)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(leftWallX, calY1);
    ctx.lineTo(rightWallX, calY1);
    ctx.moveTo(leftWallX, calY1 - 3);
    ctx.lineTo(leftWallX, calY1 + 3);
    ctx.moveTo(rightWallX, calY1 - 3);
    ctx.lineTo(rightWallX, calY1 + 3);
    ctx.stroke();
    ctx.font = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`d_gill = ${gillSpacingUm.toFixed(0)} µm`, (leftWallX + rightWallX) * 0.5, calY1 - 3);

    // Lower stopping-distance caliper: d_stop
    const calY2 = h - padB - 8;
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.85)';
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.moveTo(leftWallX, calY2);
    ctx.lineTo(stopX, calY2);
    ctx.moveTo(leftWallX, calY2 - 3);
    ctx.lineTo(leftWallX, calY2 + 3);
    ctx.moveTo(stopX, calY2 - 3);
    ctx.lineTo(stopX, calY2 + 3);
    ctx.stroke();
    ctx.textAlign = 'left';
    ctx.fillText(`d_stop ${dStopUm.toFixed(0)} µm`, leftWallX + 4, calY2 - 4);
    ctx.restore();

    // 7. Animated Buller's Drop + Spore across phase [0..1]
    const phase = ((state.phase % 1) + 1) % 1;
    const sporeVisR = Math.max(3.6, Math.min(6.2, state.params.sporeRadiusM * 1e6 * pxPerUm * 1.15));
    const dropMaxR = Math.max(2.6, Math.min(4.8, state.params.bullerDropRadiusM * 1e6 * pxPerUm * 1.15));

    if (phase < 0.38) {
      // Condensation stage: spore poised on sterigma while Buller's drop inflates
      const growT = phase / 0.38;
      this.drawSpore(ctx, originX + sporeVisR * 0.8, launchY - 2, sporeVisR, 0.15);

      // Adaxial water film shimmer
      ctx.save();
      ctx.strokeStyle = `rgba(125, 211, 252, ${0.35 + 0.45 * growT})`;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(originX + sporeVisR * 0.8, launchY - 2, sporeVisR + 1.2, -Math.PI * 0.65, Math.PI * 0.1);
      ctx.stroke();

      // Inflating spherical Buller's drop at hilar base
      const curR = Math.max(0.8, dropMaxR * Math.pow(growT, 0.7));
      const dx = originX + 1.5;
      const dy = launchY + 2.2;
      const dGrad = ctx.createRadialGradient(dx - curR * 0.3, dy - curR * 0.3, 0.3, dx, dy, curR);
      dGrad.addColorStop(0, '#e0f2fe');
      dGrad.addColorStop(0.65, '#38bdf8');
      dGrad.addColorStop(1, 'rgba(14, 165, 233, 0.45)');
      ctx.fillStyle = dGrad;
      ctx.beginPath();
      ctx.arc(dx, dy, curR, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    } else if (phase < 0.45) {
      // Coalescence surface-energy snap!
      const snapT = (phase - 0.38) / 0.07;
      const sx = originX + sporeVisR * 0.8 + snapT * 4;
      const sy = launchY - 2;
      ctx.save();
      ctx.strokeStyle = `rgba(56, 189, 248, ${1 - snapT})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, sporeVisR + snapT * 12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      this.drawSpore(ctx, sx, sy, sporeVisR, 0);
    } else {
      // Ballistic launch + exponential Stokes braking along trajectory
      const flightT = (phase - 0.45) / 0.55;
      let sx = originX;
      let sy = launchY;
      if (traj.length > 1) {
        const fIdx = Math.min(traj.length - 1, flightT * (traj.length - 1));
        const i0 = Math.floor(fIdx);
        const i1 = Math.min(traj.length - 1, i0 + 1);
        const frac = fIdx - i0;
        const xM = traj[i0].xM * (1 - frac) + traj[i1].xM * frac;
        const yM = traj[i0].yM * (1 - frac) + traj[i1].yM * frac;
        sx = Math.min(rightWallX - sporeVisR, leftWallX + Math.max(0, xM * 1e6) * pxPerUm);
        sy = Math.min(h - padB + 2, Math.max(padT, launchY - yM * 1e6 * vertScalePxPerUm));

        // Glowing solid trail up to current spore position
        ctx.save();
        ctx.beginPath();
        for (let i = 0; i <= i0; i++) {
          const tx = Math.min(rightWallX, leftWallX + Math.max(0, traj[i].xM * 1e6) * pxPerUm);
          const ty = Math.min(h - padB + 2, Math.max(padT, launchY - traj[i].yM * 1e6 * vertScalePxPerUm));
          if (i === 0) ctx.moveTo(tx, ty);
          else ctx.lineTo(tx, ty);
        }
        ctx.lineTo(sx, sy);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.0;
        ctx.stroke();
        ctx.restore();
      }
      this.drawSpore(ctx, sx, sy, sporeVisR, -0.1);
    }

    // 8. CLEAR / COLLIDE clearance badge (top-right inside reticle)
    const clears = state.launch.clearsOppositeGill;
    const badgeW = 60;
    const badgeH = 16;
    const badgeX = w - badgeW - 8;
    const badgeY = padT + 2;
    ctx.save();
    ctx.fillStyle = clears ? 'rgba(16, 185, 129, 0.22)' : 'rgba(239, 68, 68, 0.25)';
    ctx.strokeStyle = clears ? 'rgba(52, 211, 153, 0.78)' : 'rgba(248, 113, 113, 0.85)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 3);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = clears ? '#34d399' : '#f87171';
    ctx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(clears ? 'CLEAR' : 'COLLIDE', badgeX + badgeW * 0.5, badgeY + badgeH * 0.52);
    ctx.restore();

    // 9. Exact SI Kinematics Footer Strip
    const footY = h - padB + 8;
    ctx.save();
    ctx.fillStyle = 'rgba(12, 17, 25, 0.92)';
    ctx.fillRect(0, footY - 2, w, padB - 6);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
    ctx.beginPath();
    ctx.moveTo(0, footY - 2);
    ctx.lineTo(w, footY - 2);
    ctx.stroke();

    const v0 = state.launch.v0Ms;
    const accelKg = state.launch.peakAccelG / 1000;
    const deltaEfJ = state.launch.deltaSurfaceEnergyJ * 1e15;
    const vTermMmS = state.launch.terminalVelMs * 1e3;

    const cols = [
      { label: 'v₀', val: `${v0.toFixed(2)} m/s`, color: '#38bdf8' },
      { label: 'a_peak', val: `${accelKg.toFixed(1)}k g`, color: '#fbbf24' },
      { label: 'ΔE_surf', val: `${deltaEfJ.toFixed(1)} fJ`, color: '#a78bfa' },
      { label: 'v_term', val: `${vTermMmS.toFixed(2)} mm/s`, color: '#34d399' },
    ];
    const colW = w / cols.length;
    ctx.textAlign = 'center';
    for (let i = 0; i < cols.length; i++) {
      const cx = colW * (i + 0.5);
      ctx.fillStyle = 'rgba(148, 163, 184, 0.78)';
      ctx.font = '500 8.5px ui-sans-serif, system-ui, sans-serif';
      ctx.fillText(cols[i].label, cx, footY + 9);
      ctx.fillStyle = cols[i].color;
      ctx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(cols[i].val, cx, footY + 22);
    }
    ctx.restore();
  }

  private drawSpore(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    r: number,
    tilt: number,
  ): void {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(tilt);
    const grad = ctx.createRadialGradient(-r * 0.25, -r * 0.25, r * 0.1, 0, 0, r * 1.25);
    grad.addColorStop(0, '#fef3c7');
    grad.addColorStop(0.55, '#d97706');
    grad.addColorStop(1, '#78350f');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(0, 0, r * 1.22, r * 0.82, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(254, 243, 199, 0.8)';
    ctx.lineWidth = 0.9;
    ctx.stroke();
    ctx.restore();
  }
}
