// ============================================================================
// SPOROCARP — Layer 5A: Micro-Scale Buller's Drop Surface-Tension Catapult
// Exact coalescence surface energy release, viscous/detachment launch
// kinematics (~25,000-60,000 g), and 4th-order Runge-Kutta Stokes drag +
// gravity trajectory integration inside the inter-gill air channel.
// ============================================================================

import {
  GAMMA_WATER,
  GRAVITY,
  MU_AIR,
  RHO_WATER,
  type BullerLaunchResult,
  type BullerParams,
  type BullerTrajectoryPoint,
} from '../../types/contract.ts';

const TWO_TO_TWO_THIRDS = Math.pow(2.0, 2.0 / 3.0);

export class BullerCatapultSolver {
  /**
   * Computes exact Buller's drop coalescence surface energy release and launch kinematics.
   *
   * Surface energy released when spherical Buller's drop (r_d) coalesces with
   * equivalent adaxial water film (r_f = r_d):
   *   DeltaE_surf = gamma * 4 * pi * r_d^2 * (2 - 2^(2/3))
   *
   * Launch kinetic energy strictly obeys energy conservation:
   *   E_k = eta * DeltaE_surf <= gamma * DeltaA   (with eta clamped to [0, 1])
   *
   * Stokes viscous braking time:
   *   tau_s = (2 * rho_s * r_s^2) / (9 * mu_air)
   * Analytical horizontal stopping distance:
   *   d_stop = v0 * tau_s
   */
  public static computeLaunchKinematics(params: BullerParams): BullerLaunchResult {
    const gamma = params.gammaWater ?? GAMMA_WATER;
    const muAir = params.muAir ?? MU_AIR;
    const rs = Math.max(1e-8, params.sporeRadiusM);
    const rd = Math.max(1e-8, params.bullerDropRadiusM);
    const rhoS = Math.max(1.0, params.sporeDensityKgM3);
    // Strict thermodynamic energy conservation: launch efficiency cannot exceed 1.0
    const eta = Math.max(0.0, Math.min(1.0, params.launchEfficiency));

    const deltaAreaM2 = 4.0 * Math.PI * rd * rd * (2.0 - TWO_TO_TWO_THIRDS);
    const deltaSurfaceEnergyJ = gamma * deltaAreaM2;
    const launchKineticEnergyJ = eta * deltaSurfaceEnergyJ;

    const sporeMassKg = (4.0 / 3.0) * Math.PI * rs * rs * rs * rhoS;
    const v0Ms = Math.sqrt((2.0 * launchKineticEnergyJ) / sporeMassKg);

    // Capillary coalescence launch duration ~ 2*pi * sqrt(rho_w * r_d^3 / gamma) (~2-4 us)
    const tauCoalesceSec = 2.0 * Math.PI * Math.sqrt((RHO_WATER * rd * rd * rd) / gamma);
    const peakAccelG = tauCoalesceSec > 0 ? v0Ms / (tauCoalesceSec * GRAVITY) : 0;

    // Exact Stokes viscous drag relaxation time: m_s / (6 * pi * mu_air * r_s)
    const tauStokesSec = (2.0 * rhoS * rs * rs) / (9.0 * muAir);
    const analyticalStoppingDistM = v0Ms * tauStokesSec;
    const terminalVelMs = GRAVITY * tauStokesSec;
    const clearsOppositeGill = analyticalStoppingDistM < params.gillSpacingM;

    return {
      deltaSurfaceEnergyJ,
      launchKineticEnergyJ,
      v0Ms,
      peakAccelG,
      tauStokesSec,
      analyticalStoppingDistM,
      terminalVelMs,
      clearsOppositeGill,
    };
  }

  /**
   * Integrates exact Stokes viscous drag + gravity ODE:
   *   dv_x/dt = -v_x / tau_s
   *   dv_y/dt = -v_y / tau_s - g
   *   dx/dt = v_x,   dy/dt = v_y
   * using classical 4th-order Runge-Kutta (RK4) with CFL-safe micro-steps
   * (<= 0.02 * tau_s) so simulated stopping distance matches analytical
   * d_stop = v0 * tau_s to << 0.1%.
   */
  public static simulateTrajectory(
    params: BullerParams,
    durationSec: number,
    dtSec?: number,
  ): BullerTrajectoryPoint[] {
    const launch = BullerCatapultSolver.computeLaunchKinematics(params);
    const tau = launch.tauStokesSec;
    const invTau = 1.0 / tau;
    const totalDur = Math.max(0, durationSec);
    const reqDt = dtSec && dtSec > 0 ? dtSec : Math.min(2.0e-6, tau * 0.02);

    // Ensure RK4 internal step is always <= 0.02 * tau for < 1e-7 relative error
    const maxSafeDt = Math.min(reqDt, tau * 0.02);
    const totalSteps = Math.max(1, Math.ceil(totalDur / maxSafeDt));
    const h = totalDur / totalSteps;

    // Limit stored array length to keep micro-view rendering fast while always preserving exact endpoints
    const maxOutputPoints = 1200;
    const emitStride = Math.max(1, Math.floor(totalSteps / maxOutputPoints));

    const points: BullerTrajectoryPoint[] = [];
    let t = 0;
    let x = 0;
    let y = 0;
    let vx = launch.v0Ms;
    let vy = 0;
    let stuckToOppositeGill = false;

    points.push({ tSec: 0, xM: 0, yM: 0, vxMs: vx, vyMs: vy });

    for (let step = 1; step <= totalSteps; step++) {
      if (!stuckToOppositeGill) {
        // RK4 integration of linear Stokes drag + gravity
        // k1
        const k1x = vx;
        const k1y = vy;
        const k1vx = -vx * invTau;
        const k1vy = -vy * invTau - GRAVITY;

        // k2
        const vx2 = vx + 0.5 * h * k1vx;
        const vy2 = vy + 0.5 * h * k1vy;
        const k2x = vx2;
        const k2y = vy2;
        const k2vx = -vx2 * invTau;
        const k2vy = -vy2 * invTau - GRAVITY;

        // k3
        const vx3 = vx + 0.5 * h * k2vx;
        const vy3 = vy + 0.5 * h * k2vy;
        const k3x = vx3;
        const k3y = vy3;
        const k3vx = -vx3 * invTau;
        const k3vy = -vy3 * invTau - GRAVITY;

        // k4
        const vx4 = vx + h * k3vx;
        const vy4 = vy + h * k3vy;
        const k4x = vx4;
        const k4y = vy4;
        const k4vx = -vx4 * invTau;
        const k4vy = -vy4 * invTau - GRAVITY;

        x += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        y += (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        vx += (h / 6.0) * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx);
        vy += (h / 6.0) * (k1vy + 2.0 * k2vy + 2.0 * k3vy + k4vy);

        // If opposite gill wall is closer than stopping distance, spore impacts & adheres
        if (params.gillSpacingM > 0 && x >= params.gillSpacingM) {
          x = params.gillSpacingM;
          vx = 0;
          vy = 0;
          stuckToOppositeGill = true;
        }
      }

      t = step * h;
      if (step % emitStride === 0 || step === totalSteps) {
        points.push({ tSec: t, xM: x, yM: y, vxMs: vx, vyMs: vy });
      }
    }

    return points;
  }

  // Instance wrappers so both static and instantiated calls work identically
  public computeLaunchKinematics(params: BullerParams): BullerLaunchResult {
    return BullerCatapultSolver.computeLaunchKinematics(params);
  }

  public simulateTrajectory(
    params: BullerParams,
    durationSec: number,
    dtSec?: number,
  ): BullerTrajectoryPoint[] {
    return BullerCatapultSolver.simulateTrajectory(params, durationSec, dtSec);
  }
}
