// ============================================================================
// Render quality defaults, shared by every pass that must agree on them.
//
// Defaults favour frame rate on large / high-DPI displays: the scene is rendered
// at 1 CSS pixel per pixel. URL overrides exist for screenshots and testing:
//   ?dpr=2      render at up to 2x device pixel ratio
//   ?msaa=1|4   MSAA sample count (WebGPU only supports 1 or 4)
//   ?dof=0|1    force depth-of-field off / on (otherwise adaptive)
// ============================================================================

function readParam(name: string): string | null {
  try {
    return new URLSearchParams(globalThis.location?.search ?? '').get(name);
  } catch {
    return null;
  }
}

const dprParam = Number(readParam('dpr'));
const msaaParam = Number(readParam('msaa'));
const dofParam = readParam('dof');

/** Default MSAA sample count. WebGPU render targets only allow 1 or 4 samples. */
const DEFAULT_MSAA_SAMPLES: 1 | 4 = 4;

export const MSAA_SAMPLES: 1 | 4 =
  msaaParam === 1 || msaaParam === 4 ? (msaaParam as 1 | 4) : DEFAULT_MSAA_SAMPLES;

/** Backing-store pixels per CSS pixel. Default 1; `?dpr=` raises it up to the device ratio (max 2). */
export function renderPixelRatio(): number {
  const device = globalThis.devicePixelRatio || 1;
  if (Number.isFinite(dprParam) && dprParam > 0) {
    return Math.min(device, dprParam, 2);
  }
  return 1;
}

/** Depth-of-field forced on/off by URL, or null to let the adaptive governor decide. */
export const DOF_OVERRIDE: boolean | null = dofParam === '0' ? false : dofParam === '1' ? true : null;
