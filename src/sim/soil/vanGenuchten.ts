// ============================================================================
// SPOROCARP — Layer 1: van Genuchten–Mualem Soil Hydraulics & Millington–Quirk
// Pure SI biophysical parameter tables and constitutive equations.
// Citations:
//   - van Genuchten (1980), Soil Sci. Soc. Am. J., 44(5), 892–898
//   - Carsel & Parrish (1988), Water Resour. Res., 24(5), 755–769
//   - Millington & Quirk (1961), Trans. Faraday Soc., 57, 1200–1207
// ============================================================================

import {
  PA_PER_M_HEAD,
  SoilMaterialId,
  type SoilMaterialParams,
} from '../../types/contract.ts';

export const D_O2_AIR = 2.0e-5; // m^2 / s at 20 C
export const D_CO2_AIR = 1.6e-5; // m^2 / s at 20 C
export const D_AQ_GAS = 2.0e-9; // m^2 / s dissolved aqueous gas diffusion floor

export const SOIL_MATERIALS: Record<SoilMaterialId, SoilMaterialParams> = {
  [SoilMaterialId.SAND]: {
    id: SoilMaterialId.SAND,
    name: 'Sand',
    thetaR: 0.045,
    thetaS: 0.43,
    alpha: 14.5,
    n: 2.68,
    m: 1 - 1 / 2.68,
    ks: 8.25e-5,
    initialLignin: 0.5,
    initialSolubleCarbon: 0.05,
    thermalDiffusivityDry: 2.5e-7,
    thermalDiffusivityWet: 7.8e-7,
    colorDry: [0.76, 0.67, 0.48],
    colorWet: [0.46, 0.39, 0.25],
  },
  [SoilMaterialId.LOAM]: {
    id: SoilMaterialId.LOAM,
    name: 'Forest Silt-Loam',
    thetaR: 0.067,
    thetaS: 0.45,
    alpha: 2.0,
    n: 1.41,
    m: 1 - 1 / 1.41,
    ks: 1.25e-6,
    initialLignin: 12.0,
    initialSolubleCarbon: 0.6,
    thermalDiffusivityDry: 2.2e-7,
    thermalDiffusivityWet: 6.2e-7,
    colorDry: [0.42, 0.31, 0.21],
    colorWet: [0.2, 0.14, 0.09],
  },
  [SoilMaterialId.CLAY]: {
    id: SoilMaterialId.CLAY,
    name: 'Heavy Clay',
    thetaR: 0.068,
    thetaS: 0.38,
    alpha: 0.8,
    n: 1.09,
    m: 1 - 1 / 1.09,
    ks: 5.56e-7,
    initialLignin: 6.0,
    initialSolubleCarbon: 0.3,
    thermalDiffusivityDry: 2.0e-7,
    thermalDiffusivityWet: 5.4e-7,
    colorDry: [0.48, 0.33, 0.24],
    colorWet: [0.28, 0.17, 0.12],
  },
  [SoilMaterialId.HUMUS]: {
    id: SoilMaterialId.HUMUS,
    name: 'Forest Humus / Litter',
    thetaR: 0.08,
    thetaS: 0.65,
    alpha: 4.5,
    n: 1.55,
    m: 1 - 1 / 1.55,
    ks: 3.5e-5,
    initialLignin: 45.0,
    initialSolubleCarbon: 1.8,
    thermalDiffusivityDry: 1.8e-7,
    thermalDiffusivityWet: 4.8e-7,
    colorDry: [0.24, 0.17, 0.11],
    colorWet: [0.09, 0.06, 0.04],
  },
  [SoilMaterialId.WOOD]: {
    id: SoilMaterialId.WOOD,
    name: 'Rotting Oak Log',
    thetaR: 0.1,
    thetaS: 0.72,
    alpha: 1.2,
    n: 1.35,
    m: 1 - 1 / 1.35,
    ks: 2.5e-6,
    initialLignin: 240.0,
    initialSolubleCarbon: 3.2,
    thermalDiffusivityDry: 1.6e-7,
    thermalDiffusivityWet: 4.2e-7,
    colorDry: [0.44, 0.28, 0.16],
    colorWet: [0.22, 0.12, 0.06],
  },
  [SoilMaterialId.STONE]: {
    id: SoilMaterialId.STONE,
    name: 'Impermeable Stone',
    thetaR: 0.0,
    thetaS: 0.001,
    alpha: 0.1,
    n: 1.5,
    m: 1 - 1 / 1.5,
    ks: 0.0,
    initialLignin: 0.0,
    initialSolubleCarbon: 0.0,
    thermalDiffusivityDry: 8.0e-7,
    thermalDiffusivityWet: 8.0e-7,
    colorDry: [0.42, 0.43, 0.41],
    colorWet: [0.26, 0.28, 0.27],
  },
};

/** Fast indexed lookup table matching SoilMaterialId 0..5. */
export const SOIL_MATERIAL_LIST: readonly SoilMaterialParams[] = [
  SOIL_MATERIALS[SoilMaterialId.SAND],
  SOIL_MATERIALS[SoilMaterialId.LOAM],
  SOIL_MATERIALS[SoilMaterialId.CLAY],
  SOIL_MATERIALS[SoilMaterialId.HUMUS],
  SOIL_MATERIALS[SoilMaterialId.WOOD],
  SOIL_MATERIALS[SoilMaterialId.STONE],
];

/**
 * van Genuchten effective saturation S_e(h) in [0, 1] for pressure head h (m).
 * S_e(h) = (1 + (alpha * |h|)^n)^(-m) for h < 0, and 1 for h >= 0.
 */
export function vgEffectiveSaturation(hMeters: number, mat: SoilMaterialParams): number {
  if (hMeters >= 0) return 1.0;
  const ah = mat.alpha * -hMeters;
  if (ah <= 0) return 1.0;
  const ahPowN = Math.pow(ah, mat.n);
  if (!Number.isFinite(ahPowN)) return 0.0;
  return Math.pow(1.0 + ahPowN, -mat.m);
}

/**
 * Volumetric water content theta(h) = theta_r + (theta_s - theta_r) * S_e(h).
 */
export function vgThetaFromHead(hMeters: number, mat: SoilMaterialParams): number {
  return mat.thetaR + (mat.thetaS - mat.thetaR) * vgEffectiveSaturation(hMeters, mat);
}

/**
 * Inverse van Genuchten retention curve h(theta) <= 0 (meters).
 * Overflow-safe for small m (e.g. heavy clay m ~ 0.0826).
 */
export function vgHeadFromTheta(theta: number, mat: SoilMaterialParams): number {
  const span = mat.thetaS - mat.thetaR;
  if (span <= 0) return 0.0;
  const seRaw = (theta - mat.thetaR) / span;
  if (seRaw >= 1.0 - 1e-9) return 0.0;
  const se = seRaw <= 1e-6 ? 1e-6 : seRaw;

  const logTerm = -Math.log(se) / mat.m;
  let h: number;
  if (logTerm > 40.0) {
    // Asymptotic branch: (exp(logTerm) - 1)^(1/n) ~ exp(logTerm / n)
    const expArg = logTerm / mat.n;
    h = expArg > 12.0 ? -1.0e4 : -Math.exp(expArg) / mat.alpha;
  } else {
    const inner = Math.exp(logTerm) - 1.0;
    h = inner <= 0.0 ? 0.0 : -Math.pow(inner, 1.0 / mat.n) / mat.alpha;
  }
  return h < -1.0e4 ? -1.0e4 : h;
}

/**
 * Matric water potential psi(theta) = rho_w * g * h(theta) <= 0 (Pa).
 */
export function vgPsiFromTheta(theta: number, mat: SoilMaterialParams): number {
  return PA_PER_M_HEAD * vgHeadFromTheta(theta, mat);
}

/**
 * Mualem–van Genuchten unsaturated hydraulic conductivity K(theta) (m / s).
 * K(S_e) = K_s * S_e^(0.5) * [1 - (1 - S_e^(1/m))^m]^2
 */
export function vgConductivity(theta: number, mat: SoilMaterialParams): number {
  if (mat.ks <= 0) return 0.0;
  const span = mat.thetaS - mat.thetaR;
  if (span <= 0) return 0.0;
  const se = (theta - mat.thetaR) / span;
  if (se <= 1e-9) return 0.0;
  if (se >= 1.0 - 1e-12) return mat.ks;

  const sePowInvM = Math.pow(se, 1.0 / mat.m);
  const inner = 1.0 - sePowInvM;
  const bracket = 1.0 - Math.pow(inner <= 0.0 ? 0.0 : inner, mat.m);
  return mat.ks * Math.sqrt(se) * bracket * bracket;
}

/**
 * Millington–Quirk effective pore gas diffusivity D_eff(theta) (m^2 / s):
 *   D_eff = D_0 * eps_a^(10/3) / theta_s^2 + D_aq * theta^(10/3) / theta_s^2
 * where eps_a = max(0, theta_s - theta) is air-filled porosity.
 */
export function millingtonQuirkDiffusivity(
  theta: number,
  thetaS: number,
  d0: number,
): number {
  if (thetaS <= 1e-6) return 0.0;
  const thetaClamped = theta < 0.0 ? 0.0 : theta > thetaS ? thetaS : theta;
  const epsA = thetaS - thetaClamped;
  const invThetaS2 = 1.0 / (thetaS * thetaS);
  const gasPart = epsA > 0.0 ? d0 * Math.pow(epsA, 10.0 / 3.0) * invThetaS2 : 0.0;
  const aqPart =
    thetaClamped > 0.0 ? D_AQ_GAS * Math.pow(thetaClamped, 10.0 / 3.0) * invThetaS2 : 0.0;
  return gasPart + aqPart;
}
