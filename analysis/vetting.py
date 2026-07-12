"""
analysis/vetting.py
-------------------
Three independent photometric vetting score functions that provide
additive bonus/penalty points to the base confidence score.

These allow ExoDetect to reach >95% confidence for truly clean signals.
"""

import numpy as np


# ─────────────────────────────────────────────
# Module A: Gaia DR3 Contamination Vetting
# ─────────────────────────────────────────────
def gaia_contamination_score(gaia_data: dict) -> float:
    """
    Score contribution from Gaia DR3 neighbor contamination and RUWE check.

    Args:
        gaia_data: dict with keys:
            ruwe           - Gaia astrometric quality metric
            neighbor_count - bright Gaia sources within 21 arcsec
            dilution_factor- fractional flux from neighbours (0-1)

    Returns:
        float score delta (positive = good, negative = bad)
    """
    score = 0.0
    ruwe           = gaia_data.get("ruwe", 1.0)
    neighbor_count = gaia_data.get("neighbor_count", 0)
    dilution       = gaia_data.get("dilution_factor", 0.0)

    # --- RUWE check ---
    # RUWE < 1.2: perfectly well-behaved single star -> bonus
    # RUWE 1.2-1.4: marginally acceptable
    # RUWE > 1.4: likely unresolved binary -> penalty
    if ruwe < 1.2:
        score += 8.0
    elif ruwe < 1.4:
        score += 3.0
    elif ruwe < 2.0:
        score -= 10.0
    else:
        score -= 20.0   # strong binary signal

    # --- Neighbour contamination check ---
    if neighbor_count == 0:
        score += 7.0    # no contaminating sources inside aperture
    elif neighbor_count == 1:
        score += 2.0    # one faint source, marginally acceptable
    elif neighbor_count <= 3:
        score -= 8.0    # crowded field, dilution risk
    else:
        score -= 15.0   # heavily crowded, false-positive risk very high

    # --- Dilution factor check ---
    if dilution < 0.02:
        pass            # neutral, already counted in neighbor_count
    elif dilution < 0.10:
        score -= 5.0    # mild dilution: depth is underestimated
    elif dilution < 0.30:
        score -= 12.0   # significant dilution: transit depth unreliable
    else:
        score -= 20.0   # severe dilution: false positive very likely

    return round(score, 1)


# ─────────────────────────────────────────────────────────────────
# Module B: Stellar Density Consistency (Keplerian Vetting)
# ─────────────────────────────────────────────────────────────────
def stellar_density_consistency_score(
    period: float,
    a_over_rs: float,
    stellar_density_catalog: float,
    duration_hours: float
) -> float:
    """
    Compare photometric stellar density (derived from Kepler's Third Law
    using the transit geometry) against the catalog TIC stellar density.
    A >40% disagreement is a strong indicator of a false positive.

    Args:
        period                  - orbital period in days
        a_over_rs               - semi-major axis / stellar radius ratio
        stellar_density_catalog - catalog stellar density (solar units)
        duration_hours          - transit duration in hours

    Returns:
        float score delta
    """
    # Guard: insufficient transit duration for a reliable estimate
    if duration_hours < 1.0 or period <= 0 or a_over_rs <= 0:
        return 0.0   # neutral; cannot perform the test

    # Photometric stellar density from Kepler's Third Law
    # rho_star = (3*pi / G * P^2) * (a/R_star)^3  in SI
    G = 6.674e-11
    P_sec = period * 86400.0
    rho_photometric_si = (3.0 * np.pi / (G * P_sec**2)) * (a_over_rs**3)
    # Convert from kg/m^3 to solar density units (rho_sun = 1408 kg/m^3)
    rho_photometric_solar = rho_photometric_si / 1408.0

    if stellar_density_catalog <= 0 or rho_photometric_solar <= 0:
        return 0.0

    ratio = rho_photometric_solar / stellar_density_catalog
    disagreement = abs(ratio - 1.0)

    if disagreement < 0.20:
        return 12.0    # excellent consistency - strong planet indicator
    elif disagreement < 0.40:
        return 5.0     # acceptable scatter
    elif disagreement < 0.70:
        return -8.0    # geometry inconsistent with catalog star
    else:
        return -20.0   # strong mismatch - likely background/companion event


# ─────────────────────────────────────────────────────────────────
# Module C: Multi-Sector Period & Depth Stability
# ─────────────────────────────────────────────────────────────────
def multi_sector_stability_score(
    sector_depths: list,
    sector_periods: list
) -> float:
    """
    Check that the transit depth and period are stable across multiple
    TESS sectors. Real planets show rock-solid consistency; variable
    stars and eclipsing binaries often drift.

    Args:
        sector_depths   - list of transit depths, one per sector
        sector_periods  - list of measured periods, one per sector

    Returns:
        float score delta
    """
    if len(sector_depths) < 2 or len(sector_periods) < 2:
        return 0.0   # neutral; single-sector data, cannot test stability

    depths  = np.array([d for d in sector_depths  if d > 0], dtype=float)
    periods = np.array([p for p in sector_periods if p > 0], dtype=float)

    if len(depths) < 2 or len(periods) < 2:
        return 0.0

    depth_cv  = np.std(depths)  / np.mean(depths)
    period_cv = np.std(periods) / np.mean(periods)

    score = 0.0

    # Depth stability
    if depth_cv < 0.05:
        score += 6.0    # depths rock-solid across sectors
    elif depth_cv < 0.15:
        score += 2.0    # small variation, acceptable
    elif depth_cv < 0.30:
        score -= 5.0    # notable variability
    else:
        score -= 15.0   # large variability - likely stellar activity

    # Period stability
    if period_cv < 0.001:
        score += 4.0    # period perfectly stable
    elif period_cv < 0.01:
        score += 1.0    # marginally stable
    else:
        score -= 10.0   # period drifting - not a clean planetary signal

    return round(score, 1)
