import numpy as np

def odd_even_depth_check(lc_flat, period):
    """
    Compare odd and even transits using the raw light curve to rule out eclipsing binaries.
    """
    time = lc_flat.time.value
    flux = lc_flat.flux.value
    
    # Estimate the transit epoch (mid-transit time)
    # Fold light curve to locate epoch where transit dip center is at phase 0.0
    folded = lc_flat.fold(period=period)
    t0 = time[np.argmin(folded.flux.value)]
    
    # Calculate cycle number since the epoch for each data exposure
    cycles = np.round((time - t0) / period)
    
    # Phase offset for each exposure relative to the center of its transit cycle
    phases = (time - t0 - cycles * period) / period
    
    # Core transit window (typically width of phase 0.02)
    in_transit = np.abs(phases) < 0.015
    
    odd_cycles = (cycles % 2 == 1)
    even_cycles = (cycles % 2 == 0)
    
    # Out of transit baseline to measure depth relative to
    oot = np.abs(phases) > 0.04
    baseline = np.median(flux[oot]) if np.sum(oot) > 10 else 1.0
    
    odd_flux = flux[odd_cycles & in_transit]
    even_flux = flux[even_cycles & in_transit]
    
    odd_depth = baseline - np.median(odd_flux) if len(odd_flux) > 3 else 0.0
    even_depth = baseline - np.median(even_flux) if len(even_flux) > 3 else 0.0
    
    # Ensure depths are non-negative
    return max(0.0, float(odd_depth)), max(0.0, float(even_depth))



def planet_likeness_score(depth, odd_depth, even_depth):
    """
    Simple heuristic scoring for planet-likeness.
    """
    score = 0

    # Transit depth check
    if depth < 0.02:
        score += 1

    # Odd-even consistency
    if abs(odd_depth - even_depth) < 0.001:
        score += 1

    if score == 2:
        verdict = "Likely Planet Candidate 🪐"
    elif score == 1:
        verdict = "Uncertain Signal ⚠️"
    else:
        verdict = "Likely False Positive ❌"

    return verdict
import numpy as np

def compute_snr(folded_lc, depth, phase_width=0.04):
    """
    Robust SNR estimation using in-transit vs out-of-transit
    with MAD-based noise (NASA-style).
    """

    phase = folded_lc.phase.value
    flux = folded_lc.flux.value

    # In-transit window
    in_transit = np.abs(phase) < phase_width

    if np.sum(in_transit) < 10:
        return 0.0

    # Out-of-transit
    oot = ~in_transit

    # Robust noise estimate (MAD)
    mad = np.median(np.abs(flux[oot] - np.median(flux[oot])))
    sigma_oot = 1.4826 * mad  # MAD → sigma

    if sigma_oot <= 0:
        return 0.0

    # Effective noise reduction
    snr = depth / (sigma_oot / np.sqrt(np.sum(in_transit)))

    return float(snr)

def secondary_eclipse_depth(folded_lc):
    phase = folded_lc.phase.value
    flux = folded_lc.flux.value
    secondary = flux[(phase > 0.45) & (phase < 0.55)]
    if len(secondary) == 0:
        return 0.0
    return 1 - np.median(secondary)


import scipy.optimize as opt

def vet_transit_shape(folded_lc, depth_est, duration_phase, period):
    """
    Fit trapezoid (U-shape) and triangle (V-shape) models to the folded transit.
    Returns:
        fit_ratio: float, ratio of trapezoid SSR to triangle SSR (lower = U-shaped)
        is_v_shape: bool, True if the transit shape is V-shaped (likely eclipsing binary)
    """
    # Normalize phase to dimensionless units (-0.5 to 0.5) by dividing by period
    phase = folded_lc.phase.value / period
    flux = folded_lc.flux.value
    
    # We focus on the transit window (phase within ± 0.05, representing 10% of orbit)
    mask = np.abs(phase) < 0.05
    x = phase[mask]
    y = flux[mask]
    
    if len(x) < 15:
        return 1.0, False  # Not enough points to fit
        
    # Model 1: Trapezoid (U-Shape)
    def trapezoid(t, depth, duration, ingress, t0):
        y_fit = np.ones_like(t)
        abs_t = np.abs(t - t0)
        ing = max(ingress, 1e-5)
        t_start_slope = duration / 2.0 + ing / 2.0
        t_end_slope = duration / 2.0 - ing / 2.0
        
        mask_out = (abs_t >= t_start_slope)
        mask_bottom = (abs_t <= t_end_slope)
        mask_slope = ~mask_out & ~mask_bottom
        
        y_fit[mask_bottom] = 1.0 - depth
        y_fit[mask_slope] = 1.0 - depth * (t_start_slope - abs_t[mask_slope]) / ing
        return y_fit

    # Model 2: Triangle (V-Shape)
    def triangle(t, depth, duration, t0):
        abs_t = np.abs(t - t0)
        y_fit = np.ones_like(t)
        mask_in = abs_t < duration
        y_fit[mask_in] = 1.0 - depth * (1.0 - abs_t[mask_in] / duration)
        return y_fit

    # Setup fitting bounds and guesses in dimensionless phase units
    depth_guess = max(depth_est, 1e-4)
    duration_guess = max(duration_phase, 0.005)
    
    p0_trap = [depth_guess, duration_guess, duration_guess / 3.0, 0.0]
    bounds_trap = ([0.0, 0.001, 1e-5, -0.02], [1.0, 0.15, 0.1, 0.02])
    
    p0_tri = [depth_guess, duration_guess, 0.0]
    bounds_tri = ([0.0, 0.001, -0.02], [1.0, 0.15, 0.02])
    
    try:
        popt_trap, _ = opt.curve_fit(trapezoid, x, y, p0=p0_trap, bounds=bounds_trap)
        ssr_trap = np.sum((y - trapezoid(x, *popt_trap))**2)
    except Exception as e:
        ssr_trap = np.sum((y - 1.0)**2)  # fallback to flat baseline
        
    try:
        popt_tri, _ = opt.curve_fit(triangle, x, y, p0=p0_tri, bounds=bounds_tri)
        ssr_tri = np.sum((y - triangle(x, *popt_tri))**2)
    except Exception as e:
        ssr_tri = np.sum((y - 1.0)**2)

    # Avoid divide-by-zero
    if ssr_tri <= 0:
        return 1.0, False
        
    fit_ratio = float(ssr_trap / ssr_tri)
    
    # If the trapezoid fit is not significantly better than the triangle, it is V-shaped (binary)
    # Standard threshold: ratio > 0.85
    is_v_shape = fit_ratio > 0.85
    
    return fit_ratio, is_v_shape

