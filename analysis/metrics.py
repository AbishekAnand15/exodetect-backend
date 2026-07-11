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
