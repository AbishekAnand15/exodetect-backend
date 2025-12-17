import numpy as np

def odd_even_depth_check(folded_lc):
    """
    Compare odd and even transits to rule out eclipsing binaries.
    """
    phase = folded_lc.phase.value
    flux = folded_lc.flux.value

    odd = flux[(phase > -0.5) & (phase < 0)]
    even = flux[(phase > 0) & (phase < 0.5)]

    odd_depth = 1 - np.median(odd)
    even_depth = 1 - np.median(even)

    return odd_depth, even_depth


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
