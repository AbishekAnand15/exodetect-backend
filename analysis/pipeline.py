import numpy as np
from analysis.loader import load_tess_lightcurve, get_star_radius
from analysis.preprocess import clean_and_flatten
from analysis.transit import detect_transit, fold_lightcurve
from analysis.metrics import (
    odd_even_depth_check,
    compute_snr,
    secondary_eclipse_depth,
    vet_transit_shape,
    compute_stellar_scatter,
)

# ----------------------------
# Human interpretation
# ----------------------------
def generate_interpretation(period, depth, odd_depth, even_depth, snr, verdict, fit_ratio, is_v_shape):
    lines = []

    if verdict == "No Significant Transit Detected":
        lines.append(
            "No statistically significant transit signal is detected above the noise level."
        )
        lines.append(
            f"A periodic signal near {period:.2f} days is present, but the signal-to-noise ratio ({snr:.2f}) is below the detection threshold."
        )
        lines.append(
            "This signal is likely dominated by noise or stellar variability rather than a true planetary transit."
        )
        return " ".join(lines)

    # Otherwise, meaningful detection
    lines.append(
        f"A periodic transit signal with a period of {period:.2f} days is detected."
    )

    lines.append(
        f"The transit depth of {depth:.4f} corresponds to a planet-to-star radius ratio of approximately {np.sqrt(depth):.3f}."
    )

    # Transit Shape Profile feedback
    if is_v_shape:
        lines.append(
            f"Crucially, fitting models show a V-shaped profile (shape ratio {fit_ratio:.2f}), which strongly suggests an eclipsing binary star system rather than a planetary transit."
        )
    else:
        lines.append(
            f"Fitting models confirm a flat-bottomed U-shape profile (shape ratio {fit_ratio:.2f}), which is highly characteristic of a transiting exoplanet."
        )

    if abs(odd_depth - even_depth) < 0.002:
        lines.append(
            "Odd and even transits show consistent depths, reducing the likelihood of an eclipsing binary."
        )
    else:
        lines.append(
            "Odd and even transits show depth inconsistencies, which may indicate a false positive."
        )

    if period < 10:
        lines.append(
            "The short orbital period suggests a close-in planet orbiting its host star."
        )

    if not is_v_shape:
        lines.append(
            "Overall, the signal exhibits characteristics consistent with a transiting exoplanet candidate."
        )
    else:
        lines.append(
            "Overall, the signal exhibits characteristics consistent with an eclipsing binary false positive."
        )

    return " ".join(lines)



# ----------------------------
# 8-status classifier
# ----------------------------
def classify_from_confidence(conf):
    if conf < 10:
        return "No Significant Transit Detected"
    elif conf < 30:
        return "Likely False Positive ❌"
    elif conf < 50:
        return "Marginal Planet Candidate ⚠️"
    elif conf < 70:
        return "Planet Candidate 🪐"
    elif conf < 85:
        return "Strong Planet Candidate 🟢"
    else:
        return "High-Confidence Planet Candidate 🪐🟢"

def confidence_score(depth, snr, odd_depth, even_depth, secondary_depth, transit_points, period, fit_ratio, is_v_shape, stellar_scatter):
    # 🚨 Only true hard gate
    if snr < 3:
        return 5.0

    score = 0.0

    # 1) SNR (nonlinear, saturating)
    snr_score = 50.0 * (1 - np.exp(-(snr - 3) / 5))
    score += min(50.0, snr_score)

    # 🚨 Depth-based false positive suppression
    if depth > 0.002:
        score -= 25.0

    # 🚨 Short-period binary prior
    if period < 1.5:
        score -= 30.0
    elif period < 3.0:
        score -= 15.0

    # 2) Depth realism & scatter vetting
    if depth < 0.02:
        score += 15.0
    elif depth < 0.05:
        score += 8.0

    # Stellar scatter check (depth compared to local baseline noise)
    if stellar_scatter > 0:
        depth_to_scatter = depth / stellar_scatter
        if depth_to_scatter < 1.5:
            score -= 30.0
        elif depth_to_scatter < 2.0:
            score -= 15.0
        else:
            score += 10.0

    # 3) Odd–even consistency
    if abs(odd_depth - even_depth) < 0.002:
        score += 15.0
    else:
        score -= 10.0

    # 4) Secondary eclipse
    if secondary_depth < depth * 0.3:
        score += 10.0
    else:
        score -= 15.0

    # 5) Transit support
    if transit_points >= 50:
        score += 10.0
    elif transit_points >= 20:
        score += 5.0
    else:
        score -= 10.0

    # 6) Curve shape profile vetting (U-shape vs. V-shape)
    if is_v_shape:
        # Penalize V-shapes heavily since it indicates a binary star
        score -= 40.0
    else:
        # Boost clean flat-bottomed U-shapes
        if fit_ratio < 0.60:
            score += 15.0
        else:
            score += 5.0

    return round(max(0.0, min(95.0, score)), 1)





# ----------------------------
# Main pipeline
# ----------------------------
def run_exoplanet_pipeline(tic_id: int):
    # 1️⃣ Load light curve
    lc = load_tess_lightcurve(tic_id)
    if lc is None:
        return {"error": "No TESS light curve found for this TIC ID"}

    # 2️⃣ Clean & flatten
    lc_clean, lc_flat = clean_and_flatten(lc)

    # 3️⃣ Transit detection
    transit_result = detect_transit(lc_flat)
    period = float(transit_result["period"])
    depth = float(transit_result["depth"])
    epoch = float(transit_result["transit_time"])

    # 4️⃣ Fold light curve
    folded = fold_lightcurve(lc_flat, period, epoch_time=epoch)

    # 5️⃣ Vetting metrics
    odd_depth, even_depth = odd_even_depth_check(lc_flat, period)
    snr = compute_snr(folded, depth)
    secondary_depth = secondary_eclipse_depth(folded)

    # 5.5️⃣ Dynamic star catalog lookup & physical radius scaling
    star_radius = get_star_radius(str(tic_id))
    planet_radius = star_radius * np.sqrt(depth) * 109.2

    # 5.6️⃣ V-shape vs U-shape profile fitting
    duration_days = float(transit_result["duration"])
    duration_phase = duration_days / period
    fit_ratio, is_v_shape = vet_transit_shape(folded, depth, duration_phase, period)

    phase = folded.phase.value
    transit_points = int(np.sum((phase > -0.05) & (phase < 0.05)))

    # 5.7️⃣ Compute stellar scatter baseline noise
    stellar_scatter = compute_stellar_scatter(folded)

    # 6️⃣ Confidence score (FIRST)
    conf = confidence_score(
        depth=depth,
        snr=snr,
        odd_depth=odd_depth,
        even_depth=even_depth,
        secondary_depth=secondary_depth,
        transit_points=transit_points,
        period=period,
        fit_ratio=fit_ratio,
        is_v_shape=is_v_shape,
        stellar_scatter=stellar_scatter
    )

    # 7️⃣ Status derived from confidence (SECOND)
    verdict = classify_from_confidence(conf)

    # 8️⃣ Interpretation (THIRD)
    interpretation = generate_interpretation(
        period, depth, odd_depth, even_depth, snr, verdict, fit_ratio, is_v_shape
    )

    



    # 7️⃣ RETURN JSON
    return {
    "period": float(period),
    "depth": float(depth),
    "snr": float(snr),
    "odd_depth": float(odd_depth),
    "even_depth": float(even_depth),
    "secondary_depth": float(secondary_depth),
    "transit_points": int(transit_points),
    "verdict": verdict,
    "confidence": conf,
    "interpretation": interpretation,
    "star_radius": float(star_radius),
    "planet_radius": float(planet_radius),
    "fit_ratio": float(fit_ratio),
    "is_v_shape": bool(is_v_shape),
    "stellar_scatter": float(stellar_scatter),

    # Raw light curve
    "time": lc_clean.time.value.tolist(),
    "flux": lc_clean.flux.value.tolist(),

    # Folded light curve
    "phase": folded.phase.value.tolist(),
    "folded_flux": folded.flux.value.tolist(),
}

