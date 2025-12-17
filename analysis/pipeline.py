import numpy as np
from analysis.loader import load_tess_lightcurve
from analysis.preprocess import clean_and_flatten
from analysis.transit import detect_transit, fold_lightcurve
from analysis.metrics import (
    odd_even_depth_check,
    compute_snr,
    secondary_eclipse_depth,
)

# ----------------------------
# Human interpretation
# ----------------------------
def generate_interpretation(period, depth, odd_depth, even_depth, snr, verdict):
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

    if abs(odd_depth - even_depth) < 0.001:
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

    lines.append(
        "Overall, the signal exhibits characteristics consistent with a transiting exoplanet candidate."
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

def confidence_score(depth, snr, odd_depth, even_depth, secondary_depth, transit_points, period):
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

    # 2) Depth realism
    if depth < 0.02:
        score += 15.0
    elif depth < 0.05:
        score += 8.0

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

    # 4️⃣ Fold light curve
    folded = fold_lightcurve(lc_flat, period)

    # 5️⃣ Vetting metrics
    odd_depth, even_depth = odd_even_depth_check(folded)
    snr = compute_snr(folded, depth)
    secondary_depth = secondary_eclipse_depth(folded)

    phase = folded.phase.value
    transit_points = int(np.sum((phase > -0.05) & (phase < 0.05)))

    

    # 6️⃣ Confidence score (FIRST)
    conf = confidence_score(
        depth=depth,
        snr=snr,
        odd_depth=odd_depth,
        even_depth=even_depth,
        secondary_depth=secondary_depth,
        transit_points=transit_points,
        period=period
)

# 7️⃣ Status derived from confidence (SECOND)
    verdict = classify_from_confidence(conf)

# 8️⃣ Interpretation (THIRD)
    interpretation = generate_interpretation(
    period, depth, odd_depth, even_depth, snr, verdict
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

    # Raw light curve
    "time": lc_clean.time.value.tolist(),
    "flux": lc_clean.flux.value.tolist(),

    # Folded light curve
    "phase": folded.phase.value.tolist(),
    "folded_flux": folded.flux.value.tolist(),
}

