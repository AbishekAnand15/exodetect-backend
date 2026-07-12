import numpy as np
from analysis.loader import load_tess_lightcurve, get_stellar_properties, get_gaia_vetting
from analysis.vetting import (
    gaia_contamination_score,
    stellar_density_consistency_score,
    multi_sector_stability_score,
)
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
# 8-status classifier (emojis removed)
# ----------------------------
def classify_from_confidence(conf, has_external_confirmation=False):
    if conf < 10:
        return "No Significant Transit Detected"
    elif conf < 30:
        return "Likely False Positive"
    elif conf < 50:
        return "Marginal Planet Candidate"
    elif conf < 70:
        return "Planet Candidate"
    elif conf < 85:
        return "Strong Planet Candidate"
    elif conf <= 95:
        return "High-Confidence Planet Candidate"
    else:
        if has_external_confirmation:
            return "Validation-Level Candidate"
        else:
            return "High-Confidence Planet Candidate"


def confidence_score(depth, snr, odd_depth, even_depth, secondary_depth, transit_points, period, fit_ratio, is_v_shape, stellar_scatter, planet_density=None, planet_radius=None):
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

    # 7) Physical density constraints (algorithm tuning)
    if planet_density is not None and planet_radius is not None:
        if planet_density > 35.0 or planet_density < 0.05:
            score -= 25.0  # Physically implausible density
        elif planet_radius > 4.0 and planet_density > 12.0:
            score -= 20.0  # Gas giant with rocky density is unlikely (likely a binary star)
        elif planet_radius < 2.0 and planet_density < 0.5:
            score -= 15.0  # Very small planet with gas-giant density is implausible

    return round(max(0.0, min(100.0, score)), 1)


# ----------------------------
# Main pipeline
# ----------------------------
def run_exoplanet_pipeline(tic_id: int):
    # 1️⃣ Load light curve (collect per-sector data for multi-sector stability test)
    import lightkurve as lk
    sector_depths = []
    sector_periods = []

    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
    if len(search) == 0:
        return {"error": "No TESS light curve found for this TIC ID"}

    import os
    table = search.table[:3]
    download_dir = "/tmp" if os.name != 'nt' else search._default_download_dir()
    lcs = []
    for i in range(len(table)):
        row = table[i:i+1]
        path = os.path.join(
            download_dir.rstrip("/"), "mastDownload",
            row["obs_collection"][0], row["obs_id"][0], row["productFilename"][0]
        )
        if not os.path.exists(path):
            try:
                from astroquery.mast import Observations
                dl = Observations.download_products(row, mrp_only=False, download_dir=download_dir)[0]
                if dl["Status"] == "COMPLETE":
                    path = dl["Local Path"]
            except Exception:
                continue
        if os.path.exists(path):
            try:
                lcs.append(lk.read(path))
            except Exception:
                pass

    if len(lcs) == 0:
        return {"error": "No TESS light curve found for this TIC ID"}

    # Per-sector depth and period for stability test
    from analysis.preprocess import clean_and_flatten
    from analysis.transit import detect_transit
    for _lc in lcs:
        try:
            _, _flat = clean_and_flatten(_lc)
            _tr = detect_transit(_flat)
            sector_depths.append(float(_tr["depth"]))
            sector_periods.append(float(_tr["period"]))
        except Exception:
            pass

    lc = lk.LightCurveCollection(lcs).stitch() if len(lcs) > 1 else lcs[0]

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
    star_props = get_stellar_properties(str(tic_id))
    star_radius = star_props["rad"]
    star_temp = star_props["teff"]
    star_mass = star_props["mass"]

    planet_radius = star_radius * np.sqrt(depth) * 109.2

    # 5.5.1️⃣ Advanced Keplerian & physical features (Tuned Algorithm)
    duration_days = float(transit_result["duration"])
    duration_hours = duration_days * 24.0
    
    # Semi-major axis in AU (Kepler's Third Law)
    semi_major_axis = ((star_mass * (period / 365.25)**2)) ** (1/3) if period > 0 else 0.0
    # Semi-major axis in Solar Radii
    semi_major_axis_solar = semi_major_axis * 215.032
    # a/Rs ratio
    a_over_rs = semi_major_axis_solar / star_radius if star_radius > 0 else 0.0
    
    # Equilibrium Temperature in K (Assuming albedo A = 0.3)
    equilibrium_temp = star_temp * 0.9147 * np.sqrt(star_radius / (2.0 * semi_major_axis_solar)) if semi_major_axis_solar > 0 else 0.0
    
    # Insolation flux relative to Earth
    insolation_flux = (star_radius**2 / semi_major_axis**2) * (star_temp / 5778.0)**4 if semi_major_axis > 0 else 0.0
    
    # Stellar density in Solar Units (g/cm^3 relative to Sun)
    stellar_density = star_mass / (star_radius**3) if star_radius > 0 else 0.0
    
    # Estimated planet mass in Earth masses (empirical scaling tuned to solar-system & exoplanet values)
    if planet_radius < 1.5:
        # Rocky regime
        planet_mass = planet_radius**3.7
    elif planet_radius < 4.0:
        # Sub-Neptune / Neptune regime
        planet_mass = 1.43 * (planet_radius**1.7)
    elif planet_radius < 11.0:
        # Sub-Saturn / Saturn / Jovian regime
        planet_mass = 0.8 * (planet_radius**2.1)
    else:
        # Giant Jovian regime (capped around Jupiter mass of 317.8 Earth masses)
        planet_mass = 317.8
        
    # Planet density in g/cm^3
    planet_density = 5.515 * (planet_mass / (planet_radius**3)) if planet_radius > 0 else 0.0

    # 5.6️⃣ V-shape vs U-shape profile fitting
    duration_phase = duration_days / period
    fit_ratio, is_v_shape = vet_transit_shape(folded, depth, duration_phase, period)

    phase = folded.phase.value
    transit_points = int(np.sum((phase > -0.05) & (phase < 0.05)))

    # 5.7️⃣ Compute stellar scatter baseline noise
    stellar_scatter = compute_stellar_scatter(folded)

    # 6️⃣ Confidence score (FIRST) - incorporating physical constraints
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
        stellar_scatter=stellar_scatter,
        planet_density=planet_density,
        planet_radius=planet_radius
    )

    # 6.1 Gaia DR3 vetting (Module A)
    gaia_data = get_gaia_vetting(str(tic_id))
    gaia_bonus = gaia_contamination_score(gaia_data)

    # 6.2 Stellar density consistency (Module B)
    density_bonus = stellar_density_consistency_score(
        period=period,
        a_over_rs=a_over_rs,
        stellar_density_catalog=stellar_density,
        duration_hours=duration_hours
    )

    # 6.3 Multi-sector stability (Module C)
    stability_bonus = multi_sector_stability_score(sector_depths, sector_periods)

    # 6.4 Query NASA Exoplanet Archive for external confirmation
    from analysis.loader import check_exoplanet_archive_confirmation
    external_planets = check_exoplanet_archive_confirmation(str(tic_id))
    has_external_confirmation = len(external_planets) > 0

    # Combine vetting bonuses into total confidence (capped at 100)
    conf = round(max(0.0, min(100.0, conf + gaia_bonus + density_bonus + stability_bonus)), 1)
    
    # Validation-Level Candidate (>95) requires external confirmation source.
    # Otherwise, cap confidence at 95.0.
    if not has_external_confirmation:
        conf = min(95.0, conf)

    # 7️⃣ Status derived from confidence (SECOND)
    verdict = classify_from_confidence(conf, has_external_confirmation)

    # 8️⃣ Interpretation (THIRD)
    interpretation = generate_interpretation(
        period, depth, odd_depth, even_depth, snr, verdict, fit_ratio, is_v_shape
    )

    # 8.5️⃣ Optional AI Interpretation with NVIDIA Llama 3.1
    from analysis.ai import generate_ai_interpretation
    ai_results = generate_ai_interpretation({
        "tic_id": tic_id,
        "period": period,
        "depth": depth,
        "snr": snr,
        "odd_depth": odd_depth,
        "even_depth": even_depth,
        "secondary_depth": secondary_depth,
        "star_radius": star_radius,
        "star_temp": star_temp,
        "star_mass": star_mass,
        "stellar_density": stellar_density,
        "planet_radius": planet_radius,
        "planet_mass": planet_mass,
        "planet_density": planet_density,
        "duration_hours": duration_hours,
        "semi_major_axis": semi_major_axis,
        "semi_major_axis_solar": semi_major_axis_solar,
        "equilibrium_temp": equilibrium_temp,
        "insolation_flux": insolation_flux,
        "is_v_shape": is_v_shape,
        "fit_ratio": fit_ratio,
        "stellar_scatter": stellar_scatter,
        "local_confidence": conf,
        "local_verdict": verdict,
        # Vetting module results
        "gaia_ruwe": gaia_data.get("ruwe", 1.0),
        "gaia_neighbor_count": gaia_data.get("neighbor_count", 0),
        "gaia_dilution_factor": gaia_data.get("dilution_factor", 0.0),
        "gaia_bonus": gaia_bonus,
        "density_bonus": density_bonus,
        "stability_bonus": stability_bonus,
        "sector_count": len(sector_depths),
        "external_confirmations": external_planets,
    })

    # 9️⃣ RETURN JSON
    res = {
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
        "star_temp": float(star_temp),
        "star_mass": float(star_mass),
        "stellar_density": float(stellar_density),
        "planet_radius": float(planet_radius),
        "planet_mass": float(planet_mass),
        "planet_density": float(planet_density),
        "duration_hours": float(duration_hours),
        "semi_major_axis": float(semi_major_axis),
        "equilibrium_temp": float(equilibrium_temp),
        "insolation_flux": float(insolation_flux),
        "fit_ratio": float(fit_ratio),
        "is_v_shape": bool(is_v_shape),
        "stellar_scatter": float(stellar_scatter),
        # Vetting module outputs
        "gaia_ruwe": float(gaia_data.get("ruwe", 1.0)),
        "gaia_neighbor_count": int(gaia_data.get("neighbor_count", 0)),
        "gaia_dilution_factor": float(gaia_data.get("dilution_factor", 0.0)),
        "gaia_score": float(gaia_bonus),
        "density_consistency_score": float(density_bonus),
        "stability_score": float(stability_bonus),
        "sector_count": int(len(sector_depths)),
        "external_confirmations": external_planets,
        "ai_used": False,


        # Raw light curve
        "time": lc_clean.time.value.tolist(),
        "flux": lc_clean.flux.value.tolist(),

        # Folded light curve
        "phase": folded.phase.value.tolist(),
        "folded_flux": folded.flux.value.tolist(),
    }

    if ai_results.get("ai_used"):
        res["ai_used"] = True
        res["ai_verdict"] = ai_results["ai_verdict"]
        res["ai_confidence"] = ai_results["ai_confidence"]
        res["ai_interpretation"] = ai_results["ai_interpretation"]

    return res
