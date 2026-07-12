import os
import re
import json
import requests

def get_env_variable(name, default=None):
    """
    Robust helper to fetch environment variables, prioritizing a local .env file
    first, and falling back to os.environ.
    """
    try:
        # Check local .env file (parent directory since this script is in analysis/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(base_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == name:
                            return v.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Error reading .env file: {e}")

    if name in os.environ:
        return os.environ[name]
        
    return default

def generate_ai_interpretation(metrics: dict) -> dict:
    """
    Call NVIDIA NIM to get a high-quality scientific exoplanet analysis.
    If the key is missing or the request fails, return a dictionary with ai_used=False
    so the pipeline falls back gracefully.
    """
    api_key = get_env_variable("NVIDIA_API_KEY")
    if not api_key:
        print("NVIDIA_API_KEY not found. Falling back to rule-based interpretation.")
        return {"ai_used": False}

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Construct instructions for Llama 3.1
    system_prompt = (
        "You are an AI-powered Kepler and TESS exoplanet validation expert. Your job is to analyze "
        "raw transit metrics and output a JSON response containing a verdict/classification, a confidence score (0-100), "
        "and a scientific interpretation (3-5 sentences) summarizing the data.\n\n"
        "CRITICAL: Do NOT include any emojis in your output. The 'verdict' and 'interpretation' fields "
        "must be clean plain text without any emojis (like 🪐, ❌, ✨, ⚠️, 🟢, etc.).\n\n"
        "Guidance on Classification Taxonomy:\n"
        "- Classify candidate planets by radius (R_p). Use ONLY the following boundaries and labels:\n"
        "  * R_p < 1.5 Earth Radii      : Rocky planet (Earth-sized or smaller)\n"
        "  * 1.5 <= R_p < 4.0 Earth Radii : Super-Earth\n"
        "  * 4.0 <= R_p < 6.0 Earth Radii : Mini-Neptune / Sub-Neptune\n"
        "  * 6.0 <= R_p < 10.0 Earth Radii: Sub-Saturn (also acceptable: 'inflated sub-Saturn', 'sub-Jovian gas giant'; do NOT call this Jupiter-sized)\n"
        "  * 10.0 <= R_p < 15.0 Earth Radii: Jupiter-sized gas giant (Jupiter = 11.2 Earth Radii)\n"
        "  * R_p >= 15.0 Earth Radii     : Likely stellar companion, brown dwarf candidate, or eclipsing binary\n"
        "  CRITICAL: A planet of 6.85 Earth Radii is SUB-SATURN, NOT Jupiter-sized. Jupiter is 11.2 Earth Radii.\n\n"
        "- Interpret candidate density (rho_p):\n"
        "  * rho_p > 5.0 g/cm3: Dense rocky planet\n"
        "  * 1.5 <= rho_p <= 5.0 g/cm3: Moderate density sub-Neptune\n"
        "  * 0.5 <= rho_p < 1.5 g/cm3: Gaseous planet of typical density (Saturn ~0.69, Jupiter ~1.33, Neptune ~1.64 g/cm3). For a sub-Saturn sized object (6-10 R_earth) with density ~0.69-1.3 g/cm3, explicitly say 'low-density gaseous planet in the sub-Saturn regime'.\n"
        "  * rho_p < 0.5 g/cm3: Highly inflated gaseous planet or super-puff world (extremely low density — do NOT call it dense).\n\n"
        "Vetting Rules:\n"
        "- A periodic transit (depth) corresponding to a radius ratio of a planet is characteristic of a transiting exoplanet.\n"
        "- A V-shaped profile (is_v_shape = True) strongly suggests an eclipsing binary star system rather than a planet.\n"
        "- Depth differences between odd and even transits (odd_depth vs even_depth) point to a binary star system.\n"
        "- A significant secondary eclipse (secondary_depth > 30% of primary depth) indicates a binary system.\n"
        "- Signal-to-Noise Ratio (snr) must be high enough (typically snr >= 10 is secure; snr < 3 is noise).\n"
        "- Short periods (period < 1.5 days) have high binary probability but can be Hot Jupiters.\n\n"
        "Scientific Phrasing Guidelines:\n"
        "- Use the following scientific phrasing: 'is consistent with a planetary interpretation and shows no strong evidence for an eclipsing binary scenario' instead of 'likely a single planet candidate rather than an eclipsing binary system'.\n"
        "- Do NOT mention 'a single planet crossing' or claim there is only one planet. Instead, use: 'The U-shaped transit profile is consistent with a planet transiting the host star.' or 'The U-shaped transit profile is characteristic of a planetary transit event.'\n"
        "- When discussing shape profiles, use 'U-shaped transit profile' instead of 'U-shape profile' or 'U-shaped profile'.\n"
        "- When discussing evidence, use the phrase 'support a planetary interpretation' instead of 'support a planetary origin' to align with photometric conventions.\n"
        "- Do not report negative or zero values for the secondary eclipse depth (e.g. do not write 'secondary eclipse depth of -0.000000'). Instead, write 'No significant secondary eclipse was detected' or 'Secondary eclipse depth is consistent with zero'.\n"
        "- Refer to the photometric validation score as 'photometric consistency score' or 'vetting score' instead of 'local heuristic score'.\n"
        "- When explaining the confidence of a planet candidate, explicitly support it by highlighting the lack of odd-even variations, the absence of a secondary eclipse, and the physically plausible radius and density.\n\n"
        "Confidence Score Vetting Scale (CRITICAL CAPPING LIMITS):\n"
        "- Use the following confidence range mapping:\n"
        "  * 70.0% to 80.0%  : Planet Candidate\n"
        "  * 80.0% to 90.0%  : Strong Planet Candidate\n"
        "  * 90.0% to 95.0%  : High-Confidence Planet Candidate (ONLY if ALL four checks pass: RUWE < 1.2, neighbor count = 0, dilution factor < 0.02, AND stellar density consistency score > 0)\n"
        "  * NEVER output a score above 95.0% under any circumstances.\n"
        "  * If any of the four Gaia/density checks fail, cap confidence at 90.0%.\n\n"
        "Format the output strictly as a JSON object with these exact keys:\n"
        "{\n"
        "  \"verdict\": \"<A concise label, e.g. High-Confidence Planet Candidate, Planet Candidate, Likely False Positive, etc. Do NOT include any emojis.>\",\n"
        "  \"confidence\": <float percentage 0.0 to 95.0>,\n"
        "  \"interpretation\": \"<A detailed, scientifically accurate explanation of the metrics. Mention the shape profile, density, estimated temperature, and albedo. Mention Gaia contamination checks and stellar density consistency where relevant. Use the sizing and density taxonomy terms correctly. Do NOT include emojis.>\"\n"
        "}\n"
        "Do not include any formatting or explanation outside the JSON block. Output ONLY the JSON."
    )

    user_content = f"""
Stellar Transit Parameters for analysis (TIC ID: {metrics.get('tic_id', 'Unknown')}):
- Period: {metrics.get('period', 0.0):.4f} days
- Transit Depth: {metrics.get('depth', 0.0):.6f} (Normalized flux drop)
- Transit Duration: {metrics.get('duration_hours', 0.0):.2f} hours
- Ingress/Duration Ratio: {metrics.get('ingress_ratio', 0.0):.3f}
- Signal-to-Noise Ratio (SNR): {metrics.get('snr', 0.0):.2f}
- Odd Transit Depth: {metrics.get('odd_depth', 0.0):.6f}
- Even Transit Depth: {metrics.get('even_depth', 0.0):.6f}
- Secondary Eclipse: {"No significant secondary eclipse detected" if abs(metrics.get('secondary_depth', 0.0)) < 0.000001 else f"Depth = {metrics.get('secondary_depth', 0.0):.6f}"}
- Host Star Radius: {metrics.get('star_radius', 0.0):.2f} Solar Radii
- Host Star Temperature: {metrics.get('star_temp', 0.0):.1f} K
- Host Star Mass: {metrics.get('star_mass', 0.0):.2f} Solar Masses
- Stellar Density: {metrics.get('stellar_density', 0.0):.4f} Solar Units
- Calculated Planet Radius: {metrics.get('planet_radius', 0.0):.2f} Earth Radii
- Estimated Planet Mass: {metrics.get('planet_mass', 0.0):.2f} Earth Masses
- Estimated Planet Density: {metrics.get('planet_density', 0.0):.3f} g/cm^3
- Semi-Major Axis: {metrics.get('semi_major_axis', 0.0):.4f} AU ({metrics.get('semi_major_axis_solar', 0.0):.2f} Solar Radii)
- Planet Equilibrium Temperature: {metrics.get('equilibrium_temp', 0.0):.1f} K
- Insolation Flux: {metrics.get('insolation_flux', 0.0):.2f} Earth Units
- Profile Shape: {"V-Shape" if metrics.get('is_v_shape') else "U-Shape"} (Shape Fit Ratio: {metrics.get('fit_ratio', 0.0):.2f})
- Stellar Baseline Noise (Scatter): {metrics.get('stellar_scatter', 0.0):.6f}
- Photometric Consistency Score: {metrics.get('local_confidence', 0.0)}%
- Photometric Vetting Verdict: "{metrics.get('local_verdict', '')}"

Gaia DR3 Contamination Vetting:
- Gaia RUWE: {metrics.get('gaia_ruwe', 1.0):.3f} (< 1.2 = well-behaved single star; > 1.4 = likely unresolved binary)
- Bright Neighbours within 1 TESS Pixel (21 arcsec): {metrics.get('gaia_neighbor_count', 0)}
- Aperture Flux Dilution Factor: {metrics.get('gaia_dilution_factor', 0.0):.4f} (< 0.02 = negligible contamination)
- Gaia Vetting Score Contribution: {metrics.get('gaia_bonus', 0.0):+.1f} pts

Keplerian Stellar Density Consistency:
- Density Consistency Score Contribution: {metrics.get('density_bonus', 0.0):+.1f} pts  (positive = transit geometry matches catalog star; negative = mismatch suggests background event)

Multi-Sector Stability:
- Number of TESS Sectors Analysed: {metrics.get('sector_count', 1)}
- Stability Score Contribution: {metrics.get('stability_bonus', 0.0):+.1f} pts  (positive = depth and period stable across sectors; negative = variable)
"""

    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }

    try:
        # Use a 15-second timeout to prevent requests from failing prematurely
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            result_text = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean up potential markdown wrapping
            json_match = re.search(r"(\{.*\})", result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
                
            ai_data = json.loads(result_text)
            verdict_str = ai_data.get("verdict")
            interpretation_str = ai_data.get("interpretation")
            return {
                "ai_used": True,
                "ai_verdict": verdict_str.strip() if verdict_str else None,
                "ai_confidence": float(ai_data.get("confidence", 0.0)),
                "ai_interpretation": interpretation_str.strip() if interpretation_str else None
            }
        else:
            print(f"NVIDIA API Error (Status {response.status_code}): {response.text}")
    except Exception as e:
        print(f"Error during AI interpretation request: {e}")

    return {"ai_used": False}

def test_ai_module():
    """
    Self-test verification function that runs a mock exoplanet analysis.
    """
    mock_metrics = {
        "tic_id": 141872132,
        "period": 0.837495,
        "depth": 0.000078,
        "snr": 31.42,
        "odd_depth": 0.000079,
        "even_depth": 0.000077,
        "secondary_depth": 0.000002,
        "star_radius": 1.06,
        "planet_radius": 1.42,
        "is_v_shape": False,
        "fit_ratio": 0.35,
        "stellar_scatter": 0.000012,
        "local_confidence": 92.5,
        "local_verdict": "High-Confidence Planet Candidate"
    }
    
    print("Testing AI module with mock metrics...")
    res = generate_ai_interpretation(mock_metrics)
    print("Result of test:")
    print(json.dumps(res, indent=2))
    
    # Assert result is either AI enhanced or fallback (no crash)
    assert isinstance(res, dict)
    assert "ai_used" in res
    if res["ai_used"]:
        assert "ai_verdict" in res
        assert "ai_confidence" in res
        assert "ai_interpretation" in res
        print("Success: AI responded correctly and returned structured results.")
    else:
        print("Warning: AI API returned fallback mode (key not configured or request failed).")
