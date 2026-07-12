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
        "- Classify candidate planets by radius (R_p):\n"
        "  * R_p < 1.25 Earth Radii: Earth-sized planet\n"
        "  * 1.25 <= R_p < 2.0 Earth Radii: Super-Earth\n"
        "  * 2.0 <= R_p < 4.0 Earth Radii: Sub-Neptune / Neptune-like\n"
        "  * 4.0 <= R_p < 8.0 Earth Radii: Sub-Saturn / Inflated Neptune / Mini gas giant / Hot sub-Saturn (do NOT aggressively call this a gas giant)\n"
        "  * 8.0 <= R_p < 15.0 Earth Radii: Jupiter-sized gas giant\n"
        "  * R_p >= 15.0 Earth Radii: Likely stellar companion or eclipsing binary rather than a planet\n"
        "- Interpret candidate density (rho_p):\n"
        "  * rho_p > 5.0 g/cm3: Dense rocky planet\n"
        "  * 1.5 <= rho_p <= 5.0 g/cm3: Moderate density sub-Neptune / planet\n"
        "  * 0.5 <= rho_p < 1.5 g/cm3: Gaseous planet of typical density (Saturn/Jupiter/Neptune-like). If the planet is in the sub-Saturn size regime (4.0 to 8.0 Earth Radii) and has this density (e.g. 0.78 g/cm3), explicitly describe it as a 'low-density gaseous planet in the sub-Saturn regime' (do not call it 'inflated Neptune-like').\n"
        "  * rho_p < 0.5 g/cm3: Highly inflated gaseous planet or super-puff world (CRITICAL: Describe this as extremely low density or inflated, do NOT call it dense)\n\n"
        "Vetting Rules:\n"
        "- A periodic transit (depth) corresponding to a radius ratio of a planet is characteristic of a transiting exoplanet.\n"
        "- A V-shaped profile (is_v_shape = True) strongly suggests an eclipsing binary star system rather than a planet.\n"
        "- Depth differences between odd and even transits (odd_depth vs even_depth) point to a binary star system.\n"
        "- A significant secondary eclipse (secondary_depth > 30% of primary depth) indicates a binary system.\n"
        "- Signal-to-Noise Ratio (snr) must be high enough (typically snr >= 10 is secure; snr < 3 is noise).\n"
        "- Short periods (period < 1.5 days) have high binary probability but can be Hot Jupiters.\n\n"
        "Scientific Phrasing Guidelines:\n"
        "- Use the following scientific phrasing: 'consistent with a planetary origin and shows no strong evidence for an eclipsing binary scenario' instead of 'likely a single planet candidate rather than an eclipsing binary system'.\n"
        "- When discussing shape profiles, use 'U-shaped transit profile' instead of 'U-shape profile' or 'U-shaped profile'.\n"
        "- When discussing evidence, use the phrase 'support a planetary interpretation' instead of 'support a planetary origin' to align with photometric conventions.\n"
        "- When explaining the confidence of a planet candidate, explicitly support it by highlighting the lack of odd-even variations, the absence of a secondary eclipse, and the physically plausible radius and density.\n\n"
        "Format the output strictly as a JSON object with these exact keys:\n"
        "{\n"
        "  \"verdict\": \"<A concise label, e.g. High-Confidence Planet Candidate, Planet Candidate, Likely False Positive, etc. Do NOT include any emojis.>\",\n"
        "  \"confidence\": <float percentage 0.0 to 100.0>,\n"
        "  \"interpretation\": \"<A detailed, scientifically accurate explanation of the metrics. Mention the shape profile, density, estimated temperature, and albedo. Use the sizing and density taxonomy terms correctly. Do NOT include emojis.>\"\n"
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
- Secondary Eclipse Depth: {metrics.get('secondary_depth', 0.0):.6f}
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
- Local Heuristic Score: {metrics.get('local_confidence', 0.0)}%
- Local Heuristic Verdict: "{metrics.get('local_verdict', '')}"
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
