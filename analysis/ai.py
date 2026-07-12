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

        # ── Comprehensive radius taxonomy ─────────────────────────────────
        "Planet Size Classification — use ONLY these boundaries and names:\n"
        "  R_p < 0.5 R_earth         : Sub-Mercury (dwarf rocky body, iron-rich)\n"
        "  0.5 <= R_p < 1.0 R_earth  : Earth-sized (rocky, terrestrial)\n"
        "  1.0 <= R_p < 1.5 R_earth  : Super-Earth (rocky, dense)\n"
        "  1.5 <= R_p < 2.0 R_earth  : Super-Earth / Mini-Neptune transition\n"
        "                               (Fulton radius gap ~1.7-2.0 R_earth separates rocky from volatile-rich)\n"
        "  2.0 <= R_p < 4.0 R_earth  : Sub-Neptune / Mini-Neptune (volatile envelope, Neptune = 3.86 R_earth)\n"
        "  4.0 <= R_p < 6.0 R_earth  : Neptune-class (Uranus = 4.01 R_earth)\n"
        "  6.0 <= R_p < 10.0 R_earth : Sub-Saturn [acceptable: 'inflated sub-Saturn', 'sub-Jovian']\n"
        "                               (Saturn = 9.45 R_earth — never call this Jupiter-sized)\n"
        "  10.0 <= R_p < 13.0 R_earth: Jupiter-class gas giant (Jupiter = 11.2 R_earth)\n"
        "  13.0 <= R_p < 22.0 R_earth: Inflated Hot Jupiter / Super-Jupiter\n"
        "  R_p >= 22.0 R_earth       : Brown dwarf candidate or stellar companion (not a planet)\n\n"
        "CRITICAL RADIUS RULES:\n"
        "  - NEVER call any R_p < 10.0 R_earth object 'Jupiter-sized' or 'Hot Jupiter'.\n"
        "  - A 6.85 R_earth planet is a Sub-Saturn. Saturn itself is only 9.45 R_earth.\n"
        "  - 'Hot Jupiter' label is ONLY valid for R_p >= 10.0 R_earth with T_eq > 700 K.\n\n"

        # ── Temperature prefix naming convention ─────────────────────────
        "Temperature Prefix Naming (combine with size class above):\n"
        "  T_eq < 250 K    : prefix 'Cold'        e.g. Cold Sub-Neptune, Cold Super-Earth\n"
        "  250-700 K       : prefix 'Warm'         e.g. Warm Sub-Saturn, Warm Jupiter\n"
        "  700-1500 K      : prefix 'Hot'          e.g. Hot Sub-Saturn, Hot Jupiter\n"
        "  > 1500 K        : prefix 'Ultra-Hot'    e.g. Ultra-Hot Sub-Saturn, Ultra-Hot Jupiter\n"
        "RULE: Always derive the prefix from T_eq AND the class label from radius. "
        "A 6.85 R_earth planet at 1369 K is a 'Hot Sub-Saturn', NEVER a 'Hot Jupiter'.\n\n"

        # ── Density taxonomy ─────────────────────────────────────────────
        "Density Classification (reference: Earth=5.51, Neptune=1.64, Jupiter=1.33, Saturn=0.69 g/cm3):\n"
        "  rho_p > 5.5 g/cm3         : Dense rocky planet (Earth-like composition)\n"
        "  3.0-5.5 g/cm3             : Rocky with thin volatile layer / dense sub-Neptune\n"
        "  1.5-3.0 g/cm3             : Moderate-density sub-Neptune / water-world candidate\n"
        "  0.5-1.5 g/cm3             : Typical gas giant density (Saturn-like to Neptune-like).\n"
        "                               For sub-Saturn (6-10 R_earth): 'low-density gaseous planet in the sub-Saturn regime'.\n"
        "  0.1-0.5 g/cm3             : Inflated / low-density gas planet — significant atmospheric inflation.\n"
        "  < 0.1 g/cm3               : Super-puff world — extremely inflated, puffy atmosphere. NEVER call dense.\n\n"

        # ── Equilibrium temperature ───────────────────────────────────────
        "Equilibrium Temperature Interpretation (MANDATORY — follow exactly):\n"
        "  T_eq < 200 K    : Extremely cold — frozen world, far from host star.\n"
        "  200-320 K       : Temperate — potentially within habitable zone. "
        "ONLY in this range may you mention liquid water or habitability.\n"
        "  320-700 K       : Warm — too hot for liquid water. 'Warm irradiated atmosphere.'\n"
        "  700-1500 K      : Hot — molten-rock surface conditions, no liquid water possible. "
        "Use 'Hot [size class]' naming (e.g., 'Hot Sub-Saturn'). "
        "NEVER mention habitability.\n"
        "  1500-2500 K     : Ultra-Hot — thermal dissociation of molecules, iron vaporization. "
        "Use 'Ultra-Hot [size class]' naming. Day-night temperature contrast is extreme.\n"
        "  > 2500 K        : Extreme — approaching stellar photosphere temperatures. "
        "Atmospheric ablation and mass loss likely.\n"
        "ABSOLUTE RULE: NEVER suggest liquid water, habitable conditions, or biosignature potential "
        "for any planet with T_eq > 320 K. Water boils at 373 K at Earth pressure.\n\n"

        # ── Insolation flux ───────────────────────────────────────────────
        "Insolation Flux Interpretation (Earth = 1.0 F_earth):\n"
        "  < 0.3 F_earth   : Cold outer system — far beyond habitable zone\n"
        "  0.3-0.95        : Outer habitable zone edge — cold but potentially habitable\n"
        "  0.95-1.5        : Near habitable zone — only range where habitability is relevant\n"
        "  1.5-10          : Warm inner system — too hot for surface liquid water\n"
        "  10-100          : Hot — highly irradiated, inside inner habitable zone boundary\n"
        "  > 100 F_earth   : Extreme irradiation — hot sub-Saturn / hot Jupiter / ultra-hot regime. "
        "NEVER associate with habitability or liquid water.\n\n"

        # ── Vetting rules ─────────────────────────────────────────────────
        "Photometric Vetting Rules:\n"
        "- U-shaped transit profile (is_v_shape = False) is consistent with a planetary transit.\n"
        "- V-shaped profile (is_v_shape = True) strongly suggests an eclipsing binary.\n"
        "- Odd/even transit depth difference > 0.002 points to a binary system.\n"
        "- Secondary eclipse depth > 30% of primary depth indicates a binary.\n"
        "- SNR >= 10 is secure detection; SNR < 3 is noise.\n"
        "- Short periods (< 1.5 days) have elevated binary probability but can be Hot Jupiters.\n\n"

        # ── Phrasing standards ────────────────────────────────────────────
        "Scientific Phrasing Standards (follow exactly):\n"
        "- Shape: write 'U-shaped transit profile', not 'U-shape profile'.\n"
        "- Binary exclusion: write 'shows no strong evidence for an eclipsing binary scenario'.\n"
        "- Planetary claim: write 'consistent with a planetary interpretation', not 'planetary origin'.\n"
        "- Do NOT say 'a single planet crossing'. Say: 'The U-shaped transit profile is consistent with "
        "a planet transiting the host star.'\n"
        "- Secondary eclipse: never write negative or zero numeric values. Write 'No significant "
        "secondary eclipse was detected.' if |secondary_depth| < 0.000001.\n"
        "- Score label: use 'photometric consistency score', not 'local heuristic score'.\n"
        "- Mention: lack of odd-even variation, absence of secondary eclipse, and physically plausible "
        "radius and density when explaining confidence.\n"
        "- When mentioning Gaia RUWE, note that RUWE < 1.2 indicates a well-behaved single star.\n"
        "- Do NOT invent values not present in the metrics (e.g., albedo is not provided).\n\n"

        # ── Confidence cap ────────────────────────────────────────────────
        "Confidence Score Rules (CRITICAL):\n"
        "  70-80%  : Planet Candidate\n"
        "  80-90%  : Strong Planet Candidate\n"
        "  90-95%  : High-Confidence Planet Candidate — ONLY if ALL four pass: "
        "RUWE < 1.2, neighbor_count = 0, dilution_factor < 0.02, density_bonus > 0.\n"
        "  NEVER output confidence > 95.0% under any circumstances.\n"
        "  If any of the four Gaia/density checks fail, cap at 90.0%.\n\n"

        # ── Output format ─────────────────────────────────────────────────
        "Output STRICTLY as a JSON object with exactly these three keys:\n"
        "{\n"
        "  \"verdict\": \"<concise label: High-Confidence Planet Candidate | Strong Planet Candidate | "
        "Planet Candidate | Marginal Planet Candidate | Likely False Positive | No Significant Transit Detected. "
        "No emojis.>\",\n"
        "  \"confidence\": <float 0.0-95.0>,\n"
        "  \"interpretation\": \"<3-5 sentences. Cover: (1) transit shape and what it implies, "
        "(2) planet size class and density regime using the taxonomy above, "
        "(3) equilibrium temperature description using the mandatory temperature table above — "
        "NEVER mention habitability if T_eq > 320 K, "
        "(4) Gaia RUWE and contamination result if available, "
        "(5) overall vetting conclusion. No emojis. No albedo speculation.>\"\n"
        "}\n"
        "Do not include any text, markdown, or explanation outside the JSON block. Output ONLY the JSON."
    )

    user_content = f"""
Stellar Transit Parameters for analysis (TIC ID: {metrics.get('tic_id', 'Unknown')}):
- Period: {metrics.get('period', 0.0):.4f} days
- Transit Depth: {metrics.get('depth', 0.0):.6f} (Normalized flux drop)
- Transit Duration: {metrics.get('duration_hours', 0.0):.2f} hours
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
        "temperature": 0.05,
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
