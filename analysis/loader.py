import os
import lightkurve as lk
from astroquery.mast import Observations

def load_tess_lightcurve(tic_id: str):
    """
    Load TESS light curve files for a given TIC ID, stitching up to 3 sectors to maximize baseline.
    """
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")

    if len(search) == 0:
        return None

    # Limit to a maximum of 3 sectors to avoid long Vercel serverless request timeouts
    table = search.table[:3]
    
    # Determine download directory
    download_dir = "/tmp" if os.name != 'nt' else search._default_download_dir()
    
    lcs = []
    for i in range(len(table)):
        row = table[i:i+1]
        path = os.path.join(
            download_dir.rstrip("/"),
            "mastDownload",
            row["obs_collection"][0],
            row["obs_id"][0],
            row["productFilename"][0],
        )
        
        if not os.path.exists(path):
            try:
                download_response = Observations.download_products(
                    row, mrp_only=False, download_dir=download_dir
                )[0]
                if download_response["Status"] == "COMPLETE":
                    path = download_response["Local Path"]
            except Exception as e:
                print(f"Failed to download sector {i}: {e}")
                continue
        
        if os.path.exists(path):
            try:
                lc = lk.read(path)
                lcs.append(lc)
            except Exception as e:
                print(f"Error reading lightcurve file {path}: {e}")
                
    if len(lcs) == 0:
        return None
        
    # Stitch multiple light curves together
    if len(lcs) > 1:
        try:
            stitched_lc = lk.LightCurveCollection(lcs).stitch()
            return stitched_lc
        except Exception as e:
            print(f"Stitching failed: {e}. Returning first sector.")
            return lcs[0]
    else:
        return lcs[0]

def get_stellar_properties(tic_id: str):
    """
    Get stellar properties (radius, teff, mass) from TESS Input Catalog (TIC)
    with robust lookups and fallbacks.
    """
    import json
    import urllib.request
    import urllib.parse

    # Hardcoded values for quick-select targets to ensure instant, stable response
    lookups = {
        141872132: {"rad": 1.06, "teff": 5627.0, "mass": 0.90, "ra": 285.6795, "dec": -0.5319,  "Gmag": 10.95},  # Kepler-10
        25155310:  {"rad": 1.28, "teff": 5800.0, "mass": 1.10, "ra": 75.5396,  "dec": -16.3143, "Gmag": 11.46},  # WASP-126
        100100827: {"rad": 1.10, "teff": 6100.0, "mass": 1.15, "ra": None,     "dec": None,     "Gmag": 12.0},   # Typical binary star primary
    }
    
    try:
        val = int(tic_id)
        if val in lookups:
            return lookups[val]
    except Exception:
        pass
        
    props = {"rad": 1.0, "teff": 5778.0, "mass": 1.0, "ra": None, "dec": None, "Gmag": 12.0}
    
    # Query MAST API JSON service directly to retrieve the star parameters
    try:
        payload = {
            "service": "Mast.Catalogs.Filtered.Tic.Rows",
            "format": "json",
            "params": {
                "columns": "rad,teff,mass,ra,dec,Gmag",
                "filters": [
                    {
                        "paramName": "ID",
                        "values": [str(tic_id)]
                    }
                ]
            }
        }
        post_data = urllib.parse.urlencode({"request": json.dumps(payload)}).encode('utf-8')
        url = "https://mast.stsci.edu/api/v0.1/json"
        
        req = urllib.request.Request(url, data=post_data, headers={
            'User-Agent': 'Mozilla/5.0',
            'Content-Type': 'application/x-www-form-urlencoded'
        })
        
        # Use a short timeout of 3 seconds to prevent API from hanging if MAST is down
        with urllib.request.urlopen(req, timeout=3) as res:
            res_data = json.loads(res.read().decode())
            if res_data and 'data' in res_data and len(res_data['data']) > 0:
                row = res_data['data'][0]
                rad = row.get('rad')
                teff = row.get('teff')
                mass = row.get('mass')
                ra_val = row.get('ra')
                dec_val = row.get('dec')
                gmag_val = row.get('Gmag')
                
                if rad is not None:
                    props["rad"] = float(rad)
                if teff is not None:
                    props["teff"] = float(teff)
                if mass is not None:
                    props["mass"] = float(mass)
                else:
                    props["mass"] = props["rad"] # Empirically mass approximates radius on MS
                if ra_val is not None:
                    props["ra"] = float(ra_val)
                if dec_val is not None:
                    props["dec"] = float(dec_val)
                if gmag_val is not None:
                    props["Gmag"] = float(gmag_val)
                return props
    except Exception as e:
        print(f"MAST direct query failed: {e}")
        
    # Standard fallback if query fails
    props["mass"] = props["rad"]
    return props

def get_star_radius(tic_id: str):
    """
    Get the star radius in Solar Radii (kept for backward compatibility).
    """
    return get_stellar_properties(tic_id)["rad"]


# In-process cache to avoid repeat Gaia calls within same request lifecycle
_gaia_cache = {}

def get_gaia_vetting(tic_id: str, ra: float = None, dec: float = None):
    """
    Query Gaia DR3 to retrieve contamination vetting metrics for a given TIC target:
      - ruwe: Gaia astrometric quality metric (>1.4 suggests unresolved binary)
      - neighbor_count: bright Gaia sources within 1 TESS pixel (21 arcsec)
      - dilution_factor: fractional flux contribution from neighbours inside aperture
    Falls back to neutral values if the query fails or times out.
    """
    cache_key = str(tic_id)
    if cache_key in _gaia_cache:
        return _gaia_cache[cache_key]

    result = {"ruwe": 1.0, "neighbor_count": 0, "dilution_factor": 0.0}

    # If RA/Dec not supplied, look them up from MAST TIC
    if ra is None or dec is None:
        # Reuse the stellar_properties query (already makes a MAST call) to get ra/dec/Gmag
        try:
            star_props = get_stellar_properties(tic_id)
            ra = star_props.get("ra")
            dec = star_props.get("dec")
            target_gmag = star_props.get("Gmag", 12.0)
        except Exception as e:
            print(f"TIC coordinate lookup failed: {e}")
            _gaia_cache[cache_key] = result
            return result
    else:
        target_gmag = 12.0  # default if not retrieved

    if ra is None or dec is None:
        _gaia_cache[cache_key] = result
        return result

    try:
        from astroquery.gaia import Gaia
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")

        # 1 arcmin cone to find the target star and measure RUWE
        j = Gaia.cone_search_async(coord, radius=u.Quantity(60, u.arcsec), verbose=False)
        r = j.get_results()

        if r is None or len(r) == 0:
            _gaia_cache[cache_key] = result
            return result

        # Sort by angular separation to find the closest match (the target itself)
        r.sort("dist")
        target_row = r[0]

        # RUWE
        ruwe_val = float(target_row["ruwe"]) if target_row["ruwe"] is not None else 1.0
        result["ruwe"] = ruwe_val

        # Neighbour contamination within 1 TESS pixel = 21 arcsec
        neighbors_21 = r[r["dist"] <= 21.0 / 3600.0]  # dist is in degrees
        target_flux = 10 ** (-0.4 * float(target_gmag))

        neighbor_count = 0
        neighbor_flux = 0.0
        for row in neighbors_21:
            g = row.get("phot_g_mean_mag")
            if g is None:
                continue
            g = float(g)
            # Only count neighbours brighter than target + 3 magnitudes
            if g < (target_gmag + 3.0) and row["dist"] > 0:
                neighbor_count += 1
                neighbor_flux += 10 ** (-0.4 * g)

        result["neighbor_count"] = neighbor_count
        # Dilution: fraction of total aperture flux from neighbours
        total_flux = target_flux + neighbor_flux
        result["dilution_factor"] = round(neighbor_flux / total_flux, 4) if total_flux > 0 else 0.0

    except Exception as e:
        print(f"Gaia vetting query failed: {e}")

    _gaia_cache[cache_key] = result
    return result

_exoarchive_cache = {}

def check_exoplanet_archive_confirmation(tic_id: str) -> list:
    """
    Query the NASA Exoplanet Archive TAP service to see if this TIC ID
    already has confirmed exoplanets, including their discovery methods.
    """
    tic_id = str(tic_id).strip()
    if tic_id in _exoarchive_cache:
        return _exoarchive_cache[tic_id]
        
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = f"select pl_name, hostname, discoverymethod from ps where tic_id = 'TIC {tic_id}' or tic_id = '{tic_id}'"
    params = {
        "query": query,
        "format": "json"
    }
    
    try:
        import requests
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # De-duplicate entries by pl_name
            seen = set()
            unique_planets = []
            for item in data:
                pl_name = item.get("pl_name")
                if pl_name and pl_name not in seen:
                    seen.add(pl_name)
                    unique_planets.append({
                        "pl_name": pl_name,
                        "hostname": item.get("hostname"),
                        "discoverymethod": item.get("discoverymethod")
                    })
            _exoarchive_cache[tic_id] = unique_planets
            return unique_planets
    except Exception as e:
        print(f"NASA Exoplanet Archive lookup failed: {e}")
        
    _exoarchive_cache[tic_id] = []
    return []

