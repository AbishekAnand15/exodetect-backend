import os
import lightkurve as lk
from astroquery.mast import Observations

def load_tess_lightcurve(tic_id: str):
    """
    Load a TESS light curve for a given TIC ID.
    """
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")

    if len(search) == 0:
        return None

    # Get the table representing the first search result
    table = search.table[:1]
    
    # Compute the local cache path (use /tmp on Vercel/Linux, or default directory on Windows)
    download_dir = "/tmp" if os.name != 'nt' else search._default_download_dir()
    path = os.path.join(
        download_dir.rstrip("/"),
        "mastDownload",
        table["obs_collection"][0],
        table["obs_id"][0],
        table["productFilename"][0],
    )
    
    if not os.path.exists(path):
        # Download the file using astroquery download_products
        download_response = Observations.download_products(
            table, mrp_only=False, download_dir=download_dir
        )[0]
        if download_response["Status"] != "COMPLETE":
            return None
        path = download_response["Local Path"]
    
    # Read the lightcurve directly without passing the quality_bitmask keyword argument
    lc = lk.read(path)
    return lc

def get_star_radius(tic_id: str):
    """
    Get the star radius in Solar Radii from the TESS Input Catalog (TIC) with robust lookups and fallbacks.
    """
    import json
    import urllib.request
    import urllib.parse

    # Hardcoded values for quick-select targets to ensure instant, stable response
    lookups = {
        141872132: 1.06,  # Kepler-10 (Solar Radii)
        25155310: 1.28,   # WASP-126 (Solar Radii)
        100100827: 1.10   # Typical binary star primary
    }
    
    try:
        val = int(tic_id)
        if val in lookups:
            return lookups[val]
    except Exception:
        pass
        
    # Query MAST API JSON service directly to retrieve the star radius
    try:
        payload = {
            "service": "Mast.Catalogs.Filtered.Tic.Rows",
            "format": "json",
            "params": {
                "columns": "rad",
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
                rad = res_data['data'][0].get('rad')
                if rad is not None:
                    return float(rad)
    except Exception as e:
        print(f"MAST direct query failed: {e}")
        
    return 1.0  # default fallback (1 Solar Radius)


