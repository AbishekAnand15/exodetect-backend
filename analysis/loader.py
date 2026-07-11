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


