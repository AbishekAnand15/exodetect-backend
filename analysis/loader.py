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
    
    # Compute the local cache path
    download_dir = search._default_download_dir()
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

