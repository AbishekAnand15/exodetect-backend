import lightkurve as lk

def load_tess_lightcurve(tic_id: str):
    """
    Load a TESS light curve for a given TIC ID.
    """
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")

    if len(search) == 0:
        return None

    lc = search.download()
    return lc
