import numpy as np

def clean_and_flatten(lc):
    """
    Clean NaNs/outliers, truncate edge cadences to prevent boundary systematic artifacts,
    and dynamically flatten the light curve to remove long-term stellar trends.
    """
    # Remove NaNs
    lc_clean = lc.remove_nans()

    # Remove outliers (cosmic rays, bad points)
    lc_clean = lc_clean.remove_outliers(sigma=5)
    
    # Truncate edge cadences (prune the first and last 15 points to eliminate FITS header/t0 edge artifacts)
    if len(lc_clean) > 50:
        lc_clean = lc_clean[15:-15]

    # Calculate median exposure cadence in days
    times = lc_clean.time.value
    if len(times) > 10:
        cadence_days = np.nanmedian(np.diff(times))
    else:
        cadence_days = 0.00138  # fallback 2-minute default
        
    # Prevent divide by zero or extreme cadences
    if cadence_days <= 0 or np.isnan(cadence_days):
        cadence_days = 0.00138
        
    # Set window length to cover exactly 1.0 day of observations
    # S-G window must be odd and greater than 5
    window_length = int(1.0 / cadence_days)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(7, window_length)
    
    # Ensure window_length doesn't exceed total light curve size
    if window_length >= len(lc_clean):
        window_length = int(len(lc_clean) / 2)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(5, window_length)

    # Flatten the light curve dynamically
    lc_flat = lc_clean.flatten(window_length=window_length)

    return lc_clean, lc_flat
