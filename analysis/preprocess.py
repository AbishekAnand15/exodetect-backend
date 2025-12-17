def clean_and_flatten(lc):
    """
    Clean NaNs/outliers and flatten the light curve
    to remove long-term stellar trends.
    """
    # Remove NaNs
    lc_clean = lc.remove_nans()

    # Remove outliers (cosmic rays, bad points)
    lc_clean = lc_clean.remove_outliers(sigma=5)

    # Flatten the light curve (important step)
    lc_flat = lc_clean.flatten(window_length=401)

    return lc_clean, lc_flat
