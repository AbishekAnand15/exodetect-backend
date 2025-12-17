from astropy.timeseries import BoxLeastSquares
import numpy as np

def detect_transit(lc_flat):
    """
    Apply Box Least Squares to detect transit signals.
    """
    time = lc_flat.time.value
    flux = lc_flat.flux.value

    bls = BoxLeastSquares(time, flux)

    periods = np.linspace(0.5, 20, 10000)  # days
    durations = np.linspace(0.01, 0.3, 10)

    results = bls.power(periods, durations)

    best_index = results.power.argmax()

    best_period = results.period[best_index]
    best_duration = results.duration[best_index]
    best_depth = results.depth[best_index]

    return {
        "period": best_period,
        "duration": best_duration,
        "depth": best_depth,
        "bls_results": results
    }
def fold_lightcurve(lc_flat, period):
    """
    Phase-fold the light curve using the detected period.
    """
    folded = lc_flat.fold(period=period)
    return folded
