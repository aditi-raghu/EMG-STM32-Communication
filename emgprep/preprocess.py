
import numpy as np
from scipy.signal import butter, sosfiltfilt


def clean_emg(x, fs=400):
    """Bandpass-filter a raw EMG signal."""

    x = np.asarray(x, dtype=float)

    if not np.all(np.isfinite(x)):
        raise ValueError("EMG signal contains NaN or infinity")

    # 4th-order Butterworth bandpass filter
    sos = butter(
        4,
        [20, 180],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, x)
