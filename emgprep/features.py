"""Time- and frequency-domain features for one analysis window."""
import numpy as np
from scipy import signal


def window_features(w, raw, fs):
    """w: filtered window, raw: unfiltered window (for the LF artifact marker)."""
    d = np.diff(w)
    f, p = signal.periodogram(w, fs)
    P = p.sum() + 1e-12
    c = np.cumsum(p)
    fr, pr = signal.periodogram(raw - raw.mean(), fs)
    return dict(
        MAV=np.mean(np.abs(w)),
        RMS=np.sqrt(np.mean(w ** 2)),
        WL=np.sum(np.abs(d)) / len(w),
        ZC=np.mean(np.diff(np.sign(w)) != 0),
        SSC=np.mean(np.diff(np.sign(d)) != 0),
        MNF=(f * p).sum() / P,
        MDF=f[np.searchsorted(c, c[-1] / 2)],
        LF=pr[fr < 20].sum() / (pr.sum() + 1e-12),  # share of raw power < 20 Hz
    )
