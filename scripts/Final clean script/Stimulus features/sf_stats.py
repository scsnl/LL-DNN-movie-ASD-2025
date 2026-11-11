"""
Surrogate inference for autocorrelated stimulus time series.

circular shift       one series (or the window mask) is rolled by every non-zero
                     lag, or by N random lags when there are more lags than N.
phase randomisation  the amplitude spectrum is kept and the phases are replaced
                     by uniform random phases (Theiler et al. 1992).

Two-sided p = (1 + #{|null| >= |observed|}) / (1 + n_null).
"""
import numpy as np
from scipy import stats


def _z(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def _corr(a, b):
    if a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def phase_randomise(x, rng):
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=n)


def _p(null, obs):
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null)))


def surrogate_corr(x, y, n_surrogate, seed=0, label=""):
    """Pearson r with circular-shift and phase-randomisation p values."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    out = {"label": label, "n": int(n)}
    if n < 10 or x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
        out.update(r=np.nan, p_naive=np.nan, p_circshift=np.nan, p_phaserand=np.nan)
        return out
    r_obs = _corr(x, y)
    out.update(r=r_obs, p_naive=float(stats.pearsonr(x, y)[1]))
    xz, yz = _z(x), _z(y)
    lags = np.arange(1, n)
    if len(lags) > n_surrogate:
        lags = np.random.default_rng(seed).choice(lags, size=n_surrogate, replace=False)
    # cc[k] == corr(np.roll(x, k), y)
    cc = np.fft.irfft(np.conj(np.fft.rfft(xz, n=n)) * np.fft.rfft(yz, n=n), n=n) / (n - 1)
    null_c = cc[lags]
    rng = np.random.default_rng(seed + 1)
    null_p = np.array([_corr(phase_randomise(xz, rng), yz) for _ in range(n_surrogate)])
    out.update(p_circshift=_p(null_c, r_obs), p_phaserand=_p(null_p, r_obs),
               null_circshift_lo=float(np.percentile(null_c, 2.5)),
               null_circshift_hi=float(np.percentile(null_c, 97.5)),
               null_phaserand_lo=float(np.percentile(null_p, 2.5)),
               null_phaserand_hi=float(np.percentile(null_p, 97.5)))
    return out


def surrogate_mean_difference(x, mask, n_surrogate, seed=0, label=""):
    """
    Mean of x inside minus outside a window mask. Circular shift rolls the mask;
    phase randomisation randomises the data.
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=bool)
    n = len(x)
    out = {"label": label, "n": int(n), "n_in": int(m.sum()), "n_out": int((~m).sum())}
    if m.sum() < 3 or (~m).sum() < 3:
        out["diff"] = np.nan
        return out

    def stat(v, mm):
        return float(v[mm].mean() - v[~mm].mean())

    d_obs = stat(x, m)
    tt = stats.ttest_ind(x[m], x[~m], equal_var=False)
    out.update(mean_in=float(x[m].mean()), mean_out=float(x[~m].mean()),
               sd_in=float(x[m].std(ddof=1)), sd_out=float(x[~m].std(ddof=1)), diff=d_obs,
               t_welch=float(tt.statistic), p_naive_welch=float(tt.pvalue))
    lags = np.arange(1, n)
    if len(lags) > n_surrogate:
        lags = np.random.default_rng(seed).choice(lags, size=n_surrogate, replace=False)
    null_c = np.array([stat(x, np.roll(m, int(k))) for k in lags])
    rng = np.random.default_rng(seed + 1)
    mu, sd, xz = x.mean(), x.std(ddof=1), _z(x)
    null_p = np.array([stat(phase_randomise(xz, rng) * sd + mu, m) for _ in range(n_surrogate)])
    out.update(p_circshift=_p(null_c, d_obs), p_phaserand=_p(null_p, d_obs),
               null_circshift_lo=float(np.percentile(null_c, 2.5)),
               null_circshift_hi=float(np.percentile(null_c, 97.5)),
               null_phaserand_lo=float(np.percentile(null_p, 2.5)),
               null_phaserand_hi=float(np.percentile(null_p, 97.5)))
    return out
