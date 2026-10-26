"""
Verification suite for the 2026-10-26 bug-fix batch (v0.3 development).

Each check corresponds to a documented defect found by probing v0.2:

V1  ci_band matches the exact mean-response formula t*s*sqrt(1/n + (x-xbar)^2/Sxx)
    with df = n - k (old code: missing outer sqrt, Sxx under a sqrt, df = n - 2
    -> bands ~7.6x too narrow).
V2  high-degree monomial fits no longer silently corrupt: deg-10 fit of x^2 on
    [0,1] returns 0.25 at x=0.5 (old code returned O(2e2)); orthogonality of the
    QR-fallback basis is preserved; low degrees keep the determinant route and
    still agree with numpy.polyfit.
V3  near-colliding abscissas are no longer merged by the old absolute 1e-4 spike
    window (hourly data over a multi-year span has gaps < 1e-4 in normalized units).
V4  Time2Interval preserves sub-day resolution (old `.days` truncated intraday
    timestamps onto whole days) and maps endpoints exactly to [lower, upper].
V5  score() is R^2 per the scikit-learn convention; mse() keeps the old weighted
    MSE semantics.
V6  elec_cost.csv fit still agrees with numpy.polyfit (regression check).
V7  TimeSeriesCV: default single split is unchanged, n_splits >= 2 gives a leak-free
    expanding walk-forward window, degenerate/invalid inputs raise ValueError.

Run:  python tests/test_fixes_20261026.py   (plain asserts, no framework needed)
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GeneralRegression.NpyProximation import HilbertRegressor
from GeneralRegression.ModelSelection import TimeSeriesCV
from GeneralRegression.extras import Time2Interval


def v1_ci_band_exact():
    rng = np.random.default_rng(0)
    n = 25
    X = rng.uniform(0, 1, n).reshape(-1, 1)
    y = (3.0 - 2.0 * X.ravel() + 0.5 * np.sin(4.0 * X.ravel()) + 0.1 * rng.standard_normal(n)).ravel()

    # --- Part A: k = 2 (deg 1), where the closed form is exactly valid --------
    model = HilbertRegressor(deg=1).fit(X, y)          # k = 2 parameters
    xq = np.linspace(0, 1, 9).reshape(-1, 1)
    _ = model.predict(xq)                              # side effect: sets ci_band
    band_pkg = model.ci_band

    k = 2
    df = n - k
    yhat = np.array([float(np.ravel(model.apprx(x_))[0]) for x_ in X]).ravel()
    s = float(np.sqrt(np.sum((y - yhat) ** 2) / df))
    # model.c_limit stores the raw two-sided level (e.g. .95); one-sided quantile arg is (level+1)/2:
    tval = t.ppf(0.5 * (1.0 + model.c_limit), df)
    xbar = X.ravel().mean()
    Sxx = float(np.sum((X.ravel() - xbar) ** 2))
    band_ref = tval * s * np.sqrt(1.0 / n + (xq.ravel() - xbar) ** 2 / Sxx)

    ratio = band_pkg / band_ref
    assert np.allclose(ratio, 1.0, rtol=1e-8), f"ci_band ratio vs exact formula: {ratio}"
    print(f"V1a OK  ci_band == t*s*sqrt(1/n + (x-xbar)^2/Sxx), df={df}; max|ratio-1| = "
          f"{np.max(np.abs(ratio - 1)):.2e}")

    # --- Part B: k = 4 (deg 3): Monte-Carlo validation of the general ---------
    # formula Var(yhat(x*)) = sigma^2 b(x*)' (B'B)^-1 b(x*). The closed form of
    # Part A does NOT hold for k > 2, so instead we estimate the sampling sd of
    # yhat at a query point by repeated refits under additive noise.
    rng2 = np.random.default_rng(1)
    deg, k4, reps, sigma = 3, 4, 2000, 0.1
    m_ref = HilbertRegressor(deg=deg).fit(X, y)        # fixed design X reused each rep
    xstar = np.array([[0.4]])
    A = np.column_stack([X.ravel() ** j for j in range(k4)])   # raw monomial design
    b_star = np.array([xstar[0][0] ** j for j in range(k4)])

    yhat_stars = []
    for _ in range(reps):
        y_rep = (3.0 - 2.0 * X.ravel() + 0.5 * np.sin(4.0 * X.ravel())
                 + sigma * rng2.standard_normal(n))
        m_rep = HilbertRegressor(deg=deg).fit(X, y_rep)
        yhat_stars.append(float(np.ravel(m_rep.apprx(xstar))[0]))
    sd_emp = float(np.std(yhat_stars, ddof=1))

    s2_mle = float(sigma ** 2)                         # noise sd known by construction
    var_theory = s2_mle * b_star @ np.linalg.solve(A.T @ A, b_star)
    ratio_mc = sd_emp / float(np.sqrt(var_theory))
    assert 0.9 < ratio_mc < 1.1, f"MC sampling sd {sd_emp} vs theory " \
                                 f"{float(np.sqrt(var_theory))}: ratio {ratio_mc}"

    # and the package's band at xstar must equal t * s * sqrt(theoretical var)
    m_ref.predict(xstar)
    df4 = n - k4
    yhat_fit = np.array([float(np.ravel(m_ref.apprx(x_))[0]) for x_ in X]).ravel()
    s_pkg = float(np.sqrt(np.sum((y - yhat_fit) ** 2) / df4))
    # band = t * s_pkg * sqrt(b'(A'A)^-1 b); var_theory carries sigma^2, so strip it:
    band_theory_xstar = (t.ppf(0.5 * (1.0 + m_ref.c_limit), df4) * s_pkg
                         * float(np.sqrt(var_theory)) / sigma)
    assert abs(float(m_ref.ci_band[0]) - band_theory_xstar) < 1e-8 * band_theory_xstar, \
        (m_ref.ci_band[0], band_theory_xstar)
    print(f"V1b OK  k=4: MC sd/sd_theory = {ratio_mc:.4f}; package band at x*=0.4 matches "
          f"t*s*sqrt(b'(B'B)^-1 b) to 1e-8")


def v2_conditioning():
    n = 40
    X = np.linspace(0, 1, n).reshape(-1, 1)
    y_true = lambda xv: xv ** 2
    y = y_true(X.ravel())

    # deg 10: monomial Gram cond ~ 4e14 -> QR fallback must kick in and be exact
    m10 = HilbertRegressor(deg=10).fit(X, y)
    p10 = float(np.ravel(m10.predict(np.array([[0.5]])))[0])
    assert abs(p10 - 0.25) < 1e-9, f"deg-10 fit at x=0.5 gave {p10} (expected 0.25)"

    # orthogonality of the fallback basis on the support (exact for discrete measures)
    fs = m10.regressor.orth
    keys = list(fs.measure.supp)
    wv = np.array([fs.measure.density(k_) for k_ in keys], dtype=float)
    EO = np.array([[float(np.ravel(phi(np.asarray(k_))) [0]) for phi in fs.orth_base]
                   for k_ in keys])
    G2 = (EO.T * wv) @ EO
    off = G2 - np.diag(np.diag(G2))
    # composed lambdas re-introduce O(eps*|C|) cancellation, so allow a loose band;
    # the point is that we are 13+ orders of magnitude away from the broken state:
    assert np.max(np.abs(off)) < 1e-5, f"fallback basis not orthogonal: {np.max(np.abs(off))}"
    assert np.all(np.diag(G2) > 0.9) and np.all(np.diag(G2) < 1.1), "fallback basis not normalized"

    # deg 5: below the condition limit -> determinant route (unchanged path) vs numpy
    m5 = HilbertRegressor(deg=5).fit(X, y)
    coef = np.polyfit(X.ravel(), y, 5)
    grid = np.linspace(0.1, 0.9, 37).reshape(-1, 1)
    diff = float(np.max(np.abs(m5.predict(grid) - np.polyval(coef, grid.ravel()))))
    assert diff < 1e-6, f"deg-5 fit deviates from numpy.polyfit by {diff}"
    print(f"V2 OK   deg-10 via QR fallback exact ({p10:.12f}); orthogonality max off-diag "
          f"{np.max(np.abs(off)):.2e}; deg-5 (det route) vs polyfit: {diff:.2e}")


def v3_near_collision():
    # two abscissas 9e-5 apart in normalized units (< old absolute spike window 1e-4):
    # hourly timestamps over a multi-year span have exactly this spacing.
    X = np.array([[0.3], [0.3 + 9e-5]])
    y = np.array([1.0, -1.0])
    m = HilbertRegressor(deg=1).fit(X, y)             # k = n: interpolation, no CI (df=0)
    pred = m.predict(X)
    assert abs(pred[0] - 1.0) < 1e-8 and abs(pred[1] + 1.0) < 1e-8, f"preds {pred}"
    assert m.ci_available is False and m.df == 0, "df bookkeeping wrong for n == k"
    print(f"V3 OK   near-colliding abscissas kept distinct: preds = {np.round(pred, 10)}")


def v4_time2interval():
    d0 = datetime(2022, 1, 1)
    dmax = datetime(2024, 12, 31, 23)
    t2i = Time2Interval(min_date=d0, max_date=dmax, lower=0.0, upper=1.0)

    v0 = t2i.date2num(d0)
    v1h = t2i.date2num(d0 + timedelta(hours=1))        # same day, 1 h later
    assert abs(v0 - 0.0) < 1e-12 and v1h > 0.0 and v1h != v0, (v0, v1h)
    v_end = t2i.date2num(dmax)
    assert abs(v_end - 1.0) < 1e-9, f"endpoint mapped to {v_end}"

    # the old `.days` code collapsed this pair: both had .days == 0 from min_date
    assert v1h < 1e-3 and v1h > 0, "hourly resolution should sit near the start of [0,1]"
    print(f"V4 OK   hour 1 -> {v1h:.6e} (distinct from {v0}); endpoint -> {v_end:.12f}")


def v5_score_mse():
    rng = np.random.default_rng(7)
    n = 50
    X = rng.uniform(0, 1, n).reshape(-1, 1)
    y = (5.0 * X.ravel() + 0.3 * rng.standard_normal(n)).ravel()

    m = HilbertRegressor(deg=1).fit(X, y)
    r2_np = float(m.score(X, y))
    # independent numpy reference of the same OLS fit
    A = np.column_stack([np.ones(n), X.ravel()])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2_ref = 1.0 - float(resid @ resid) / float(np.sum((y - y.mean()) ** 2))
    mse_ref = float(np.mean(resid ** 2))

    assert abs(r2_np - r2_ref) < 1e-8, f"score {r2_np} vs R^2 ref {r2_ref}"
    assert 0.0 <= r2_np <= 1.0
    mse = m.mse(X, y)
    assert abs(mse - mse_ref) < 1e-9, f"mse {mse} vs weighted-MSE ref {mse_ref}"
    print(f"V5 OK   score() = R^2 = {r2_np:.6f} (ref {r2_ref:.6f}); "
          f"mse() = {mse:.6e} (old `score` semantics preserved)")


def v6_elec_cost():
    import pandas as pd

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(here, "data", "elec_cost.csv"))
    dates = pd.to_datetime(df.iloc[:, 0])
    yv = df.iloc[:, 2].to_numpy(dtype=float)          # Cost column (col 1 is stored T)

    dmin, dmax = dates.min().to_pydatetime(), dates.max().to_pydatetime()
    t2i = Time2Interval(min_date=dmin, max_date=dmax, lower=0.0, upper=1.0)
    xv = np.array([t2i.date2num(d) for d in dates]).reshape(-1, 1)

    # the stored T column was generated with the old .days code; for midnight
    # timestamps total_seconds()/86400 == .days exactly, so values must coincide:
    t_stored = df.iloc[:, 1].to_numpy(dtype=float)
    assert np.max(np.abs(xv.ravel() - t_stored)) < 1e-9, "Time2Interval drifts from stored T"

    m = HilbertRegressor(deg=5).fit(xv, yv)
    coef = np.polyfit(xv.ravel(), yv, 5)
    diff = float(np.max(np.abs(m.predict(xv) - np.polyval(coef, xv.ravel()))))
    assert diff < 1e-6, f"elec_cost fit deviates from numpy.polyfit by {diff}"
    print(f"V6 OK   elec_cost (n={len(yv)}, deg=5) vs polyfit: max |diff| = {diff:.2e}; "
          f"T-column cross-check passes")


def v7_timeseriescv():
    rng = np.random.default_rng(3)
    n = 100
    tcol = np.arange(n, dtype=float) + rng.normal(0.0, 0.05, n)   # noisy times -> sort matters
    X = np.column_stack([tcol, np.arange(n, dtype=float)])

    # default (n_splits=1): must reproduce the original single-split behavior exactly —
    # first train_ratio of observations in time order train, remainder test.
    cv = TimeSeriesCV()
    assert cv.get_n_splits() == 1
    splits = list(cv.split(X))
    assert len(splits) == 1
    tr, te = splits[0]
    order = np.argsort(tcol, kind="stable")
    assert list(tr) == list(order[:80]), "default single split no longer matches v0.2"
    assert list(te) == list(order[80:]), "default single split no longer matches v0.2"

    # n_splits=3: test region = last 20 (chunks 7/7/6), expanding train, no leakage
    cv3 = TimeSeriesCV(n_splits=3)
    assert cv3.get_n_splits() == 3
    folds = list(cv3.split(X))
    assert len(folds) == 3
    bounds = [80, 87, 94]
    sizes = [7, 7, 6]
    for i, (tr_i, te_i) in enumerate(folds):
        # leak-free: every training observation is strictly earlier than every test one
        assert float(tcol[tr_i].max()) < float(tcol[te_i].min()), f"fold {i}: time leakage"
        assert len(tr_i) == bounds[i], f"fold {i} train size {len(tr_i)} != {bounds[i]}"
        assert len(te_i) == sizes[i], f"fold {i} test size {len(te_i)} != {sizes[i]}"
        # training window is nested: fold i trains on a superset of fold i-1's data
        if i > 0:
            assert set(folds[i - 1][0].tolist()) <= set(tr_i.tolist()), "window not expanding"

    # degenerate / invalid inputs raise cleanly (split is a generator -> iterate to trigger)
    for bad in (lambda: TimeSeriesCV(n_splits=0),
                lambda: list(TimeSeriesCV(index=-1).split(X)),
                lambda: list(TimeSeriesCV(index=3).split(X)),
                lambda: list(TimeSeriesCV(test_ratio=0.05).split(np.ones((4, 2))))):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    # sklearn cloneability: cross_val_score clones the estimator per fold; HilbertRegressor
    # used to fail here because __init__ stored (c_limit+1)/2 instead of the raw parameter.
    from sklearn.base import clone as _clone
    m_c = _clone(HilbertRegressor(deg=1, c_limit=.95))
    assert m_c.c_limit == 0.95 and m_c.deg == 1

    # and it plugs into sklearn's cross_val_score with the package's own regressor
    from sklearn.model_selection import cross_val_score
    yv = np.sin(2.0 * tcol) + 0.05 * rng.standard_normal(n)
    scores = cross_val_score(HilbertRegressor(deg=1), X, yv, cv=cv3)
    assert len(scores) == 3 and all(np.isfinite(s) for s in scores), f"scores {scores}"

    print("V7 OK   TimeSeriesCV default split unchanged; walk-forward n_splits=3 leak-free "
          "(expanding); invalid inputs raise ValueError; estimator is sklearn-cloneable and "
          "cross_val_score integration works")


if __name__ == "__main__":
    v1_ci_band_exact()
    v2_conditioning()
    v3_near_collision()
    v4_time2interval()
    v5_score_mse()
    v6_elec_cost()
    v7_timeseriescv()
    print("\nALL CHECKS PASSED")
