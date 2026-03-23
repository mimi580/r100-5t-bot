"""
R_100 5-TICK SIGNAL ENGINE
============================
Two-layer architecture:

LAYER 1 — Medium-term context (300-tick Fisher + Hurst + MTF)
  Identical to the 5-minute bot. Answers: is there a real trend?
  If no → skip. Direction comes from here.

LAYER 2 — Micro-structure confirmation (15-30 tick window)
  Answers: is right NOW a good entry point within that trend?
  Four models:
    A. Tick Momentum Score  — directional acceleration
    B. Short-Scale ACF      — positive serial correlation (momentum burst)
    C. Volatility Burst     — vol spike in trend direction
    D. Price Z-Score        — price displaced but not overextended

Both layers must agree. Layer 1 sets direction. Layer 2 confirms timing.

WHY TWO LAYERS:
  At 5-tick expiry, the 300-tick Fisher drift is still the real edge —
  it tells us the market has directional bias. But entering at a random
  point within a trend gives a worse expected value than entering when
  the micro-structure confirms momentum is active right now.
  Layer 2 is the entry timing filter.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy import stats as scipy_stats

import settings as S


@dataclass
class ModelResult:
    name:       str
    tradeable:  bool
    confidence: float
    direction:  Optional[str]   # "RISE" | "FALL" | None
    detail:     dict = field(default_factory=dict)


@dataclass
class TickSignal:
    tradeable:    bool
    direction:    Optional[str]
    confidence:   float
    hurst:        float
    p_value:      float
    layer1_score: int
    layer2_score: int
    reasons:      list
    models:       dict = field(default_factory=dict)
    skip_reason:  str = ""


# ─────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────

def _ewma_vol(prices: np.ndarray, halflife: int) -> float:
    rets  = np.diff(prices[-min(len(prices), max(halflife*4, 20)):])
    if len(rets) == 0:
        return 1e-8
    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    var   = float(rets[-1] ** 2)
    for r in reversed(rets[:-1]):
        var = alpha * r**2 + (1.0 - alpha) * var
    return float(np.sqrt(max(var, 1e-16)))


def _acf_lag(rets: np.ndarray, lag: int) -> float:
    if len(rets) <= lag:
        return 0.0
    mu  = float(np.mean(rets))
    var = float(np.var(rets, ddof=1))
    if var == 0:
        return 0.0
    return float(np.mean((rets[:-lag] - mu) * (rets[lag:] - mu)) / var)


# ─────────────────────────────────────────────────────────────────
# ══ LAYER 1 — MEDIUM-TERM CONTEXT ══
# ─────────────────────────────────────────────────────────────────

# ── Volatility regime filter ─────────────────────────────────────

def volatility_filter(prices: np.ndarray) -> Tuple[bool, str]:
    if len(prices) < S.VOL_LONG + 1:
        return True, ""
    vol_s = _ewma_vol(prices[-S.VOL_SHORT:], S.VOL_SHORT // 2)
    vol_l = _ewma_vol(prices[-S.VOL_LONG:],  S.VOL_LONG  // 4)
    if vol_l == 0:
        return True, ""
    ratio = vol_s / vol_l
    if ratio > S.VOL_SPIKE_RATIO:
        return False, f"VOL_SPIKE {ratio:.2f}>{S.VOL_SPIKE_RATIO}"
    return True, ""


# ── Hurst gate ───────────────────────────────────────────────────

def _hurst_rs(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 50:
        return 0.5
    lags, rs_vals = [], []
    for lag in [n//8, n//6, n//4, n//3, n//2, n]:
        if lag < 8:
            continue
        sub   = arr[:lag]
        mean_ = np.mean(sub)
        dev   = np.cumsum(sub - mean_)
        R     = float(np.max(dev) - np.min(dev))
        S_std = float(np.std(sub, ddof=1))
        if S_std == 0 or R == 0:
            continue
        rs_vals.append(np.log(R / S_std))
        lags.append(np.log(lag))
    if len(lags) < 3:
        return 0.5
    A = np.vstack([np.array(lags), np.ones(len(lags))]).T
    try:
        h, _ = np.linalg.lstsq(A, np.array(rs_vals), rcond=None)[0]
        return float(np.clip(h, 0.0, 1.0))
    except Exception:
        return 0.5


def hurst_gate(prices: np.ndarray) -> Tuple[float, bool, str]:
    if len(prices) < S.HURST_WINDOW + 1:
        return 0.5, True, ""
    rets = np.diff(prices[-S.HURST_WINDOW:])
    h    = _hurst_rs(rets)
    if h < S.HURST_BLOCK:
        return h, False, f"HURST_LOW H={h:.4f}<{S.HURST_BLOCK}"
    return h, True, ""


# ── Fisher combined test ─────────────────────────────────────────

def fisher_combined_test(prices: np.ndarray) -> Tuple[Optional[str], float, dict]:
    """
    Three independent tests combined via Fisher's method.
    Same implementation as the 5-minute bot.
    300-tick window, p_threshold=0.04 (tighter than 5-min's 0.05).
    """
    if len(prices) < S.FISHER_WINDOW + 5:
        return None, 1.0, {"reason": "INSUFFICIENT_DATA"}

    rets = np.diff(prices[-S.FISHER_WINDOW:])
    n    = len(rets)

    # Test 1: t-test
    _, p_t = scipy_stats.ttest_1samp(rets, 0)
    p_t    = float(p_t)

    # Test 2: Binomial sign test
    n_pos  = int(np.sum(rets > 0))
    p_sign = float(scipy_stats.binomtest(n_pos, n, 0.5).pvalue)

    # Test 3: Wald-Wolfowitz runs test
    signs = np.sign(rets); signs = signs[signs != 0]
    ns    = len(signs)
    if ns < 10:
        p_runs = 0.5
    else:
        n1, n2 = int(np.sum(signs > 0)), int(np.sum(signs < 0))
        nn     = n1 + n2
        if n1 == 0 or n2 == 0:
            p_runs = 0.5
        else:
            n_runs = 1 + int(np.sum(signs[:-1] != signs[1:]))
            mu_r   = 2.0 * n1 * n2 / nn + 1.0
            var_r  = (2.0 * n1 * n2 * (2.0 * n1 * n2 - nn) /
                      (nn**2 * (nn - 1) + 1e-8))
            z_r    = (n_runs - mu_r) / float(np.sqrt(max(var_r, 1e-8)))
            p_runs = float(2.0 * scipy_stats.norm.sf(abs(z_r)))

    pvals     = [max(float(p), 1e-10) for p in [p_t, p_sign, p_runs]]
    chi2_stat = float(-2.0 * np.sum(np.log(pvals)))
    p_fisher  = float(scipy_stats.chi2.sf(chi2_stat, df=6))

    mean_ret  = float(np.mean(rets))
    direction = "RISE" if mean_ret > 0 else "FALL"

    details = {
        "p_ttest":  round(p_t, 5),
        "p_sign":   round(p_sign, 5),
        "p_runs":   round(p_runs, 5),
        "p_fisher": round(p_fisher, 6),
        "chi2":     round(chi2_stat, 3),
        "mean_ret": round(mean_ret, 6),
        "n_pos":    n_pos,
        "n":        n,
    }

    if p_fisher >= S.FISHER_P_THRESHOLD:
        return None, p_fisher, {**details, "reason": f"P_LARGE {p_fisher:.4f}"}

    return direction, p_fisher, details


# ── MTF direction check ──────────────────────────────────────────

def mtf_check(prices: np.ndarray, required_dir: str) -> Tuple[bool, str]:
    agree = total = 0
    for w in S.MTF_WINDOWS:
        if len(prices) < w + 1:
            continue
        d = "RISE" if np.mean(np.diff(prices[-w:])) > 0 else "FALL"
        total += 1
        if d == required_dir:
            agree += 1
    if total == 0:
        return True, "0/0"
    if agree < S.MTF_MIN_AGREE:
        return False, f"MTF_SPLIT {agree}/{total} agree with {required_dir}"
    return True, f"MTF {agree}/{total}"


# ─────────────────────────────────────────────────────────────────
# ══ LAYER 2 — MICRO-STRUCTURE CONFIRMATION ══
# ─────────────────────────────────────────────────────────────────

# ── Model A: Tick Momentum Score ─────────────────────────────────

def model_momentum(prices: np.ndarray, required_dir: str) -> ModelResult:
    """
    Exponentially weighted momentum of last MOMENTUM_WINDOW ticks.
    Measures: are recent ticks accelerating in the required direction?

    Two checks:
    1. Weighted momentum (all window ticks) must be in required direction
       and exceed MOMENTUM_MIN magnitude
    2. Acceleration: recent half must have stronger momentum than older half
       (price is picking up speed, not slowing down)
    """
    name = "momentum"
    fail = ModelResult(name, False, 0.0, None)

    if len(prices) < S.MOMENTUM_WINDOW + 1:
        return fail

    rets  = np.diff(prices[-S.MOMENTUM_WINDOW:])
    n     = len(rets)
    alpha = 1.0 - np.exp(-np.log(2) / S.MOMENTUM_HALFLIFE)

    # Exponentially weighted mean — recent ticks dominate
    weights = np.array([alpha * (1 - alpha)**i for i in range(n)])[::-1]
    weights /= weights.sum()
    momentum = float(np.dot(weights, rets))

    # Direction check
    mom_dir = "RISE" if momentum > 0 else "FALL"
    if mom_dir != required_dir:
        return ModelResult(name, False, 0.0, None, {
            "momentum": round(momentum, 6),
            "mom_dir":  mom_dir,
            "reason":   f"WRONG_DIR {mom_dir}!={required_dir}",
        })

    if abs(momentum) < S.MOMENTUM_MIN:
        return ModelResult(name, False, 0.0, None, {
            "momentum": round(momentum, 6),
            "reason":   "MOMENTUM_WEAK",
        })

    # Acceleration: compare recent half vs older half
    half      = n // 2
    mom_old   = float(np.mean(rets[:half]))
    mom_new   = float(np.mean(rets[half:]))
    # Accelerating = new momentum in same direction and stronger
    accel     = (mom_new - mom_old) if required_dir == "RISE" else (mom_old - mom_new)
    if accel < S.MOMENTUM_ACCEL_MIN:
        return ModelResult(name, False, 0.0, None, {
            "momentum": round(momentum, 6),
            "accel":    round(accel, 6),
            "reason":   "DECELERATING",
        })

    conf = float(np.clip(abs(momentum) / (S.MOMENTUM_MIN * 5.0 + 1e-8), 0.0, 1.0))

    return ModelResult(name, True, conf, required_dir, {
        "momentum": round(momentum, 6),
        "accel":    round(accel, 6),
        "mom_old":  round(mom_old, 6),
        "mom_new":  round(mom_new, 6),
    })


# ── Model B: Short-Scale ACF ─────────────────────────────────────

def model_acf_short(prices: np.ndarray, required_dir: str) -> ModelResult:
    """
    Lag-1 and lag-2 autocorrelation on last ACF_SHORT_WINDOW ticks.

    At 5-tick scale, POSITIVE ACF-1 indicates momentum burst:
    each tick continues the previous one (serial correlation).
    This is the OPPOSITE of what EXPIRYRANGE wants (negative ACF).

    For a 5-tick RISE/FALL bet we want:
      ACF-1 > ACF_LAG1_MIN  (positive — ticks are trending)
      ACF-2 > ACF_LAG2_MIN  (positive — multi-lag persistence)

    Direction is confirmed by the sign of the mean return.
    """
    name = "acf_short"
    fail = ModelResult(name, False, 0.0, None)

    if len(prices) < S.ACF_SHORT_WINDOW + 2:
        return fail

    rets  = np.diff(prices[-S.ACF_SHORT_WINDOW:])
    acf1  = _acf_lag(rets, 1)
    acf2  = _acf_lag(rets, 2)

    if acf1 < S.ACF_LAG1_MIN or acf2 < S.ACF_LAG2_MIN:
        return ModelResult(name, False, 0.0, None, {
            "acf1":   round(acf1, 5),
            "acf2":   round(acf2, 5),
            "reason": f"ACF_LOW acf1={acf1:.3f} acf2={acf2:.3f}",
        })

    # Direction from mean return
    mean_ret  = float(np.mean(rets))
    direction = "RISE" if mean_ret > 0 else "FALL"

    if direction != required_dir:
        return ModelResult(name, False, 0.0, None, {
            "acf1":    round(acf1, 5),
            "acf2":    round(acf2, 5),
            "acf_dir": direction,
            "reason":  f"WRONG_DIR {direction}!={required_dir}",
        })

    conf = float(np.clip(acf1 * 10.0, 0.0, 1.0))

    return ModelResult(name, True, conf, direction, {
        "acf1":     round(acf1, 5),
        "acf2":     round(acf2, 5),
        "mean_ret": round(mean_ret, 7),
    })


# ── Model C: Volatility Burst ────────────────────────────────────

def model_vol_burst(prices: np.ndarray, required_dir: str) -> ModelResult:
    """
    Detects sudden volatility expansion in the required direction.

    A vol burst = short EWMA vol / medium EWMA vol > VOL_BURST_RATIO.
    When a burst fires, the direction of the burst (sign of recent returns)
    must match the required direction.

    Rationale: vol bursts on R_100 tend to continue for a few ticks
    before mean-reverting. A 5-tick contract entered at burst onset
    captures that initial continuation.
    """
    name = "vol_burst"
    fail = ModelResult(name, False, 0.0, None)

    if len(prices) < S.VOL_BURST_MIN_TICKS + 1:
        return fail

    vol_short  = _ewma_vol(prices, S.VOL_BURST_SHORT)
    vol_medium = _ewma_vol(prices, S.VOL_BURST_MEDIUM)

    if vol_medium == 0:
        return fail

    ratio = vol_short / vol_medium

    if ratio < S.VOL_BURST_RATIO:
        return ModelResult(name, False, 0.0, None, {
            "ratio":  round(ratio, 3),
            "reason": f"NO_BURST {ratio:.2f}<{S.VOL_BURST_RATIO}",
        })

    # Direction of the burst = sign of last 3 ticks
    last_rets = np.diff(prices[-4:])
    if len(last_rets) == 0:
        return fail

    burst_dir = "RISE" if float(np.mean(last_rets)) > 0 else "FALL"

    if burst_dir != required_dir:
        return ModelResult(name, False, 0.0, None, {
            "ratio":     round(ratio, 3),
            "burst_dir": burst_dir,
            "reason":    f"BURST_WRONG_DIR {burst_dir}!={required_dir}",
        })

    conf = float(np.clip((ratio - S.VOL_BURST_RATIO) / (1.0 + 1e-6), 0.0, 1.0))

    return ModelResult(name, True, conf, burst_dir, {
        "ratio":     round(ratio, 3),
        "vol_short": round(vol_short, 7),
        "vol_med":   round(vol_medium, 7),
    })


# ── Model D: Price Level Z-Score (micro) ─────────────────────────

def model_zscore_micro(prices: np.ndarray, required_dir: str) -> ModelResult:
    """
    Short-window Z-score of current price vs recent mean.

    Confirms that price is displaced from its micro mean in the
    required direction — slightly elevated for RISE, slightly
    depressed for FALL. Not overextended (too far = likely reversal).

    ZSCORE_MICRO_MIN: must have some displacement (price is moving)
    ZSCORE_MICRO_MAX: must not be overextended (price hasn't run too far)

    This is a timing quality filter — we want to enter when the trend
    is active but not exhausted.
    """
    name = "zscore_micro"
    fail = ModelResult(name, False, 0.0, None)

    if len(prices) < S.ZSCORE_MICRO_WINDOW + 1:
        return fail

    window  = prices[-S.ZSCORE_MICRO_WINDOW:]
    mean_p  = float(np.mean(window))
    std_p   = float(np.std(window, ddof=1))
    current = float(prices[-1])

    if std_p == 0:
        return fail

    z = (current - mean_p) / std_p

    # Direction from Z-score sign
    z_dir = "RISE" if z > 0 else "FALL"

    if z_dir != required_dir:
        return ModelResult(name, False, 0.0, None, {
            "z":      round(z, 3),
            "z_dir":  z_dir,
            "reason": f"WRONG_DIR {z_dir}!={required_dir}",
        })

    abs_z = abs(z)
    if abs_z < S.ZSCORE_MICRO_MIN:
        return ModelResult(name, False, 0.0, None, {
            "z":      round(z, 3),
            "reason": f"Z_TOO_SMALL {abs_z:.3f}<{S.ZSCORE_MICRO_MIN}",
        })

    if abs_z > S.ZSCORE_MICRO_MAX:
        return ModelResult(name, False, 0.0, None, {
            "z":      round(z, 3),
            "reason": f"Z_OVEREXTENDED {abs_z:.3f}>{S.ZSCORE_MICRO_MAX}",
        })

    # Confidence: best in middle of range
    mid  = (S.ZSCORE_MICRO_MIN + S.ZSCORE_MICRO_MAX) / 2
    conf = float(np.clip(
        1.0 - abs(abs_z - mid) / (mid - S.ZSCORE_MICRO_MIN + 1e-6),
        0.0, 1.0))

    return ModelResult(name, True, conf, required_dir, {
        "z":       round(z, 3),
        "mean_p":  round(mean_p, 4),
        "std_p":   round(std_p, 5),
    })


# ─────────────────────────────────────────────────────────────────
# MAIN EVALUATE — TWO-LAYER CONFLUENCE
# ─────────────────────────────────────────────────────────────────

def evaluate(prices: np.ndarray) -> TickSignal:
    """
    Two-layer evaluation:

    LAYER 1 (medium-term context):
      1. Volatility filter
      2. Hurst gate
      3. Fisher combined test → direction + p-value
      4. MTF direction check

    LAYER 2 (micro-structure confirmation):
      All 4 models receive the direction from Layer 1.
      They confirm whether RIGHT NOW is a good entry.
      LAYER2_REQUIRED models must vote in the same direction.

    Both layers must pass for a trade to fire.
    """
    def skip(reason, h=0.5, p=1.0):
        return TickSignal(False, None, 0.0, h, p, 0, 0, [],
                          skip_reason=reason)

    # ── Layer 1 ───────────────────────────────────────────────────

    ok, reason = volatility_filter(prices)
    if not ok:
        return skip(reason)

    h, h_pass, h_reason = hurst_gate(prices)
    if not h_pass:
        return skip(h_reason, h=h)

    direction, p_val, fisher_det = fisher_combined_test(prices)
    if direction is None:
        return skip(f"FISHER p={p_val:.4f}>={S.FISHER_P_THRESHOLD}", h=h, p=p_val)

    mtf_ok, mtf_info = mtf_check(prices, direction)
    if not mtf_ok:
        return skip(mtf_info, h=h, p=p_val)

    # Layer 1 passed — direction established
    l1_reasons = [
        f"H={h:.3f}",
        f"p={p_val:.5f}",
        f"chi2={fisher_det.get('chi2',0):.1f}",
        f"{mtf_info}",
    ]

    # ── Layer 2 ───────────────────────────────────────────────────

    mA = model_momentum(prices,    direction)
    mB = model_acf_short(prices,   direction)
    mC = model_vol_burst(prices,   direction)
    mD = model_zscore_micro(prices, direction)

    l2_models = {"momentum": mA, "acf_short": mB, "vol_burst": mC, "zscore_micro": mD}
    l2_passing = {k: v for k, v in l2_models.items() if v.tradeable}
    l2_score   = len(l2_passing)

    l2_reasons = [
        f"{k}:{'OK' if v.tradeable else 'NO'}({v.confidence:.2f})"
        for k, v in l2_models.items()
    ]

    if l2_score < S.LAYER2_REQUIRED:
        return TickSignal(
            False, None, 0.0, h, p_val, 4, l2_score,
            l1_reasons + l2_reasons,
            models={"fisher": fisher_det,
                    **{k: v.detail for k, v in l2_models.items()}},
            skip_reason=f"LAYER2_WEAK {l2_score}/{len(l2_models)}"
        )

    # All passing Layer 2 models must agree with required direction
    wrong_dir = [k for k, v in l2_passing.items()
                 if v.direction and v.direction != direction]
    if wrong_dir:
        return TickSignal(
            False, None, 0.0, h, p_val, 4, l2_score,
            l1_reasons + l2_reasons,
            models={"fisher": fisher_det,
                    **{k: v.detail for k, v in l2_models.items()}},
            skip_reason=f"LAYER2_DIR_MISMATCH {wrong_dir}"
        )

    # ── Final confidence ──────────────────────────────────────────
    # Layer 1 confidence from p-value
    denom = np.log(0.001 / S.FISHER_P_THRESHOLD)
    l1_conf = float(np.clip(
        1.0 - np.log(max(p_val, 1e-6) / S.FISHER_P_THRESHOLD) / (denom + 1e-8),
        0.0, 1.0))

    # Layer 2 confidence from passing models
    l2_conf = float(np.mean([v.confidence for v in l2_passing.values()]))

    # Combined: Layer 1 weighted 60%, Layer 2 weighted 40%
    confidence = round(float(0.6 * l1_conf + 0.4 * l2_conf), 4)

    all_reasons = l1_reasons + l2_reasons

    return TickSignal(
        tradeable    = True,
        direction    = direction,
        confidence   = confidence,
        hurst        = round(h, 4),
        p_value      = round(p_val, 6),
        layer1_score = 4,
        layer2_score = l2_score,
        reasons      = all_reasons,
        models       = {"fisher": fisher_det,
                        **{k: v.detail for k, v in l2_models.items()}}
    )
