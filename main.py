#!/usr/bin/env python3
"""GTA VI PS5 trophy earn-rate forecaster (gta6-trophy-rarity-forecast-01).

Method -- earners / players decomposition, fitted per trophy in log time:
  * A trophy's Exophase earn rate is earners / players-tracked. Both move:
    players-tracked grows as more players register on Exophase (dilution
    pulls every rate down), while earners grow as players progress through
    the game (pulls rates up). Fitting the rate alone confuses the two: a
    refresh that jumps players-tracked looks like a sudden fall in every
    rate, and a pure rate trend extrapolates that fall.
  * So for each trophy we fit, on the case's distinct snapshots
    (consecutive identical reads collapsed), recency-weighted least squares
    of log(earners) and of log(players) against log(1 + hours since the
    2026-11-19T00:00:00Z release), and of logit(rate) against the same
    clock, then forecast earners and players at the target and divide.
    The slopes come from the last 24 hours of snapshots only: the
    growth curve is concave in log time (adoption saturates), so a slope
    fitted over the whole history over-extrapolates — on synthetic
    final-like cases the full-history slope overshot the true move by
    more than 2x, the 24-hour slope did not.
  * The median is a shrinkage blend of the decomposition forecast, the
    direct logit-trend forecast and persistence (the last read), so a wild
    extrapolation is pulled back toward what was last seen.
  * Quantiles are normal on the logit scale: q_tau = sigmoid(median_logit
    + z_tau * sigma), with a per-trophy sigma from fit residuals,
    extrapolation distance, horizon and recent dilution volatility.
  * When the history is completely flat (the launch-night practice cases),
    no trend can be read; the rules record that Exophase's first refresh
    took players-tracked 638 -> 3868 and every rate but the one at 100%
    down, so the median is shifted down by a dilution factor that is
    smaller for easier (higher-rate) trophies, with a wide spread.
    The factor is 0.165 * (1 + 1.5 * r / r_max): pure dilution would be
    0.165 for every trophy, while earners also grew while the page was
    frozen, more for easier trophies. The constants were picked by
    scoring the practice cases against the dilution scenarios the rules
    imply (earners frozen, up to 3x, up to 5x growth), choosing the
    setting with the best worst case across them, not the best on any
    single scenario.
  * Deterministic: pure arithmetic, no randomness, fixed output ordering.
  * Robust: missing trophies, 0% / 100% rates and empty histories are all
    handled; output is always one complete row per trophy in the case.
"""

import json
import math
import os
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Recency half-life (hours) for the weighted fits: the recent regime
# matters most, but the early shape still anchors the log-time curve.
HALF_LIFE_H = 12.0
# Blend weights for the median (logit space).
W_TREND = 0.6
W_PERSIST = 0.4


def parse_time(s):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_rate(r):
    return min(99.99, max(0.01, r))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    return round(x * 10000.0) / 10000.0


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, resid_sd, n, sxx). resid_sd is the weighted residual
    standard deviation (0 when the fit is exact or degenerate).
    """
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.0, 0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, n, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        ss = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        sd = math.sqrt(max(ss, 0.0) / (sw * (n - 2) / n)) if sw > 0 else 0.0
        # Unweighted-style scale: use plain mean of squared residuals.
        ss2 = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(n - 2, 1)
        sd = math.sqrt(max(ss2, 0.0))
    else:
        sd = 0.0
    return a, b, sd, n, sxx


def distinct_snapshots(history):
    """Collapse consecutive reads with identical players + rates.

    Exophase only refreshes occasionally, so most reads repeat the last
    snapshot verbatim; keeping them all would overweight long plateaus.
    Returns [(at_datetime, players_tracked, rates_tuple)] oldest first.
    """
    out = []
    prev_key = None
    for h in history:
        try:
            at = parse_time(h["at"])
        except Exception:
            continue
        players = h.get("players_tracked")
        rates = h.get("rates")
        if players is None or rates is None:
            continue
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        out.append((at, players, tuple(rates)))
    return out


def flat_branch(last_rate, r_max, tier):
    """Forecast when the history shows no movement at all.

    No trend can be read, so fall back on the launch-night refresh the
    rules record: players-tracked jumped 638 -> 3868 and every rate but
    the one at 100% fell. Pure dilution would multiply every rate by
    638/3868 = 0.165; earners also grew while the page was frozen, more
    for easier trophies, so the factor is 0.165 * (1 + 4 * r / r_max),
    capped at 0.9. Wide spread: the refresh is a jump we cannot time.
    """
    if last_rate >= 99.99:
        return logit(99.95), 0.15
    if last_rate <= 0.01:
        return logit(0.05), 1.2
    dil = 0.165 * (1.0 + 1.5 * last_rate / max(r_max, 0.01))
    dil = min(dil, 0.9)
    return logit(last_rate * dil), 0.6


TIER_PRIOR = {"platinum": 0.5, "gold": 0.8, "silver": 0.7, "bronze": 1.0}


def forecast_trophy(idx, trophies, snaps, last_rate, n_fit, x_target,
                    x_last, horizon_days, r_max):
    """Return (median_logit, sigma) for trophy index idx."""
    tier = trophies[idx].get("tier", "bronze")
    if last_rate is None:
        # Trophy never observed in any read: tier prior, very wide.
        return logit(TIER_PRIOR.get(tier, 1.0)), 1.2
    if last_rate >= 99.99:
        # Earned by essentially everyone tracked (e.g. a start-the-game
        # trophy): it stays at the top, tiny spread.
        return logit(99.95), 0.15

    pts = []
    for at, players, rates in snaps:
        r = rates[idx]
        if r is None or players is None or players <= 0:
            continue
        t_h = (at - RELEASE).total_seconds() / 3600.0
        x = math.log(max(1.0 + t_h, 0.05))
        pts.append((x, r, players, at))
    if len(pts) < 3:
        return flat_branch(last_rate, r_max, tier)
    rates_seen = {p[1] for p in pts}
    players_seen = {p[2] for p in pts}
    if len(rates_seen) <= 1 and len(players_seen) <= 1:
        return flat_branch(last_rate, r_max, tier)

    t_last = pts[-1][3]
    # Slope from the recent window only: the growth curve is concave in
    # log time (adoption saturates), so a slope fitted over the whole
    # history over-extrapolates. Fall back to all points when the recent
    # window is too thin.
    recent = [p for p in pts
              if (t_last - p[3]).total_seconds() <= 24.0 * 3600.0]
    fit_pts = recent if len(recent) >= 4 else pts
    xs, ys_logit, ys_earn, ws = [], [], [], []
    for x, r, players, at in fit_pts:
        w = 0.5 ** ((t_last - at).total_seconds() / 3600.0 / HALF_LIFE_H)
        xs.append(x)
        ys_logit.append(logit(r))
        earners = max(r / 100.0 * players, 0.5)
        ys_earn.append(math.log(earners))
        ws.append(w)

    a_d, b_d, s_d, _, _ = wls(xs, ys_logit, ws)
    a_e, b_e, s_e, _, _ = wls(xs, ys_earn, ws)
    # Earners and players-tracked can only grow; a negative fitted slope
    # is rounding noise on 2-decimal rates, not a real decline.
    b_e = min(max(b_e, 0.0), 4.0)
    a_n, b_n, s_n, _, _ = n_fit
    b_n = min(max(b_n, 0.0), 4.0)

    # Decomposition forecast: earners(target) / players(target).
    ln_e_t = a_e + b_e * x_target
    ln_n_t = a_n + b_n * x_target
    m_dec = logit(100.0 * math.exp(ln_e_t - ln_n_t))
    # Direct logit-trend forecast (the house-curve shape, full history).
    m_dir = a_d + b_d * x_target
    m_per = logit(last_rate)
    m_trend = 0.5 * (m_dec + m_dir)
    m = W_TREND * m_trend + W_PERSIST * m_per

    # Spread: fit noise + random-walk horizon term + extrapolation
    # distance in log time + dilution volatility from players growth.
    gap = max(x_target - x_last, 0.0)
    s_fit = min(max(0.5 * (s_d + s_e), 0.10), 1.0)
    s_hor = 0.20 * math.sqrt(max(horizon_days, 0.0))
    s_ext = 0.25 * gap
    s_dil = 0.15 * min(max(b_n, 0.0), 1.0)
    sigma = math.sqrt(0.25 ** 2 + s_fit ** 2 + s_hor ** 2
                      + s_ext ** 2 + s_dil ** 2)
    sigma = min(max(sigma, 0.30), 1.6)
    m = min(max(m, logit(0.02)), logit(99.98))
    return m, sigma


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = math.log(max(1.0 + (target - RELEASE).total_seconds() / 3600.0,
                            0.05))
    snaps = distinct_snapshots(history)
    if not snaps:
        # No usable history at all: tier priors, very wide.
        rows = []
        for t in trophies:
            m = logit(TIER_PRIOR.get(t.get("tier", "bronze"), 1.0))
            rows.append((t["id"], m, 1.2))
        return rows

    x_last = math.log(max(1.0 + (snaps[-1][0] - RELEASE).total_seconds()
                          / 3600.0, 0.05))
    # Global players-tracked fit: log(N) vs log(1 + hours since release).
    xs_n, ys_n, ws_n = [], [], []
    t_last = snaps[-1][0]
    recent_snaps = [s for s in snaps
                    if (t_last - s[0]).total_seconds() <= 24.0 * 3600.0]
    fit_snaps = recent_snaps if len(recent_snaps) >= 4 else snaps
    for at, players, _rates in fit_snaps:
        if players is None or players <= 0:
            continue
        t_h = (at - RELEASE).total_seconds() / 3600.0
        xs_n.append(math.log(max(1.0 + t_h, 0.05)))
        ys_n.append(math.log(players))
        ws_n.append(0.5 ** ((t_last - at).total_seconds() / 3600.0
                            / HALF_LIFE_H))
    n_fit = wls(xs_n, ys_n, ws_n) if len(xs_n) >= 2 else (math.log(638.0),
                                                         0.0, 0.0, 0, 0.0)

    last_rates = snaps[-1][2]
    seen_rates = [r for _at, _p, rates in snaps for r in rates
                  if r is not None and r < 99.99]
    r_max = max(seen_rates) if seen_rates else 1.0

    rows = []
    for i, t in enumerate(trophies):
        m, sigma = forecast_trophy(i, trophies, snaps, last_rates[i], n_fit,
                                   x_target, x_last, horizon_days, r_max)
        rows.append((t["id"], m, sigma))
    return rows


def quantiles(m, sigma):
    qs = []
    for z in Z:
        qs.append(round4(clamp_rate(sigmoid(m + z * sigma))))
    # Rounding can only break monotonicity at the extremes; enforce it.
    for j in range(1, len(qs)):
        if qs[j] < qs[j - 1]:
            qs[j] = qs[j - 1]
    return qs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        cid = fname[:-len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        rows = forecast_case(case)
        trophies = [{"id": tid, "q": quantiles(m, s)} for tid, m, s in rows]
        out = {"trophies": trophies}
        with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
