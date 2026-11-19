# GTA VI Trophy Rarity Forecaster — make.fun `gta6-trophy-rarity-forecast-01`

Forecasts every GTA VI PS5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z
as five quantiles (5 / 25 / 50 / 75 / 95%), scored by mean pinball loss on
the logit scale.

## Method

An earn rate is earners / players-tracked, and the two move in opposite
directions: players-tracked grows as more players register on Exophase
(dilution pulls every rate down) while earners grow as players progress
(pulls rates up). Fitting the rate alone confuses the two — a refresh that
jumps players-tracked looks like a sudden fall in every rate.

So for each trophy the program fits, on the case's distinct snapshots
(consecutive identical reads collapsed), recency-weighted least squares of
log(earners) and of log(players) against log(1 + hours since release), plus
a direct logit(rate) trend fit, then forecasts earners and players at the
target and divides. Slopes come from the last 24 hours of snapshots only:
the growth curve is concave in log time (adoption saturates), so a slope
fitted over the whole history over-extrapolates. The median blends the
trend forecast 60% with persistence (the last read) 40%.

Quantiles are normal on the logit scale, with a per-trophy spread from fit
residuals, horizon, extrapolation distance and dilution volatility. When
the history is completely flat (the launch-night practice cases) no trend
can be read, so the median uses the dilution the rules record for
Exophase's first refresh (players 638 → 3868, every rate but the 100% one
down), smaller for easier trophies, with a wide spread.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic, standard library only, ~0.2 s per case.

## Local verification

- Practice cases p00/p01: exit 0, valid output (55 rows, monotone
  quantiles, 3229 bytes), two runs byte-identical.
- Practice pinball loss vs the dilution scenario the rules imply
  (rates × 638/3868 at the first refresh): **0.28** vs house baseline
  **1.69** and naive persistence **1.78**.
- Synthetic final-like cases (2.5 days of history, 4.5-day horizon,
  refresh plateaus, three worlds + two noisy worlds): **0.17** vs house
  baseline **0.27**.
