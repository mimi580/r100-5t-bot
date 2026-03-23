"""
R_100 5-TICK RISE/FALL BOT — SETTINGS
========================================
Contract : CALL (Rise) / PUT (Fall) on R_100
Expiry   : 5 ticks (~5 seconds)
Symbol   : R_100 (Volatility 100 Index)

TWO-LAYER SIGNAL ARCHITECTURE:
  Layer 1 — Medium-term context (300-tick lookback):
    Fisher combined test  — is there statistically significant drift?
    Hurst gate            — is market trending not mean-reverting?
    MTF direction check   — do short/medium/long windows agree?

  Layer 2 — Micro-structure confirmation (15-30 tick lookback):
    Tick Momentum Score   — directional acceleration right now
    Short-Scale ACF       — are ticks serially correlated (momentum burst)?
    Volatility Burst      — is a vol spike happening in predicted direction?
    Price Level Z-Score   — micro mean-reversion confirmation

  Both layers must agree on direction before a trade fires.
  Layer 1 sets the context. Layer 2 confirms the entry timing.

PAYOUT NOTE:
  5-tick contracts pay ~60-70% on Deriv (vs ~80-90% for 5-minute).
  Break-even WR shifts from ~53% to ~59-62%.
  Fisher p-threshold tightened to 0.04 (vs 0.05 for 5-minute) to
  demand stronger statistical evidence before entering.
"""

import os

# ── Connection ────────────────────────────────────────────────────
DERIV_APP_ID    = "1089"
DERIV_API_TOKEN = os.environ.get("DERIV_API_TOKEN", "")
DERIV_WS_URL    = "wss://ws.derivws.com/websockets/v3"

# ── Contract ──────────────────────────────────────────────────────
SYMBOL          = "R_100"
EXPIRY_TICKS    = 5
EXPIRY_UNIT     = "t"

# ── Warmup ────────────────────────────────────────────────────────
# Must have 300 ticks before ANY model fires.
# Layer 1 models need 300 ticks. Layer 2 models need 30.
# Warmup ensures both are satisfied from the start.
WARMUP_TICKS    = 300

# ── Tick buffer ───────────────────────────────────────────────────
TICK_BUFFER     = 1000

# ─────────────────────────────────────────────────────────────────
# LAYER 1 — MEDIUM-TERM CONTEXT (300-tick window)
# Identical calibration to the 5-minute bot.
# Lookback window is the same — only expiry changes.
# ─────────────────────────────────────────────────────────────────

# ── Fisher combined test ──────────────────────────────────────────
# Tighter than 5-minute (0.05) because 5-tick payout is lower.
# At p=0.04: fires ~24% on strong trend, ~5% on random walk (4.8x ratio)
FISHER_P_THRESHOLD  = 0.04
FISHER_WINDOW       = 300      # ticks of history for Fisher test

# ── Hurst gate ────────────────────────────────────────────────────
# Block mean-reverting markets. Computed on RETURNS (not prices).
# Same threshold as 5-minute bot.
HURST_WINDOW        = 300
HURST_BLOCK         = 0.40

# ── Multi-timeframe direction check ───────────────────────────────
# Same windows as 5-minute bot — checking consistency at 1/2/5 min scales
MTF_WINDOWS         = [60, 150, 300]
MTF_MIN_AGREE       = 2        # of 3 windows must agree

# ── Volatility regime filter ──────────────────────────────────────
VOL_SHORT           = 30
VOL_LONG            = 150
VOL_SPIKE_RATIO     = 1.6

# ─────────────────────────────────────────────────────────────────
# LAYER 2 — MICRO-STRUCTURE CONFIRMATION (15-30 tick window)
# These models look at what the market is doing RIGHT NOW.
# They confirm entry timing within the trend Layer 1 identified.
# ─────────────────────────────────────────────────────────────────

# ── Model A: Tick Momentum Score ──────────────────────────────────
# Exponentially weighted sum of recent returns.
# Measures directional consistency + acceleration over last N ticks.
MOMENTUM_WINDOW     = 15       # ticks
MOMENTUM_HALFLIFE   = 3        # ticks — recent returns weighted heavily
MOMENTUM_MIN        = 0.008    # minimum weighted momentum magnitude to vote
MOMENTUM_ACCEL_MIN  = 0.0      # acceleration (recent half > older half) required

# ── Model B: Short-Scale ACF (Lag-1 and Lag-2) ───────────────────
# At 5-tick scale, POSITIVE ACF-1 means momentum burst (ticks trending).
# Opposite of what we want for EXPIRYRANGE (which needs negative ACF).
# Both lag-1 and lag-2 must be positive for momentum confirmation.
ACF_SHORT_WINDOW    = 30       # ticks
ACF_LAG1_MIN        = 0.03     # ACF-1 must exceed this (positive = momentum)
ACF_LAG2_MIN        = 0.0      # ACF-2 must be positive (consistency)

# ── Model C: Volatility Burst ─────────────────────────────────────
# Detects sudden vol expansion in predicted direction.
# Short EWMA vol vs medium EWMA vol — spike = burst.
# Direction of burst must match Layer 1 direction.
VOL_BURST_SHORT     = 5        # ticks (halflife)
VOL_BURST_MEDIUM    = 20       # ticks (halflife)
VOL_BURST_RATIO     = 1.5      # short/medium must exceed to detect burst
VOL_BURST_MIN_TICKS = 25       # minimum ticks before burst model activates

# ── Model D: Price Level Z-Score (micro) ─────────────────────────
# Short-window Z-score of current price vs recent mean.
# Used as confirmation: price slightly displaced from mean in
# the direction of the trend = good entry (not overextended).
# Price too far from mean = overextended = skip.
ZSCORE_MICRO_WINDOW = 20       # ticks
ZSCORE_MICRO_MIN    = 0.1      # minimum displacement (price moving)
ZSCORE_MICRO_MAX    = 2.2      # maximum displacement (not overextended)

# ── Layer 2 confluence ────────────────────────────────────────────
# How many Layer 2 models must vote in the same direction as Layer 1
LAYER2_REQUIRED     = 2        # of 4 micro models must confirm
LAYER2_DIRECTION_MATCH = True  # all voting Layer 2 models must agree with Layer 1

# ─────────────────────────────────────────────────────────────────
# TRADE MANAGEMENT
# ─────────────────────────────────────────────────────────────────

# ── Martingale ────────────────────────────────────────────────────
FIRST_STAKE         = 0.35
MARTINGALE_FACTOR   = 1.89
MARTINGALE_AFTER    = 1
MARTINGALE_MAX_STEP = 4

# ── Risk ──────────────────────────────────────────────────────────
TARGET_PROFIT       = 30.0
STOP_LOSS           = 40.0

# ── Cooldown ──────────────────────────────────────────────────────
# 5 ticks settle in ~5 seconds. Short cooldown appropriate.
# But don't trade every 5 seconds — need genuine condition change.
COOLDOWN_MIN        = 15       # seconds
COOLDOWN_MAX        = 30       # seconds

# ── Reconnect ─────────────────────────────────────────────────────
RECONNECT_BASE      = 3
RECONNECT_MAX       = 60

# ── Telegram ──────────────────────────────────────────────────────
TELEGRAM_TOKEN      = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Logging ───────────────────────────────────────────────────────
LOG_FILE            = "/tmp/r100_5t_bot.log"
TRADES_FILE         = "/tmp/r100_5t_trades.csv"
STATS_FILE          = "/tmp/r100_5t_stats.json"
