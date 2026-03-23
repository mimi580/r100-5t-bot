"""Logger, CSV trade recorder, Telegram alerter."""

import csv, json, logging, logging.handlers, threading, urllib.request, urllib.parse
from datetime import datetime
from pathlib import Path
import settings as S

FIELDS = [
    "timestamp", "direction", "expiry_ticks", "stake", "profit", "win",
    "confidence", "score", "hurst", "mg_step", "balance",
]


def setup_logger() -> logging.Logger:
    Path(S.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    fmt  = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    log  = logging.getLogger("Bot")
    log.setLevel(logging.DEBUG)
    ch   = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, dfmt))
    fh   = logging.handlers.RotatingFileHandler(
        S.LOG_FILE, maxBytes=8*1024*1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, dfmt))
    log.addHandler(ch)
    log.addHandler(fh)
    return log


class TradeLogger:
    def __init__(self):
        p = Path(S.TRADES_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with open(p, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def record(self, direction, expiry, stake, profit, win,
               confidence, score, hurst, mg_step, balance):
        row = {
            "timestamp":    datetime.now().isoformat(),
            "direction":    direction,
            "expiry_ticks": expiry,
            "stake":        stake,
            "profit":       round(profit, 2),
            "win":          win,
            "confidence":   round(confidence, 4),
            "score":        score,
            "hurst":        round(hurst, 4) if hurst else 0.0,
            "mg_step":      mg_step,
            "balance":      round(balance, 2),
        }
        with open(S.TRADES_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


class Telegram:
    def __init__(self):
        self._enabled = bool(S.TELEGRAM_TOKEN and S.TELEGRAM_CHAT_ID)
        self._url     = f"https://api.telegram.org/bot{S.TELEGRAM_TOKEN}/sendMessage"

    def send(self, text: str):
        if not self._enabled:
            return
        def _fire():
            try:
                data = urllib.parse.urlencode({
                    "chat_id":    S.TELEGRAM_CHAT_ID,
                    "text":       text,
                    "parse_mode": "HTML",
                }).encode()
                urllib.request.urlopen(self._url, data=data, timeout=6)
            except Exception:
                pass
        threading.Thread(target=_fire, daemon=True).start()
