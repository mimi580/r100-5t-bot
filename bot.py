"""
R_100 5-TICK RISE/FALL BOT
============================
Trades CALL/PUT on R_100 with 5-tick expiry.
Two-layer signal: Fisher+Hurst+MTF context → micro-structure entry timing.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio, json, random, time, uuid
from collections import deque
from typing import Optional

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

import settings as S
from models  import evaluate, TickSignal
from risk    import RiskManager
from logger  import setup_logger, TradeLogger, Telegram

log = setup_logger()


class R100FiveTickBot:

    def __init__(self):
        self.ws              = None
        self.connected       = False
        self.balance         = 0.0
        self.risk            = None
        self.tlog            = TradeLogger()
        self.tg              = Telegram()

        self._prices         = deque(maxlen=S.TICK_BUFFER)
        self._tick_count     = 0

        self.in_trade        = False
        self._cooldown_until = 0.0
        self._pending        = {}

        self._req_id         = 0
        self._msg_buffer     = []

    def _nid(self):
        self._req_id += 1
        return self._req_id

    def _prices_arr(self):
        return np.array(self._prices, dtype=np.float64)

    # ─── Run loop ─────────────────────────────────────────────────

    async def run(self):
        log.info("=" * 65)
        log.info("R_100 5-TICK RISE/FALL BOT")
        log.info(f"Symbol: {S.SYMBOL} | Expiry: {S.EXPIRY_TICKS} ticks")
        log.info(f"Layer 1: Fisher(p<{S.FISHER_P_THRESHOLD}) + Hurst(H>{S.HURST_BLOCK}) + MTF")
        log.info(f"Layer 2: Momentum + ACF_Short + VolBurst + ZScore ({S.LAYER2_REQUIRED}/4 needed)")
        log.info(f"Warmup: {S.WARMUP_TICKS} ticks | Cooldown: {S.COOLDOWN_MIN}-{S.COOLDOWN_MAX}s")
        log.info("=" * 65)

        delay, attempt = S.RECONNECT_BASE, 0
        while True:
            try:
                attempt += 1
                log.info(f"Connecting (attempt {attempt})...")
                await self._connect()
                delay = S.RECONNECT_BASE; attempt = 0
            except (ConnectionClosed, WebSocketException) as e:
                log.warning(f"WS closed: {e}")
            except asyncio.TimeoutError:
                log.warning("Connection timed out")
            except Exception as e:
                log.error(f"Unexpected: {e}", exc_info=True)

            self.connected = False
            if self.in_trade:
                log.warning("Disconnected mid-trade — releasing lock")
                self.in_trade = False

            log.info(f"Reconnecting in {delay}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, S.RECONNECT_MAX)

    # ─── Connection ───────────────────────────────────────────────

    async def _connect(self):
        url = f"{S.DERIV_WS_URL}?app_id={S.DERIV_APP_ID}"
        async with websockets.connect(
            url, ping_interval=30, ping_timeout=20, close_timeout=5
        ) as ws:
            self.ws = ws; self.connected = True

            await self._send_raw({"authorize": S.DERIV_API_TOKEN})
            auth = json.loads(await asyncio.wait_for(ws.recv(), 15.0))
            if auth.get("error"):
                raise Exception(f"Auth failed: {auth['error']['message']}")

            self.balance = float(auth["authorize"]["balance"])
            log.info(f"Authorized | Balance: ${self.balance:.2f}")

            if self.risk is None:
                self.risk = RiskManager(self.balance)
            else:
                self.risk.sync(self.balance)

            await self._send_raw({"balance": 1, "subscribe": 1})
            await self._send_raw({"ticks": S.SYMBOL, "subscribe": 1})
            log.info(f"Subscribed to {S.SYMBOL} | Warming up {S.WARMUP_TICKS} ticks...")

            self.tg.send(
                f"R_100 5T Bot STARTED\n"
                f"Balance: ${self.balance:.2f}\n"
                f"RISE/FALL | {S.EXPIRY_TICKS} ticks | p<{S.FISHER_P_THRESHOLD}"
            )

            async for raw in ws:
                if not self.connected: break
                try:
                    await self._handle(json.loads(raw))
                except Exception as e:
                    log.error(f"Handle error: {e}", exc_info=True)
                while self._msg_buffer:
                    msg = self._msg_buffer.pop(0)
                    try:
                        await self._handle(msg)
                    except Exception as e:
                        log.error(f"Buffer error: {e}", exc_info=True)

    async def _send_raw(self, payload):
        rid = self._nid()
        payload["req_id"] = rid
        if self.ws and self.connected:
            try:
                await asyncio.wait_for(self.ws.send(json.dumps(payload)), 10.0)
            except Exception as e:
                log.warning(f"Send error: {e}")
        return rid

    # ─── Message router ───────────────────────────────────────────

    async def _handle(self, msg):
        t = msg.get("msg_type")
        if   t == "tick":
            await self._on_tick(msg["tick"])
        elif t == "balance":
            self.balance = float(msg["balance"]["balance"])
            if self.risk: self.risk.sync(self.balance)
        elif t == "proposal_open_contract":
            await self._on_contract(msg["proposal_open_contract"])
        elif t == "buy":
            log.debug(f"Stray buy: req_id={msg.get('req_id')}")
        elif t == "error":
            log.error(f"API error: {msg.get('error',{}).get('message','?')}")

    # ─── Tick processing ──────────────────────────────────────────

    async def _on_tick(self, tick):
        price = float(tick["quote"])
        self._prices.append(price)
        self._tick_count += 1

        if self._tick_count < S.WARMUP_TICKS:
            if self._tick_count % 60 == 0:
                log.info(f"Warming up {self._tick_count}/{S.WARMUP_TICKS}")
            return

        if self.in_trade or not self.risk or not self.risk.bot_active:
            return

        if time.time() < self._cooldown_until:
            return

        await self._evaluate()

    # ─── Signal evaluation ────────────────────────────────────────

    async def _evaluate(self):
        prices = self._prices_arr()
        signal = evaluate(prices)

        if not signal.tradeable:
            sr = signal.skip_reason
            # Layer 1 skips at INFO (important — market condition)
            # Layer 2 skips at DEBUG (frequent — entry timing misses)
            if any(k in sr for k in ["HURST", "FISHER", "MTF", "VOL_SPIKE"]):
                log.info(f"SKIP | {sr}")
            elif "LAYER2" in sr:
                log.debug(f"SKIP | {sr}")
            else:
                log.debug(f"SKIP | {sr}")
            return

        stake = self.risk.stake()
        if stake == 0:
            return

        self.in_trade = True

        log.info(
            f"SIGNAL | {signal.direction} | "
            f"H={signal.hurst:.3f} | p={signal.p_value:.5f} | "
            f"conf={signal.confidence:.4f} | "
            f"L1=pass L2={signal.layer2_score}/4 | "
            f"stake=${stake:.2f} | "
            f"{' | '.join(signal.reasons)}"
        )

        await self._place_trade(signal, stake)

    # ─── Trade placement ──────────────────────────────────────────

    async def _place_trade(self, signal: TickSignal, stake: float):
        trade_id = str(uuid.uuid4())[:8].upper()
        rid      = self._nid()
        contract = "CALL" if signal.direction == "RISE" else "PUT"

        payload = {
            "buy": 1,
            "price": stake,
            "req_id": rid,
            "parameters": {
                "contract_type": contract,
                "symbol":        S.SYMBOL,
                "duration":      S.EXPIRY_TICKS,
                "duration_unit": S.EXPIRY_UNIT,
                "basis":         "stake",
                "amount":        stake,
                "currency":      "USD",
            },
        }

        try:
            await asyncio.wait_for(self.ws.send(json.dumps(payload)), 10.0)
        except Exception as e:
            log.error(f"Send failed: {e}")
            self._release_lock(); return

        # Direct recv — buffer everything else
        buy_resp = None
        deadline = time.time() + 15.0

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0: break
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            except Exception as e:
                log.error(f"WS recv: {e}"); break

            msg = json.loads(raw)
            if msg.get("msg_type") == "buy" and msg.get("req_id") == rid:
                buy_resp = msg; break
            else:
                self._msg_buffer.append(msg)

        if buy_resp is None:
            log.error("Buy timeout — lock released")
            self._release_lock(); return

        if "error" in buy_resp:
            log.error(f"Buy rejected: {buy_resp['error'].get('message','?')}")
            self._release_lock(); return

        contract_id = buy_resp.get("buy", {}).get("contract_id")
        if not contract_id:
            log.error("No contract_id")
            self._release_lock(); return

        log.info(f"PLACED {trade_id} | cid={contract_id} | "
                 f"{contract} | {S.EXPIRY_TICKS}t | ${stake:.2f}")

        self.tg.send(
            f"TRADE [R_100 5T]\n"
            f"{contract} {signal.direction} | {S.EXPIRY_TICKS} ticks\n"
            f"Stake: ${stake:.2f} | H={signal.hurst:.3f} | "
            f"p={signal.p_value:.4f} | conf={signal.confidence:.3f}"
        )

        self._pending = {
            "trade_id":    trade_id,
            "contract_id": contract_id,
            "stake":       stake,
            "signal":      signal,
            "mg_step":     self.risk._mg_step,
        }

        await self._send_raw({
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
        })

        # 5-tick contracts settle in ~5 seconds
        # Poller starts after 10s buffer
        asyncio.create_task(self._poll(contract_id))

    def _release_lock(self):
        self.in_trade        = False
        self._cooldown_until = time.time() + S.COOLDOWN_MIN

    # ─── Settlement push ──────────────────────────────────────────

    async def _on_contract(self, contract):
        if contract.get("status") not in ("won", "lost", "sold"):
            return
        cid  = contract.get("contract_id")
        meta = self._pending
        if not meta or meta.get("contract_id") != cid:
            return
        await self._settle(contract, meta)

    # ─── Fallback poller ──────────────────────────────────────────

    async def _poll(self, contract_id: str):
        """
        5-tick contracts settle in ~5 seconds.
        Start polling after 10s to give push settlement a chance first.
        """
        try:
            await asyncio.sleep(10)
            for attempt in range(1, 15):
                meta = self._pending
                if not meta or meta.get("contract_id") != contract_id:
                    return   # already settled via push
                if not self.connected:
                    return
                log.debug(f"Polling {contract_id} ({attempt}/14)")
                await self._send_raw({
                    "proposal_open_contract": 1,
                    "contract_id": contract_id,
                })
                await asyncio.sleep(3)

            log.error(f"Contract {contract_id} never settled")
            self._pending = {}
            self._release_lock()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Poll error: {e}", exc_info=True)
            self._release_lock()

    # ─── Settlement ───────────────────────────────────────────────

    async def _settle(self, contract, meta):
        profit = float(contract.get("profit", 0))
        win    = contract.get("status") == "won"
        stake  = meta["stake"]
        signal = meta["signal"]

        alerts = self.risk.record(profit, win)

        self.tlog.record(
            direction  = signal.direction,
            expiry     = S.EXPIRY_TICKS,
            stake      = stake,
            profit     = profit,
            win        = win,
            confidence = signal.confidence,
            score      = signal.layer2_score,
            models     = signal.models,
            mg_step    = meta["mg_step"],
            balance    = self.balance,
        )

        icon    = "WIN" if win else "LOSS"
        summary = self.risk.summary_line()
        log.info(f"[{icon}] P&L ${profit:+.2f} | {summary}")

        self.tg.send(
            f"[{icon}] R_100 5T\n"
            f"P&L: ${profit:+.2f} | Bal: ${self.balance:.2f}\n{summary}"
        )

        for kind, msg in alerts.items():
            self.tg.send(f"ALERT {kind}: {msg}")

        cooldown = random.uniform(S.COOLDOWN_MIN, S.COOLDOWN_MAX)
        self._cooldown_until = time.time() + cooldown
        log.info(f"Breathing {cooldown:.0f}s")

        self._pending = {}
        self.in_trade = False

        self._scoreboard(win, profit, signal)

    # ─── Scoreboard ───────────────────────────────────────────────

    def _scoreboard(self, win: bool, profit: float, signal: TickSignal):
        G="\033[92m"; R="\033[91m"; Y="\033[93m"
        B="\033[94m"; M="\033[90m"; X="\033[0m"; BO="\033[1m"

        r   = self.risk
        wr  = r.wins/r.total_trades*100 if r.total_trades else 0
        wc  = G if wr>=62 else Y if wr>=55 else R
        pc  = G if r.daily_pnl>=0 else R
        mgc = R if r._mg_step>=2 else Y if r._mg_step==1 else M
        res = f"{G}WIN ✅{X}" if win else f"{R}LOSS ❌{X}"
        pnl = f"{G}+${profit:.2f}{X}" if profit>=0 else f"{R}-${abs(profit):.2f}{X}"
        dc  = G if signal.direction=="RISE" else R

        print(f"\n{BO}{'─'*65}{X}")
        print(f"  {BO}[R_100 5T]{X}  {res}  "
              f"{dc}{signal.direction}{X}  "
              f"H={signal.hurst:.3f}  p={signal.p_value:.4f}  "
              f"L2={signal.layer2_score}/4  "
              f"P&L {pnl}  Bal {B}${r.balance:.2f}{X}")
        print(f"  {M}{'─'*63}{X}")
        print(f"  {r.total_trades:>5}T  "
              f"{G}{r.wins}W{X}/{R}{r.losses}L{X}  "
              f"{wc}{wr:.1f}%{X}  "
              f"{pc}P&L ${r.daily_pnl:+.2f}{X}  "
              f"{Y}${r.current_stake:.2f}{X}  "
              f"{mgc}MG={'s'+str(r._mg_step) if r._mg_step else 'base'}{X}")
        print(f"{BO}{'─'*65}{X}\n", flush=True)


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not S.DERIV_API_TOKEN:
        print("ERROR: Set DERIV_API_TOKEN environment variable")
        sys.exit(1)
    try:
        asyncio.run(R100FiveTickBot().run())
    except KeyboardInterrupt:
        log.info("Stopped by user.")
