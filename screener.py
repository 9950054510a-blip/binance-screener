"""
Binance USDT-M Futures Trend-Pullback Screener

ENV:
  BINANCE_BASE_URL=https://fapi.binance.com
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...                 # required for commands and automatic alerts
  DB_PATH=./screener.db
  HEALTH_HOST=0.0.0.0
  HEALTH_PORT=8080
  MIN_QUOTE_VOLUME=15000000
  MAX_SYMBOLS=0                         # 0 = all eligible symbols
  SCAN_INTERVAL_SEC=300
  SCORE_THRESHOLD=70
  VOLUME_RATIO_MIN=1.30
  EXTENSION_MAX_ATR=0.50
  SIGNAL_LIFETIME_BARS=2
  EPISODE_COOLDOWN_BARS=3
  MAX_CONCURRENCY=8
  REQUEST_TIMEOUT_SEC=12
  PARAMS_VERSION=v1.2.5-production-hardening

Install:
  Python 3.10+
  pip install -r requirements.txt

Run:
  python screener.py

No order execution is implemented. This program only scans and sends alerts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
from aiohttp import web
import aiosqlite
from aiogram import Bot, Dispatcher, Router
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.types import Message

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --------------------------- configuration ---------------------------

BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com").rstrip("/")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DB_PATH = os.getenv("DB_PATH", "./screener.db")
HEALTH_HOST = os.getenv("HEALTH_HOST", "0.0.0.0")
HEALTH_PORT = int(os.getenv("PORT") or os.getenv("HEALTH_PORT", "8080"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "15000000"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "70"))
VOLUME_RATIO_MIN = float(os.getenv("VOLUME_RATIO_MIN", "1.30"))
EXTENSION_MAX_ATR = float(os.getenv("EXTENSION_MAX_ATR", "0.50"))
SIGNAL_LIFETIME_BARS = int(os.getenv("SIGNAL_LIFETIME_BARS", "2"))
EPISODE_COOLDOWN_BARS = int(os.getenv("EPISODE_COOLDOWN_BARS", "3"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "12"))
PARAMS_VERSION = os.getenv("PARAMS_VERSION", "v1.2.6-production-hardening")
TELEGRAM_ADMIN_USER_ID = os.getenv("TELEGRAM_ADMIN_USER_ID", "").strip()

KLINE_LIMIT_1H = 520
MIN_1H_BARS = 500
KLINE_LIMIT_15M = 250
KLINE_LIMIT_5M = 80

logger = logging.getLogger("screener")

# --------------------------- models ---------------------------

@dataclass(slots=True)
class Candle:
    open_ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_ts: int
    quote_volume: float
    taker_buy_quote: float

@dataclass(slots=True)
class Signal:
    symbol: str
    direction: str
    breakout_open_ts: int
    episode_ts: int
    entry_ref: float
    atr15: float
    atr_ratio: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    score: float
    s_vol: float
    s_rsi: float
    s_trend: float
    vol_ratio: float
    taker_ratio: float
    rsi15: float
    zone_width_atr: float
    trend_dist_atr: float
    funding: Optional[float]
    signal_id: Optional[int] = None

# --------------------------- indicators ---------------------------

def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out = [values[0]]
    prev = values[0]
    for x in values[1:]:
        prev = alpha * x + (1.0 - alpha) * prev
        out.append(prev)
    return out


def sma(values: list[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def true_ranges(candles: list[Candle]) -> list[float]:
    if not candles:
        return []
    out = [candles[0].high - candles[0].low]
    for i in range(1, len(candles)):
        c, p = candles[i], candles[i - 1]
        out.append(max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close)))
    return out


def atr(candles: list[Candle], period: int = 14) -> Optional[float]:
    trs = true_ranges(candles)
    if len(trs) < period:
        return None
    value = sum(trs[:period]) / period
    for tr in trs[period:]:
        value = ((period - 1) * value + tr) / period
    return value


def atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    trs = true_ranges(candles)
    if len(trs) < period:
        return []
    value = sum(trs[:period]) / period
    out = [float("nan")] * (period - 1) + [value]
    for tr in trs[period:]:
        value = ((period - 1) * value + tr) / period
        out.append(value)
    return out


def rsi(candles: list[Candle], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    closes = [c.close for c in candles]
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = ((period - 1) * avg_gain + gain) / period
        avg_loss = ((period - 1) * avg_loss + loss) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def median_atr100(candles: list[Candle], period: int = 14, window: int = 100) -> Optional[float]:
    series = atr_series(candles, period)
    vals = [x for x in series[-window:] if math.isfinite(x)]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

# --------------------------- helpers ---------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def utc_text(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(x: Any) -> float:
    try:
        value = float(x)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"

# --------------------------- database ---------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS scans (
    scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_ts INTEGER NOT NULL,
    params_version TEXT NOT NULL,
    universe INTEGER NOT NULL DEFAULT 0,
    btc_pass INTEGER NOT NULL DEFAULT 0,
    trend1h INTEGER NOT NULL DEFAULT 0,
    pullback15 INTEGER NOT NULL DEFAULT 0,
    breakout5 INTEGER NOT NULL DEFAULT 0,
    volume INTEGER NOT NULL DEFAULT 0,
    taker INTEGER NOT NULL DEFAULT 0,
    score_pass INTEGER NOT NULL DEFAULT 0,
    final INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    stale INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    breakout_open_ts INTEGER NOT NULL,
    episode_ts INTEGER NOT NULL,
    kind TEXT NOT NULL,
    fail_reason TEXT,
    entry_ref REAL,
    atr15 REAL,
    atr_ratio REAL,
    sl REAL,
    tp1 REAL,
    tp2 REAL,
    tp3 REAL,
    score REAL,
    s_vol REAL,
    s_rsi REAL,
    s_trend REAL,
    vol_ratio REAL,
    taker_ratio REAL,
    rsi15 REAL,
    zone_width_atr REAL,
    trend_dist_atr REAL,
    funding REAL,
    params_version TEXT NOT NULL,
    created_ts INTEGER NOT NULL,
    telegram_sent INTEGER NOT NULL DEFAULT 0,
    UNIQUE(symbol, direction, breakout_open_ts)
);

CREATE TABLE IF NOT EXISTS signal_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    type TEXT NOT NULL,
    price REAL,
    extension_atr REAL,
    FOREIGN KEY(signal_id) REFERENCES signals(signal_id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    signal_id INTEGER PRIMARY KEY,
    resolved_ts INTEGER,
    exit_reason TEXT,
    tp1_hit INTEGER DEFAULT 0,
    tp2_hit INTEGER DEFAULT 0,
    tp3_hit INTEGER DEFAULT 0,
    sl_hit INTEGER DEFAULT 0,
    mfe_r REAL,
    mae_r REAL,
    time_to_outcome_s INTEGER,
    gross_r REAL,
    fee_r REAL,
    slippage_r REAL,
    funding_r REAL,
    net_r REAL,
    FOREIGN KEY(signal_id) REFERENCES signals(signal_id)
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, breakout_open_ts);
CREATE INDEX IF NOT EXISTS idx_signals_kind ON signals(kind);
"""

class Database:
    def __init__(self, path: str):
        self.path = path
        self.db: Optional[aiosqlite.Connection] = None
        self.lock = asyncio.Lock()

    async def open(self) -> None:
        self.db = await aiosqlite.connect(self.path)
        self.db.row_factory = aiosqlite.Row
        await self.db.execute("PRAGMA foreign_keys=ON")
        await self.db.execute("PRAGMA busy_timeout=5000")
        await self.db.executescript(SCHEMA)
        cur = await self.db.execute("PRAGMA table_info(scans)")
        columns = {row[1] for row in await cur.fetchall()}
        if "errors" not in columns:
            await self.db.execute("ALTER TABLE scans ADD COLUMN errors INTEGER NOT NULL DEFAULT 0")
        if "stale" not in columns:
            await self.db.execute("ALTER TABLE scans ADD COLUMN stale INTEGER NOT NULL DEFAULT 0")
        cur = await self.db.execute("PRAGMA table_info(signals)")
        signal_columns = {row[1] for row in await cur.fetchall()}
        if "telegram_sent" not in signal_columns:
            await self.db.execute("ALTER TABLE signals ADD COLUMN telegram_sent INTEGER NOT NULL DEFAULT 0")
            await self.db.execute("UPDATE signals SET telegram_sent=1 WHERE kind='SIGNAL'")
        await self.db.execute(
            "UPDATE signals SET telegram_sent=1 WHERE kind='SIGNAL' AND params_version != ?",
            (PARAMS_VERSION,),
        )
        await self.db.execute(
            "UPDATE scans SET slot_ts = slot_ts * 1000 "
            "WHERE slot_ts > 0 AND slot_ts < 100000000000"
        )
        await self.db.execute(
            "DELETE FROM signal_events "
            "WHERE event_id NOT IN ("
            "SELECT MIN(event_id) FROM signal_events GROUP BY signal_id, type"
            ")"
        )
        await self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_signal_events_signal_type "
            "ON signal_events(signal_id, type)"
        )
        await self.db.commit()

    async def close(self) -> None:
        if self.db:
            await self.db.close()
            self.db = None

    async def execute(self, sql: str, params: tuple = ()) -> None:
        assert self.db is not None
        async with self.lock:
            await self.db.execute(sql, params)
            await self.db.commit()

    async def write_scan(self, slot_ts: int, counts: dict[str, int], duration_ms: int) -> int:
        assert self.db is not None
        async with self.lock:
            cur = await self.db.execute(
                """INSERT INTO scans(slot_ts,params_version,universe,btc_pass,trend1h,pullback15,
                   breakout5,volume,taker,score_pass,final,duration_ms,errors,stale)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (slot_ts, PARAMS_VERSION, counts.get("universe", 0), counts.get("btc_pass", 0),
                 counts.get("trend1h", 0), counts.get("pullback15", 0), counts.get("breakout5", 0),
                 counts.get("volume", 0), counts.get("taker", 0), counts.get("score_pass", 0),
                 counts.get("final", 0), duration_ms, counts.get("errors", 0), counts.get("stale", 0)),
            )
            await self.db.commit()
            return int(cur.lastrowid)

    async def insert_signal(self, signal: Signal, scan_id: int, kind: str, fail_reason: Optional[str] = None) -> Optional[int]:
        assert self.db is not None
        async with self.lock:
            cur = await self.db.execute(
                """INSERT OR IGNORE INTO signals
                (scan_id,symbol,direction,breakout_open_ts,episode_ts,kind,fail_reason,
                 entry_ref,atr15,atr_ratio,sl,tp1,tp2,tp3,score,s_vol,s_rsi,s_trend,
                 vol_ratio,taker_ratio,rsi15,zone_width_atr,trend_dist_atr,funding,params_version,created_ts,telegram_sent)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (scan_id, signal.symbol, signal.direction, signal.breakout_open_ts, signal.episode_ts,
                 kind, fail_reason, signal.entry_ref, signal.atr15, signal.atr_ratio, signal.sl,
                 signal.tp1, signal.tp2, signal.tp3, signal.score, signal.s_vol, signal.s_rsi,
                 signal.s_trend, signal.vol_ratio, signal.taker_ratio, signal.rsi15,
                 signal.zone_width_atr, signal.trend_dist_atr, signal.funding, PARAMS_VERSION, now_ms(), 0),
            )
            await self.db.commit()
            if cur.rowcount == 0:
                return None
            return int(cur.lastrowid)

    async def insert_signal_with_lifecycle(self, signal: Signal, scan_id: int) -> Optional[int]:
        assert self.db is not None
        async with self.lock:
            cur = await self.db.execute(
                """INSERT OR IGNORE INTO signals
                (scan_id,symbol,direction,breakout_open_ts,episode_ts,kind,fail_reason,
                 entry_ref,atr15,atr_ratio,sl,tp1,tp2,tp3,score,s_vol,s_rsi,s_trend,
                 vol_ratio,taker_ratio,rsi15,zone_width_atr,trend_dist_atr,funding,params_version,created_ts,telegram_sent)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (scan_id, signal.symbol, signal.direction, signal.breakout_open_ts, signal.episode_ts,
                 "SIGNAL", None, signal.entry_ref, signal.atr15, signal.atr_ratio, signal.sl,
                 signal.tp1, signal.tp2, signal.tp3, signal.score, signal.s_vol, signal.s_rsi,
                 signal.s_trend, signal.vol_ratio, signal.taker_ratio, signal.rsi15,
                 signal.zone_width_atr, signal.trend_dist_atr, signal.funding, PARAMS_VERSION, now_ms(), 0),
            )
            if cur.rowcount == 0:
                await self.db.rollback()
                return None
            signal_id = int(cur.lastrowid)
            ts = now_ms()
            await self.db.execute(
                "INSERT INTO signal_events(signal_id,ts,type,price,extension_atr) VALUES(?,?,?,?,?)",
                (signal_id, ts, "DETECTED", signal.entry_ref, 0.0),
            )
            await self.db.execute(
                "INSERT INTO signal_events(signal_id,ts,type,price,extension_atr) VALUES(?,?,?,?,?)",
                (signal_id, ts, "VALID", signal.entry_ref, 0.0),
            )
            await self.db.commit()
            return signal_id

    async def mark_telegram_sent(self, signal_id: int) -> None:
        await self.execute("UPDATE signals SET telegram_sent=1 WHERE signal_id=?", (signal_id,))

    async def pending_telegram_signals(self, limit: int = 5) -> list[aiosqlite.Row]:
        assert self.db is not None
        async with self.lock:
            cur = await self.db.execute(
                """SELECT * FROM signals
                   WHERE kind='SIGNAL' AND COALESCE(telegram_sent,0)=0
                   ORDER BY signal_id ASC LIMIT ?""",
                (max(1, int(limit)),),
            )
            return await cur.fetchall()

    async def event(self, signal_id: int, event_type: str, price: Optional[float], extension: Optional[float]) -> None:
        await self.execute(
            "INSERT INTO signal_events(signal_id,ts,type,price,extension_atr) VALUES(?,?,?,?,?)",
            (signal_id, now_ms(), event_type, price, extension),
        )

    async def add_lifecycle_event(
        self,
        signal_id: int,
        event_type: str,
        price: Optional[float],
        extension: Optional[float],
    ) -> bool:
        assert self.db is not None
        async with self.lock:
            cur = await self.db.execute(
                "SELECT 1 FROM signal_events WHERE signal_id=? AND type=? LIMIT 1",
                (signal_id, event_type),
            )
            if await cur.fetchone() is not None:
                return False
            await self.db.execute(
                "INSERT INTO signal_events(signal_id,ts,type,price,extension_atr) VALUES(?,?,?,?,?)",
                (signal_id, now_ms(), event_type, price, extension),
            )
            await self.db.commit()
            return True

    async def active_signals(self) -> list[aiosqlite.Row]:
        assert self.db is not None
        async with self.lock:
            cur = await self.db.execute(
                """
                SELECT s.signal_id, s.symbol, s.direction, s.breakout_open_ts,
                       s.entry_ref, s.atr15,
                       COALESCE((SELECT e.type FROM signal_events e
                                 WHERE e.signal_id=s.signal_id
                                 ORDER BY e.event_id DESC LIMIT 1), 'DETECTED') AS lifecycle_state
                FROM signals s
                WHERE s.kind='SIGNAL'
                  AND NOT EXISTS (
                      SELECT 1 FROM signal_events x
                      WHERE x.signal_id=s.signal_id AND x.type='EXPIRED'
                  )
                ORDER BY s.breakout_open_ts ASC
                """
            )
            return await cur.fetchall()

    async def episode_exists(self, symbol: str, direction: str, episode_ts: int, cooldown_ms: int) -> bool:
        assert self.db is not None
        async with self.lock:
            cur = await self.db.execute(
                """SELECT 1 FROM signals WHERE symbol=? AND direction=?
                   AND kind='SIGNAL' AND created_ts >= ? LIMIT 1""",
                (symbol, direction, now_ms() - cooldown_ms),
            )
            row = await cur.fetchone()
            return row is not None

class StaleKlineError(RuntimeError):
    pass

# --------------------------- Binance client ---------------------------

class BinanceClient:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.sem = asyncio.Semaphore(MAX_CONCURRENCY)
        self.exchange_cache: tuple[float, list[dict[str, Any]]] | None = None
        self.ticker_cache: tuple[float, dict[str, dict[str, Any]]] | None = None
        self.time_offset_ms = 0
        self.time_last_sync = 0.0
        self.server_anchor_ms = 0
        self.server_anchor_mono = 0.0
        self.time_lock = asyncio.Lock()
        self.cooldown_until_mono = 0.0
        self.cooldown_lock = asyncio.Lock()

    async def _wait_global_cooldown(self) -> None:
        while True:
            async with self.cooldown_lock:
                remaining = self.cooldown_until_mono - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 5.0))

    async def _set_global_cooldown(self, seconds: float) -> None:
        seconds = min(max(float(seconds), 1.0), 300.0)
        async with self.cooldown_lock:
            self.cooldown_until_mono = max(self.cooldown_until_mono, time.monotonic() + seconds)

    async def get(self, path: str, params: Optional[dict[str, Any]] = None, retries: int = 3) -> Any:
        url = BINANCE_BASE_URL + path
        delay = 0.8
        for attempt in range(retries):
            try:
                await self._wait_global_cooldown()
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
                async with self.sem:
                    async with self.session.get(url, params=params, timeout=timeout) as r:
                        if r.status == 200:
                            return await r.json()
                        retry_after = r.headers.get("Retry-After")
                        body = await r.text()
                        if r.status == 418:
                            try:
                                ban_delay = float(retry_after) if retry_after else 60.0
                            except (TypeError, ValueError):
                                ban_delay = 60.0
                            await self._set_global_cooldown(ban_delay)
                            raise RuntimeError(f"Binance HTTP 418 (IP ban): {body[:300]}")
                        if r.status in (429, 500, 502, 503, 504):
                            try:
                                retry_delay = float(retry_after) if retry_after else delay
                            except (TypeError, ValueError):
                                retry_delay = delay
                            retry_delay = min(max(0.2, retry_delay), 30.0)
                            if r.status == 429:
                                await self._set_global_cooldown(retry_delay)
                        else:
                            raise RuntimeError(f"Binance HTTP {r.status}: {body[:300]}")
                if attempt + 1 >= retries:
                    break
                logger.warning("Binance HTTP %s; retrying in %.2fs", r.status, retry_delay)
                await asyncio.sleep(retry_delay)
                delay = min(delay * 2, 10)
                continue
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                if attempt + 1 >= retries:
                    raise
                logger.warning("Binance request retry: %s", exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)
        raise RuntimeError("Binance request failed after retries")

    async def sync_server_time(self, force: bool = False) -> int:
        now_mono = time.monotonic()
        if not force and self.time_last_sync and now_mono - self.time_last_sync < 3600:
            return self.time_offset_ms
        async with self.time_lock:
            now_mono = time.monotonic()
            if not force and self.time_last_sync and now_mono - self.time_last_sync < 3600:
                return self.time_offset_ms
            data = await self.get("/fapi/v1/time", retries=3)
            server_ms = int(data["serverTime"])
            self.server_anchor_ms = server_ms
            self.server_anchor_mono = time.monotonic()
            self.time_offset_ms = server_ms - now_ms()
            self.time_last_sync = self.server_anchor_mono
            logger.info("Binance time synchronized: offset_ms=%d", self.time_offset_ms)
            return self.time_offset_ms

    async def server_now_ms(self) -> int:
        await self.sync_server_time()
        elapsed_ms = int((time.monotonic() - self.server_anchor_mono) * 1000)
        return self.server_anchor_ms + elapsed_ms

    async def exchange_symbols(self) -> list[dict[str, Any]]:
        if self.exchange_cache and time.monotonic() - self.exchange_cache[0] < 3600:
            return self.exchange_cache[1]
        data = await self.get("/fapi/v1/exchangeInfo")
        symbols = []
        for s in data.get("symbols", []):
            if (s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL"
                    and s.get("quoteAsset") == "USDT"):
                symbols.append(s)
        self.exchange_cache = (time.monotonic(), symbols)
        return symbols

    async def tickers(self) -> dict[str, dict[str, Any]]:
        if self.ticker_cache and time.monotonic() - self.ticker_cache[0] < 20:
            return self.ticker_cache[1]
        data = await self.get("/fapi/v1/ticker/24hr")
        out = {x["symbol"]: x for x in data if x.get("symbol")}
        self.ticker_cache = (time.monotonic(), out)
        return out

    async def klines(self, symbol: str, interval: str, limit: int, expected_last_open_ts: Optional[int] = None) -> list[Candle]:
        for attempt in range(3):
            data = await self.get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
            out: list[Candle] = []
            current_ms = await self.server_now_ms()
            for k in data:
                open_ts = int(k[0])
                close_ts = int(k[6])
                if expected_last_open_ts is not None:
                    if open_ts > expected_last_open_ts:
                        continue
                elif close_ts >= current_ms:
                    continue
                out.append(Candle(
                    open_ts=open_ts, open=safe_float(k[1]), high=safe_float(k[2]),
                    low=safe_float(k[3]), close=safe_float(k[4]), volume=safe_float(k[5]),
                    close_ts=close_ts, quote_volume=safe_float(k[7]),
                    taker_buy_quote=safe_float(k[10]),
                ))
            if expected_last_open_ts is None:
                return out
            interval_ms = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000}.get(interval)
            if interval_ms and len(out) >= 2:
                for prev, cur in zip(out, out[1:]):
                    if cur.open_ts - prev.open_ts != interval_ms:
                        raise RuntimeError(f"{symbol} {interval}: kline gap detected at {utc_text(cur.open_ts)}")
            if out and out[-1].open_ts == expected_last_open_ts and out[-1].close_ts < current_ms:
                return out
            if attempt < 2:
                await asyncio.sleep(1.0)
        raise StaleKlineError(
            f"{symbol} {interval}: expected last closed open_ts={expected_last_open_ts}, "
            f"got={out[-1].open_ts if out else None}"
        )

    async def funding(self, symbol: str) -> Optional[float]:
        try:
            data = await self.get("/fapi/v1/premiumIndex", {"symbol": symbol})
            if not data or data.get("lastFundingRate") is None:
                return None
            value = float(data["lastFundingRate"])
            return value if math.isfinite(value) else None
        except Exception:
            return None

# --------------------------- strategy ---------------------------

class Strategy:
    async def evaluate(self, symbol: str, btc1h: list[Candle], h1: list[Candle], m15: list[Candle], m5: list[Candle], funding: Optional[float]) -> tuple[Optional[Signal], str]:
        if len(h1) < MIN_1H_BARS or len(m15) < 60 or len(m5) < 25 or len(btc1h) < 200:
            return None, "insufficient_history"

        hclose = h1[-1].close
        e20 = ema([c.close for c in h1], 20)[-1]
        e50 = ema([c.close for c in h1], 50)[-1]
        e200 = ema([c.close for c in h1], 200)[-1]
        h_atr = atr(h1, 14)
        if h_atr is None or not math.isfinite(h_atr) or h_atr <= 0:
            return None, "no_h1_atr"

        btc_close = btc1h[-1].close
        btc_e200 = ema([c.close for c in btc1h], 200)[-1]
        if btc_close > btc_e200:
            market_direction = "LONG"
        elif btc_close < btc_e200:
            market_direction = "SHORT"
        else:
            return None, "btc_neutral"

        if market_direction == "LONG":
            if not (e20 > e50 and hclose > e200):
                return None, "trend1h"
            direction = "LONG"
        else:
            if not (e20 < e50 and hclose < e200):
                return None, "trend1h"
            direction = "SHORT"

        atr15 = atr(m15, 14)
        rsi15 = rsi(m15, 14)
        if atr15 is None or not math.isfinite(atr15) or atr15 <= 0 or rsi15 is None or not math.isfinite(rsi15):
            return None, "15m_indicators"
        e20_15 = ema([c.close for c in m15], 20)[-1]
        e50_15 = ema([c.close for c in m15], 50)[-1]
        last3 = m15[-3:]
        zone_low = min(e20_15, e50_15)
        zone_high = max(e20_15, e50_15)
        zone_width_atr = (zone_high - zone_low) / atr15 if atr15 else 0.0

        if direction == "LONG":
            if not e20_15 > e50_15:
                return None, "pullback_ema_direction"
            touched = min(c.low for c in last3) <= zone_high
            depth = min(c.low for c in last3)
            if not touched or depth < zone_low - 0.3 * atr15:
                return None, "pullback_zone"
            if m15[-1].close <= zone_low:
                return None, "pullback_hold"
            if not 35 <= rsi15 <= 48:
                return None, "pullback_rsi"
        else:
            if not e20_15 < e50_15:
                return None, "pullback_ema_direction"
            touched = max(c.high for c in last3) >= zone_low
            depth = max(c.high for c in last3)
            if not touched or depth > zone_high + 0.3 * atr15:
                return None, "pullback_zone"
            if m15[-1].close >= zone_high:
                return None, "pullback_hold"
            if not 52 <= rsi15 <= 65:
                return None, "pullback_rsi"

        breakout = m5[-1]
        prev3 = m5[-4:-1]
        if len(prev3) != 3:
            return None, "breakout_history"
        if direction == "LONG":
            if breakout.close <= max(c.high for c in prev3):
                return None, "breakout5"
        else:
            if breakout.close >= min(c.low for c in prev3):
                return None, "breakout5"

        prev20 = m5[-21:-1]
        vol_sma = sma([c.volume for c in prev20], 20)
        if not vol_sma or vol_sma <= 0:
            return None, "volume_sma"
        vol_ratio = breakout.volume / vol_sma
        if not math.isfinite(vol_ratio) or vol_ratio < VOLUME_RATIO_MIN:
            return None, "volume"

        taker_ratio = breakout.taker_buy_quote / breakout.quote_volume if breakout.quote_volume > 0 else 0.5
        atr_ratio_base = median_atr100(m15, 14, 100)
        atr_ratio = atr15 / atr_ratio_base if atr_ratio_base and atr_ratio_base > 0 else 1.0
        if not math.isfinite(atr_ratio):
            return None, "non_finite"

        trend_dist_atr = abs(hclose - e200) / h_atr
        if not math.isfinite(trend_dist_atr):
            return None, "non_finite"
        s_vol = min(100.0, vol_ratio * 50.0)
        target = 41.5 if direction == "LONG" else 58.5
        s_rsi = max(0.0, 100.0 - abs(rsi15 - target) / 6.5 * 100.0)
        s_trend = min(100.0, trend_dist_atr * 20.0)
        score = 0.40 * s_vol + 0.30 * s_rsi + 0.30 * s_trend
        entry = breakout.close
        if not all(math.isfinite(x) for x in (entry, s_vol, s_rsi, s_trend, score)):
            return None, "non_finite"

        if direction == "LONG":
            sl = entry - 1.5 * atr15
            tp1 = entry + 3.0 * atr15
            tp2 = entry + 4.5 * atr15
            tp3 = entry + 6.0 * atr15
        else:
            sl = entry + 1.5 * atr15
            tp1 = entry - 3.0 * atr15
            tp2 = entry - 4.5 * atr15
            tp3 = entry - 6.0 * atr15

        episode_candidates = []
        for c in last3:
            touched_bar = c.low <= zone_high if direction == "LONG" else c.high >= zone_low
            if touched_bar:
                episode_candidates.append(c.open_ts)
        episode_ts = max(episode_candidates) if episode_candidates else m15[-1].open_ts

        signal = Signal(
            symbol=symbol, direction=direction, breakout_open_ts=breakout.open_ts,
            episode_ts=episode_ts, entry_ref=entry, atr15=atr15, atr_ratio=atr_ratio,
            sl=sl, tp1=tp1, tp2=tp2, tp3=tp3, score=score, s_vol=s_vol,
            s_rsi=s_rsi, s_trend=s_trend, vol_ratio=vol_ratio, taker_ratio=taker_ratio,
            rsi15=rsi15, zone_width_atr=zone_width_atr, trend_dist_atr=trend_dist_atr,
            funding=funding,
        )
        if score < SCORE_THRESHOLD:
            return signal, "score"
        return signal, "final"

# --------------------------- application ---------------------------

class App:
    def __init__(self) -> None:
        self.db = Database(DB_PATH)
        self.bot: Optional[Bot] = Bot(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
        self.dp = Dispatcher()
        self.router = Router()
        self.dp.include_router(self.router)
        self.running = True
        self.auto_scan = True
        self.scan_lock = asyncio.Lock()
        self.last_scan_ms = 0
        self.last_error: Optional[str] = None
        self.started_ms = now_ms()
        self.scan_errors = 0
        self.session: Optional[aiohttp.ClientSession] = None
        self.binance: Optional[BinanceClient] = None
        self.strategy = Strategy()
        self.health_task: Optional[asyncio.Task] = None
        self.scan_task: Optional[asyncio.Task] = None
        self.lifecycle_task: Optional[asyncio.Task] = None
        self.lifecycle_lock = asyncio.Lock()
        self._stopped = False
        self.started_mono = time.monotonic()
        self.last_scan_mono = 0.0

        self.router.message.register(self.cmd_scan, Command("scan"))
        self.router.message.register(self.cmd_auto, Command("auto"))
        self.router.message.register(self.cmd_status, Command("status"))
        self.router.message.register(self.cmd_help, Command("help"))

    def _supervise_background_task(self, name: str, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            self.last_error = f"background task {name} stopped: {exc!r}"
            logger.error(
                "Background task %s terminated unexpectedly; entering fail-closed state",
                name,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self.running = False

    def _validate_config(self) -> None:
        if SCAN_INTERVAL_SEC != 300:
            raise RuntimeError("SCAN_INTERVAL_SEC must be 300 for the fixed 5m strategy cadence")
        if MAX_CONCURRENCY < 1:
            raise RuntimeError("MAX_CONCURRENCY must be >= 1")
        if REQUEST_TIMEOUT_SEC <= 0:
            raise RuntimeError("REQUEST_TIMEOUT_SEC must be > 0")
        if MIN_QUOTE_VOLUME < 0:
            raise RuntimeError("MIN_QUOTE_VOLUME must be >= 0")
        if MAX_SYMBOLS < 0:
            raise RuntimeError("MAX_SYMBOLS must be >= 0")
        if not 0 <= SCORE_THRESHOLD <= 100:
            raise RuntimeError("SCORE_THRESHOLD must be between 0 and 100")
        if VOLUME_RATIO_MIN <= 0:
            raise RuntimeError("VOLUME_RATIO_MIN must be > 0")
        if EXTENSION_MAX_ATR < 0:
            raise RuntimeError("EXTENSION_MAX_ATR must be >= 0")
        if SIGNAL_LIFETIME_BARS < 1:
            raise RuntimeError("SIGNAL_LIFETIME_BARS must be >= 1")
        if EPISODE_COOLDOWN_BARS < 0:
            raise RuntimeError("EPISODE_COOLDOWN_BARS must be >= 0")

    async def start(self) -> None:
        if not TELEGRAM_BOT_TOKEN:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
        self._validate_config()
        self.session = aiohttp.ClientSession()
        self.binance = BinanceClient(self.session)
        await self.binance.sync_server_time(force=True)
        await self.db.open()
        self.health_task = asyncio.create_task(self.health_server(), name="health_server")
        self.scan_task = asyncio.create_task(self.scheduler(), name="scheduler")
        self.lifecycle_task = asyncio.create_task(self.lifecycle_worker(), name="lifecycle_worker")
        self.health_task.add_done_callback(lambda t: self._supervise_background_task("health_server", t))
        self.scan_task.add_done_callback(lambda t: self._supervise_background_task("scheduler", t))
        self.lifecycle_task.add_done_callback(lambda t: self._supervise_background_task("lifecycle_worker", t))
        logger.info("Started. DB=%s params=%s", DB_PATH, PARAMS_VERSION)
        if not TELEGRAM_CHAT_ID:
            logger.warning("TELEGRAM_CHAT_ID is empty: automatic alerts and bot commands are disabled")
        try:
            await self.dp.start_polling(self.bot, handle_signals=False)
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        current = asyncio.current_task()
        tasks = (self.health_task, self.scan_task, self.lifecycle_task)
        for task in tasks:
            if task and task is not current and not task.done():
                task.cancel()
        for task in tasks:
            if task and task is not current:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if self.session:
            with contextlib.suppress(Exception):
                await self.session.close()
            self.session = None
        await self.db.close()
        if self.bot:
            with contextlib.suppress(Exception):
                await self.bot.session.close()
            self.bot = None

    def is_admin(self, message: Message) -> bool:
        if not TELEGRAM_CHAT_ID or str(message.chat.id) != TELEGRAM_CHAT_ID:
            return False
        if TELEGRAM_ADMIN_USER_ID:
            return bool(message.from_user and str(message.from_user.id) == TELEGRAM_ADMIN_USER_ID)
        return True

    async def cmd_help(self, message: Message) -> None:
        if not self.is_admin(message):
            return
        await message.answer("/scan — запустить скан вручную\n/auto — включить/выключить автоскан\n/status — состояние скринера")

    async def cmd_scan(self, message: Message) -> None:
        if not self.is_admin(message):
            return
        if self.scan_lock.locked():
            await message.answer("Скан уже выполняется.")
            return
        result = await self.scan_once(manual=True)
        await message.answer(result)

    async def cmd_auto(self, message: Message) -> None:
        if not self.is_admin(message):
            return
        self.auto_scan = not self.auto_scan
        await message.answer(f"AUTO_SCAN: {'ON' if self.auto_scan else 'OFF'}")

    async def cmd_status(self, message: Message) -> None:
        if not self.is_admin(message):
            return
        await message.answer(
            f"Screener: {'RUNNING' if self.running else 'STOPPING'}\n"
            f"AUTO_SCAN: {'ON' if self.auto_scan else 'OFF'}\n"
            f"Last scan: {utc_text(self.last_scan_ms) if self.last_scan_ms else 'never'}\n"
            f"Params: {PARAMS_VERSION}\n"
            f"Last error: {self.last_error or 'none'}"
        )

    async def scheduler(self) -> None:
        interval_ms = 5 * 60 * 1000
        last_slot_ms: Optional[int] = None
        while self.running:
            try:
                server_now = await self.binance.server_now_ms()
                current_boundary = (server_now // interval_ms) * interval_ms
                next_boundary = current_boundary + interval_ms
                sleep_s = max(0.1, (next_boundary + 2000 - server_now) / 1000.0)
                await asyncio.sleep(sleep_s)
                if not self.auto_scan:
                    continue

                latest_now = await self.binance.server_now_ms()
                latest_slot = (latest_now // interval_ms) * interval_ms
                if last_slot_ms is not None and latest_slot <= last_slot_ms:
                    continue

                if last_slot_ms is not None and latest_slot - last_slot_ms > interval_ms:
                    logger.warning(
                        "Scheduler skipped stale slots after delay: last=%s latest=%s",
                        utc_text(last_slot_ms), utc_text(latest_slot),
                    )

                slot_ms = latest_slot
                result = await self.scan_once(manual=False, slot_ms=slot_ms)
                if self.last_scan_ms >= slot_ms:
                    last_slot_ms = slot_ms
                logger.info("Scheduler result slot=%s: %s", utc_text(slot_ms), result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.scan_errors += 1
                self.last_error = repr(exc)
                logger.exception("Scheduler loop failed; entering fail-closed state")
                self.running = False
                return

    async def health_server(self) -> None:
        app = web.Application()
        app.router.add_get("/", self.health_handler)
        app.router.add_get("/health", self.health_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, HEALTH_HOST, HEALTH_PORT)
        await site.start()
        logger.info("Health server listening on %s:%s", HEALTH_HOST, HEALTH_PORT)
        try:
            while self.running:
                await asyncio.sleep(60)
        finally:
            await runner.cleanup()

    async def health_handler(self, request: web.Request) -> web.Response:
        now_mono = time.monotonic()
        grace_s = max(10 * 60, SCAN_INTERVAL_SEC * 3)
        fresh = self.last_scan_mono > 0 and (now_mono - self.last_scan_mono) < grace_s
        startup_grace = self.last_scan_mono == 0 and (now_mono - self.started_mono) < grace_s
        ok = self.running and (not self.auto_scan or fresh or startup_grace)
        payload = {
            "ok": ok,
            "last_scan_ms": self.last_scan_ms,
            "last_error": self.last_error,
            "params_version": PARAMS_VERSION,
            "auto_scan": self.auto_scan,
            "scan_errors": self.scan_errors,
        }
        return web.json_response(payload, status=200 if ok else 503)

    async def universe(self) -> list[str]:
        assert self.binance is not None
        symbols_info = await self.binance.exchange_symbols()
        tickers = await self.binance.tickers()
        selected = []
        for s in symbols_info:
            sym = s["symbol"]
            qv = safe_float(tickers.get(sym, {}).get("quoteVolume"))
            if qv >= MIN_QUOTE_VOLUME:
                selected.append((sym, qv))
        selected.sort(key=lambda x: x[1], reverse=True)
        if MAX_SYMBOLS > 0:
            selected = selected[:MAX_SYMBOLS]
        if not selected:
            raise RuntimeError("empty universe")
        return [x[0] for x in selected]

    async def scan_once(self, manual: bool = False, slot_ms: Optional[int] = None) -> str:
        if self.scan_lock.locked():
            return "Скан уже выполняется."
        async with self.scan_lock:
            started = time.perf_counter()
            if slot_ms is None:
                slot_ms = (await self.binance.server_now_ms() // (5 * 60 * 1000)) * (5 * 60 * 1000)
            self.last_error = None
            assert self.binance is not None
            try:
                symbols = await self.universe()
                counts = {k: 0 for k in ("universe", "btc_pass", "trend1h", "pullback15", "breakout5", "volume", "taker", "score_pass", "final", "errors", "stale")}
                counts["universe"] = len(symbols)

                btc_expected = (slot_ms // 3_600_000) * 3_600_000 - 3_600_000
                try:
                    btc1h = await self.binance.klines("BTCUSDT", "1h", KLINE_LIMIT_1H, btc_expected)
                except StaleKlineError as exc:
                    counts["stale"] += 1
                    logger.warning("BTC 1H stale for slot=%s: %s", utc_text(slot_ms), exc)
                    await self.db.write_scan(slot_ms, counts, int((time.perf_counter() - started) * 1000))
                    return f"BTC 1H данные ещё не готовы для слота {utc_text(slot_ms)}"
                if len(btc1h) < MIN_1H_BARS:
                    logger.warning("Недостаточно истории BTC 1H: got=%d need=%d", len(btc1h), MIN_1H_BARS)
                    counts["stale"] += 1
                    await self.db.write_scan(slot_ms, counts, int((time.perf_counter() - started) * 1000))
                    return f"Недостаточно истории BTC 1H: {len(btc1h)}/{MIN_1H_BARS}"
                btc_e200 = ema([c.close for c in btc1h], 200)[-1]
                btc_direction = "LONG" if btc1h[-1].close > btc_e200 else "SHORT"
                counts["btc_pass"] = 1

                results: list[Signal] = []
                near_miss: list[tuple[Signal, str]] = []

                async def process(sym: str) -> None:
                    try:
                        h1_expected = (slot_ms // 3_600_000) * 3_600_000 - 3_600_000
                        h1 = await self.binance.klines(sym, "1h", KLINE_LIMIT_1H, h1_expected)
                        if len(h1) < MIN_1H_BARS:
                            return
                        hclose = h1[-1].close
                        h20 = ema([c.close for c in h1], 20)[-1]
                        h50 = ema([c.close for c in h1], 50)[-1]
                        h200 = ema([c.close for c in h1], 200)[-1]
                        trend_ok = (h20 > h50 and hclose > h200) if btc_direction == "LONG" else (h20 < h50 and hclose < h200)
                        if not trend_ok:
                            return
                        counts["trend1h"] += 1

                        m15_expected = (slot_ms // 900_000) * 900_000 - 900_000
                        m15 = await self.binance.klines(sym, "15m", KLINE_LIMIT_15M, m15_expected)
                        if len(m15) < 60:
                            return
                        atr15 = atr(m15, 14)
                        r15 = rsi(m15, 14)
                        if not atr15 or r15 is None:
                            return
                        e20 = ema([c.close for c in m15], 20)[-1]
                        e50 = ema([c.close for c in m15], 50)[-1]
                        if btc_direction == "LONG":
                            pre = e20 > e50 and 35 <= r15 <= 48 and min(c.low for c in m15[-3:]) <= max(e20, e50) and m15[-1].close > min(e20, e50)
                        else:
                            pre = e20 < e50 and 52 <= r15 <= 65 and max(c.high for c in m15[-3:]) >= min(e20, e50) and m15[-1].close < max(e20, e50)
                        if not pre:
                            return
                        counts["pullback15"] += 1

                        m5_expected = (slot_ms // 300_000) * 300_000 - 300_000
                        m5 = await self.binance.klines(sym, "5m", KLINE_LIMIT_5M, m5_expected)
                        if len(m5) < 25:
                            return
                        breakout = m5[-1]
                        prev3 = m5[-4:-1]
                        breakout_ok = (breakout.close > max(c.high for c in prev3)) if btc_direction == "LONG" else (breakout.close < min(c.low for c in prev3))
                        if not breakout_ok:
                            return
                        counts["breakout5"] += 1
                        prev20 = m5[-21:-1]
                        vol_sma = sma([c.volume for c in prev20], 20)
                        if not vol_sma or breakout.volume / vol_sma < VOLUME_RATIO_MIN:
                            return
                        counts["volume"] += 1
                        taker_ratio = breakout.taker_buy_quote / breakout.quote_volume if breakout.quote_volume else 0.5
                        counts["taker"] += 1
                        funding = await self.binance.funding(sym)
                        sig, reason = await self.strategy.evaluate(sym, btc1h, h1, m15, m5, funding)
                        if not sig:
                            return
                        if reason == "score":
                            near_miss.append((sig, reason))
                            return
                        counts["score_pass"] += 1
                        cooldown_ms = EPISODE_COOLDOWN_BARS * 5 * 60 * 1000
                        if await self.db.episode_exists(sym, sig.direction, sig.episode_ts, cooldown_ms):
                            return
                        results.append(sig)
                    except StaleKlineError as exc:
                        counts["stale"] += 1
                        logger.debug("%s stale: %s", sym, exc)
                    except Exception as exc:
                        counts["errors"] += 1
                        logger.debug("%s failed: %s", sym, exc, exc_info=True)

                await asyncio.gather(*(process(s) for s in symbols))
                counts["final"] = len(results)
                duration_ms = int((time.perf_counter() - started) * 1000)
                scan_id = await self.db.write_scan(slot_ms, counts, duration_ms)

                sent = 0
                for sig, reason in near_miss:
                    try:
                        await self.db.insert_signal(sig, scan_id, "NEAR_MISS", reason)
                    except Exception:
                        logger.exception("Failed to persist near-miss: %s %s", sig.symbol, reason)

                delivered = 0
                for sig in sorted(results, key=lambda x: x.score, reverse=True):
                    sid = await self.db.insert_signal_with_lifecycle(sig, scan_id)
                    if sid is None:
                        continue
                    sig.signal_id = sid
                    sent += 1
                    if await self.send_signal(sig):
                        delivered += 1

                for row in await self.db.pending_telegram_signals(limit=5):
                    try:
                        pending = self.signal_from_row(row)
                        if await self.send_signal(pending):
                            delivered += 1
                    except Exception:
                        logger.exception("Telegram reconciliation failed: signal_id=%s", row["signal_id"])

                self.last_scan_ms = await self.binance.server_now_ms()
                self.last_scan_mono = time.monotonic()
                self.last_error = None
                if counts["errors"] > max(1, int(counts["universe"] * 0.10)):
                    logger.warning("High scan error rate: errors=%d universe=%d stale=%d", counts["errors"], counts["universe"], counts["stale"])
                logger.info(
                    "SCAN slot=%s universe=%d btc=%d trend=%d pullback=%d breakout=%d volume=%d taker=%d score=%d final=%d errors=%d stale=%d duration_ms=%d",
                    utc_text(slot_ms), counts["universe"], counts["btc_pass"], counts["trend1h"],
                    counts["pullback15"], counts["breakout5"], counts["volume"], counts["taker"],
                    counts["score_pass"], counts["final"], counts["errors"], counts["stale"], duration_ms,
                )
                return f"Скан завершён: universe={counts['universe']}, финальных сигналов={sent}, доставлено={delivered}, near-miss={len(near_miss)}, {duration_ms} ms"
            except Exception as exc:
                self.scan_errors += 1
                self.last_error = repr(exc)
                logger.exception("Scan failed")
                return f"Ошибка скана: {exc}"

    def signal_from_row(self, row: aiosqlite.Row) -> Signal:
        return Signal(
            symbol=str(row["symbol"]), direction=str(row["direction"]),
            breakout_open_ts=int(row["breakout_open_ts"]), episode_ts=int(row["episode_ts"]),
            entry_ref=float(row["entry_ref"]), atr15=float(row["atr15"]), atr_ratio=float(row["atr_ratio"]),
            sl=float(row["sl"]), tp1=float(row["tp1"]), tp2=float(row["tp2"]), tp3=float(row["tp3"]),
            score=float(row["score"]), s_vol=float(row["s_vol"]), s_rsi=float(row["s_rsi"]),
            s_trend=float(row["s_trend"]), vol_ratio=float(row["vol_ratio"]),
            taker_ratio=float(row["taker_ratio"]), rsi15=float(row["rsi15"]),
            zone_width_atr=float(row["zone_width_atr"]), trend_dist_atr=float(row["trend_dist_atr"]),
            funding=float(row["funding"]) if row["funding"] is not None else None,
            signal_id=int(row["signal_id"]),
        )

    async def lifecycle_worker(self) -> None:
        last_slot = None
        while self.running:
            try:
                assert self.binance is not None
                server_now = await self.binance.server_now_ms()
                slot_ms = (server_now // 300_000) * 300_000
                if last_slot != slot_ms:
                    async with self.lifecycle_lock:
                        await self.process_signal_lifecycle(slot_ms)
                    last_slot = slot_ms
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = repr(exc)
                logger.exception("Lifecycle worker failed; entering fail-closed state")
                self.running = False
                return

    async def process_signal_lifecycle(self, slot_ms: int) -> None:
        assert self.binance is not None
        active = await self.db.active_signals()
        if not active:
            return

        for row in active:
            try:
                signal_id = int(row["signal_id"])
                symbol = str(row["symbol"])
                direction = str(row["direction"])
                breakout_open_ts = int(row["breakout_open_ts"])
                entry_ref = float(row["entry_ref"])
                atr15 = float(row["atr15"])
                state = str(row["lifecycle_state"])

                expected = (slot_ms // 300_000) * 300_000 - 300_000
                if expected <= breakout_open_ts:
                    continue

                age_bars = (expected - breakout_open_ts) // 300_000
                if age_bars < 1:
                    continue

                if age_bars >= SIGNAL_LIFETIME_BARS:
                    price = None
                    extension = None
                    try:
                        candles = await self.binance.klines(symbol, "5m", 3, expected)
                        if candles and candles[-1].open_ts == expected:
                            price = candles[-1].close
                            if math.isfinite(atr15) and atr15 > 0 and math.isfinite(entry_ref) and entry_ref > 0:
                                extension = (price - entry_ref) / atr15 if direction == "LONG" else (entry_ref - price) / atr15
                                if not math.isfinite(extension):
                                    extension = None
                    except Exception as exc:
                        logger.warning(
                            "Lifecycle expiry without price: signal_id=%s symbol=%s age_bars=%s error=%s",
                            signal_id, symbol, age_bars, exc,
                        )
                    changed = await self.db.add_lifecycle_event(signal_id, "EXPIRED", price, extension)
                    if changed:
                        logger.info(
                            "LIFECYCLE EXPIRED symbol=%s direction=%s signal_id=%d price=%s extension=%s age_bars=%d",
                            symbol, direction, signal_id, price, extension, age_bars,
                        )
                    continue

                if not math.isfinite(atr15) or not math.isfinite(entry_ref) or atr15 <= 0 or entry_ref <= 0:
                    logger.warning("Lifecycle skipped invalid frozen refs: signal_id=%s", signal_id)
                    continue

                candles = await self.binance.klines(symbol, "5m", 3, expected)
                if not candles or candles[-1].open_ts != expected:
                    raise StaleKlineError(
                        f"{symbol} 5m lifecycle: expected={expected}, got={candles[-1].open_ts if candles else None}"
                    )
                price = candles[-1].close
                extension = (price - entry_ref) / atr15 if direction == "LONG" else (entry_ref - price) / atr15
                if not math.isfinite(extension):
                    raise RuntimeError(f"Lifecycle produced non-finite extension for signal_id={signal_id}")

                if state not in {"TOO_EXTENDED", "EXPIRED"} and extension > EXTENSION_MAX_ATR:
                    changed = await self.db.add_lifecycle_event(signal_id, "TOO_EXTENDED", price, extension)
                    if changed:
                        logger.info(
                            "LIFECYCLE TOO_EXTENDED symbol=%s direction=%s signal_id=%d price=%s extension=%.3f age_bars=%d",
                            symbol, direction, signal_id, price, extension, age_bars,
                        )
                    state = "TOO_EXTENDED"

                if state == "DETECTED" and extension <= EXTENSION_MAX_ATR:
                    changed = await self.db.add_lifecycle_event(signal_id, "VALID", price, extension)
                    if changed:
                        logger.info(
                            "LIFECYCLE VALID symbol=%s direction=%s signal_id=%d price=%s extension=%.3f age_bars=%d",
                            symbol, direction, signal_id, price, extension, age_bars,
                        )
            except StaleKlineError as exc:
                logger.debug("Lifecycle stale for signal_id=%s: %s", row["signal_id"], exc)
            except Exception:
                logger.exception("Lifecycle processing failed for signal_id=%s", row["signal_id"])

    async def send_signal(self, s: Signal) -> bool:
        if not self.bot or not TELEGRAM_CHAT_ID or not s.signal_id:
            return False
        rr1 = 2.0
        rr2 = 3.0
        rr3 = 4.0
        funding_line = f"Funding: {s.funding:.6f}" if s.funding is not None else "Funding: n/a"
        text = (
            f"🚨 <b>{s.symbol}</b> {s.direction}\n\n"
            f"Quality Score: <b>{s.score:.0f}/100</b>\n"
            f"Entry ref: <code>{s.entry_ref:.8g}</code>\n"
            f"SL: <code>{s.sl:.8g}</code>\n"
            f"TP1: <code>{s.tp1:.8g}</code>  ({rr1:.1f}R)\n"
            f"TP2: <code>{s.tp2:.8g}</code>  ({rr2:.1f}R)\n"
            f"TP3: <code>{s.tp3:.8g}</code>  ({rr3:.1f}R)\n\n"
            f"1H trend distance: {s.trend_dist_atr:.2f} ATR\n"
            f"15M RSI: {s.rsi15:.1f}\n"
            f"15M zone width: {s.zone_width_atr:.2f} ATR\n"
            f"5M VolRatio: {s.vol_ratio:.2f}x\n"
            f"Taker Buy Ratio: {s.taker_ratio:.3f}\n"
            f"ATR regime: {s.atr_ratio:.2f}x median\n"
            f"{funding_line}"
        )
        delay = 1.0
        for attempt in range(3):
            try:
                await self.bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode="HTML")
            except TelegramRetryAfter as exc:
                wait_s = min(max(1.0, float(exc.retry_after)), 60.0)
                logger.warning("Telegram rate limit for signal_id=%s; retrying in %.1fs", s.signal_id, wait_s)
                if attempt + 1 >= 3:
                    return False
                await asyncio.sleep(wait_s)
                continue
            except (TelegramBadRequest, TelegramForbiddenError) as exc:
                logger.error("Permanent Telegram delivery error: signal_id=%s error=%s", s.signal_id, exc)
                return False
            except Exception as exc:
                if attempt + 1 >= 3:
                    logger.exception("Telegram send failed: signal_id=%s", s.signal_id)
                    return False
                logger.warning("Telegram send retry signal_id=%s: %s", s.signal_id, exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)
                continue

            try:
                await self.db.mark_telegram_sent(s.signal_id)
            except Exception:
                logger.exception("Telegram delivered but DB acknowledgement failed: signal_id=%s", s.signal_id)
                return False
            return True
        return False

# --------------------------- main ---------------------------

async def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = App()
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def request_stop() -> None:
        if not stop_event.is_set():
            stop_event.set()
            app.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, request_stop)

    task = asyncio.create_task(app.start())
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, _ = await asyncio.wait(
            {task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            task.result()
        else:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    finally:
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task
        await app.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
