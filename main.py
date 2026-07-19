import asyncio
import os
import logging
import html
import sqlite3
import json
from datetime import datetime, timedelta, timezone

import aiohttp
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove

# -------------------------- ИНИЦИАЛИЗАЦИЯ --------------------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN")

if not BOT_TOKEN or not TIINGO_TOKEN:
    raise ValueError("КРИТИЧЕСКАЯ ОШИБКА: Проверьте токены BOT_TOKEN и TIINGO_TOKEN в .env!")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("CryptoQuantScorer")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

active_tasks = {}
ACTIVE_TASKS_LOCK = asyncio.Lock()
GLOBAL_CACHE = {}
CACHE_LOCK = asyncio.Lock()
ACTIVE_SUBSCRIPTIONS = {}
SUBS_LOCK = asyncio.Lock()
API_SEMAPHORE = asyncio.Semaphore(2)

ASSET_CONFIG = {
    "BTC/USDT": {"source": "bybit", "symbol": "BTCUSDT", "category": "spot"},
    "ETH/USDT": {"source": "bybit", "symbol": "ETHUSDT", "category": "spot"},
    "SOL/USDT": {"source": "bybit", "symbol": "SOLUSDT", "category": "spot"},
    "TON/USDT": {"source": "bybit", "symbol": "TONUSDT", "category": "spot"},
    "EUR/USD": {"source": "tiingo", "ticker": "eurusd"},
    "GBP/USD": {"source": "tiingo", "ticker": "gbpusd"},
    "USD/JPY": {"source": "tiingo", "ticker": "usdjpy"},
    "XAU/USD": {"source": "tiingo", "ticker": "xauusd"},
}

BYBIT_INTERVALS = {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}
TIINGO_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1hour"}

# -------------------------- КЛАВИАТУРЫ --------------------------
def get_assets_kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="BTC/USDT"), KeyboardButton(text="ETH/USDT")],
            [KeyboardButton(text="SOL/USDT"), KeyboardButton(text="TON/USDT")],
            [KeyboardButton(text="EUR/USD"), KeyboardButton(text="GBP/USD")],
            [KeyboardButton(text="USD/JPY"), KeyboardButton(text="XAU/USD")],
            [KeyboardButton(text="❓ Помощь")]
        ],
        resize_keyboard=True
    )

def get_tf_kb():
    return ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="1m"), KeyboardButton(text="5m")], [KeyboardButton(text="🔙 Назад")]], resize_keyboard=True)

def get_modes_kb():
    return ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="⚡ Быстрый")], [KeyboardButton(text="🎯 Снайпер")], [KeyboardButton(text="🔙 Назад")]], resize_keyboard=True)

def get_stop_kb():
    return ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="🛑 Остановить анализ")]], resize_keyboard=True)

# -------------------------- ИНДИКАТОРЫ И PRICE ACTION --------------------------
def compute_indicators(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 50:
        return {}

    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    volume = df.get('volume', pd.Series(1, index=close.index)).astype(float)

    # Трендовые
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    typical_price = (high + low + close) / 3
    rolling_vwap = ((typical_price * volume).rolling(window=50).sum() / (volume.rolling(window=50).sum() + 1e-10))

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-10)))

    vol_sma = volume.rolling(20).mean()
    vol_ratio = volume / (vol_sma + 1e-10)

    # Локальные экстремумы (для определения пробоя)
    local_high = high.shift(1).rolling(10).max()
    local_low = low.shift(1).rolling(10).min()

    # Свинг-уровни (20 свечей)
    swing_high = high.shift(1).rolling(20).max()
    swing_low = low.shift(1).rolling(20).min()

    # Предыдущая свеча для паттернов поглощения
    prev_open = open_p.shift(1).iloc[-1] if len(open_p) >= 2 else None
    prev_close = close.shift(1).iloc[-1] if len(close) >= 2 else None
    prev_high = high.shift(1).iloc[-1] if len(high) >= 2 else None
    prev_low = low.shift(1).iloc[-1] if len(low) >= 2 else None

    return {
        "open": open_p.iloc[-1], "high": high.iloc[-1], "low": low.iloc[-1], "close": close.iloc[-1],
        "ema9": ema9.iloc[-1], "ema21": ema21.iloc[-1], "ema50": ema50.iloc[-1], "VWAP": rolling_vwap.iloc[-1],
        "RSI": rsi.iloc[-1], "ATR": atr.iloc[-1], "vol_ratio": vol_ratio.iloc[-1],
        "break_high": bool(close.iloc[-1] > local_high.iloc[-1]),
        "break_low": bool(close.iloc[-1] < local_low.iloc[-1]),
        "swing_high": swing_high.iloc[-1] if not pd.isna(swing_high.iloc[-1]) else None,
        "swing_low": swing_low.iloc[-1] if not pd.isna(swing_low.iloc[-1]) else None,
        "prev_open": prev_open,
        "prev_close": prev_close,
        "prev_high": prev_high,
        "prev_low": prev_low,
    }

# -------------------------- СВЕЧНЫЕ ПАТТЕРНЫ --------------------------
def is_hammer(open_, close_, high_, low_):
    body = abs(close_ - open_)
    lower_wick = min(open_, close_) - low_
    upper_wick = high_ - max(open_, close_)
    total_range = high_ - low_
    if total_range == 0: return False
    return (body < total_range * 0.3) and (lower_wick >= body * 2) and (upper_wick < body * 0.5)

def is_shooting_star(open_, close_, high_, low_):
    body = abs(close_ - open_)
    upper_wick = high_ - max(open_, close_)
    lower_wick = min(open_, close_) - low_
    total_range = high_ - low_
    if total_range == 0: return False
    return (body < total_range * 0.3) and (upper_wick >= body * 2) and (lower_wick < body * 0.5)

def is_bullish_engulfing(open_cur, close_cur, open_prev, close_prev):
    if open_prev is None or close_prev is None: return False
    return (close_prev < open_prev) and (close_cur > open_cur) and \
           (close_cur > open_prev) and (open_cur < close_prev)

def is_bearish_engulfing(open_cur, close_cur, open_prev, close_prev):
    if open_prev is None or close_prev is None: return False
    return (close_prev > open_prev) and (close_cur < open_cur) and \
           (open_cur > close_prev) and (close_cur < open_prev)

def is_doji(open_, close_, high_, low_):
    body = abs(close_ - open_)
    total_range = high_ - low_
    return total_range > 0 and body <= total_range * 0.05

# -------------------------- ЗАПРОСЫ ДАННЫХ --------------------------
async def fetch_candles_bybit(symbol: str, category: str, interval: str, limit: int = 300) -> pd.DataFrame:
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": category, "symbol": symbol, "interval": BYBIT_INTERVALS[interval], "limit": limit}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data["retCode"] == 0:
                        rows = data["result"]["list"]
                        columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
                        df = pd.DataFrame(rows, columns=columns)
                        df = df.iloc[::-1]
                        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
                        df["timestamp"] = pd.to_numeric(df["timestamp"]) // 1000
                        return df
    except Exception as e:
        logger.error(f"Ошибка Bybit: {e}")
    return pd.DataFrame()

async def fetch_candles_tiingo(ticker: str, freq: str, lookback_minutes: int = 600) -> pd.DataFrame:
    url = "https://api.tiingo.com/tiingo/fx/prices"
    headers = {"Authorization": f"Token {TIINGO_TOKEN}"}
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(minutes=lookback_minutes)
    params = {
        "tickers": ticker, "startDate": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "endDate": end_date.strftime("%Y-%m-%dT%H:%M:%S"), "resampleFreq": freq
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=15) as resp:
                if resp.status == 429: raise Exception("Tiingo rate limit")
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        df = pd.DataFrame(data)
                        df = df.rename(columns={"date": "timestamp"})
                        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(np.int64) // 10**9
                        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
                        return df[["timestamp", "open", "high", "low", "close"]]
    except Exception as e:
        logger.error(f"Ошибка Tiingo: {e}")
        raise
    return pd.DataFrame()

async def fetch_with_retry(fetch_func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            result = await fetch_func(*args, **kwargs)
            if not result.empty: return result
        except Exception:
            await asyncio.sleep(2 ** attempt)
    return pd.DataFrame()

async def fetch_asset_data(asset: str, timeframe: str, target_candle_key: int):
    config = ASSET_CONFIG[asset]
    source = config["source"]
    higher_tf = {"1m": "15m", "5m": "1h"}[timeframe]

    async with API_SEMAPHORE:
        try:
            if source == "bybit":
                df_worker = await fetch_with_retry(fetch_candles_bybit, config["symbol"], config["category"], timeframe)
                await asyncio.sleep(0.3)
                df_higher = await fetch_with_retry(fetch_candles_bybit, config["symbol"], config["category"], higher_tf)
            else:
                df_worker = await fetch_candles_tiingo(config["ticker"], TIINGO_FREQ[timeframe])
                await asyncio.sleep(0.3)
                df_higher = await fetch_candles_tiingo(config["ticker"], TIINGO_FREQ[higher_tf])
        except Exception:
            return None

    if df_worker.empty: return None

    low_indicators = compute_indicators(df_worker)
    high_indicators = compute_indicators(df_higher) if not df_higher.empty else None

    return {
        "low": low_indicators,
        "high": high_indicators,
        "candle_key": int(df_worker["timestamp"].iloc[-1])
    }

async def global_data_puller():
    while True:
        async with SUBS_LOCK:
            timeframes = {tf for (_, tf), cnt in ACTIVE_SUBSCRIPTIONS.items() if cnt > 0}
        interval = 60 if '1m' in timeframes else 300
        await asyncio.sleep(interval)

        tasks, keys = [], []
        async with SUBS_LOCK:
            for (asset, tf), count in ACTIVE_SUBSCRIPTIONS.items():
                if count > 0:
                    tasks.append(fetch_asset_data(asset, tf, 0))
                    keys.append((asset, tf))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            async with CACHE_LOCK:
                for key, res in zip(keys, results):
                    if isinstance(res, dict) and res: GLOBAL_CACHE[key] = res

# -------------------------- БАЗА ДАННЫХ --------------------------
DB_PATH = "signals.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY, ts TEXT, asset TEXT, tf TEXT, mode TEXT,
        verdict TEXT, score REAL, close_val REAL, features TEXT
    )""")
    conn.commit()
    conn.close()

def log_signal(asset, timeframe, mode, verdict, score, close_val, features):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO signals (ts, asset, tf, mode, verdict, score, close_val, features) VALUES (?,?,?,?,?,?,?,?)",
                 (datetime.now(timezone.utc).isoformat(), asset, timeframe, mode, verdict, score, close_val, json.dumps(features)))
    conn.commit()
    conn.close()

# -------------------------- СКОРИНГ СИГНАЛОВ (НОВАЯ ФУНКЦИЯ) --------------------------
def calculate_metrics(mode, data_pack, timeframe, asset):
    ind = data_pack.get("low", {})
    history = data_pack.get("history", [])

    # Нужно минимум 3 свечи
    if not ind or len(history) < 3:
        return "⚪️ НАБЛЮДЕНИЕ", 0, {"block": "Нет данных"}, 0, 0

    c1 = history[-1]
    c2 = history[-2]
    c3 = history[-3]

    def is_green(c): return c['close'] > c['open']
    def is_red(c): return c['close'] < c['open']

    body1 = abs(c1['close'] - c1['open'])
    
    # Берем волатильность и отношение объема (Volume / SMA Volume)
    atr = ind.get("ATR", 0.0001)
    vol_ratio = ind.get("vol_ratio", 1)
    price = c1['close']

    # Настройки порогов для режимов
    if mode == "⚡ Быстрый":
        vol_threshold = 1.5  # Объем в 1.5 раза выше среднего
        body_threshold = atr * 0.8
    else: # Снайпер
        vol_threshold = 2.5  # Экстремальный объем (в 2.5 раза выше)
        body_threshold = atr * 1.2

    # ЛОВУШКА 1: Кульминация покупок (Blow-off Top)
    # Три зеленые свечи подряд, последняя - огромная И с аномальным объемом
    if is_green(c1) and is_green(c2) and is_green(c3):
        if body1 > body_threshold and vol_ratio > vol_threshold:
            # Толпа в эйфории залетает в лонг. Мы идем с ними, чтобы слить на откате.
            return "📈 ВВЕРХ", 5, {"pattern": "🚨 Кульминация покупок (Лонг на хае)"}, price, atr

    # ЛОВУШКА 2: Капитуляция (Panic Bottom)
    # Три красные свечи подряд, последняя - огромная И с аномальным объемом
    if is_red(c1) and is_red(c2) and is_red(c3):
        if body1 > body_threshold and vol_ratio > vol_threshold:
            # Толпа в панике закрывает лонги. Мы шортим в самое дно.
            return "📉 ВНИЗ", -5, {"pattern": "🚨 Капитуляция (Шорт в дно)"}, price, atr

    return "⚪️ НАБЛЮДЕНИЕ", 0, {"block": "Ждем кульминации"}, price, atr

# -------------------------- ФОРМАТИРОВАНИЕ СООБЩЕНИЙ --------------------------
def format_output_message(asset, tf, mode, data):
    verdict, score, details, price, spread = calculate_metrics(mode, data, tf, asset)
    if 'block' in details:
        score_bar = "⚪️ Контекст неблагоприятный"
    else:
        norm = max(-4, min(4, int(score / 1.5)))
        score_bar = "🟢 " + "▪️" * (norm + 4) + "|⚽️|" + "▪️" * (4 - norm) + " 🔴"

    msg = (
        f"📊 <b>АНАЛИЗ | {html.escape(asset)}</b>\n"
        f"⏱ ТФ: {html.escape(tf)} | Режим: <code>{html.escape(mode)}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📋 Контекст: <b>{html.escape(details.get('structure', 'Нейтрально'))}</b>\n"
    )
    if 'block' in details:
        msg += f"⚠️ {html.escape(details['block'])}\n"
    if 'pattern' in details:
        msg += f"🕯 Паттерн: {html.escape(details['pattern'])}\n"
    if 'triggers' in details:
        msg += f"🎯 Триггеры: {html.escape(details['triggers'])}\n"
    if 'warning' in details:
        msg += f"{html.escape(details['warning'])}\n"
    if 'swing' in details:
        msg += f"{html.escape(details['swing'])}\n"

    msg += (
        f"<code>{html.escape(score_bar)}</code>\n\n"
        f"🔥 Сигнал: <b>{html.escape(verdict)}</b> (Сила: {score})\n"
        f"📍 Цена: <code>{price:.5f}</code> | ATR: <code>{spread:.5f}</code>"
    )
    return msg

async def user_monitoring_cycle(user_id: int, asset: str, timeframe: str, mode: str):
    sub_key = (asset, timeframe)
    async with SUBS_LOCK: ACTIVE_SUBSCRIPTIONS[sub_key] = ACTIVE_SUBSCRIPTIONS.get(sub_key, 0) + 1
    try:
        await bot.send_message(user_id, f"📡 Сигнальный мониторинг {asset} запущен ({timeframe}/{mode})")
        last_candle = None
        while True:
            async with CACHE_LOCK: new_data = GLOBAL_CACHE.get(sub_key)
            if new_data and new_data["candle_key"] != last_candle:
                last_candle = new_data["candle_key"]
                await bot.send_message(user_id, format_output_message(asset, timeframe, mode, new_data), parse_mode="HTML")
            await asyncio.sleep(2)
    finally:
        async with ACTIVE_TASKS_LOCK: active_tasks.pop(user_id, None)
        async with SUBS_LOCK: ACTIVE_SUBSCRIPTIONS[sub_key] = max(0, ACTIVE_SUBSCRIPTIONS.get(sub_key, 0) - 1)

class ScannerStates(StatesGroup):
    choosing_asset, choosing_timeframe, choosing_mode, monitoring = State(), State(), State(), State()

@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await state.set_state(ScannerStates.choosing_asset)
    await message.answer("Какой актив анализируем?", reply_markup=get_assets_kb())

@dp.message(ScannerStates.choosing_asset, F.text.in_(ASSET_CONFIG.keys()))
async def process_asset(message: Message, state: FSMContext):
    await state.update_data(asset=message.text)
    await state.set_state(ScannerStates.choosing_timeframe)
    await message.answer("Выберите таймфрейм:", reply_markup=get_tf_kb())

@dp.message(ScannerStates.choosing_timeframe, F.text.in_(["1m", "5m"]))
async def process_tf(message: Message, state: FSMContext):
    await state.update_data(timeframe=message.text)
    await state.set_state(ScannerStates.choosing_mode)
    await message.answer("Выберите режим скоринга:", reply_markup=get_modes_kb())

@dp.message(ScannerStates.choosing_mode, F.text.in_(["⚡ Быстрый", "🎯 Снайпер"]))
async def process_mode(message: Message, state: FSMContext):
    data = await state.get_data()
    asset, tf, mode = data['asset'], data['timeframe'], message.text
    user_id = message.from_user.id
    async with ACTIVE_TASKS_LOCK:
        if old := active_tasks.pop(user_id, None): old.cancel()
        await state.set_state(ScannerStates.monitoring)
        active_tasks[user_id] = asyncio.create_task(user_monitoring_cycle(user_id, asset, tf, mode))
    await message.answer("Слежение активировано.", reply_markup=get_stop_kb())

@dp.message(F.text == "🛑 Остановить анализ")
async def stop_monitoring(message: Message, state: FSMContext):
    async with ACTIVE_TASKS_LOCK:
        if task := active_tasks.pop(message.from_user.id, None): task.cancel()
    await state.clear()
    await state.set_state(ScannerStates.choosing_asset)
    await message.answer("Остановлено. Какой актив выберем?", reply_markup=get_assets_kb())

puller_task = None
async def on_startup(bot: Bot):
    global puller_task
    init_db()
    puller_task = asyncio.create_task(global_data_puller())

async def on_shutdown(bot: Bot):
    if puller_task: puller_task.cancel()

dp.startup.register(on_startup)
dp.shutdown.register(on_shutdown)

async def main():
    try: await dp.start_polling(bot)
    finally: await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())