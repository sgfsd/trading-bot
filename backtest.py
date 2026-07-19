import asyncio
import logging
from datetime import datetime, timezone
import pandas as pd

# Импорт необходимых функций из основного модуля
from main import (
    fetch_candles_bybit,
    compute_indicators,
    calculate_metrics,
    ASSET_CONFIG,
    init_db,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Backtest")

# Отключаем запись сигналов в БД во время тестирования, чтобы не засорять
import main
main.log_signal = lambda *args, **kwargs: None

def run_backtest_engine(df, df_high, asset, timeframe, mode, invert_strategy=False):
    results = []
    for i in range(200, len(df)):
        window = df.iloc[:i]
        ind = compute_indicators(window)
        if not ind:
            continue

        history = window.tail(30)[['open','high','low','close','volume']].to_dict('records') if 'volume' in window.columns else \
                  window.tail(30)[['open','high','low','close']].to_dict('records')

        data_pack = {"low": ind, "high": None, "history": history, "candle_key": int(df.iloc[i-1]["timestamp"])}
        try:
            verdict, score, details, _, _ = calculate_metrics(mode, data_pack, timeframe, asset)
        except Exception as e:
            continue

        if verdict == "⚪️ НАБЛЮДЕНИЕ":
            continue

        # Инвертируем сигнал, если нужно
        if invert_strategy:
            if "ВВЕРХ" in verdict:
                verdict = "📉 ВНИЗ"
            elif "ВНИЗ" in verdict:
                verdict = "📈 ВВЕРХ"

        next_close = df.iloc[i]["close"]
        next_open = df.iloc[i]["open"]
        correct = None
        if verdict.startswith("📈"):
            correct = next_close > next_open
        elif verdict.startswith("📉"):
            correct = next_close < next_open

        if correct is not None:
            results.append({
                "verdict": verdict,
                "correct": correct
            })
    return pd.DataFrame(results)

def analyze_results(df_res, asset, timeframe, mode_name, test_type, total_candles, start_dt=None, end_dt=None):
    signals_count = len(df_res)
    if signals_count == 0:
        success_rate = 0.0
        success_signals = 0
    else:
        success_signals = df_res["correct"].sum()
        success_rate = df_res["correct"].mean() * 100

    print("=" * 60)
    print(f"📊 {asset} ({timeframe}) | {mode_name} | {test_type}")
    if start_dt and end_dt:
        print(f"Период: {start_dt.strftime('%Y-%m-%d %H:%M')} — {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print("-" * 60)
    print(f"Всего свечей в выборке: {total_candles}")
    print(f"Сигналов сгенерировано: {signals_count} ({signals_count / total_candles:.1%})")
    print(f"Успешные исходы: {int(success_signals)}")
    print(f"🔥 Точность сигналов (Winrate): {success_rate:.2f}%")
    print("=" * 60 + "\n")

async def main_backtest():
    init_db()
    modes = ["⚡ Быстрый", "🎯 Снайпер"]
    timeframes = ["1m", "5m"]
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    for asset in assets:
        if asset not in ASSET_CONFIG:
            continue
        config = ASSET_CONFIG[asset]
        for tf in timeframes:
            higher_tf = {"1m": "15m", "5m": "1h"}[tf]
            print(f"\n📥 Загрузка истории BYBIT для {asset} [{tf}]...")
            try:
                df = await fetch_candles_bybit(config["symbol"], config["category"], tf, limit=1000)
                await asyncio.sleep(0.5)
                df_high = await fetch_candles_bybit(config["symbol"], config["category"], higher_tf, limit=1000)
            except Exception as e:
                logger.error(f"Ошибка загрузки данных {asset} {tf}: {e}")
                continue

            if df.empty:
                logger.warning("Нет данных")
                continue

            df = df.sort_values("timestamp").reset_index(drop=True)
            ts_start = df["timestamp"].iloc[0]
            ts_end = df["timestamp"].iloc[-1]
            if ts_start > 1e11:
                ts_start /= 1000
            if ts_end > 1e11:
                ts_end /= 1000
            start_dt = datetime.fromtimestamp(ts_start, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(ts_end, tz=timezone.utc)

            for mode in modes:
                df_res_direct = run_backtest_engine(df, df_high, asset, tf, mode, invert_strategy=False)
                analyze_results(df_res_direct, asset, tf, mode, "🩸 ПРЯМОЙ СИГНАЛ", len(df), start_dt, end_dt)

                df_res_invert = run_backtest_engine(df, df_high, asset, tf, mode, invert_strategy=True)
                analyze_results(df_res_invert, asset, tf, mode, "🔄 ИНВЕРСИЯ (ANTI-STRATEGY)", len(df), start_dt, end_dt)

                await asyncio.sleep(0.3)
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main_backtest())