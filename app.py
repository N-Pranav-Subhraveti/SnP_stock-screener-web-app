from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import yfinance as yf
import requests
import pandas_ta as ta
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, Optional, List

# Initialize Flask App
app = Flask(__name__, static_folder='static')
CORS(app)
session = requests.Session()

# --- Route to serve the frontend ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# --- Helper & Data Logic ---

def get_tickers(index_name: str = "S&P 500") -> List[str]:
    """Fetches list of tickers from Wikipedia."""
    wiki_pages = {
        "S&P 500": {'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'table_index': 0, 'ticker_col': 'Symbol'},
        "S&P 100": {'url': 'https://en.wikipedia.org/wiki/S%26P_100', 'table_index': 2, 'ticker_col': 'Symbol'}
    }
    if index_name not in wiki_pages: return []
    try:
        page_info = wiki_pages[index_name]
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(page_info['url'], headers=headers)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        index_table = tables[page_info['table_index']]
        if isinstance(index_table.columns, pd.MultiIndex):
            index_table.columns = index_table.columns.get_level_values(0)
        return index_table[page_info['ticker_col']].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        print(f"Error fetching {index_name} tickers: {e}")
        return []

# --- Individual Strategy Checkers (Now with Parameters) ---

def check_ma_crossover_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks for a bullish SMA crossover using user-defined windows."""
    short_window = params.get('short_window', 50)
    long_window = params.get('long_window', 200)
    
    df[f'SMA{short_window}'] = df['Close'].rolling(window=short_window).mean()
    df[f'SMA{long_window}'] = df['Close'].rolling(window=long_window).mean()
    df.dropna(inplace=True)
    if len(df) < 2: return False, None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if prev[f'SMA{short_window}'] < prev[f'SMA{long_window}'] and last[f'SMA{short_window}'] > last[f'SMA{long_window}']:
        return True, {f"SMA{short_window}": f"{last[f'SMA{short_window}']:.2f}", f"SMA{long_window}": f"{last[f'SMA{long_window}']:.2f}"}
    return False, None

def check_rsi_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks if RSI is below a user-defined oversold threshold."""
    length = params.get('rsi_length', 14)
    threshold = params.get('rsi_threshold', 30)
    
    df.ta.rsi(length=length, append=True)
    df.dropna(inplace=True)
    if df.empty: return False, None
    
    last_rsi = df.iloc[-1][f'RSI_{length}']
    if last_rsi < threshold:
        return True, {"RSI": f"{last_rsi:.2f}"}
    return False, None

def check_supertrend_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks if the Supertrend signal is bullish using user-defined parameters."""
    length = params.get('supertrend_length', 7)
    multiplier = params.get('supertrend_multiplier', 3.0)
    
    df.ta.supertrend(length=length, multiplier=multiplier, append=True)
    df.dropna(inplace=True)
    if df.empty or f'SUPERTd_{length}_{multiplier}' not in df.columns: return False, None
    
    if df.iloc[-1][f'SUPERTd_{length}_{multiplier}'] == 1:
        return True, {"Supertrend": f"{df.iloc[-1][f'SUPERT_{length}_{multiplier}']:.2f}", "Direction": "Up"}
    return False, None

def filter_by_ha_pattern(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Filters by a specific user-defined Heikin-Ashi pattern."""
    pattern = params.get('pattern', 'RRGG')
    df.ta.ha(append=True)
    df.dropna(inplace=True)
    num_candles = len(pattern)
    if len(df) < num_candles: return False, None
    
    latest_candles = df.iloc[-num_candles:]
    for i in range(num_candles):
        candle = latest_candles.iloc[i]
        actual_color = 'G' if candle['HA_close'] > candle['HA_open'] else 'R'
        if actual_color != pattern[i].upper(): return False, None
    return True, {"Pattern": pattern}

# --- ADAPTIVE UPTREND FILTER ---
def check_uptrend_filter(df: pd.DataFrame, strategy_name: str) -> Tuple[bool, Dict[str, Any]]:
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df.ta.rsi(length=14, append=True)
    df['Volume_Avg_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_Avg_20']
    df['High_20'] = df['High'].rolling(window=20).max()
    df['Price_vs_High_20'] = df['Close'] / df['High_20']
    df.dropna(inplace=True)

    if df.empty: return False, {}
    
    last = df.iloc[-1]
    price, ema_20, ema_50, rsi = last['Close'], last['EMA_20'], last['EMA_50'], last['RSI_14']
    volume_ratio, price_vs_high = last['Volume_Ratio'], last['Price_vs_High_20']

    if strategy_name == 'rsi':
        conditions = { 'price_above_ema20': price > ema_20 * 0.98, 'price_above_ema50': price > ema_50 * 0.97, 'rsi_reasonable': 30 <= rsi <= 75, 'volume_adequate': volume_ratio > 0.7 }
        required_passes = 3
    elif strategy_name == 'ma_crossover':
        conditions = { 'price_above_ema20': price > ema_20, 'price_above_ema50': price > ema_50, 'rsi_reasonable': 35 <= rsi <= 80, 'volume_adequate': volume_ratio > 0.6, 'near_highs': price_vs_high > 0.85 }
        required_passes = 3
    elif strategy_name in ['ha_pattern', 'supertrend']:
        conditions = { 'price_above_ema50': price > ema_50 * 0.95, 'rsi_not_extreme': rsi > 25, 'volume_adequate': volume_ratio > 0.5 }
        required_passes = 2
    else:
        conditions = { 'price_above_ema20': price > ema_20 * 0.98, 'price_above_ema50': price > ema_50, 'rsi_reasonable': 30 <= rsi <= 80, 'volume_adequate': volume_ratio > 0.6 }
        required_passes = 3

    pass_count = sum(conditions.values())
    
    if pass_count >= required_passes:
        trend_strength = "Strong" if pass_count == len(conditions) else "Moderate"
        return True, {"Trend Strength": trend_strength}
    else:
        return False, {"Trend Strength": "Weak"}

# --- Core Data Processor ---
def process_ticker_data(ticker: str, data: pd.DataFrame, strategy: str, params: Dict, timeframe: str, apply_uptrend_filter: bool) -> Optional[Dict[str, Any]]:
    try:
        if data.empty or len(data) < 52: return None

        if timeframe == 'weekly':
            if not isinstance(data.index, pd.DatetimeIndex): data.index = pd.to_datetime(data.index)
            data = data.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
            if len(data) < 52: return None

        strategy_funcs = {
            'ma_crossover': check_ma_crossover_signal, 'rsi': check_rsi_signal,
            'supertrend': check_supertrend_signal, 'ha_pattern': filter_by_ha_pattern
        }
        
        if strategy not in strategy_funcs: return None
        
        # --- THIS IS THE FIX ---
        # Removed the `**` before `params` to pass it as a single dictionary.
        is_signal, signal_data = strategy_funcs[strategy](data.copy(), params)
        # --- END FIX ---

        if not is_signal: return None

        uptrend_data = {}
        if apply_uptrend_filter:
            is_uptrend, uptrend_data = check_uptrend_filter(data.copy(), strategy)
            if not is_uptrend: return None
        
        result = {'Ticker': ticker, 'Close': f"${data.iloc[-1]['Close']:.2f}"}
        if signal_data: result.update(signal_data)
        if uptrend_data: result.update(uptrend_data)
        return result
    except Exception as e:
        print(f"  -> Error processing {ticker}: {e}")
        return None

# --- Main API Endpoint ---
@app.route('/run-screener', methods=['POST'])
def handle_screener_request():
    try:
        req_data = request.get_json()
        params = req_data.get('params', {})
        print(f"Request: {req_data.get('index')}, {req_data.get('strategy')}, Params: {params}, Uptrend Filter: {req_data.get('applyUptrendFilter')}")
        
        tickers = get_tickers(req_data.get('index'))
        if not tickers: return jsonify({"error": f"Could not fetch tickers for {req_data.get('index')}."}), 500

        all_data = yf.download(tickers, period="2y", auto_adjust=True, session=session, progress=False, group_by='ticker')
        
        matching_stocks = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            ticker_data_map = {t: all_data[t] for t in tickers if isinstance(all_data[t], pd.DataFrame) and not all_data[t].empty}
            futures = [executor.submit(process_ticker_data, ticker, df, req_data.get('strategy'), params, req_data.get('timeframe'), req_data.get('applyUptrendFilter')) for ticker, df in ticker_data_map.items()]
            for future in futures:
                result = future.result()
                if result: matching_stocks.append(result)

        total_in_index, failed = len(tickers), list(set(tickers) - set(ticker_data_map.keys()))
        print(f"Scan complete. Found {len(matching_stocks)} matching stocks.")
        return jsonify({"matching_stocks": matching_stocks, "total_scanned": len(ticker_data_map), "total_in_index": total_in_index, "failed_tickers": failed})
    except Exception as e:
        print(f"[SERVER ERROR] An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)

