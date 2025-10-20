from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests
import pandas_ta as ta
import io
import time 
from typing import Dict, Any, Tuple, Optional, List

# Initialize Flask App
app = Flask(__name__, static_folder='static')
CORS(app)
session = curl_requests.Session(impersonate="chrome110")

# --- Route to serve the frontend ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# --- Helper & Data Logic (with Robust Ticker Fetching) ---
def get_tickers(index_name: str = "S&P 500") -> List[str]:
    # --- THIS IS THE FIX ---
    # Wikipedia frequently changes column names. This logic now tries multiple
    # common names to make the scraper more robust.
    wiki_pages = {
        "S&P 500": {'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'possible_cols': ['Symbol']},
        "S&P 100": {'url': 'https://en.wikipedia.org/wiki/S%26P_100', 'possible_cols': ['Symbol', 'Ticker symbol']}
    }
    # --- END FIX ---
    if index_name not in wiki_pages: return []
    try:
        page_info = wiki_pages[index_name]
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = session.get(page_info['url'], headers=headers)
        response.raise_for_status()
        
        table = None
        found_col = None
        for col_name in page_info['possible_cols']:
            try:
                tables = pd.read_html(io.StringIO(response.text), match=col_name, flavor='lxml')
                if tables:
                    table = tables[0]
                    found_col = col_name
                    break
            except ValueError:
                continue # This column name was not found, try the next one

        if table is None or found_col is None:
             raise ValueError(f"Could not find a valid ticker table for {index_name}.")
        
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.get_level_values(0)
            
        return table[found_col].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        print(f"Error fetching {index_name} tickers: {e}")
        return []

# --- Individual Strategy Checkers (No changes needed) ---
def check_ma_crossover_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    short_window, long_window = params.get('short_window', 50), params.get('long_window', 200)
    df[f'SMA{short_window}'] = df['Close'].rolling(window=short_window).mean()
    df[f'SMA{long_window}'] = df['Close'].rolling(window=long_window).mean()
    df.dropna(inplace=True)
    if len(df) < 2: return False, None
    last, prev = df.iloc[-1], df.iloc[-2]
    if prev[f'SMA{short_window}'] < prev[f'SMA{long_window}'] and last[f'SMA{short_window}'] > last[f'SMA{long_window}']:
        return True, {f"SMA{short_window}": f"{last[f'SMA{short_window}']:.2f}", f"SMA{long_window}": f"{last[f'SMA{long_window}']:.2f}"}
    return False, None

def check_rsi_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    length, threshold = params.get('rsi_length', 14), params.get('rsi_threshold', 30)
    df.ta.rsi(length=length, append=True)
    df.dropna(inplace=True)
    if df.empty: return False, None
    last_rsi = df.iloc[-1][f'RSI_{length}']
    if last_rsi < threshold: return True, {"RSI": f"{last_rsi:.2f}"}
    return False, None

def check_supertrend_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    length, multiplier = params.get('supertrend_length', 7), params.get('supertrend_multiplier', 3.0)
    df.ta.supertrend(length=length, multiplier=multiplier, append=True)
    df.dropna(inplace=True)
    if df.empty or f'SUPERTd_{length}_{multiplier}' not in df.columns: return False, None
    if df.iloc[-1][f'SUPERTd_{length}_{multiplier}'] == 1:
        return True, {"Supertrend": f"{df.iloc[-1][f'SUPERT_{length}_{multiplier}']:.2f}", "Direction": "Up"}
    return False, None

def filter_by_ha_pattern(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    pattern = params.get('pattern', 'RRGG')
    df.ta.ha(append=True)
    df.dropna(inplace=True)
    if len(df) < len(pattern): return False, None
    latest_candles = df.iloc[-len(pattern):]
    for i in range(len(pattern)):
        actual_color = 'G' if latest_candles.iloc[i]['HA_close'] > latest_candles.iloc[i]['HA_open'] else 'R'
        if actual_color != pattern[i].upper(): return False, None
    return True, {"Pattern": pattern}

def check_uptrend_filter(df: pd.DataFrame, strategy_name: str) -> Tuple[bool, Dict[str, Any]]:
    df['EMA_20'], df['EMA_50'] = ta.ema(df['Close'], length=20), ta.ema(df['Close'], length=50)
    df.ta.rsi(length=14, append=True)
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['Price_vs_High_20'] = df['Close'] / df['High'].rolling(window=20).max()
    df.dropna(inplace=True)
    if df.empty: return False, {}
    last = df.iloc[-1]
    price, ema_20, ema_50, rsi = last['Close'], last['EMA_20'], last['EMA_50'], last['RSI_14']
    volume_ratio, price_vs_high = last['Volume_Ratio'], last['Price_vs_High_20']
    if strategy_name == 'rsi':
        conditions = {'price_above_ema20': price>ema_20*0.98, 'price_above_ema50': price>ema_50*0.97, 'rsi_reasonable': 30<=rsi<=75, 'volume_adequate': volume_ratio>0.7}
        required_passes = 3
    elif strategy_name == 'ma_crossover':
        conditions = {'price_above_ema20': price>ema_20, 'price_above_ema50': price>ema_50, 'rsi_reasonable': 35<=rsi<=80, 'volume_adequate': volume_ratio>0.6, 'near_highs': price_vs_high>0.85}
        required_passes = 3
    elif strategy_name in ['ha_pattern', 'supertrend']:
        conditions = {'price_above_ema50': price>ema_50*0.95, 'rsi_not_extreme': rsi>25, 'volume_adequate': volume_ratio>0.5}
        required_passes = 2
    else:
        conditions = {'price_above_ema20': price>ema_20*0.98, 'price_above_ema50': price>ema_50, 'rsi_reasonable': 30<=rsi<=80, 'volume_adequate': volume_ratio > 0.6}
        required_passes = 3

    pass_count = sum(conditions.values())
    if pass_count >= required_passes:
        return True, {"Trend Strength": "Strong" if pass_count == len(conditions) else "Moderate"}
    return False, {"Trend Strength": "Weak"}

def process_ticker_data(ticker: str, data: pd.DataFrame, strategy: str, params: Dict, timeframe: str, apply_uptrend_filter: bool) -> Optional[Dict[str, Any]]:
    try:
        if data.empty or len(data) < 52: return None
        if timeframe == 'weekly':
            if not isinstance(data.index, pd.DatetimeIndex): data.index = pd.to_datetime(data.index)
            data = data.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            if len(data) < 52: return None
        
        strategy_funcs = {
            'ma_crossover': check_ma_crossover_signal, 'rsi': check_rsi_signal,
            'supertrend': check_supertrend_signal, 'ha_pattern': filter_by_ha_pattern
        }
        if strategy not in strategy_funcs: return None
        
        is_signal, signal_data = strategy_funcs[strategy](data.copy(), params)
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

@app.route('/run-screener', methods=['POST'])
def handle_screener_request():
    try:
        req_data = request.get_json()
        params = req_data.get('params', {})
        print(f"Request: {req_data.get('index')}, {req_data.get('strategy')}, Params: {params}, Uptrend: {req_data.get('applyUptrendFilter')}")
        
        tickers = get_tickers(req_data.get('index'))
        if not tickers: return jsonify({"error": f"Could not fetch tickers for {req_data.get('index')}."}), 500

        matching_stocks, failed_tickers, total_scanned = [], [], 0
        
        chunk_size = 30 
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        for i, chunk in enumerate(ticker_chunks):
            print(f"Processing micro-batch {i+1}/{len(ticker_chunks)} ({len(chunk)} tickers)...")
            
            all_data = yf.download(chunk, period="2y", auto_adjust=True, session=session, progress=False, group_by='ticker')
            
            ticker_data_map = {t: all_data[t] for t in chunk if isinstance(all_data.get(t), pd.DataFrame) and not all_data.get(t).empty}
            total_scanned += len(ticker_data_map)
            failed_tickers.extend(list(set(chunk) - set(ticker_data_map.keys())))

            for ticker, df in ticker_data_map.items():
                result = process_ticker_data(ticker, df, req_data.get('strategy'), params, req_data.get('timeframe'), req_data.get('applyUptrendFilter'))
                if result:
                    matching_stocks.append(result)
            
            print(f"Batch {i+1} complete. Pausing for 2 seconds...")
            time.sleep(2)

        print(f"Scan complete. Found {len(matching_stocks)} matching stocks.")
        return jsonify({
            "matching_stocks": matching_stocks,
            "total_scanned": total_scanned,
            "total_in_index": len(tickers),
            "failed_tickers": failed_tickers
        })

    except Exception as e:
        print(f"[SERVER ERROR] An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error. Please try again later."}), 500

if __name__ == '__main__':
    app.run(debug=True)

