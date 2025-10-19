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
import time

# Initialize Flask App
app = Flask(__name__, static_folder='static')
CORS(app)
session = requests.Session()

# --- Route to serve the frontend ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# --- Helper & Data Logic ---

def get_tickers(index_name: str = "S&P 500") -> List[str]:
    """Fetches list of tickers from Wikipedia with fallback options."""
    wiki_pages = {
        "S&P 500": {'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'ticker_col': 'Symbol'},
        "S&P 100": {'url': 'https://en.wikipedia.org/wiki/S%26P_100', 'ticker_col': 'Symbol'}
    }
    
    if index_name not in wiki_pages: 
        return []
    
    try:
        page_info = wiki_pages[index_name]
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(page_info['url'], headers=headers, timeout=10)
        response.raise_for_status()
        
        # Try multiple table reading strategies
        tables = pd.read_html(io.StringIO(response.text), match=page_info['ticker_col'])
        if not tables:
            # Fallback: try to get any table and find the ticker column
            tables = pd.read_html(io.StringIO(response.text))
            
        if not tables:
            raise ValueError(f"Could not find any tables on the Wikipedia page for {index_name}.")
            
        index_table = tables[0]
        
        # Handle multi-index columns
        if isinstance(index_table.columns, pd.MultiIndex):
            index_table.columns = index_table.columns.get_level_values(0)
        
        # Find the ticker column (case insensitive)
        ticker_col = None
        for col in index_table.columns:
            if page_info['ticker_col'].lower() in col.lower():
                ticker_col = col
                break
        
        if not ticker_col:
            # Use first column as fallback
            ticker_col = index_table.columns[0]
            
        tickers = index_table[ticker_col].dropna().astype(str).str.replace('.', '-', regex=False).tolist()
        return [t for t in tickers if t != 'nan']
        
    except Exception as e:
        print(f"Error fetching {index_name} tickers: {e}")
        # Fallback to a hardcoded list if scraping fails
        return get_fallback_tickers(index_name)

def get_fallback_tickers(index_name: str) -> List[str]:
    """Fallback ticker lists in case Wikipedia scraping fails."""
    if index_name == "S&P 100":
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'BRK-B', 'UNH', 'JNJ', 'XOM', 
                'JPM', 'V', 'PG', 'NVDA', 'HD', 'CVX', 'LLY', 'MA', 'BAC', 'ABBV', 'PFE', 'AVGO', 
                'KO', 'WMT', 'TMO', 'DIS', 'PEP', 'CSCO', 'VZ', 'ADBE', 'CMCSA', 'NFLX', 'ABT', 
                'DHR', 'ACN', 'NKE', 'CRM', 'TXN', 'COST', 'MRK', 'WFC', 'PM', 'LIN', 'RTX', 
                'AMD', 'BMY', 'UPS', 'SBUX', 'T', 'QCOM', 'HON', 'AMGN', 'INTC', 'COP', 'LOW', 
                'INTU', 'SPGI', 'UNP', 'IBM', 'CAT', 'GS', 'PLD', 'DE', 'NOW', 'SYK', 'ELV', 
                'AXP', 'LMT', 'BKNG', 'MDT', 'GE', 'ISRG', 'BLK', 'AMT', 'C', 'TJX', 'VRTX', 
                'PGR', 'ADI', 'MMC', 'ZTS', 'REGN', 'TGT', 'GILD', 'LRCX', 'CVS', 'SCHW', 'MDLZ', 
                'MO', 'CI', 'PNC', 'DUK', 'SO', 'BDX', 'NEE', 'ETN', 'CL', 'BSX', 'APD', 'SLB', 
                'FIS', 'EMR', 'EOG', 'ITW', 'HCA', 'WM', 'CCI']
    else:  # S&P 500 fallback (top 100 as sample)
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'BRK-B', 'UNH', 'JNJ', 'XOM', 
                'JPM', 'V', 'PG', 'NVDA', 'HD', 'CVX', 'LLY', 'MA', 'BAC', 'ABBV']

# --- Individual Strategy Checkers ---

def check_ma_crossover_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks for a bullish SMA crossover using user-defined windows."""
    try:
        short_window = params.get('short_window', 50)
        long_window = params.get('long_window', 200)
        
        df[f'SMA{short_window}'] = df['Close'].rolling(window=short_window).mean()
        df[f'SMA{long_window}'] = df['Close'].rolling(window=long_window).mean()
        df.dropna(inplace=True)
        if len(df) < 2: 
            return False, None
        
        last = df.iloc[-1]
        prev = df.iloc[-2]

        if (prev[f'SMA{short_window}'] < prev[f'SMA{long_window}'] and 
            last[f'SMA{short_window}'] > last[f'SMA{long_window}']):
            return True, {
                f"SMA{short_window}": f"{last[f'SMA{short_window}']:.2f}", 
                f"SMA{long_window}": f"{last[f'SMA{long_window}']:.2f}"
            }
        return False, None
    except Exception as e:
        print(f"Error in MA crossover: {e}")
        return False, None

def check_rsi_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks if RSI is below a user-defined oversold threshold."""
    try:
        length = params.get('rsi_length', 14)
        threshold = params.get('rsi_threshold', 30)
        
        df.ta.rsi(length=length, append=True)
        df.dropna(inplace=True)
        if df.empty: 
            return False, None
        
        last_rsi = df.iloc[-1][f'RSI_{length}']
        if last_rsi < threshold:
            return True, {"RSI": f"{last_rsi:.2f}"}
        return False, None
    except Exception as e:
        print(f"Error in RSI check: {e}")
        return False, None

def check_supertrend_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks if the Supertrend signal is bullish using user-defined parameters."""
    try:
        length = params.get('supertrend_length', 7)
        multiplier = params.get('supertrend_multiplier', 3.0)
        
        df.ta.supertrend(length=length, multiplier=multiplier, append=True)
        df.dropna(inplace=True)
        if df.empty or f'SUPERTd_{length}_{multiplier}' not in df.columns: 
            return False, None
        
        if df.iloc[-1][f'SUPERTd_{length}_{multiplier}'] == 1:
            return True, {
                "Supertrend": f"{df.iloc[-1][f'SUPERT_{length}_{multiplier}']:.2f}", 
                "Direction": "Up"
            }
        return False, None
    except Exception as e:
        print(f"Error in Supertrend: {e}")
        return False, None

def filter_by_ha_pattern(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Filters by a specific user-defined Heikin-Ashi pattern."""
    try:
        pattern = params.get('pattern', 'RRGG')
        df.ta.ha(append=True)
        df.dropna(inplace=True)
        num_candles = len(pattern)
        if len(df) < num_candles: 
            return False, None
        
        latest_candles = df.iloc[-num_candles:]
        for i in range(num_candles):
            candle = latest_candles.iloc[i]
            actual_color = 'G' if candle['HA_close'] > candle['HA_open'] else 'R'
            if actual_color != pattern[i].upper(): 
                return False, None
        return True, {"Pattern": pattern}
    except Exception as e:
        print(f"Error in HA pattern: {e}")
        return False, None

# --- ADAPTIVE UPTREND FILTER ---
def check_uptrend_filter(df: pd.DataFrame, strategy_name: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df.ta.rsi(length=14, append=True)
        df['Volume_Avg_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_Avg_20']
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Price_vs_High_20'] = df['Close'] / df['High_20']
        df.dropna(inplace=True)

        if df.empty: 
            return False, {}
        
        last = df.iloc[-1]
        price, ema_20, ema_50, rsi = last['Close'], last['EMA_20'], last['EMA_50'], last['RSI_14']
        volume_ratio, price_vs_high = last['Volume_Ratio'], last['Price_vs_High_20']

        if strategy_name == 'rsi':
            conditions = { 
                'price_above_ema20': price > ema_20 * 0.98, 
                'price_above_ema50': price > ema_50 * 0.97, 
                'rsi_reasonable': 30 <= rsi <= 75, 
                'volume_adequate': volume_ratio > 0.7 
            }
            required_passes = 3
        elif strategy_name == 'ma_crossover':
            conditions = { 
                'price_above_ema20': price > ema_20, 
                'price_above_ema50': price > ema_50, 
                'rsi_reasonable': 35 <= rsi <= 80, 
                'volume_adequate': volume_ratio > 0.6, 
                'near_highs': price_vs_high > 0.85 
            }
            required_passes = 3
        elif strategy_name in ['ha_pattern', 'supertrend']:
            conditions = { 
                'price_above_ema50': price > ema_50 * 0.95, 
                'rsi_not_extreme': rsi > 25, 
                'volume_adequate': volume_ratio > 0.5 
            }
            required_passes = 2
        else:
            conditions = { 
                'price_above_ema20': price > ema_20 * 0.98, 
                'price_above_ema50': price > ema_50, 
                'rsi_reasonable': 30 <= rsi <= 80, 
                'volume_adequate': volume_ratio > 0.6 
            }
            required_passes = 3

        pass_count = sum(conditions.values())
        
        if pass_count >= required_passes:
            trend_strength = "Strong" if pass_count == len(conditions) else "Moderate"
            return True, {"Trend Strength": trend_strength}
        else:
            return False, {"Trend Strength": "Weak"}
    except Exception as e:
        print(f"Error in uptrend filter: {e}")
        return False, {}

# --- Core Data Processor ---
def process_ticker_data(ticker: str, data: pd.DataFrame, strategy: str, params: Dict, timeframe: str, apply_uptrend_filter: bool) -> Optional[Dict[str, Any]]:
    try:
        if data.empty or len(data) < 52: 
            return None

        if timeframe == 'weekly':
            if not isinstance(data.index, pd.DatetimeIndex): 
                data.index = pd.to_datetime(data.index)
            data = data.resample('W').agg({
                'Open': 'first', 
                'High': 'max', 
                'Low': 'min', 
                'Close': 'last', 
                'Volume': 'sum'
            }).dropna()
            if len(data) < 52: 
                return None

        strategy_funcs = {
            'ma_crossover': check_ma_crossover_signal, 
            'rsi': check_rsi_signal,
            'supertrend': check_supertrend_signal, 
            'ha_pattern': filter_by_ha_pattern
        }
        
        if strategy not in strategy_funcs: 
            return None
        
        is_signal, signal_data = strategy_funcs[strategy](data.copy(), params)

        if not is_signal: 
            return None

        uptrend_data = {}
        if apply_uptrend_filter:
            is_uptrend, uptrend_data = check_uptrend_filter(data.copy(), strategy)
            if not is_uptrend: 
                return None
        
        result = {'Ticker': ticker, 'Close': f"${data.iloc[-1]['Close']:.2f}"}
        if signal_data: 
            result.update(signal_data)
        if uptrend_data: 
            result.update(uptrend_data)
        return result
    except Exception as e:
        print(f"  -> Error processing {ticker}: {e}")
        return None

# --- Main API Endpoint ---
@app.route('/run-screener', methods=['POST'])
def handle_screener_request():
    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({"error": "No JSON data received"}), 400
            
        params = req_data.get('params', {})
        index = req_data.get('index', 'S&P 500')
        strategy = req_data.get('strategy', 'ma_crossover')
        timeframe = req_data.get('timeframe', 'daily')
        apply_uptrend_filter = req_data.get('applyUptrendFilter', False)
        
        print(f"Request: {index}, {strategy}, Timeframe: {timeframe}, Uptrend Filter: {apply_uptrend_filter}")
        
        tickers = get_tickers(index)
        if not tickers: 
            return jsonify({"error": f"Could not fetch tickers for {index}."}), 500

        print(f"Downloading data for {len(tickers)} tickers...")
        all_data = yf.download(
            tickers, 
            period="2y", 
            auto_adjust=True, 
            session=session, 
            progress=False, 
            group_by='ticker',
            threads=True
        )
        
        matching_stocks = []
        ticker_data_map = {}
        
        # Process ticker data
        for ticker in tickers:
            if isinstance(all_data, pd.DataFrame) and len(all_data.columns) > 0:
                # Single ticker case
                if len(tickers) == 1:
                    ticker_data_map[ticker] = all_data
                else:
                    # Multi-ticker case
                    if ticker in all_data.columns.get_level_values(0):
                        ticker_data_map[ticker] = all_data[ticker]
            else:
                break

        print(f"Successfully downloaded data for {len(ticker_data_map)} tickers")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for ticker, df in ticker_data_map.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    future = executor.submit(
                        process_ticker_data, 
                        ticker, df, strategy, params, timeframe, apply_uptrend_filter
                    )
                    futures.append(future)
            
            for future in futures:
                result = future.result(timeout=30)  # 30 second timeout
                if result: 
                    matching_stocks.append(result)

        total_in_index, failed = len(tickers), list(set(tickers) - set(ticker_data_map.keys()))
        print(f"Scan complete. Found {len(matching_stocks)} matching stocks.")
        
        return jsonify({
            "matching_stocks": matching_stocks, 
            "total_scanned": len(ticker_data_map), 
            "total_in_index": total_in_index, 
            "failed_tickers": failed
        })
        
    except Exception as e:
        print(f"[SERVER ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)