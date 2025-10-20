from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import yfinance as yf
import requests
import pandas_ta as ta
import io
import time
from typing import Dict, Any, Tuple, Optional, List

# Initialize Flask App
app = Flask(__name__, static_folder='static')
CORS(app)

# --- Route to serve the frontend ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# --- Helper & Data Logic ---
def get_tickers(index_name: str = "S&P 500") -> List[str]:
    wiki_pages = {
        "S&P 500": {'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'possible_cols': ['Symbol']},
        "S&P 100": {'url': 'https://en.wikipedia.org/wiki/S%26P_100', 'possible_cols': ['Symbol', 'Ticker symbol']}
    }
    if index_name not in wiki_pages: return []
    try:
        page_info = wiki_pages[index_name]
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(page_info['url'], headers=headers)
        response.raise_for_status()
        table, found_col = None, None
        for col_name in page_info['possible_cols']:
            try:
                tables = pd.read_html(io.StringIO(response.text), match=col_name, flavor='lxml')
                if tables:
                    table, found_col = tables[0], col_name
                    break
            except ValueError: continue
        if table is None: raise ValueError(f"Could not find a valid ticker table for {index_name}.")
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.get_level_values(0)
        return table[found_col].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        print(f"Error fetching {index_name} tickers: {e}")
        return []

# --- Individual Strategy Checkers ---
def check_ma_crossover_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        short_window = params.get('short_window', 50)
        long_window = params.get('long_window', 200)
        
        # Calculate SMAs
        df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
        
        # Remove NaN values
        df_clean = df.dropna(subset=['SMA_short', 'SMA_long'])
        
        if len(df_clean) < 2:
            return False, None
            
        # Check for crossover
        current_row = df_clean.iloc[-1]
        previous_row = df_clean.iloc[-2]
        
        # Golden cross: short MA crosses above long MA
        if (previous_row['SMA_short'] <= previous_row['SMA_long'] and 
            current_row['SMA_short'] > current_row['SMA_long']):
            return True, {
                f"SMA{short_window}": f"{current_row['SMA_short']:.2f}",
                f"SMA{long_window}": f"{current_row['SMA_long']:.2f}",
                "Signal": "Golden Cross"
            }
            
        return False, None
        
    except Exception as e:
        print(f"Error in MA crossover check: {e}")
        return False, None

def check_rsi_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        length = params.get('rsi_length', 14)
        threshold = params.get('rsi_threshold', 30)
        
        # Calculate RSI
        df['RSI'] = ta.rsi(df['Close'], length=length)
        df_clean = df.dropna(subset=['RSI'])
        
        if df_clean.empty:
            return False, None
            
        current_rsi = df_clean.iloc[-1]['RSI']
        
        # Check if RSI is below oversold threshold
        if current_rsi < threshold:
            return True, {
                "RSI": f"{current_rsi:.2f}",
                "Signal": "Oversold"
            }
            
        return False, None
        
    except Exception as e:
        print(f"Error in RSI check: {e}")
        return False, None

def check_supertrend_signal(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        length = params.get('supertrend_length', 7)
        multiplier = params.get('supertrend_multiplier', 3.0)
        
        # Calculate supertrend
        supertrend_df = ta.supertrend(df['High'], df['Low'], df['Close'], length=length, multiplier=multiplier)
        
        # Merge with original dataframe
        df = pd.concat([df, supertrend_df], axis=1)
        
        # Get the supertrend column names
        supertrend_col = f'SUPERT_{length}_{multiplier}'
        supertrend_dir_col = f'SUPERTd_{length}_{multiplier}'
        
        df_clean = df.dropna(subset=[supertrend_col, supertrend_dir_col])
        
        if df_clean.empty:
            return False, None
            
        current_row = df_clean.iloc[-1]
        current_close = current_row['Close']
        current_supertrend = current_row[supertrend_col]
        current_direction = current_row[supertrend_dir_col]
        
        # Check if price is above supertrend AND supertrend direction is up
        if current_close > current_supertrend and current_direction == 1:
            return True, {
                "Supertrend": f"{current_supertrend:.2f}",
                "Close": f"{current_close:.2f}",
                "Position": "Above",
                "Direction": "Up",
                "Distance %": f"{(current_close - current_supertrend) / current_supertrend * 100:.2f}%"
            }
            
        return False, None
        
    except Exception as e:
        print(f"Error in supertrend check: {e}")
        return False, None

def filter_by_ha_pattern(df: pd.DataFrame, params: Dict) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        pattern = params.get('pattern', 'RRGG')
        
        # Calculate Heikin-Ashi
        ha_df = ta.ha(df['Open'], df['High'], df['Low'], df['Close'])
        df = pd.concat([df, ha_df], axis=1)
        
        df_clean = df.dropna(subset=['HA_close', 'HA_open'])
        
        if len(df_clean) < len(pattern):
            return False, None
            
        # Check the pattern
        latest_candles = df_clean.iloc[-len(pattern):]
        
        for i in range(len(pattern)):
            candle = latest_candles.iloc[i]
            actual_color = 'G' if candle['HA_close'] > candle['HA_open'] else 'R'
            expected_color = pattern[i].upper()
            
            if actual_color != expected_color:
                return False, None
                
        return True, {
            "Pattern": pattern,
            "Signal": f"Heikin-Ashi {pattern} pattern detected"
        }
        
    except Exception as e:
        print(f"Error in HA pattern check: {e}")
        return False, None

def check_uptrend_filter(df: pd.DataFrame, strategy_name: str, filter_params: Dict) -> Tuple[bool, Dict[str, Any]]:
    try:
        # Get filter parameters with defaults
        ema_short_period = filter_params.get('ema_short_period', 20)
        ema_long_period = filter_params.get('ema_long_period', 50)
        rsi_period = filter_params.get('rsi_period', 14)
        volume_period = filter_params.get('volume_period', 20)
        
        # Calculate technical indicators for trend filtering
        df['EMA_short'] = ta.ema(df['Close'], length=ema_short_period)
        df['EMA_long'] = ta.ema(df['Close'], length=ema_long_period)
        df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=volume_period).mean()
        df['Price_vs_High'] = df['Close'] / df['High'].rolling(window=volume_period).max()
        
        df_clean = df.dropna()
        
        if df_clean.empty:
            return False, {"Error": "Insufficient data for trend filter"}
            
        last = df_clean.iloc[-1]
        price = last['Close']
        ema_short = last['EMA_short']
        ema_long = last['EMA_long']
        rsi = last['RSI']
        volume_ratio = last['Volume_Ratio']
        price_vs_high = last['Price_vs_High']
        
        # Define conditions based on strategy with user-defined parameters
        if strategy_name == 'rsi':
            price_vs_ema20_threshold = filter_params.get('price_vs_ema20', 98) / 100
            price_vs_ema50_threshold = filter_params.get('price_vs_ema50', 97) / 100
            rsi_min = filter_params.get('rsi_min', 30)
            rsi_max = filter_params.get('rsi_max', 75)
            volume_ratio_min = filter_params.get('volume_ratio_min', 0.7)
            required_passes = filter_params.get('required_conditions', 3)
            
            conditions = {
                'price_above_ema20': price > ema_short * price_vs_ema20_threshold,
                'price_above_ema50': price > ema_long * price_vs_ema50_threshold,
                'rsi_reasonable': rsi_min <= rsi <= rsi_max,
                'volume_adequate': volume_ratio > volume_ratio_min
            }
            
        elif strategy_name == 'ma_crossover':
            price_vs_ema20_threshold = filter_params.get('price_vs_ema20', 100) / 100
            price_vs_ema50_threshold = filter_params.get('price_vs_ema50', 100) / 100
            rsi_min = filter_params.get('rsi_min', 35)
            rsi_max = filter_params.get('rsi_max', 80)
            volume_ratio_min = filter_params.get('volume_ratio_min', 0.6)
            price_vs_high_threshold = filter_params.get('price_vs_high', 85) / 100
            required_passes = filter_params.get('required_conditions', 3)
            
            conditions = {
                'price_above_ema20': price > ema_short * price_vs_ema20_threshold,
                'price_above_ema50': price > ema_long * price_vs_ema50_threshold,
                'rsi_reasonable': rsi_min <= rsi <= rsi_max,
                'volume_adequate': volume_ratio > volume_ratio_min,
                'near_highs': price_vs_high > price_vs_high_threshold
            }
            
        elif strategy_name in ['ha_pattern', 'supertrend']:
            price_vs_ema50_threshold = filter_params.get('price_vs_ema50', 95) / 100
            rsi_min = filter_params.get('rsi_min', 25)
            volume_ratio_min = filter_params.get('volume_ratio_min', 0.5)
            required_passes = filter_params.get('required_conditions', 2)
            
            conditions = {
                'price_above_ema50': price > ema_long * price_vs_ema50_threshold,
                'rsi_not_extreme': rsi > rsi_min,
                'volume_adequate': volume_ratio > volume_ratio_min
            }
            
        else:
            price_vs_ema20_threshold = filter_params.get('price_vs_ema20', 98) / 100
            price_vs_ema50_threshold = filter_params.get('price_vs_ema50', 100) / 100
            rsi_min = filter_params.get('rsi_min', 30)
            rsi_max = filter_params.get('rsi_max', 80)
            volume_ratio_min = filter_params.get('volume_ratio_min', 0.6)
            required_passes = filter_params.get('required_conditions', 3)
            
            conditions = {
                'price_above_ema20': price > ema_short * price_vs_ema20_threshold,
                'price_above_ema50': price > ema_long * price_vs_ema50_threshold,
                'rsi_reasonable': rsi_min <= rsi <= rsi_max,
                'volume_adequate': volume_ratio > volume_ratio_min
            }
            
        pass_count = sum(conditions.values())
        
        if pass_count >= required_passes:
            return True, {
                "Trend Strength": "Strong" if pass_count == len(conditions) else "Moderate",
                "Conditions Passed": f"{pass_count}/{len(conditions)}"
            }
        return False, {
            "Trend Strength": "Weak",
            "Conditions Passed": f"{pass_count}/{len(conditions)}"
        }
        
    except Exception as e:
        print(f"Error in uptrend filter: {e}")
        return False, {"Error": str(e)}

def process_ticker_data(ticker: str, data: pd.DataFrame, strategy: str, params: Dict, timeframe: str, apply_uptrend_filter: bool, filter_params: Dict) -> Optional[Dict[str, Any]]:
    try:
        if data.empty or len(data) < 50:
            return None
            
        # Handle timeframe resampling
        if timeframe == 'weekly':
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            weekly_data = data.resample('W').agg({
                'Open': 'first', 
                'High': 'max', 
                'Low': 'min', 
                'Close': 'last', 
                'Volume': 'sum'
            }).dropna()
            if len(weekly_data) < 20:
                return None
            data = weekly_data
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return None
            
        strategy_funcs = {
            'ma_crossover': check_ma_crossover_signal,
            'rsi': check_rsi_signal,
            'supertrend': check_supertrend_signal,
            'ha_pattern': filter_by_ha_pattern
        }
        
        if strategy not in strategy_funcs:
            return None
            
        # Apply strategy
        is_signal, signal_data = strategy_funcs[strategy](data.copy(), params)
        
        if not is_signal:
            return None
            
        # Apply uptrend filter if enabled
        uptrend_data = {}
        if apply_uptrend_filter:
            is_uptrend, uptrend_data = check_uptrend_filter(data.copy(), strategy, filter_params)
            if not is_uptrend:
                return None
        
        # Build result
        result = {
            'Ticker': ticker, 
            'Close': f"${data.iloc[-1]['Close']:.2f}",
            'Strategy': strategy.replace('_', ' ').title()
        }
        
        if signal_data:
            result.update(signal_data)
        if uptrend_data:
            result.update(uptrend_data)
            
        return result
        
    except Exception as e:
        print(f"  -> Error processing {ticker} ({strategy}): {e}")
        return None

# --- Main API Endpoint ---
@app.route('/run-screener', methods=['POST'])
def handle_screener_request():
    try:
        req_data = request.get_json()
        params = req_data.get('params', {})
        filter_params = req_data.get('filterParams', {})
        apply_uptrend_filter = req_data.get('applyUptrendFilter', False)
        
        print(f"Request: {req_data.get('index')}, {req_data.get('strategy')}, Params: {params}, Uptrend: {apply_uptrend_filter}, FilterParams: {filter_params}")
        
        tickers = get_tickers(req_data.get('index'))
        if not tickers:
            return jsonify({"error": f"Could not fetch tickers for {req_data.get('index')}."}), 500

        matching_stocks, failed_tickers = [], []
        
        # Process all tickers
        print(f"Processing all {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers):
            print(f"Processing {ticker} ({i+1}/{len(tickers)})...")
            try:
                # Use yf.Ticker for more reliable data
                stock = yf.Ticker(ticker)
                data = stock.history(period="1y", auto_adjust=True)
                
                if data.empty:
                    failed_tickers.append(ticker)
                    continue

                result = process_ticker_data(
                    ticker, data, 
                    req_data.get('strategy'), 
                    params, 
                    req_data.get('timeframe'), 
                    apply_uptrend_filter,
                    filter_params
                )
                if result:
                    matching_stocks.append(result)

                # Small delay to be gentle on APIs
                time.sleep(0.05)
                
            except Exception as e:
                print(f"  -> Failed to download or process {ticker}: {e}")
                failed_tickers.append(ticker)

        print(f"Scan complete. Found {len(matching_stocks)} matching stocks.")
        return jsonify({
            "matching_stocks": matching_stocks,
            "total_scanned": len(tickers) - len(failed_tickers),
            "total_in_index": len(tickers),
            "failed_tickers": failed_tickers
        })

    except Exception as e:
        print(f"[SERVER ERROR] An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error. Please try again later."}), 500

if __name__ == '__main__':
    app.run(debug=True)