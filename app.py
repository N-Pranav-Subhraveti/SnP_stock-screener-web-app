from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import yfinance as yf
import requests
import pandas_ta as ta
import io

# Initialize Flask App
app = Flask(__name__)
# Enable CORS for cross-origin requests
CORS(app)

# --- Screener & Data Logic ---

def get_tickers(index_name="S&P 500"):
    """Fetches tickers for a specified index from Wikipedia."""
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

def add_heikin_ashi(df):
    """Calculates and appends Heikin-Ashi candles to the DataFrame."""
    df.columns = [col.lower() for col in df.columns]
    df.ta.ha(append=True)
    df.rename(columns={"ha_open": "HA_open", "ha_high": "HA_high", "ha_low": "HA_low", "ha_close": "HA_close"}, inplace=True)
    return df

def filter_by_pattern(df, pattern):
    """Checks if the latest candles match the specified red/green pattern."""
    ha_open_col, ha_close_col, num_candles = 'HA_open', 'HA_close', len(pattern)
    if len(df) < num_candles or ha_open_col not in df.columns: return False
    latest_candles = df.iloc[-num_candles:]
    for i in range(num_candles):
        candle = latest_candles.iloc[i]
        if pd.isna(candle[ha_close_col]) or pd.isna(candle[ha_open_col]): return False
        actual_color = 'G' if candle[ha_close_col] > candle[ha_open_col] else 'R'
        if actual_color != pattern[i]: return False
    return True

def run_screener_logic(index_to_scan, pattern_str, timeframe):
    """Core screener logic."""
    pattern, tickers = list(pattern_str.upper()), get_tickers(index_to_scan)
    if not tickers: return {"error": "Could not fetch tickers."}
    
    matching_tickers, failed_tickers, total_scanned = [], [], 0
    print(f"\n--- Starting Scan (Timeframe: {timeframe}) ---")
    for ticker in tickers:
        try:
            print(f"Scanning {ticker}...")
            stock = yf.Ticker(ticker)
            daily_data = stock.history(period="6mo", auto_adjust=True)
            if daily_data.empty: continue
            
            data_to_process = daily_data.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna() if timeframe == 'weekly' else daily_data.copy()
            if len(data_to_process) < len(pattern): 
                total_scanned += 1
                continue

            data_with_ha = add_heikin_ashi(data_to_process)
            if filter_by_pattern(data_with_ha, pattern):
                matching_tickers.append(ticker)
            total_scanned += 1
        except Exception as e:
            print(f"  -> Failed to process data for {ticker}: {e}")
            failed_tickers.append(ticker)
    print("--- Scan Complete ---\n")
    return {"matching_tickers": matching_tickers, "total_scanned": total_scanned, "total_in_index": len(tickers), "failed_tickers": failed_tickers}

# --- API Endpoint ---

@app.route('/run-screener', methods=['POST'])
def handle_screener_request():
    """API endpoint to run the screener."""
    data = request.get_json()
    index, pattern, timeframe = data.get('index'), data.get('pattern'), data.get('timeframe')
    if not all([index, pattern, timeframe]):
        return jsonify({"error": "Missing parameters."}), 400
    print(f"Received request: Index={index}, Pattern={pattern}, Timeframe={timeframe}")
    results = run_screener_logic(index, pattern, timeframe)
    print(f"Found {len(results.get('matching_tickers', []))} matching tickers.")
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

