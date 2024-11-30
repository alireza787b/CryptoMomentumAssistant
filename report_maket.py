"""
Crypto Data Fetcher and Indicator Calculator

This script fetches cryptocurrency market data from APIs, calculates various
technical indicators, and saves the data in CSV format. The script is fully 
configurable to allow flexible workflows.

## Features
1. Configurable Parameters:
   - `symbol`: Cryptocurrency symbol (e.g., BTC, ETH).
   - `timeframes`: Choose one or multiple timeframes (e.g., 1m, 5m, 1h).
   - `data_limit`: Number of data points to fetch.
   - `indicators`: Select which technical indicators to calculate.

2. Flexible Output:
   - `output_mode`: 
       - "single_file": Combines data from all timeframes into one file.
       - "separate_files": Saves each timeframe's data into separate files.

3. Technical Indicators:
   - RSI, EMA (50 & 200), SMA (50 & 200), Bollinger Bands, MACD, ATR, VWAP.

4. Scalable and Modular:
   - Can be extended with new indicators or features.

## Usage
1. Configure the `CONFIG` dictionary at the top.
2. Run the script: `python crypto_data_fetcher.py`.
3. Outputs CSV files in the specified directory.
"""

import pandas as pd
from datetime import datetime
import os
from data_fetcher import fetch_crypto_data
from ta import momentum, trend, volatility

# Configuration Parameters
CONFIG = {
    "symbol": "BTC",                  # Cryptocurrency symbol
    "timeframes": ["15m", "1h", "4h"],  # Timeframes to fetch data for
    "data_limit": 200,                # Number of data points to fetch per timeframe
    "output_dir": "./data",           # Directory to save CSV files
    "output_mode": "single_file",     # Options: "single_file" or "separate_files"
    "indicators": [                   # List of indicators to calculate
        "RSI", "EMA_50", "EMA_200", "SMA_50", "SMA_200", "BollingerBands",
        "MACD", "ATR", "VWAP"
    ]
}

# Create output directory if it doesn't exist
os.makedirs(CONFIG["output_dir"], exist_ok=True)

def calculate_indicators(df):
    """Add technical indicators to the DataFrame."""
    if "RSI" in CONFIG["indicators"]:
        df['RSI'] = momentum.RSIIndicator(df['close'], window=14).rsi()
    if "EMA_50" in CONFIG["indicators"]:
        df['EMA_50'] = trend.EMAIndicator(df['close'], window=50).ema_indicator()
    if "EMA_200" in CONFIG["indicators"]:
        df['EMA_200'] = trend.EMAIndicator(df['close'], window=200).ema_indicator()
    if "SMA_50" in CONFIG["indicators"]:
        df['SMA_50'] = df['close'].rolling(window=50).mean()
    if "SMA_200" in CONFIG["indicators"]:
        df['SMA_200'] = df['close'].rolling(window=200).mean()
    if "BollingerBands" in CONFIG["indicators"]:
        bollinger = volatility.BollingerBands(df['close'], window=20)
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
    if "MACD" in CONFIG["indicators"]:
        macd = trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
    if "ATR" in CONFIG["indicators"]:
        df['ATR'] = volatility.AverageTrueRange(
            df['high'], df['low'], df['close']).average_true_range()
    if "VWAP" in CONFIG["indicators"]:
        df['VWAP'] = (df['close'] * df['volumeto']).cumsum() / df['volumeto'].cumsum()
    return df

def process_timeframe(symbol, timeframe, data_limit):
    """Fetch and process data for a specific timeframe."""
    print(f"Fetching data for {symbol} in {timeframe} timeframe...")
    df = fetch_crypto_data(symbol, timeframe, data_limit)
    
    print(f"Calculating indicators for {timeframe} timeframe...")
    df = calculate_indicators(df)
    
    return df

def save_to_csv(dataframes, symbol, mode):
    """Save data to CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if mode == "single_file":
        output_file = f"{CONFIG['output_dir']}/{symbol}_all_timeframes_{timestamp}.csv"
        print(f"Saving combined data to {output_file}...")
        combined_df = pd.concat(dataframes, keys=CONFIG["timeframes"], names=["Timeframe", "Index"])
        combined_df.reset_index(level=1, drop=True, inplace=True)
        combined_df.to_csv(output_file, index=False)
    
    elif mode == "separate_files":
        for tf, df in zip(CONFIG["timeframes"], dataframes):
            output_file = f"{CONFIG['output_dir']}/{symbol}_{tf}_{timestamp}.csv"
            print(f"Saving {tf} data to {output_file}...")
            df.to_csv(output_file, index=False)
    else:
        raise ValueError("Invalid output_mode in CONFIG. Use 'single_file' or 'separate_files'.")

def main():
    """Main workflow to fetch, process, and save cryptocurrency data."""
    all_data = []
    
    for timeframe in CONFIG["timeframes"]:
        df = process_timeframe(CONFIG["symbol"], timeframe, CONFIG["data_limit"])
        all_data.append(df)
    
    save_to_csv(all_data, CONFIG["symbol"], CONFIG["output_mode"])
    print("Data processing complete.")

if __name__ == "__main__":
    main()
