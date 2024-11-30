import os
import pandas as pd
from data_fetcher import fetch_crypto_data
from config import output_timeframes_config, output_dir, desired_fields, timeframe_offsets
from analysis import calculate_smi, calculate_rsi, normalize_smi

def compile_and_calculate_metrics(crypto):
    """
    Fetches data for specified crypto across different timeframes,
    calculates various metrics including RSI and Squeeze Momentum indicators,
    and saves individual timeframe data with these metrics.
    """
    compiled_data = {}
    oldest_times = []

    # Directory setup for individual CSVs
    individual_output_dir = os.path.join(output_dir, 'individual_timeframes')
    os.makedirs(individual_output_dir, exist_ok=True)

    # Find the smallest timeframe for resampling in merging
    smallest_timeframe = min(output_timeframes_config, key=lambda k: output_timeframes_config[k]['weight'])

    for timeframe in output_timeframes_config:
        try:
            print(f"Processing {timeframe} data...")
            df = fetch_crypto_data(crypto, timeframe, 2000)
            if not df.empty:
                oldest_times.append(df['time'].min())
                df = calculate_indicators(df)  # Calculate all indicators
                
                # Save the processed dataframe for this timeframe
                df.to_csv(os.path.join(individual_output_dir, f"{crypto}_{timeframe}.csv"), index=False)
                compiled_data[timeframe] = df

        except Exception as e:
            print(f"Failed for {timeframe}: {e}")

    common_start_time = max(oldest_times)
    return compiled_data, smallest_timeframe, common_start_time

def calculate_indicators(df):
    """
    Calculate RSI, Squeeze Momentum Indicator, and other related metrics.
    """
    df['rsi'] = calculate_rsi(df)
    calculate_smi(df)  # Calculates SMI components and adds them to df
    normalize_smi(df)  # Adds normalized SMI to df
    return df

def align_and_merge_data(crypto, compiled_data, smallest_timeframe, common_start_time):
    """
    Merges processed dataframes from different timeframes into a single dataframe,
    ensuring consistent intervals and including all calculated metrics.
    """
    dfs = []

    for timeframe, df in compiled_data.items():
        # Prepare dataframe for merging
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df.resample(f'{timeframe_offsets[smallest_timeframe]}').ffill().bfill()
        dfs.append(df.add_suffix(f'_{timeframe}'))

    # Merge and fill gaps
    merged_df = pd.concat(dfs, axis=1).ffill().bfill()

    # Save merged data
    merged_filename = os.path.join(output_dir, f"{crypto}_merged.csv")
    merged_df.to_csv(merged_filename)
    print(f"Merged data saved to {merged_filename}")

    return merged_df



# Main execution
if __name__ == "__main__":
    crypto = "BTC"  # Example, replace with your target cryptocurrency symbol
    compiled_data, smallest_timeframe, common_start_time = compile_and_calculate_metrics(crypto)
    merged_data = align_and_merge_data(crypto, compiled_data, smallest_timeframe, common_start_time)
