import matplotlib.pyplot as plt
import numpy as np
from config import timeframes_config

def evaluate_optimized_strategy(timeframe_data):
    strategy_scores = {}
    
    for tf, df in timeframe_data.items():
        if len(df) < 4:
            print(f"Not enough data for timeframe {tf} to evaluate strategy with optimized confirmation.")
            strategy_scores[tf] = None
            continue

        current_momentum = df['val'].iloc[-1]
        previous_momentum = df['val'].iloc[-2]
        pre_previous_momentum = df['val'].iloc[-3]
        pre_pre_previous_momentum = df['val'].iloc[-4]

        initial_momentum_change = pre_previous_momentum - pre_pre_previous_momentum
        confirmation_momentum_change = current_momentum - previous_momentum
        confirmed_reversal_signal = (confirmation_momentum_change * initial_momentum_change < 0) and (abs(confirmation_momentum_change) > abs(initial_momentum_change))
        
        max_val_abs = max(abs(df['val'].max()), abs(df['val'].min()))
        if max_val_abs == 0:
            max_val_abs = 1

        magnitude_of_change = abs(confirmation_momentum_change)
        score_factor = np.sign(confirmation_momentum_change)
        
        if confirmed_reversal_signal:
            score = 100 * (magnitude_of_change / max_val_abs) * score_factor
        else:
            score = 50 * (magnitude_of_change / max_val_abs) * score_factor if initial_momentum_change * confirmation_momentum_change > 0 else 0
        
        strategy_scores[tf] = score

    return strategy_scores


def calculate_strategy_score(df, timeframe):
    """
    Calculate the strategy score (sqzScore) for the given DataFrame and timeframe.
    """
    # Mock function to represent the structure of evaluate_optimized_strategy
    # Assumes evaluate_optimized_strategy takes a dictionary of dataframes keyed by timeframe
    # and returns a dictionary of scores keyed by the same timeframes.
    # This wrapper is to fit that into our existing per-timeframe calculation structure.
    timeframe_data = {timeframe: df}
    strategy_scores = evaluate_optimized_strategy(timeframe_data)
    return strategy_scores.get(timeframe, 0)


def generate_strategy_summary(strategy_scores, weighted_average_score, timeframe_weights):
    bullish_timeframes = [tf for tf, score in strategy_scores.items() if score and score > 0]
    bearish_timeframes = [tf for tf, score in strategy_scores.items() if score and score < 0]

    if weighted_average_score > 0:
        overall_sentiment = "bullish"
    elif weighted_average_score < 0:
        overall_sentiment = "bearish"
    else:
        overall_sentiment = "neutral"

    strongest_bullish_signal = max(bullish_timeframes, key=lambda tf: strategy_scores[tf], default="None")
    strongest_bearish_signal = max(bearish_timeframes, key=lambda tf: strategy_scores[tf], default="None")
    
    recommendation = "Market sentiment appears neutral; it may be wise to wait for clearer signals before taking a position."
    if overall_sentiment == "bullish":
        recommendation = "Considering a long position, particularly focusing on timeframes with the strongest bullish signals."
    elif overall_sentiment == "bearish":
        recommendation = "Considering a short position, particularly focusing on timeframes with the strongest bearish signals."
    
    confidence = min(abs(weighted_average_score) / 100, 1) * 100

    summary = (
        f"Overall Market Sentiment: {overall_sentiment.capitalize()}.\n"
        f"Strongest Bullish Signal: {strongest_bullish_signal}.\n"
        f"Strongest Bearish Signal: {strongest_bearish_signal}.\n"
        f"{recommendation}\n"
        f"Confidence Level: {confidence:.2f}%.\n"
    )
    
    print(summary)

def calculate_weighted_average_score(strategy_scores, timeframe_weights):
    weighted_sum = sum(strategy_scores[tf] * timeframes_config[tf]['weight'] for tf in strategy_scores if strategy_scores[tf] is not None)
    total_weight = sum(timeframes_config[tf]['weight'] for tf in strategy_scores if strategy_scores[tf] is not None)

    weighted_average_score = weighted_sum / total_weight
    return weighted_average_score

import matplotlib.pyplot as plt
import numpy as np

def plot_combined_strategy_visualization(strategy_scores, timeframe_data):
    """
    Combine strategy scores and intuitive momentum changes into a single figure with subplots.
    
    Parameters:
    - strategy_scores: Dict of scores for each timeframe.
    - timeframe_data: Dict of DataFrames with SMI values for each timeframe.
    """
    fig = plt.figure(figsize=(20, 6))
    
    # Subplot 1: Strategy Scores
    ax1 = fig.add_subplot(2, 1, 1)
    timeframes = list(strategy_scores.keys())
    scores = list(strategy_scores.values())
    colors = ['red' if score < 0 else 'green' for score in scores]
    ax1.bar(timeframes, scores, color=colors)
    ax1.axhline(0, color='grey', linestyle='--')
    ax1.set_ylabel('Strategy Score (-100 to +100)')
    ax1.set_title('Strategy Evaluation Scores Across Timeframes')

    # Subplot 2: Intuitive Momentum Changes
    ax2 = fig.add_subplot(2, 1, 2)
    timeframes = list(timeframe_data.keys())
    current_momentum = [df['val'].iloc[-1] for df in timeframe_data.values()]
    previous_momentum = [df['val'].iloc[-2] if len(df) > 1 else 0 for df in timeframe_data.values()]
    colors = ['green' if val > 0 else 'red' for val in current_momentum]
    ypos = np.arange(len(timeframes))
    ax2.barh(ypos, current_momentum, color=colors, tick_label=timeframes)
    for i, (cur, prev) in enumerate(zip(current_momentum, previous_momentum)):
        marker = '^' if cur > prev else 'v'
        marker_color = 'lime' if cur > prev else 'darkred'
        ax2.scatter(cur, i, color=marker_color, marker=marker, s=100, zorder=5)
    ax2.axvline(0, color='grey', linestyle='--')
    ax2.set_xlabel('Momentum Value')
    ax2.set_title('Momentum and Direction Change Across Timeframes')

    plt.tight_layout()
    plt.show(block=False)

def calculate_vwap(timeframe_data, price_column='close', volume_column='volumeto', window=20):
    """
    Calculate the VWAP (Volume Weighted Average Price) over a specified rolling window.

    Parameters:
    - timeframe_data: A DataFrame containing at least 'price' and 'volume'.
    - price_column: The column name for the price data (e.g., 'close').
    - volume_column: The column name for the volume data (e.g., 'volume').
    - window: The rolling window size to calculate the VWAP.

    Returns:
    - A Series representing the VWAP values.
    """
    # Multiply price by volume to get volume-weighted values
    price_volume = timeframe_data[price_column] * timeframe_data[volume_column]
    
    # Sum up the price-volume products and divide by total volume over the rolling window
    cumulative_price_volume = price_volume.rolling(window=window, min_periods=1).sum()
    cumulative_volume = timeframe_data[volume_column].rolling(window=window, min_periods=1).sum()
    
    vwap = cumulative_price_volume / cumulative_volume
    return vwap
