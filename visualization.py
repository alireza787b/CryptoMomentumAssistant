import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Ensure pandas is imported if you need to manipulate or inspect dataframe within the plotting functions
import math
from config import timeframes_config


def plot_squeeze_momentum(timeframe_data, subplot_mode=False, num_periods=50):
    """
    Plot the Squeeze Momentum Indicator for specified timeframes.
    Can plot each timeframe individually or all in subplots based on the subplot_mode flag.
    Assumes timeframe_data is either a single DataFrame (when subplot_mode=False) 
    and tf is specified, or a dictionary with timeframes as keys and DataFrames as values 
    (when subplot_mode=True).
    
    Parameters:
    - timeframe_data: DataFrame or dict of DataFrames
    - subplot_mode: bool, default False. If True, plots all timeframes in a subplot grid.
    - num_periods: int, default 50. Number of periods to display from the end of the DataFrame.
    """
    if subplot_mode:
        num_timeframes = len(timeframe_data)
        grid_size = math.ceil(np.sqrt(num_timeframes))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 10))
        fig.suptitle('Squeeze Momentum Indicator Across Timeframes')
        
        axs = axs.flatten()
        for i, (tf, df) in enumerate(timeframe_data.items()):
            if 'val' in df.columns:
                df_last_N = df.tail(num_periods)
                colors = df_last_N['val'].apply(lambda x: 'green' if x > 0 else 'red')
                axs[i].bar(df_last_N.index, df_last_N['val'], color=colors)
                axs[i].set_title(f'{tf} timeframe')
            else:
                print(f"No 'val' column found in dataframe for {tf} timeframe.")
                axs[i].set_visible(False)
        
        # Hide unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].set_visible(False)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)
    else:
        for tf, df in timeframe_data.items():
            plt.figure(figsize=(10, 6))
            if 'val' in df.columns:
                df_last_N = df.tail(num_periods)
                colors = df_last_N['val'].apply(lambda x: 'green' if x > 0 else 'red')
                plt.bar(df_last_N.index, df_last_N['val'], color=colors)
                plt.title(f'Squeeze Momentum Indicator for {tf} timeframe')
            else:
                print(f"No 'val' column found in dataframe for {tf} timeframe.")
            plt.show(block=False)




def plot_momentum_with_direction_changes(timeframe_data):
    """
    Plot momentum with directional changes for each timeframe.
    Assumes timeframe_data is a dictionary with timeframes as keys and dataframes as values.
    """
    plt.figure(figsize=(14, 8))
    markers = {'up': '^', 'down': 'v'}
    
    for tf, df in timeframe_data.items():
        if not df.empty and 'val' in df.columns:
            current_momentum = df['val'].iloc[-1]
            previous_momentum = df['val'].iloc[-2] if len(df) > 1 else 0
            
            # Determine marker type based on momentum change
            marker = markers['up'] if current_momentum > previous_momentum else markers['down']
            color = 'green' if current_momentum > 0 else 'red'
            
            plt.plot(df.index[-2:], df['val'].iloc[-2:], marker=marker, color=color, label=f"{tf} momentum change")

    plt.legend()
    plt.show()

def plot_strategy_scores(strategy_scores):
    """
    Plot strategy evaluation scores across different timeframes.
    Assumes strategy_scores is a dictionary with timeframes as keys and scores as values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    timeframes = list(strategy_scores.keys())
    scores = list(strategy_scores.values())
    
    colors = ['red' if score < 0 else 'green' for score in scores]
    ax.bar(timeframes, scores, color=colors)
    ax.axhline(0, color='grey', linestyle='--')
    ax.set_ylabel('Strategy Score (-100 to +100)')
    ax.set_title('Strategy Evaluation Scores Across Timeframes')
    plt.show(block=False)

def plot_intuitive_momentum_changes(timeframe_data):
    """
    Plot intuitive momentum changes across different timeframes.
    Assumes timeframe_data is a dictionary with timeframes as keys and dataframes as values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    timeframes = list(timeframes_config.keys())


    current_momentum = [df['val'].iloc[-1] if 'val' in df.columns else 0 for df in timeframe_data.values()]
    ypos = np.arange(len(timeframes))
    
    colors = ['green' if val > 0 else 'red' for val in current_momentum]
    ax.barh(ypos, current_momentum, color=colors, tick_label=timeframes)
    
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_xlabel('Momentum Value')
    ax.set_title('Momentum and Direction Change Across Timeframes')
    
    plt.tight_layout()
    plt.show(block=False)


def plot_vwaipt_deviation_grid(timeframe_data, vwaipt_values_dict, num_periods=50):
    """
    Plot the deviation from VWAIPT for specified timeframes using subplots.
    
    Parameters:
    - timeframe_data: dict of DataFrames containing price information.
    - vwaipt_values_dict: dict of Series/arrays with VWAIPT values corresponding to each timeframe.
    - num_periods: int, default 50. Number of periods to display from the end of the DataFrame.
    """
    num_timeframes = len(timeframe_data)
    grid_size = math.ceil(np.sqrt(num_timeframes))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 10))
    fig.suptitle('VWAIPT Deviation Across Timeframes')
    
    axs = axs.flatten()
    for i, (tf, df) in enumerate(timeframe_data.items()):
        if tf in vwaipt_values_dict:
            # Fetch the last N periods and corresponding VWAIPT values
            df_last_N = df.tail(num_periods)
            vwaipt_values = vwaipt_values_dict[tf].tail(num_periods)

            # Calculate the percentage deviation and the direction changes
            vwaipt_percentage = 100 * (df_last_N['close'] - vwaipt_values) / vwaipt_values
            colors = np.where(vwaipt_percentage > 0, 'green', 'red')
            previous_percentage = vwaipt_percentage.shift(1).fillna(0)
            markers = np.where(vwaipt_percentage > previous_percentage, '^', 'v')

            # Plot the VWAIPT deviation with directional markers
            for time, percentage, color, marker in zip(df_last_N.index, vwaipt_percentage, colors, markers):
                axs[i].scatter(time, percentage, color=color, marker=marker)
            
            axs[i].axhline(0, color='grey', linestyle='--')
            axs[i].set_title(f'{tf} timeframe')
            axs[i].set_ylabel('Deviation (%)')
        else:
            print(f"No VWAIPT values found for {tf} timeframe.")
            axs[i].set_visible(False)
    
    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)
    
    
def normalize_series(series):
    """
    Normalize a pandas Series to the range [-1, +1].
    
    Parameters:
    - series: The Series to normalize.
    
    Returns:
    - Normalized Series.
    """
    return 2 * ((series - series.min()) / (series.max() - series.min())) - 1

def get_deviation_color(value):
    """
    Determine the color intensity for VWAIPT deviation markers using numerical RGB values.

    Parameters:
    - value: Deviation value to color code.

    Returns:
    - A list representing RGB values.
    """
    if value > 0:
        intensity = min(1, value)  # Cap at 1 to prevent extreme intensity
        return [0, min(1, intensity), 0]  # More green intensity
    else:
        intensity = min(1, abs(value))  # Cap at 1 for negative values
        return [min(1, intensity), 0, 0]  # More red intensity

def plot_combined_squeeze_vwap_subplots(timeframe_data, vwaipt_values_dict, num_periods=50):
    """
    Plot a combined visualization of normalized Squeeze Momentum and VWAIPT deviations across all timeframes as subplots.

    Parameters:
    - timeframe_data: Dictionary of DataFrames for each timeframe.
    - vwaipt_values_dict: Dictionary of Series containing VWAP values for each timeframe.
    - num_periods: int, default 50. Number of periods to display from the end of each DataFrame.
    """
    num_timeframes = len(timeframe_data)
    grid_size = math.ceil(np.sqrt(num_timeframes))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle('Combined Squeeze Momentum and VWAP Deviation Across Timeframes')

    axs = axs.flatten()
    for i, tf in enumerate(timeframe_data.keys()):
        df = timeframe_data[tf]
        vwaipt_values = vwaipt_values_dict[tf]
        
        # Fetch the last N periods of data
        df_last_N = df.tail(num_periods)
        vwaipt_last_N = vwaipt_values.tail(num_periods)

        # Normalize both Squeeze Momentum and VWAIPT deviations to [-1, +1]
        normalized_squeeze_momentum = normalize_series(df_last_N['val'])
        vwaipt_percentage = 100 * (df_last_N['close'] - vwaipt_last_N) / vwaipt_last_N
        normalized_vwaipt_deviation = normalize_series(vwaipt_percentage)

        # Set bar colors for Squeeze Momentum based on positive or negative values
        bar_colors = normalized_squeeze_momentum.apply(lambda x: 'green' if x > 0 else 'red')

        # Set numerical RGB values for VWAIPT deviation markers
        marker_colors = [get_deviation_color(value) for value in normalized_vwaipt_deviation]

        # Plot the normalized Squeeze Momentum as bars
        axs[i].bar(df_last_N.index, normalized_squeeze_momentum, color=bar_colors, label='Squeeze Momentum')

        # Plot the normalized VWAIPT deviation as lines with smaller dots
        axs[i].plot(df_last_N.index, normalized_vwaipt_deviation, '-o', c='darkorange', markersize=6, linewidth=1.5, label='VWAIPT Deviation %')
        
        axs[i].set_title(f'{tf} Timeframe')
        axs[i].axhline(0, color='grey', linestyle='--')
        axs[i].legend()

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)