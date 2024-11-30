from matplotlib import pyplot as plt
from data_fetcher import fetch_crypto_data
from analysis import calculate_squeeze_momentum, calculate_smi, normalize_smi
import strategy
import config
from visualization import plot_combined_squeeze_vwap_subplots, plot_squeeze_momentum, plot_vwaipt_deviation_grid

def process_timeframe_data(timeframe):
    """
    Fetch and process data for a given timeframe, calculating SMI and VWAIPT values.
    
    Parameters:
    - timeframe: String representing the timeframe to process.

    Returns:
    - smi_df: Processed DataFrame containing Squeeze Momentum data.
    - vwaipt_values: Series containing VWAIPT values.
    """
    print(f"Fetching data for {timeframe} timeframe...")
    # Fetch cryptocurrency data for the specified timeframe
    df = fetch_crypto_data(config.CRYPTO_SYMBOL, timeframe, config.DATA_LIMIT)

    print(f"Calculating Squeeze Momentum for {timeframe} timeframe...")
    # Calculate the Squeeze Momentum Indicator and normalize the data
    smi_df = calculate_squeeze_momentum(df)
    df = calculate_smi(smi_df)
    smi_df = normalize_smi(df)

    # Calculate VWAIPT values based on the processed SMI data
    vwaipt_values = strategy.calculate_vwap(smi_df)
    
    return smi_df, vwaipt_values

def main():
    # Initialize dictionaries to hold SMI data and VWAIPT values
    timeframe_data = {}
    vwaipt_values_dict = {}

    # Iterate over each timeframe defined in the configuration
    for tf in config.timeframes_config.keys():
        smi_df, vwaipt_values = process_timeframe_data(tf)
        
        # Store the processed data for each timeframe
        timeframe_data[tf] = smi_df
        vwaipt_values_dict[tf] = vwaipt_values

    # Decide whether to display all plots in a single subplot window or individually
    subplot_mode = True  # Change to False to plot each timeframe individually
    # Display the Squeeze Momentum Indicator plots
    print("Displaying Squeeze Momentum Indicator plots...")
    #plot_squeeze_momentum(timeframe_data, subplot_mode=subplot_mode)
    
    # Evaluate the trading strategy based on the SMI data and calculate scores for each timeframe
    print("Evaluating the trading strategy...")
    strategy_scores = strategy.evaluate_optimized_strategy(timeframe_data)
    
    # Plot the combined strategy visualization including strategy scores and intuitive momentum changes
    print("Displaying combined strategy visualization...")
    #strategy.plot_combined_strategy_visualization(strategy_scores, timeframe_data)
    
    # Calculate the weighted average score of the strategy across all timeframes
    weighted_average_score = strategy.calculate_weighted_average_score(strategy_scores, config.timeframes_config)
    # Generate and print a summary of the strategy evaluation
    print("Generating strategy evaluation summary...")
    strategy.generate_strategy_summary(strategy_scores, weighted_average_score, config.timeframes_config)

    # Plot VWAIPT deviations using subplots across all timeframes
    print("Plotting VWAIPT deviation grid...")
    #plot_vwaipt_deviation_grid(timeframe_data, vwaipt_values_dict)

    # Display combined Squeeze Momentum and VWAIPT deviation across all timeframes as subplots
    print("Displaying combined Squeeze Momentum and VWAIPT deviation across all timeframes...")
    plot_combined_squeeze_vwap_subplots(timeframe_data, vwaipt_values_dict)

if __name__ == "__main__":
    main()
    
    # Keep the script running to allow viewing the plots
    input("Press Enter to exit and close all plots.")
