# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Import your custom modules for fetching and processing data
from data_fetcher import fetch_crypto_data
from analysis import calculate_smi, normalize_smi
import config

# Function to fetch and prepare data from multiple timeframes
def fetch_and_prepare_data(crypto='BTC', timeframes=['1h', '4h', '1d'], limit=2000):
    all_data = []
    for timeframe in timeframes:
        data = fetch_crypto_data(crypto, timeframe, limit)
        data = calculate_smi(data)
        data = normalize_smi(data)
        all_data.append(data[['time', 'normalized_SMI', 'volumeto', 'close']])
    
    # Align data from different timeframes based on timestamps
    aligned_data = pd.concat(all_data, axis=1).dropna().reset_index(drop=True)
    return aligned_data

# Function to normalize input features
def normalize_features(data):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_columns = [col for col in data.columns if col not in ['time', 'close']]
    data[feature_columns] = feature_scaler.fit_transform(data[feature_columns])
    return data, feature_scaler

# Adjusted function to normalize the target column
def normalize_target(data):
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    # Assuming 'close' column is uniquely identifiable and last 'close' is the target
    close_column = data.filter(regex='^close$').columns[-1]  # This ensures we're only selecting the target 'close' column
    data[[close_column]] = target_scaler.fit_transform(data[[close_column]])
    return data, target_scaler, close_column  # Return the column name to keep track

# Function to create sequences for LSTM training
def create_sequences(data, n_steps_in, close_column):
    X, y = [], []
    feature_columns = [col for col in data.columns if col not in ['time'] + list(data.filter(regex='close').columns)]  # Exclude all 'close' columns
    
    for i in range(len(data) - n_steps_in):
        end_ix = i + n_steps_in
        if end_ix > len(data):
            break
        seq_x, seq_y = data.iloc[i:end_ix][feature_columns].values.astype(np.float32), data.iloc[end_ix][close_column].astype(np.float32)
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to print an example of input data and target
def print_example_data(X, y):
    example_X = X[0]
    example_y = y[0]
    print("Example Input Features:")
    example_df = pd.DataFrame(example_X, columns=[f'Feature_{i+1}' for i in range(example_X.shape[1])])
    print(example_df)
    print("\nCorresponding Target:")
    print(example_y)

# Function to evaluate the model's performance
def evaluate_model(model, X_test, y_test, target_scaler):
    y_pred = model.predict(X_test)
    print("y_pred shape (before reshape):", y_pred.shape)  # Debugging line
    
    # Ensure y_pred is correctly shaped for inverse transformation
    y_pred_inverse = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    # Explicitly reshape y_test for inverse scaling to ensure compatibility
    y_test_reshaped = y_test.reshape(-1, 1)
    y_test_inverse = target_scaler.inverse_transform(y_test_reshaped)

    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    mse = mean_squared_error(y_test_inverse, y_pred_inverse)

    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inverse, label='Actual Close Price')
    plt.plot(y_pred_inverse, label='Predicted Close Price', linestyle='--')
    plt.title('Actual vs Predicted Close Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    prepared_data = fetch_and_prepare_data()
    prepared_data, feature_scaler = normalize_features(prepared_data)
    prepared_data, target_scaler, close_column = normalize_target(prepared_data)
    
    features_set, labels = create_sequences(prepared_data, config.N_last_SMI, close_column)
    X_train, X_test, y_train, y_test = train_test_split(features_set, labels, test_size=0.2, random_state=42)
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    
    #evaluate_model(model, X_test, y_test, target_scaler)

    print_example_data(X_train, y_train)
