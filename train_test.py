import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random

# Step 1: Load the dataset
file_path = 'out/BTC_merged_with_metrics.csv'  # Update this to your actual file path
data = pd.read_csv(file_path)

# Selecting the 'close_5m' column
close_prices = data[['close_5m']].values

# Normalize the close prices
scaler = MinMaxScaler(feature_range=(0,1))
scaled_close_prices = scaler.fit_transform(close_prices)

# Function to create the dataset with lookback
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Split into train and test sets
look_back = 1
X, Y = create_dataset(scaled_close_prices, look_back)
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
trainX, testX = X[0:train_size], X[train_size:len(X)]
trainY, testY = Y[0:train_size], Y[train_size:len(Y)]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Step 4: Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Predicting the future points for backtesting
def predict_future_points(base_input, steps=10):
    future_predictions = []
    current_input = base_input
    for _ in range(steps):
        predicted_point = model.predict(current_input.reshape(1, 1, look_back))
        future_predictions.append(predicted_point[0][0])
        current_input = np.array([predicted_point])
    return np.array(future_predictions).reshape(-1, 1)

# Backtesting on random test points
samples_to_show = 5
indexes = random.sample(range(len(testX) - 10), samples_to_show)

plt.figure(figsize=(15, 15))

for i, index in enumerate(indexes):
    plt.subplot(samples_to_show, 1, i+1)
    actual_future_values = scaler.inverse_transform(testX[index:index+10].reshape(-1, 1)).flatten()
    predicted_future_values = scaler.inverse_transform(predict_future_points(testX[index], 10)).flatten()
    plt.plot(range(10), actual_future_values, label="Actual", marker='o', color='blue')
    plt.plot(range(10), predicted_future_values, label="Predicted", marker='x', color='red')
    plt.legend()
    plt.title(f"Back Test from Point {index}")

plt.tight_layout()
plt.show()

# Final forecast for the next 10 points
final_base_input = scaled_close_prices[-1]
final_predictions = scaler.inverse_transform(predict_future_points(final_base_input, 10)).flatten()

plt.figure(figsize=(10, 6))
plt.plot(range(10), final_predictions, label='Forecasted Next 10 Prices', marker='o', color='red')
plt.title('Forecast for Next 10 Points in the Future')
plt.xlabel('Point')
plt.ylabel('Price')
plt.legend()
plt.show()
