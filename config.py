from API_KEY_SECRET import API_KEY_SECRET

BASE_URL = 'https://min-api.cryptocompare.com/data/v2/'


# Configuration for fetching data
API_KEY = API_KEY_SECRET
CRYPTO_SYMBOL = 'BTC'
timeframes_config = {
    #'1m': {'endpoint': ('histominute', 1), 'weight': 0.5},
    '5m': {'endpoint': ('histominute', 5), 'weight': 0.5},
    '15m': {'endpoint': ('histominute', 15), 'weight': 1},
    '30m': {'endpoint': ('histominute', 30), 'weight': 1},
    '1h': {'endpoint': ('histohour', 1), 'weight': 2},
    '4h': {'endpoint': ('histohour', 4), 'weight': 3},
    '1d': {'endpoint': ('histoday', 1), 'weight': 4},
}


output_timeframes_config = {
    #'1m': {'endpoint': ('histominute', 1), 'weight': 0.5},
    '5m': {'endpoint': ('histominute', 5), 'weight': 1},
    '1h': {'endpoint': ('histohour', 1), 'weight': 2},
    '1d': {'endpoint': ('histoday', 1), 'weight': 4},
}


# Timeframe conversion dictionary updated for readability and completeness
timeframe_offsets = {
    '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
    '1h': '1H', '4h': '4H', '1d': '1D'
}

DATA_LIMIT = 2000

# config.py

N_last_SMI = 3  # Example: Use the last 3 SMI values
Prediction_Window = '1h'  # Example: Predict the next 1 hour close price

output_dir = 'out'


desired_fields = ['time', 'high', 'low', 'open', 'volumeto', 'close']
