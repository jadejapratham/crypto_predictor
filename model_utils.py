import numpy as np
import pandas as pd  # Added missing import
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(data: pd.DataFrame, window_size: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
        
    if 'close' not in data.columns:
        raise ValueError("DataFrame must contain 'close' column")
        
    if len(data) < window_size + 10:  # Minimum data points needed
        raise ValueError(f"Need at least {window_size + 10} data points, got {len(data)}")

    try:
        # Scale data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['close']])
        
        # Create sequences
        x, y = [], []
        for i in range(window_size, len(scaled_data)):
            x.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        
        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Reshape for LSTM input (samples, timesteps, features)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        return x, y, scaler
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def build_model(input_shape: tuple) -> Sequential:
    
    try:
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=32),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        logger.info("Model built successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        raise