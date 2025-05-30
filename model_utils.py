import numpy as np
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(data: pd.DataFrame, window_size: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
        
    if 'close' not in data.columns:
        raise ValueError("DataFrame must contain 'close' column")
        
    if len(data) < window_size + 10:
        raise ValueError(f"Need at least {window_size + 10} data points, got {len(data)}")

    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['close']])

        x, y = [], []
        for i in range(window_size, len(scaled_data)):
            x.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])

        x = np.array(x)
        y = np.array(y)

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
