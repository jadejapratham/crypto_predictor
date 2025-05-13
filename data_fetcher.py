import yfinance as yf
import ccxt
import pandas as pd
import datetime
import logging
from typing import Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch historical stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL")
        period (str): Time period to fetch (default "1y")
        interval (str): Data interval (default "1d")
        
    Returns:
        pd.DataFrame: DataFrame with stock data or None on error
    """
    if not ticker or not isinstance(ticker, str):
        logger.error("Invalid ticker parameter")
        return None

    try:
        logger.info(f"Fetching stock data for {ticker}")
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            logger.error(f"No data found for ticker: {ticker}")
            return None
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        return None

def get_crypto_data(symbol: str = "BTC/USDT", days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch historical cryptocurrency data from Binance
    
    Args:
        symbol (str): Crypto pair symbol (e.g., "BTC/USDT")
        days (int): Number of days of data to fetch
        
    Returns:
        pd.DataFrame: DataFrame with crypto data or None on error
    """
    if not symbol or not isinstance(symbol, str):
        logger.error("Invalid symbol parameter")
        return None

    try:
        exchange = ccxt.binance()
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        
        logger.info(f"Fetching crypto data for {symbol}")
        bars = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since)
        
        if not bars:
            logger.error(f"No data found for symbol: {symbol}")
            return None
            
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except ccxt.NetworkError as e:
        logger.error(f"Network error fetching crypto data: {str(e)}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error fetching crypto data: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching crypto data: {str(e)}")
        return None