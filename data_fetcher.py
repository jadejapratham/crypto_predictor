import requests
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_crypto_data(symbol: str = "bitcoin", days: int = 365) -> pd.DataFrame:
    """
    Fetch historical cryptocurrency price data from CoinGecko.

    Args:
        symbol (str): CoinGecko coin ID (e.g., 'bitcoin')
        days (int): Number of days of historical data to fetch

    Returns:
        pd.DataFrame: DataFrame with 'date' and 'price' columns
    """
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            "vs_currency": "inr",
            "days": days,
            "interval": "daily"
        }
        logger.info(f"Fetching crypto data for {symbol} from CoinGecko")
        response = requests.get(url, params=params)
        response.raise_for_status()

        prices = response.json().get("prices", [])
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit='ms')
        return df[["date", "price"]]

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching crypto data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching crypto data: {e}")
        return pd.DataFrame()
