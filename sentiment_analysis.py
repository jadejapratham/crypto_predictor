import requests
from textblob import TextBlob
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

def fetch_news(query="Bitcoin"):
    """
    Fetch news articles for a given query using NewsAPI
    
    Args:
        query (str): Search query for news articles
        
    Returns:
        list: List of article dictionaries or empty list on error
    """
    if not query or not isinstance(query, str):
        logger.error("Invalid query parameter")
        return []
        
    if not NEWS_API_KEY:
        logger.error("News API key not configured")
        return []

    try:
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        articles = response.json().get("articles", [])
        if not articles:
            logger.info(f"No articles found for query: {query}")
            
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return []

def analyze_sentiment(text):
    """
    Analyze sentiment of given text using TextBlob
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Sentiment polarity between -1 (negative) and 1 (positive)
    """
    if not text or not isinstance(text, str):
        return 0.0
        
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return 0.0