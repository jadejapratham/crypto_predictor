import requests
from textblob import TextBlob
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

def fetch_news(query="Bitcoin"):
   
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
    
    if not text or not isinstance(text, str):
        return 0.0
        
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return 0.0
