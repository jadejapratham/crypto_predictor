import streamlit as st
import requests
import os
from textblob import TextBlob
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def fetch_news(query="Bitcoin"):

    if not query or not isinstance(query, str):
        logger.error("Invalid query parameter")
        return []
        
    if not NEWS_API_KEY:
        logger.error("News API key not configured")
        return []

    try:
        api_key = st.secrets["NEWS_API_KEY"]
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}"
        response = requests.get(url)
    
        if response.status_code != 200:
            st.error(f"API error: {response.status_code} - {response.text}")
            return []
    
        articles = response.json().get("articles", [])
        return articles
        
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
