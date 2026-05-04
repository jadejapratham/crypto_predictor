---
title: Crypto Trend Analyzer
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
tags:
- streamlit
- finance
- machine-learning
- predictive-modeling
- crypto
pinned: false
short_description: AI-powered cryptocurrency price forecasting and market sentiment analysis.
---

# 📈 Crypto Trend Analyzer

An interactive web application deployed on Hugging Face Spaces that combines historical market data, machine learning, and natural language processing to analyze and forecast cryptocurrency trends. 

### ✨ Key Features

* **Live Market Data:** Fetches up-to-date cryptocurrency prices and historical data directly from the CoinGecko API.
* **AI Price Forecasting:** Utilizes a custom-built Neural Network (trained on a 60-day rolling window) to predict price movements up to 7 days into the future.
* **Market Sentiment Analysis:** Scrapes the latest news headlines related to the selected cryptocurrency via NewsAPI and scores current market sentiment using TextBlob NLP.
* **Interactive Visualizations:** Features responsive, interactive charts built with Plotly for deep-dive technical analysis.

### 🛠️ Technology Stack

* **Frontend:** Streamlit, Plotly
* **Backend:** Python 3.13
* **Machine Learning:** TensorFlow / Keras, Scikit-learn, NumPy, Pandas
* **NLP & Data APIs:** TextBlob, CoinGecko API, NewsAPI
* **Deployment:** Docker, Hugging Face Spaces

### 🚀 How to Use

1. Enter a valid cryptocurrency ID (e.g., `bitcoin`, `ethereum`, `solana`) in the sidebar.
2. Adjust the slider to select how many days of historical data you want to visualize (minimum 60 days required for the ML model).
3. Select your prediction window (1 to 7 days ahead).
4. Click **Run Forecast** to generate the charts and fetch the latest market sentiment.

---
*Note: This project is built for educational and portfolio purposes. The machine learning predictions provided by this application do not constitute financial advice.*
