import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from data_fetcher import get_stock_data, get_crypto_data
from model_utils import prepare_data, build_model
from sentiment_analysis import fetch_news, analyze_sentiment
import time
from typing import Optional

st.set_page_config(
    page_title="Stock & Crypto Predictor",
    layout="wide",
    page_icon="📈"
)

st.markdown("""
    <style>
    .stButton>button {
        
        color: white;
        font-weight: bold;
    }
    .stAlert {
        padding: 20px;
        border-radius: 5px;
    }
    .positive-sentiment {
        color: green;
        font-weight: bold;
    }
    .negative-sentiment {
        color: red;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Stock & Crypto Price Predictor with Sentiment Analysis")
st.markdown("""
    Predict future prices using LSTM models and analyze market sentiment from news articles.
    """)

with st.sidebar:
    st.header("Settings")
    prediction_days = st.slider("Days to look back for prediction", 30, 90, 60)
    epochs = st.slider("Training epochs", 1, 50, 10)
    st.markdown("---")
    st.info("Note: Predictions are based on historical patterns and may not be accurate.")

col1, col2 = st.columns(2)
with col1:
    option = st.selectbox("Choose Market", ["Crypto"])
with col2:
    default_ticker = "BTC/USDT"
    ticker = st.text_input("Enter Ticker Symbol", default_ticker)

if st.button("Fetch & Predict"):
    with st.spinner('Fetching data...'):
        start_time = time.time()
        try:
            if option == "Stock":
                data = get_stock_data(ticker)
            else:
                data = get_crypto_data(ticker)
                
            if data is None:
                st.error("❌ Failed to fetch data. Please check your connection and try again.")
                st.stop()

            if 'close' not in data.columns:
                alternatives = ['Close', 'price', 'last', '4']
                found = False
                
                for alt in alternatives:
                    if alt in data.columns:
                        data = data.rename(columns={alt: 'close'})
                        found = True
                        break
                
                if not found:
                    st.error(f"""
                    ❌ Data doesn't contain expected 'close' prices. 
                    Columns found: {list(data.columns)}
                    """)
                    st.stop()

            if not pd.api.types.is_numeric_dtype(data['close']):
                try:
                    data['close'] = pd.to_numeric(data['close'])
                except:
                    st.error("❌ Could not convert 'close' prices to numeric values")
                    st.stop()
                    
            if len(data) < 60:
                st.error(f"❌ Insufficient data points ({len(data)}). Need at least 60.")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.stop()
            
    st.success(f"✅ Successfully fetched {len(data)} data points for {ticker}")
    st.write(f"**Date Range:** {data.index[0].date()} to {data.index[-1].date()}")
    
    st.subheader("📊 Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['close'], 
        name='Close Price',
        line=dict(color='royalblue', width=2)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🤖 Model Prediction")
    with st.spinner('Training model and making prediction...'):
        try:
            x, y, scaler = prepare_data(data, window_size=prediction_days)
            
            model = build_model((x.shape[1], 1))
            model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
            
            last_n = data['close'].values[-prediction_days:].reshape(-1, 1)
            scaled = scaler.transform(last_n)
            X_test = np.reshape(scaled, (1, prediction_days, 1))
            predicted_price = model.predict(X_test)
            predicted_price = scaler.inverse_transform(predicted_price)
            
            current_price = data['close'].iloc[-1]
            percent_change = ((predicted_price[0][0] - current_price) / current_price) * 100
            change_color = "green" if percent_change >= 0 else "red"
            change_icon = "↑" if percent_change >= 0 else "↓"
            
            st.markdown(f"""
                <div style="background-color:black;padding:20px;border-radius:10px;">
                    <h3 style="margin-top:0;">Prediction Results</h3>
                    <p><b>Current Price:</b> ${current_price:.2f}</p>
                    <p><b>Predicted Next Close:</b> ${predicted_price[0][0]:.2f}</p>
                    <p><b>Predicted Change:</b> <span style="color:{change_color}">
                        {change_icon} {abs(percent_change):.2f}%
                    </span></p>
                </div>
                """, unsafe_allow_html=True)
                
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")
            st.stop()
    
    st.subheader("📰 Market Sentiment Analysis")
    with st.spinner('Fetching and analyzing news...'):
        try:
            news = fetch_news(ticker.split('/')[0] if option == "Crypto" else ticker)
            
            if not news:
                st.warning("No news articles found for this ticker.")
                st.stop()
                
            st.info(f"Analyzing sentiment from {len(news[:3])} recent news articles...")
            
            for i, article in enumerate(news[:3]):
                with st.expander(f"News {i+1}: {article['title']}"):
                    st.write(article['description'] or "No description available.")
                    
                    if article['publishedAt']:
                        published_date = pd.to_datetime(article['publishedAt']).strftime('%B %d, %Y')
                        st.caption(f"Published: {published_date}")
                    
                    text = f"{article['title']} {article['description']}"
                    sentiment = analyze_sentiment(text)
                    
                    sentiment_class = "positive-sentiment" if sentiment >= 0 else "negative-sentiment"
                    sentiment_emoji = "😊" if sentiment >= 0 else "😞"
                    
                    st.markdown(f"""
                        <div class="{sentiment_class}">
                            {sentiment_emoji} Sentiment Score: {sentiment:.2f} 
                            ({'Positive' if sentiment >= 0 else 'Negative'})
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if article['url']:
                        st.markdown(f"[Read full article]({article['url']})")
                        
        except Exception as e:
            st.error(f"❌ Error in sentiment analysis: {str(e)}")
            st.stop()
    
    # Show performance metrics
    st.markdown("---")
    st.caption(f"Completed in {time.time() - start_time:.2f} seconds")
