import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from data_fetcher import get_crypto_data
from model_utils import prepare_data, build_model
from sentiment_analysis import fetch_news, analyze_sentiment
import numpy as np

st.set_page_config(page_title="Crypto Price Predictor", layout="wide", page_icon="📈")
st.title("📈 Crypto Price Predictor with Sentiment Analysis")

st.sidebar.header("🔧 Settings")
ticker = st.sidebar.text_input("Crypto ID (e.g., bitcoin, ethereum)", "bitcoin")

days = st.sidebar.slider("Days of historical data", min_value=60, max_value=365, value=180)
prediction_days = st.sidebar.slider("Days to predict ahead", min_value=1, max_value=7, value=3)

if st.sidebar.button("Run Forecast"):
    df = get_crypto_data(ticker, days)

    if df.empty:
        st.error("❌ Failed to fetch crypto data. Please check the crypto ID or try again later.")
    else:
        st.subheader("📊 Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["price"], name='Close Price'))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🤖 Model Prediction")
        df.set_index("date", inplace=True)
        df.rename(columns={"price": "close"}, inplace=True)
        x, y, scaler = prepare_data(df)
        model = build_model((x.shape[1], 1))
        model.fit(x, y, epochs=5, batch_size=32, verbose=0)

        predictions = []
        last_60 = df['close'].values[-60:].reshape(-1, 1)

        for _ in range(prediction_days):
            scaled = scaler.transform(last_60)
            X_test = np.reshape(scaled, (1, 60, 1))
            pred = model.predict(X_test)
            predictions.append(pred[0][0])
            last_60 = np.append(last_60, scaler.inverse_transform(pred)[0][0])[-60:].reshape(-1, 1)

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=prediction_days)
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices})

        st.dataframe(forecast_df)
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Historical'))
        forecast_fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted Price"], name='Forecast'))
        st.plotly_chart(forecast_fig, use_container_width=True)

        st.subheader("📰 Sentiment Analysis (Latest News)")
        news = fetch_news(ticker)
        if not news:
            st.warning("⚠️ No news articles found or failed to fetch news.")
        else:
            for article in news[:3]:
                title = article.get('title') or ""
                description = article.get('description') or ""
                
                if title:
                    st.markdown(f"**{title}**")
                if description:
                    st.write(description)

                combined_text = f"{title} {description}".strip()
                sentiment = analyze_sentiment(combined_text)
                
                st.write(f"🧠 Sentiment Score: `{sentiment:.2f}`")
                st.markdown("---")
