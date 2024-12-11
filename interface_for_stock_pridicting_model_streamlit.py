import numpy as np
import yfinance as yf
import datetime as dt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow.keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model = load_model('stock_price_prediction_model.keras')


scaler = MinMaxScaler(feature_range=(0, 1))


def fetch_data(start_date, end_date):
    tesla_data = yf.download('TSLA', start=start_date, end=end_date)
    return tesla_data

def prepare_data(tesla_data, prediction_days=100):
    scaled_data = scaler.fit_transform(tesla_data[['Close']].values)
    model_inputs = scaled_data[-prediction_days:]
    return model_inputs


def predict_for_date(start_date, end_date):
    tesla_data = fetch_data(start_date, end_date)
    model_inputs = prepare_data(tesla_data)

    predicted_price = model.predict(model_inputs)

    predicted_price = scaler.inverse_transform(
        np.concatenate((predicted_price, np.zeros_like(predicted_price)), axis=1)
    )[:, 0]

    return predicted_price[0]

def app():
    st.title("📈 Tesla Stock Price Prediction")
    st.write("Predict Tesla's stock price for a specific date using machine learning!")

    # Date input for prediction
    input_date = st.date_input("Enter a date", dt.datetime.today())

    if st.button("🔍 Predict Stock Price for Selected Date"):
        with st.spinner("Fetching data and making predictions..."):
            try:
                start_date = dt.datetime(2024, 1, 1)
                end_date = input_date.strftime('%Y-%m-%d')

                predicted_price = predict_for_date(start_date, end_date)
                st.success(f"🚀 Predicted Tesla stock price for {input_date}: **${predicted_price:.2f}**")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.write("---")
    st.info("Disclaimer: Predictions are based on historical data and may not reflect actual future prices. Always perform your own analysis before making financial decisions.")

if __name__ == '__main__':
    app()
