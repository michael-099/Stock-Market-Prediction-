import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model('tesla_stock_price_prediction_model.keras')
print("Model loaded successfully.")
scaler = MinMaxScaler(feature_range=(0, 1))


start = '2015-01-01'
end = '2023-12-01'
tesla_data = yf.download('TSLA', start=start, end=end)


last_5_prices = tesla_data['Close'].tail(100).values
def predict_price():
    print("\n=== Tesla Stock Price Prediction ===")

    try:
        input_data = last_5_prices.reshape(-1, 1)
        input_scaled = scaler.fit_transform(input_data)
        input_scaled = np.reshape(input_scaled, (1, input_scaled.shape[0], 1))  

        prediction = model.predict(input_scaled)
        predicted_price = scaler.inverse_transform(prediction)  

        print(f"\nPredicted Closing Price for the Next Day: ${predicted_price[0][0]:.2f}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    predict_price()
