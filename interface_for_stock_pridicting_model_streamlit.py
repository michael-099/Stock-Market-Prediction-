import numpy as np
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model

model = load_model('stock_price_prediction_model.keras')

scaler = MinMaxScaler(feature_range=(0, 1))

def fetch_data():
    today = dt.datetime.today()
    start_date = (today - dt.timedelta(days=200)).strftime('%Y-%m-%d') 
    tesla_data = yf.download('TSLA', start=start_date, end=today.strftime('%Y-%m-%d'))
    return tesla_data

def prepare_data(tesla_data, prediction_days=50):  # Adjust to 50 time steps
    scaled_data = scaler.fit_transform(tesla_data[['Close', 'Volume']].values)
    
    # Use the last 50 time steps
    model_inputs = scaled_data[-prediction_days:]
    
    # Reshape to (1, time_steps, features)
    model_inputs = np.reshape(model_inputs, (1, model_inputs.shape[0], 2))  # 1 sample, 50 time steps, 2 features
    return model_inputs

def predict_tomorrow():
    tesla_data = fetch_data()
    model_inputs = prepare_data(tesla_data)

    # Make prediction
    predicted_price = model.predict(model_inputs)
    predicted_price = scaler.inverse_transform(np.concatenate((predicted_price, np.zeros_like(predicted_price)), axis=1))[:, 0]

    print(f"Predicted Tesla stock price for tomorrow: ${predicted_price[0]:.2f}")

predict_tomorrow()
