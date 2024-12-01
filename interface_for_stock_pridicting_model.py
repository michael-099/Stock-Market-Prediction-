import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
import tkinter as tk
from tkinter import messagebox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime as dt

model = load_model('tesla_stock_price_prediction_model.keras')


scaler = MinMaxScaler(feature_range=(0, 1))


def fetch_data(start_date, end_date):
    tesla_data = yf.download('TSLA', start=start_date, end=end_date)
    return tesla_data

def plot_data(actual_prices, predicted_price):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color='black', label='Actual Price')
    plt.plot(predicted_price, color='green', label='Predicted Price')
    plt.title("Tesla Share Price Prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def predict_price():
    try:
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()

       
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')

      
        tesla_data = fetch_data(start_date, end_date)

        
        scaled_data = scaler.fit_transform(tesla_data[['Close', 'Volume']].values)
        prediction_days = 100
        x_train, y_train = [], []
        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, :])
            y_train.append(scaled_data[x, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

        
        model_inputs = tesla_data[['Close', 'Volume']].tail(len(tesla_data) - prediction_days).values
        model_inputs = scaler.transform(model_inputs)
        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, :])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))

        predicted_price = model.predict(x_test)
        predicted_price = scaler.inverse_transform(np.concatenate((predicted_price, np.zeros_like(predicted_price)), axis=1))[:, 0]
        actual_prices = tesla_data['Close'].values

        mae = mean_absolute_error(actual_prices[-len(predicted_price):], predicted_price)
        mse = mean_squared_error(actual_prices[-len(predicted_price):], predicted_price)
        rmse = np.sqrt(mse)

        messagebox.showinfo("Error Metrics", f"Mean Absolute Error: {mae}\nMean Squared Error: {mse}\nRoot Mean Squared Error: {rmse}")

        
        plot_data(actual_prices[-len(predicted_price):], predicted_price)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


root = tk.Tk()
root.title("Tesla Stock Price Prediction")


tk.Label(root, text="Start Date (YYYY-MM-DD):").pack(pady=5)
start_date_entry = tk.Entry(root)
start_date_entry.pack(pady=5)

tk.Label(root, text="End Date (YYYY-MM-DD):").pack(pady=5)
end_date_entry = tk.Entry(root)
end_date_entry.pack(pady=5)

predict_button = tk.Button(root, text="Predict", command=predict_price)
predict_button.pack(pady=20)

exit_button = tk.Button(root, text="Exit", command=root.quit)
exit_button.pack(pady=5)

root.mainloop()
