import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
import numpy as np
import pandas as pd
import random
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set seed to ensure reproducibility
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

# Load data for ARIMA model
def create_arima_model(data, arima_order=(5, 1, 0)):
    model = ARIMA(data, order=arima_order)
    model_fit = model.fit()
    return model_fit

# Ensemble prediction function
def ensemble_prediction(arima_pred, lstm_pred, weight_arima=0.5, weight_lstm=0.5):
    # Make sure the arrays are of the same length
    min_length = min(len(arima_pred), len(lstm_pred))
    arima_pred_trimmed = arima_pred[-min_length:]
    lstm_pred_trimmed = lstm_pred[-min_length:]

    # Ensemble prediction as a weighted sum
    return weight_arima * arima_pred_trimmed + weight_lstm * lstm_pred_trimmed

# Task 5: Load data for multistep LSTM prediction
def load_data_multistep(ticker, n_steps=50, scale=True, lookup_step=5, split_by_date=True, test_size=0.2, feature_columns=['adjclose']):
    df = si.get_data(ticker)

    # Ensure the specified feature columns exist
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' does not exist in the dataframe.")

    # Add 'date' column if not present
    df["date"] = df.index

    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
    else:
        column_scaler = None

    # Create `lookup_step` future columns
    for i in range(1, lookup_step + 1):
        df[f'future_{i}'] = df['adjclose'].shift(-i)

    df.dropna(inplace=True)  # Drop any rows with NaNs after shifting

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, *targets in zip(df[feature_columns].values,
                               *[df[f'future_{i}'].values for i in range(1, lookup_step + 1)]):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), targets])

    X, y = [], []
    for seq, targets in sequence_data:
        X.append(seq)
        y.append(targets)  # Store the entire target sequence (e.g., 5 days of future prices)

    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        X_train, y_train = X[:train_samples], y[:train_samples]
        X_test, y_test = X[train_samples:], y[train_samples:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "column_scaler": column_scaler
    }

# Create the LSTM model
def create_lstm_model(sequence_length, n_features, units=256, dropout=0.3, loss="mean_squared_error", optimizer="adam", lookup_step=5):
    model = Sequential()
    model.add(LSTM(units, return_sequences=False, input_shape=(sequence_length, n_features)))
    model.add(Dropout(dropout))
    model.add(Dense(lookup_step))  # Predict multiple time steps
    model.compile(loss=loss, optimizer=optimizer)
    return model

# Training and Evaluation
if __name__ == "__main__":
    ticker = 'AAPL'

    # Load data for multistep LSTM prediction
    data = load_data_multistep(ticker, n_steps=50, lookup_step=5)

    # Prepare data
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Train ARIMA model
    arima_model = create_arima_model(data['X_train'][:, -1])  # Using adjclose as ARIMA target
    arima_pred = arima_model.forecast(steps=len(X_test))

    # Train LSTM model
    n_features = X_train.shape[2]
    lstm_model = create_lstm_model(sequence_length=50, n_features=n_features, lookup_step=5)
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # LSTM predictions
    lstm_pred = lstm_model.predict(X_test)
    lstm_pred_rescaled = lstm_pred.flatten()

    # Ensemble prediction
    ensemble_pred = ensemble_prediction(arima_pred[-len(lstm_pred_rescaled):], lstm_pred_rescaled)

    # Ensure consistent lengths for y_test and ensemble_pred
    y_test_trimmed = y_test.flatten()[-len(ensemble_pred):]

    # Now calculate the MSE for the trimmed y_test and ensemble prediction
    mse_ensemble = mean_squared_error(y_test_trimmed, ensemble_pred)
    print(f'Ensemble Model MSE: {mse_ensemble}')

    # Plot actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_trimmed, label="Actual Prices", color="blue")
    plt.plot(lstm_pred_rescaled, label="LSTM Predicted Prices", color="green")
    plt.plot(ensemble_pred, label="Ensemble Predicted Prices", color="red")
    plt.title(f"Actual vs Predicted Stock Prices (ARIMA + LSTM Ensemble)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

