import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
import yfinance as yf
import os

# Define constants and configuration parameters
N_STEPS = 50
LOOKUP_STEP = 1
TEST_SIZE = 0.2
FEATURE_COLUMNS = ['Adj Close', 'Volume']
LOSS = 'mean_squared_error'
UNITS = 256
N_LAYERS = 2
DROPOUT = 0.2
OPTIMIZER = 'adam'
BIDIRECTIONAL = False
SCALE = True
SPLIT_BY_DATE = True
SHUFFLE = False
MODEL_NAME = 'model_v0.1'
SAVE_LOCAL = True
DATA_DIR = 'local_data'
NAN_STRATEGY = 'fill_mean'

# Load data from Yahoo Finance
def load_data(ticker, start_date, end_date, feature_columns=FEATURE_COLUMNS):
    local_file_path = f"{DATA_DIR}/{ticker}_data.csv"
    if SAVE_LOCAL and os.path.exists(local_file_path):
        data = pd.read_csv(local_file_path, index_col=0, parse_dates=True)
        print(f"Loaded data from local storage: {local_file_path}")
    else:
        data = yf.download(ticker, start_date, end_date)
        if SAVE_LOCAL:
            os.makedirs(DATA_DIR, exist_ok=True)
            data.to_csv(local_file_path)
            print(f"Data saved locally to: {local_file_path}")

    # Check available columns
    print("Available columns:", data.columns)

    # Handle missing values
    if NAN_STRATEGY == 'drop':
        data.dropna(inplace=True)
    elif NAN_STRATEGY == 'fill_mean':
        data.fillna(data.mean(), inplace=True)
    else:
        raise ValueError(f"Unknown NaN strategy: {NAN_STRATEGY}. Use 'drop' or 'fill_mean'.")

    # Scale features
    if SCALE:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[feature_columns] = scaler.fit_transform(data[feature_columns])
    else:
        scaler = None

    # Split data
    if SPLIT_BY_DATE:
        train_data = data[:int(len(data) * (1 - TEST_SIZE))]
        test_data = data[int(len(data) * (1 - TEST_SIZE)):]
    else:
        train_data, test_data = train_test_split(data, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=42)

    X_train, y_train = create_xy(train_data, N_STEPS, LOOKUP_STEP, feature_columns)
    X_test, y_test = create_xy(test_data, N_STEPS, LOOKUP_STEP, feature_columns)

    return X_train, y_train, X_test, y_test, scaler


# Create X and y arrays from the data
def create_xy(data, n_steps, lookup_step, feature_columns):
    """
    Create X (features) and y (labels) arrays from the data.

    Parameters:
    data (pd.DataFrame): Data containing features
    n_steps (int): Number of time steps to use as input
    lookup_step (int): Number of days to look ahead for the label
    feature_columns (list): List of feature columns to use

    Returns:
    X (numpy array): Array of feature data
    y (numpy array): Array of labels
    """
    X, y = [], []
    data = data[feature_columns].values
    for i in range(len(data) - n_steps - lookup_step):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps+lookup_step-1][0])  # Use the first column for the label (e.g., 'adjclose')
    return np.array(X), np.array(y)

# Construct the LSTM model
def create_model(n_steps, n_features, loss, units, n_layers, dropout, optimizer, bidirectional):
    """
    Create an LSTM model for time series prediction.

    Parameters:
    n_steps (int): Number of time steps in the input
    n_features (int): Number of features in the input
    loss (str): Loss function to use
    units (int): Number of LSTM units in each layer
    n_layers (int): Number of LSTM layers
    dropout (float): Dropout rate to prevent overfitting
    optimizer (str): Optimizer for training
    bidirectional (bool): Whether to use bidirectional LSTM layers

    Returns:
    model (Sequential): Compiled LSTM model
    """
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=(n_steps, n_features)))
            else:
                model.add(LSTM(units, return_sequences=True, input_shape=(n_steps, n_features)))
        elif i == n_layers - 1:
            if bidirectional:
                model.add(Bidirectional(LSTM(units)))
            else:
                model.add(LSTM(units))
        else:
            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=True)))
            else:
                model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Dense(1))  # Output layer for predicting a single value
    model.compile(loss=loss, optimizer=optimizer)
    return model

# Load and preprocess the data
X_train, y_train, X_test, y_test, scaler = load_data('AAPL', '2010-01-01', '2022-02-26')

# Check the number of features (columns) in the training data
n_features = X_train.shape[2]

# Create and compile the model
model = create_model(N_STEPS, n_features, LOSS, UNITS, N_LAYERS, DROPOUT, OPTIMIZER, BIDIRECTIONAL)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.2f}")

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
predicted_prices = model.predict(X_test)

# Inverse transform to get actual price values (if scaling was applied)
if scaler:
    predicted_prices = scaler.inverse_transform(np.concatenate((predicted_prices, np.zeros((predicted_prices.shape[0], len(FEATURE_COLUMNS) - 1))), axis=1))[:, 0]
    true_prices = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(FEATURE_COLUMNS) - 1))), axis=1))[:, 0]
else:
    true_prices = y_test

# Plot the true prices vs. predicted prices
plt.figure(figsize=(10, 6))
plt.plot(true_prices, color='black', label='Actual Prices')
plt.plot(predicted_prices, color='green', label='Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Save the model to a file for future use
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")

# Load the model from the saved file (for demonstration)
# model = keras.models.load_model(MODEL_NAME)

# Predict the future price using the last available data point
last_sequence = X_test[-1].reshape(1, N_STEPS, n_features)
future_price = model.predict(last_sequence)
if scaler:
    future_price = scaler.inverse_transform(np.concatenate((future_price, np.zeros((1, len(FEATURE_COLUMNS) - 1))), axis=1))[:, 0]
print(f"Predicted future price: {future_price[0]:.2f}")