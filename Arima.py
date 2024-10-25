## Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf  # Changed from yahoo_fin to yfinance
from collections import deque
import numpy as np
import pandas as pd
import random
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

# Function to create and fit an ARIMA model
def create_arima_model(data, arima_order=(5, 1, 0)):
    # Initialize the ARIMA model with the given order
    model = ARIMA(data, order=arima_order)
    # Fit the model to the data
    model_fit = model.fit()
    return model_fit

# Function to perform ensemble prediction
def ensemble_prediction(predictions, weights=None):
    # Convert predictions to numpy arrays
    predictions = [np.array(pred) for pred in predictions]

    # Find the minimum length among predictions
    min_length = min(len(pred) for pred in predictions)
    # Trim predictions to the same length
    predictions = [pred[-min_length:] for pred in predictions]

    # If weights are not provided, assign equal weights
    if weights is None:
        weights = [1/len(predictions)] * len(predictions)

    # Normalize weights to sum to 1
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Initialize ensemble prediction array
    ensemble_pred = np.zeros_like(predictions[0])
    # Compute the weighted sum of predictions
    for weight, pred in zip(weights, predictions):
        ensemble_pred += weight * pred
    return ensemble_pred

# Function to load and preprocess data for multistep prediction
def load_data_multistep(ticker, n_steps=50, scale=True, lookup_step=1, split_by_date=True, test_size=0.2, feature_columns=['Adj Close']):
    # Fetch historical stock data using yfinance
    df = yf.download(ticker)

    # Ensure feature columns exist in the dataframe
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' does not exist in the dataframe.")

    # Add 'date' column if not present
    df['Date'] = df.index

    # Scale data if required
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
    else:
        column_scaler = None

    # Create future column for prediction target
    df['future'] = df['Adj Close'].shift(-lookup_step)

    # Drop rows with NaN values after shifting
    df.dropna(inplace=True)

    # Initialize sequences deque with a maximum length
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    # Iterate over the dataframe to create sequences and targets
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Separate sequences and targets
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
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
        "column_scaler": column_scaler,
        "df": df  # Return df for inverse transformation if needed
    }

# Function to create an LSTM model
def create_lstm_model(sequence_length, n_features, units=256, dropout=0.3, loss="mean_squared_error", optimizer="adam"):
    # Initialize a sequential model
    model = Sequential()
    # Add LSTM layer
    model.add(LSTM(units, return_sequences=False, input_shape=(sequence_length, n_features)))
    # Add dropout layer
    model.add(Dropout(dropout))
    # Add output layer
    model.add(Dense(1))
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer)
    return model

# Function to create a GRU model
def create_gru_model(sequence_length, n_features, units=256, dropout=0.3, loss="mean_squared_error", optimizer="adam"):
    # Initialize a sequential model
    model = Sequential()
    # Add GRU layer
    model.add(GRU(units, return_sequences=False, input_shape=(sequence_length, n_features)))
    # Add dropout layer
    model.add(Dropout(dropout))
    # Add output layer
    model.add(Dense(1))
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer)
    return model

# Main execution block
if __name__ == "__main__":
    try:
        print("Starting the stock price prediction script...")
        # Define stock ticker symbol
        ticker = 'AAPL'

        # Load and preprocess data with lookup_step=1
        print("Loading data...")
        data = load_data_multistep(ticker, n_steps=50, lookup_step=1)
        print("Data loaded successfully.")

        # Extract training and testing data
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        df = data["df"]
        scaler = data["column_scaler"]["Adj Close"]  # For inverse transformation

        # Reshape data for Random Forest Regressor
        X_train_rf = X_train.reshape((X_train.shape[0], -1))
        X_test_rf = X_test.reshape((X_test.shape[0], -1))

        print("Training models...")
        # --- Model Training and Prediction ---

        # ARIMA Model
        # Use the last feature in each sequence as input for ARIMA
        arima_input = X_train[:, -1, 0]  # Assuming 'Adj Close' is at index 0
        # Create and fit ARIMA model
        arima_model = create_arima_model(arima_input)
        # Forecast using ARIMA model
        arima_pred = arima_model.forecast(steps=len(X_test))

        # LSTM Model
        n_features = X_train.shape[2]
        # Create LSTM model
        lstm_model = create_lstm_model(sequence_length=50, n_features=n_features, units=256, dropout=0.3)
        # Train LSTM model
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
        # Predict using LSTM model
        lstm_pred = lstm_model.predict(X_test).flatten()

        # GRU Model
        # Create GRU model
        gru_model = create_gru_model(sequence_length=50, n_features=n_features, units=256, dropout=0.3)
        # Train GRU model
        gru_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
        # Predict using GRU model
        gru_pred = gru_model.predict(X_test).flatten()

        # Random Forest Regressor
        # Initialize Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=314)
        # Train Random Forest model
        rf_model.fit(X_train_rf, y_train)
        # Predict using Random Forest model
        rf_pred = rf_model.predict(X_test_rf)

        print("Models trained successfully.")

        # --- Evaluate Individual Model Performance ---

        from sklearn.metrics import mean_squared_error

        # Ensure y_test and predictions are aligned
        min_length = min(len(y_test), len(lstm_pred), len(gru_pred), len(rf_pred), len(arima_pred))
        y_test_trimmed = y_test[-min_length:]
        lstm_pred_trimmed = lstm_pred[-min_length:]
        gru_pred_trimmed = gru_pred[-min_length:]
        rf_pred_trimmed = rf_pred[-min_length:]
        arima_pred_trimmed = arima_pred[-min_length:]

        # Calculate MSE for each model
        mse_lstm = mean_squared_error(y_test_trimmed, lstm_pred_trimmed)
        mse_gru = mean_squared_error(y_test_trimmed, gru_pred_trimmed)
        mse_rf = mean_squared_error(y_test_trimmed, rf_pred_trimmed)
        mse_arima = mean_squared_error(y_test_trimmed, arima_pred_trimmed)

        print(f'LSTM Model MSE: {mse_lstm}')
        print(f'GRU Model MSE: {mse_gru}')
        print(f'Random Forest Model MSE: {mse_rf}')
        print(f'ARIMA Model MSE: {mse_arima}')

        # --- Adjust Ensemble Weights Based on Performance ---

        # Calculate inverse MSE for weighting
        inv_mse_lstm = 1 / mse_lstm
        inv_mse_gru = 1 / mse_gru
        inv_mse_rf = 1 / mse_rf
        inv_mse_arima = 1 / mse_arima

        # Sum of inverse MSEs
        inv_mse_sum = inv_mse_lstm + inv_mse_gru + inv_mse_rf + inv_mse_arima

        # Compute weights proportional to inverse MSE
        weights = [
            inv_mse_arima / inv_mse_sum,
            inv_mse_lstm / inv_mse_sum,
            inv_mse_gru / inv_mse_sum,
            inv_mse_rf / inv_mse_sum,
        ]

        print(f'Computed Weights: {weights}')

        # --- Ensemble Predictions with Adjusted Weights ---

        predictions = [arima_pred_trimmed, lstm_pred_trimmed, gru_pred_trimmed, rf_pred_trimmed]
        ensemble_pred = ensemble_prediction(predictions, weights=weights)

        # Calculate Ensemble MSE
        mse_ensemble = mean_squared_error(y_test_trimmed, ensemble_pred)
        print(f'Adjusted Ensemble Model MSE: {mse_ensemble}')

        # --- Plotting the Results ---

        # Inverse transform the scaled data to original values
        y_test_inv = scaler.inverse_transform(y_test_trimmed.reshape(-1, 1)).flatten()
        lstm_pred_inv = scaler.inverse_transform(lstm_pred_trimmed.reshape(-1, 1)).flatten()
        ensemble_pred_inv = scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()

        # Plot actual vs predicted prices
        plt.figure(figsize=(14, 7))
        plt.plot(y_test_inv, label="Actual Prices", color="blue")
        plt.plot(lstm_pred_inv, label="LSTM Predicted Prices", color="green")
        plt.plot(ensemble_pred_inv, label="Ensemble Predicted Prices", color="red")
        plt.title(f"Actual vs Predicted Stock Prices (Ensemble vs. LSTM)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        print("Script completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

