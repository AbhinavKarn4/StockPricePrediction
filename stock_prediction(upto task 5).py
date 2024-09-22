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

# Set seed to ensure reproducibility
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


def shuffle_in_unison(a, b):
    # Shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


# ------------------- TASK 5 FUNCTIONS -------------------

def load_data_multistep(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=5, split_by_date=True,
                        test_size=0.2, feature_columns=['adjclose']):
    """
    Loads stock data and prepares it for multistep prediction (i.e., predicting `lookup_step` days into the future).
    """
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

    # For multistep, we need to create sequences of targets (multiple future values)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "column_scaler": column_scaler
    }


def load_data_multivariate_multistep(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=5, split_by_date=True,
                                     test_size=0.2,
                                     feature_columns=['open', 'high', 'low', 'close', 'volume', 'adjclose']):
    """
    Loads stock data for multivariate and multistep prediction.
    Predicts multiple future steps using multiple features as input.
    """
    df = si.get_data(ticker)

    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' does not exist in the dataframe.")

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
        y.append(targets)

    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        X_train, y_train = X[:train_samples], y[:train_samples]
        X_test, y_test = X[train_samples:], y[train_samples:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "column_scaler": column_scaler
    }


# ------------------- MODEL CREATION -------------------

def create_multistep_multivariate_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                                        loss="mean_squared_error", optimizer="adam", bidirectional=False,
                                        lookup_step=5):
    """
    Create a deep learning model for multistep and multivariate prediction.
    """
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(sequence_length, n_features)))
        elif i == n_layers - 1:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))

    model.add(Dense(lookup_step))  # Output multiple time steps (e.g., 5 days into the future)
    model.compile(loss=loss, optimizer=optimizer)
    return model


# ------------------- TRAINING AND EVALUATION -------------------

# Example usage: Load data and train the model for multistep, multivariate prediction
if __name__ == "__main__":
    ticker = 'AAPL'

    # Load data for multivariate, multistep prediction
    data = load_data_multivariate_multistep(ticker, n_steps=50, lookup_step=5)

    # Prepare data
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Get the number of features
    n_features = X_train.shape[2]

    # Create the model
    model = create_multistep_multivariate_model(sequence_length=50, n_features=n_features, lookup_step=5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")

    # Predict future stock prices (for 5 days into the future)
    predicted = model.predict(X_test)
    print("Predicted future prices:", predicted[:5])  # Show the first 5 predictions
