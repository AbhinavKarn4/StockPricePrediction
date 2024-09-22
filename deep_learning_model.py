import tensorflow as tf
from stock_prediction import load_data, create_model
import numpy as np

# Load the data
data = load_data('AAPL', n_steps=50, scale=True, split_by_date=True)

# Extract training and test data
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# Convert the data to float32 (to avoid type errors)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Get the number of features (e.g., adjclose, volume, etc.)
n_features = X_train.shape[2]

# Create a 3-layer LSTM model
lstm_model = create_model(sequence_length=50,
                          n_features=n_features,
                          units=256,
                          cell=tf.keras.layers.LSTM,
                          n_layers=3,
                          dropout=0.3,
                          optimizer='adam',
                          loss='mean_squared_error',
                          bidirectional=False)

# Train the LSTM model
history_lstm = lstm_model.fit(X_train, y_train,
                              epochs=50,
                              batch_size=32,
                              validation_data=(X_test, y_test))

# Evaluate the model on test data
test_loss = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Test Loss: {test_loss}")

# Example for creating and training a GRU model
gru_model = create_model(sequence_length=50,
                         n_features=n_features,
                         units=256,
                         cell=tf.keras.layers.GRU,
                         n_layers=3,
                         dropout=0.3,
                         optimizer='adam',
                         loss='mean_squared_error',
                         bidirectional=False)

# Train the GRU model
history_gru = gru_model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_test, y_test))

# Evaluate the GRU model on test data
test_loss_gru = gru_model.evaluate(X_test, y_test)
print(f"GRU Test Loss: {test_loss_gru}")
