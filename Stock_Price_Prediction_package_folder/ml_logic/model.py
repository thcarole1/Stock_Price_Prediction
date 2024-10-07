
# Basic imports
import numpy as np

# Create RNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

def reshape_input_data(X_train : np.array, X_test : np.array):
    '''
    Reshape data for LSTM (samples, time_steps, features)
    '''

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    print("✅ Input data has been reshaped for LSTM model")
    return X_train, X_test

def define_LSTM_model(X_train : np.array):
    '''
    This function creates an architecture and compilation conditions for the deep learning model
    '''
    # 1- RNN Architecture
    model = Sequential()
    model.add(layers.LSTM(units=80,
                          activation='tanh',
                          input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(layers.Dense(1, activation="linear"))

    # 2- Compilation
    model.compile(loss='mse', optimizer='adam', metrics =['mse', 'mae', RootMeanSquaredError()])
    print("✅ LSTM model has been defined")
    return model

def train_model(model : Sequential, X_train : np.array, y_train : np.array ):
    '''
    This function trains the model.
    '''
    # Fitting model
    es = EarlyStopping(patience = 50, restore_best_weights=True)

    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=16,
                        epochs=1000,
                        verbose=0,
                        callbacks=[es],
                        validation_split=0.3,
                        shuffle=False)
    print("✅ LSTM model has been trained")
