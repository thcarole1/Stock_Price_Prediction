
# Basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import exists

import warnings
warnings.filterwarnings(action = 'ignore')


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
    print("✅ LSTM model : Training in progress ....")
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

def plot_train_actual_predictions(y_train : np.array,
         y_test : np.array,
         y_pred : np.array,
         y_train_dates : np.array,
         y_test_dates : np.array,
         currency : str,
         short_name : str):

    '''
    This function plots train data, test data and predictions.
    Returns a png image stored locally.
    '''

    #Delete previous any previous png file
    file_path = 'data/processed_data/train_test_pred.png'

    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    # Plot stock prices : actual vs predicted
    plt.figure(figsize=(16,8))
    plt.title(f'Prediction of {short_name} stock price with LSTM')
    plt.plot(y_train_dates, y_train)
    plt.plot(y_test_dates, y_test)
    plt.plot(y_test_dates, y_pred)
    plt.legend(['Training data', 'Actual data', 'Predictions'], loc='lower right')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(f'Close price in {currency}', fontsize=18)
    plt.savefig(f'data/processed_data/train_test_pred.png')
    plt.close()
    print("✅ Figure of train data, test data and predictions created !")

def plot_actual_predictions(y_test : np.array,
                            y_pred : np.array,
                            y_test_dates : np.array,
                            currency : str,
                            short_name : str):

    '''
    This function plots test data and predictions.
    Returns a png image stored locally.
    '''

    #Delete previous any previous png file
    file_path = 'data/processed_data/test_pred.png'

    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    # Plot stock prices : actual vs predicted (ONLY predictions and actual. No train data displayed)
    plt.figure(figsize=(16,8))
    plt.title(f'Prediction of {short_name} stock price with LSTM')
    plt.plot(y_test_dates, y_test)
    plt.plot(y_test_dates, y_pred)
    plt.legend(['Actual data', 'Predictions'], loc='lower right')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(f'Close price in {currency}', fontsize=18)
    plt.savefig(f'data/processed_data/test_pred.png')
    plt.close()
    print("✅ Figure of test data and predictions created !")

def plot_actual_predictions_last_values(y_test : np.array,
                                        y_pred : np.array,
                                        y_test_dates : np.array,
                                        currency : str,
                                        short_name : str):
    '''
    This function plots test data and predictions on a limited number of values.
    Returns a png image stored locally.
    '''

    #Delete previous any previous png file
    file_path = 'data/processed_data/test_pred_limited.png'

    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    # Plot stock prices : actual vs predicted (ONLY predictions and actual. No train data displayed)
    number_last = 100

    plt.figure(figsize=(16,8))
    plt.title(f'Prediction of {short_name} stock price with LSTM on the last {number_last} days')
    plt.plot(y_test_dates[-number_last:], y_test[-number_last:])
    plt.plot(y_test_dates[-number_last:], y_pred[-number_last:])
    plt.legend(['Actual data', 'Predictions'], loc='lower right')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(f'Close price in {currency}', fontsize=18)
    plt.savefig(f'data/processed_data/test_pred_limited.png')
    plt.close()
    print("✅ Figure of test data and predictions (limited values) created !")

def create_summary(y_test : np.array,
                   y_pred : np.array,
                   y_test_dates : np.array):

    '''
    This function creates a summary dataframe describing
    actual unseen values (y_test), predictions (y_pred)
    and delta (absolute value btw both).
    Returns a pandas dataframe.
    '''
    # Display latest data (actual and predicted)
    def retrieve_element(s):
        return s[0]

    # Create Pandas dataframe with actual data and predicted data with time index
    actual_and_pred = pd.DataFrame({'actual' : y_test.tolist(),
                                    'predictions' : y_pred.tolist()},
                                    index = y_test_dates)

    # Formatting
    actual_and_pred['actual'] = actual_and_pred['actual'].apply(retrieve_element)
    actual_and_pred['predictions'] = actual_and_pred['predictions'].apply(retrieve_element)

    # New column with absolute value of difference between actual and predicted values
    actual_and_pred['delta'] = np.abs(actual_and_pred['actual'] - actual_and_pred['predictions'])

    # Description of dataframe
    summary = actual_and_pred.describe()
    print("✅ Summary dataframe created !")
    return summary
