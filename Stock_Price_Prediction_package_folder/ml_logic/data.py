
# Retrieve data
import yfinance as yf
import pandas as pd
import numpy as np


def retrieve_data(ticker : str):
    '''
    This function retrieves data from Yahoo Finance through a ticker.
    Returns type <class 'yfinance.ticker.Ticker'>
    '''
    print("✅ data has been retrieved from yfinance")
    return  yf.Ticker(ticker)

def retrieve_historical_data(data : yf.ticker.Ticker, target : str):
    '''
    This function retrieves temporal data from Yahoo Finance through yf.ticker.Ticker object.
    The period of extraction is fixed to 10 years max.
    Returns a Pandas Dataframe with extracted target only.
    '''
    period = '10y'
    print("✅ historical data has been retrieved from yfinance")
    return data.history(period=period)[[target]]

def retrieve_currency(data : yf.ticker.Ticker):
    '''
    This function retrieves the currency in which the stock is traded in.
    Source is Yahoo Finance.
    Returns a string corresponding to the currency.
    '''
    print("✅ Currency has been retrieved from yfinance")
    return data._price_history._history_metadata['currency']

def retrieve_short_name(data : yf.ticker.Ticker):
    '''
    This function retrieves the name of the company from yahoo Finance.
    Returns a string.
    '''
    print("✅ Short_name has been retrieved from yfinance")
    return data._price_history._history_metadata['shortName']

def memorize_dates(historical_data : pd.DataFrame) :
    ''''
    This function memorized the historical dates for vizualisation purpose.
    Returns a numpy.array.
    '''
    print("✅ Dates have been memorized")
    return historical_data.index.values.reshape(-1,1)

def reshape_historical_data(historical_data : pd.DataFrame):
    '''
    This function reshape the historical data.
    Returns a Numpy array.
    '''
    print("✅ Historical data has been reshaped")
    return historical_data.values.reshape(-1,1)

def define_train_test_data(historical_data : np.ndarray, dates : np.ndarray):
    '''
    This function create train data (80% of total data) and test data.
    Returns train dataset, a test dataset and (train dates and test dates)
    '''
    # Define size of train data
    data_percentage = 0.8
    train_size = int(len(historical_data) * data_percentage)

    # Create train data and test data
    train_data = historical_data[:train_size]
    test_data = historical_data[train_size:]

    # Create train data dates and test data dates (datetime)
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]

    print("✅ Train data and test data has been defined")
    return train_data, test_data, train_dates, test_dates

def create_features_target(dataset : pd.DataFrame, time_step : int, dates : np.ndarray):
    ''''
    This function creates X and y with associated dates (for later vizualisation).
    Returns Numpy arrays.
    '''

    X, y = [], []
    time = []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
        time.append(dates[i,0])
    print("✅ Features and target have been created")
    return np.array(X), np.array(y), np.array(time)
