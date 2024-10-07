import pandas as pd
import numpy as np
from scipy.stats import iqr

# Scaling the data
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def check_outliers(historical_data : pd.DataFrame):
    '''
    This function checks for outliers in the historical data.
    Returns 2 lists of strings :
        - a list of columns WITH outliers
        - a list of columns WITHOUT outliers
    '''

# Identify the columns with outliers
    numerical_columns_w_outliers = []
    numerical_columns_no_outliers = []

    for col in historical_data.columns:
        # Calculate IQR
        iqr_value = iqr(historical_data[col])

        #Calculate 1st quartile
        q1 = np.percentile(historical_data[col],25)

        #Calculate 3rd quartile
        q3 = np.percentile(historical_data[col],75)

        #Calculate lower limit below which data point is considered an outlier
        outlier_lim_low = q1 - 1.5 * iqr_value

        #Calculate higher limit above which data point is considered an outlier
        outlier_lim_high = q3 + 1.5 * iqr_value

        #Calculate number of 'low' outliers
        outlier_condition_low = historical_data[col] < outlier_lim_low
        number_outliers_low = len(historical_data[outlier_condition_low][col])

        #Calculate number of 'high' outliers
        outlier_condition_high = historical_data[col] > outlier_lim_high
        number_outliers_high = len(historical_data[outlier_condition_high][col])

        #Calculate total number of outliers
        number_outliers_total = number_outliers_low + number_outliers_high

        #If any outliers in column, column is added to a list of columns with outliers
        if number_outliers_total > 0:
            numerical_columns_w_outliers.append(col)
        elif number_outliers_total == 0:
            numerical_columns_no_outliers.append(col)

        print("✅ Check for outliers has been done !")
        return numerical_columns_w_outliers, numerical_columns_no_outliers

def scale_with_outliers(historical_data : pd.DataFrame, numerical_columns_w_outliers : list):
    '''
    This function scales the data that contains outliers, with a RobustScaler.
    Returns a pandas dataframe with scaled data (with Robust Scaler).
    '''

    # This is executed ONLY if there is at least 1 column with outliers
    if len(numerical_columns_w_outliers) != 0:

        # Instantiate the robust scaler
        rb_scaler = RobustScaler()

        # Fit the robust scaler on X_train
        rb_scaler.fit(historical_data[numerical_columns_w_outliers])

        # Transform X_train and X_test through the fitted robust scaler
        historical_data[numerical_columns_w_outliers] = rb_scaler.transform(historical_data[numerical_columns_w_outliers])
        print("✅ Historical data has been scaled with Robust Scaler !")
        return historical_data, rb_scaler

def scale_without_outliers(historical_data : pd.DataFrame, numerical_columns_no_outliers : list):
    '''
    This function scales the data that DOES NOT contain outliers, with a MinMax Scaler.
    Returns a pandas dataframe with scaled data (with MinMax Scaler).
    '''

    # This is executed ONLY if there is at least 1 column with outliers
    if len(numerical_columns_no_outliers) != 0:

        # Instantiate the robust scaler
        minmax_scaler = MinMaxScaler()

        # Fit the robust scaler on X_train
        minmax_scaler.fit(historical_data[numerical_columns_no_outliers])

        # Transform X_train and X_test through the fitted robust scaler
        historical_data[numerical_columns_no_outliers] = minmax_scaler.transform(historical_data[numerical_columns_no_outliers])
        print("✅ Historical data has been scaled with MinMax Scaler !")
        return historical_data, minmax_scaler
