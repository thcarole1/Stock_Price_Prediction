
# Import from .py files
from ml_logic.data import retrieve_data, retrieve_historical_data,\
                          retrieve_currency, retrieve_short_name,\
                            memorize_dates, define_train_test_data,\
                            create_features_target, reshape_historical_data

from ml_logic.preprocessor import  check_outliers, scale_with_outliers, scale_without_outliers
from ml_logic.model import reshape_input_data, define_LSTM_model,\
                            train_model, plot_train_actual_predictions,\
                            plot_actual_predictions, \
                            plot_actual_predictions_last_values,\
                            create_summary

def predict_stock_price():
    # Enter ticker and target
    # ticker = "AAPL"
    # ticker = "MSFT"
    # ticker = "NVDA"
    # ticker = "GOOGL"
    # ticker = "AMZN"
    # ticker = "META"
    # ticker = "TSLA"
    ticker = "STLA"

    # Choose target
    target = 'Close'

    # Retrieve global data
    data = retrieve_data(ticker)

    # Retrieve historical data
    historical_data = retrieve_historical_data(data, target)

    # Retrieve Currency
    currency = retrieve_currency(data)

    # Retrieve short name of company
    short_name = retrieve_short_name(data)

    # Check for outliers
    numerical_columns_w_outliers, numerical_columns_no_outliers = check_outliers(historical_data)
    # print(f"Columns WITH outliers : {numerical_columns_w_outliers}")
    # print(f"Columns WITHOUT outliers : {numerical_columns_no_outliers}")

    # Scaling temporal data WITH outliers
    if len(numerical_columns_w_outliers) != 0:
        data_with_outliers, rb_scaler = scale_with_outliers(historical_data, numerical_columns_w_outliers)
        # print(data_with_outliers.shape)

    # Scaling temporal data WITHOUT outliers
    if len(numerical_columns_no_outliers) != 0:
        data_without_outliers, minmax_scaler = scale_without_outliers(historical_data, numerical_columns_no_outliers)
        # print(data_without_outliers.shape)

    # Memorize dates for vizualisation purpose
    dates = memorize_dates(historical_data)

    # Reshape data
    historical_data = reshape_historical_data(historical_data)

    # Define train data, test data, train dates, test dates
    train_data, test_data, train_dates, test_dates = define_train_test_data(historical_data, dates)

    # Create train datasets (X and y) and test datasets (X and y) and associated dates
    # time_step of past data to predict next data
    time_step = 1

    # Get X_train, X_test, y_train, y_test, y_train_dates and y_test_dates
    X_train, y_train, y_train_dates = create_features_target(train_data, time_step, train_dates)
    X_test, y_test, y_test_dates = create_features_target(test_data, time_step, test_dates)

    # Reshape data for LSTM (samples, time_steps, features)
    X_train, X_test = reshape_input_data(X_train, X_test)

    # Create LSTM model
    model = define_LSTM_model(X_train)
    # print(type(model))
    # print(type(y_train))

    # Train model
    train_model(model, X_train, y_train )

    # Predictions from unseen data
    y_pred = model.predict(X_test)

    # Let's reshape the data
    y_test = y_test.reshape(-1,1)
    y_train = y_train.reshape(-1,1)

    # Convert back to original price
    if target in numerical_columns_w_outliers:
        scaler = rb_scaler

    if target in numerical_columns_no_outliers:
        scaler = minmax_scaler

    # Convert back to original price
    y_test = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_train = scaler.inverse_transform(y_train)

    # ****************  Visualization  ******************************************
    # Plot the train data, the actual unseen data (y_test) and the predictions (y_pred)
    test = plot_train_actual_predictions(y_train, y_test, y_pred,
                                  y_train_dates,y_test_dates,
                                  currency,short_name)

    print(f"Image name type  :{type(test[0])}")
    print(f"Image data type  :{type(test[1])}")

    # breakpoint()

    # Plot the train data, the actual unseen data (y_test) and the predictions (y_pred)
    plot_actual_predictions(y_test, y_pred,y_test_dates, currency,short_name)

    # Plot stock prices : actual vs predicted  (limited values)
    # (ONLY predictions and actual. No train data displayed)
    plot_actual_predictions_last_values(y_test, y_pred,y_test_dates, currency,short_name)

    # ***************************************************************************

    # This function creates a summary dataframe describing
    # actual unseen values (y_test), predictions (y_pred)
    # and delta (absolute value btw both)
    summary = create_summary(y_test, y_pred, y_test_dates)
    print(summary)


def say_hello():
    print('Hello World !')

if __name__ == '__main__':
    try:
        predict_stock_price()
        # say_hello()

    except:
        import sys
        import traceback
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
