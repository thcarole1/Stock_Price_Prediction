
from Stock_Price_Prediction_package_folder.api_functions.data_api import retrieve_data_api,\
retrieve_historical_data_api, retrieve_currency_api,\
retrieve_short_name_api, memorize_dates_api,\
reshape_historical_data_api, define_train_test_data_api,\
create_features_target_api

from Stock_Price_Prediction_package_folder.api_functions.preprocessor_api import check_outliers_api,\
scale_with_outliers_api, scale_without_outliers_api

from Stock_Price_Prediction_package_folder.api_functions.model_api import reshape_input_data_api,\
define_LSTM_model_api, train_model_api, plot_train_actual_predictions_api,\
plot_actual_predictions_api, plot_actual_predictions_last_values_api,\
create_summary_api

def predict_stock_price_api(query):
    '''
    This function predicts the stock price value based on an ticker input.
    Returns xxx
    '''
    # Choose target
    target = 'Close'

    # Retrieve global data
    data = retrieve_data_api(query)

    # Retrieve historical data
    historical_data = retrieve_historical_data_api(data, target)

    # Retrieve Currency
    currency = retrieve_currency_api(data)

    # Retrieve short name of company
    short_name = retrieve_short_name_api(data)

    # Check for outliers
    numerical_columns_w_outliers, numerical_columns_no_outliers = check_outliers_api(historical_data)

    # Scaling temporal data WITH outliers
    if len(numerical_columns_w_outliers) != 0:
        data_with_outliers, rb_scaler = scale_with_outliers_api(historical_data, numerical_columns_w_outliers)

    # Scaling temporal data WITHOUT outliers
    if len(numerical_columns_no_outliers) != 0:
        data_without_outliers, minmax_scaler = scale_without_outliers_api(historical_data, numerical_columns_no_outliers)

    # Memorize dates for vizualisation purpose
    dates = memorize_dates_api(historical_data)

    # Reshape data
    historical_data = reshape_historical_data_api(historical_data)

    # Define train data, test data, train dates, test dates
    train_data, test_data, train_dates, test_dates = define_train_test_data_api(historical_data, dates)

    # Create train datasets (X and y) and test datasets (X and y) and associated dates
    # time_step of past data to predict next data
    time_step = 1

    # Get X_train, X_test, y_train, y_test, y_train_dates and y_test_dates
    X_train, y_train, y_train_dates = create_features_target_api(train_data, time_step, train_dates)
    X_test, y_test, y_test_dates = create_features_target_api(test_data, time_step, test_dates)

    # Reshape data for LSTM (samples, time_steps, features)
    X_train, X_test = reshape_input_data_api(X_train, X_test)

    # Create LSTM model
    model = define_LSTM_model_api(X_train)
    # print(type(model))
    # print(type(y_train))

    # Train model
    train_model_api(model, X_train, y_train )

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
    image_dict = {}
    # Plot the train data, the actual unseen data (y_test) and the predictions (y_pred)
    image_dict['train_actual_predictions_api'] = plot_train_actual_predictions_api(y_train, y_test, y_pred,
                                                                                y_train_dates,y_test_dates,
                                                                                currency,short_name)

    # Plot the train data, the actual unseen data (y_test) and the predictions (y_pred)
    image_dict['actual_predictions_api'] = plot_actual_predictions_api(y_test, y_pred,y_test_dates, currency,short_name)

    # Plot stock prices : actual vs predicted  (limited values)
    # (ONLY predictions and actual. No train data displayed)
    image_dict['actual_predictions_last_values_api'] = plot_actual_predictions_last_values_api(y_test, y_pred,y_test_dates, currency,short_name)

    # ***************************************************************************

    # This function creates a summary dataframe describing
    # actual unseen values (y_test), predictions (y_pred)
    # and delta (absolute value btw both)
    summary = create_summary_api(y_test, y_pred, y_test_dates)

    return image_dict, summary
