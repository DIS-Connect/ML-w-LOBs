import tensorflow as tf 
from tensorflow import keras
from keras.layers import Input
import pandas as pd
from datetime import datetime
from datetime import timedelta
from order_book.loader import *
from order_book.transformation import *
import holidays
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge

def get_simple_single_product_model(input_size):

    """
    max bid: 1
    min ask: 1
    seconds to delivery: 1
    holiday: 1
    dow: 6
    bids: 8
    asks: 8
    """

    inputs = Input(shape=(input_size))
    
    dense1 = keras.layers.Dense(1)(inputs)
    # output = keras.layers.Dense(1, activation='relu')(dense1)
    
    model = keras.models.Model(inputs, dense1)
    return model



def zero_pad(n):
    if n < 10:
        return f"0{n}"
    else:
        return str(n)



def one_hot_dow(dow):
    if dow == 0:
        return np.array([0,0,0,0,0,0])
    elif dow == 1:
        return np.array([1,0,0,0,0,0])
    elif dow == 2:
        return np.array([0,1,0,0,0,0])
    elif dow == 3:
        return np.array([0,0,1,0,0,0])
    elif dow == 4:
        return np.array([0,0,0,1,0,0])
    elif dow == 5:
        return np.array([0,0,0,0,1,0])
    elif dow == 6:
        return np.array([0,0,0,0,0,1])
    else:
        raise ValueError(f"{dow} is not in [1,7]")
        
    
def is_holiday(date):
    de_holidays = holidays.Germany(years=date.year)
    return date in de_holidays
        


def preprocess_data(
        start_date,
        end_date,
        del_start,
        del_end,
        ticks = 2000,               # must be multiple of 500
        target_index = "ID1"        # can be "ID1" or "ID3"

):
    
    x_data = []
    y_data = []
    
    # Set Up
    dl = DataLoader()
    indices = dl.load_indices()
    delta = end_date - start_date
    exp_ob_vecs_by_day = None
    last_month = 0

    for i in range(delta.days + 1):

        # 1. Set up important variables for this day
        current_date = start_date + timedelta(days=i)
        year = current_date.year
        month = current_date.month
        day = current_date.day
        del_start_hour = del_start[0]
        del_start_minute = del_start[1]
        dow = one_hot_dow(current_date.weekday())
        holiday = 1 if is_holiday(current_date) else 0

        # 2. Load the relevant index value and load the exp order book vectors for this day
        del_start_string = f"{year}-{zero_pad(month)}-{zero_pad(day)}T{zero_pad(del_start_hour)}:{zero_pad(del_start_minute)}:00Z"
        index_value = indices[(indices["TimeResolution"] == "15min") &(indices["DeliveryStart"] == del_start_string) & (indices["IndexName"] == target_index)]["IndexPrice"].iloc[0]
        

        if last_month != month:
            exp_ob_vecs_by_day = dl.get_ob_vec_series_for_month(
                year=year,
                month=month,
                delivery_start = del_start,
                delivery_end = del_end)
        
        
        exp_ob_vecs = exp_ob_vecs_by_day[f"day_{day}"]
        
        # 3. Calulate the time of the last order book depending on the index
        delivery_start = datetime(year, month, day, del_start_hour, del_start_minute)
        
        if target_index == "ID1":
            last_ob_time = delivery_start - timedelta(hours=1)
        elif target_index == "ID3":
            last_ob_time = delivery_start - timedelta(hours=1)
        else:
            raise ValueError("I do not know this index")


        # 4. Build x_data and y_data
        for i in range(0, len(exp_ob_vecs), int(ticks/500)):
            meta_data, bids, asks = exp_ob_vecs[i]

            unix_time_stamp  = meta_data[0]
            highest_bid = meta_data[1]
            lowest_ask = meta_data[2]

            time_stamp = datetime.fromtimestamp(unix_time_stamp)
            if time_stamp > last_ob_time:
                break

            seconds_to_delivery = (delivery_start - time_stamp).seconds
            
            input_row = np.hstack([[highest_bid, lowest_ask, seconds_to_delivery, holiday], dow, bids, asks])
            # input_row = np.hstack([[highest_bid, lowest_ask, seconds_to_delivery, holiday], dow])
            output_row = index_value


            x_data.append(input_row)
            y_data.append(output_row)
        
    return np.array(x_data), np.array(y_data)

    


def compare_agains_naive(ssp_model, x_val, y_val):

    naive_prediction = []
    for i in range(len(x_val)):
        naive_prediction.append((x_val[i][0] + x_val[i][1])/2)      # mid price

    naive_prediction = np.array(naive_prediction)
    ssp_prediction = ssp_model.predict(x_val)

    print(y_val.shape)
    print(naive_prediction.shape)
    mse_naive = mean_squared_error(y_val, naive_prediction)
    mse_ssp = mean_squared_error(y_val, ssp_prediction)

    print(f"mse_naive:\t {mse_naive}")
    print(f"mse_ssp:  \t {mse_ssp}")



def compare_agains_OLS(ssp_model,x_train, y_train, x_val, y_val):
    
    ols_model = LinearRegression()
    ols_model.fit(x_train, y_train)
    ols_prediction = ols_model.predict(x_val)
    mse_ols = mean_squared_error(y_val, ols_prediction)
    print(f"mse_ols:\t {mse_ols}")


    lasso = Lasso(alpha=0.1)
    lasso.fit(x_train, y_train)
    lasso_prediction = lasso.predict(x_val)
    mse_lasso = mean_squared_error(y_val, lasso_prediction)
    print(f"mse_lasso:\t {mse_lasso}")


    ridge = Ridge(alpha=0.5) 
    ridge.fit(x_train, y_train)
    ridge_prediction = ridge.predict(x_val)
    mse_ridge = mean_squared_error(y_val, ridge_prediction)
    print(f"mse_ridge:\t {mse_ridge}")

    
    ssp_prediction = ssp_model.predict(x_val)
    mse_ssp = mean_squared_error(y_val, ssp_prediction)
    print(f"mse_ssp:\t {mse_ssp}")
        

