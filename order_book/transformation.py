import pandas as pd
from typeguard import typechecked
from typing import Dict
import numpy as np
from datetime import datetime, timedelta


@typechecked
def get_visible_product_ob_at(at : str, m7_market_orders : pd.DataFrame) -> pd.DataFrame:


    # Filtered specified prodict and before relevant time
    m7_market_orders = m7_market_orders[m7_market_orders["TransactionTime"] <= at]
      

    # Remove all: deleted by user (D) / completely matched (M) / deleted by trading system (X)
    delete_ids = m7_market_orders[m7_market_orders["ActionCode"].isin(["D", "M", "X"])]["OrderId"]
    m7_market_orders = m7_market_orders[~m7_market_orders["OrderId"].isin(delete_ids)]

    id_highest_rev_no = m7_market_orders.groupby('OrderId')['RevisionNo'].idxmax()
    m7_market_orders = m7_market_orders.loc[id_highest_rev_no]

    # Filter out orders that are hibernated
    m7_market_orders = m7_market_orders[m7_market_orders["ActionCode"] != "H"]


    # Test
    # m7_market_orders = m7_market_orders[m7_market_orders["ParentId"].notna()]

    return m7_market_orders


def transform_ob_to_perc_vector(ob):
    # [0% - 2%], [2% - 5%],[5% - 10%], [10% - 20%], [20% - 50%], [50% - infty]
    bids = ob[ob["Side"]== "BUY"]
    asks = ob[ob["Side"]== "SELL"]
    
    price = (bids["Price"].max() + asks["Price"].min()) / 2


    # calculate limits
    bucket_limits = [0.02, 0.05, 0.1, 0.2, 0.5]

    bid_vec = []
    ask_vec = []

    for limit in bucket_limits:
        vol_bid = bids[bids["Price"] >= (1 - limit) * price]["Volume"].sum()
        bid_vec.append(vol_bid)
    
        vol_ask = asks[asks["Price"] <= (1 + limit) * price]["Volume"].sum()
        ask_vec.append(vol_ask)


    return price, np.array(bid_vec), np.array(ask_vec)



def transform_ob_to_tensor(order_book : pd.DataFrame, num_orders):


    bids = order_book[order_book["Side"] == "BUY"]
    asks = order_book[order_book["Side"] == "SELL"]

    bids = bids.groupby("Price")["Volume"].sum().reset_index()
    asks = asks.groupby("Price")["Volume"].sum().reset_index()

    bids.sort_values(by="Price")
    asks.sort_values(by="Price")

    bids = bids[-num_orders:]
    asks = asks[:num_orders]
    
    
    bids_price = bids["Price"]
    bids_volume = bids["Volume"]
    
    asks_price = asks["Price"]
    asks_volume = asks["Volume"]
    
    if len(bids_price) < num_orders or len(asks_price) < num_orders:
        
        # calculate pads
        bid_price_pad = bids_price[0] if len(bids_price) > 0 else asks_price[0]
        asks_price_pad = asks_price[-1] if len(asks_price) > 0 else bids_price[-1]
        
        # pad bids
        bids_price = np.pad(bids_price, (num_orders - len(bids_price), 0), mode='constant', constant_values=bid_price_pad)
        bids_volume = np.pad(bids_volume, (0, num_orders - len(bids_volume)), mode='constant')
        
        # pad asks
        asks_price = np.pad(asks_price, (0, num_orders - len(asks_price)), mode='constant', constant_values=asks_price_pad)
        asks_volume = np.pad(asks_volume, (0, num_orders - len(asks_volume)), mode='constant')
        
        
    bid_tensor = np.vstack((bids_price, bids_volume))
    ask_tensor = np.vstack((asks_price, asks_volume))
    
    return np.array([bid_tensor, ask_tensor])

def to_ob_series_by_ticks(
    order_data,
    num_orders,
    ticks,
    start = None,
    end = None):
    
    
    
    if start is None:
        start = order_data["TransactionTime"].min()
        start = datetime.strptime(start, '%Y-%m-%dT%H:%M:%S.%fZ')
        start = start.replace(hour=16, minute=0, second=0, microsecond=0)
        start = start.strftime('%Y-%m-%dT%H:%M:%S')
        
    
    if end is None:
        end = order_data["DeliveryStart"][0]
        end = datetime.strptime(end, '%Y-%m-%dT%H:%M:%SZ')
        end -= timedelta(minutes=10)
        end = end.strftime('%Y-%m-%dT%H:%M:%S')
        
    lobs = []
        
    transac_times = np.array(order_data["TransactionTime"])
    transac_times = transac_times[transac_times >= start]
    transac_times = transac_times[transac_times < end]
    transac_times.sort()
    
                                 
    time_steps = transac_times[0::ticks]
    
    for i in range(len(time_steps)):
        
        curr_str = time_steps[i]
        ob = get_visible_product_ob_at(curr_str, order_data)
        if(len(ob) != 0):
        
            ob_tensor = transform_ob_to_tensor(ob, num_orders)
            lobs.append((curr_str, ob_tensor))
        
    return lobs
    
@typechecked
def to_ob_series_by_timedelta(
    order_data,
    num_orders,
    delta: timedelta,
    start = None,
    end = None):

    
    
    # Calculate the total duration
    if start is None:
        start = order_data["TransactionTime"].min()
        start = datetime.strptime(start, '%Y-%m-%dT%H:%M:%S.%fZ')
        start = start.replace(hour=16, minute=0, second=0, microsecond=0)
        
    if end is None:
        end = order_data["DeliveryStart"][0]
        end = datetime.strptime(end, '%Y-%m-%dT%H:%M:%SZ')
        end -= timedelta(minutes=10)
        
    
    lobs = []
    current_time = start
    while current_time < end:
        
        curr_str = current_time.strftime('%Y-%m-%dT%H:%M:%S')
        order_book = get_visible_product_ob_at(curr_str, order_data)
        
        ob_tensor = transform_ob_to_tensor(order_book, num_orders)
        
        lobs.append((curr_str, ob_tensor))
        
        current_time += delta
        
    return lobs

def get_price_trends(ob_series, horizons=6):
    
    trends = []
    num_obs = len(ob_series)
    
    for i in range(num_obs):
        
        max_bid = ob_series[i][1][0][0][-1]
        min_ask = ob_series[i][1][1][0][0]
        
        trend = []
        
        for j in range(1, 1 + min(horizons, num_obs-i-1)):
            
            tmp_max_bid = ob_series[i+j][1][0][0][-1]
            tmp_min_ask = ob_series[i+j][1][1][0][0]

            if min_ask < tmp_max_bid:
                trend.append([1,0,0])
            elif max_bid > tmp_min_ask:
                trend.append([0,0,1])
            else:
                trend.append([0,1,0])
                        
        trends.append(trend)
    
    return np.array(trends)
            
def ob_series_to_vectors(ob_series):
    
    num_obs = len(ob_series)
    ob_vectors = []
    change_vectors = []
    y_data = []
    
    # do the first one outside the for loop
    ob = ob_series[0][1]
    ob_vector = np.concatenate((ob[0][0], ob[1][0], ob[0][1], ob[1][1]), axis=0)
    ob_vectors.append(ob_vector)
    
    # fill the first change vector with only ones
    change_vec = np.ones(len(ob_vector))
    change_vectors.append(change_vec)
    
    for i in range(1, num_obs):
        ob = ob_series[i][1]
        ob_vector = np.concatenate((ob[0][0], ob[1][0], ob[0][1], ob[1][1]), axis=0)
        
        change_vector = (ob_vector / ob_vectors[i-1])
        change_vector = np.sqrt(change_vector)
        ob_vectors.append(ob_vector)
        change_vectors.append(change_vector)
        
    return np.array(ob_vectors), np.array(change_vectors)
       
def vecs_to_model_input(vecs, obs_per_input, price_trends, horizons):

    num_obs = len(vecs)
    x_data = []
    y_data = []
    
    # loop: (obs_per_input-1) to (num_obs- horizons -1)
    for i in range(obs_per_input-1, num_obs-horizons):
        
        x_sample = []
        
        # loop: (obs_per_input - 1) to 0
        for j in range(obs_per_input-1, -1,-1):
            x_sample.append(vecs[i-j])
            
        x_data.append(x_sample)
        y_data.append(price_trends[i])
        
    return np.array(x_data), np.array(y_data)
    
def ob_series_to_model_input(ob_series, obs_per_input, price_trends, trends_per_output):
        
    num_obs = len(ob_series)
    x_data = []
    y_data = []
    
    for i in range(num_obs - obs_per_input - trends_per_output):
        
        x_sample = []
        
        for j in range(obs_per_input):
            ob = ob_series[i+j][1]
            ob_vector = np.concatenate((ob[0][1], ob[1][1], ob[0][0], ob[1][0]), axis=0)
            x_sample.append(ob_vector)
            
        x_data.append(x_sample)
        y_data.append(price_trends[i+obs_per_input-1])
        
    return np.array(x_data), np.array(y_data)



def get_obs_by_ticks(
    order_data,
    ticks,
    start = None,
    end = None):
    
    
    
    if start is None:
        start = order_data["TransactionTime"].min()
        start = datetime.strptime(start, '%Y-%m-%dT%H:%M:%S.%fZ')
        start = start.replace(hour=16, minute=0, second=0, microsecond=0)
        start = start.strftime('%Y-%m-%dT%H:%M:%S')
        
    
    if end is None:
        end = order_data["DeliveryStart"][0]
        end = datetime.strptime(end, '%Y-%m-%dT%H:%M:%SZ')
        end -= timedelta(minutes=10)
        end = end.strftime('%Y-%m-%dT%H:%M:%S')
        
    lobs = []
        
    transac_times = np.array(order_data["TransactionTime"])
    transac_times = transac_times[transac_times >= start]
    transac_times = transac_times[transac_times < end]
    transac_times.sort()
    
                                 
    time_steps = transac_times[0::ticks]
    
    for i in range(len(time_steps)):
        
        curr_str = time_steps[i]
        ob = get_visible_product_ob_at(curr_str, order_data)
        if(len(ob) != 0):
            lobs.append((curr_str, ob))
        
    return lobs
