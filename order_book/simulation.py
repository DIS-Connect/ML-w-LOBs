import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ipywidgets import interact, FloatSlider, fixed




@typechecked
def get_product_orders(produkt : Dict, m7_market_orders : pd.DataFrame) -> pd.DataFrame:
    """
    returns all orders relating to a product

    @Params:
    products: {"Product" : "Intraday_Quarter_Hour_Power", "DeliveryStart": datetime}
    """
    
    m7_market_orders = m7_market_orders[(m7_market_orders["DeliveryEnd"] == produkt["DeliveryEnd"]) & (m7_market_orders["DeliveryStart"] == produkt["DeliveryStart"])]
    return m7_market_orders

@typechecked
def get_visible_ob_at(produkt : Dict, at : str, m7_market_orders : pd.DataFrame) -> pd.DataFrame:
    """
    returns the state of the order book of the specified product at the specified time

    @Params:
    products: {"Product" : "Intraday_Quarter_Hour_Power", "DeliveryStart": datetime}
    """

    # Filtered specified prodict and before relevant time
    m7_market_orders = m7_market_orders[(m7_market_orders["DeliveryEnd"] == produkt["DeliveryEnd"]) & (m7_market_orders["DeliveryStart"] == produkt["DeliveryStart"])]
    m7_market_orders = m7_market_orders[m7_market_orders["TransactionTime"] <= at]
      

    # Remove all: deleted by user (D) / completely matched (M) / deleted by trading system (X)
    delete_ids = m7_market_orders[m7_market_orders["ActionCode"].isin(["D", "M", "X"])]["OrderId"]
    m7_market_orders = m7_market_orders[~m7_market_orders["OrderId"].isin(delete_ids)]

    id_highest_rev_no = m7_market_orders.groupby('OrderId')['RevisionNo'].idxmax()
    m7_market_orders = m7_market_orders.loc[id_highest_rev_no]



    # Test
    # m7_market_orders = m7_market_orders[m7_market_orders["ParentId"].notna()]

    return m7_market_orders


def interact_with_sliders(df, product, at_prefix, bin_width, book_width, mid, hour, minute, second):

    h = str(int(hour)) if hour > 9 else "0"+str(int(hour))
    m = str(int(minute)) if minute > 9 else "0"+str(int(minute))
    s = str(int(second)) if second > 9 else "0"+str(int(second))

    time = at_prefix + f"{h}:{m}:{s}Z"
    
    ob = get_visible_ob_at(product, time, df)
    
    start = product["DeliveryStart"]
    end = product["DeliveryEnd"]
    print(f"Product: {start} - {end}")
    print(f"Order Book at: {time}")
    visualize_ob(ob, bin_width, book_width, mid)

def visualize_interactive(df, product, at_prefix):

    bin_width_slider = FloatSlider(min=0.00, max=5, step=0.1, value=0.5, description='Bin Width:')
    book_width_slider = FloatSlider(min=0.0, max=100, step=1, value=100, description='Book Width:')
    mid_slider = FloatSlider(min=0, max=100, step=1, value=50, description='Mid:')

    hour_slider = FloatSlider(min=0, max=23, step=1, value=12, description='Hour:')
    minute_slider = FloatSlider(min=0, max=60, step=1, value=0, description='Minute:')
    second_slider = FloatSlider(min=0, max=60, step=1, value=0, description='Second:')


    interact(interact_with_sliders,df=fixed(df),product=fixed(product), at_prefix=fixed(at_prefix), bin_width=bin_width_slider, book_width=book_width_slider, mid=mid_slider, hour = hour_slider,minute=minute_slider, second = second_slider )





@typechecked
def visualize_ob(order_book : pd.DataFrame, bin_width, book_width, mid=None):


    bids = order_book[order_book["Side"] == "BUY"]
    asks = order_book[order_book["Side"] == "SELL"]

    price = (asks['Price'].min() + bids['Price'].max())/2
    spread = asks['Price'].min() - bids['Price'].max()
    
    if mid is None:
        mid = price
        
    max = mid + book_width
    min = mid - book_width

    bids = bids[bids["Price"] >= min]
    asks = asks[asks["Price"] <= max]

    max_vol = 100# max(asks['Volume'].max(), bids['Volume'].max())

    bids = bids.groupby("Price")["Volume"].sum().reset_index()
    asks = asks.groupby("Price")["Volume"].sum().reset_index()




    if(len(asks) > 0):
        ask_bins = pd.cut(asks['Price'], bins=np.arange(int(asks['Price'].min()), int(asks['Price'].max()) + 1, bin_width))
        binned_asks = asks.groupby(ask_bins)['Volume'].sum().reset_index()
        ask_midpoints = [(bin.left + bin.right) / 2 for bin in binned_asks['Price']]
        plt.bar(ask_midpoints, binned_asks['Volume'], width=bin_width, color='red', alpha=0.7)

    if(len(bids) > 0):
        bid_bins = pd.cut(bids['Price'], bins=np.arange(int(bids['Price'].min()), int(bids['Price'].max()) + 1, bin_width))
        binned_bids = bids.groupby(bid_bins)['Volume'].sum().reset_index()
        bid_midpoints = [(bin.left + bin.right) / 2 for bin in binned_bids['Price']]
        plt.bar(bid_midpoints, binned_bids['Volume'], width=bin_width, color='green', alpha=0.7)

    plt.xlim(min, max)
    plt.yscale('log')

    plt.axvline(x=price, color='blue', linestyle='-', linewidth=1, label='Price')
    
    



    # Visualize Spread
    if(len(bids) > 0) and (len(asks) > 0):
        rectangle = Rectangle((bids['Price'].max(), 0), spread, max_vol, linewidth=0, fill=True, facecolor=(0.5, 0.5, 0.5, 0.2), label='Rectangle')
        plt.gca().add_patch(rectangle)

    

    plt.xlabel('Price')
    plt.ylabel('Volume')
    plt.title('Bids / Asks')
    plt.show()


    print(f"Price: {price} EUR/MWH")
    if(len(bids) > 0) and (len(asks) > 0):
        print(f"Spread: {int(spread*100)/100.0} EUR/MWH")
    else:
        print(f"Spread: 0.00 EUR/MWH")
    print("Bids:")
    
    


def visualize_ob_data_series(ob_data_series):
    return None