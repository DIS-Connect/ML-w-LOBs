import pandas as pd
from typeguard import typechecked
from typing import Dict
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import calendar
from order_book.simulation import *
from order_book.transformation import *
from order_book.loader import *



def preprocess_obs(
    year,
    months,
    delivery_start, # tuple (hour, minute)
    delivery_end,   # tuple (hour, minute)
    ticks=500
    ):

    dl = DataLoader()

    for month in months:

        obs_to_save = {}

        last_day = calendar.monthrange(year, month)[1]
        for day in range(1, last_day + 1, 1):
            print(f"processing: {year}-{month}-{day}")

            start = datetime(year, month, day, delivery_start[0], delivery_start[1])
            end = datetime(year, month, day, delivery_end[0], delivery_end[1])

            order_data = dl.get_raw_product_orders(start, end)

            time_steps = get_time_steps_by_ticks(order_data, ticks)

            ob_vecs = get_exp_ob_vec_by_time_steps(order_data, time_steps)


            ob_name = f"day_{day}"
            obs_to_save[ob_name] = ob_vecs

        file_name = f"{year}-{month}_{delivery_start[0]}:{delivery_start[1]}_{delivery_end[0]}:{delivery_end[1]}.npz"

        np.savez_compressed("data/preprocessed/"+file_name, **obs_to_save)
        return "data/preprocessed/"+file_name









