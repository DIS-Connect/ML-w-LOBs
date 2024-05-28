import os
import pandas as pd
import boto3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from order_book.simulation import *
from tqdm import tqdm
import awswrangler as wr

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class DataLoader():

    def __init__(self):
        
        self.s3 = boto3.resource('s3', )

        bucket_raw = "da-historical-raw-001"
        self.raw_bucket = self.s3.Bucket(bucket_raw)

        bucket_processed = "da-historical-processed"
        self.processed_bucket = self.s3.Bucket(bucket_processed)
        
        self.columns = ['OrderId', 'InitialId', 'ParentId', 'Side', 'Product', 'DeliveryStart',
       'DeliveryEnd', 'CreationTime', 'DeliveryArea', 'ExecutionRestriction',
       'UserdefinedBlock', 'LinkedBasketId', 'RevisionNo', 'ActionCode',
       'TransactionTime', 'ValidityTime', 'Price', 'Currency', 'Quantity',
       'QuantityUnit', 'Volume', 'VolumeUnit']


        
    def get_local_file_path(self, deliverystart, deliveryend):
        
        minute_start = deliverystart.strftime("%M")
            
        minute_end = deliveryend.strftime("%M")
            
        hour_start = deliverystart.strftime("%H")
            
        hour_end = deliveryend.strftime("%H")
            
        day = deliverystart.strftime("%d")

        month = deliverystart.strftime("%m")
        
        year = deliverystart.strftime("%Y")
        
        folder_name = f"data/orders/{year}/{month}/{day}"
        
        file_name = f"/{hour_start}-{minute_start}_{hour_end}-{minute_end}.csv"
        
        return folder_name, file_name
    
    def store_product_orders(self, order_data):
        
        
        deliverystart = datetime.strptime(order_data["DeliveryStart"][0], "%Y-%m-%dT%H:%M:%SZ")
        
        deliveryend = datetime.strptime(order_data["DeliveryEnd"][0], "%Y-%m-%dT%H:%M:%SZ")
        
        folder_name, file_name = self.get_local_file_path(deliverystart, deliveryend)
        

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
                    
        order_data.to_csv(folder_name + file_name, index=False)
        
                
        
    def get_raw_product_orders(self, deliverystart, deliveryend):
        
        folder_name, file_name = self.get_local_file_path(deliverystart, deliveryend)
        
        if os.path.isfile(folder_name + file_name):
            return pd.read_csv(folder_name + file_name)
        
        start = deliverystart.strftime('%Y-%m-%dT%H:%M:%SZ')
        end = deliveryend.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        query = f"""
                SELECT *
                FROM epex_orderbooks_raw_orders
                WHERE deliverystart = '{start}' AND deliveryend = '{end}'
                """
        
        
        database = "etlgluedb-pfncms8nqjjp"
        order_data =  wr.athena.read_sql_query(query, database)
        order_data.columns = self.columns
        
        self.store_product_orders(order_data)
        
        return order_data

    
    def get_file_name_for_date(self, year, month, day):

        str_month = str(month) if month > 9 else f"0{month}"
        str_day = str(day) if day > 9 else f"0{day}"
        
        folder = "epex-spot-data/de/ic/orders/" + str(year) + "/" + str_month
        files_in_s3 = [f.key.split(folder + "/")[1] for f in self.raw_bucket.objects.filter(Prefix=folder).all()]

        file_start = "Continuous_Orders-DE-" + str(year) + str_month + str_day
        
        for file in files_in_s3:
            if file.startswith(file_start):
                return  str(year) + "/" + str_month + "/" + file

        return None


    def get_order_data_unprocessed(self, year, month, day):
    
        s3_path = "s3://da-historical-raw-001/epex-spot-data/de/ic/orders/" + self.get_file_name_for_date(year, month, day)
        df = pd.read_csv(s3_path, compression='zip', skiprows=1)
        return df


    def get_order_data_processed(self, year, month, day, interval=(0,24)):

        day_folder = f"epex-spot/de/ic/orders/year={year}/month={month}/day={day}/"
        s3_path = "s3://da-historical-processed/" + day_folder
        
        # s3_path = "s3:///epex-spot/de/ic/orders/year=2024/month=1/day=19/"
        #files_in_s3 = [f.key.split(s3_path + "/")[1] for f in s3_bucket.objects.filter(Prefix=s3_path).all()]

        all_files = []
        for hour in range(interval[1]+1):
            hour_folder = day_folder + "hour=" + str(hour)
            files = self.processed_bucket.objects.filter(Prefix=hour_folder).all()
            
            for file in files:
                all_files.append(file.key[len(day_folder):])

        order_data = None
        for file in tqdm(all_files, desc=f"Downloading {year}-{month}-{day}", unit='item'):
            
            df = wr.s3.read_parquet(s3_path + file)
            
            if order_data is None:
                order_data = df
            else:
                order_data = pd.concat([order_data, df], ignore_index=True)
                
            
            
            
        order_data.columns = self.columns
        return order_data

        #pd.read_parquet(s3_path)
            #files_in_s3 = [f.key.split(hour_folder + "/")[1] for f in processed_bucket.objects.filter(Prefix=hour_folder).all()]
            #print(files_in_s3)

            
        
