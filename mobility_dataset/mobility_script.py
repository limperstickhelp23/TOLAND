import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Replace 'file.pkl' with the path to your pickle file
# mohawkap = pd.read_pickle('mobility_objects/mohawk_1000dev_100ap_wifi_lte_100m_s42.pkl')
# sample=pd.DataFrame(list(mohawkap.values())[0])


#NOTE: Inspecting the data I know between 5/10 and 5/20 we should have at least 100 farily-active devices
START_DATE="2020-05-10"
END_DATE="2020-05-20"

dvc = pd.read_csv('RVF_ATX_PID_HZ-2020-05.tsv', sep='\t')
venues = pd.read_csv('RVF_ATX_PID_HZ_Places_Lookup.tsv', sep='\t')
print(dvc.columns)
print(venues.columns)

def sort_dictionary():
    return

def find_top_100_devices():
    return

# Filter + Convert the combined string to a datetime object
data=dvc[["persistentid","local_date","local_hour","venueid"]]
data['datetime'] = pd.to_datetime(data['local_date'] + ' ' + data['local_hour'].astype(str) + ':00')

data=data[ (data.datetime <= START_DATE) & (data.datetime <= START_DATE)]

# NOTE: Top 100 From this Time Period
counts=dvc['persistentid'].value_counts()
top_100=pd.DataFrame(counts.iloc[0:100])
top_100.reset_index(inplace=True)
top_100["id"]=top_100.index
filtered = data[data['persistentid'].isin(set(top_100.persistentid))]
filtered = pd.merge(top_100[["id","persistentid"]],filtered,right_on="persistentid",left_on="persistentid")

# Enrich Filered Data:
verbose=["persistentid","id","local_date","local_hour","venueid", "city","state",'geolat', 'geolong', 'datetime']
data=pd.merge(filtered,venues,right_on="venueid", left_on ="venueid")[verbose]
data.to_csv("context.csv")
# print(data[["city","state", "datetime"]].head())
data.drop(columns=["local_date","local_hour","venueid", "city","state"],inplace=True)

def interploate_multiple_devices():
    #TODO
    return

def manually_find_best_starting_date(df,num=100):

    start_date = df.iloc[0].local_date
    end_date = df.iloc[-1].local_date

    encountered=[]
    date_list = pd.date_range(start=start_date, end=end_date,freq="h").tolist()

    for date in date_list:

        check=list(df[df.datetime == date].id)
        encountered.extend(check)
        encountered=list(set(encountered))

        if len(encountered)==100:
            for i in sorted(encountered):
                print(i)
            return date
    
def filter_on_time_of_day(start_hour=8,end_hour=19):
    return

def split_into_unique_id_frames(df,method="ffill"):

    """ Fill Methods:   "ffill",   "interpolate"
    """

    # Create a uniform datetime index for everyone | Set the datetime as index
    start_date=df.iloc[0].datetime # Best start date: 2020-05-07 11:00:00
    end_date=df.iloc[-1].datetime # max date
    df=df[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

    def interpolate(series): 
        return series.interpolate(method='linear')

    # os.mkdir("device_movements/",exists_ok=True)
    for dvc in range(100):
        
        sub=df.loc[df.id == dvc]
        sub.set_index('datetime', inplace=True)
        sub=sub.groupby(sub.index.floor('h')).agg(
            {
                "persistentid": 'last',
                'id': 'last',
                'geolat': 'mean',
                'geolong': 'mean'
            }
        )
        sub['is_active']=1.0
        active=sub['is_active']
        sub.drop(columns=["is_active","persistentid"],inplace=True)

        sub=sub.reindex(pd.date_range(start=start_date, end=end_date, freq='h'))
        # print(sub.head())
    
        # print("Before: \n" , sub.head(20))
        if method == "ffill":
            sub=sub.resample("h").ffill()
    
        if method == "interpolate":
            sub=sub.resample("h").interpolate(method="linear")
        # print("AFter Interpolated: \n",sub.head())
        
        
        sub=pd.merge(sub,active,how="left",left_index=True,right_index=True).fillna(0)
        sub.to_pickle(f"device_movements/log_{dvc}.pkl")


# Process/ Make Files
split_into_unique_id_frames(data,method="interpolate")



