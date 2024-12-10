import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# NOTE: Replace 'file.pkl' with the path to your pickle file
# mohawkap = pd.read_pickle('mobility_objects/mohawk_1000dev_100ap_wifi_lte_100m_s42.pkl')
# sample=pd.DataFrame(list(mohawkap.values())[0])

#NOTE: Inspecting the data I know between 5/10 and 5/20 we should have at least 100 farily-active devices
START_DATE="2020-05-01"
END_DATE="2020-05-20"

#NOTE Mohawk Box
BBOX = [-97.8395, -97.6819, 30.1961, 30.3511]


def prepare_dataset(num_devices=100):
    dvc = pd.read_csv('RVF_ATX_PID_HZ-2020-05.tsv', sep='\t') # print(dvc.columns) # print(venues.columns)
    venues = pd.read_csv('RVF_ATX_PID_HZ_Places_Lookup.tsv', sep='\t')
    
    # dvc processing
    data=dvc[["persistentid","local_date","local_hour","venueid"]]
    data['datetime'] = pd.to_datetime(data['local_date'] + ' ' + data['local_hour'].astype(str) + ':00')
    data=filter_on_dates(data)

    # Enrich with Geo Information
    keep=["persistentid","local_date","local_hour","venueid", "city","state",'geolat', 'geolong', 'datetime']
    data=pd.merge(data,venues,right_on="venueid", left_on ="venueid")[keep]

    # Filter to top N devices
    data=check_in_bounds(data)
    data=find_top_N_occurring_devices(data,N=num_devices)
    data.to_csv("context.csv")
    data.drop(columns=["venueid", "city","state"],inplace=True)
    
    # print(data[["geolat","geolong"]].head(10))
    print(data.persistentid.value_counts())
    print("Check N dvcs: ", data['persistentid'].nunique())

    return data

# def sort_dictionary():
#     return

def filter_on_dates(df,start=START_DATE,stop=END_DATE):
    return df[(start <= df.datetime) & (df.datetime <= stop)]


def filter_on_time_of_day(df,start_hour=8,end_hour=19):
    if 'datetime' not in df.columns:
        return df[(start_hour <= df.index.hour) & (df.index.hour <= end_hour)]
    else:
        return df[(start_hour <= df.datetime.dt.hour) & (df.datetime.dt.hour <= end_hour)]


def find_top_N_occurring_devices(data,N=100):
    top=pd.DataFrame(data['persistentid'].value_counts().iloc[0:N])  
    # top["id"]=[int(i) for i in range(N)]
    # print(top.head())
    # data=pd.merge(data, top, right_index=True, left_on="persistentid",how="left")
    # print(data.id.head(20))
    return data[data['persistentid'].isin(set(top.index))]

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

def check_in_bounds(df):
    leftbound = df['geolong'] >= BBOX[0]
    rightbound = df['geolong'] <= BBOX[1]
    upperbound = df['geolat'] >= BBOX[2]
    lowerbound = df['geolat'] <= BBOX[3]
    condition = leftbound & rightbound & upperbound & lowerbound
    df = df[condition]
    return df


# # NOTE: Top 100 From this Time Period
# counts=dvc['persistentid'].value_counts()
# top_100=pd.DataFrame(counts.iloc[0:100])
# top_100.reset_index(inplace=True)
# top_100["id"]=top_100.index
# filtered = data[data['persistentid'].isin(set(top_100.persistentid))]
# filtered = pd.merge(top_100[["id","persistentid"]],filtered,right_on="persistentid",left_on="persistentid")

# # Enrich Filered Data:
# verbose=["persistentid","id","local_date","local_hour","venueid", "city","state",'geolat', 'geolong', 'datetime']
# data=pd.merge(filtered,venues,right_on="venueid", left_on ="venueid")[verbose]
# data.to_csv("context.csv")
# # print(data[["city","state", "datetime"]].head())
# data.drop(columns=["local_date","local_hour","venueid", "city","state"],inplace=True)



def split_save_device_frames(df,start=START_DATE,stop=END_DATE,method="ffill"):
    """ Fill Methods:   "ffill",   "interpolate"
    """

    df=filter_on_time_of_day(df,start_hour=8,end_hour=19)
    devices=list(df.persistentid.value_counts().index)

    # os.mkdir("device_movements/",exists_ok=True)
    for (n,dvc) in enumerate(devices):

        sub=df.loc[df.persistentid == dvc]
        sub.set_index('datetime', inplace=True)
        sub=sub.groupby(sub.index.floor('h')).agg(
            {
                "persistentid": 'last',
                'geolat': 'mean',
                'geolong': 'mean'
            }
        )
        sub['is_active']=1.0
        active=sub['is_active']
        sub.drop(columns=["is_active","persistentid"],inplace=True)

        sub=sub.reindex(pd.date_range(start=start, end=stop, freq='h'))
        

        # print("Before: \n" , sub.head(20))
        if method == "ffill":
            sub=sub.resample("h").ffill()
    
        if method == "interpolate":
            sub=sub.resample("h").interpolate(method="linear").bfill()
   
        # print("AFter Interpolated: \n",sub.head())
        sub=filter_on_time_of_day(sub,start_hour=8,end_hour=19)
        print(sub.head(8)),print("\n\n")

        
        sub=pd.merge(sub,active,how="left",left_index=True,right_index=True).fillna(0)
        sub['id']=n
        sub.to_pickle(f"device_movements/log_{n}.pkl")


if __name__ =="__main__":
# Process/ Make Files
# split_into_unique_id_frames(data,method="interpolate")
    data=prepare_dataset(num_devices=100)
    split_save_device_frames(data,start=START_DATE,stop=END_DATE,method="interpolate")





