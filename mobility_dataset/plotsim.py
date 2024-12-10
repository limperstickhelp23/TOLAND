import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
# from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from pathlib import Path

# Choose Hours
START_HOUR=8
STOP_HOUR=19
MAPIMG='austin_map.png'
ASPECT=688/1160 # NOTE: set manually
ROOT=Path(__file__).resolve().parent.parent
MOVEMENTS=os.path.join(ROOT,"devices/movements")
BBOX = [-97.8395, -97.6819, 30.1961, 30.3511]  #NOTE: Same as MOHAWK Box


def load_austin_screenshot(ax):  
    img = mpimg.imread(MAPIMG)
    ax.imshow(img, aspect=ASPECT, extent=[BBOX[0],BBOX[1], BBOX[2], BBOX[3]])

def plot_hour(datehour,logs,ax):
    active_count=0
    which=[]
    scatters,labels=[],[]
    for (k,log) in logs.items():
        try:
            row=log.loc[datehour]
            active_count+=row.is_active
            which.append(k)
        except (KeyError):
            continue
        scatters.append(ax.scatter(row['geolong'], row['geolat'], c='blue', label=row['id'], alpha=0.7))
        labels.append(ax.annotate(int(row['id']), (row['geolong'], row['geolat']), textcoords="offset points", xytext=(0,5), ha='center'))
    ax.set_title(f"{datehour} Positions")
    return scatters,labels

def plot_all_hours(logs):
    
    start_date=logs[0].index[0]
    end_date=logs[0].index[-1]
    _,ax=plt.subplots(figsize=(12, 12*(ASPECT)))
    load_austin_screenshot(ax)
    ax.set_xlim(BBOX[0],BBOX[1])
    ax.set_ylim(BBOX[2], BBOX[3])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    for (n,datehour) in enumerate(pd.date_range(start=start_date, end=end_date,freq="H").tolist()):
        if datehour.hour < START_HOUR or  (datehour.hour > STOP_HOUR):
            continue
        scatters,labels=plot_hour(datehour,logs,ax)
        plt.pause(.20)
        for lbl,scatter in zip(scatters,labels):
            scatter.remove()  # Remove each scatter from the axes
            lbl.remove()
        if n>200: 
            break

if __name__ == "__main__":
    logs={}
    for n in range(100):
        df=pd.read_pickle(f"{MOVEMENTS}/log_{n}.pkl")
        try:
            logs[n]=df[(df.index.hour >= START_HOUR) & (df.index.hour <= STOP_HOUR)]
        except:
            print(n, " failed")
            continue
    plot_all_hours(logs)