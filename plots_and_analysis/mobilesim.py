import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.animation import FuncAnimation
import osmnx as ox
import matplotlib.image as mpimg

# Choose Hours
START_HOUR=8
STOP_HOUR=19

MINLON=-98.4739
MAXLON=-96.901
MINLAT=29.64
MAXLAT=30.86

### TODO: Make Sure to Point this to Proper Files !!! ############



def load_austin_screenshot(ax):
    img = mpimg.imread('austin.png')
    ax.imshow(img, extent=[MINLON,MAXLON, MINLAT, MAXLAT])


def plot_hour(datehour,logs,ax,fig):

    ax.clear()
    load_austin_screenshot(ax)
    ax.set_xlim(MINLON, MAXLON)
    ax.set_ylim(MINLAT, MAXLAT)
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    ax.set_xticks([])
    ax.set_yticks([])

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
        # print("coords: ", row['geolong'], row['geolat'])
        scatters.append(ax.scatter(row['geolong'], row['geolat'], c='blue', label=row['id'], alpha=0.7))
        labels.append(ax.annotate(int(row['id']), (row['geolong'], row['geolat']), textcoords="offset points", xytext=(0,5), ha='center'))
    # print(f"{datehour}, # actually active : {active_count}, # appeared in the dataset {len(which)}")
    ax.set_title(f"{datehour} Positions",fontsize=28)
    fig.tight_layout()

def animate_all_hours(logs,output_file="device_mobility_austin.mp4"):
    
    start_date=logs[0].index[0]
    end_date=logs[0].index[-1]

    fig,ax=plt.subplots(figsize=(14, 11))

    hours_to_plot=[]
    for (n,datehour) in enumerate(pd.date_range(start=start_date, end=end_date,freq="H").tolist()):
        if datehour.hour < START_HOUR or  (datehour.hour > STOP_HOUR):
            continue
        hours_to_plot.append(datehour)
        if n > 120:
            break

    def update(idx):
        datehour = hours_to_plot[idx]
        plot_hour(datehour,logs,ax,fig)
        # for lbl,scatter in zip(scat,lab):
        #     scatter.remove()  # Remove each scatter from the axes
        #     lbl.remove()

    ani = FuncAnimation(fig, update, frames=len(hours_to_plot), repeat=False)
    ani.save(output_file, writer='ffmpeg', fps=1.75)
    print(f"Animation saved to {output_file}")
    
    # plt.axhline(0, color='black', lw=0.5, ls='--')  # Optional: horizontal line at y=0
    # plt.axvline(0, color='black', lw=0.5, ls='--')  # Optional: vertical line at x=0
        
## NOTE:
# Based on the Folium Pictures -- this data is all the way in like separate cities -- Austin isn't relaly beyond latittdue 30.3
# TODO (?) Pick new devices based on this ? Top 100 and they also have to be inside the boundaries ??


if __name__ == "__main__":
    
    logs={}
    for n in range(100):
        df=pd.read_pickle(f"device_movements/log_{n}.pkl")
        try:
            logs[n]=df[(df.index.hour >= START_HOUR) & (df.index.hour <= STOP_HOUR)]
        except:
            print(n, " failed")
            continue
    
    animate_all_hours(logs)

