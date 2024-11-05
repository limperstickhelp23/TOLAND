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

def map_of_austin_opmnx(fig,ax):
    """WARNING: 
        Will not load the larger long/latitiude range or is extremely slow

        Looking for other solutions (like screenshot of Folium view)
    """
    # Define the latitude and longitude range
    north, south, east, west = 31.00, 29.00, -96.5, -98.00

    north, south, east, west = 30.3, 30.25, -97.70, -97.75

    # Create a bounding box for the specified area
    bounding_box = (north, south, east, west)

    # Download the street network for the bounding box
    # graph = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    buildings = ox.geometries_from_bbox(north, south, east, west, tags={'building': True})
    water = ox.geometries_from_bbox(north, south, east, west, tags={'natural': 'water'})
    parks = ox.geometries_from_bbox(north, south, east, west, tags={'leisure': 'park'})

    # Plot buildings on the map
    buildings.plot(ax=ax, color='lightgray')
    parks.plot(ax=ax, color='lightgreen',alpha=.5)
    water.plot(ax=ax, color='skyblue',alpha=.8)
    # ox.plot_graph(graph, node_size=0, edge_color='black', ax=ax,show=False)
    print("stop showing")

    # plt.title('Buildings and Street Network of Austin, Texas')


def load_austin_screenshot(ax):

    img = mpimg.imread('austin.png')

    # Display the image
    ax.imshow(img, extent=[-98.5, -97., 29, 31])  # Adjust the extent based on your map bounds


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
        # print("coords: ", row['geolong'], row['geolat'])
        scatters.append(ax.scatter(row['geolong'], row['geolat'], c='blue', label=row['id'], alpha=0.7))
        labels.append(ax.annotate(int(row['id']), (row['geolong'], row['geolat']), textcoords="offset points", xytext=(0,5), ha='center'))
    # print(f"{datehour}, # actually active : {active_count}, # appeared in the dataset {len(which)}")
    ax.set_title(f"{datehour} Positions")
    return scatters,labels

def plot_all_hours(logs):
    
    start_date=logs[0].index[0]
    end_date=logs[0].index[-1]
    fig,ax=plt.subplots(figsize=(12, 8))
    # load_austin_screenshot(ax)
    xmin,xmax=logs[0]['geolong'].min(),logs[0]['geolong'].max()
    ymin,ymax=logs[0]['geolat'].min(),logs[0]['geolat'].max()
    ax.set_xlim(xmin - .5, xmax + .5)
    ax.set_ylim(ymin - .5, ymax + .5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid()
    # plt.axhline(0, color='black', lw=0.5, ls='--')  # Optional: horizontal line at y=0
    # plt.axvline(0, color='black', lw=0.5, ls='--')  # Optional: vertical line at x=0
    
    for (n,datehour) in enumerate(pd.date_range(start=start_date, end=end_date,freq="H").tolist()):
        if datehour.hour < START_HOUR or  (datehour.hour > STOP_HOUR):
            continue
        scatters,labels=plot_hour(datehour,logs,ax)
        plt.pause(.25)
        for lbl,scatter in zip(scatters,labels):
            scatter.remove()  # Remove each scatter from the axes
            lbl.remove()

        if n>200: 
            break

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

    plot_all_hours(logs)






# def plot_coordinates(df):
#     unique_hours = df['local_hour'].unique()

#     # Create a figure for plotting
#     plt.figure(figsize=(12, 8))

#     for hour in unique_hours:
#         hour_data = df[df['local_hour'] == hour]
#         plt.clf()  # Clear the current figure
#         plt.title(f'Hour: {hour}')

#         # Plot each device's coordinates
#         for index, row in hour_data.iterrows():
#             plt.scatter(row['geolong'], row['geolat'], c='blue', label=row['id'], alpha=0.7)
#             plt.annotate(row['id'], (row['geolong'], row['geolat']), textcoords="offset points", xytext=(0,5), ha='center')

#         plt.xlim(df['geolong'].min() - .05, df['geolong'].max() + 1)
#         plt.ylim(df['geolat'].min() - .05, df['geolat'].max() + 1)
#         plt.xlabel('Longitude')
#         plt.ylabel('Latitude')
#         plt.axhline(0, color='black', lw=0.5, ls='--')  # Optional: horizontal line at y=0
#         plt.axvline(0, color='black', lw=0.5, ls='--')  # Optional: vertical line at x=0
#         # plt.grid()
#         plt.pause(.5)  #




# def animation_dev():
#     # Create a scatter plot for the animation with 100 points
#     scat = ax.scatter([], [], color='red')

#     # Initialize the scatter plot and annotations
#     def init():
#         scat.set_offsets([])
#         for label in ax.texts:  # Remove any existing labels
#             label.remove()
#         return scat,

#     # Update the scatter plot for each frame
#     def update(frame):
#         # Get the current positions for all 100 points at the given frame
#         current_positions = movement_data[:, frame, :]  # Shape (100, 2)
#         scat.set_offsets(current_positions)  # Update the scatter point positions

#         # Clear previous annotations
#         for label in ax.texts:
#             label.remove()

#         # Annotate each point with its corresponding device ID
#         for idx, (x, y) in enumerate(current_positions):
#             ax.annotate(device_ids[idx], (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

#         return scat,

#     # Create the animation
#     ani = FuncAnimation(fig, update, frames=10, init_func=init, blit=True)
