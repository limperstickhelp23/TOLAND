# TOLAND
Adaptive Topology structures for Federated Learning


## Changes:

### *NetSim Folder*
- Network Class
    - Purpose is to be a coordinator between:
        - (1) Mobility Sim 
        - (2) Network Communication Cost 
        - (3) Device Activity (logs devices) 
        - (4) Training Loop/Progress 
        - (5) Plotting Utilities

    - Stores a list of Device Objects, which used to store copies of models
        - Change we discussed: get rid of device models to instead point to files where they keep their weights 
        - Therefore the Device object is a conveniences data structure to store some meta data for coordinating (i.e. path to its parameters, id, possibly a pointer to it's current access point, route to its AP, etc.)
        - Might also add some methods at Device level like "load_model_parameters()" but this could be done outside too
- Algorithm
    - Currently extends the Network class to instantiate the different custom policies including: ScaleFree Baseline & Cosine Similarity
    - There are some common mehthods defined at Algorithm level
    - Also stores a map of AP: [ list of members ] which could be dynamic

- Customs
    - Extend Algorithm
    - Would be nice for a common interface (TODO) i.e. each custom algorithm must implement two callable methods:
        - "global_round" 
        - "local_round"


Lastly all the plotting stuff is stored in this folder right now. Don't worry too much about that. I'll see if I can maybe separate that out into a totally separate process which would allow us to clean up some things.

For example "reset_colors()" is purely for visualizing

### Results Folder:
- Structure changed -- check to see the difference

### "Plotter" Function
- Can Ignore
- Can Run "mobility_plotter.py" to playback device movements ( austin map is still WIP)


### Device Movements
- Comes from processing the FourSquare data
- These files store the full simulated movements of devices 1 to 100 
- Date Range: 5/10 to 5/20/2020





## MileStone 2 Efforts
- Implement Dynamics using the mobility (mohawk) dataset
    - Allen recommended stitching together multiple days of data -- TODO

- "Static" Baselines (results without device mobility)
- IID Tests
    - Star Topology -- 100 devices -- DONE
    - Scalefree -- 5 seeds same topology on 100 devices -- DONE
    - Scalefree Alternating -- TODO 
    - Proximity + Cosine Comparison -- TODO

- Non-IID Tests (build a way to visualize this ?)
    - Star Topology -- 100 devices -- TODO
    - Scalefree -- 5 seeds same topology on 100 devices --  TODO
    - Scalefree Alternating - TODO

- CIFAR Data still to do (better computing resources needed ?)

- Equations for Cost of Communication -- TODO
    - distance^2 currently being used
