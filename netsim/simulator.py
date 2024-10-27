from algorithms import *
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

NUM_DEVICES=100
ROUNDS=20
FPS=2
SAVE=False
PLOTOP=2
PROXIMITY=5
DISPERSION=20
NUM_APS=5

#NOTE: check do I have an edge case issue with the 0 node ?
#NOTE: TODO -- Min Span Tree With Girvan-Newman or Louvain/ Modularity too

#NOTE: TODO
# - Work on Implementing Algorithms with the Training Scheme Dynamically
# - Test Self-Organizing Community Assignment Problem
#     --  Mainly just see if I can get devices to correctly find the right groups on static topology (DONE)


def run_rounds(server: MobileNet, animate=True):
    if (animate==True):
        run_simulate_animate(server,fname="")
        # animate_nodes(server)
    else:
        for rnd in range(ROUNDS):
            run_round(server,round=rnd)
    return

def run_round(server,round,folder=""):
    server.move()
    server.rewire_round(round_num=round)
    server.calculate_topology_cost()
    if (SAVE):
        server.assignments_to_yaml(os.path.join(server.outdir,folder+f"_round_{round}.yaml"))
        with open(os.path.join(server.outdir,folder+"costs.txt"), "a") as file:
            file.write(f"Round {round + 1} Cost: {server.topology_cost}\n")
        if (server.mobile == True) or ((server.mobile == False) and (round == 0)):
            server.positions_to_yaml(os.path.join(server.outdir,folder+f"coordinates_round_{round}.yaml"))

def run_simulate_animate(server,fname="",top=2):
    fig, ax = plt.subplots(figsize=(12,10)) # brew install ffmpeg
    ax.set_xlim(-100, 100)  # Set x-axis limits
    ax.set_ylim(-100, 100)  # Set y-axis limits
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    with writer.saving(fig, "animation.mp4", dpi=100):
        for rnd in range(ROUNDS):
            run_round(server,round=rnd)
            server.plot_topology(PLOTOP)
            plt.title(f"Global Round: {rnd}", fontsize=20)
            writer.grab_frame()
            plt.clf()

def animate_nodes(server,fname="",top=2):
    fig, ax = plt.subplots(figsize=(12,10)) # brew install ffmpeg
    ax.set_xlim(-100, 100)  # Set x-axis limits
    ax.set_ylim(-100, 100)  # Set y-axis limits
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    with writer.saving(fig, "animation.mp4", dpi=100):
        for rnd in range(ROUNDS):
            run_round(server,round=rnd)
            plt.title(f"Global Round: {rnd}", fontsize=20)
            server.plot_nodes(PLOTOP)
            writer.grab_frame()
            # server.plot_edges(PLOTOP)
            # writer.grab_frame()
            plt.clf()

if __name__ == "__main__":
    
    ## (1) Scale Free Topologies
    #-------------------------------#
    ## Static Devices, Scale Free Tops at every round
    # static_sf = StaticServer(num_devices=NUM_DEVICES)
    # run_rounds(static,animate=True)
   
    ## Dynamic Devices, Scale Free Tops at every round
    # dynamic_sf = StaticServer(num_devices=NUM_DEVICES)
    # run_rounds(dynamic_sf,animate=True)
    
    ## (2) Effcient Paper
    #-------------------------------#
    efficient=EfficientLessCentralizedServer(num_devices=NUM_DEVICES)
    run_rounds(efficient,animate=True)
    
    
    ## (3) MinSpan Tree Algorithm
    #-------------------------------#
    # minspan =MinSpanTreeServer(num_devices=NUM_DEVICES,λ=DISPERSION)
    # minspan.num_apoints=NUM_APS
    # minspan.mobile=True
    # minspan.method="fully_connected"
    # # minspan.method="proximity"
    # minspan.threshold=PROXIMITY
    # run_rounds(minspan,animate=True)

    ## (4) Self-Assignment
    #-------------------------------#
    # cosine=CosineReassignment()
    # TODO -- fill in details