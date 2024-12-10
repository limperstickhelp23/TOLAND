import matplotlib.gridspec as gridspec
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import matplotlib.image as mpimg

CMAP="jet"
# CMAP="viridis"
# CMAP="magma"
# CMAP="tab20"
# CMAP="Paired"
CMAP="Accent"
# CMAP="turbo"
# CMAP="Set3"

MINLON=-98.4739
MAXLON=-96.901
MINLAT=29.64
MAXLAT=30.86
PAD=0.033
EDGE_ALPHA=.8


def load_austin_screenshot(ax):
    img = mpimg.imread('austin.png')
    ax.imshow(img, extent=[MINLON,MAXLON, MINLAT, MAXLAT])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def load_gexf_files(directory):
    gexf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".gexf")]
    gexf_files.sort()  # Ensure a consistent order (e.g., by name)
    graphs = []
    for (n,file) in enumerate(gexf_files):
        if n%2==0:
            continue
        G = nx.read_gexf(file)
        graphs.append((os.path.basename(file), G))
    return graphs

def draw_frame_with_histogram(graph, ax_graph, ax_hist, title, fig, map=True):
    ax_graph.clear()
    ax_hist.clear()

    if (map):
        load_austin_screenshot(ax_graph)
        PAD=0

    ax_graph.spines['top'].set_visible(False)
    ax_graph.spines['right'].set_visible(False)
    ax_graph.spines['bottom'].set_visible(False)
    ax_graph.spines['left'].set_visible(False)
    
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.spines['bottom'].set_visible(False)
    ax_hist.spines['left'].set_visible(False)
    
    m=min(data['lat'] for _, data in graph.nodes(data=True))
    mx=max(data['lat'] for _, data in graph.nodes(data=True))
    # Set axis limits
    ax_graph.set_xlim(MINLON - PAD, MAXLON + PAD)
    ax_graph.set_ylim(MINLAT - PAD, MAXLAT + PAD)

    # Set aspect ratio to equal to ensure nodes are proportionally placed
    ax_graph.set_aspect('equal')

    all_communities = Counter()
    for _, data in graph.nodes(data=True):
        community = data.get('gephi_community')
        if community is not None:
            all_communities[community] += 1

    # Sort communities by total node count and assign consistent colors
    sorted_communities = [comm for comm, _ in all_communities.most_common()]
    color_map = plt.cm.get_cmap(CMAP, len(sorted_communities))
    community_colors = {comm: color_map(i) for i, comm in enumerate(sorted_communities)}

    # Extract positions
    pos = {node: (data['lon'], data['lat']) for node, data in graph.nodes(data=True)}
    ap_nodes = [(node,data) for node, data in graph.nodes(data=True) if data.get('is_ap', True)]
    non_ap_nodes = [(node,data) for node, data in graph.nodes(data=True) if not data.get('is_ap', False)]
    ap_pos = {node: pos[node] for node,_ in ap_nodes}
    non_ap_pos = {node: pos[node] for node,_ in non_ap_nodes}

    for u, v in list(graph.edges()):
        if graph.nodes[u].get('gephi_community') != graph.nodes[v].get('gephi_community'):
            graph.remove_edge(u, v)
    
    # Get edge colors based on source node
    edge_colors = [community_colors.get(graph.nodes[u].get('gephi_community'), 'gray') for u, v in graph.edges()]

    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, ax=ax_graph, edge_color=edge_colors, alpha=EDGE_ALPHA, width=0.67)
    
    # Draw nodes
    # nx.draw_networkx_nodes(graph, pos, ax=ax_graph, node_color=node_colors, node_size=30)
    
    # Non-AP nodes: smaller size, default circle shape
    ax_graph.scatter(
        [x for x, y in non_ap_pos.values()],
        [y for x, y in non_ap_pos.values()],
        s=120,  # Smaller size
        c=[community_colors.get(data.get('gephi_community'), 'gray') for _,data in non_ap_nodes],  # Color for non-AP nodes
        marker='o',  # Circle shape
        label='Non-AP Nodes',  # for node in non_ap_nodes
        edgecolors='black'
    )
    #NOTE: AP nodes AFTER for more emphasis
    ax_graph.scatter(
        [x for x, y in ap_pos.values()],
        [y for x, y in ap_pos.values()],
        s=1500,  # Larger size
        c=[community_colors.get(data.get('gephi_community'), 'gray') for _,data in ap_nodes],  # Color for AP nodes
        marker='*',  # Star shape
        label='AP Nodes',
        edgecolors='black'
    )
    
    ## Add labels (optional)
    # nx.draw_networkx_labels(graph, pos, ax=ax_graph, font_size=8)
    
    # Title for the graph
    ax_graph.set_title(f"{title}",fontsize=28)
    ax_graph.set_aspect('equal')

    # Histogram for community counts
    # community_counts = Counter(data.get('gephi_community') for _, data in graph.nodes(data=True))
    bar_colors = [community_colors[comm] for comm in community_colors.keys()]
    
    bars=ax_hist.barh(sorted([i+1 for i in sorted_communities]), sorted(all_communities.values(),reverse=True), color=bar_colors)
    # Add the counts on top of each bar
    for bar in bars:
        width = bar.get_width()  # Get the width of the bar (count)
        ax_hist.text(width + 0.1, bar.get_y() + bar.get_height() / 2,  # Adjust position slightly to the right of the bar
                     str(width), va='center', ha='left', fontsize=10, color='black')

    # ax_hist.set_xlabel("Member Counts",fontsize=16)
    # ax_hist.set_ylabel("Top 5 Descending",fontsize=16)
    ax_hist.set_xticks([])
    ax_hist.set_yticks([])
    # ax_hist.set_title("Community Distribution",fontsize=20)
    ax_hist.set_xlim([0,90])
    fig.tight_layout()
    

def animate_graph_with_histogram(graphs, output_file="dynamic_graph_with_histogram.mp4"):
    # Load all graphs
    # graphs = [load_gexf_files(gephi_files)]
    names,graphs= zip(*graphs)
    # community_colors = prepare_community_colors(graphs)

    # Setup figure and axes
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.2])
    ax_graph = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    def update(idx):
        graph = graphs[idx]
        draw_frame_with_histogram(graph, ax_graph, ax_hist, title=f"Frame {idx+1}",fig=fig)

    ani = FuncAnimation(fig, update, frames=len(graphs), repeat=False)
    
    # Save animation
    ani.save(output_file, writer='ffmpeg', fps=1.5)
    print(f"Animation saved to {output_file}")

def static_community_plots(graphs, indices):
    names,graphs= zip(*graphs)
    for idx in indices:
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.2])
        ax_graph = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
        graph=graphs[idx]
        draw_frame_with_histogram(graph, ax_graph, ax_hist, title=f"Frame {idx+1}",fig=fig)
        plt.savefig(f"comm_plot_round_{idx}")
    return

if __name__ == "__main__":
    # gephi_files = ["frame1.gexf", "frame2.gexf", "frame3.gexf"]  # List of GEXF files
    gephi_path="gephi/DPP_CIFAR"
    output_animation = "dynamic_graph_with_histogram.mp4"
    graphs = load_gexf_files(gephi_path)
    
    #NOTE Animation
    # animate_graph_with_histogram(graphs, output_animation)
    
    #NOTE Static Plotter
    # static_community_plots(graphs, indices=[0,1,2,3,4,5,6,7,8,9,19,29]) # indices=server round index
