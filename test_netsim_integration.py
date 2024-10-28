from netsim.network import MobileNet
from netsim.algorithms import *
from data_prepare import *
from utils import *

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pickle import EMPTY_DICT

from tqdm import tqdm

import hydra
import omegaconf
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from client import local_train, test
from data_prepare import prepare_dataset
from models import Net
import colorama

#SETTINGS
METRIC_PATH="cosine_demo/metrics"
FIGPATH="cosine_demo/figures/"
CONFIG_NAME="network"
SAVE_RESULTS=True
SAVE_FIGURES=True

@hydra.main(config_path="configs", config_name=CONFIG_NAME, version_base=None)
def cloud(cfg:DictConfig):
    omegaconf.OmegaConf.to_yaml(cfg)
    
    # save_path = HydraConfig.get().runtime.output_dir  # NOTE: or put paths in config files
    path = f"{METRIC_PATH}/run_{len(os.listdir(METRIC_PATH))}/"
    figpath=f"{FIGPATH}/run_{len(os.listdir(METRIC_PATH))}/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(figpath, exist_ok=True)
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size, iid=cfg.iid)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    NETWORK=CosineReassignment(num_devices=cfg.num_clients, num_classes=cfg.num_classes,perceptual_map=cfg.plot_colormap)
    aggregation_rounds = cfg['aggregation_rounds']
    ap_avg_state_dict,global_state_dict=None,None
    global_model = Net(cfg.num_classes)

    # NOTE: all of these things using configs currently are stored in the Algorithm Objects 
    # TODO: decide if we deprecate(?) -- algorithm dynamically updates 
    # access_points = cfg.neighbor_lists['AP']
    # ap_routes = {AP: cfg.neighbor_lists[AP].route for AP in access_points} # TODO: deprecate(?) can store in NET object
    # access_points=NETWORK.curr_apoints 

    for server_round in range(cfg.num_rounds):
        print(colorama.Fore.LIGHTBLUE_EX+f'Starting server round {server_round}')

        # NOTE: global reassignment/relinking/device movement steps can go here
        # If dynamic network, add new logic for using old communities to assign new ones
        #NETWORK.move()
        #NOTE/TODO : also should we send the global model back to community AP nodes ?

        if server_round == 0: # Initial Round Only / Static Case Builds Proximity Graph Once
            NETWORK.threshold=10
            NETWORK.build_proximity_graph() # allow_isolates = False
            NETWORK.select_access_points_on_betweenness() 
            NETWORK.init_community_models()
        print(
        "\nCheck Access Point Assignments\n" +
            "\nCommunities:\n" +"\n".join([f"{k} : {v}" for (k,v) in NETWORK.ap_member_map.items()]) + 
                "\n\nDevice to Parent Node Map:\n"+"\n".join([f"{d.id}: {d.parent_point}" for d in NETWORK.device_list])
        )#input("\nHit Enter to Continue:\n")

        # Demo/Monitor Self-Assignment Process
        if SAVE_FIGURES:
            NETWORK.plot_communities(spring=True)
            plt.title(f"Round {server_round} Communities")
            plt.savefig(figpath+f"{server_round}.png")

        for aggr_round in range(aggregation_rounds):
            
            # Run a Local Round
            pool=ThreadPoolExecutor(max_workers=10)
            futures=[pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
                                   global_state_dict, cfg.config_fit, device) for i in range(cfg.num_clients)]
            results=[ future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients")]
            pool.shutdown(wait=True)

            # NOTE: NEW -- Prior to Local Aggregation, Run the Reassignment Step
            NETWORK.map_results(results)
            NETWORK.compare_communities(max_iters=3)  # Re-assignment is inside (at the end of) compare_communities()
            # NETWORK.integrated_local_aggregation_round() # Specific to Cosine Compare Algorithm

            #NOTE: aggregation in the final step -- local_aggregation_round, maps community models internally. It also returns averages
            ap_avg_state_dict = NETWORK.local_aggregation_round()
            get_ap_metrics(ap_avg_state_dict, path, server_round, aggr_round, Net(cfg.num_classes), validationloaders, device)

        # Process Server Round Results
        global_state_dict = aggregate_params(list(ap_avg_state_dict.values()))
        global_model.load_state_dict(global_state_dict)
        g_loss, g_accuracy = test(global_model, testloader, device)

        with open(path+f'global_model_eval.txt', 'a') as file:
            file.write(f'round {server_round}: \n loss: {g_loss} \n accuracy: {g_accuracy}\n')
        #TODO: save in dataframe or some other more flexible format


if __name__ == '__main__':
    cloud()

