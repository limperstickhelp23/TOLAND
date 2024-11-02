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
METRIC_PATH="cosine_demo/metrics/"
FIGPATH="cosine_demo/figures/"
CONFIG_NAME="cosine"
SAVE_RESULTS=True
SAVE_FIGURES=True
STAR_BASELINE=True
DATA={}
DATA["cloud"]={"losses":[],"accuracies":[]}

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
    aggregation_rounds = cfg['aggregation_rounds']
    ap_avg_state_dict,global_state_dict=None,None
    global_model = Net(cfg.num_classes)

    NETWORK=CosineReassignment(num_devices=cfg.num_clients, num_classes=cfg.num_classes,perceptual_map=cfg.plot_colormap)

    for server_round in range(cfg.num_rounds):
        
        print(colorama.Fore.LIGHTBLUE_EX+f'Starting server round {server_round}')
        DATA[server_round]={}
        if server_round == 0: # Initial Round Only / Static Case Builds Proximity Graph Once
            NETWORK.threshold=10
            NETWORK.build_proximity_graph() # allow_isolates = False
            NETWORK.select_access_points_on_betweenness() 
            NETWORK.init_community_models()
        print("\nCheck Access Point Assignments\n" + "\nCommunities:\n" +"\n".join([f"{k} : {v}" for (k,v) in NETWORK.ap_member_map.items()]))    # Demo/Monitor Self-Assignment Process
        
        if SAVE_FIGURES:
            print(NETWORK.colormap)
            NETWORK.plot_communities(spring=True)
            plt.title(f"Round {server_round} Communities")
            plt.savefig(figpath+f"{server_round}.png")

        for aggr_round in range(aggregation_rounds):
            
            pool=ThreadPoolExecutor(max_workers=10)
            futures=[pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
                                    global_state_dict, cfg.config_fit, device) for i in range(cfg.num_clients)]
            results=[ future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients")]
            pool.shutdown(wait=True)
            
            if STAR_BASELINE == True:
                print(results[0])
                ap_avg_state_dict={res[0]:res[1] for res in results}
                continue
            else:
                # NOTE: NEW -- Prior to Local Aggregation, Run the Reassignment Step
                NETWORK.map_results(results)
                NETWORK.compare_communities(max_iters=3)  
                ap_avg_state_dict = NETWORK.local_aggregation_round()
                DATA[server_round][aggr_round]=update_ap_metrics(ap_avg_state_dict, Net(cfg.num_classes), validationloaders, device)

        # Process Server Round Results
        global_state_dict = aggregate_params(list(ap_avg_state_dict.values()))
        global_model.load_state_dict(global_state_dict)
        g_loss, g_accuracy = test(global_model, testloader, device)

        DATA["cloud"]["losses"].append(g_loss)
        DATA["cloud"]["accuracies"].append(g_accuracy)


if __name__ == '__main__':
    cloud()
    path = f"{METRIC_PATH}/run_{len(os.listdir(METRIC_PATH))}_"
    with open(path+"model_eval.json", "w") as file:
        json.dump(DATA, file, indent=4)
    

