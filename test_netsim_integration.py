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

#SET
METRIC_PATH="metrics/cosine_test/"
CONFIG_NAME="network"

# Algorithms
# algorithm=CosineReassignment(num_devices=25)
# algorithm=MinSpanTreeServer(num_devices=10)
# algorithm=StaticServer(num_devices=10)
# algorithm=EfficientLessCentalized(num_devices=10)

"""
NOTE: STILL WIP!! Testing the full pipeline to chekc data passing properly

"""


@hydra.main(config_path="configs", config_name=CONFIG_NAME, version_base=None)
def cloud(cfg:DictConfig):
    omegaconf.OmegaConf.to_yaml(cfg)
    
    # save_path = HydraConfig.get().runtime.output_dir  # NOTE: or put paths in config files
    path = f"{METRIC_PATH}/run_{len(os.listdir(METRIC_PATH))}/"
    os.makedirs(path, exist_ok=True)
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size, iid=cfg.iid)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    NETWORK=CosineReassignment(num_devices=25, num_classes=cfg.num_classes)
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

        # NOTE: possible reassignment/relinking steps can go here
        #NETWORK.move()
        #NETWORK.select_access_points_on_betweenness()
        #access_points=NETWORK.curr_apoints

        for aggr_round in range(aggregation_rounds):
            
            # Run a Local Round
            pool=ThreadPoolExecutor(max_workers=10)
            futures=[pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
                                   global_state_dict, cfg.config_fit, device) for i in range(cfg.num_clients)]
            results=[ future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients")]
            pool.shutdown(wait=True)
            print(results[0].keys())


            # NOTE: NEW -- Prior to Local Aggregation, Run the Reassignment Step
            NETWORK.map_results(results)
            NETWORK.compare_communities(max_iters=3)
            # NETWORK.assign_communities() # Assignment is inside (at the end of) compare_communities()
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


if __name__ == '__main__':
    cloud()

