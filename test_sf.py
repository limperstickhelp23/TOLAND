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
CONFIG_NAME="scalefree"
SAVE_RESULTS=True
WORKERS=10
DYNAMIC=True


@hydra.main(config_path="configs", config_name=CONFIG_NAME, version_base=None)
def cloud(cfg:DictConfig):

    omegaconf.OmegaConf.to_yaml(cfg)

    if (SAVE_RESULTS):
        os.makedirs(f"{cfg.figure_path}/run_{len(os.listdir(cfg.metric_path))}/", exist_ok=True)
    
    DATA={}
    DATA["cloud"]={"losses":[],"accuracies":[]}

    aggregation_rounds = cfg['aggregation_rounds']
    ap_avg_state_dict,global_state_dict=None,None

    net_model = Net(cfg.num_classes)
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size, iid=cfg.iid)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    NETWORK=Baselines(topology="scalefree", num_devices=cfg.num_clients, num_classes=cfg.num_classes,perceptual_map=cfg.plot_colormap)
    NETWORK.num_apoints=min(NETWORK.sf_seeds,5)
    if (DYNAMIC):
        NETWORK.dynamic=True

    for server_round in range(cfg.num_rounds):

        NETWORK.rewire_round()
        DATA[server_round]={}
    
        print(colorama.Fore.LIGHTBLUE_EX+f'Starting server round {server_round}')
        pool = ThreadPoolExecutor(max_workers=WORKERS)
        results = []

        futures = [pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
                               net_model.state_dict(), cfg.config_fit, device) for i in range(cfg.num_clients)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients"):
            result = future.result()
            results.append(result)
        pool.shutdown(wait=True)
        
        aggr_round=0
       
        # Locally Pass to AP's
        NETWORK.map_results(results)
        ap_avg_state_dict = NETWORK.local_aggregation_round()
        print(ap_avg_state_dict.keys())
        DATA[server_round][aggr_round]=update_ap_metrics(ap_avg_state_dict, Net(cfg.num_classes), validationloaders, device)
        net_state_dict = aggregate_params(list(ap_avg_state_dict.values()))

        #NOTE: Evaluate
        net_model.load_state_dict(net_state_dict)
        g_loss, g_accuracy = test(net_model, testloader, device)
        DATA["cloud"]["losses"].append(g_loss)
        DATA["cloud"]["accuracies"].append(g_accuracy)

    if (SAVE_RESULTS):
        lgth=len(os.listdir(f"{cfg.metric_path}/"))
        with open(f"{cfg.metric_path}/run_{lgth}.json", "w") as file:
            json.dump(DATA, file, indent=4)


if __name__ == '__main__':
    cloud()
        


# else:
#     aggr_round=0
#     print("\nCheck Access Point Assignments\n" +
#         "\nCommunities:\n" +"\n".join([f"{k} : {v}" for (k,v) in NETWORK.ap_member_map.items()]))
#     NETWORK.map_results(results)
#     ap_avg_state_dict = NETWORK.local_aggregation_round()
#     DATA[server_round][aggr_round]=update_ap_metrics(ap_avg_state_dict, Net(cfg.num_classes), validationloaders, device)
#     global_state_dict = aggregate_params(list(ap_avg_state_dict.values()))
