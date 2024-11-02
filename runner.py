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
METRIC_PATH="scalefree_baseline/metrics/"
FIGPATH="scalefree_baseline/figures/"
CONFIG_NAME="scalefree"
SAVE_RESULTS=True
SAVE_FIGURES=True
STAR=True
DATA={}
DATA["cloud"]={"losses":[],"accuracies":[]}

MODE="_star"
CONFIG="network"
METRIC_PATH="baselines/star"
WORKERS=5
DATA={}
DATA["cloud"]={"losses":[],"accuracies":[]}

@hydra.main(config_path="configs", config_name=CONFIG, version_base=None)
def run(cfg:DictConfig):
    from baselines.star.cloud import cloud
    omegaconf.OmegaConf.to_yaml(cfg)
    cloud(cfg)

run()



# @hydra.main(config_path="configs", config_name=CONFIG_NAME, version_base=None)


# class DynFLRunner:

#     def __init__(self,ALGORITHM):

#         if ALGORITHM == "star_baseline":
#             self.algortihm = Baselines(topology="star")
        
    

#     def run_star():
        
#         METRIC_PATH="scalefree_baseline/metrics/"
#         FIGPATH="scalefree_baseline/figures/"
#         CONFIG_NAME="scalefree"
#         SAVE_RESULTS=True
#         SAVE_FIGURES=True
#         STAR=True
#         DATA={}
#         DATA["cloud"]={"losses":[],"accuracies":[]}


#         omegaconf.OmegaConf.to_yaml(cfg)
#         save_path = HydraConfig.get().runtime.output_dir

#         path = f"{METRIC_PATH}/run_{len(os.listdir(METRIC_PATH))}/"
#         os.makedirs(path, exist_ok=True)


#         trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

#         device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#         net_model = Net(cfg.num_classes)

#         for server_round in range(cfg.num_rounds):
#             print(colorama.Fore.LIGHTBLUE_EX+f'Starting server round {server_round}')
#             pool = ThreadPoolExecutor(max_workers=WORKERS)
#             results = []

#             futures = [pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
#                                 net_model.state_dict(), cfg.config_fit, device) for i in range(cfg.num_clients)]

#             for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients"):
#                 result = future.result()
#                 results.append(result)
#             pool.shutdown(wait=True)

#             params = [result[1] for result in results]
#             #NOTE: testing is this what makes diffference ? 
#             # if server_round != 0:
#             #     params.append(net_model.state_dict())

#             net_state_dict = aggregate_params(params)

#             net_model.load_state_dict(net_state_dict)

#             g_loss, g_accuracy = test(net_model, testloader, device)

#             with open(path+f'global_model_eval.txt', 'a') as file:




