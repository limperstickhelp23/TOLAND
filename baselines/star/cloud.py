import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pickle import EMPTY_DICT
from tqdm import tqdm

import numpy as np
import torch

from client import local_train, test
from data_prepare import prepare_dataset
from models import Net
import colorama
import json



#TODO: Would be nice if I coudl figure out how to make it more modular
#           without just copy pastign code everywhere


def aggregate_params(model_params):
    averaged_state_dict = {}

    for key in model_params[0].keys(): #conv1_
        param_stack = torch.stack([state_dict[key] for state_dict in model_params], dim=0)
        avg_params = torch.mean(param_stack, dim=0)
        averaged_state_dict[key] = avg_params
    return averaged_state_dict

def cloud(cfg):

    # TODO: add to configs probably
    METRIC_PATH="baselines/star"
    WORKERS=5
    DATA={}
    DATA["cloud"]={"losses":[],"accuracies":[]}

    path = f"{METRIC_PATH}/results/run_{len(os.listdir(METRIC_PATH))}/"
    os.makedirs(path, exist_ok=True)

    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    net_model = Net(cfg.num_classes)

    for server_round in range(cfg.num_rounds):
        
        print(colorama.Fore.LIGHTBLUE_EX+f'Starting server round {server_round}')
        pool = ThreadPoolExecutor(max_workers=WORKERS)
        results = []
        futures = [pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
                               net_model.state_dict(), cfg.config_fit, device) for i in range(cfg.num_clients)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients"):
            result = future.result()
            results.append(result)
        pool.shutdown(wait=True)

        params = [result[1] for result in results]
        #NOTE: testing is this what makes diffference ? 
        # if server_round != 0:
        #     params.append(net_model.state_dict())

        net_state_dict = aggregate_params(params)
        net_model.load_state_dict(net_state_dict)
        g_loss, g_accuracy = test(net_model, testloader, device)

        DATA["cloud"]["losses"].append(g_loss)
        DATA["cloud"]["accuracies"].append(g_accuracy)
        print(g_accuracy, " " ,g_loss)

    lgth=len(os.listdir(f"{METRIC_PATH}/results/"))
    with open(f"{METRIC_PATH}/results/run_{lgth}.json", "w") as file:
        json.dump(DATA, file, indent=4)
    
    return DATA



# if __name__ == '__main__':
#     cloud()
#     lgth=len(os.listdir(f"{METRIC_PATH}/results/"))
#     with open(f"{METRIC_PATH}/results/run_{lgth}.json", "w") as file:
#         json.dump(DATA, file, indent=4)

