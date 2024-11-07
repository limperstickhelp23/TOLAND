import copy
import argparse
import pickle

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

from deprecate.client import local_train, test
from deprecate.data_prepare import prepare_dataset
from deprecate.models import Net
import colorama


# NOTE: Possibly delete this -- use structure in  "cloud_template.py" to be able to call every algorthm from single script


@hydra.main(config_path="configs", config_name="network", version_base=None)
def cloud(cfg:DictConfig):
    omegaconf.OmegaConf.to_yaml(cfg)
    save_path = HydraConfig.get().runtime.output_dir

    run = cfg['run_id']

    path = f"metrics/run_{run}/"
    os.makedirs(path, exist_ok=True)

    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size, iid=cfg.iid)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    net_model = Net(cfg.num_classes)
    #Dictionary of {Access Point : neighbors to aggregate from}
    access_points = cfg.neighbor_lists['AP']
    aggregation_rounds = cfg['aggregation_rounds']
    ap_routes = {AP: cfg.neighbor_lists[AP].route for AP in access_points}
    ap_avg_state_dict = None
    net_state_dict = None
    for server_round in range(cfg.num_rounds):
        print(colorama.Fore.LIGHTBLUE_EX+f'Starting server round {server_round}')
        for aggr_round in range(aggregation_rounds):
            pool = ThreadPoolExecutor(max_workers=10)
            results = []

            futures = [pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
                                   net_state_dict, cfg.config_fit, device) for i in range(cfg.num_clients)]

            #get_parameters(ap_avg_state_dict, ap_routes, i)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients"):
                result = future.result()
                results.append(result)
            pool.shutdown(wait=True)

            ap_avg_state_dict = ap_aggregate(results, ap_routes) # {AP: avg_parameter(state_dict)}
            get_ap_metrics(ap_avg_state_dict, path, server_round, aggr_round, Net(cfg.num_classes), validationloaders, device)
            avg_params = list(ap_avg_state_dict.values())


        avg_params = list(ap_avg_state_dict.values())
        # if server_round != 0:
        #     avg_params.append(net_model.state_dict())
        net_state_dict = aggregate_params(avg_params)

        net_model.load_state_dict(net_state_dict)

        g_loss, g_accuracy = test(net_model, testloader, device)

        with open(path+f'global_model_eval.txt', 'a') as file:
            file.write(f'round {server_round}: \n loss: {g_loss} \n accuracy: {g_accuracy}\n')


def aggregate_params(model_params):
    averaged_state_dict = {}

    for key in model_params[0].keys(): #conv1_
        param_stack = torch.stack([state_dict[key] for state_dict in model_params], dim=0)

        avg_params = torch.mean(param_stack, dim=0)

        averaged_state_dict[key] = avg_params

    return averaged_state_dict

def aggregate_at_server(ap_avg_state_dict):
    return aggregate_params(ap_avg_state_dict)


def ap_aggregate(results, ap_routes)-> [torch.Tensor]:
    ap_params = {}
    for AP, peers in ap_routes.items():
        chosen_params = [result[1] for result in results if result[0] in peers]

        avg_ap_state_dict = aggregate_params(chosen_params)
        ap_params[AP] = avg_ap_state_dict

    return ap_params

def get_parameters(ap_state_dict, ap_routes, node_id):
    if ap_state_dict is None:
        return None
    key = None
    for AP, peers in ap_routes.items():
        if  node_id in peers:
            key = AP
            break

    return ap_state_dict[key]

def get_ap_metrics(ap_avg_state_dict, path, server_round, a, model, testloaders, device):
    file_path = path+f'access_point_eval_{server_round}/aggregate_{a}.txt'

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        for AP, state_dict in ap_avg_state_dict.items():
            model.load_state_dict(state_dict)
            loss, accuracy = test(model, testloaders[AP], device)
            file.write(f"{AP} : \n\tloss: {loss} \n\taccuracy: {accuracy}\n")


if __name__ == '__main__':
    cloud()

