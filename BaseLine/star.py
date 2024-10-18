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

from client import local_train, test
from data_prepare import prepare_dataset
from models import Net
import colorama




@hydra.main(config_path="configs", config_name="network", version_base=None)
def cloud(cfg:DictConfig):
    omegaconf.OmegaConf.to_yaml(cfg)
    save_path = HydraConfig.get().runtime.output_dir

    run = cfg['run_id']

    path = f"metrics/run_{run}/"
    os.makedirs(path, exist_ok=True)


    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    net_model = Net(cfg.num_classes)

    ap_avg_state_dict = None
    for server_round in range(cfg.num_rounds):
        print(colorama.Fore.LIGHTBLUE_EX+f'Starting server round {server_round}')
        pool = ThreadPoolExecutor(max_workers=10)
        results = []

        futures = [pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
                               net_model.state_dict(), cfg.config_fit, device) for i in range(cfg.num_clients)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients"):
            result = future.result()
            results.append(result)
        pool.shutdown(wait=True)

        params = [result[1] for result in results]
        if server_round != 0:
            params.append(net_model.state_dict())

        net_state_dict = aggregate_params(params)

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

if __name__ == '__main__':
    cloud()

