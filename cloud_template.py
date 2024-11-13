# from netsim.network import Network
from netsim.algorithms import *
from utils import *
from utils import Net
# from deprecate.client import local_train, test
# from deprecate.data_prepare import prepare_dataset

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pickle import EMPTY_DICT

from tqdm import tqdm

import colorama
import hydra
import omegaconf
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig


import warnings # Suppress all warnings # NOTE: some complaints from torch and plotting stuff
warnings.filterwarnings("ignore") 

#NOTE SETTINGS / Set algorithm here
CONFIG_NAME="scalefree"  # "scalefree", "network", "star", "cosine"
SAVE_RESULTS=False
WORKERS=10
THRESHOLD=10 # NOTE: does nothing -- threshold parameter is set at algorithm level
SUFFIX="_TEST"

@hydra.main(config_path="configs", config_name=CONFIG_NAME, version_base=None)
def cloud(cfg:DictConfig):
    
    ### Exp Setup Info
    omegaconf.OmegaConf.to_yaml(cfg)
    DATA,NETWORK={},None
    DATA["cloud"]={"losses":[],"accuracies":[]}
    iid="iid" if cfg.iid else "non_iid"
    metpath=f"{cfg.metric_path}/{iid}/"
    figpath=f"{cfg.figure_path}/{iid}/"+f"run_{len(os.listdir(f"{cfg.figure_path}/{iid}/"))}/"

    ### Data Setup/Loading
    net_model = Net(cfg.num_classes)
    trainloaders, validationloaders, testloader = prepare_dataset(num_partitions=cfg.num_clients, batch_size=cfg.batch_size,iid=cfg.iid,alpha=cfg.alpha)
    check_iidness(trainloaders[0]),print(cfg.iid, cfg.alpha)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    ###NOTE Select Algorithm
    if cfg.algorithm == "scalefree":
        NETWORK=ScaleFreeRewiring(num_devices=cfg.num_clients,num_classes=cfg.num_classes,threshold=THRESHOLD,perceptual_map=cfg.plot_colormap)
    
    elif cfg.algorithm == "cosine":
        NETWORK=CosineReassignment(num_devices=cfg.num_clients,num_classes=cfg.num_classes,threshold=THRESHOLD,perceptual_map=cfg.plot_colormap)
    
    elif cfg.algorithm == "prox_preferential":
        NETWORK=ProximityPreferentialAttachment(num_devices=cfg.num_clients,num_classes=cfg.num_classes,threshold=THRESHOLD,perceptual_map=cfg.plot_colormap)
        print("hre")

    else:
        NETWORK=Algorithm(num_devices=cfg.num_clients,num_classes=cfg.num_classes,threshold=THRESHOLD) # Some Default Behavior
    
    # elif cfg.algorothm == "star":
    #     NETWORK=StarBaseline(TODO)

    ###NOTE: Run
    for server_round in range(cfg.num_rounds):

        NETWORK.run_global_round_setup_steps(server_round)
        DATA[server_round]={}
        
        if (SAVE_RESULTS):
            os.makedirs(figpath, exist_ok=True)
            NETWORK.plot_communities(spring=False)
            plt.title(f"Round {server_round} Communities")
            plt.savefig(figpath+f"{server_round}_{SUFFIX}.png")
        
        print(colorama.Fore.LIGHTBLUE_EX+f'Starting server round {server_round}')
        pool = ThreadPoolExecutor(max_workers=WORKERS)

        futures = [
            pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
                               net_model.state_dict(), cfg.config_fit, device) for i in range(cfg.num_clients)
        ]
        results= [
            future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients")
        ]
        pool.shutdown(wait=True)
        
        aggr_round=0 #TODO -- Might need additional logic for multiple rounds
        # Locally Passing to AP's (TODO: Multiple Rounds)
        NETWORK.map_results_to_files(results)
        ap_avg_state_dict = NETWORK.run_local_aggregation_round()
        DATA[server_round][aggr_round]=update_ap_metrics(ap_avg_state_dict, Net(cfg.num_classes), validationloaders, device)
        net_state_dict = aggregate_params(list(ap_avg_state_dict.values()))

        #NOTE: Globally Evaluate
        net_model.load_state_dict(net_state_dict)
        g_loss, g_accuracy = test(net_model, testloader, device)
        DATA["cloud"]["losses"].append(g_loss)
        DATA["cloud"]["accuracies"].append(g_accuracy)
        print("\nCheck Round Loss: ", g_loss, ", Accuracy: ", g_accuracy,"\n")

    
    if (SAVE_RESULTS):
        lgth=len(os.listdir(f"{metpath}/"))
        with open(f"{metpath}/run_{lgth}_{SUFFIX}.json", "w") as file:
            json.dump(DATA, file, indent=4)
        
        print(f"Done.\nMetrics saved to {metpath}/run_{lgth}.json")
        print(f"Plots saved to {figpath}/")


if __name__ == "__main__":
    cloud()



# parser = argparse.ArgumentParser(description="Run a script with a specified config file.")
# parser.add_argument(
#     'casename',
#     type=str,
#     # required=True,
#     help="Path to the configuration file."
# )
# args = parser.parse_args()
# print(f"Running config: {args.casename}")
# CONFIG_NAME = args.casename