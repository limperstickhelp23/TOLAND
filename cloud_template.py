# from netsim.network import Network
import warnings  # Suppress all warnings # NOTE: some complaints from torch and plotting stuff
from concurrent.futures import ThreadPoolExecutor, as_completed

import colorama
import hydra
import omegaconf
from tqdm import tqdm

from data_prepare import *
from netsim.algorithms import *
from utils import *
from utils import Net

warnings.filterwarnings("ignore")

#NOTE SETTINGS / Set algorithm here
CONFIG_NAME= "network"  # "scalefree", "network", "star", "cosine"
WORKERS=10
THRESHOLD=10 # NOTE: does nothing -- threshold parameter is set at algorithm level


@hydra.main(config_path="configs", config_name=CONFIG_NAME, version_base=None)
def cloud(cfg:DictConfig):
    
    ### Exp Setup Info
    omegaconf.OmegaConf.to_yaml(cfg)
    cfg.config_data.num_partitions = cfg.num_clients
    SAVE_RESULTS = cfg.save_results
    DATA,NETWORK={},None
    DATA["cloud"]={"losses":[],"accuracies":[]}

    SUFFIX = f"_TEST_{cfg.config_data.dataset}"

    ### Data Setup/Loading
    network_model = Net(cfg.num_classes, cfg.input_len)
    trainloaders, validationloaders, testloader = prepare_dataset(cfg=cfg.config_data)
    #check_iidness(trainloaders[0]),print(cfg.config_data.iid, cfg.config_data.alpha)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    file_path = ''
    ###NOTE Select Algorithm
    if cfg.algorithm == "scalefree":
        file_path = f'baselines/{cfg.algorithm}'
        NETWORK=ScaleFreeRewiring(num_devices=cfg.num_clients,num_classes=cfg.num_classes,threshold=THRESHOLD,perceptual_map=cfg.plot_colormap)
    
    elif cfg.algorithm == "cosine":
        file_path = f'cosine_assignment'
        NETWORK=CosineReassignment(num_devices=cfg.num_clients,num_classes=cfg.num_classes,threshold=THRESHOLD,perceptual_map=cfg.plot_colormap)
    elif cfg.algorithm == "prox_preferential":
        file_path = f'proxpref'
        NETWORK=ProximityPreferentialAttachment(num_devices=cfg.num_clients,num_classes=cfg.num_classes,threshold=THRESHOLD,perceptual_map=cfg.plot_colormap)
    elif cfg.algorithm == "star":
        file_path = f'baselines/{cfg.algorithm}'
        return
    elif cfg.algorithm == "DPP":
        file_path = f'DPP'
        NETWORK = DPP(num_devices=cfg.num_clients,num_classes=cfg.num_classes, threshold=THRESHOLD, perceptual_map=cfg.plot_colormap)
    else:
        NETWORK=Algorithm(num_devices=cfg.num_clients,num_classes=cfg.num_classes,threshold=THRESHOLD) # Some Default Behavior

    #Directory setup
    iid="iid" if cfg.config_data.iid else "non_iid"
    cfg.metric_path = cfg.metric_path.format(algorithm=file_path)
    cfg.figure_path = cfg.figure_path.format(algorithm=file_path)
    metpath=f"{cfg.metric_path}/{iid}/"
    figpath=f"{cfg.figure_path}/{iid}/"+f"run_{len(os.listdir(f'{cfg.figure_path}/{iid}/'))}/"

    ###NOTE: Run
    for server_round in range(1, cfg.num_rounds+1):
        print(colorama.Fore.LIGHTBLUE_EX + f'Starting server round {server_round}'+ colorama.Style.RESET_ALL)
        NETWORK.run_global_round_setup_steps(server_round)
        DATA[server_round]={}
        ap_avg_state_dict = []
        for aggr_round in range(1, cfg.aggregation_rounds+1):
            if SAVE_RESULTS:
                os.makedirs(figpath, exist_ok=True)
                NETWORK.plot_communities(spring=True)
                plt.title(f"Round {server_round} Communities")
                plt.savefig(figpath+f"{server_round}_{aggr_round}_{SUFFIX}.jpeg")

            state_dict_map = {}
            if aggr_round == 1:
                state_dict_map = {i: network_model.state_dict() for i in range(cfg.num_clients)}
            else :
                state_dict_map = { member: ap_avg_state_dict[AP]
                    for AP, members in NETWORK.ap_member_map.items()
                    for member in members
                }

            pool = ThreadPoolExecutor(max_workers=WORKERS)
            futures = [
                pool.submit(local_train, i, Net(cfg.num_classes, cfg.input_len), trainloaders[i], validationloaders[i],
                                   state_dict_map[i], cfg.config_fit, device) for i in range(cfg.num_clients)
            ]
            state_dict_map.clear()
            results= [
                future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients")
            ]
            pool.shutdown(wait=True)

            NETWORK.map_results_to_files(results)
            ap_avg_state_dict = NETWORK.run_local_aggregation_round()
            DATA[server_round][aggr_round]=update_ap_metrics(ap_avg_state_dict, Net(cfg.num_classes, cfg.input_len), validationloaders, device)

        net_state_dict = aggregate_params(list(ap_avg_state_dict.values()))

        #NOTE: Globally Evaluate
        network_model.load_state_dict(net_state_dict)
        g_loss, g_accuracy = test(network_model, testloader, device)
        DATA["cloud"]["losses"].append(g_loss)
        DATA["cloud"]["accuracies"].append(g_accuracy)
        print(colorama.Fore.LIGHTGREEN_EX+"\nCheck Round Loss: ", g_loss, ", Accuracy: ", g_accuracy,"\n"+colorama.Style.RESET_ALL)


    if SAVE_RESULTS:
        lgth=len(os.listdir(f"{metpath}/"))
        with open(f"{metpath}/run_{lgth}_{SUFFIX}.json", "w") as file:
            json.dump(DATA, file, indent=4)
        
        print(f"Done.\nMetrics saved to {metpath}/run_{lgth}.json")
        print(f"Plots saved to {figpath}")


if __name__ == "__main__":
    cloud()