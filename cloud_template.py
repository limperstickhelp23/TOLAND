# from netsim.network import Network
import time
import warnings  # Suppress all warnings # NOTE: some complaints from torch and plotting stuff
from os.path import exists

import colorama
import hydra
import omegaconf

import matplotlib.pyplot as plt
from data_prepare import *
from netsim.algorithms import *
from utils import *

warnings.filterwarnings("ignore")

#NOTE SETTINGS / Set algorithm here
CONFIG_NAME= "network"  # "scalefree", "network", "star", "cosine"
WORKERS=20
THRESHOLD=0.65




@hydra.main(config_path="configs", config_name=CONFIG_NAME, version_base=None)
def cloud(cfg:DictConfig):
    ### Exp Setup Info
    omegaconf.OmegaConf.to_yaml(cfg)
    cfg.config_data.num_partitions = cfg.num_clients
    SAVE_RESULTS = cfg.save_results
    SAVE_FIGURES = cfg.save_figures
    DATA,NETWORK={},None
    DATA["cloud"]={"losses":[],"accuracies":[], "Wall_Clock":[]}
    CLASS_DIST = {}

    SUFFIX = f"{cfg.config_data.dataset}"

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
        NETWORK=Algorithm(num_devices=cfg.num_clients,num_classes=cfg.num_classes, threshold=THRESHOLD,perceptual_map=cfg.plot_colormap, star=True)
    elif cfg.algorithm == "DPP":
        file_path = f'DPP'
        NETWORK = DPP(num_devices=cfg.num_clients,num_classes=cfg.num_classes, threshold=THRESHOLD, perceptual_map=cfg.plot_colormap)
    else:
        NETWORK=Algorithm(num_devices=cfg.num_clients,num_classes=cfg.num_classes,threshold=THRESHOLD) # Some Default Behavior

    #Directory setup
    iid="iid" if cfg.config_data.iid else "non_iid"
    cfg.metric_path = cfg.metric_path.format(algorithm=file_path)
    cfg.figure_path = cfg.figure_path.format(algorithm=file_path)
    cfg.gephi_path = cfg.gephi_path.format(algorithm=file_path)

    os.makedirs(f"{cfg.gephi_path}/{iid}/", exist_ok=True)

    metpath=f"{cfg.metric_path}/{iid}/"
    figpath=f"{cfg.figure_path}/{iid}/"+f"run_{len(os.listdir(f'{cfg.figure_path}/{iid}/'))}/"
    gephipath=f"{cfg.gephi_path}/{iid}/"+f"run_{len(os.listdir(f'{cfg.gephi_path}/{iid}/'))}/"
    
    if not os.path.exists(metpath):
        os.mkdir(metpath)
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    if not os.path.exists(gephipath):
        os.mkdir(gephipath)

    print(colorama.Fore.MAGENTA+ f'{cfg.algorithm} algorithm'+ colorama.Style.RESET_ALL)
    ###NOTE: Run
    for server_round in range(cfg.num_rounds):
        if server_round > 0 and DATA["cloud"]["accuracies"][-1] >= 0.945:
            break
        print(colorama.Fore.LIGHTBLUE_EX + f'Starting server round {server_round+1}'+ colorama.Style.RESET_ALL)

        NETWORK.run_global_round_setup_steps(server_round, threshold=THRESHOLD)

        if server_round == 0 and cfg.algorithm != "star" and SAVE_FIGURES:
            CLASS_DIST["Starting Communities"] = get_community_class_distributions(trainloaders, NETWORK.ap_member_map)

        DATA[server_round]={}
        ap_avg_state_dict = []

        # Distribution + Aggregation Cost
        if (cfg.algorithm == "star"):
            print("Star Costs TOO much")
            NETWORK.calculate_server_distribution_aggregation_cost(mode="star")
            NETWORK.calculate_server_distribution_aggregation_cost(mode="star") # 2x for agg and distribution
        else:
            NETWORK.calculate_server_distribution_aggregation_cost(mode="ap")
            NETWORK.calculate_server_distribution_aggregation_cost(mode="ap") # 2x for agg and distribution

        start_time = time.time()
        for aggr_round in range(cfg.aggregation_rounds):

            # TODO: DELETE / Moved this below because star doesn't really plot anything anyway
            # if SAVE_RESULTS:
            #     os.makedirs(figpath, exist_ok=True)
            #     NETWORK.plot_communities(spring=True)
            #     plt.title(f"Round {server_round} Communities")
            #     plt.savefig(figpath+f"{server_round}_{aggr_round}_{SUFFIX}.jpeg")
            #     #TODO: check on making GEPHI files
            
            if (cfg.algorithm != "star"):
                NETWORK.calculate_ap_distribution_aggregation_cost()
                NETWORK.calculate_ap_distribution_aggregation_cost() # 2x for agg and distribution
                
                if SAVE_RESULTS:
                    os.makedirs(figpath, exist_ok=True)
                    os.makedirs(gephipath, exist_ok=True)
                    NETWORK.plot_communities(spring=False)
                    plt.title(f"Round {server_round} Communities")
                    plt.savefig(figpath+f"{server_round}_{aggr_round}_{SUFFIX}.jpeg")

                    # Ensure Communities Passed as Integers:
                    for (n,members) in enumerate(NETWORK.ap_member_map.values()):
                        for m in members:
                            NETWORK.NXG2.nodes[m]['gephi_community'] = n
                            NETWORK.NXG2.nodes[m]['lat'] = NETWORK.device_list[m].x
                            NETWORK.NXG2.nodes[m]['lon'] = NETWORK.device_list[m].y
                            NETWORK.NXG2.nodes[m]['is_ap'] = 1*(m in NETWORK.curr_apoints)
                    nx.write_gexf(NETWORK.NXG2, gephipath+f"{server_round}_{aggr_round}_{SUFFIX}.gexf") #TODO: check on making GEPHI files
            
            state_dict_map = {}
            print(f'aggr_round: {aggr_round+1}')
            if aggr_round == 0:
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
            if (cfg.algorithm == "star"):
                break # No additional steps needed
            DATA[server_round][aggr_round]=update_ap_metrics(ap_avg_state_dict, Net(cfg.num_classes, cfg.input_len), validationloaders, device, MAX_WORKERS=WORKERS)

        net_state_dict = aggregate_params(list(ap_avg_state_dict.values()))

        #NOTE: Globally Evaluate
        network_model.load_state_dict(net_state_dict)
        end_time = time.time()
        g_loss, g_accuracy = test(network_model, testloader, device)
        DATA["cloud"]["losses"].append(g_loss)
        DATA["cloud"]["accuracies"].append(g_accuracy)
        DATA["cloud"]["Wall_Clock"].append(end_time-start_time)
        print(colorama.Fore.LIGHTGREEN_EX+"\nCheck Round Loss: ", g_loss, ", Accuracy: ", g_accuracy,"\n"+colorama.Style.RESET_ALL)

    DATA["total_run_cost"] = NETWORK.total_cost
    if cfg.algorithm != "star" and SAVE_FIGURES:
        CLASS_DIST["Ending_Communities"] = get_community_class_distributions(trainloaders, NETWORK.ap_member_map)
    print(f"{cfg.algorithm } cost ", NETWORK.total_cost)
    print(CLASS_DIST)
    if SAVE_RESULTS:
        lgth=len(os.listdir(f"{metpath}/"))
        with open(f"{metpath}/run_{lgth}_{SUFFIX}.json", "w") as file:
            json.dump(DATA, file, indent=4)
        with open(f"{metpath}/class_dist_{lgth}_{SUFFIX}.json", "w") as file:
            json.dump(CLASS_DIST, file, indent=4)
        print(f"Done.\nMetrics saved to {metpath}/run_{lgth}.json")
        print(f"Plots saved to {figpath}")


if __name__ == "__main__":
    cloud()