from concurrent.futures import ThreadPoolExecutor, as_completed

import colorama
import hydra
import omegaconf
from tqdm import tqdm

from deprecate.client import local_train
from deprecate.data_prepare import prepare_dataset
from deprecate.models import Net
from netsim.algorithms import *
from utils import *

#SETTINGS
CONFIG_NAME="cosine"
SAVE_RESULTS=True

@hydra.main(config_path="configs", config_name=CONFIG_NAME, version_base=None)
def cloud(cfg:DictConfig):
    omegaconf.OmegaConf.to_yaml(cfg)
    DATA={}
    DATA["cloud"]={"losses":[],"accuracies":[]}
    iid="iid" if cfg.iid else "non_iid"
    
    metpath=f"{cfg.metric_path}/{iid}"
    figpath=f"{cfg.figure_path}/{iid}/"+f"run_{len(os.listdir(f'{cfg.figure_path}/{iid}/'))}"

    # save_path = HydraConfig.get().runtime.output_dir  # NOTE: or put paths in config files
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size, iid=cfg.iid, alpha=cfg.alpha)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap_avg_state_dict,global_state_dict,net_model=None,None,Net(cfg.num_classes)
    check_iidness(trainloaders[0])

    #NOTE INIT Network:
    NETWORK=CosineReassignment(num_devices=cfg.num_clients, num_classes=cfg.num_classes,perceptual_map=cfg.plot_colormap)
    NETWORK.threshold=10
    NETWORK.build_proximity_graph() # allow_isolates = False
    NETWORK.select_access_points_on_betweenness() 
    
    for server_round in range(cfg.num_rounds):
        print(colorama.Fore.LIGHTBLUE_EX+f'Starting server round {server_round}')
        
        DATA[server_round]={}
        # print("\nCheck Access Point Assignments\n" + "\nCommunities:\n" +"\n".join([f"{k} : {v}" for (k,v) in NETWORK.ap_member_map.items()]))    # Demo/Monitor Self-Assignment Process
        
        #NOTE: Plot Each Global Round
        if (SAVE_RESULTS):
            os.makedirs(figpath, exist_ok=True)
            NETWORK.plot_communities(spring=False,which=1)
            plt.title(f"Server Round {server_round} Communities")
            plt.savefig(figpath+f"/{server_round}.png")

        for aggr_round in range(cfg['aggregation_rounds']):
            
            pool=ThreadPoolExecutor(max_workers=10)
            futures=[pool.submit(local_train, i, Net(cfg.num_classes), trainloaders[i], validationloaders[i],
                                    global_state_dict, cfg.config_fit, device) for i in range(cfg.num_clients)]
            results=[ future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Training clients")]
            pool.shutdown(wait=True)
            
            # NOTE: Run the Reassignment Step
            NETWORK.map_results(results)
            NETWORK.compare_communities(max_iters=3)  
            ap_avg_state_dict = NETWORK.local_aggregation_round()
            DATA[server_round][aggr_round]=update_ap_metrics(ap_avg_state_dict, Net(cfg.num_classes), validationloaders, device)

        # Process Server Round Results
        global_state_dict = aggregate_params(list(ap_avg_state_dict.values()))
        net_model.load_state_dict(global_state_dict)
        g_loss, g_accuracy = test(net_model, testloader, device)

        #NOTE: Evaluate/Save
        net_model.load_state_dict(global_state_dict)
        g_loss, g_accuracy = test(net_model, testloader, device)
        DATA["cloud"]["losses"].append(g_loss)
        DATA["cloud"]["accuracies"].append(g_accuracy)
        print("loss: ", g_loss, "accuracy: ", g_accuracy)

    if (SAVE_RESULTS):
        with open(f"{metpath}/run_{len(os.listdir(f'{metpath}/'))}.json", "w") as file:
            json.dump(DATA, file, indent=4)


if __name__ == '__main__':
    cloud()


    

