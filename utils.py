import torch
import os
from typing import List
from client import test
import pandas as pd
import numpy as np
from collections import Counter

def aggregate_params(model_params):
    
    averaged_state_dict = {}
    for key in model_params[0].keys(): #conv1_
        param_stack = torch.stack([state_dict[key] for state_dict in model_params], dim=0)
        avg_params = torch.mean(param_stack, dim=0)
        averaged_state_dict[key] = avg_params

    return averaged_state_dict

def aggregate_at_server(ap_avg_state_dict):
    return aggregate_params(ap_avg_state_dict)

def ap_aggregate(results, ap_routes)-> List[torch.Tensor]:
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

def update_ap_metrics(ap_avg_state_dict,model,testloaders,device):
    losses,accuracies=[],[]
    for (AP, state_dict) in ap_avg_state_dict.items():
        model.load_state_dict(state_dict),
        loss, accuracy = test(model, testloaders[AP], device)
        losses.append(loss)
        accuracies.append(accuracy)
    
    return {
        "ap_nodes":list(ap_avg_state_dict.keys()),
        "losses":losses, 
        "accuracies":accuracies
    }

def normalize_dict(d):
    D=np.sum([v for v in list(d.values())])
    for (k,v) in d.items():
        d[k]=np.round(v/D,3)
    return d

def get_class_distribution(dataloader):
    class_counts = Counter()
    for _, labels in dataloader:
        class_counts.update(labels.tolist())
    return dict(class_counts)

def check_iidness(dataloader):
    d=normalize_dict(get_class_distribution(dataloader))
    for (k,v) in d.items():
        print(f"{k}: {v}")


# def get_ap_metrics_pandas(ap_avg_state_dict, path, server_round, a, model, testloaders, device):
#     """TODO WIP """
    
#     file_path=f'access_point_eval_{server_round}/aggregate_{a}.txt'

#     for AP, state_dict in ap_avg_state_dict.items():
#             model.load_state_dict(state_dict)
#             loss, accuracy = test(model, testloaders[AP], device)
#             row={
#                 "loss":{}


#             }
#             file.write(f"{AP} : \n\tloss: {loss} \n\taccuracy: {accuracy}\n")

#             new_row = pd.DataFrame([{"Name": f"Person_{i}", "Age": 20 + i, "City": f"City_{i}"}])
#             rows.append(new_row)  # Add row to the list