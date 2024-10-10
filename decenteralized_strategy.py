from typing import Dict, List, Tuple, OrderedDict

from flwr.common import Scalar, Parameters, EvaluateIns, FitIns
from flwr.server import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import flwr
from omegaconf import DictConfig, OmegaConf

from utils import test


class DecentralizedStrategy(FedAvg):

   def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reverse_cid_map = None
        self.cid_map = {}

   def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
       """Configure the next round of training."""

       # Sample clients
       sample_size, min_num_clients = self.num_fit_clients(
           client_manager.num_available()
       )
       #clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

       clients = client_manager.all()

       tmp = list(clients.keys())[0]
       print("\ncid from key: " + tmp + ", cid from value: " + clients[tmp].cid+"\n" )

       if self.cid_map is None:
           self.cid_map = {client.cid : idx for idx, client in enumerate(clients.values())}
           self.reverse_cid_map = {value: key for key, value in self.cid_map.items()}

       client_ids = [self.cid_map[client.cid] for client in clients.values()]


       config_list = []

       for cid in client_ids:
           d = self.on_fit_config_fn(server_round, cid)
           peers = [clients[self.reverse_cid_map[idx]] for idx in d["peers"]]
           d["peers"] = peers
           config_list.append(d)

       fit_in_list = [FitIns(parameters, config) for config in config_list]
       # Return client/config pairs
       return [(client, fit_ins) for client, fit_ins in zip(clients.values(), fit_in_list)]


# def get_evaluate_fn(num_classes: int, testloader):
#     """Define function for global evaluation on the server."""
#
#     def evaluate_fn(server_round: int, parameters, config):
#         # This function is called by the strategy's `evaluate()` method
#         # and receives as input arguments the current round number and the
#         # parameters of the global model.
#         # this function takes these parameters and evaluates the global model
#         # on a evaluation / test dataset.
#
#         model = Net(num_classes)
#
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)
#
#         # Here we evaluate the global model on the test set. Recall that in more
#         # realistic settings you'd only do this at the end of your FL experiment
#         # you can use the `server_round` input argument to determine if this is the
#         # last round. If it's not, then preferably use a global validation set.
#         loss, accuracy = test(model, testloader, device)
#
#         # Report the loss and any other metric (inside a dictionary). In this case
#         # we report the global test accuracy.
#         return loss, {"accuracy": accuracy}
#
#     return evaluate_fn


def get_on_fit_config_fn(config: DictConfig):
    """Return a function to configure the client's fit."""
    def fit_config_fn(server_round: int, cid: int):
        neighbor_lists = OmegaConf.load(config.path) #try to leverage Hydra to call neighbors file

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
            "peers" : neighbor_lists[cid],
        }

    return fit_config_fn