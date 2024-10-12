# import pickle
# from pathlib import Path

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from data_prepare import prepare_dataset

from strategy import DecentralizedStrategy, get_on_fit_config_fn, get_evaluate_fn
from device import generate_client_fn


from models import Net
# from client import generate_client_fn
# from dataset import prepare_dataset


# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="configs", config_name="network.yaml")
def cloud(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    OmegaConf.to_yaml(cfg)
    save_path = HydraConfig.get().runtime.output_dir

    trainloaders, validationloaders, testloader = prepare_dataset(
        num_partitions=cfg.num_clients, batch_size=cfg.batch_size
    )

    ## 3. Define your clients
    # Unlike in standard FL (e.g. see the quickstart-pytorch or quickstart-tensorflow examples in the Flower repo),
    # in simulation we don't want to manually launch clients. We delegate that to the VirtualClientEngine.
    # What we need to provide to start_simulation() with is a function that can be called at any point in time to
    # create a client. This is what the line below exactly returns.

    # client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model)
    # clients = [client_fn(cid) for cid in range(cfg.num_clients)]
    #
    # for client in clients:
    #     client.peers = [peer for peer in clients if peer != client]
    #
    # client_fn_buffer = lambda cid: clients[int(cid)]


    ## 4. Define your strategy
    # A flower strategy orchestrates your FL pipeline. Although it is present in all stages of the FL process
    # each strategy often differs from others depending on how the model _aggregation_ is performed. This happens
    # in the strategy's `aggregate_fit()` method. In this tutorial we choose FedAvg, which simply takes the average
    # of the models received from the clients that participated in a FL round doing fit().
    # You can implement a custom strategy to have full control on all aspects including: how the clients are sampled,
    # how updated models from the clients are aggregated, how the model is evaluated on the server, etc
    # To control how many clients are sampled, strategies often use a combination of two parameters `fraction_{}` and `min_{}_clients`
    #     # evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    # )  # a function to run on the server side to evaluate the global model.
    access_points = cfg.neighbor_lists['AP']
    strategy = instantiate(cfg.strategy, ap_routes= access_points, evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader))
    print(strategy)


    ## 5. Start Simulation
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!

    history = fl.simulation.start_simulation(
        client_fn=generate_client_fn(trainloaders, validationloaders, cfg.model),  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
        },  # (optional) controls the degree of parallelism of your simulation.
        # Lower resources per client allow for more clients to run concurrently
        # (but need to be set taking into account the compute/memory footprint of your run)
        # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
        # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs.
    )

    # ^ Following the above comment about `client_resources`. if you set `num_gpus` to 0.5 and you have one GPU in your system,
    # then your simulation would run 2 clients concurrently. If in your round you have more than 2 clients, then clients will wait
    # until resources are available from them. This scheduling is done under-the-hood for you so you don't have to worry about it.
    # What is really important is that you set your `num_gpus` value correctly for the task your clients do. For example, if you are training
    # a large model, then you'll likely see `nvidia-smi` reporting a large memory usage of you clients. In those settings, you might need to
    # leave `num_gpus` as a high value (0.5 or even 1.0). For smaller models, like the one in this tutorial, your GPU would likely be capable
    # of running at least 2 or more (depending on your GPU model.)
    # Please note that GPU memory is only one dimension to consider when optimising your simulation. Other aspects such as compute footprint
    # and I/O to the filesystem or data preprocessing might affect your simulation  (and tweaking `num_gpus` would not translate into speedups)
    # Finally, please note that these gpu limits are not enforced, meaning that a client can still go beyond the limit initially assigned, if
    # this happens, your might get some out-of-memory (OOM) errors.

    ## 6. Save your results
    # (This is one way of saving results, others are of course valid :) )
    # Now that the simulation is completed, we could save the results into the directory
    # that Hydra created automatically at the beginning of the experiment.

    # results_path = Path(save_path) / "results.pkl"

    # add the history returned by the strategy into a standard Python dictionary
    # you can add more content if you wish (note that in the directory created by
    # Hydra, you'll already have the config used as well as the log)

    # results = {"history": history, "anythingelse": "here"}

    # save the results as a python pickle

    # with open(str(results_path), "wb") as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    cloud()