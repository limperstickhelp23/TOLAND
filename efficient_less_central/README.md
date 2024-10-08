## Working on a first implementation of [*Efficient and Less Centralized Federated Learning*](https://arxiv.org/pdf/2106.06627)

This uses an algorithm they call "FedP2P" which has many things in common with our discussed strategy. It works by:
  - Selecting L out of N devices (randomly) to recieve the global model at each update
  - These L devices have communities which they talk to locally to aggregate a new model (after local training)
  - The local update is FedAvg but weighted by the size of each dataset. Thus nobody shares what data they have, just how much they have
  - I guess, for more privacy, they could even share weights/N and not N itself
  - The representative node sends this back to the server for global aggregation


## Practical Implementation Discussion
- Flower has a function "run_simulation" which handles much of the under-the-hood management of concurrent local training
- However, if we simply call this function for T rounds, we may not have much control over the inter-round stages
- I think there might be a way to do it, by writing elaborate custom strategies, but it's not obvious to me yet
- A work-around would be to spin out a flower "ServerApp" at each of the *access points* instead, and thus manage the workflow with multiple instances of Flower ServerApps running
- Last, a note on fully decentralized -- in this case, I'm not sure Flower really adds anything at all, we might as well just work with threading or other parallel processing libraries
