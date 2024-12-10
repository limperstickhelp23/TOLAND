from netsim.milestone2 import ModularDPP

# TODO : Still need to see how this works in training loop
def modular_dpp_unit_test():
    Net=ModularDPP(80,5)
    for round in range(5):
        Net.run_global_round_setup_steps(round,plot=False)

    # NOTE: test rate parameter
    # Net.run_community_division_step(step=1,rate=.1)
    # Net.run_community_division_step(step=3,rate=.1)
    # Net.run_community_division_step(step=10,rate=.1)

    """ (1) Assign AP using MinSpanTree inside a Cluster (easy)
            
            (a) Actually could use Local Star Instead ....
            (b) Actually should do -- Local DPP as this is the Original Idea , then take modularity to get communities

        (2) Next, we need some logic for running the local rounds, either:

            (a) infrequent cosim comparisons (hub sends information back to nodes -- they try to classify themselves)

            (b) running DPP at every round (related to b above)
    """

if __name__ == "__main__":
    modular_dpp_unit_test()

    
