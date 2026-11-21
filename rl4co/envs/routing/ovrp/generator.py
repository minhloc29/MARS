from rl4co.envs.routing.cvrp.generator import CVRPGenerator


class OVRPGenerator(CVRPGenerator):
    """Data generator for the Open Vehicle Routing Problem (OVRP).

    Instance data is identical to the CVRP (depot + ``num_loc`` customers with
    demand, capacity constraints enforce route splits). The only difference is
    how the tour cost is defined at decode time: the Open Vehicle Routing
    Problem does not charge the final return-to-depot arc of each route. We
    reuse the CVRP realization exactly so that a model trained on CVRP *is*
    the same distribution, and the "open" aspect is purely a cost/reward
    choice made by :class:`OVRPEnv`.
    """
