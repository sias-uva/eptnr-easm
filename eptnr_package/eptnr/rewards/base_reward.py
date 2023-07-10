import abc
import igraph as ig
import pandas as pd
from typing import List
from .utils.graph_computation_utils import get_tt_df
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class BaseReward(abc.ABC):

    def __init__(self, census_data: pd.DataFrame, groups: List[str] = None, 
                 verbose: bool = False, reward_scaling=False) -> None:

        self.census_data = census_data
        self.verbose = verbose
        self.reward_scaling = reward_scaling

        self.groups = groups

    def retrieve_tt_df(self, g: ig.Graph) -> pd.DataFrame:
        g_prime = g.subgraph_edges(g.es.select(active_eq=1), delete_vertices=False)
        tt_samples = get_tt_df(g_prime, self.census_data)

        return tt_samples

    @abc.abstractmethod
    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def _reward_scaling(self, reward: float) -> float:
        raise NotImplementedError()

    def evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        g_prime = g.subgraph_edges(g.es.select(active_eq=1), delete_vertices=False)
        if self.verbose:
            logger.info("Received graph with:\n"
                        f"\tn_edges={len(g.es)}"
                        f"\tn_vertices={len(g.vs)}\n"
                        f"Created subgraph:\n"
                        f"\tn_edges={len(g_prime.es)}\n"
                        f"\tn_vertices={len(g_prime.vs)}")
        calculated_reward = self._evaluate(g_prime, *args, **kwargs)
        if self.reward_scaling:
            scaled_reward = self._reward_scaling(calculated_reward)
        else:
            scaled_reward = calculated_reward
        if self.verbose:
            logger.info(f"Resulting rewards:\n"
                        f"\t{calculated_reward=}\n"
                        f"\t{scaled_reward=}")

        return scaled_reward
