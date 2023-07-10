import igraph as ig
from typing import List
from ....rewards.base_reward import BaseReward
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_rewards_over_removals(base_graph: ig.Graph, reward: BaseReward,
                                  removed_edge_list: List[int], verbose: bool = False) -> List[float]:
    rewards_per_removal = []
    for i in range(0, len(removed_edge_list)):
        g_prime = base_graph.copy()
        g_prime.es[removed_edge_list[0:i + 1]]['active'] = 0
        if verbose:
            logger.info(reward.evaluate(g_prime))
        rewards_per_removal.append(reward.evaluate(g_prime))

    return rewards_per_removal
