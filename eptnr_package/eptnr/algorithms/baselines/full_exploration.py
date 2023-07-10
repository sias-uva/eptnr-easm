import itertools as it
from typing import Dict, List
import igraph as ig
from ...rewards import BaseReward
from tqdm import tqdm
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def full_exploration_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str],
                              budget: int, verbose: bool = False) -> Dict[str, float]:
    """

    Args:
        budget:
        verbose:
        g:
        reward:
        edge_types:

    Returns:

    """
    assert 0 < len(g.es.select(type_in=edge_types))

    nr_edges_to_remove = budget or len(g.es.select(type_in=edge_types))
    assert 0 < nr_edges_to_remove

    # logger.info(f"Possible states: {possible_combinations}")
    rewards = {}

    for current_k in range(1, nr_edges_to_remove + 1):
        removable_edges = g.es.select(type_in=edge_types, active_eq=1)
        possible_combinations = [[e.index for e in es]
                                 for es in it.combinations(removable_edges, current_k)]
        for i, candidate in enumerate(tqdm(possible_combinations)):
            g_prime = g.copy()
            g_prime.es[candidate]['active'] = 0
            rewards[str(candidate)] = reward.evaluate(g_prime)
            if verbose:
                logger.info(f"For configuration {candidate} obtained rewards {rewards[str(candidate)]}")

    return rewards
