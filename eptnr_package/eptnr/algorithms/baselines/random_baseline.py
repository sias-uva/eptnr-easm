from typing import List, Tuple
import random
import igraph as ig
from ...rewards import BaseReward


def random_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str], 
                    budget: int = None) -> Tuple[List[float], List[int]]:
    """

    Args:
        budget:
        reward:
        edge_types:
        g:

    Returns:

    """
    g_prime = g.copy()
    nr_removable_edges = budget or len(g.es.select(type_in=edge_types))

    assert 0 < nr_removable_edges

    removed_edges = []
    rewards_per_removal = []

    for _ in range(nr_removable_edges):
        removable_edges = g_prime.es.select(type_in=edge_types, active_eq=1)
        edge_to_remove = random.sample(list(removable_edges), 1)[0].index
        removed_edges.append(edge_to_remove)

        g_prime.es[edge_to_remove]['active'] = 0
        r = reward.evaluate(g_prime)
        rewards_per_removal.append(r)

    return rewards_per_removal, removed_edges
