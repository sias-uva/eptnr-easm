import itertools as it
from typing import Tuple, List
import igraph as ig
import numpy as np
from ...rewards import BaseReward
from tqdm import tqdm
import logging
from .utils.compute_rewards import compute_rewards_over_removals

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _optimal_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str],
                      current_k: int, verbose: bool = True) -> List[Tuple[List[float], List[int]]]:
    """

    Args:
        g:
        reward:
        edge_types:

    Returns:

    """
    assert 0 < len(g.es.select(type_in=edge_types))

    removable_edges = g.es.select(type_in=edge_types, active_eq=1)
    possible_combinations = [[e.index for e in es]
                             for es in it.combinations(removable_edges, current_k)]

    # logger.info(f"Possible states: {possible_combinations}")
    rewards = -np.ones(len(possible_combinations)) * np.inf

    for i, candidate in enumerate(tqdm(possible_combinations)):
        g_prime = g.copy()
        g_prime.es[candidate]['active'] = 0
        rewards[i] = reward.evaluate(g_prime)
        if verbose:
            logger.info(f"For configuration {candidate} obtained rewards {rewards[i]}")

    max_reward_candidates_idxs = np.where(rewards == rewards.max())[0]

    optimal_solutions_and_rewards_per_removal = []
    if verbose:
        logger.info("OPTIMAL CONFIGURATIONS:")
    for cand_i in max_reward_candidates_idxs:
        if verbose:
            logger.info(f"For configuration {possible_combinations[cand_i]} obtained rewards {rewards[cand_i]}")
        es_idx_list = possible_combinations[cand_i]
        rewards_per_removal = compute_rewards_over_removals(g, reward, es_idx_list)
        optimal_solutions_and_rewards_per_removal.append((rewards_per_removal, es_idx_list))

    return optimal_solutions_and_rewards_per_removal


def optimal_max_baseline(g: ig.Graph, reward: BaseReward,
                         edge_types: List[str], budget: int = None,
                         verbose: bool = True) -> List[Tuple[List[float], List[int]]]:
    """
    Args:
        verbose:
        budget:
        g:
        reward:
        edge_types:

    Returns:
        List of optimal configuration reaching maximum rewards over all solutions in S as a
        list of rewards over each removal in that solution and the edges removed.
    """
    nr_edges_to_remove = budget or len(g.es.select(type_in=edge_types))
    assert 0 < nr_edges_to_remove

    all_opt = []
    for k in range(1, nr_edges_to_remove + 1):
        opt_solutions = _optimal_baseline(g, reward, edge_types, k, verbose)

        for (rewards_over_removals, edges) in opt_solutions:
            max_reward_idx = np.argmax(rewards_over_removals)
            res_tuple = (
                [
                    rewards_over_removals[max_reward_idx],
                    edges[:max_reward_idx+1]
                ],
            )
            all_opt.extend(res_tuple)

    if verbose:
        logger.info(f"All optimal solutions are: {all_opt}")
    all_opt = np.array(all_opt, dtype=object)
    opt_idxs = np.argmax(all_opt[:, 0]).tolist()
    opt_idxs = [opt_idxs] if isinstance(opt_idxs, int) else opt_idxs

    output = []

    for idx in opt_idxs:
        output.append(tuple(all_opt[idx, :].tolist()))

    return output

