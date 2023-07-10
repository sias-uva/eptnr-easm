from typing import List, Tuple
import igraph as ig
import numpy as np
import pygad
import logging
from ...rewards import BaseReward
from .utils.compute_rewards import compute_rewards_over_removals

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _ga_runner(g: ig.Graph, reward: BaseReward, edge_types: List[str], fitness_func: callable,
               budget: int = 5, num_generations: int = 50,
               num_parents_mating: int = 10, sol_per_pop: int = 30, crossover_probability: float = 0.4,
               mutation_probability: float = 0.4, saturation: int = 20) -> Tuple[List[float], List[int]]:
    """

    Args:
        reward:
        g:
        edge_types:
        budget:
        num_generations:
        num_parents_mating:
        sol_per_pop:
        crossover_probability:
        mutation_probability:
        saturation:

    Returns:

    """
    removable_edges = g.es.select(type_in=edge_types, active_eq=1).indices

    assert 0 < budget < len(removable_edges)

    def callback_gen(ga_instance: pygad.GA):
        print("Generation : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", ga_instance.best_solution()[1])

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        initial_population=None,
        sol_per_pop=sol_per_pop,
        num_genes=budget,
        gene_type=int,
        parent_selection_type="sss",
        crossover_type="single_point",
        crossover_probability=crossover_probability,
        mutation_type="random",
        mutation_probability=mutation_probability,
        mutation_by_replacement=False,
        mutation_percent_genes="default",
        mutation_num_genes=None,
        gene_space=list(removable_edges),
        # on_start=None,
        # on_fitness=None,
        # on_parents=None,
        # on_crossover=None,
        # on_mutation=None,
        on_generation=callback_gen,
        # on_stop=None,
        delay_after_gen=0.0,
        save_best_solutions=True,
        save_solutions=True,
        suppress_warnings=False,
        stop_criteria=f'saturate_{saturation}',
    )

    logger.info("Starting GA run")
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    rewards = compute_rewards_over_removals(
        base_graph=g,
        reward=reward,
        removed_edge_list=list(solution),
        verbose=False
    )

    logger.info("Parameters of the best solution : {solution}".format(solution=solution))
    logger.info("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    return rewards, solution


def ga_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str],
                budget: int = 5, num_generations: int = 50,
                num_parents_mating: int = 10, sol_per_pop: int = 30, crossover_probability: float = 0.4,
                mutation_probability: float = 0.4, saturation: int = 20) -> Tuple[List[float], List[int]]:
    def fitness_func(solution: List[int], solution_idx: int):
        edges_to_remove = [e.index for e in g.es[list(solution)]]

        # Encode that the same edge cannot be removed multiple times
        if np.unique(np.array(edges_to_remove)).size != np.array(edges_to_remove).size:
            return -100

        g_prime = g.copy()
        g_prime.es[edges_to_remove]['active'] = False
        r = reward.evaluate(g_prime)
        return r

    return _ga_runner(g=g, reward=reward, edge_types=edge_types, fitness_func=fitness_func,
                      budget=budget, num_generations=num_generations,
                      num_parents_mating=num_parents_mating, sol_per_pop=sol_per_pop,
                      crossover_probability=crossover_probability,
                      mutation_probability=mutation_probability, saturation=saturation)


def ga_max_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str],
                    budget: int = 5, num_generations: int = 50,
                    num_parents_mating: int = 10, sol_per_pop: int = 30, crossover_probability: float = 0.4,
                    mutation_probability: float = 0.4, saturation: int = 20) -> Tuple[List[float], List[int]]:
    def fitness_func(solution: List[int], solution_idx: int):
        edges_to_remove = [e.index for e in g.es[list(solution)]]

        # Encode that the same edge cannot be removed multiple times
        if np.unique(np.array(edges_to_remove)).size != np.array(edges_to_remove).size:
            return -100

        g_prime = g.copy()

        rewards = []
        for edge in edges_to_remove:
            g_prime.es[edge]['active'] = False
            r = reward.evaluate(g_prime)
            rewards.append(r)

        return max(rewards)

    rewards, solution = _ga_runner(g=g, reward=reward, edge_types=edge_types, fitness_func=fitness_func,
                                   budget=budget, num_generations=num_generations,
                                   num_parents_mating=num_parents_mating, sol_per_pop=sol_per_pop,
                                   crossover_probability=crossover_probability,
                                   mutation_probability=mutation_probability, saturation=saturation)
    max_idx = np.argmax(rewards) + 1
    return rewards[:max_idx], solution[:max_idx]
