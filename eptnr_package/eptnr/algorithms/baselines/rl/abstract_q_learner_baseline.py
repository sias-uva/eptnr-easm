import os
import pickle

import igraph as ig
from typing import Tuple, List, Optional, Union
import abc
import math

from ....rewards import BaseReward
from ...q_learning_utils.epsilon_schedule import EpsilonSchedule
from ....exceptions.q_learner_exceptions import ActionAlreadyTakenError

import numpy as np

import itertools as it
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class AbstractQLearner(abc.ABC):
    def __init__(self, base_graph: ig.Graph, reward: BaseReward,
                 edge_types: List[str], episodes: int, step_size: float = 1.0,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 200, static_eps_steps: int = 0,
                 discount_factor: float = 1.0) -> None:
        self.base_graph = base_graph
        self.reward = reward
        self.episodes = episodes
        self.alpha = step_size
        self.gamma = discount_factor

        self.starting_state: Tuple[int] = ()
        self.wrong_action_reward: int = -100

        self.actions = np.array([e.index for e in self.base_graph.es.select(type_in=edge_types, active_eq=1)])

        self.q_values = self._get_q_value_dict()

        self.eps_schedule = EpsilonSchedule(eps_start=eps_start, eps_end=eps_end,
                                            eps_decay=eps_decay, static_eps_steps=static_eps_steps)

        self.steps_done = 0
        self.curr_episode = 0

        self.state_visits = None
        self.reset_state_visits()

        self.trained = False

    def _increment_step(self):
        self.steps_done += 1
        self.eps_schedule.make_step()

    def _get_q_value_dict(self):
        return {
            self.get_state_key(tuple(e)): np.zeros(len(self.actions), dtype=np.float)
            for k in range(len(self.actions)) for e in it.combinations(self.actions, k)
        }

    @staticmethod
    def get_state_key(removed_edges: Tuple) -> Tuple:
        return tuple(np.sort(list(removed_edges)))

    def reset_state_visits(self):
        self.state_visits = {key: 0 for key in self.q_values.keys()}

    def step(self, state: Tuple[int], action_idx: int) -> Tuple[Tuple[int], float]:
        # TODO: Consider scaling the probabilities of not-allowed actions
        g_prime = self.base_graph.copy()

        try:
            edge_idx = self.actions[action_idx]
            if edge_idx in state:
                raise ActionAlreadyTakenError(
                    f"Cannot choose same action twice {action_idx} is already active in {state}"
                )
            next_state = state + (edge_idx,)
            g_prime.es[list(next_state)]['active'] = 0
            reward = self.reward.evaluate(g_prime)
            self.state_visits[self.get_state_key(next_state)] += 1

        except ActionAlreadyTakenError:
            reward = self.wrong_action_reward
            next_state = self.starting_state

        self._increment_step()
        return next_state, reward

    # choose an action based on epsilon greedy algorithm
    def choose_action(self, state: tuple, epsilon: float) -> int:
        available_actions = [action_idx for action_idx, action in enumerate(self.actions)
                             if action not in list(state)]

        assert len(available_actions) > 0

        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(available_actions)
        else:
            values_ = self.q_values[state]
            choosable_actions = [action_ for action_, value_ in enumerate(values_)
                                 if value_ == np.max(values_[available_actions]) and action_ in available_actions]
            return np.random.choice(choosable_actions)

    @abc.abstractmethod
    def train(self, return_rewards_over_episodes: bool = True, verbose: bool = True) -> Optional[List[float]]:
        raise NotImplementedError()

    def save_model(self, fpath: Union[str, os.PathLike]):
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(fpath: Union[str, os.PathLike]):
        with open(fpath, 'rb') as f:
            return pickle.load(f)

    def inference(self) -> Tuple[List[float], List[int]]:
        if not self.trained:
            raise RuntimeError("Please run the training before inference")

        ord_state = self.get_state_key(self.starting_state)
        rewards_per_removal = []
        edges_removed = []

        for i in range(len(self.actions)):
            action_idx = self.choose_action(ord_state, 0)
            next_state, reward = self.step(ord_state, action_idx)
            edges_removed.append(next_state[-1])
            ord_state = self.get_state_key(next_state)
            rewards_per_removal.append(reward)

        final_state = list(edges_removed)

        return rewards_per_removal, final_state
