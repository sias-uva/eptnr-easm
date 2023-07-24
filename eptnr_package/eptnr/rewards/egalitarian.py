import igraph as ig
import numpy as np
from inequality.theil import TheilD
from .base_reward import BaseReward
import logging
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


class EgalitarianTheilReward(BaseReward):   
    
    def __init__(self, census_data: pd.DataFrame, 
                 groups: List[str] = None,
                 verbose: bool = False) -> None:
        super().__init__(census_data, groups, verbose, reward_scaling=True)

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        tt_df = self.retrieve_tt_df(g)

        X = tt_df.drop(columns='group').astype(float).to_numpy()
        Y = tt_df.group
        theil_inequality = TheilD(X, Y).T[0] if X.sum() > 0 else 0.0
    
        return theil_inequality

    def _reward_scaling(self, reward: float) -> float:
        return -reward
