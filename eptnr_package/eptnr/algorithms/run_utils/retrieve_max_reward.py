from typing import List, Tuple
import numpy as np


def get_max_solutions(sol: List[Tuple[List[float], List[float]]]):
    final_sol = []
    for entry in sol:
        max_idx = np.argmax(entry[0])
        final_sol.append((entry[0][0:max_idx+1], entry[1][0:max_idx+1]))
    return final_sol
