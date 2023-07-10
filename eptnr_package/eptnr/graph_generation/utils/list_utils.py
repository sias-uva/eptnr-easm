from typing import List, Union


def check_all_lists_of_same_length(*args: List) -> None:
    if len(set(map(len, args))) not in (0, 1):
        raise ValueError('not all lists have same length!')
