# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import combinations


def generate_combs(vector, k_start, k_end):
    collector = []
    for r in range(k_start, k_end):
        variants = [el for el in combinations(vector, r)]
        collector.append(variants)
    collector = sum(collector, [])
    return collector


def add_infinity(bins: tuple) -> list:
    return [-np.inf] + list(bins) + [np.inf]


def check_mono(vector: np.ndarray) -> bool:
    """
    This function defines if vector is monotonic in any direction
    :param vector: array with data
    :return: True if monotonic else False
    """
    diffs = np.diff(vector)
    mono_inc = np.all(diffs > 0)
    mono_dec = np.all(diffs < 0)
    mono_any = mono_dec | mono_inc
    return mono_any


def gini_index(events, non_events):
    """
    Calculates Gini index in SAS format
    Args:
        events (np.array): Vector of good group sizes
        non_events (np.array): Vector of non-event group sizes
    Returns:
        Gini index (float)
    """
    p1 = float(2 * sum(events[i] * sum(non_events[:i]) for i in range(1, len(events))))
    p2 = float(sum(events * non_events))
    p3 = float(events.sum() * non_events.sum())
    coefficient = 1 - ((p1 + p2) / p3)
    index = coefficient * 100
    return index


def calculate_stats(y: np.ndarray, total_non_events: int, total_events: int, adaptor: float = 0.5):
    # global stats
    _size = len(y)
    _events = y.sum()
    _non_events = _size - _events

    # t_values
    t_events = _events / total_events
    t_non_events = _non_events / total_non_events

    # adapted t values
    t_events = adaptor if t_events == 0 else t_events
    t_non_events = adaptor if t_non_events == 0 else t_non_events
    woe_value = np.log(t_non_events / t_events)
    return _size, _events, _non_events, woe_value


def calculate_overall_stats(bins: np.ndarray, y: np.ndarray, total_non_events: int, total_events: int):
    """
    Calculate overall statistics per each bin
    :param bins:
    :param y:
    :param total_non_events:
    :param total_events:
    :return:
    """
    bin_stats = []

    for bin_v in sorted(np.unique(bins)):
        _y = y[bins == bin_v]
        bin_stats.append((bin_v,)+calculate_stats(_y, total_non_events, total_events))

    bin_stats = pd.DataFrame(bin_stats, columns=["idx", "size", "events", "non_events", "woe_value"]).set_index("idx")
    bin_stats["local_event_rate"] = bin_stats["events"] / bin_stats["size"]
    return bin_stats
