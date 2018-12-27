# -*- coding: utf-8 -*-
"""
TODO: В файл вынести статические функции, которые будет необходимо оптимизировать на C
Какие функции вообще есть и какие выносим:
make_edges
generate_combs
generate_layer_variant
add_infinity
check_mono
split_by_edges
gini_index
calc_descriptive_from_df
"""
import numpy as np
import pandas as pd
from itertools import combinations


def make_edges(X, cuts, unique=True):
    """
    Initial edges binning
    :param X: numpy numeric vector
    :param cuts: amount of cuts to be binned
    :param unique: unique flag for binning
    :return:
    """
    needs_unique = False
    edges_space = np.linspace(0, 1, num=cuts)
    # Делаем проверку потому, что нарезка может быть неудачной из за слишком большого перевеса
    try:
        splits, edges = pd.qcut(X, q=cuts, retbins=True)
    except:
        print("Too oversampled dataset for qcut, will be used only unique values for splitting")
        needs_unique = True
    if needs_unique:
        splits, edges = pd.qcut(np.unique(X), q=cuts, retbins=True)
    edges = np.array([-np.inf] + list(edges[1:-1]) + [np.inf])
    if unique:
        edges = np.unique(edges)
    return edges

def generate_combs(vector, k, k_start=1):
    """
    Генерирует перестановки в виде:
        C(n,1) + C(n,2) + ... + C(n,k)
    Args:
        vector (np.array):
            Вектор для генерации перестановок
        k (int):
            Макс. размер перестановок
        k_start (int):
            С какого k начинать.
    """
    collector = []
    for r in range(k_start, k + 1):
        variants = [el for el in combinations(vector, r)]
        collector.append(variants)
    collector = sum(collector, [])
    return collector


def add_infinity(vector):
    """
    Adds inf values to vector
    Args:
        vector (np.array):
            Object to add infs
    Returns:
        vector (np.array):
            object with inf
    """
    inf_vector = np.concatenate([[-np.inf], list(vector), [np.inf]])
    return inf_vector


def check_mono(vector):
    """
    This function defines does vector is monotonic
    Args:
        vector (np.array): Vector Array of data
    Returns:
        is_mono (bool)
    """
    diffs = np.diff(vector)
    mono_inc = np.all(diffs > 0)
    mono_dec = np.all(diffs < 0)
    mono_any = mono_dec | mono_inc
    return mono_any

def split_by_edges(vector, edges):
    """
    Splits input vector by edges and returns index of each value
    Args:
        vector (np.array): array to split
        edges (np.array): array of edges
    Returns:
        bins: (np.array): array of len(vector) with index of each element
    """
    # bins = np.digitize(vector,edges,right=True)
    bins = np.digitize(vector, edges)
    return bins

def calculate_loc_woe(vect, goods, bads):
    """
    Calculates woe in bucket
    Args:
        vector (pd.Series): Vector with keys "good" and "bad"
        goods (int): total amount of "event" in frame
        bads (int): total amount of "non-event" in frame
    """
    t_good = np.float(vect["good"]) / np.float(goods)
    t_bad = np.float(vect["bad"]) / np.float(bads)
    t_bad = 0.5 if t_bad == 0 else t_bad
    t_good = 0.5 if t_good == 0 else t_good
    return np.log(t_bad / t_good)

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
    if p3 == 0.0:
        return 0
    else:
        coefficient = 1 - ((p1 + p2) / p3)
        index = coefficient * 100
        return index


def calc_descriptive_from_vector(bins,y,total_good,total_bad):
    """
    Calculates IV/WoE + other descriptive data in df by grouper column
    Args:
        df (pd.DataFrame): dataframe with vectors X,y
        grouper (str): grouper of df
    Returns:
        woe_df with information about woe, lre and other.
    """
    df = pd.DataFrame(np.array([bins,y]).T,columns=["grp","y"])
    tg_good = df.groupby("grp")["y"].sum()
    tg_all = df.groupby("grp")["y"].count()
    tg_bad = tg_all - tg_good
    woe_df = pd.concat([tg_good, tg_bad, tg_all], axis=1)
    woe_df.columns = ["good", "bad", "total"]
    woe_df["woe"] = woe_df.apply(lambda row: calculate_loc_woe(row, total_good, total_bad), axis=1)
    woe_df["local_event_rate"] = woe_df["good"] / tg_all
    return woe_df


def check_variant(bins, y, total_good, total_bad):
    """
    Splits X vector
    Функция разбивает вектор X по edges
    Считает WoE по разбитым группам
    Проверяет является ли оно монотонным
    Если нет - gini=None
    Если да - считает gini
    :param bins:
        Вектор примененных границ групп (с бесконечностями по краям)
    :param X:
        Вектор X для разбиения
    :param y:
        Вектор y для расчета gini
    :return:
        edges:
            Исходный набор границ
        is_mono:
            Является ли разбиение монотонным
        gini:
            Если is_mono=False, возращает None
            Если is_mono=True, возвращает значение Gini Index
    """
    wdf = calc_descriptive_from_vector(bins, y, total_good, total_bad)
    if check_mono(wdf["woe"]):
        wdf = wdf.sort_values(by="local_event_rate", ascending=False)
        gini_index_value = gini_index(wdf["good"].values, wdf["bad"].values)
        return True, gini_index_value
    else:
        return False, None
