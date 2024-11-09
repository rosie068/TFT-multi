import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict, Tuple, Optional
import copy
import math
from omegaconf import OmegaConf,DictConfig

from functools import partial
from tqdm import tqdm
import torch.nn.init as init

from typing import Dict, List, Union, Callable, Optional
from IPython.display import display
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter


def aggregate_weights(output_arr: np.ndarray,
                      prctiles: List[float],
                      feat_names: List[str]) -> pd.DataFrame:
    prctiles_agg = []
    for q in prctiles: 
        if len(output_arr.shape) > 2:
            flatten_time = output_arr.reshape(-1, output_arr.shape[-1])
        else: 
            flatten_time = output_arr
        prctiles_agg.append(np.percentile(flatten_time, q=q, axis=0))

    agg_df = pd.DataFrame({prctile: aggs for prctile, aggs in zip(prctiles, prctiles_agg)})
    agg_df.index = feat_names

    return agg_df


def display_selection_weights_stats(outputs_dict: Dict[str, np.ndarray],
                                    prctiles: List[float],
                                    mapping: Dict,
                                    sort_by: Optional[float] = None,
                                    ):
    if not sort_by:
        sort_by = prctiles[0]
    else:
        assert sort_by in prctiles, "Cannot sort by a percentile which was not listed"

    for name, config in mapping.items():
        weights_agg = aggregate_weights(output_arr=outputs_dict[config['arr_key']],
                                        prctiles=prctiles,
                                        feat_names=config['feat_names'])
        print(name)
        print('=========')
        display(weights_agg.sort_values([sort_by], ascending=False).style.background_gradient(cmap='viridis'))


def display_attention_scores(attention_scores: np.ndarray,
                             horizons: Union[int, List[int]],
                             prctiles: Union[float, List[float]],
                             unit: Optional[str] = 'Units'):
    
    if not isinstance(horizons, list):
        horizons = [horizons]
    if not isinstance(prctiles, list):
        prctiles = [prctiles]

    assert len(prctiles) == 1 or len(horizons) == 1

    attn_stats = {}
    for prctile in prctiles:
        attn_stats[prctile] = np.percentile(attention_scores, q=prctile, axis=0)

    fig, ax = plt.subplots(figsize=(20, 5))

    if len(prctiles) == 1: 
        relevant_prctile = prctiles[0]
        title = f"Multi-Step - Attention ({relevant_prctile}% Percentile)"
        scores_percentile = attn_stats[relevant_prctile]
        for horizon in horizons: 
            siz = scores_percentile.shape
            x_axis = np.arange(siz[0] - siz[1], siz[0])
            ax.plot(x_axis, scores_percentile[horizon - 1], lw=1, label=f"t + {horizon} scores", marker='o')

    else:
        title = f"{horizons[0]} Steps Ahead - Attention Scores"
        for prctile, scores_percentile in attn_stats.items():  # for each percentile
            siz = scores_percentile.shape
            x_axis = np.arange(siz[0] - siz[1], siz[0])
            ax.plot(x_axis, scores_percentile[0], lw=1, label=f"{prctile}%", marker='o')

    ax.axvline(x=0, lw=1, color='r', linestyle='--')
    ax.grid(True)
    ax.set_xlabel(f"Relative Time-step [{unit}]")
    ax.set_ylabel('Attention Scores')
    ax.set_title(title)
    ax.legend()
    plt.show()


def display_target_trajectory(signal_history: np.ndarray,
                              signal_future: np.ndarray,
                              model_preds: np.ndarray,
                              observation_index: int,
                              model_quantiles: List[float],
                              transformation: Optional[Callable] = None,
                              unit: Optional[str] = 'Units'):
    
    past = signal_history[observation_index, ...]
    future = signal_future[observation_index, ...]
    preds = model_preds[observation_index, ...]

    win_len = past.shape[0]
    max_horizon = future.shape[0]

    if transformation is None:
        transformation = lambda x: x

    fig, ax = plt.subplots(figsize=(20, 5))

    past_x = np.arange(1 - win_len, 1) 
    fut_x = np.arange(1, max_horizon + 1)

    ax.plot(past_x, transformation(past[np.newaxis, ...]).reshape(-1), lw=3, label='observed', marker='o')
    ax.plot(fut_x, transformation(future[np.newaxis, ...]).reshape(-1), lw=3, label='target', marker='o')

    # for each predicted quantile, plot the quantile prediction
    for idx, quantile in enumerate(model_quantiles):
        ax.plot(fut_x, transformation(preds[np.newaxis, ..., idx]).reshape(-1), linestyle='--', lw=2, marker='s',
                label=f"predQ={quantile}")
    ax.fill_between(fut_x,
                    transformation(preds[np.newaxis, ..., 0]).reshape(-1),
                    transformation(preds[np.newaxis, ..., -1]).reshape(-1),
                    color='gray', alpha=0.3, label=None)
    # add a line at time 0
    ax.axvline(x=0.5, linestyle='--', lw=3, label=None, color='k')
    ax.grid(True)
    ax.set_xlabel(f"Relative Time-Step [{unit}]")
    ax.set_ylabel('Target Variable')
    ax.legend()
    plt.show()


def display_sample_wise_attention_scores(attention_scores: np.ndarray,
                                         observation_index: int,
                                         horizons: Union[int, List[int]],
                                         unit: Optional[str] = None):
    if isinstance(horizons, int):
        horizons = [horizons]

    sample_attn_scores = attention_scores[observation_index, ...]

    fig, ax = plt.subplots(figsize=(25, 10))

    attn_shape = sample_attn_scores.shape
    x_axis = np.arange(attn_shape[0] - attn_shape[1], attn_shape[0])

    for step in horizons:
        ax.plot(x_axis, sample_attn_scores[step - 1], marker='o', lw=3, label=f"t+{step}")

    ax.axvline(x=-0.5, lw=1, color='k', linestyle='--')
    ax.grid(True)
    ax.legend()

    ax.set_xlabel('Relative Time-Step ' + (f"[{unit}]" if unit else ""))
    ax.set_ylabel('Attention Score')
    ax.set_title('Attention Mechanism Scores - Per Horizon')
    plt.show()


def display_sample_wise_selection_stats(weights_arr: np.ndarray,
                                        observation_index: int,
                                        feature_names: List[str],
                                        top_n: Optional[int] = None,
                                        title: Optional[str] = '',
                                        historical: Optional[bool] = True,
                                        rank_stepwise: Optional[bool] = False):
    num_temporal_steps = None

    weights_shape = weights_arr.shape
    num_features = weights_shape[-1]
    is_temporal: bool = len(weights_shape) > 2

    top_n = min(num_features, top_n) if top_n else num_features

    sample_weights = weights_arr[observation_index, ...]

    if is_temporal:
        num_temporal_steps = weights_shape[1]
        sample_weights_trans = sample_weights.T
        weights_df = pd.DataFrame({'weight': sample_weights_trans.mean(axis=1)}, index=feature_names)
    else:
        weights_df = pd.DataFrame({'weight': sample_weights}, index=feature_names)

    fig, ax = plt.subplots(figsize=(20, 10))
    weights_df.sort_values('weight', ascending=False).iloc[:top_n].plot.bar(ax=ax)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(11)
        tick.label.set_rotation(45)

    ax.grid(True)
    ax.set_xlabel('Feature Name')
    ax.set_ylabel('Selection Weight')
    ax.set_title(title + (" - " if title != "" else "") + \
                 f"Selection Weights " + ("Aggregation " if is_temporal else "") + \
                 (f"- Top {top_n}" if top_n < num_features else ""))
    plt.show()

    if is_temporal:
        order = sample_weights_trans.mean(axis=1).argsort()[::-1]

        ordered_weights = sample_weights_trans[order]
        ordered_names = [feature_names[i] for i in order.tolist()]

        if rank_stepwise:
            ordered_weights = np.argsort(ordered_weights, axis=0)

        fig, ax = plt.subplots(figsize=(30, 20))

        if historical:
            map_x = {idx: val for idx, val in enumerate(np.arange(- num_temporal_steps, 1))}
        else:
            map_x = {idx: val for idx, val in enumerate(np.arange(1, num_temporal_steps + 1))}

        def format_fn(tick_val, tick_pos):
            if int(tick_val) in map_x:
                return map_x[int(tick_val)]
            else:
                return ''

        im = ax.pcolor(ordered_weights, edgecolors='gray', linewidths=2)
        ax.yaxis.set_ticks(np.arange(len(ordered_names)))
        ax.set_yticklabels(ordered_names)

        ax2 = ax.twiny()
        ax2.set_xticks([])
        ax2.xaxis.set_ticks_position('top')
        ax.set_xlabel(('Historical' if historical else 'Future') + ' Steps')
        ax2.set_xlabel(('Historical' if historical else 'Future') + ' Steps')

        ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
        fig.colorbar(im, orientation="horizontal", pad=0.05, ax=ax2)
        plt.show()