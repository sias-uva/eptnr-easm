import igraph
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from ..rewards.utils.graph_computation_utils import get_tt_df


def get_melted_tt_df(graph: igraph.Graph, census: pd.DataFrame):
    tt_df = get_tt_df(graph, census)
    temp_tt_df = tt_df.copy()
    categories = pd.Categorical(temp_tt_df['group'])
    temp_tt_df['group'] = categories
    temp_tt_df = temp_tt_df.pivot(columns='group')

    melted_temp_tt = temp_tt_df.melt()[['group', 'value']]
    melted_temp_tt.value = pd.to_numeric(melted_temp_tt.value)
    melted_temp_tt['travel time'] = melted_temp_tt.value
    del melted_temp_tt['value']

    return melted_temp_tt


def plot_travel_time_histogram(graph: igraph.Graph, census: pd.DataFrame, fig=None, ax=None,
                               min_x=None, max_x=None, min_y=None, max_y=None):

    melted_temp_tt = get_melted_tt_df(graph, census)
    group_set = melted_temp_tt['group'].unique()

    if not ax:
        fig, ax = plt.subplots()
    sns.histplot(
        melted_temp_tt,
        x='travel time',
        hue='group',
        multiple='dodge',
        stat='proportion',
        kde=True,
        shrink=.75,
        bins=100,
        palette='Set1',
        ax=ax
    )
    ax.set_xlabel('Travel Time')

    ymax = ax.get_ylim()[1]

    for group, color in zip(group_set, sns.color_palette('Set1')[:len(group_set)]):
        ax.vlines(
            melted_temp_tt[melted_temp_tt['group'] == group]['travel time'].mean(),
            ymin=0,
            ymax=ymax,
            linestyles="dashed",
            colors=color,
            label=f"avg.tt. {group}",
        )

    # Set the last quantile to be the 99th quantile
    nn_percentile = melted_temp_tt.quantile(0.99)['travel time']
    ax.set_xlim(None, nn_percentile)

    if (min_x is not None) and (max_x is not None):
        ax.set_xlim(min_x, max_x)
    if (min_y is not None) and (max_y is not None):
        ax.set_ylim(min_y, max_y)

    return fig, ax
