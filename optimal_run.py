from pathlib import Path
import json
import random

import numpy as np
import torch
import pandas as pd
import geopandas as gpd
import igraph as ig

import matplotlib.pyplot as plt
import seaborn as sns

from eptnr.rewards.egalitarian import EgalitarianTheilReward
from eptnr.plotting.data_exploration import plot_travel_time_histogram, get_melted_tt_df
from eptnr.algorithms.baselines import optimal_max_baseline

import click
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.option('--edge_types', type=click.Choice(['METRO', 'BUS', 'TRAM', 'TRAIN']),
              multiple=True, default={'METRO'}, help='Type of edge')
@click.option('--budget', type=click.IntRange(0, 10), default=2, help='Maximal edge budget [int]')
@click.option('--groups', type=click.Choice(['w', 'nw']), default=None, multiple=True, help='Demographic groups')
@click.argument('graph_gml_file_path', type=click.Path(exists=True))
@click.argument('census_gdf_file_path', type=click.Path(exists=True))
@click.option('--random_seed', type=int, default=2048, help='Random seed value')
@click.option('--output_file', type=str, default='./results/optimal_run.json', help='Output file for optimal run')
def run_experiment(edge_types, budget, groups, graph_gml_file_path, census_gdf_file_path, random_seed, output_file):

    if Path(output_file).exists():
        logger.info(f"Skipping run as outputfile exists ({output_file})")
        return

    # Load Data
    graph: ig.Graph = ig.read(graph_gml_file_path)
    census: gpd.GeoDataFrame = gpd.read_parquet(census_gdf_file_path)

    # Set that all groups that should be considered and the edge types that can be removed
    edge_types = set(edge_types)

    reward = EgalitarianTheilReward(census_data=census, groups=groups, verbose=False)

    def set_seeds(seed: int = 2048):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Get total number of metro edges
    nr_total_edges = len(graph.es.select(type_in=edge_types))
    # Reduce by certain amount of percent
    print(
        f"Budget is {budget}/{nr_total_edges} of graph edges of "
        f"type{'s' if len(edge_types) > 1 else ''} {', '.join(edge_types)}"
    )

    set_seeds(random_seed)
    opt_edges = optimal_max_baseline(g=graph, reward=reward, edge_types=edge_types, budget=budget, verbose=False)

    edges_to_remove = opt_edges[0][1]
    t_inequality = opt_edges[0][0]
    opt_reduced_graph = graph.copy()
    opt_reduced_graph.delete_edges(edges_to_remove)

    logger.info(f"Removed edge{'s' if len(edges_to_remove) > 1 else ''} {str(edges_to_remove)} resulting "
                f"in a Theil inequality of {t_inequality}")

    def get_tt_stats(g: ig.Graph, census_df: pd.DataFrame):
        melted_tt_df = get_melted_tt_df(g, census_df)
        return melted_tt_df.groupby('group')['travel time'].agg(['mean', 'median', 'var']).reset_index()

    tt_df = get_tt_stats(opt_reduced_graph, census)
    logger.info('\n' + tt_df.to_string(index=False))

    logger.info(f"Storing JSON with results in {output_file}")
    json.dump({'edges_to_remove': edges_to_remove}, open(output_file, 'w'))


if __name__ == '__main__':
    run_experiment()
