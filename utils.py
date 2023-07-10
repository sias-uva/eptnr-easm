from pathlib import Path
import os

import igraph as ig
import pandas as pd
import geopandas as gpd
from shapely import wkt
from typing import List
from matplotlib import pyplot as plt
from eptnr.plotting.data_exploration import plot_travel_time_histogram, get_melted_tt_df
import matplotlib
import streamlit as st
from eptnr.rewards import EgalitarianTheilReward
import math


@st.cache_data
def load_filenames(folder_path='.'):
    return os.listdir(folder_path)


@st.cache_resource
def load_graph(gml_file_path: Path):
    """Load GML file and return igraph object"""
    graph: ig.Graph = ig.read(gml_file_path)
    return graph


@st.cache_data
def load_census_data(census_file):
    """Load parquet census file and return pandas DataFrame"""
    census_data = gpd.read_parquet(census_file)
    return census_data


# @st.cache_data
def compute_equality(graph: ig.Graph, census_data: gpd.GeoDataFrame):
    reward = EgalitarianTheilReward(census_data)
    equality = -reward.evaluate(graph)
    return equality


@st.cache_data
def get_edge_types(g):
    """Return a list of unique edge types in the graph"""
    df = g.get_edge_dataframe()
    return df['type'].unique().tolist()


@st.cache_data
def get_available_vertex_names(g, edge_types: List[str] = None, origin_vertex: ig.Vertex = None,
                               target_vertex: ig.Vertex = None):
    """Returns a list of vertices"""
    edges = g.es.select(type_in=edge_types)
    if origin_vertex and target_vertex:
        raise ValueError("Cannot set both origin and target vertices")

    if origin_vertex:
        edges = edges.select(origin=origin_vertex)
    if target_vertex:
        edges = edges.select(origin=target_vertex)

    return set([e.source for e in edges] + [e.target for e in edges])


@st.cache_data
def remove_edges(g, edge_list):
    """Remove selected edges from the graph and return updated graph"""
    g_copy = g.copy()
    g_copy.delete_edges(edge_list)
    return g_copy


# @st.cache_data
def get_tt_stats(g: ig.Graph, census_data: gpd.GeoDataFrame, round_to_digit: int = -1):
    melted_tt_df = get_melted_tt_df(g, census_data)
    out = melted_tt_df.groupby('group')['travel time'].agg(['mean', 'median', 'var']).reset_index()
    if round_to_digit > 0:
        out = out.round(round_to_digit)
    return out


# @st.cache_data
def get_equality_hist_plot(graph: ig.Graph, census_data: gpd.GeoDataFrame):
    fig, ax = plt.subplots(figsize=(10, 10))
    base_hist = plot_travel_time_histogram(graph, census_data, fig, ax)
    equality_hist = base_hist[0]
    return equality_hist


# @st.cache_data
def plot_map(graph: ig.Graph, census_data: gpd.GeoDataFrame, removed_edges: List[int] = []):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    g_transit = graph.subgraph_edges(graph.es.select(type_ne='walk'))

    base = census_data.boundary.plot(figsize=(15, 15), edgecolor="purple", alpha=0.3, ax=ax)
    _ = ig.plot(g_transit, target=base, edge_curved=[0] * len(g_transit.es), vertex_color=[(0, 0, 0, 0.1)],
                vertex_size=2)

    arrows = [e for e in base.get_children() if
              isinstance(e, matplotlib.patches.FancyArrowPatch)]  # This is a PathCollection

    label_set = False
    for j, (arrow, edge) in enumerate(zip(arrows, g_transit.es)):
        if edge.index in removed_edges:
            arrow.set_color('tomato')
            arrow.set_linewidth(3)
            # Make sure label is only set once
            if not label_set:
                arrow.set_label('removed')
                label_set = True
        else:
            arrow.set_color('gray')
            arrow.set_alpha(0.8)

    return fig


@st.cache_resource
def get_reduced_graph(graph: ig.Graph, edges_to_remove: List[int]):
    g_reduced = graph.copy()
    g_reduced.delete_edges(edges_to_remove)
    return g_reduced


def save_plot(fig: plt.Figure, filename: str):
    fig.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
