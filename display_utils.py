import igraph as ig
import geopandas as gpd
import streamlit as st
import math
import os
from pathlib import Path
from utils import (
    compute_equality,
    get_equality_hist_plot,
    plot_map,
    get_tt_stats,
)

demo_groups = {
    'nw': 'Non White',
    'w': 'White',
}


def display_file_selector(label: str, type: str = None, folder_path: str = '.') -> Path:
    filelist = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filename = os.path.join(root, file)
            filelist.append(filename)
    if type:
        filelist = [f for f in filelist if type in f]
    selected_filename = st.selectbox(label, filelist)
    return Path(os.path.join(folder_path, selected_filename))


def display_city_metrics(graph: ig.Graph, census_data: gpd.GeoDataFrame, reduced_graph: ig.Graph = None):
    st.write("### City Information")

    col1_numbers, col2_numbers = st.columns(2)

    col1_numbers.metric("Residential centroids", len(graph.vs.select(type_eq='rc_node')))
    col2_numbers.metric("Points of interest", len(graph.vs.select(type_eq='poi_node')))

    st.write("### Transit Information")
    col1_pt, col2_pt = st.columns(2)

    ptn_stations_base = len(graph.vs.select(type_eq='pt_node'))
    ptn_edges_base = len(graph.es.select(type_ne='walk'))

    if reduced_graph:
        ptn_stations_new = len(reduced_graph.vs.select(type_eq='pt_node'))
        ptn_edges_new = len(reduced_graph.es.select(type_ne='walk'))

        col1_pt.metric("Public transit stations", ptn_stations_base, ptn_stations_new - ptn_stations_base)
        col2_pt.metric("Public transit edges", ptn_edges_base, ptn_edges_new - ptn_edges_base)
    else:
        col1_pt.metric("Public transit stations", ptn_stations_base)
        col2_pt.metric("Public transit edges", ptn_edges_base)

    st.write("### Demographics")
    col1_inh, col2_inh, col3_inh = st.columns(3)

    total_inhabitants = census_data['n_inh'].sum()
    col1_inh.metric("Total inhabitants", total_inhabitants)
    col2_inh.metric("White", census_data['n_w'].sum())
    col3_inh.metric("Non-white", census_data['n_nw'].sum())


def display_equality_metric(graph: ig.Graph, census_data: gpd.GeoDataFrame, reduced_graph: ig.Graph = None):
    st.write("### Equality")
    equality = compute_equality(graph, census_data)
    total_inhabitants = census_data['n_inh'].sum()
    # st.write(f"The Theil T index indicates the prevalent inequality of access to the socio-economic POIs.")
    # st.write(f"The closer the index is to 0, the more equal access is distributed. "
    #          f"The closer it is to {round(math.log(total_inhabitants), 4)} (maximum), "
    #          f"the more unequal the distribution of access.")
    if reduced_graph:
        equality_new = compute_equality(reduced_graph, census_data)
        st.metric(label="Theil T Inequality", value=round(equality_new, 4), delta=round(equality_new - equality, 4),
                  delta_color='inverse')
    else:
        st.metric(label="Theil T Inequality", value=round(equality, 4))


def display_tt_hist(graph: ig.Graph, census_data: gpd.GeoDataFrame):
    st.write("### Travel time distribution")
    hist = get_equality_hist_plot(graph, census_data)
    st.pyplot(hist)
    return hist


def display_per_group_tt_metrics(graph: ig.Graph, census_data: gpd.GeoDataFrame, reduced_graph: ig.Graph = None):
    st.write("### Per-group travel time:")
    base_stats_df = get_tt_stats(graph, census_data, 4)
    col1_stats, col2_stats, col3_stats = st.columns(3)

    reduced_stats_df = get_tt_stats(reduced_graph, census_data, 4) if reduced_graph else None

    for group in base_stats_df['group']:
        base_mean = base_stats_df[base_stats_df['group'] == group]['mean']
        base_median = base_stats_df[base_stats_df['group'] == group]['median']
        base_var = base_stats_df[base_stats_df['group'] == group]['var']

        if reduced_graph:
            reduced_mean = reduced_stats_df[reduced_stats_df['group'] == group]['mean']
            reduced_median = reduced_stats_df[reduced_stats_df['group'] == group]['median']
            reduced_var = reduced_stats_df[reduced_stats_df['group'] == group]['var']

            delta_mean = reduced_mean.item() - base_mean.item()
            delta_mean = round(delta_mean, 4)
            col1_stats.metric(f"Mean {demo_groups[group]}", reduced_mean, delta_mean, delta_color='inverse')

            delta_median = reduced_median.item() - base_median.item()
            delta_median = round(delta_median, 4)
            col2_stats.metric(f"Median {demo_groups[group]}", reduced_median, delta_median, delta_color='inverse')

            delta_var = reduced_var.item() - base_var.item()
            delta_var = round(delta_var, 4)
            col3_stats.metric(f"Variance {demo_groups[group]}", reduced_var, delta_var, delta_color='inverse')
        else:
            col1_stats.metric(f"Mean {demo_groups[group]}", base_mean)
            col2_stats.metric(f"Median {demo_groups[group]}", base_median)
            col3_stats.metric(f"Variance {demo_groups[group]}", base_var)


def display_graph(graph: ig.Graph, census_data: gpd.GeoDataFrame, removed_edges: list[int] = []):
    st.write("### Visualization")
    graph_plot_fig = plot_map(graph, census_data, removed_edges)
    st.pyplot(graph_plot_fig)
    return graph_plot_fig
