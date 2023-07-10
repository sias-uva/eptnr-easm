# Standard library imports
import logging
import json

# External package imports
import streamlit as st

# Internal package imports (eptnr)
from eptnr.plotting.data_exploration import plot_travel_time_histogram
from utils import (
    load_graph,
    load_census_data,
    save_plot,
    get_reduced_graph,
)
from display_utils import (
    display_city_metrics,
    display_equality_metric,
    display_tt_hist,
    display_graph,
    display_per_group_tt_metrics,
    display_file_selector,
)


logger = logging.getLogger(__file__)

# Set up the Streamlit application
st.title("Equality in Public Transportation Network Removals (EPTNR)")


# Init session state
st.session_state.processed = False


with st.container():
    st.write("## Data Selection")

    gml_file_path = display_file_selector(label="Upload an EPTNR problem graph GML file", type="gml")
    census_file_path = display_file_selector(label="Upload a Census data file", type="parquet")
    opt_run_file_path = display_file_selector(label="Upload an optimal run JSON file", type="json")

    if st.button("Process"):
        st.session_state.processed = True

    if st.session_state.processed:
        # Load data into memory
        if gml_file_path is not None:
            g = load_graph(gml_file_path)
        if census_file_path is not None:
            census_data = load_census_data(census_file_path)

with st.container():
    if st.session_state.processed:
        st.write("## Base Graph")

        display_city_metrics(g, census_data)
        display_equality_metric(g, census_data)

        current_equality_hist = display_tt_hist(g, census_data)
        save_plot(current_equality_hist, 'current_equality_hist')

        display_per_group_tt_metrics(g, census_data)

        current_equality_graph = display_graph(g, census_data)
        save_plot(current_equality_graph, 'current_equality_graph')

with st.container():
    if st.session_state.processed and opt_run_file_path:
        st.write("## Optimally Reduced Graph")

        edges_to_remove = json.load(open(opt_run_file_path, 'r'))['edges_to_remove']

        g_reduced = get_reduced_graph(g, edges_to_remove)

        display_city_metrics(g, census_data, g_reduced)
        display_equality_metric(g, census_data)

        current_equality_hist = display_tt_hist(g_reduced, census_data)
        save_plot(current_equality_hist, 'new_equality_hist')

        display_per_group_tt_metrics(g, census_data, g_reduced)

        new_equality_graph = display_graph(g, census_data, edges_to_remove)
        save_plot(new_equality_graph, 'new_equality_graph')
