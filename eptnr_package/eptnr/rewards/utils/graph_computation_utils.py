import igraph as ig
import pandas as pd
import numpy as np


def _evaluate_graph(g: ig.Graph) -> pd.DataFrame:
    nb_nodes = g.vs.select(type_eq='rc_node')
    poi_nodes = g.vs.select(type_eq='poi_node')

    shortest_paths_tt = g.shortest_paths(nb_nodes, poi_nodes, weights='tt')

    # Travel Time
    tt_mx = np.array(shortest_paths_tt)
    # Assign max over both dimensions to inf values
    tt_mx[tt_mx == np.inf] = tt_mx.max(1).max()

    poi_node_names = [f'POI_{name}' if 'POI' not in name else name for name in poi_nodes['name']]
    rc_node_names = [f'RC_{name}' if 'RC' not in name else name for name in nb_nodes['name']]

    df_tt = pd.DataFrame(tt_mx, columns=poi_node_names)
    df_tt['rc'] = rc_node_names

    return df_tt


def _generate_samples(metric_df: pd.DataFrame, inh_per_group: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(metric_df, inh_per_group, how='left', left_on='rc', right_on='neighborhood')

    city = pd.DataFrame(columns=['metric_value', 'group'])

    for group in merged_df.loc[:, merged_df.columns.str.contains('n_')].columns:
        samples_group = np.repeat(merged_df['average'].values, merged_df[group].to_numpy())
        samples_group = samples_group.reshape((-1, 1))

        labels = np.repeat(group.replace('n_', ''), samples_group.shape[0])
        labels = labels.reshape((-1, 1))

        group_individuals = np.hstack([samples_group, labels])
        df = pd.DataFrame(group_individuals, columns=['metric_value', 'group'])
        city = pd.concat([city, df], axis=0)

    return city


def get_tt_df(g: ig.Graph, census_data: pd.DataFrame) -> pd.DataFrame:
    metric_df = _evaluate_graph(g)
    groups_census = census_data.drop(columns=['n_inh', 'geometry'])

    metric_df['average'] = metric_df.loc[:, metric_df.columns.str.contains('POI')].mean(axis=1)
    tt_samples = _generate_samples(metric_df, groups_census)

    return tt_samples
