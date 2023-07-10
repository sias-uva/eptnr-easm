import os
from typing import List
from pathlib import Path
from zipfile import ZipFile
from enum import Enum
import math
from tqdm import tqdm

import networkx as nx
import urbanaccess as ua
import pandas as pd
import subprocess
import numpy as np
from haversine import haversine, Unit

from .utils.graph_helper_utils import (
    ua_transit_network_to_nx,
    append_length_attribute,
)
from .utils.osm_utils import get_bbox
from ..exceptions.graph_generation_exceptions import GraphGenerationError
from .utils.file_management_utils import (
    remove_files_in_dir,
)
from ..constants.gtfs_network_types import GTFSNetworkTypes
import igraph as ig
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GTFSGraphGenerator:

    def __init__(self, city: str, gtfs_zip_file_path: Path, out_dir_path: Path,
                 day: str, time_from: str, time_to: str,
                 agencies: List[str] = None, contract_vertices: bool = False,
                 modalities: List[str] = None) -> None:
        self.city = city
        bbox = get_bbox(city)
        # (lng_max, lat_min, lng_min, lat_max)
        self.bbox = (bbox['west'], bbox['south'], bbox['east'], bbox['north'])
        self.gtfs_file_path = gtfs_zip_file_path
        self.out_dir_path = out_dir_path
        self.agencies = agencies
        self.day = day
        self.time_from = time_from
        self.time_to = time_to
        self.contract_vertices = contract_vertices
        self.modalities = modalities

    def _filter_gtfs(self):
        out_path = self.gtfs_file_path.parent \
            .joinpath(f"{self.gtfs_file_path.with_suffix('').name}-filtered-by-{'_'.join(self.agencies)}.zip")

        if not os.path.exists(out_path):
            # Needed to make sure that the agencies that we want to filter for
            # are actually available
            with ZipFile(self.gtfs_file_path, 'r') as gtfs:
                with gtfs.open('agency.txt') as agencies:
                    df = pd.read_csv(agencies)
                    available_agencies = set(df.agency_id.to_list())

            available_agencies = available_agencies.intersection(set(self.agencies))
            args = [e for sublist in [["-extract-agency", agency] for agency in available_agencies] for e in sublist]
            tool = "transitland"
            subprocess.run([tool, "extract", *args, self.gtfs_file_path, out_path])

        return out_path

    def _loaf_ua_feed(self):
        with ZipFile(self.gtfs_file_path) as ref:
            # First extract all GTFS files as UA cannot load ZIPs directly
            # Could use tmpfile but it's a major pain and the out dir is there anyway
            ref.extractall(self.out_dir_path)
            # Load feed into memory
            loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=str(self.out_dir_path.absolute()),
                                                       validation=True,
                                                       verbose=True,
                                                       bbox=self.bbox,
                                                       remove_stops_outsidebbox=True,
                                                       append_definitions=True)
            # Remove all the extracted files as they are in memory now
            remove_files_in_dir(self.out_dir_path, 'txt')
        return loaded_feeds

    def generate_and_store_graph(self) -> Path:
        # If agencies are provided filter GTFS by them
        if self.agencies:
            self.gtfs_file_path = self._filter_gtfs()

        # Check if file exists already
        with ZipFile(self.gtfs_file_path, 'a') as ref:
            with ref.open('calendar_dates.txt', 'r') as calendar_dates:
                calendar_dates_df = pd.read_csv(calendar_dates)

        # Extract the date from the current GTFS file
        dates = calendar_dates_df['date'].unique()
        gml_file_name = f"{self.city}_pt_network_monday_{dates.min()}_{dates.max()}.gml"
        gml_out_path = self.out_dir_path.joinpath(gml_file_name)

        if not gml_out_path.exists():
            # Load GTFS data
            loaded_feeds = self._loaf_ua_feed()

            # Create the transit network graph from GTFS feeds using the urbanaccess library
            try:
                transit_net = ua.gtfs.network.create_transit_net(
                    gtfsfeeds_dfs=loaded_feeds,
                    calendar_dates_lookup={'unique_feed_id': f"{self.out_dir_path.name}_1"},
                    day=self.day,
                    timerange=[self.time_from, self.time_to],
                )

                # Generate transit graph WITHOUT headways
                G_transit = ua_transit_network_to_nx(transit_net)
                G_transit = append_length_attribute(G_transit)

                def get_distance(origin_station_id, destination_station_id, trip_id):
                    osid = origin_station_id
                    dsid = destination_station_id
                    tid = trip_id

                    stdf = loaded_feeds.stop_times
                    dist = stdf[stdf.trip_id == tid]
                    d_origin = dist[dist['stop_id'] == osid]['shape_dist_traveled'].max()
                    d_destination = dist[dist['stop_id'] == dsid]['shape_dist_traveled'].max()
                    distance = np.round(abs(d_destination - d_origin), decimals=2)

                    if math.isinf(distance):
                        sdf = loaded_feeds.stops
                        origin_loc = sdf[sdf['stop_id'] == osid][['stop_lat', 'stop_lon']].tuple()
                        destination_loc = sdf[sdf['stop_id'] == dsid][['stop_lat', 'stop_lon']].tuple()
                        distance = haversine(origin_loc, destination_loc, unit=Unit.METERS)
                    return distance

                # Add edge attributes
                edge_attrs = {}
                for node1, node2, key, data in tqdm(G_transit.edges(keys=True, data=True)):
                    rt = GTFSNetworkTypes(int(data['route_type'])).name

                    osid = node1.split('_')[0]
                    dsid = node2.split('_')[0]
                    tid = data['unique_trip_id'].split('_')[0]

                    dist = get_distance(osid, dsid, tid)

                    rid = loaded_feeds.trips[loaded_feeds.trips.trip_id == tid].unique_route_id.tolist()[0]
                    rid = rid.split('_')[0]
                    trip_ids = loaded_feeds.trips[loaded_feeds.trips.route_id == rid].trip_id
                    stop_times_filter = loaded_feeds.stop_times.trip_id.isin(trip_ids)
                    all_trips_on_a_route: pd.DataFrame = loaded_feeds.stop_times[stop_times_filter]

                    all_trips_on_a_route['orig_stop_id'] = all_trips_on_a_route['stop_id']
                    del all_trips_on_a_route['stop_id']

                    dests = all_trips_on_a_route.groupby('trip_id')['orig_stop_id'].shift(-1)
                    dests_str_list = dests.astype('Int64', errors='ignore').astype('str', errors='ignore').tolist()
                    all_trips_on_a_route['dest_stop_id'] = dests_str_list

                    trip_count = all_trips_on_a_route[
                        (all_trips_on_a_route.orig_stop_id == osid) &
                        (all_trips_on_a_route.dest_stop_id == dsid)].shape[0]

                    trip_count += all_trips_on_a_route[
                        (all_trips_on_a_route.dest_stop_id == dsid) &
                        (all_trips_on_a_route.orig_stop_id == osid)].shape[0]

                    entry = {
                        'type': rt,
                        'tt': np.round(data['travel_time'], decimals=2),
                        'distance': dist,
                        'name': data['unique_route_id'] + '_' + str(data['sequence']),
                        'color': 'BLACK',
                    }
                    edge_attrs[(node1, node2, key)] = entry

                malformed_edge_attr = {k:v for k, v in edge_attrs.items() if math.isnan(v['distance']) or math.isinf(v['distance'])}

                route_types = {data['type'] for _, data in edge_attrs.items()}
                avg_speeds = {}

                for rt in route_types:
                    rt_entries = {k:v for k, v in edge_attrs.items() if v['type'] == rt and not math.isnan(v['distance'])}
                    rt_tt = np.array([data['tt'] for _, data in rt_entries.items()])
                    rt_dist = np.array([data['distance'] for _, data in rt_entries.items()])
                    rt_speeds = rt_dist / rt_tt
                    logger.debug(f"For type {rt}, mean is {np.average(rt_speeds)} with std {np.std(rt_speeds)}")
                    avg_speeds[rt] = np.average(rt_speeds)

                for k, v in malformed_edge_attr.items():
                    tt = v['tt']
                    distance = tt * avg_speeds[v['type']]
                    edge_attrs[k].update({'distance': distance})

                nx.set_edge_attributes(G_transit, edge_attrs)

                # Add node attributes
                node_attrs = {
                    node: {
                        'name': data['stop_name'],
                        'color': 'BLUE',
                        'modality_type': GTFSNetworkTypes(int(data['route_type'])).name.lower(),
                    } for node, data in G_transit.nodes(data=True)
                }

                nx.set_node_attributes(G_transit, node_attrs)

                if self.modalities:
                    non_modality_nodes = [n for n, v in G_transit.nodes(data=True)
                                          if v['modality_type'] not in self.modalities]
                    G_transit.remove_nodes_from(non_modality_nodes)

                # Contract vertices if requested
                if self.contract_vertices:
                    ig_G_transit = ig.Graph.from_networkx(G_transit)
                    g_stop_name_clustering = ig.clustering.VertexClustering.FromAttribute(ig_G_transit, "stop_name")
                    membership = g_stop_name_clustering.membership
                    # Is in-place
                    ig_G_transit.contract_vertices(membership, combine_attrs='first')
                    ig.write(ig_G_transit, gml_out_path)
                else:
                    nx.write_gml(G_transit, gml_out_path)

            except Exception as e:
                logger.error(str(e))
                raise GraphGenerationError(self.out_dir_path)

        return gml_out_path
