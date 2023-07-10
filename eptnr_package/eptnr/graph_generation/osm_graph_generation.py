import osmnx as ox
import networkx as nx
from .utils.osm_utils import get_bbox
from pathlib import Path
from datetime import datetime
from ..constants.osm_network_types import OSMNetworkTypes
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OSMGraphGenerator:

    def __init__(self, city: str, network_type: OSMNetworkTypes, graph_out_path: Path):
        self.city = city
        self.bbox = get_bbox(self.city)
        self.network_type = network_type.value
        self.graph_out_path = graph_out_path

    def generate_and_store_graph(self) -> Path:
        file_path = self.graph_out_path.joinpath(f'{self.city}_osm_graph_{datetime.now().date().isoformat()}.gpickle')

        if not file_path.exists():
            g = ox.graph_from_bbox(self.bbox['south'], self.bbox['north'], self.bbox['west'], self.bbox['east'],
                                   network_type=self.network_type, simplify=True)

            nx.write_gpickle(g, file_path)
        return file_path
