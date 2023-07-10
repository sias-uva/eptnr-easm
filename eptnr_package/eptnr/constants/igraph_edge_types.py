from enum import Enum


class IGraphEdgeTypes(Enum):
    """
    Edge types for igraph edges can be any of the GTFS and OSM network types. As both of them are not prone to
    change in the near future, this is some duplicated information to disentangle the workflow.
    """
    WALK = "walk"
    BIKE = "bike"
    DRIVE = "drive"
    TRAM = "tram"
    STREETCAR = "streetcar"
    LIGHT_RAIL = "light_rail"
    METRO = "metro"
    SUBWAY = "subway"
    RAIL = "rail"
    BUS = "bus"
    FERRY = "ferry"
    CABLE_TRAM = "cable_tram"
    AERIAL_LIFT = "aerial_lift"
    SUSPENDED_CABLE_CAR = "suspended_cable_car"
    GONDOLA_LIFT = "gondola_lift"
    AERIAL_TRAMWAY = "aerial_tramway"
    FUNICULAR = "funicular"
    TROLLEYBUS = "trolleybus"
    MONORAIL = "monorail"
