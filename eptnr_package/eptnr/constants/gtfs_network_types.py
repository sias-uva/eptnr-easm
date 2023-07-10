from enum import Enum


# https://developers.google.com/transit/gtfs/reference#routestxt
# 0 - Tram, Streetcar, Light rail. Any light rail or street level system within a metropolitan area.
# 1 - Subway, Metro. Any underground rail system within a metropolitan area.
# 2 - Rail. Used for intercity or long-distance travel.
# 3 - Bus. Used for short- and long-distance bus routes.
# 4 - Ferry. Used for short- and long-distance boat service.
# 5 - Cable tram. Used for street-level rail cars where the cable runs beneath the vehicle,
#     e.g., cable car in San Francisco.
# 6 - Aerial lift, suspended cable car (e.g., gondola lift, aerial tramway). Cable transport where cabins, cars,
#     gondolas or open chairs are suspended by means of one or more cables.
# 7 - Funicular. Any rail system designed for steep inclines.
# 11 - Trolleybus. Electric buses that draw power from overhead wires using poles.
# 12 - Monorail.


class GTFSNetworkTypes(Enum):
    TRAM = 0
    STREETCAR = 0
    LIGHT_RAIL = 0
    METRO = 1
    SUBWAY = 1
    RAIL = 2
    BUS = 3
    FERRY = 4
    CABLE_TRAM = 5
    AERIAL_LIFT = 6
    SUSPENDED_CABLE_CAR = 6
    GONDOLA_LIFT = 6
    AERIAL_TRAMWAY = 6
    FUNICULAR = 7
    TROLLEYBUS = 11
    MONORAIL = 12
