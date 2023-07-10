import osmnx as ox
from pathlib import Path


def get_bbox(city: str) -> dict:
    """Get the bounding box of a city.

    Args:
        city (str): City to get the bounding box of.

    Returns:
        dict: Dictionary containing the bounding box of the city.
    """
    gdf = ox.geocode_to_gdf({'city': city})
    return dict(
        west=gdf.loc[0, 'bbox_west'],
        south=gdf.loc[0, 'bbox_south'],
        east=gdf.loc[0, 'bbox_east'],
        north=gdf.loc[0, 'bbox_north']
    )

def get_pois_gdf(city: str, poi_types: dict) -> None:
    pois = ox.geometries_from_place(city, tags=poi_types)
    pois = pois.reset_index()
    pois = pois.set_index('osmid')
    
    pois.loc[pois.name.isnull(), 'name'] = pois.loc[pois.name.isnull(), :].apply(lambda x: str(x.name), axis=1)
    pois.loc[pois.name.duplicated(), 'name'] = pois.loc[pois.name.duplicated(), :].apply(lambda x: str(x['name']) + '_' + str(x.name), axis=1)

    assert pois['name'].is_unique
    assert pois['name'].notnull().all()
    assert pois['geometry'].is_unique
    assert pois['geometry'].notnull().all()
    
    return pois[pois['element_type'] == 'node'][['name', 'geometry']]
