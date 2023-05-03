import geopandas as gpd
import h3
import folium
import osmnx as ox
from shapely import wkt
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import mapping
RESOLUTION = 8

import os
from pyproj import Geod


def osm_query(city, tag):
    """extracts all elements that satisfy the tag passed as an argument and returns them in a geopandas

    Args:
        city (_type_): string describing the city in question
        tag (_type_): OSM tags to filter elements

    Returns:

    """
    gdf = ox.geometries_from_place(city, tag).reset_index()
    print(gdf.shape)
    return gdf


def coordinates_to_h3(lat, lng, res=RESOLUTION):
    return h3.geo_to_h3(lat, lng, res)


# get the h3 address of the hexagone for which the geometry belongs to
def get_h3_geo(g, res=RESOLUTION):
    # longitude
    lng = g.x if isinstance(g, Point) else g.centroid.x     
    # latitude 
    lat = g.y if isinstance(g, Point) else g.centroid.y
    # get h3
    h3_addr = h3.geo_to_h3(lat=lat, lng=lng, resolution=res)
    return h3_addr, lat, lng


# this function returns the area of the passed geometry divided by the scaling constant
def get_area(geo):
    geod = Geod(ellps="WGS84") 
    return abs(geod.geometry_area_perimeter(geo)[0])


def get_h3_point(point):
    # return the h3 address of the hexagone for which the point belongs to
    h3_a, lat, lng = get_h3_geo(point)
    return [h3_a]    

def get_h3_lineString(ls: LineString):
    """this function considers all the points belonging to LineString object and returns a set of all h3 addresses of hexagones
    intersecting this geometry

    Args:
        ls (LineString): A lineString object representing a OpenStreetMap object, generally: highway, road, street...

    Returns:
        set: the set of all h3_address of the hexagones that are spanned by the LineString object
    """
    
    h3_s = set()
    x, y = ls.coords.xy
    for p_x, p_y in zip(x, y):
        # получить адрес h3 шестиугольника, к которому принадлежит эта конкретная точка
        
        h3_addr = h3.geo_to_h3(lat=p_y, lng=p_x, resolution=RESOLUTION)
        
        # добавьте адрес в набор
        h3_s.add(h3_addr)
    return h3_s

def get_h3_Poly(poly: Polygon):
    """this function considers all the points belonging to Polygon object and returns a set of all h3 addresses of hexagones
    intersecting this geometry

    Args:
        poly (Polygone): A Polygone object representing a OpenStreetMap element: large enough not be represented as a point

    Returns:
        set: the set of all h3_address of the hexagones that are spanned by the Polygon object
    """
    h3_s = set()
    x, y = poly.exterior.coords.xy
    for p_x,p_y in zip(x, y):
        # get the h3 address of the hexagone to which this particular point belongs
        h3_addr = h3.geo_to_h3(lat=p_y, lng=p_x, resolution=RESOLUTION)
        # add the address to the set
        h3_s.add(h3_addr)
    return h3_s


def get_h3_MultiPoly(m_poly: MultiPolygon):
    h3_s = set()
    for poly in m_poly.geoms:
        h3_s.update(get_h3_Poly(poly))
    return h3_s

def get_h3_area(g):
    """This function gets the area as well as the set of h3 addresses of hexagones spanned by the given geometry"""
    
    # first get the area
    area = get_area(g)
    addresses = None

    # call the corresponding function depending on the geometry's data type
        
    if isinstance(g, Point):
        addresses = get_h3_point(g)
    elif isinstance(g, LineString):
        addresses = get_h3_lineString(g)
    elif isinstance(g, Polygon):
        addresses = get_h3_Poly(g)
    else:
        addresses = get_h3_MultiPoly(g)
    return area, addresses


import pandas as pd

COLS = ['h3', 'count', 'area']

def get_count_area_tag(city, tag):
    
    # extract the data from osm
    g = osm_query(city, tag).loc[:, 'geometry']
        
    
    # h3_count: maps each h3 address to the number of elements in geopandas "g" that intersect the hexagone h3
    # h3_area: maos each h3 address to the sum of areas of elements in geopandas "g" that intersect the hexagone h3
    h3_count = {}
    h3_area = {}
                
    for geo in g:
        # get the area and the unique address associated with this element: geometry
        area, adds = get_h3_area(geo)

        # for each address
        for ad in adds:
            # if this address was encoutered before increment its count and add the area of the element to its h3_area
            if ad in h3_count:
                h3_count[ad] += 1
                h3_area[ad] += area
            else:
                # if this address if first encountered then set its count to 1 and its area to the element's
                h3_count[ad] = 1
                h3_area[ad] = area

    # convert the dictionaries into dataframe                
    res = pd.DataFrame({"h3": list(h3_area.keys()), "count": list(h3_count.values()), "area": list(h3_area.values())})
    return res

import numpy as np

def get_tags(tags, name):
    df_res = pd.DataFrame(data=[], columns=COLS)
    # for each specific tag
    for t in tags:
        # create the count and area dataframe associated with the specific dataframe
        df_tag = get_count_area_tag(t)
        # add it to previous dataframes    
        df_res = pd.concat([df_res, df_tag])
    
    # as several h3 address can repeat accross different tags: we group by h3 addresses
    # and sum up all their associated areas and counts
    
    df_res_area = pd.pivot_table(df_res, index='h3', values=['area'], aggfunc=[np.sum])
    df_res_count = pd.pivot_table(df_res, index='h3', values=['count'], aggfunc=[np.sum])
    final_df = pd.merge(df_res_count, df_res_area, right_index=True, left_index=True)
    final_df.columns = [f"{name}_count", f"{name}_area"]
    # final_df.to_excel(os.path.join("osm_features", f"{name}.xlsx"))
    return final_df
