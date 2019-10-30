import os
import csv
import pickle
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
from math import radians, cos, sin, asin, sqrt

def create_folder(string):
    folder = ''
    for substr in string.split('/'):
        if substr == '' or substr == '..': continue
        folder += substr + '/'
        if not os.path.exists(folder): os.mkdir(folder)

def haversine_distance(pt1, pt2):
    lon1, lat1 = pt1
    lon2, lat2 = pt2

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    m = 1000 * km
    return m

# Get LineString from string (printed LineStirng)
def get_LineString(string):
    string = string.replace('LINESTRING', '').replace('(', '').replace(')', '').replace(',', '')

    points = []
    coords = string.split()
    for i in range(len(coords) >> 1):
        index = i << 1
        points.append(Point(float(coords[index]), float(coords[index + 1])))

    return LineString(coordinates=points)

# Get list from string (printed set or list)
def get_list(string):
    string = string.replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace(',', '').replace('\'', '')

    res = []
    nodes = string.split()
    for node in nodes:
        res.append(node)

    return res

class PickleDataWriter:
    def __init__(self, filename, clear=False):
        self.filename = filename
        if clear:
            self.clear()

    def clear(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def write(self, obj):
        with open(self.filename, "ab") as F:
            pickle.dump(obj, F)
    def write_with_truncating(self, obj):
        with open(self.filename, "wb") as F:
            pickle.dump(obj, F)

    def print_data(self):
        #num = 0
        with open(self.filename, "rb") as F:
            while True:
                try:
                    db = pickle.load(F)
                    if db is not None:
                        #print(db)
                        for key, value in db.items():
                            print(key, value)
                        #num += 1
                except EOFError:
                    break
        #print(num)

    def get_data_obj(self):
        with open(self.filename, "rb") as F:
            while True:
                try:
                    db = pickle.load(F)
                    if db is not None:
                        return db
                except EOFError:
                    break
        return None

    def get_data_obj_list(self):
        obj = []
        with open(self.filename, "rb") as F:
            while True:
                try:
                    db = pickle.load(F)
                    if db is not None:
                        obj.append(db)
                except EOFError:
                    break
        return obj

class UnionFind:

    def __init__(self, nodes, edges):
        self.__roots = None
        self.nodes = nodes
        self.__father = dict()

        for node in nodes: self.__father[node] = node

        for u, v in edges:
            fu = self.find(u)
            fv = self.find(v)
            if fu != fv: self.__father[fu] = fv

    def find(self, son):
        if self.__father[son] == son: return son
        self.__father[son] = self.find(self.__father[son])
        return self.__father[son]


    def merge(self, u, v):
        fu = self.find(u)
        fv = self.find(v)
        self.__father[fu] = fv


    def roots(self):
        #if not self.__roots is None: return self.__roots
        self.__roots = []
        for node in self.nodes:
            root = self.find(node)
            if not root in self.__roots:
                self.__roots.append(root)
        return self.__roots


    def children(self, root=None):
        if root is None:
            children = dict()
            for node in self.nodes:
                root = self.find(node)
                if not root in children:
                    children[root] = [node]
                else:
                    children[root].append(node)
        else:
            children = []
            for node in self.nodes:
                if self.find(node) == root:
                    children.append(node)
        return children


    def show_info(self):
        print('There\'re', len(self.roots()), 'connected components')

class FilesReader:
    def __init__(self, folder, city, budget):
        self.city = city
        self.folder = folder
        self.budget = budget
        self.name = FileNames(folder, city, budget)
    def update_budget(self, budget):
        self.budget = budget
        self.name = FileNames(self.folder, self.city, budget)
    def graph(self, type=0, opt=None):
        return nx.read_graphml(self.name.graph(type, opt))
    def compressed_graph(self, type=1):
        return nx.read_graphml(self.name.compressed_graph(type))
    def zone(self, type=0):
        return gpd.read_file(self.name.zone(type))
    def mapping(self, type=0):
        return PickleDataWriter(self.name.mapping(type)).get_data_obj_list()
    def flood_zone(self, type=0):
        return gpd.read_file(self.name.flood_zone(type))
    def node_distance_matrix(self, type=0, opt=None):
        return np.load(self.name.node_distance_matrix(type, opt))
    def zone_trips_count_matrix(self, type=0, opt=None):
        return np.load(self.name.zone_trips_count_matrix(type, opt))
    def travel_demand_matrix(self, type=0):
        return np.load(self.name.travel_demand_matrix(type))

class FileNames:
    def __init__(self, folder, city, budget):
        self.city = city
        self.folder = folder
        self.budget = budget
    def update_budget(self, budget):
        self.budget = budget
    def graph(self, type=0, opt=None):
        if type == -1:  return '%s/graphml/%s_original_graph_with_flood.graphml'    % (self.folder, self.city)
        elif type == 0: return '%s/graphml/%s_original_graph.graphml'               % (self.folder, self.city)
        elif type == 1: return '%s/graphml/%s_unaffected_graph.graphml'             % (self.folder, self.city)
        elif type == 2: return '%s/graphml/%s_affected_graph.graphml'               % (self.folder, self.city)
        elif type == 3:
            if self.budget == 0:
                return '%s/graphml/%s_unaffected_graph.graphml'             % (self.folder, self.city)
            else:
                if opt is None: return None
                return '%s/graphml/%s_%s_budget_%s_resulting_graph.graphml' % (self.folder, self.city, opt, self.budget)
        elif type == 4:
            if self.budget == 0:
                return '%s/graphml/%s_unaffected_graph_colored_by_connected_components.graphml'             % (self.folder, self.city)
            else:
                if opt is None: return None
                return '%s/graphml/%s_%s_budget_%s_resulting_graph_colored_by_connected_components.graphml' % (self.folder, self.city, opt, self.budget)
        else: return None
    def compressed_graph(self, type=1):
        if type == 1:   return '%s/graphml/%s_compressed_unaffected_graph.graphml'             % (self.folder, self.city)
        elif type == 2: return '%s/graphml/%s_compressed_affected_graph.graphml'               % (self.folder, self.city)
        else: return None
    def zone(self, type=0):
        if type == 0:   return '%s/shapefile/Traffic_analysis_zone/%s.shp'  % (self.folder, self.city)
        else: return None
    def mapping(self, type=0):
        if type == 0:   return '%s/mapping/%s_indices_nodes_mapping'        % (self.folder, self.city)
        elif type == 1: return '%s/mapping/%s_indices_zones_mapping'        % (self.folder, self.city)
        elif type == 2: return '%s/mapping/%s_zones_nodes_mapping'          % (self.folder, self.city)
        elif type == 3: return '%s/mapping/%s_zone_incs_node_incs_mapping'  % (self.folder, self.city)
        else: return None
    def flood_zone(self, type=0):
        if type == 0:   return '%s/shapefile/FEMA_flood_map/%s.shp'         % (self.folder, self.city)
        elif type == 1: return '%s/shapefile/EnviroAtlas_flood_map/%s.shp'  % (self.folder, self.city)
        else: return None
    def node_distance_matrix(self, type=0, opt=None):
        if type == 0:   return '%s/matrix/%s_original_node_distance_matrix.npy' % (self.folder, self.city)
        elif type == 1:
            if self.budget == 0:
                return '%s/matrix/%s_budget_%s_flooded_node_distance_matrix.npy' % (self.folder, self.city, self.budget)
            else:
                if opt is None: return None
                return '%s/matrix/%s_%s_budget_%s_flooded_node_distance_matrix.npy' % (self.folder, self.city, opt, self.budget)
        else: return None
    def zone_trips_count_matrix(self, type=0, opt=None):
        if type == 0:   return '%s/matrix/%s_original_zone_trips_count_matrix.npy' % (self.folder, self.city)
        elif type == 1:
            if self.budget == 0:
                return '%s/matrix/%s_budget_%s_flooded_zone_trips_count_matrix.npy' % (self.folder, self.city, self.budget)
            else:
                if opt is None: return None
                return '%s/matrix/%s_%s_budget_%s_flooded_zone_trips_count_matrix.npy' % (self.folder, self.city, opt, self.budget)
        else: return None
    def travel_demand_matrix(self, type=0):
        if type == 0:   return '%s/matrix/%s_original_travel_demand_matrix.npy'    % (self.folder, self.city)
        elif type == 1: return '%s/matrix/%s_compressed_travel_demand_matrix.npy'  % (self.folder, self.city)

class PriorityQueueEntry(object):
    def __init__(self, priority, data):
        self.data = data
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


