import os
import osmnx as ox
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString

from shuo.basics import haversine_distance, get_LineString, get_list
from shuo.basics import UnionFind


# [Input]
#   1. query: query string for requesting NetworkX Mulitidigraph data of a city from OSMnx
# [Output]
#   1. MDG: NetworkX Multidigraph of road network of the queried city
#   2. MG: NetworkX Multigraph of road network of the queried city
#   3. G: NetworkX Graph of road network of the queried city
def graph(query):
    # Download the Multidigraph data by OSMnx
    MDG = ox.graph_from_place(query, 'drive')
    cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    ox.save_graphml(MDG, 'tmp.graphml', cur_dir)

    # Convert Multidigraph to Multigraph by NetworkX
    MG = nx.to_undirected(nx.read_graphml(cur_dir + 'tmp.graphml'))
    os.remove(cur_dir + 'tmp.graphml')

    # Convert Multigraph to Graph by NetworkX
    G = nx.Graph()
    # Add node data into new graph NG
    for node, data in MG.nodes(data=True):
        G.add_node(node)
        G.nodes[node].update(data)
    # Add edge data into new graph NG
    for u, v, data in MG.edges(data=True):
        old_length_att = 'length'
        new_length_att = 'weight'
        data[old_length_att] = float(data[old_length_att])

        if G.has_edge(u, v):
            G[u][v][new_length_att] = min(G[u][v][new_length_att], data[old_length_att])
        else:
            G.add_edge(u, v)
            data[new_length_att] = data[old_length_att]
            del data[old_length_att]
            G[u][v].update(data)

    # Remove the self-loops
    # NG.remove_edges_from(NG.selfloop_edges())

    return MDG, MG, G
# [Input]
#   1. UG, AG: Unaffected and affected graph
#   2. total_trips_mat: total trips counting matrix at zone level
#   3. demand_mat: travel demand matrix at zone level
# [Output]
#   1. CUG: Compressed unaffected graph
#   2. CAG: Compressed affected graph
#   3. compressed_demand_mat: compressed travel demand at connected component level
def compressed_graph(UG, AG, total_trips_mat, demand_mat, mappings, edge_length_att='weight', edge_flooded_length_att='flooded_distance'):
    assert len(mappings) == 2
    dict_node_to_ind, dict_zone_ind_to_node_incs = mappings
    node_num, zone_num = len(UG.nodes), len(dict_zone_ind_to_node_incs)

    ###############################################################
    ## Calculate the cheapest edges between connected components ##
    ###############################################################
    uf = UnionFind(UG.nodes, UG.edges)
    cheapest_edges = dict()
    for edge in AG.edges(data=True):
        root_u, root_v  = uf.find(edge[0]), uf.find(edge[1])

        if root_u == root_v: continue

        # If this pair of roots has no edge, add the current edge
        if ((root_u, root_v) not in cheapest_edges) and ((root_v, root_u) not in cheapest_edges):
            cheapest_edges[(root_u, root_v)] = edge
        else:
            key = (root_u, root_v) if (root_u, root_v) in cheapest_edges else (root_v, root_u)

            # If current edge is cheaper than the saved edge, replace the old one
            if edge[2][edge_flooded_length_att] < cheapest_edges[key][2][edge_flooded_length_att]:
                cheapest_edges[key] = edge


    #################################
    ## Create the compressed graph ##
    #################################
    CAG = nx.Graph()    # Compressed Flooded Affected Graph
    CUG = nx.Graph()    # Compressed Flooded Unaffected Graph

    # Add nodes into compressed graph
    for nodes in nx.connected_components(UG):
        root = uf.find(list(nodes)[0])
        CAG.add_node(root, members=str(nodes), x=UG.nodes[root]['x'], y=UG.nodes[root]['y'])
        CUG.add_node(root, members=str(nodes), x=UG.nodes[root]['x'], y=UG.nodes[root]['y'])

    # Add edges into compressed graph
    for edge in cheapest_edges:
        root_u, root_v = edge[0], edge[1]
        edge = cheapest_edges[(root_u, root_v)]
        CAG.add_edge(root_u, root_v, edge=str([edge[0], edge[1]]), length=edge[2][edge_length_att], flooded_distance=edge[2][edge_flooded_length_att])


    ###################################################
    # Calculate estimated travel demand at node level #
    ###################################################
    node_demand_mat = np.zeros((node_num, node_num), dtype=float)

    # For each pair of (zone s, zone t)
    for zone_s_ind in range(zone_num):
        for zone_t_ind in range(zone_num):
            total_trips = total_trips_mat[zone_s_ind][zone_t_ind]
            demand = demand_mat[zone_s_ind][zone_t_ind]

            #if (total_trips == 0) and (demand != 0): print(zone_s_ind, zone_t_ind)
            if (total_trips == 0) or (demand == 0): continue

            node_demand = demand / total_trips
            node_s_incs, node_t_incs = dict_zone_ind_to_node_incs[zone_s_ind], dict_zone_ind_to_node_incs[zone_t_ind]
            for node_s_ind in node_s_incs:
                for node_t_ind in node_t_incs:
                    # Node assignment check
                    #if node_demand_mat[node_s_ind][node_t_ind] != 0: print(node_s, node_t)
                    node_demand_mat[node_s_ind][node_t_ind] = node_demand

    ############################################
    ## Create compressed travel demand matrix ##
    ############################################
    roots_num = len(uf.roots())
    dict_mega_node_to_index = {node: index for index, node in enumerate(CAG.nodes())}

    # Demand of mega-nodes (m_u, m_v) = total demand of nodes (u, v) belonging to m_u and m_v respectively.
    compressed_demand_mat = np.zeros((roots_num, roots_num), dtype=float)
    for mega_u in CAG.nodes(data=True):
        mega_u_ind = dict_mega_node_to_index[mega_u[0]]
        for mega_v in CAG.nodes(data=True):
            mega_v_ind = dict_mega_node_to_index[mega_v[0]]
            #if mega_u_ind == mega_v_ind: continue

            # Demand of nodes (u, v) = Demand of (zone_u, zone_v) / Total trips number of (zone_u, zone_v)
            for u in get_list(mega_u[1]['members']):
                for v in get_list(mega_v[1]['members']):
                    if u == v: continue

                    node_u, node_v = dict_node_to_ind[u], dict_node_to_ind[v]
                    compressed_demand_mat[mega_u_ind][mega_v_ind] += node_demand_mat[node_u][node_v]

    return CUG, CAG, compressed_demand_mat
# [Input]
#   1. G: NetworkX Graph of road network
# [Output]
#   1. dict_index_to_node: mapping from index to node id
#   2. dict_node_to_index: mapping from node id to index
def indices_and_nodes_mapping(G):
    dict_index_to_node = list(G.nodes)  # Mapping from index to node
    dict_node_to_index = {node: index for index, node in enumerate(G.nodes)}
    return dict_index_to_node, dict_node_to_index
# [Input]
#   1. G: NetworkX Graph of road network
#   2. dict_node_to_index: mapping from node id to index
# [Output]
#   1. dist_mat: node distance (nodes num * nodes num) matrix
def nodes_distances(G, dict_node_to_index):
    nodes_num = len(G.nodes)
    dist_mat = np.ones((nodes_num, nodes_num), dtype=float) * np.Inf  # Output distance matrix
    for i in range(nodes_num):
        dist_mat[i][i] = 0

    # Build a integer-indexing Graph
    NG = nx.Graph()
    for node, data in G.nodes(data=True):
        node_ind = dict_node_to_index[node]
        NG.add_node(node_ind)
        NG.nodes[node_ind].update(data)
    for u, v, data in G.edges(data=True):
        u_ind = dict_node_to_index[u]
        v_ind = dict_node_to_index[v]
        NG.add_edge(u_ind, v_ind)
        NG[u_ind][v_ind].update(data)

    # Calculate the shortest distances between all pairs of nodes
    distances = nx.all_pairs_dijkstra_path_length(NG, weight='length')

    # Reformat as distance matrix
    for source, values in distances:
        for target, distance in values.items():
            dist_mat[source][target] = distance

    return dist_mat
# [Input]
#   1. zone_gdf: GeoDataFrame of zone
#   2. zone_att: attribute name indicating zone id in "zone_gdf"
# [Output]
#   1. dict_index_to_zone: mapping from index to zone id
#   2. dict_zone_to_index: mapping from zone id to index
def indices_and_zones_mapping_from_GDF(zone_gdf, zone_att):
    dict_index_to_zone = []
    dict_zone_to_index = dict()

    for index, row in zone_gdf.iterrows():
        zone = row[zone_att]
        dict_index_to_zone.append(zone)
        dict_zone_to_index[zone] = index

    return dict_index_to_zone, dict_zone_to_index
# [Input]
#   1. G: NetworkX Graph of road network
#   2. zone_gdf: GeoDataFrame of zone
#   3. zone_att: attribute name indicating zone id in "zone_gdf"
# [Output]
#   1. dict_zone_to_nodes: mapping from zone id to node ids contained in this zone
#   2. dict_node_to_zones: mapping from node id to zone ids containing this node
def zones_and_nodes_mapping_from_GDF(G, zone_gdf, zone_att):
    dict_zone_to_nodes = dict()
    dict_node_to_zones = dict()

    for index, row in zone_gdf.iterrows():
        zone = row['geometry']  # Polygon made from points in "row['geometry']"
        zone_id = row[zone_att]
        nodes_list = []         # List of nodes of road network which is in this "zone"
        for node, data in G.nodes(data=True):
            if Point(float(data['x']), float(data['y'])).intersects(zone):
                nodes_list.append(node)
                if not node in dict_node_to_zones:
                    dict_node_to_zones[node] = [zone_id]
                else:
                    dict_node_to_zones[node].append(zone_id)
        dict_zone_to_nodes[zone_id] = nodes_list

    # Check the nodes which has no affiliating zone. Assign it to its nearest zone
    for node, data in G.nodes(data=True):
        if not node in dict_node_to_zones:
            point = Point(float(data['x']), float(data['y']))
            min_dist = 1000000
            address = ''

            for index, row in zone_gdf.iterrows():
                dist = point.distance(row['geometry'].centroid)
                if min_dist > dist:
                    min_dist = dist
                    address = row[zone_att]
            dict_zone_to_nodes[address].append(node)
            dict_node_to_zones[node] = [address]

    return dict_zone_to_nodes, dict_node_to_zones
# [Input]
#   1. G: NetworkX Graph of road network
#   2. zone_gdf: GeoDataFrame of zone
# [Output]
#   1. dict_zone_to_nodes: mapping from zone index to node indices contained in this zone
#   2. dict_node_to_zones: mapping from node index to zone indices containing this node
def zone_indices_and_node_indices_mapping_from_GDF(G, zone_gdf):
    dict_zone_to_nodes = dict()
    dict_node_to_zones = dict()

    for zone_index, row in zone_gdf.iterrows():
        zone = row['geometry']  # Polygon made from points in "row['geometry']"
        nodes_list = []  # List of nodes of road network which is in this "zone"
        for node_index, node in enumerate(G.nodes(data=True)):
            if Point(float(node[1]['x']), float(node[1]['y'])).intersects(zone):
                nodes_list.append(node_index)
                if not node_index in dict_node_to_zones:
                    dict_node_to_zones[node_index] = [zone_index]
                else:
                    dict_node_to_zones[node_index].append(zone_index)
        dict_zone_to_nodes[zone_index] = nodes_list

    # Check the nodes which has no affiliating zone. Assign it to its nearest zone
    for node_index, node in enumerate(G.nodes(data=True)):
        if not node_index in dict_node_to_zones:
            point = Point(float(node[1]['x']), float(node[1]['y']))
            min_dist = 1000000
            min_addr = ''

            for zone_index, row in zone_gdf.iterrows():
                dist = point.distance(row['geometry'].centroid)
                if min_dist > dist:
                    min_dist = dist
                    min_addr = zone_index
            dict_zone_to_nodes[min_addr].append(node_index)
            dict_node_to_zones[node_index] = [min_addr]

    return dict_zone_to_nodes, dict_node_to_zones
# [Input]
#   1. node_dist_mat: node distance (nodes num * nodes num) matrix
#   2. dict_index_to_node: mapping from index to node id
#   3. dict_node_to_zones: mapping from node id to zone ids
#   4. dict_zone_to_index: mapping from zone id to index
# [Output]
#   1. zone_dist_mat: zone distance (zones num * zones num) matrix
#   2. [feasible_trips_count, infeasible_trips_count]: trips count (2 * zones_num * zones_num) matrix
def zones_distances_and_trips_count(node_dist_mat, dict_index_to_node, dict_node_to_zones, dict_zone_to_index):
    # Number of nodes and zones
    nodes_num = len(dict_index_to_node)
    zones_num = len(dict_zone_to_index)

    # Total distance matrix at zone level
    zone_total_dist_mat = np.ones((zones_num, zones_num), dtype=float) * np.Inf
    zone_total_dist_mat_count = np.zeros((zones_num, zones_num), dtype=float)

    #for i in range(zones_num):
        #zone_total_dist_mat[i][i] = 0
        #zone_total_dist_mat_count[i][i] = 1

    # Trips count matrix
    feasible_trips_count = np.zeros((zones_num, zones_num), dtype=int)
    infeasible_trips_count = np.zeros((zones_num, zones_num), dtype=int)

    # Calculate the total distances and trips count between zones
    for i in range(nodes_num):
        for j in range(nodes_num):
            if i == j: continue

            # Current pairs of nodes: "node_s" -> "node_t" with distance = "distance"
            node_s = dict_index_to_node[i]
            node_t = dict_index_to_node[j]
            distance = node_dist_mat[i][j]

            # Corresponding lists of zones to which "node_s" and "node_t" belongs
            zones_s = dict_node_to_zones[node_s]
            zones_t = dict_node_to_zones[node_t]

            # For each pair of zones: "zone_s" -> "zone_t"
            for zone_s in zones_s:
                for zone_t in zones_t:
                    # If the source zone is the same as the target zone, ignore
                    #if zone_s == zone_t: continue

                    # Indices of "zone_s" and "zone_t"
                    zone_s_ind = dict_zone_to_index[zone_s]
                    zone_t_ind = dict_zone_to_index[zone_t]

                    # Update total distance matrix
                    zone_total_dist_mat[zone_s_ind][zone_t_ind] += distance
                    zone_total_dist_mat_count[zone_s_ind][zone_t_ind] += 1

                    # Update trips count
                    # Note that, "distance == 0" between two different nodes means an infeasible trip
                    if distance != np.Inf:
                        feasible_trips_count[zone_s_ind][zone_t_ind] += 1
                    else:
                        infeasible_trips_count[zone_s_ind][zone_t_ind] += 1

    # Calculate the average distances between zones
    zone_dist_mat = np.zeros((zones_num, zones_num), dtype=float)
    for i in range(zones_num):
        for j in range(zones_num):
            if infeasible_trips_count[i][j] != 0:
                zone_dist_mat[i][j] = zone_total_dist_mat[i][j] / infeasible_trips_count[i][j]
            #else: zone_dist_mat[i][j] = math.nan

    return zone_dist_mat, [feasible_trips_count, infeasible_trips_count]
# [Input]
#   1. G: NetworkX Graph instance of road network
#   2. flood_polygons: shapely.geometry.Polygon instance of flood areas
# [Output]
#   1. G: NetworkX Graph instance of road network with flood info
#   2. flooded_edges: flood-affected edges list
def graph_with_flood_info(G, flood_polygons):
    flooded_length_key = 'flooded_distance'

    # Convert the nodes of road network into "shapely.geometry.Point"
    points = {node: Point(float(data['x']), float(data['y'])) for node, data in G.nodes(data=True)}

    # List of flooded edges
    flooded_edges = []

    # Calculate flooded part of each edge
    for u, v, data in G.edges(data=True):
        # "line" refers to the LineString of the current edge (u, v)
        # Note that, edge (u, v) contains several line segments as a curve if "data" has 'geometry'
        if 'geometry' in data:
            line = get_LineString(data['geometry'])
        else:
            line = LineString([points[u], points[v]])

        # Flooded length of the current edge (u, v)
        data[flooded_length_key] = 0

        for polygon in flood_polygons:
            try: intersection = line.intersection(polygon)
            except: continue

            # If the intersection between "line" and this flooded zone is not empty, append it into "flooded_edges"
            if not intersection.is_empty:
                flooded_edges.append(intersection)

                # If the intersection is one single line, calculate the flooded distance of this line
                if type(intersection) is LineString:
                    last_point = None
                    for point in list(intersection.coords):
                        if not last_point is None:
                            data[flooded_length_key] += haversine_distance(last_point, point)
                        last_point = point
                # If the intersection is multiple lines, calculate the total flooded distance of these lines
                elif type(intersection) is MultiLineString:
                    for line in intersection:
                        last_point = None
                        for point in list(line.coords):
                            if not last_point is None:
                                data[flooded_length_key] += haversine_distance(last_point, point)
                            last_point = point
                #else:
                    #print(type(intersection))

    return G, flooded_edges
# [Input]
#   1. G: NetworkX Graph of road network
#   2. flooded_zone_subty: dictionary storing flooding data whose keys are zone ids and values are their flooding info
#   3. flooded_zone_subty_value: value indicating that a zone is flooded
#   4. flooded_zone_geometry: shapely.geometry.Polygon of this zone (using the same indexing (zone ids) as "flooded_zone_subty")
# [Output]
#   1. G: NetworkX Graph of road network with flood info
#   2. flooded_edges: flooded affected edges list
def graph_with_flood_info_2(G, flooded_zone_subty, flooded_zone_subty_value, flooded_zone_geometry):
    flooded_length_key = 'flooded_distance'

    # Convert the nodes of road network into "shapely.geometry.Point"
    points = {node: Point(float(data['x']), float(data['y'])) for node, data in G.nodes(data=True)}

    flooded_zones_indices = [k for k, v in flooded_zone_subty.items() if v == flooded_zone_subty_value]

    # List of flooded edges
    flooded_edges = []

    # Calculate flooded part of each edge
    for u, v, data in G.edges(data=True):
        # Delete useless info from each edge
        # if 'osmid' in data: del data['osmid']

        # "line" refers to the LineString of the current edge (u, v)
        # Note that, edge (u, v) contains several line segments as a curve if "data" has 'geometry'
        if 'geometry' in data:
            line = get_LineString(data['geometry'])
        else:
            line = LineString([points[u], points[v]])

        # Flooded length of the current edge (u, v)
        data[flooded_length_key] = 0

        for index in flooded_zones_indices:
            try: intersection = line.intersection(flooded_zone_geometry[index])
            except: continue

            # If the intersection between "line" and this flooded zone is not empty, append it into "flooded_edges"
            if not intersection.is_empty:
                flooded_edges.append(intersection)

                # If the intersection is one single line, calculate the flooded distance of this line
                if type(intersection) is LineString:
                    last_point = None
                    for point in list(intersection.coords):
                        if not last_point is None:
                            data[flooded_length_key] += haversine_distance(last_point, point)
                        last_point = point
                # If the intersection is multiple lines, calculate the total flooded distance of these lines
                else:
                    for line in intersection:
                        last_point = None
                        for point in list(line.coords):
                            if not last_point is None:
                                data[flooded_length_key] += haversine_distance(last_point, point)
                            last_point = point

    return G, flooded_edges
# [Input]
#   1. G_with_flood_info: NetworkX Graph of road network with flood info
# [Ouput]
#   1. UG: corresponding flooded unaffected NetworkX Graph of road network (containing all nodes and only edges unaffected by flood)
#   2. AG: corresponding flooded affected NetworkX Graph of road network (containing all nodes and only edges affected by flood)
def unaffected_and_affected_graph(G_with_flood_info):
    # Unaffected graph
    UG = nx.Graph()
    # Affected graph
    AG = nx.Graph()

    # Add node data into G
    for node, data in G_with_flood_info.nodes(data=True):
        UG.add_node(node)
        UG.nodes[node].update(data)
        AG.add_node(node)
        AG.nodes[node].update(data)

    for u, v, data in G_with_flood_info.edges(data=True):
        if data['flooded_distance'] == 0:
            UG.add_edge(u, v)
            UG[u][v].update(data)
        else:
            AG.add_edge(u, v)
            AG[u][v].update(data)

    return UG, AG

# Get travel demand matrix from TAZ cvs
def travel_demand_from_TAZ(valid_zone_indices, travel_demand_source_fn, dict_zone_to_index=None):
    valid_zones_num = len(valid_zone_indices)
    demand_matrix = np.zeros((valid_zones_num, valid_zones_num), dtype=float)
    with open(travel_demand_source_fn) as f:
        datareader = csv.DictReader(f)
        for row in datareader:
            data = list(row.items())

            # 1st col: source TAZ id
            taz_s = int(data[0][1])
            if not taz_s in valid_zone_indices: break

            # 2nd col: target TAZ id
            taz_t = int(data[1][1])
            if not taz_t in valid_zone_indices: continue

            # 9th col: travel demand; "-1" is to get the index
            if dict_zone_to_index is None:
                demand_matrix[taz_s - 1][taz_t - 1] = float(data[8][1])
            else:
                demand_matrix[dict_zone_to_index[taz_s]][dict_zone_to_index[taz_t]] = float(data[8][1])

    return demand_matrix

# Get travel demand matrix at node level
def node_travel_demand_matrix(total_trips_mat, demand_mat, mappings):
    assert len(mappings) == 2
    dict_node_to_ind, dict_zone_ind_to_node_incs = mappings
    node_num, zone_num = len(dict_node_to_ind), len(dict_zone_ind_to_node_incs)

    ###################################################
    # Calculate estimated travel demand at node level #
    ###################################################
    node_demand_mat = np.zeros((node_num, node_num), dtype=float)

    # For each pair of (zone s, zone t)
    for zone_s_ind in range(zone_num):
        for zone_t_ind in range(zone_num):
            total_trips = total_trips_mat[zone_s_ind][zone_t_ind]
            demand = demand_mat[zone_s_ind][zone_t_ind]

            #if (total_trips == 0) and (demand != 0): print(zone_s_ind, zone_t_ind)
            if (total_trips == 0) or (demand == 0): continue

            node_demand = demand / total_trips
            node_s_incs, node_t_incs = dict_zone_ind_to_node_incs[zone_s_ind], dict_zone_ind_to_node_incs[zone_t_ind]
            for node_s_ind in node_s_incs:
                for node_t_ind in node_t_incs:
                    # Node assignment check
                    #if node_demand_mat[node_s_ind][node_t_ind] != 0: print(node_s, node_t)
                    node_demand_mat[node_s_ind][node_t_ind] = node_demand
    return node_demand_mat
