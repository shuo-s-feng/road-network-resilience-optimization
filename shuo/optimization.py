from queue import PriorityQueue

import numpy as np
np.seterr(all="raise")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import igraph
import networkx as nx

import shuo
from shuo.basics import get_list, UnionFind, PriorityQueueEntry


#############################################################
## Minimize number of infeasible trips on compressed graph ##
#############################################################
def get_estimated_trips_count(count_possible, count_impossible, total_trips, demand_matrix):
    return np.sum(demand_matrix * (count_possible / total_trips)), np.sum(demand_matrix * (count_impossible / total_trips))
def __build_mappings_for_minimize_number_of_infeasible_trips_on_compressed_graph(unaffected_graph_fn):

    # Build mapping between indices and nodes (NetworkX)
    G = nx.read_graphml(unaffected_graph_fn)
    nx_index_to_node, nx_node_to_index = {}, {}
    for i, node in enumerate(G.nodes()):
        nx_index_to_node[i] = node
        nx_node_to_index[node] = i

    # Build mapping between indices and zones
    dict_index_to_zone, dict_zone_to_index = shuo.indices_and_nodes_mapping(G)
    zones_num = len(dict_index_to_zone)

    # Build mapping between zone indices to node indices
    dict_zone_index_to_node_indices = {i: [i] for i in range(zones_num)}
    dict_node_index_to_zone_indices = {i: [i] for i in range(zones_num)}

    # Build mapping between indices and nodes (igraph)
    G = igraph.Graph.Read_GraphML(unaffected_graph_fn)
    igraph_index_to_node, igraph_node_to_index = {}, {}
    for index, node in enumerate(G.vs):
        igraph_index_to_node[index] = node['id']
        igraph_node_to_index[node['id']] = index

    # Integrated mapping
    return nx_index_to_node, nx_node_to_index, igraph_index_to_node, igraph_node_to_index, dict_zone_index_to_node_indices, dict_node_index_to_zone_indices, dict_index_to_zone, dict_zone_to_index
def __preprocess_for_minimize_number_of_infeasible_trips_on_compressed_graph(unaffected_graph_fn, affected_graph_fn, demand_mat, budget, mappings, verbose=False):
    nx_itn, nx_nti, ig_itn, ig_nti, dict_zitni, dict_nitzi, dict_itz, dict_zti = mappings
    G = igraph.Graph.Read_GraphML(unaffected_graph_fn)
    zones_num = len(dict_zitni)

    ############################
    ## Calculate initial data ##
    ############################

    # Calculate initial node distance infinity matrix on compressed unaffected graph. For each (i, j) s.t. i != j, value is 0; 1 otherwise
    init_node_dist_inf_mat = np.ones((zones_num, zones_num), dtype=int)
    for i in range(zones_num): init_node_dist_inf_mat[i][i] = 0

    # Calculate initial zone distance matrix on compressed unaffected graph. For each (i, j) s.t. i != j, distance is infinity; 0 otherwise
    init_zone_dist_mat = np.ones((zones_num, zones_num), dtype=float) * float('Inf')
    for i in range(zones_num): init_zone_dist_mat[i][i] = 0

    # Calculate initial trips count matrix on compressed unaffected graph. For each (i, j) s.t. i != j, infeasible trips number is |number of nodes in zone i| * |number of nodes in zone j|, the total trips number of (i, j)
    init_fea_tc_mat = np.zeros((zones_num, zones_num), dtype=float)
    init_inf_tc_mat = np.zeros((zones_num, zones_num), dtype=float)

    for zone_s in G.vs:
        zone_s_index = dict_zti[zone_s['id']]
        zone_s_nodes_num = len(get_list(zone_s['members']))
        for zone_t in G.vs:
            zone_t_index = dict_zti[zone_t['id']]
            if zone_s_index == zone_t_index: continue
            zone_t_nodes_num = len(get_list(zone_t['members']))

            init_inf_tc_mat[zone_s_index][zone_t_index] = zone_s_nodes_num * zone_t_nodes_num

    init_total_tc_mat = init_fea_tc_mat + init_inf_tc_mat
    # Some zones may not contain a node
    init_total_tc_mat[init_total_tc_mat == 0] = 1

    # Calculate initial estimated trips count matrix (with travel demand)
    init_est_inf_tc_sum = get_estimated_trips_count(init_fea_tc_mat, init_inf_tc_mat, init_total_tc_mat, demand_mat)[1]

    # Calculate initial candidate edges, the flooded affected edges of affected graph
    total_flooded_distance = 0
    original_candidate_edges = nx.read_graphml(affected_graph_fn).edges(data=True)
    for edge in original_candidate_edges:

        # If current edge isn't affected, add it into final (result) graph G.
        if edge[2]['flooded_distance'] == 0:
            u_index = ig_nti[edge[0]]
            v_index = ig_nti[edge[1]]
            G.add_edge(u_index, v_index, length=edge[2]["length"])
        else:
            total_flooded_distance += edge[2]['flooded_distance']


    #####################
    ## Data structures ##
    #####################

    # Union-Find for finding the affiliation of a node
    uf = UnionFind([node['id'] for node in G.vs], [get_list(edge['edge']) for edge in G.es])

    # Priority-queue data structure to greedily inspect edges
    invalid_edges = 0
    candidate_edges = PriorityQueue()
    for edge in original_candidate_edges:
        u_index = ig_nti[edge[0]]
        v_index = ig_nti[edge[1]]
        length = edge[2]['length']
        flooded_distance = edge[2]['flooded_distance']

        # If u and v belong to different connected components and cost of repairing (u, v) is lower than budget, add (u, v) into candidates
        if (uf.find(edge[0]) != uf.find(edge[1])) and (flooded_distance <= budget):
            # Negative priority value because python PriorityQueue returns lowest priority element
            candidate_edges.put(PriorityQueueEntry(-float('inf'), {'u': u_index, 'v': v_index, 'length': length, 'flooded_distance': flooded_distance, 'valid': False}))
        else:
            invalid_edges += 1
    if verbose:
        print("Total flooded distance %0.4f m" % total_flooded_distance)
        print("Budget is %0.4f m" % budget)
        print("%d edges eliminated since they are invalid" % invalid_edges)
        print("Starting with %d candidate edges" % (candidate_edges.qsize()))

    return G, uf, candidate_edges, \
        init_node_dist_inf_mat, \
        init_fea_tc_mat, \
        init_inf_tc_mat, \
        init_total_tc_mat, \
        init_est_inf_tc_sum
def __recompute_distance_and_trips_count_matrix_for_minimize_number_of_infeasible_trips_on_compressed_graph(G, old_node_dist_inf_mat, old_fea_tc_mat, old_inf_tc_mat, mappings):

    # Mappings between indices and nodes, and between zone indices and node indices
    nx_itn, nx_nti, ig_itn, ig_nti, dict_zitni, dict_nitzi = mappings

    # Current shortest path matrix
    node_dist_inf_mat = np.isinf(np.array(G.shortest_paths(source=None, target=None, weights="length")))


    #############################################################
    ## Calculate the zones which need to recompute trips count ##
    #############################################################
    zones_to_recompute = []

    # Get newly reachable pairs of nodes
    changes = np.logical_xor(old_node_dist_inf_mat, node_dist_inf_mat)
    zone_s_indices, zone_t_indices = np.where(changes == True)

    # Enumerate each newly reachable pairs of nodes
    for zone_s_index, zone_t_index in zip(zone_s_indices, zone_t_indices):

        # find which zones i is associated with
        node_index = nx_nti[ig_itn[zone_s_index]]
        for zone_index in dict_nitzi[node_index]:
            zones_to_recompute.append(zone_index)

        # find which zones j is associated with
        node_index = nx_nti[ig_itn[zone_t_index]]
        for zone_index in dict_nitzi[node_index]:
            zones_to_recompute.append(zone_index)


    ###############################
    ## Recompute the trips count ##
    ###############################
    zones_num = len(dict_zitni)
    mask = np.zeros((zones_num, zones_num), dtype=bool)
    # Tempory infeasible or feasible trips count matrix
    tem_inf_tc_mat = np.zeros((zones_num, zones_num), dtype=float)
    tem_fea_tc_mat = np.zeros((zones_num, zones_num), dtype=float)

    # Enumerate each pair of nodes within each pair of zones
    for zone_s_index in zones_to_recompute:
        node_s_indices = dict_zitni[zone_s_index]
        for zone_t_index in zones_to_recompute:
            node_t_indices = dict_zitni[zone_t_index]

            for node_s_index in node_s_indices:
                for node_t_index in node_t_indices:
                    s = ig_nti[nx_itn[node_s_index]]
                    t = ig_nti[nx_itn[node_t_index]]

                    # Update trips count matrix
                    if node_dist_inf_mat[s, t]:
                        tem_inf_tc_mat[zone_s_index, zone_t_index] += 1
                    else:
                        tem_fea_tc_mat[zone_s_index, zone_t_index] += 1
                    mask[zone_s_index, zone_t_index] = True


    # New trips count matrix is the combination of old one and tempory one
    fea_tc_mat = old_fea_tc_mat.copy()
    fea_tc_mat[mask] = tem_fea_tc_mat[mask]
    inf_tc_mat = old_inf_tc_mat.copy()
    inf_tc_mat[mask] = tem_inf_tc_mat[mask]

    return node_dist_inf_mat, fea_tc_mat, inf_tc_mat
def minimize_number_of_infeasible_trips_on_compressed_graph(unaffected_graph_fn, affected_graph_fn, demand_matrix_fn, budget, by_ratio=True, verbose=False):
    #####################
    ## Preprocess data ##
    #####################

    mappings = __build_mappings_for_minimize_number_of_infeasible_trips_on_compressed_graph(unaffected_graph_fn)
    ig_itn = mappings[2]

    demand_mat = np.load(demand_matrix_fn)

    G, uf, candidate_edges, \
    cur_node_dist_inf_mat, \
    cur_fea_tc_mat, \
    cur_inf_tc_mat, \
    cur_total_tc_mat, \
    cur_est_inf_tc = __preprocess_for_minimize_number_of_infeasible_trips_on_compressed_graph(unaffected_graph_fn, affected_graph_fn, demand_mat, budget, mappings, verbose)
    og_est_inf_tc = cur_est_inf_tc

    ########################
    ## Greedily add edges ##
    ########################

    # Initialize variables
    cur_G = G.copy()
    added_edges_num = 0
    added_edges_list = []
    evaluations_num = 0
    remaining_budget = budget
    while (remaining_budget > 0) and (not candidate_edges.empty()):

        # Set each entry of priority queue to be invalid. Note, each "get()" will remove the element from the queue
        tmp_edges = []
        while not candidate_edges.empty():
            elem = candidate_edges.get()
            elem.data['valid'] = False
            tmp_edges.append(elem)
        for edge in tmp_edges:
            candidate_edges.put(edge)

        # Process the next valid edge (with highest priority) or preprocessing all invalid edges
        while True:
            cur_edge = candidate_edges.get()

            # If current edge is valid, add it into final graph
            if cur_edge.data['valid']:
                u_index = cur_edge.data['u']
                v_index = cur_edge.data['v']
                remaining_budget -= cur_edge.data['flooded_distance']
                added_edges_num += 1

                # Add the current edge into corresponding data structures
                cur_G.add_edge(u_index, v_index, length=cur_edge.data['length'])
                added_edges_list.append([u_index, v_index])
                uf.merge(ig_itn[u_index], ig_itn[v_index])

                # Remove edges that are no longer within budget or no longer between different connected components
                tmp_edges = []
                while not candidate_edges.empty():
                    elem = candidate_edges.get()
                    if (uf.find(ig_itn[elem.data['u']]) != uf.find(ig_itn[elem.data['v']])) and (elem.data['flooded_distance'] <= remaining_budget):
                        tmp_edges.append(elem)
                for edge in tmp_edges:
                    candidate_edges.put(edge)

                if verbose and added_edges_num % 10 == 0:
                    print('Added %d edges, budget of %0.4f remaining' % (added_edges_num, remaining_budget))
                    print("%d edges left as candidates" % candidate_edges.qsize())

                # Recompute the effect of adding this edge
                cur_node_dist_inf_mat, cur_fea_tc_mat, cur_inf_tc_mat = __recompute_distance_and_trips_count_matrix_for_minimize_number_of_infeasible_trips_on_compressed_graph(cur_G, cur_node_dist_inf_mat, cur_fea_tc_mat, cur_inf_tc_mat, mappings[:-2])
                cur_est_inf_tc = get_estimated_trips_count(cur_fea_tc_mat, cur_inf_tc_mat, cur_total_tc_mat, demand_mat)[1]

                # if added_edges_num % 10 == 0:
                # cur_G.write_graphml("%s_iter_%d.graphml" % (output_fn_pattern, added_edges_num))

                # Since we've added this edge, all of matrix need to recompute
                break

            # If current edge is invalid, validate it
            else:
                evaluations_num += 1

                # Calculate the effect of adding this edge
                tmp_G = cur_G.copy()
                tmp_G.add_edge(cur_edge.data['u'], cur_edge.data['v'], length=cur_edge.data['flooded_distance'])
                tmp_fea_tc_mat, tmp_inf_tc_mat = __recompute_distance_and_trips_count_matrix_for_minimize_number_of_infeasible_trips_on_compressed_graph(tmp_G, cur_node_dist_inf_mat, cur_fea_tc_mat, cur_inf_tc_mat, mappings[:-2])[1:]
                tmp_est_inf_tc_mat = get_estimated_trips_count(tmp_fea_tc_mat, tmp_inf_tc_mat, cur_total_tc_mat, demand_mat)[1]
                improvement = cur_est_inf_tc - tmp_est_inf_tc_mat

                # Update the priority of this edge, and add it into candidates
                delta_priority = improvement / cur_edge.data['flooded_distance'] if by_ratio else improvement
                cur_edge.priority = -delta_priority
                cur_edge.data['valid'] = True
                candidate_edges.put(cur_edge)

    print("Added a total of %d edges, with %0.4f budget remaining" % (added_edges_num, remaining_budget))
    print('Original infeasible trips:', og_est_inf_tc, 'Remaining:', cur_est_inf_tc, 'Recovered:', og_est_inf_tc - cur_est_inf_tc)
    #cur_G.write_graphml("%s_iter_%d.graphml" % (output_fn_pattern, added_edges_num))
    print()

    '''
        Note:
            The added edges given lower budget are not necessary the subset of edges given higher budget.
            For instance, 3 edges:
                1. benefit: 5; cost: 2
                2. benefit: 6; cost: 4
                3. benefit: 2; cost: 1
            Therefore, at current round of greedy algorithm, 
                if the left budget is 3: we choose edge 1 (maybe and edge 3)
                if the left budget is 4: we choose edge 2
    '''

    return cur_G, added_edges_list
def graph_of_minimizing_number_of_infeasible_trips(original_unaffected_graph_fn, original_affected_graph_fn, compressed_unaffected_graph_fn, compressed_affected_graph_fn, demand_matrix_fn, budget, by_ratio=True, verbose=False):
    # Calculate the edges list to be added (repaired)
    added_edges_list = minimize_number_of_infeasible_trips_on_compressed_graph(compressed_unaffected_graph_fn, compressed_affected_graph_fn, demand_matrix_fn, budget, by_ratio, verbose)[1]

    OUG = nx.read_graphml(original_unaffected_graph_fn)
    OAG = nx.read_graphml(original_affected_graph_fn)
    CAG = nx.read_graphml(compressed_affected_graph_fn)
    dict_index_to_zone = list(CAG.nodes)

    # Recover the result on compressed graph into the original graph
    NG = OUG.copy()
    for edge in added_edges_list:
        u, v = edge
        zone_u = dict_index_to_zone[u]
        zone_v = dict_index_to_zone[v]

        og_edge = get_list(CAG[zone_u][zone_v]['edge'])

        NG.add_edge(og_edge[0], og_edge[1])
        NG[og_edge[0]][og_edge[1]].update(OAG[og_edge[0]][og_edge[1]])
    return NG
def residual_graph_of_minimizing_number_of_infeasible_trips(original_unaffected_graph_fn, original_affected_graph_fn, compressed_unaffected_graph_fn, compressed_affected_graph_fn, demand_matrix_fn, budget, by_ratio=True, verbose=False):
    # Calculate the edges list to be added (repaired)
    added_edges_list = minimize_number_of_infeasible_trips_on_compressed_graph(compressed_unaffected_graph_fn, compressed_affected_graph_fn, demand_matrix_fn, budget, by_ratio, verbose)[1]

    OUG = nx.read_graphml(original_unaffected_graph_fn)
    OAG = nx.read_graphml(original_affected_graph_fn)
    CAG = nx.read_graphml(compressed_affected_graph_fn)
    dict_index_to_zone = list(CAG.nodes)

    # Recover the result on compressed graph into the original graph
    NG = nx.Graph()
    NG.add_nodes_from(OUG)
    for edge in added_edges_list:
        u, v = edge
        zone_u = dict_index_to_zone[u]
        zone_v = dict_index_to_zone[v]

        og_edge = get_list(CAG[zone_u][zone_v]['edge'])

        NG.add_edge(og_edge[0], og_edge[1])
        NG[og_edge[0]][og_edge[1]].update(OAG[og_edge[0]][og_edge[1]])
    return NG


#############################################
## Minimize number of connected components ##
#############################################
def graph_of_minimizing_number_of_connected_components(UG, AG, budget=5):
    flooded_length_key = 'flooded_distance'
    graph = [[u, v, d] for u, v, d in AG.edges(data=True)]
    graph = sorted(graph, key=lambda item: item[2][flooded_length_key])

    UF = UnionFind(UG.nodes, UG.edges)

    budget_left = budget
    fortified_edges_list = []
    for edge in graph:
        fu = UF.find(edge[0])
        fv = UF.find(edge[1])

        if fu != fv:
            if budget_left >= edge[2][flooded_length_key]:
                budget_left -= edge[2][flooded_length_key]
                fortified_edges_list.append(edge)
                UF.merge(fu, fv)
            else:
                break

    NG = UG.copy()
    NG.add_edges_from(fortified_edges_list)

    #last = fortified_edges_list[len(fortified_edges_list) - 1]
    #print(last[0], last[1])


    print('Given budget', budget - budget_left, ', after fortifying', len(fortified_edges_list), 'edges, there are', nx.number_connected_components(NG), 'connected components:')
    #components_count = dict()
    #for c in sorted(nx.connected_components(NG), key=len, reverse=True):
        #num = len(c)
        #if not num in components_count:
            #components_count[num] = 1
        #else:
            #components_count[num] += 1

    #for k, v in components_count.items():
        #print('size:', k, 'number:', v)
    print()

    return NG


