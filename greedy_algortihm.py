import math

import networkx as nx

import numpy as np


def _greedy_plus_plus(G, iterations):
    if G.number_of_edges() == 0:
        return 0.0, set()
    if iterations < 1:
        raise ValueError(
            f"The number of iterations must be an integer >= 1. Provided: {iterations}"
        )

    loads = dict.fromkeys(G.nodes, 0)  # Load vector for Greedy++.
    best_density = 0.0  # Highest density encountered.
    best_subgraph = set()  # Nodes of the best subgraph found.

    for _ in range(iterations):
        # Initialize heap for fast access to minimum weighted degree.
        heap = nx.utils.BinaryHeap()

        # Compute initial weighted degrees and add nodes to the heap.
        for node, degree in G.degree:
            heap.insert(node, loads[node] + degree)
        # Set up tracking for current graph state.
        remaining_nodes = set(G.nodes)
        num_edges = G.number_of_edges()
        current_degrees = dict(G.degree)

        while remaining_nodes:
            num_nodes = len(remaining_nodes)

            # Current density of the (implicit) graph
            current_density = num_edges / num_nodes

            # Update the best density.
            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(remaining_nodes)

            # Pop the node with the smallest weighted degree.
            node, _ = heap.pop()
            if node not in remaining_nodes:
                continue  # Skip nodes already removed.

            # Update the load of the popped node.
            loads[node] += current_degrees[node]

            # Update neighbors' degrees and the heap.
            for neighbor in G.neighbors(node):
                if neighbor in remaining_nodes:
                    current_degrees[neighbor] -= 1
                    num_edges -= 1
                    heap.insert(neighbor, loads[neighbor] + current_degrees[neighbor])

            # Remove the node from the remaining nodes.
            remaining_nodes.remove(node)
            

    return best_density, best_subgraph


def compute_density(G, remaining_nodes):
    H = G.subgraph(remaining_nodes)
    S = list(H.nodes)
    
    e_total = H.number_of_edges()
    
    d_orig = e_total / len(S)
    
    return d_orig

def compute_group_density(G, remaining_nodes,node_attributes,Attr):
    H = G.subgraph(remaining_nodes)
    S = list(H.nodes)
    if not S:
        return 0.0
    

    e_red = sum(1 for u, v in H.edges if node_attributes[u] == Attr or node_attributes[v] == Attr)
    

    d_red = e_red / len(S)
    
    return d_red

def compute_group_diverse_density(G, remaining_nodes, node_attributes, group_value):
    H = G.subgraph(remaining_nodes)
    S = list(H.nodes)
    if not S:
        return 0.0

    e_diverse = 0
    for u, v in H.edges:
        gu = node_attributes[u]
        gv = node_attributes[v]
        # cross edge AND incident to group_value (exactly one endpoint in the group)
        if gu != gv and (gu == group_value or gv == group_value):
            e_diverse += 1

    return e_diverse / len(S)









def compute_diverse_density(G, remaining_nodes, node_attributes, lam=0):
    # Subgraph of remaining nodes
    H = G.subgraph(remaining_nodes)
    S = list(H.nodes)
    if not S:
        return 0.0

    # Total edges
    e_total = H.number_of_edges()

    # Red-sensitive edges: at least one red endpoint
    e_red = sum(1 for u, v in H.edges if node_attributes[u] != node_attributes[v])


    d_red = e_red / len(S)

    return  d_red 




def diverse_greedy_plus_plus(G,node_attributes, iterations):
    if G.number_of_edges() == 0:
        return 0.0, set()
    if iterations < 1:
        raise ValueError(
            f"The number of iterations must be an integer >= 1. Provided: {iterations}"
        )

    loads = dict.fromkeys(G.nodes, 0)  # Load vector for Greedy++.
    best_density = 0.0  # Highest density encountered.
    best_subgraph = set()  # Nodes of the best subgraph found.

    for _ in range(iterations):
        # Initialize heap for fast access to minimum weighted degree.
        heap = nx.utils.BinaryHeap()

        # Only consider edges with at least one endpoint red (node_attributes[node]==0 or node_attributes[neighbor]==0)
        # Build a subgraph with only those edges
        red_edges = [(u, v) for u, v in G.edges if node_attributes[u] != node_attributes[v]]
        red_G = G.edge_subgraph(red_edges).copy()

        # Compute initial weighted degrees and add nodes to the heap (degree in red_G)
        for node in red_G.nodes:
            degree = red_G.degree[node]
            heap.insert(node, loads[node] + degree)

        remaining_nodes = set(red_G.nodes)
        num_edges = red_G.number_of_edges()
        current_degrees = dict(red_G.degree)

        while remaining_nodes:
            num_nodes = len(remaining_nodes)
            if num_nodes == 0:
                break

            # Current density of the (implicit) graph
            current_density = num_edges / num_nodes #need to change this

            # Update the best density.
            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(remaining_nodes)

            # Pop the node with the smallest weighted degree.
            node, _ = heap.pop()
            if node not in remaining_nodes:
                continue  # Skip nodes already removed.

            # Update the load of the popped node.
            loads[node] += current_degrees.get(node, 0)

            # Update neighbors' degrees and the heap.
            for neighbor in list(red_G.neighbors(node)):
                if neighbor in remaining_nodes:
                    # Only consider edge if at least one endpoint is red
                    if node_attributes[node] != node_attributes[neighbor]:
                        current_degrees[neighbor] -= 1
                        num_edges -= 1
                        heap.insert(neighbor, loads[neighbor] + current_degrees[neighbor])
            # Remove the node from the remaining nodes and from the graph
            remaining_nodes.remove(node)
            red_G.remove_node(node)
            

    return best_density, best_subgraph






def maxmin_group_diverse_greedy_plus_plus(G, node_attributes, iterations):
    """
    Maximize min_g |E_g^div(S)| / |S| where E_g^div(S) are diverse (cross-group) edges
    with at least one endpoint in group g (i.e., group-adjacent diverse connectivity).

    Unweighted version (same style as your old function).
    """
    if G.number_of_edges() == 0:
        return 0.0, set()
    if iterations < 1:
        raise ValueError(
            f"The number of iterations must be an integer >= 1. Provided: {iterations}"
        )

    # ignore isolated nodes (matches your previous behavior)
    base_G = G.edge_subgraph(G.edges).copy()

    # groups from attributes on remaining nodes
    groups = sorted(set(node_attributes[n] for n in base_G.nodes))

    loads = dict.fromkeys(base_G.nodes, 0)  # Load vector for Greedy++.
    best_density = -1.0
    best_subgraph = set()

    def group_adjacent_diverse_degree(node, group_value, current_degrees, neighbor_group_counts):
        """
        Diverse-incident degree of `node` w.r.t. group_value.

        We consider only diverse edges (cross-group).
        - If node is in group_value: count edges to neighbors NOT in group_value
          => deg(node) - (#neighbors in group_value)
        - Else: count edges from node to neighbors in group_value
          => #neighbors in group_value
        """
        if node_attributes[node] == group_value:
            same_g = neighbor_group_counts[node].get(group_value, 0)
            return current_degrees.get(node, 0) - same_g
        return neighbor_group_counts[node].get(group_value, 0)

    for _ in range(iterations):
        work_G = base_G.copy()

        # neighbor group-counts for all edges (needed to compute same-group counts too)
        nbr_group_cnt = {u: {} for u in work_G.nodes}
        current_degrees = dict(work_G.degree)

        # per-group adjacent diverse edge counts: |E_g^div(S)|
        num_div_edges_by_group = dict.fromkeys(groups, 0)

        # group node counts (paper-style: all groups must be present)
        group_node_counts = dict.fromkeys(groups, 0)
        for u in work_G.nodes:
            group_node_counts[node_attributes[u]] += 1

        # initialize counts
        for u, v in work_G.edges:
            gu = node_attributes[u]
            gv = node_attributes[v]

            # neighbor group counts (for degree computations)
            nbr_group_cnt[u][gv] = nbr_group_cnt[u].get(gv, 0) + 1
            nbr_group_cnt[v][gu] = nbr_group_cnt[v].get(gu, 0) + 1

            # only diverse edges contribute, credited to BOTH endpoint groups
            if gu != gv:
                num_div_edges_by_group[gu] += 1
                num_div_edges_by_group[gv] += 1

        # one heap per group (peel by current bottleneck group)
        heaps = {g: nx.utils.BinaryHeap() for g in groups}
        for node in work_G.nodes:
            for g in groups:
                d = group_adjacent_diverse_degree(node, g, current_degrees, nbr_group_cnt)
                heaps[g].insert(node, loads[node] + d)

        remaining_nodes = set(work_G.nodes)

        while remaining_nodes:
            num_nodes = len(remaining_nodes)
            if num_nodes == 0:
                break

            # require all groups present
            if any(group_node_counts[g] == 0 for g in groups):
                break

            min_edges = min(num_div_edges_by_group[g] for g in groups)
            current_density = min_edges / num_nodes

            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(remaining_nodes)

            # bottleneck group = argmin_g |E_g^div(S)|
            bottleneck_g = min(groups, key=lambda g: num_div_edges_by_group[g])

            # pop node with smallest weighted degree w.r.t. bottleneck group
            heap = heaps[bottleneck_g]
            node, _ = heap.pop()
            while node not in remaining_nodes:
                node, _ = heap.pop()

            # update load (Greedy++ style) using diverse-degree to bottleneck group
            loads[node] += group_adjacent_diverse_degree(node, bottleneck_g, current_degrees, nbr_group_cnt)

            # update group node counts
            g_node = node_attributes[node]
            group_node_counts[g_node] -= 1

            # update neighbors, degrees, and per-group diverse edge counts
            for neighbor in list(work_G.neighbors(node)):
                if neighbor in remaining_nodes:
                    gu = node_attributes[node]
                    gv = node_attributes[neighbor]

                    # if (node,neighbor) is diverse, it contributed +1 to both endpoint groups
                    if gu != gv:
                        num_div_edges_by_group[gu] -= 1
                        num_div_edges_by_group[gv] -= 1

                    # update neighbor degree
                    current_degrees[neighbor] -= 1

                    # neighbor loses one neighbor from group gu
                    c = nbr_group_cnt[neighbor].get(gu, 0) - 1
                    if c <= 0:
                        nbr_group_cnt[neighbor].pop(gu, None)
                    else:
                        nbr_group_cnt[neighbor][gu] = c

                    # update heaps keys only where they can change:
                    # 1) neighbor's own group (uses current_degrees and same-group count)
                    # 2) gu (because neighbor's count of gu-neighbors changed if gu != neighbor_group)
                    g_neighbor = node_attributes[neighbor]

                    d1 = group_adjacent_diverse_degree(neighbor, g_neighbor, current_degrees, nbr_group_cnt)
                    heaps[g_neighbor].insert(neighbor, loads[neighbor] + d1)

                    if gu != g_neighbor:
                        d2 = group_adjacent_diverse_degree(neighbor, gu, current_degrees, nbr_group_cnt)
                        heaps[gu].insert(neighbor, loads[neighbor] + d2)

            remaining_nodes.remove(node)
            work_G.remove_node(node)

    return best_density, best_subgraph

def maxmin_greedy_plus_plus(G, node_attributes, iterations):
    if G.number_of_edges() == 0:
        return 0.0, set()
    if iterations < 1:
        raise ValueError(
            f"The number of iterations must be an integer >= 1. Provided: {iterations}"
        )

    

    # like your red_G idea: ignore isolated nodes (optional but matches your behavior)
    base_G = G.edge_subgraph(G.edges).copy()
    
    # groups from attributes
    groups = sorted(set(node_attributes[n] for n in base_G.nodes))

    loads = dict.fromkeys(base_G.nodes, 0)  # Load vector for Greedy++.
    best_density = -1.0
    best_subgraph = set()

    def group_incident_degree(node, group_value, current_degrees, neighbor_group_counts):
        """
        Degree of `node` in the subgraph that keeps edges with >=1 endpoint in `group_value`.

        - if node is in group_value: every incident edge counts  -> full current degree
        - else: only edges to neighbors in group_value count     -> #neighbors in that group
        """
        if node_attributes[node] == group_value:
            return current_degrees.get(node, 0)

        return neighbor_group_counts[node].get(group_value, 0)

    for _ in range(iterations):
        # IMPORTANT: work on a fresh copy each iteration (Greedy++ keeps loads, not deletions)
        work_G = base_G.copy()

        # neighbor group-counts (for fast group_incident_degree when node not in g)
        nbr_group_cnt = {u: {} for u in work_G.nodes}
        current_degrees = dict(work_G.degree)

        # per-group edge counts: |E_g(S)| where edge counts for g if at least one endpoint is in g
        num_edges_by_group = dict.fromkeys(groups, 0)

        # group node counts (to ignore groups that are no longer present)
        group_node_counts = dict.fromkeys(groups, 0)
        for u in work_G.nodes:
            group_node_counts[node_attributes[u]] += 1

        # initialize counts
        for u, v in work_G.edges:
            gu = node_attributes[u]
            gv = node_attributes[v]

            # neighbor group counts
            nbr_group_cnt[u][gv] = nbr_group_cnt[u].get(gv, 0) + 1
            nbr_group_cnt[v][gu] = nbr_group_cnt[v].get(gu, 0) + 1

            # edge contributes to endpoint groups (once if same group, twice if different)
            num_edges_by_group[gu] += 1
            if gv != gu:
                num_edges_by_group[gv] += 1

        # one heap per group (so we can peel using the current bottleneck group)
        heaps = {g: nx.utils.BinaryHeap() for g in groups}
        for node in work_G.nodes:
            for g in groups:
                d = group_incident_degree(node, g, current_degrees, nbr_group_cnt)
                heaps[g].insert(node, loads[node] + d)

        remaining_nodes = set(work_G.nodes)

        while remaining_nodes:
            num_nodes = len(remaining_nodes)
            if num_nodes == 0:
                break

            # all groups must be present (paper definition)
            if any(group_node_counts[g] == 0 for g in groups):
                break  # any further peeling yields eFairD(S)=0
            min_edges = min(num_edges_by_group[g] for g in groups)
            current_density = min_edges / num_nodes

            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(remaining_nodes)

            # bottleneck group = argmin_g |E_g(S)|
            bottleneck_g = min(groups, key=lambda g: num_edges_by_group[g])

            # pop node with smallest weighted degree w.r.t. bottleneck group
            heap = heaps[bottleneck_g]
            node, _ = heap.pop()
            while node not in remaining_nodes:
                node, _ = heap.pop()

            # update load (Greedy++ style) using degree to bottleneck group
            loads[node] += group_incident_degree(node, bottleneck_g, current_degrees, nbr_group_cnt)

            # update group node counts
            g_node = node_attributes[node]
            group_node_counts[g_node] -= 1

            # update neighbors, degrees, and per-group edge counts
            for neighbor in list(work_G.neighbors(node)):
                if neighbor in remaining_nodes:
                    gu = node_attributes[node]
                    gv = node_attributes[neighbor]

                    # remove the edge (node, neighbor) from per-group edge counts
                    num_edges_by_group[gu] -= 1
                    if gv != gu:
                        num_edges_by_group[gv] -= 1

                    # update neighbor degree
                    current_degrees[neighbor] -= 1

                    # neighbor loses one neighbor from group gu
                    c = nbr_group_cnt[neighbor].get(gu, 0) - 1
                    if c <= 0:
                        nbr_group_cnt[neighbor].pop(gu, None)
                    else:
                        nbr_group_cnt[neighbor][gu] = c

                    # update heaps only for groups whose key can change:
                    # 1) neighbor's own group (because current_degrees changed if neighbor in that group)
                    # 2) gu (because neighbor's count of gu-neighbors changed if neighbor not in gu)
                    g_neighbor = node_attributes[neighbor]

                    d1 = group_incident_degree(neighbor, g_neighbor, current_degrees, nbr_group_cnt)
                    heaps[g_neighbor].insert(neighbor, loads[neighbor] + d1)

                    if gu != g_neighbor:
                        d2 = group_incident_degree(neighbor, gu, current_degrees, nbr_group_cnt)
                        heaps[gu].insert(neighbor, loads[neighbor] + d2)

            remaining_nodes.remove(node)
            work_G.remove_node(node)

    return best_density, best_subgraph




