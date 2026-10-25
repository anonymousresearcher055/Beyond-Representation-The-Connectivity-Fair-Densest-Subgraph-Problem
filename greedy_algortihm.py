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
    if G.number_of_edges() == 0:
        return 0.0, set(G.nodes)

    if iterations < 1:
        raise ValueError(
            f"The number of iterations must be an integer >= 1. Provided: {iterations}"
        )

    base_G = G.edge_subgraph(G.edges).copy()
    initial_nodes = tuple(base_G.nodes)

    groups_with_diverse_edges = set()

    for u, v in base_G.edges:
        if node_attributes[u] != node_attributes[v]:
            groups_with_diverse_edges.add(node_attributes[u])
            groups_with_diverse_edges.add(node_attributes[v])

    groups = sorted(groups_with_diverse_edges)

    if not groups:
        return 0.0, set(G.nodes)

    loads = dict.fromkeys(base_G.nodes, 0)
    best_density = -1.0
    best_subgraph = set()

    def group_adjacent_diverse_degree(
        node, group_value, current_degrees, nbr_group_cnt
    ):
        if node_attributes[node] == group_value:
            return (
                current_degrees[node]
                - nbr_group_cnt[node].get(group_value, 0)
            )

        return nbr_group_cnt[node].get(group_value, 0)

    for _ in range(iterations):
        work_G = base_G.copy()

        current_degrees = dict(work_G.degree)
        nbr_group_cnt = {node: {} for node in work_G.nodes}

        num_div_edges_by_group = dict.fromkeys(groups, 0)
        group_node_counts = dict.fromkeys(groups, 0)

        for node in work_G.nodes:
            g = node_attributes[node]
            if g in group_node_counts:
                group_node_counts[g] += 1

        for u, v in work_G.edges:
            gu = node_attributes[u]
            gv = node_attributes[v]

            nbr_group_cnt[u][gv] = (
                nbr_group_cnt[u].get(gv, 0) + 1
            )
            nbr_group_cnt[v][gu] = (
                nbr_group_cnt[v].get(gu, 0) + 1
            )

            if gu != gv:
                if gu in num_div_edges_by_group:
                    num_div_edges_by_group[gu] += 1

                if gv in num_div_edges_by_group:
                    num_div_edges_by_group[gv] += 1

        heaps = {
            g: nx.utils.BinaryHeap()
            for g in groups
        }

        for node in work_G.nodes:
            # Total cross-group degree in the current subgraph.
            diverse_degree = (
                current_degrees[node]
                - nbr_group_cnt[node].get(node_attributes[node], 0)
            )

            for g in groups:
                d = group_adjacent_diverse_degree(
                    node,
                    g,
                    current_degrees,
                    nbr_group_cnt
                )

                heaps[g].insert(
                    node,
                    (
                        loads[node] + d,
                        diverse_degree,
                        current_degrees[node]
                    )
                )

        remaining_nodes = set(work_G.nodes)

        removal_order = []
        best_step_this_iteration = None

        while remaining_nodes:
            num_nodes = len(remaining_nodes)

            if any(group_node_counts[g] == 0 for g in groups):
                break

            # Primary optimization objective: eFairDivD.
            min_edges = min(
                num_div_edges_by_group[g]
                for g in groups
            )

            current_density = min_edges / num_nodes

            if current_density > best_density:
                best_density = current_density
                best_step_this_iteration = len(removal_order)

            # Select the group with minimum diverse edge count.
            bottleneck_g = min(
                groups,
                key=lambda g: num_div_edges_by_group[g]
            )

            heap = heaps[bottleneck_g]

            node, _ = heap.pop()

            while node not in remaining_nodes:
                node, _ = heap.pop()

            loads[node] += group_adjacent_diverse_degree(
                node,
                bottleneck_g,
                current_degrees,
                nbr_group_cnt
            )

            g_node = node_attributes[node]

            if g_node in group_node_counts:
                group_node_counts[g_node] -= 1

            for neighbor in list(work_G.neighbors(node)):
                if neighbor not in remaining_nodes:
                    continue

                gu = node_attributes[node]
                gv = node_attributes[neighbor]

                if gu != gv:
                    if gu in num_div_edges_by_group:
                        num_div_edges_by_group[gu] -= 1

                    if gv in num_div_edges_by_group:
                        num_div_edges_by_group[gv] -= 1

                current_degrees[neighbor] -= 1

                c = nbr_group_cnt[neighbor].get(gu, 0) - 1

                if c <= 0:
                    nbr_group_cnt[neighbor].pop(gu, None)
                else:
                    nbr_group_cnt[neighbor][gu] = c

                # Update the tie-breakers dynamically.
                diverse_degree = (
                    current_degrees[neighbor]
                    - nbr_group_cnt[neighbor].get(
                        node_attributes[neighbor], 0
                    )
                )

                for g in groups:
                    d = group_adjacent_diverse_degree(
                        neighbor,
                        g,
                        current_degrees,
                        nbr_group_cnt
                    )

                    heaps[g].insert(
                        neighbor,
                        (
                            loads[neighbor] + d,
                            diverse_degree,
                            current_degrees[neighbor]
                        )
                    )

            remaining_nodes.remove(node)
            removal_order.append(node)
            work_G.remove_node(node)

        if best_step_this_iteration is not None:
            best_subgraph = set(initial_nodes)

            for node in removal_order[:best_step_this_iteration]:
                best_subgraph.remove(node)

    return best_density, best_subgraph

def maxmin_greedy_plus_plus(G, node_attributes, iterations):


    if G.number_of_edges() == 0:
        return 0.0, set(G.nodes)

    if iterations < 1:
        raise ValueError(
            f"The number of iterations must be an integer >= 1. "
            f"Provided: {iterations}"
        )

    # Preserve original behavior.
    base_G = (
        G.edge_subgraph(G.edges).copy()
    )

    initial_nodes = tuple(
        base_G.nodes
    )

    groups = sorted(
        set(
            node_attributes[n]
            for n in base_G.nodes
        )
    )

    loads = dict.fromkeys(
        base_G.nodes,
        0,
    )

    best_density = -1.0
    best_subgraph = set()

    def group_incident_degree(
        node,
        group_value,
        current_degrees,
        neighbor_group_counts,
    ):
        if (
            node_attributes[node]
            == group_value
        ):
            return (
                current_degrees.get(
                    node,
                    0,
                )
            )

        return (
            neighbor_group_counts[
                node
            ].get(
                group_value,
                0,
            )
        )

    for _ in range(iterations):
        # Keep this copy for now so this test changes
        # only incumbent storage.
        work_G = base_G.copy()

        nbr_group_cnt = {
            u: {}
            for u in work_G.nodes
        }

        current_degrees = dict(
            work_G.degree
        )

        num_edges_by_group = (
            dict.fromkeys(
                groups,
                0,
            )
        )

        group_node_counts = (
            dict.fromkeys(
                groups,
                0,
            )
        )

        for u in work_G.nodes:
            group_node_counts[
                node_attributes[u]
            ] += 1

        for u, v in work_G.edges:
            gu = node_attributes[u]
            gv = node_attributes[v]

            nbr_group_cnt[u][gv] = (
                nbr_group_cnt[u].get(
                    gv,
                    0,
                )
                + 1
            )

            nbr_group_cnt[v][gu] = (
                nbr_group_cnt[v].get(
                    gu,
                    0,
                )
                + 1
            )

            num_edges_by_group[
                gu
            ] += 1

            if gv != gu:
                num_edges_by_group[
                    gv
                ] += 1

        heaps = {
            g: nx.utils.BinaryHeap()
            for g in groups
        }

        for node in work_G.nodes:
            for g in groups:
                d = (
                    group_incident_degree(
                        node,
                        g,
                        current_degrees,
                        nbr_group_cnt,
                    )
                )

                heaps[g].insert(
                    node,
                    (loads[node] + d, current_degrees[node]),
                )

        remaining_nodes = set(
            work_G.nodes
        )

        removal_order = []
        best_step_this_iteration = None

        while remaining_nodes:
            num_nodes = len(
                remaining_nodes
            )

            if num_nodes == 0:
                break

            # Same original stopping condition.
            if any(
                group_node_counts[g] == 0
                for g in groups
            ):
                break

            min_edges = min(
                num_edges_by_group[g]
                for g in groups
            )

            current_density = (
                min_edges / num_nodes
            )

            if current_density > best_density:
                best_density = (
                    current_density
                )

                best_step_this_iteration = (
                    len(removal_order)
                )

            bottleneck_g = min(
                groups,
                key=lambda g:
                num_edges_by_group[g],
            )

            heap = heaps[
                bottleneck_g
            ]

            node, _ = heap.pop()

            while (
                node
                not in remaining_nodes
            ):
                node, _ = heap.pop()

            loads[node] += (
                group_incident_degree(
                    node,
                    bottleneck_g,
                    current_degrees,
                    nbr_group_cnt,
                )
            )

            g_node = (
                node_attributes[node]
            )

            group_node_counts[
                g_node
            ] -= 1

            for neighbor in list(
                work_G.neighbors(node)
            ):
                if (
                    neighbor
                    in remaining_nodes
                ):
                    gu = (
                        node_attributes[
                            node
                        ]
                    )

                    gv = (
                        node_attributes[
                            neighbor
                        ]
                    )

                    num_edges_by_group[
                        gu
                    ] -= 1

                    if gv != gu:
                        num_edges_by_group[
                            gv
                        ] -= 1

                    current_degrees[
                        neighbor
                    ] -= 1

                    c = (
                        nbr_group_cnt[
                            neighbor
                        ].get(
                            gu,
                            0,
                        )
                        - 1
                    )

                    if c <= 0:
                        nbr_group_cnt[
                            neighbor
                        ].pop(
                            gu,
                            None,
                        )
                    else:
                        nbr_group_cnt[
                            neighbor
                        ][gu] = c

                    # Update every group heap: current degree is the tie-breaker.
                    for g in groups:
                        d = (
                            group_incident_degree(
                                neighbor,
                                g,
                                current_degrees,
                                nbr_group_cnt,
                            )
                        )
                        heaps[g].insert(
                            neighbor,
                            (
                                loads[neighbor] + d,
                                current_degrees[neighbor],
                            ),
                        )

            remaining_nodes.remove(
                node
            )

            removal_order.append(
                node
            )

            work_G.remove_node(
                node
            )

        if (
            best_step_this_iteration
            is not None
        ):
            best_subgraph = set(
                initial_nodes
            )

            for position in range(
                best_step_this_iteration
            ):
                best_subgraph.remove(
                    removal_order[
                        position
                    ]
                )

    return (
        best_density,
        best_subgraph,
    )




