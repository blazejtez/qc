#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
from networkx.drawing.nx_pylab import draw
from networkx.algorithms.moral import moral_graph
import itertools
'''Example of creating a graph and computing a junction tree for the graph. Uses networkx library.'''
'''The factorized wave function for this example will be: psi(A)psi(B|A)psi(C|A)psi(D|B)psi(E|C)psi(F|B,E).
We see that the Bayesian graph corresponding to the above factorization has cycle.'''
G = nx.DiGraph()

G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F'])
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('B', 'F'),
                  ('E', 'F')])
'''Compute the moral graph from the Bayesian graph'''
H = moral_graph(G)


def get_edges_set(graph):
    return set(H.edges())


def minfillcost(edges, neighbor_nodes):
    elimination_clique = itertools.combinations(neighbor_nodes, 2)
    cost = sum([edge not in edges for edge in elimination_clique])
    return cost


cost = minfillcost(get_edges_set(H), 'F')
'''Find repeatedly vertex with minimal elimination cost, create elimination ordering'''
elimination_ordering = []
H_copy = H.copy()
while list(H_copy):
    edges = get_edges_set(H)
    costs = [(v, minfillcost(edges, H_copy.neighbors(v)))
             for v in list(H_copy)]
    costs.sort(key=lambda e: e[1])
    node = costs[0][0]
    elimination_ordering.append(node)
    H_copy.remove_node(node)
'''Find clusters'''
clusters = []
for node in elimination_ordering:
    neighbor_nodes = list(H.neighbors(node))
    neighbor_nodes.append(node)
    elimination_clique = itertools.combinations(neighbor_nodes, 2)
    elimination_clique_lst = list(elimination_clique)
    clusters.append(elimination_clique_lst)
    edges = get_edges_set(H)
    for edge in elimination_clique_lst:
        if edge not in edges and (edge[1], edge[0]) not in edges:
            H.add_edge(*edge)
    if len(edges) == len(elimination_clique_lst):
        break
    H.remove_node(node)
print(f"Clusters as lists of edges: {clusters}")
'''Turn clusters into sets'''
cluster_sets = []
for cluster in clusters:
    cluster_sets.append(set(itertools.chain(*cluster)))
'''Turn clusters into tuples'''
cluster_tuples = []
for cluster in cluster_sets:
    cluster_tuples.append(tuple(cluster))
print(f"Clusters as lists of nodes: {cluster_tuples}")
'''Run the maximum spanning tree algorithm (note the minuses in weights)'''
complete_graph = nx.Graph()
edges = list(itertools.combinations(cluster_tuples, 2))
weights = list(map(lambda x: len(set(x[0]).intersection(set(x[1]))), edges))
for edge, weight in zip(edges, weights):
    complete_graph.add_edge(*edge, weight=-weight)
print(f"Edges between clusters (pairs of clusters): {nx.minimum_spanning_tree(complete_graph).edges()}")
