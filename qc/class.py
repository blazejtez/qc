# -*- coding: utf-8 -*-


import networkx as nx
from networkx.drawing.nx_pylab import draw
from networkx.algorithms.moral import moral_graph
import itertools
from typing import List

class DoTheJob:
    '''constructor loads nodes and edges and stores moral graph in class'''
    def __init__(self, nodes: List, edges: List):
        G = nx.DiGraph()     #nie zrobilem directed graph atrybutem klasy - ok?
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        '''Compute the moral graph from the Bayesian graph'''
        self.H = moral_graph(G)
    
    '''Helper functions for elimination ordering'''
    def _get_edges_set(self) -> set:
        return set(self.H.edges())

    def _minfillcost(self, edges, neighbor_nodes):
        elimination_clique = itertools.combinations(neighbor_nodes, 2)
        cost = sum([edge not in edges for edge in elimination_clique])
        return cost

    
    def _find_node(self, H_copy: nx.Graph):
        
    	edges = self._get_edges_set()
    	costs = [(v, self._minfillcost(edges, H_copy.neighbors(v)))
    			 for v in list(H_copy)]
    	costs.sort(key=lambda e: e[1])
    	node = costs[0][0]
    	return node
    
# cost = minfillcost(get_edges_set(H), 'F') # to bylo tylko do przykladu, prawda?

    '''Find repeatedly vertex with minimal elimination cost, create elimination ordering'''
    def create_elimination_ordering(self):
        elimination_ordering = []
        H_copy = self.H.copy()
        while list(H_copy):
            node = self._find_node(H_copy)
            elimination_ordering.append(node)
            H_copy.remove_node(node)
        return elimination_ordering
    
    
    '''Find clusters in a moral graph using elimination ordering'''
    def find_clusters(self, elimination_ordering):
        clusters = []
        for node in elimination_ordering:
            neighbor_nodes = list(self.H.neighbors(node))
            neighbor_nodes.append(node)
            elimination_clique = itertools.combinations(neighbor_nodes, 2)
            elimination_clique_lst = list(elimination_clique)
            clusters.append(elimination_clique_lst)
            edges = self._get_edges_set()
            for edge in elimination_clique_lst:
                if edge not in edges and (edge[1], edge[0]) not in edges:
                    self.H.add_edge(*edge)
            if len(edges) == len(elimination_clique_lst):
                break
            self.H.remove_node(node)
        return clusters
    
    '''Helper functions to remove duplicates from clusters''' # czy dobrze zrozumialem intencje?
    def _turn_clusters_into_sets(self, clusters: List[tuple]) -> List[set]: 
    	cluster_sets = []
    	for cluster in clusters:
    		cluster_sets.append(set(itertools.chain(*cluster)))
    	return cluster_sets
    
    
    def _turn_clusters_into_tuples(self, cluster_sets: List[set]):
    	cluster_tuples = []
    	for cluster in cluster_sets:
    		cluster_tuples.append(tuple(cluster))
    	return cluster_tuples
    
    '''Removes duplicates from clusters, returns cluster tuples'''
    def remove_clusters_duplicates(self, clusters):
    	cluster_sets = self._turn_clusters_into_sets(clusters)
    	cluster_tuples = self._turn_clusters_into_tuples(cluster_sets)
    	return cluster_tuples
    
    
    '''using clusters tuples without duplicates, calculates complete graph'''
    def run_max_span_tree_algorithm(self, cluster_tuples):
    	complete_graph = nx.Graph()
    	edges = list(itertools.combinations(cluster_tuples, 2))
    	weights = list(map(lambda x: len(set(x[0]).intersection(set(x[1]))), edges))
    	for edge, weight in zip(edges, weights):
    		complete_graph.add_edge(*edge, weight=-weight)
    	return complete_graph
    
    '''does the code from example'''
    def DoEverything(self):
        elimination_ordering = self.create_elimination_ordering()
        
        clusters = self.find_clusters(elimination_ordering)
        print(f"Clusters as lists of edges: {clusters}")
    
        cluster_tuples = self.remove_clusters_duplicates(clusters)
        print(f"Clusters as lists of nodes: {cluster_tuples}")
        
        complete_graph = self.run_max_span_tree_algorithm(cluster_tuples)
        print(f"Edges between clusters (pairs of clusters): {nx.minimum_spanning_tree(complete_graph).edges()}")
    
if __name__ == "__main__":
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('B', 'F'),
                  ('E', 'F')]
    graph = DoTheJob(nodes, edges)
    elimination_ordering = graph.create_elimination_ordering()

    clusters = graph.find_clusters(elimination_ordering)
    print(f"Clusters as lists of edges: {clusters}")
    
    cluster_tuples = graph.remove_clusters_duplicates(clusters)
    print(f"Clusters as lists of nodes: {cluster_tuples}")
        
    complete_graph = graph.run_max_span_tree_algorithm(cluster_tuples)
    print(f"Edges between clusters (pairs of clusters): {nx.minimum_spanning_tree(complete_graph).edges()}")
    