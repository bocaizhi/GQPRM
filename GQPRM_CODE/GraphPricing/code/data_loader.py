import networkx as nx
import rdflib
import pickle
import os
from config import Config

class GraphDataLoader:
    def __init__(self):
        self.rdf_file = Config.Paths.RDF_FILE
        self.subgraph_file = Config.Paths.SUBGRAPH_FILE
        self.graph = nx.DiGraph()
        self.subgraphs = []

    def load_rdf_graph(self):
        print(f"Loading RDF Graph from {self.rdf_file}...")
        if not os.path.exists(self.rdf_file):
            print(f"Error: {self.rdf_file} not found.")
            return

        rdf_graph = rdflib.Graph()
        rdf_graph.parse(self.rdf_file, format="ttl")

        for subj, pred, obj in rdf_graph:
            s = subj.split('/')[-1]
            p = pred.split('/')[-1]
            o = obj.split('/')[-1]
            if s == o: continue
            self.graph.add_edge(s, o, label=p)
        print(f"Full Graph loaded. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")

    def load_subgraphs(self):
        print(f"Loading raw subgraphs from {self.subgraph_file}...")
        if not os.path.exists(self.subgraph_file):
            print("Warning: Subgraph file not found.")
            return

        with open(self.subgraph_file, 'rb') as f:
            self.subgraphs = pickle.load(f)
        print(f"Loaded {len(self.subgraphs)} raw subgraphs (igraph format).")

    def get_graph(self):
        return self.graph

    def get_subgraphs(self):
        return self.subgraphs
