import networkx as nx
import rdflib
import pickle

class GraphDataLoader:
    """
    A class for loading RDF graph data, constructing a NetworkX graph,
    and loading pre-split subgraphs.

    用于加载 RDF 图数据、构建 NetworkX 图结构，并加载已拆分的子图。
    """

    def __init__(self, rdf_file, subgraph_file):
        """
        Initialize the GraphDataLoader with file paths.

        初始化 GraphDataLoader，传入 RDF 文件和子图数据文件路径。

        :param rdf_file: Path to the RDF file (TTL format)
        :param subgraph_file: Path to the pickled subgraph dataset
        """
        self.rdf_file = rdf_file
        self.subgraph_file = subgraph_file
        self.graph = nx.DiGraph()
        self.subgraphs = []

    def load_rdf_graph(self):
        """
        Load RDF data and construct a NetworkX graph.

        加载 RDF 数据并构建 NetworkX 图。
        """
        rdf_graph = rdflib.Graph()
        rdf_graph.parse(self.rdf_file, format="ttl")

        for subj, pred, obj in rdf_graph:
            s, p, o = subj.split('/')[-1], pred.split('/')[-1], obj.split('/')[-1]
            
            # Skip self-loops
            if s == o:
                continue

            # Add edge with label attribute
            self.graph.add_edge(s, o, label=p)

    def load_subgraphs(self):
        """
        Load precomputed subgraphs from a pickle file.

        加载预处理的子图数据。
        """
        with open(self.subgraph_file, 'rb') as f:
            self.subgraphs = pickle.load(f)

        # # 确保所有子图是 DiGraph
        # self.subgraphs = []
        # for subG in subgraphs_raw:
        #     if isinstance(subG, nx.Graph):  # 如果是无向图，转换为有向图
        #         subG = nx.DiGraph(subG)
        #     elif not isinstance(subG, nx.DiGraph):
        #         raise TypeError(f"Invalid subgraph type: {type(subG)}. Expected nx.DiGraph.")
        #     self.subgraphs.append(subG)

    def get_graph(self):
        """
        Get the constructed NetworkX graph.

        获取构建的 NetworkX 图。
        """
        return self.graph

    def get_subgraphs(self):
        """
        Get the loaded subgraph dataset.

        获取加载的子图数据集。
        """
        return self.subgraphs

# import igraph as ig
# import rdflib
# import pickle

# class GraphDataLoader:
#     """
#     A class for loading RDF graph data, constructing an igraph graph,
#     and loading pre-split subgraphs.

#     用于加载 RDF 图数据、构建 igraph 图结构，并加载已拆分的子图。
#     """

#     def __init__(self, rdf_file, subgraph_file):
#         """
#         Initialize the GraphDataLoader with file paths.

#         初始化 GraphDataLoader，传入 RDF 文件和子图数据文件路径。

#         :param rdf_file: Path to the RDF file (TTL format)
#         :param subgraph_file: Path to the pickled subgraph dataset
#         """
#         self.rdf_file = rdf_file
#         self.subgraph_file = subgraph_file
#         self.graph = ig.Graph(directed=True)
#         self.subgraphs = []

#     def load_rdf_graph(self):
#         """
#         Load RDF data and construct an igraph graph.

#         加载 RDF 数据并构建 igraph 图。
#         """
#         rdf_graph = rdflib.Graph()
#         rdf_graph.parse(self.rdf_file, format="ttl")

#         vertex_set = set()  # 使用集合提高唯一性检查速度
#         edges = []  # 用于批量添加边

#         for subj, pred, obj in rdf_graph:
#             s, p, o = subj.split('/')[-1], pred.split('/')[-1], obj.split('/')[-1]

#             # Skip self-loops
#             if s == o:
#                 continue

#             # Add vertices only if they are new
#             if s not in vertex_set:
#                 self.graph.add_vertex(s)
#                 vertex_set.add(s)
#             if o not in vertex_set:
#                 self.graph.add_vertex(o)
#                 vertex_set.add(o)

#             # Store edges for batch processing
#             edges.append((s, o, p))

#         # Add all edges at once for efficiency
#         self.graph.add_edges([(s, o) for s, o, _ in edges])
#         self.graph.es['label'] = [p for _, _, p in edges]

#     def load_subgraphs(self):
#         """
#         Load precomputed subgraphs from a pickle file.

#         加载预处理的子图数据。
#         """
#         with open(self.subgraph_file, 'rb') as f:
#             self.subgraphs = pickle.load(f)

#     def get_graph(self):
#         """
#         Get the constructed igraph graph.

#         获取构建的 igraph 图。
#         """
#         return self.graph

#     def get_subgraphs(self):
#         """
#         Get the loaded subgraph dataset.

#         获取加载的子图数据集。
#         """
#         return self.subgraphs