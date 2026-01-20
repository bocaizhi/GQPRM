import networkx as nx
import random

def get_connected_subgraph(G, max_edges=4):
    """
    从 G 中随机生成一个边数不超过 max_edges 的连通子图。
    """
    start_node = random.choice(list(G.nodes()))
    visited_nodes = set([start_node])
    visited_edges = set()
    stack = [start_node]

    while stack and len(visited_edges) < max_edges:
        node = stack.pop()
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                visited_edges.add((node, neighbor))
                stack.append(neighbor)
                if len(visited_edges) >= max_edges:
                    break

    # 构建子图
    subgraph = nx.Graph()
    for node in visited_nodes:
        subgraph.add_node(node, **G.nodes[node])
    for u, v in visited_edges:
        if G.has_edge(u, v):
            subgraph.add_edge(u, v, **G.edges[u, v])

    return subgraph

def is_subgraph_of(subgraph, supergraph):
    """快速检查 subgraph 是否是 supergraph 的子图（边和节点是否包含）"""
    return set(subgraph.nodes()).issubset(supergraph.nodes()) and \
           set(subgraph.edges()).issubset(supergraph.edges())

class QueryGenerator:
    """
    A class for generating test queries from a given graph and initializing random prices.
    
    用于从给定图中生成测试查询，并初始化随机价格。
    """

    def __init__(self, graph, query_num=10000, max_query_edges=5):
        """
        Initialize the QueryGenerator with a graph and parameters.
        
        初始化 QueryGenerator，传入图结构和相关参数。
        
        :param graph: The input graph (networkx.Graph)
        :param query_num: Number of test queries to generate (default: 10000)
        :param max_query_edges: Maximum number of edges per query (default: 5)
        """
        self.graph = graph
        self.query_num = query_num
        self.max_query_edges = max_query_edges
        self.queries = []
        self.vertex_prices = []

    def generate_queries(self, subgraphs):
        """
        从子图列表中生成查询图：若子图较小直接使用，否则提取较小的连通子图。
        """
        self.queries = []
        while len(self.queries) < self.query_num:
            selected_graph = random.choice(subgraphs)
            if selected_graph.number_of_nodes() <= 5:
                self.queries.append(selected_graph.copy())
            else:
                subgraph = get_connected_subgraph(selected_graph, max_edges=self.max_query_edges - 1)
                if nx.is_connected(subgraph):
                    # 验证 subgraph 是否真的是 selected_graph 的子图
                    if is_subgraph_of(subgraph, selected_graph):
                        # print("yes")
                        self.queries.append(subgraph)
                    # else:
                    #     print("no")  
    # # 直接选择子图作为查询    
    # def generate_queries(self, subgraphs):
    #     """
    #     直接从给定的子图列表中随机选择查询图。
        
    #     :param subgraphs: A list of candidate graphs (networkx.Graph)
    #     """
    #     self.queries = []
    #     while len(self.queries) < self.query_num:
    #         num = random.randint(0,len(subgraphs)-1)
    #         # print('choose subgraph:',num)
    #         selected_graph = subgraphs[num]
    #         self.queries.append(selected_graph.copy())  # 防止引用冲突

    #
    # 
    # 根据G随机性很大的生成            
    # def generate_queries(self):
    #     """
    #     Generate connected, acyclic subgraphs (trees) from the input graph.
        
    #     从原始图生成连通无环子图（树）。
    #     """
    #     while len(self.queries) < self.query_num:
    #         start_node = random.choice(list(self.graph.nodes))
            
    #         nodes = set([start_node])
    #         edges = set()
    #         stack = [start_node]
            
    #         while stack and len(edges) < self.max_query_edges:
    #             node = stack.pop()
    #             neighbors = list(self.graph.neighbors(node))
    #             random.shuffle(neighbors)
                
    #             for neighbor in neighbors:
    #                 if neighbor not in nodes:
    #                     nodes.add(neighbor)
    #                     edges.add((node, neighbor))
    #                     stack.append(neighbor)
    #                     if len(edges) >= self.max_query_edges:
    #                         break

    #         # 构建一个新图，保留节点的 name 属性，但不保留边属性
    #         subgraph = nx.Graph()
    #         for node in nodes:
    #             name_attr = self.graph.nodes[node].get('name', str(node))
    #             subgraph.add_node(node, name=name_attr)

    #         for u, v in edges:
    #             subgraph.add_edge(u, v)  # 不带边属性

    #         if nx.is_tree(subgraph):
    #             self.queries.append(subgraph)

    def initialize_random_prices(self):
        """
        Generate random prices for vertices in the graph.
        
        为图中的顶点生成随机价格。
        """
        self.vertex_prices = [
            (node, c, random.randint(c + 1, 15))
            for node in self.graph.nodes
            for c in [random.randint(1, 8)]
        ]

    def get_queries(self):
        """
        Get the generated queries.
        
        获取生成的查询。
        """
        return self.queries

    def get_vertex_prices(self):
        """
        Get the initialized vertex prices.
        
        获取初始化的顶点价格。
        """
        return self.vertex_prices

# class QueryGenerator:
#     """
#     A class for generating test queries from a given graph and initializing random prices.
    
#     用于从给定图中生成测试查询，并初始化随机价格。
#     """
    
#     def __init__(self, graph, query_num=10000, max_query_edges=5):
#         """
#         Initialize the QueryGenerator with a graph and parameters.
        
#         初始化 QueryGenerator，传入图结构和相关参数。
        
#         :param graph: The input graph (networkx.Graph)
#         :param query_num: Number of test queries to generate (default: 10000)
#         :param max_query_edges: Maximum number of edges per query (default: 5)
#         """
#         self.graph = graph
#         self.query_num = query_num
#         self.max_query_edges = max_query_edges
#         self.queries = []
#         self.vertex_prices = []
    
#     def add_unique_vertex(self, node, graph):
#         """
#         Ensure a vertex is added to the graph only if it does not already exist.
        
#         确保顶点仅在不存在时添加到图中。
        
#         :param node: Vertex label
#         :param graph: The graph to add the vertex to
#         """
#         if node not in graph:
#             graph.add_node(node)
    
#     def add_unique_vertex(self, node, query_graph):
#         """确保查询图中只添加唯一节点"""
#         if node not in query_graph:
#             query_graph.add_node(node)

#     def generate_queries(self):
#         while len(self.queries) < self.query_num:
#             # 从原始图中随机选择一个起始节点
#             start_node = random.choice(list(self.graph.nodes))
            
#             # 随机生成一个连通无环的子图，最大边数为max_edges
#             nodes = set([start_node])
#             edges = set()
            
#             # 使用广度优先搜索或深度优先搜索生成子图
#             stack = [start_node]
#             while stack and len(edges) < self.max_query_edges:
#                 node = stack.pop()
#                 neighbors = list(self.graph.neighbors(node))
#                 random.shuffle(neighbors)
                
#                 for neighbor in neighbors:
#                     if neighbor not in nodes:
#                         nodes.add(neighbor)
#                         edges.add((node, neighbor))
#                         stack.append(neighbor)
#                         if len(edges) >= self.max_query_edges:
#                             break
            
#             # 构建子图
#             subgraph = self.graph.subgraph(nodes).copy()
#             subgraph.add_edges_from(edges)
            
#             # 确保子图是简单连通无环的
#             if nx.is_weakly_connected(subgraph) and nx.is_tree(subgraph):
#                 self.queries.append(subgraph)
    
#     def initialize_random_prices(self):
#         """
#         Generate random prices for vertices in the graph.
        
#         为图中的顶点生成随机价格。
#         """
#         self.vertex_prices = [
#             (node, c, random.randint(c + 1, 15))
#             for node in self.graph.nodes
#             for c in [random.randint(1, 8)]
#         ]
    
#     def get_queries(self):
#         """
#         Get the generated queries.
        
#         获取生成的查询。
#         """
#         return self.queries
    
#     def get_vertex_prices(self):
#         """
#         Get the initialized vertex prices.
        
#         获取初始化的顶点价格。
#         """
#         return self.vertex_prices