import random
import networkx as nx
from graph_data_loader import GraphDataLoader

def generate_random_subgraphs(graph, num_subgraphs, max_edges=5):
    subgraphs = []
    
    for _ in range(num_subgraphs):
        # 从原始图中随机选择一个起始节点
        start_node = random.choice(list(graph.nodes))
        
        # 随机生成一个连通无环的子图，最大边数为max_edges
        nodes = set([start_node])
        edges = set()
        
        # 使用广度优先搜索或深度优先搜索生成子图
        stack = [start_node]
        while stack and len(edges) < max_edges:
            node = stack.pop()
            neighbors = list(graph.neighbors(node))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in nodes:
                    nodes.add(neighbor)
                    edges.add((node, neighbor))
                    stack.append(neighbor)
                    if len(edges) >= max_edges:
                        break
        
        # 构建子图
        subgraph = graph.subgraph(nodes).copy()
        subgraph.add_edges_from(edges)
        
        # 确保子图是简单连通无环的
        if nx.is_weakly_connected(subgraph) and nx.is_tree(subgraph):
            subgraphs.append(subgraph)
    
    return subgraphs

# 示例图
#G = nx.erdos_renyi_graph(20, 0.3, directed=True)

loader = GraphDataLoader("dbpedia.ttl", "subgraphs.pkl")
loader.load_rdf_graph()
loader.load_subgraphs()
G = loader.get_graph()

# # 给图的每条边加上一个随机标签
# for u, v in G.edges():
#     G[u][v]['label'] = random.choice(['label1', 'label2', 'label3'])

# 生成3个查询子图，最大边数为5
subgraphs = generate_random_subgraphs(G, num_subgraphs=3, max_edges=5)

# 输出子图的节点和边
for idx, subgraph in enumerate(subgraphs):
    print(f"Subgraph {idx+1}:")
    print(f"Nodes: {subgraph.nodes()}")
    print(f"Edges: {subgraph.edges()}")
    print()

    print("Subgraph edges:", subgraph.edges(data=True))

# import networkx as nx
# import random
# import matplotlib.pyplot as plt

# # 创建一个有向图
# G = nx.DiGraph()

# # 添加顶点并为每个顶点指定名称
# nodes = ['A', 'B', 'C', 'D', 'E']
# G.add_nodes_from(nodes)

# # 添加带有 label 属性的边
# edges = [('A', 'B', {'label': 'ab'}), 
#          ('A', 'C', {'label': 'ac'}), 
#          ('B', 'D', {'label': 'bd'}), 
#          ('C', 'D', {'label': 'cd'}), 
#          ('D', 'E', {'label': 'de'}),
#          ('E', 'A', {'label': 'ea'})]
# G.add_edges_from(edges)

# # 指定要选择的边数
# num_edges_to_select = 3

# # 获取图中的所有边
# edges_with_labels = list(G.edges(data=True))

# # 随机选择指定数量的边
# selected_edges = random.sample(edges_with_labels, num_edges_to_select)

# # 创建包含这些选定边的子图
# subgraph = G.edge_subgraph([(u, v) for u, v, _ in selected_edges])

# # 打印子图信息
# print("Subgraph nodes:", subgraph.nodes())
# print("Subgraph edges:", subgraph.edges(data=True))

# # 可视化子图
# pos = nx.spring_layout(subgraph)
# nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
# edge_labels = nx.get_edge_attributes(subgraph, 'label')
# nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)
# plt.show()
