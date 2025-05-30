import random
import pickle
import networkx as nx
from graph_data_loader import GraphDataLoader

# 从 subgraphs 中随机抽取 30000 个子图
def sample_subgraphs(subgraphs, sample_size=30000):
    return random.sample(subgraphs, sample_size)

# 保存为 pkl 文件
def save_to_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def convert_igraph_to_networkx(ig_graph):
    """
    将 igraph.Graph 转换为 networkx.DiGraph（有向图）

    :param ig_graph: igraph.Graph（可能是有向或无向）
    :return: networkx.DiGraph
    """
    nx_graph = nx.DiGraph()  # 确保是有向图

    # 添加节点及其属性
    for v in ig_graph.vs:
        nx_graph.add_node(v.index, **v.attributes())

    # 添加边及其属性
    for e in ig_graph.es:
        nx_graph.add_edge(e.source, e.target, **e.attributes())

    return nx_graph


# 示例调用方法
if __name__ == '__main__':
    loader = GraphDataLoader("dbpedia.ttl", "subgraphs.pkl")
    loader.load_rdf_graph()
    loader.load_subgraphs()
    G = loader.get_graph()
    subgraphs_ig = loader.get_subgraphs()
    subgraphs = []
    for subG in subgraphs_ig:
        subg = convert_igraph_to_networkx(subG)
        subgraphs.append(subg)
    print(len(subgraphs))
    # 假设 subgraphs 已定义
    sampled_subgraphs = sample_subgraphs(subgraphs, sample_size=30000)
    save_to_pkl(sampled_subgraphs, 'queries_subgraphs_30000.pkl')
    print("保存成功，文件名为：sampled_subgraphs_30000.pkl")