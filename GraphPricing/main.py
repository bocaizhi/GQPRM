import random
from graph_data_loader import GraphDataLoader
from query_generator import QueryGenerator
import graph_pricing_transaction
from graph_pricing_transaction import TransactionManager
from graph_pricing_transaction import FullTransactionLogger
from  graph_pricing_transaction import TransactionEvaluator
import networkx as nx
import time
import hashlib
import pickle

def load_graphs_from_pickle(filename):
    """
    从 pkl 文件加载图列表。
    
    :param filename: 存储的 .pkl 文件名
    :return: 读取后的图列表
    """
    with open(filename, 'rb') as f:
        graph_list = pickle.load(f)
    return graph_list

def initialize_vertex_edge_price_list(G):
    """
    初始化基于“顶点 + 边”结构的价格集合。
    
    :param G: NetworkX 图
    :return: (vertex_price_list, edge_price_list)
    """
    vertex_price_list = {}
    edge_price_list = {}
    seen_vertex_names = set()
    seen_edge_keys = set()

    # 初始化顶点价格
    for node in G.nodes:
        name = G.nodes[node].get('name', node)  # 获取 'name' 属性，若无则使用节点 ID

        if name in seen_vertex_names:
            continue  # 跳过重复顶点名
        seen_vertex_names.add(name)

        cost = random.randint(1, 4)
        price = random.randint(cost + 1, cost + 6)

        vertex_price_list[name] = {'cost': cost, 'price': price}

    # 初始化边价格
    for u, v, attr in G.edges(data=True):
        name_u = G.nodes[u].get('name', u)
        name_v = G.nodes[v].get('name', v)

        # 无向图：确保边唯一标识（字典顺序）
        edge_key = tuple(sorted((name_u, name_v)))
        if edge_key in seen_edge_keys:
            continue  # 跳过重复边

        seen_edge_keys.add(edge_key)

        edge_weight = attr.get('weight', random.randint(1, 3))
        edge_price = random.randint(edge_weight + 1, edge_weight + 6)

        edge_price_list[edge_key] = {
            'weight': edge_weight,
            'price': edge_price
        }

    return vertex_price_list, edge_price_list

# def initialize_vertex_price_list(G):
#     """
#     以顶点 'name' 属性为单位初始化 vertex_price_list，避免重复顶点，确保价格不为 inf。

#     :param G: NetworkX 图guo
#     :return: 顶点价格字典 {name: {'cost': c, 'price': p}}
#     """
#     vertex_price_list = {}
#     seen_names = set()

#     for node in G.nodes:
#         name = G.nodes[node].get('name', node)  # 获取 'name' 属性，若无则用节点 ID
        
#         if name in seen_names:
#             continue  # 跳过重复顶点名

#         seen_names.add(name)
        
#         cost = random.randint(1, 8)
#         price = random.randint(cost + 1, 15)
#         if isinstance(price, float) and not price.is_integer() == True:
#                 print(1)
#                 raise ValueError(f"顶点 '{name}' 的价格为小数")

#         vertex_price_list[name] = {'cost': cost, 'price': price}

#     return vertex_price_list

# def graph_signature(G):
#     """
#     用于为无向图生成唯一的同构签名（无属性版本）。
#     使用 Graph canonical labeling 来做图结构签名。
#     """
#     gm = nx.convert_node_labels_to_integers(G, label_attribute="old_label")
#     return nx.weisfeiler_lehman_graph_hash(gm)
def custom_weisfeiler_lehman_hash(G, node_attr="label", edge_attr=None, digest_size=16):
    """
    自定义 WL 哈希函数，支持 UTF-8 编码，避免 UnicodeEncodeError。
    """
    # 初始化节点标签
    node_labels = {n: str(d.get(node_attr, n)) for n, d in G.nodes(data=True)}

    for _ in range(G.number_of_nodes()):  # 迭代次数，可设固定值
        new_labels = {}
        for node in G.nodes():
            neighbors = []
            for nbr in sorted(G.neighbors(node)):
                if edge_attr and G.has_edge(node, nbr):
                    edge_label = G[node][nbr].get(edge_attr, "")
                    neighbors.append(f"{node_labels[nbr]}-{edge_label}")
                else:
                    neighbors.append(node_labels[nbr])
            # 拼接自身标签和邻居标签
            label_str = node_labels[node] + "".join(sorted(neighbors))
            hash_val = hashlib.blake2b(label_str.encode("utf-8"), digest_size=digest_size).hexdigest()
            new_labels[node] = hash_val
        node_labels = new_labels

    # 最终图标签
    return hashlib.blake2b("".join(sorted(node_labels.values())).encode("utf-8"), digest_size=digest_size).hexdigest()

def graph_signature(G):
    """
    为图生成结构 + 属性的唯一签名（兼容非ASCII）。
    节点使用 'name'，边使用 'label'。
    """
    G_copy = nx.Graph()

    for n, attr in G.nodes(data=True):
        name = attr.get('name', str(n))
        G_copy.add_node(n, label=name)

    for u, v, attr in G.edges(data=True):
        label = attr.get('label', '')
        G_copy.add_edge(u, v, label=label)

    return custom_weisfeiler_lehman_hash(G_copy, node_attr='label', edge_attr='label')

def initialize_buyer_price_dict(queries):
    """
    使用图结构签名初始化买家的预期价格字典（加速后续查找）。

    :param queries: 查询图列表 (networkx.Graph)
    :return: 字典 {graph_signature(Q): expected_price}
    """
    buyer_price_dict = {}
    for Q in queries:
        expected_price = Q.number_of_nodes() * 4 + random.uniform(0, 15)
        sig = graph_signature(Q)
        buyer_price_dict[sig] = expected_price
    return buyer_price_dict

# def initialize_buyer_price_list(queries):
#     """
#     随机初始化买家心理预期价格列表，以查询图为单位。

#     :param queries: 需要定价的查询图列表 (NetworkX 图列表)
#     :return: 预期价格列表 (列表的每个元素是 [查询图, 预期价格])
#     """
#     buyer_price_list = []
#     for Q in queries:
#         expected_price = Q.number_of_nodes() * 5 + random.uniform(0, 15)  # 根据查询图大小生成预期价格
#         buyer_price_list.append([Q, expected_price])
    
#     return buyer_price_list

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

# def initialize_buyer_price_list(queries):
#     """
#     随机初始化买家心理预期价格列表，以查询图为单位。

#     :param queries: 需要定价的查询图列表
#     :return: 预期价格列表
#     """
#     buyer_price_list = []
#     for Q in queries:
#         expected_price = Q.vcount() * 5 + random.uniform(0, 10)  # 根据查询图大小生成预期价格
#         buyer_price_list.append([Q, expected_price])
#     return buyer_price_list

# 按装订区域中的绿色按钮以运行脚本。
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

    

    # print(type(subgraphs[0]))
    generator = QueryGenerator(G, query_num=150000, max_query_edges=5)
    # 加载之前保存的子图列表
    loaded_graphs = load_graphs_from_pickle('queries_subgraphs_30000.pkl')
    generator.generate_queries(loaded_graphs)  # subgraph_list 为你预先构造好的图列表
    # print(list(G.nodes))  # 获取所有节点的名称（即节点的 key）
    # generator = QueryGenerator(G,query_num=100)
    # generator.generate_queries()
    # generator.initialize_random_prices()
    queries = generator.get_queries()
    # for q in queries:
    #     print(q.number_of_nodes(),q.number_of_edges())
    # queries = []
    # queries.append(subgraphs[1])
    # prices = generator.get_vertex_prices()
    print(len(queries))
    # print(type(queries[0]))

    # 初始化顶点价格列表
    vertex_price_list, edge_price_list = initialize_vertex_edge_price_list(G)
    # print(len(G.nodes))
    # print(vertex_price_list)
    

    # 初始化买家心理预期价格列表
    buyer_price_list = initialize_buyer_price_dict(queries)
    # print(buyer_price_list)
    print('len of expected price list:',len(buyer_price_list))
    # buyer_price_list = []
    # print(len(buyer_price_list))
    start_time = time.time()
    # 运行交易系统 (Run the transaction system)
    transaction_manager = TransactionManager()
    full_transaction = FullTransactionLogger()
    evaluator = TransactionEvaluator(buyer_price_list)
    graph_pricing_transaction.process_transactions(queries, subgraphs, vertex_price_list, edge_price_list, transaction_manager, full_transaction, 
                                                   buyer_price_list, evaluator, start_time, xpricing=8, centrality_file= 'my_graph_centrality.pkl')
    # print("交易成功率:", evaluator.evaluate_transactions())