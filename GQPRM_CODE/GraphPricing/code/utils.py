import networkx as nx
import hashlib
import random
import pickle
from config import Config

def custom_weisfeiler_lehman_hash(G, node_attr="label", edge_attr=None, digest_size=16):
    """自定义 WL 哈希函数，用于生成图的唯一指纹"""
    node_labels = {n: str(d.get(node_attr, n)) for n, d in G.nodes(data=True)}
    for _ in range(G.number_of_nodes()):
        new_labels = {}
        for node in G.nodes():
            neighbors = []
            for nbr in sorted(G.neighbors(node)):
                if edge_attr and G.has_edge(node, nbr):
                    edge_label = G[node][nbr].get(edge_attr, "")
                    neighbors.append(f"{node_labels[nbr]}-{edge_label}")
                else:
                    neighbors.append(node_labels[nbr])
            label_str = node_labels[node] + "".join(sorted(neighbors))
            hash_val = hashlib.blake2b(label_str.encode("utf-8"), digest_size=digest_size).hexdigest()
            new_labels[node] = hash_val
        node_labels = new_labels
    return hashlib.blake2b("".join(sorted(node_labels.values())).encode("utf-8"), digest_size=digest_size).hexdigest()

def graph_signature(G):
    """生成包含节点名称和边标签的唯一签名"""
    G_copy = nx.Graph()
    for n, attr in G.nodes(data=True):
        name = attr.get('name', str(n))
        G_copy.add_node(n, label=name)
    for u, v, attr in G.edges(data=True):
        label = attr.get('label', '')
        G_copy.add_edge(u, v, label=label)
    return custom_weisfeiler_lehman_hash(G_copy, node_attr='label', edge_attr='label')

def convert_igraph_to_networkx(ig_graph):
    """将 igraph 对象转换为 networkx 对象"""
    nx_graph = nx.DiGraph()
    for v in ig_graph.vs:
        nx_graph.add_node(v.index, **v.attributes())
    for e in ig_graph.es:
        nx_graph.add_edge(e.source, e.target, **e.attributes())
    return nx_graph

def initialize_vertex_edge_price_list(G):
    """初始化卖家的成本(Cost)和价格(Price)"""
    vertex_price_list = {}
    edge_price_list = {}
    seen_vertex_names = set()
    seen_edge_keys = set()
    
    IP = Config.InitParams

    # 初始化节点
    for node in G.nodes:
        name = G.nodes[node].get('name', node)
        if name in seen_vertex_names: continue
        seen_vertex_names.add(name)
        
        cost = random.randint(IP.VERTEX_COST_MIN, IP.VERTEX_COST_MAX)
        price_add = random.randint(IP.VERTEX_PRICE_ADD_MIN, IP.VERTEX_PRICE_ADD_MAX)
        vertex_price_list[name] = {'cost': cost, 'price': cost + price_add}

    # 初始化边
    for u, v, attr in G.edges(data=True):
        name_u = G.nodes[u].get('name', u)
        name_v = G.nodes[v].get('name', v)
        edge_key = tuple(sorted((name_u, name_v))) # 无向处理用于定价键值
        
        if edge_key in seen_edge_keys: continue
        seen_edge_keys.add(edge_key)
        
        # 原代码逻辑：边的 weight 即为 cost
        edge_weight = attr.get('weight', random.randint(IP.EDGE_WEIGHT_MIN, IP.EDGE_WEIGHT_MAX))
        price_add = random.randint(IP.EDGE_PRICE_ADD_MIN, IP.EDGE_PRICE_ADD_MAX)
        
        edge_price_list[edge_key] = {
            'weight': edge_weight, # This is COST
            'price': edge_weight + price_add
        }
    return vertex_price_list, edge_price_list

def initialize_buyer_price_dict(queries):
    """初始化买家心理预期价格"""
    buyer_price_dict = {}
    IP = Config.InitParams
    for Q in queries:
        expected_price = (Q.number_of_nodes() * IP.BUYER_NODE_MULTIPLIER + 
                          random.uniform(0, IP.BUYER_RANDOM_FLUCTUATION))
        sig = graph_signature(Q)
        buyer_price_dict[sig] = expected_price
    return buyer_price_dict
