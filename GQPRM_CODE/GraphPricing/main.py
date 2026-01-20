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
    with open(filename, 'rb') as f:
        graph_list = pickle.load(f)
    return graph_list

def initialize_vertex_edge_price_list(G):
    vertex_price_list = {}
    edge_price_list = {}
    seen_vertex_names = set()
    seen_edge_keys = set()

    for node in G.nodes:
        name = G.nodes[node].get('name', node)

        if name in seen_vertex_names:
            continue
        seen_vertex_names.add(name)

        cost = random.randint(1, 4)
        price = random.randint(cost + 1, cost + 6)

        vertex_price_list[name] = {'cost': cost, 'price': price}

    for u, v, attr in G.edges(data=True):
        name_u = G.nodes[u].get('name', u)
        name_v = G.nodes[v].get('name', v)

        edge_key = tuple(sorted((name_u, name_v)))
        if edge_key in seen_edge_keys:
            continue

        seen_edge_keys.add(edge_key)

        edge_weight = attr.get('weight', random.randint(1, 3))
        edge_price = random.randint(edge_weight + 1, edge_weight + 6)

        edge_price_list[edge_key] = {
            'weight': edge_weight,
            'price': edge_price
        }

    return vertex_price_list, edge_price_list

def custom_weisfeiler_lehman_hash(G, node_attr="label", edge_attr=None, digest_size=16):
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
    G_copy = nx.Graph()
    for n, attr in G.nodes(data=True):
        name = attr.get('name', str(n))
        G_copy.add_node(n, label=name)
    for u, v, attr in G.edges(data=True):
        label = attr.get('label', '')
        G_copy.add_edge(u, v, label=label)
    return custom_weisfeiler_lehman_hash(G_copy, node_attr='label', edge_attr='label')

def initialize_buyer_price_dict(queries):
    buyer_price_dict = {}
    for Q in queries:
        expected_price = Q.number_of_nodes() * 5 + random.uniform(0, 15)
        sig = graph_signature(Q)
        buyer_price_dict[sig] = expected_price
    return buyer_price_dict

def convert_igraph_to_networkx(ig_graph):
    nx_graph = nx.DiGraph()
    for v in ig_graph.vs:
        nx_graph.add_node(v.index, **v.attributes())
    for e in ig_graph.es:
        nx_graph.add_edge(e.source, e.target, **e.attributes())
    return nx_graph

if __name__ == '__main__':
    # ca-AstroPh 数据加载配置
    loader = GraphDataLoader("ca-AstroPh.txt.gz", "subgraphs_email_cutto10.pkl")
    loader.load_graph()
    loader.load_subgraphs()
    G = loader.get_graph()
    print(G.number_of_nodes())
    subgraphs = loader.get_subgraphs()

    print(len(subgraphs))
    print(type(subgraphs[0]))
  
    # 配置
    xpricing = 8
    ep = 1
    result_file_name = f"EM_x{xpricing}_{ep}_50k_revenue_evaluation_results.txt"

    generator = QueryGenerator(G, query_num=50000, max_query_edges=5)
    
    # 注释掉外部文件加载，直接使用 subgraphs 生成，保持代码独立可运行性
    # loaded_graphs = load_graphs_from_pickle('queries_EMAIL_50000.pkl')
    # generator.generate_queries(loaded_graphs)
    generator.generate_queries(subgraphs)

    queries = generator.get_queries()
    print(len(queries))

    vertex_price_list, edge_price_list = initialize_vertex_edge_price_list(G)

    buyer_price_list = initialize_buyer_price_dict(queries)
    print('len of expected price list:', len(buyer_price_list))
    
    start_time = time.time()
    
    transaction_manager = TransactionManager()
    full_transaction = FullTransactionLogger()
    evaluator = TransactionEvaluator(buyer_price_list)
    
    graph_pricing_transaction.process_transactions(
        queries, subgraphs, vertex_price_list, edge_price_list, 
        transaction_manager, full_transaction, 
        buyer_price_list, evaluator, start_time, 
        xpricing=xpricing, 
        centrality_file='EMAIL_centrality.pkl',
        result_file_name=result_file_name
    )
