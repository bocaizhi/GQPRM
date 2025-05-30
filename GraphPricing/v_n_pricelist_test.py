import random
from graph_data_loader import GraphDataLoader
from query_generator import QueryGenerator
import graph_pricing_transaciton
from graph_pricing_transaciton import TransactionManager
from graph_pricing_transaciton import TransactionEvaluator
import networkx as nx
import time

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

        cost = random.randint(1, 8)
        price = random.randint(cost + 1, 15)

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

        edge_weight = attr.get('weight', random.randint(1, 5))
        edge_price = random.randint(edge_weight + 1, edge_weight + 10)

        edge_price_list[edge_key] = {
            'weight': edge_weight,
            'price': edge_price
        }

    return vertex_price_list, edge_price_list

loader = GraphDataLoader("dbpedia.ttl", "subgraphs.pkl")
loader.load_rdf_graph()
loader.load_subgraphs()
G = loader.get_graph()
print(G.number_of_nodes())
subgraphs = loader.get_subgraphs()
# subgraphs = []
# for subG in subgraphs:
#     subg = convert_igraph_to_networkx(subG)
#     subgraphs.append(subg)
print(len(subgraphs))
print(type(subgraphs[0]))

vertex_price_list, edge_price_list = initialize_vertex_edge_price_list(G)
print(len(vertex_price_list),len(edge_price_list))

#原compute_cost
def computecost_G(Q, vertex_price_list, edge_price_list):
    """
    计算查询图 Q 的执行成本（包括顶点成本 + 边权重）

    :param Q: 查询图 (networkx.Graph)
    :param vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}
    :param edge_price_list: 边价格字典 {(u, v): {'weight': w, 'price': p}}
    :return: 查询图的总成本
    """
    total_cost = 0
    used_vertex_names = set()
    used_edge_keys = set()

    for node in Q.nodes:
        name = Q.nodes[node].get('name', node)
        if name in vertex_price_list and name not in used_vertex_names:
            total_cost += vertex_price_list[name]['cost']
            used_vertex_names.add(name)

    for u, v in Q.edges:
        name_u = Q.nodes[u].get('name', u)
        name_v = Q.nodes[v].get('name', v)
        edge_key = tuple(sorted((name_u, name_v)))

        if edge_key in edge_price_list and edge_key not in used_edge_keys:
            total_cost += edge_price_list[edge_key]['weight']
            used_edge_keys.add(edge_key)

    return total_cost


#原compute_price
def calculate_query_price(query_graph, vertex_price_list, edge_price_list):
    """
    计算一个查询图的价格，按顶点价格 + 边价格累加。
    
    :param query_graph: 查询图（NetworkX 格式，具有 'name' 属性）
    :param vertex_price_list: 顶点价格字典
    :param edge_price_list: 边价格字典
    :return: 查询图总价格
    """
    total_price = 0
    used_vertices = set()

    # 累加顶点价格（按 name）
    for node in query_graph.nodes:
        name = query_graph.nodes[node].get('name', node)
        if name in vertex_price_list and name not in used_vertices:
            total_price += vertex_price_list[name]['price']
            used_vertices.add(name)

    # 累加边价格（按 name_name 组合）
    for u, v in query_graph.edges:
        name_u = query_graph.nodes[u].get('name', u)
        name_v = query_graph.nodes[v].get('name', v)
        edge_key = tuple(sorted((name_u, name_v)))
        if edge_key in edge_price_list:
            total_price += edge_price_list[edge_key]['price']

    return total_price

#原adjust_vertex_price
def update_prices_by_feedback_proportional(query_graph, vertex_price_list, edge_price_list, new_price):
    """
    根据新的目标价格 new_price，按当前成本比例等比调整查询图中顶点与边的价格，
    保证调整后总价为 new_price。

    :param query_graph: 查询图 (networkx.Graph)
    :param vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}
    :param edge_price_list: 边价格字典 {(u, v): {'weight': w, 'price': p}}
    :param new_price: 查询图的目标价格 (float)
    :return: 更新后的 vertex_price_list 和 edge_price_list
    """
    used_vertex_names = set()
    used_edge_keys = set()

    # 计算 query_graph 的当前成本总和（顶点 + 边）
    total_cost = 0

    for node in query_graph.nodes:
        name = query_graph.nodes[node].get('name', node)
        if name in vertex_price_list and name not in used_vertex_names:
            total_cost += vertex_price_list[name]['cost']
            used_vertex_names.add(name)

    for u, v in query_graph.edges:
        name_u = query_graph.nodes[u].get('name', u)
        name_v = query_graph.nodes[v].get('name', v)
        edge_key = tuple(sorted((name_u, name_v)))

        if edge_key in edge_price_list and edge_key not in used_edge_keys:
            total_cost += edge_price_list[edge_key]['weight']
            used_edge_keys.add(edge_key)

    if total_cost == 0:
        raise ValueError("查询图的总成本为 0，无法进行价格等比调整")

    # 等比调整比例
    adjustment_ratio = new_price / total_cost

    # 更新顶点价格
    for name in used_vertex_names:
        cost = vertex_price_list[name]['cost']
        new_p = round(cost * adjustment_ratio, 4)
        vertex_price_list[name]['price'] = new_p

    # 更新边价格
    for edge_key in used_edge_keys:
        weight = edge_price_list[edge_key]['weight']
        new_p = round(weight * adjustment_ratio, 4)
        edge_price_list[edge_key]['price'] = new_p

    return vertex_price_list, edge_price_list


# def update_prices_by_feedback(query_graph, vertex_price_list, edge_price_list, feedback, lr=0.1):
#     """
#     根据反馈信息更新顶点和边的价格（简单线性更新）。
    
#     :param query_graph: 查询图
#     :param vertex_price_list: 当前顶点价格
#     :param edge_price_list: 当前边价格
#     :param feedback: 买家反馈（如成交为1，未成交为0，或任意 0~1 的数）
#     :param lr: 学习率 / 调整幅度
#     """
#     seen_vertices = set()

#     for node in query_graph.nodes:
#         name = query_graph.nodes[node].get('name', node)
#         if name in vertex_price_list and name not in seen_vertices:
#             old_price = vertex_price_list[name]['price']
#             vertex_price_list[name]['price'] += lr * (feedback - old_price)
#             vertex_price_list[name]['price'] = max(0.1, round(vertex_price_list[name]['price'], 2))  # 保证价格非负
#             seen_vertices.add(name)

#     for u, v in query_graph.edges:
#         name_u = query_graph.nodes[u].get('name', u)
#         name_v = query_graph.nodes[v].get('name', v)
#         edge_key = tuple(sorted((name_u, name_v)))
#         if edge_key in edge_price_list:
#             old_price = edge_price_list[edge_key]['price']
#             edge_price_list[edge_key]['price'] += lr * (feedback - old_price)
#             edge_price_list[edge_key]['price'] = max(0.1, round(edge_price_list[edge_key]['price'], 2))
