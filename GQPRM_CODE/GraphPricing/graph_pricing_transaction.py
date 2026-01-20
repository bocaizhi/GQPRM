from output_evaluator import TransactionEvaluator
from networkx.algorithms import isomorphism
import networkx as nx
import numpy as np
import pickle
import pandas as pd
import json
import random
import time
from collections import defaultdict
import hashlib

# ----------------------------
# 类定义
# ----------------------------

# 完整交易记录类 (Full transaction logger)
class FullTransactionLogger:
    def __init__(self):
        # 修改：存储完整交易记录增加 cost 字段: (Q, price, success, cost)
        self.history = []  
    
    def add_transaction(self, Q, price, success, cost, timestamp=None):
        """
        记录一笔新的完整交易
        :param Q: 查询图 (networkx.Graph)
        :param price: 交易价格
        :param success: 是否交易成功 (True/False)
        :param cost: 交易时的成本 (新增)
        :param timestamp: 可选交易时间戳
        """
        # 保存 cost，用于计算 Revenue
        self.history.append((Q.copy(), price, success, cost))
    
    def get_transaction_history(self, Q):
        """
        获取与 Q 同构图的交易历史
        """
        result = []
        for record in self.history:
            # record 解包: graph, price, success, cost
            graph, price, success, cost = record
            if nx.is_isomorphic(Q, graph):
                result.append(record)
        return result

    def get_history(self):
        """
        获取所有交易历史记录
        """
        return self.history


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

class TransactionManager:
    def __init__(self):
        self.transactions = {}  # key: graph_signature -> summarized transaction record

    def add_transaction(self, Q, price, success):
        key = graph_signature(Q)
        if key not in self.transactions:
            self.transactions[key] = {
                'graph': Q,
                'max_success_price': float('-inf'),
                'min_fail_price': float('inf'),
                'success_count': 0,
                'fail_count': 0
            }

        record = self.transactions[key]
        if success:
            record['max_success_price'] = max(record['max_success_price'], price)
            record['success_count'] += 1
        else:
            record['min_fail_price'] = min(record['min_fail_price'], price)
            record['fail_count'] += 1

    def get_summary(self, Q):
        key = graph_signature(Q)
        return self.transactions.get(key)

    def get_history(self):
        return list(self.transactions.values())

# ----------------------------
# 函数定义
# ----------------------------

def gequal(Q1, Q2):
    if Q1.number_of_nodes() != Q2.number_of_nodes() or Q1.number_of_edges() != Q2.number_of_edges():
        return False
    node_match = isomorphism.categorical_node_match('name', None)
    edge_match = isomorphism.categorical_edge_match('weight', None)
    gm = isomorphism.GraphMatcher(Q1, Q2, node_match=node_match, edge_match=edge_match)
    return gm.is_isomorphic()

def calculate_query_price(query_graph, vertex_price_list, edge_price_list):
    total_price = 0
    used_vertices = set()

    for node in query_graph.nodes:
        name = query_graph.nodes[node].get('name', node)
        if name in vertex_price_list and name not in used_vertices:
            total_price += vertex_price_list[name]['price']
            used_vertices.add(name)

    for u, v in query_graph.edges:
        name_u = query_graph.nodes[u].get('name', u)
        name_v = query_graph.nodes[v].get('name', v)
        edge_key = tuple(sorted((name_u, name_v)))
        if edge_key in edge_price_list:
            total_price += edge_price_list[edge_key]['price']

    return total_price

def computecost_G(Q, vertex_price_list, edge_price_list):
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

def is_subgraph_match(G1, G2):
    GM = isomorphism.GraphMatcher(
        G2, G1,
        node_match=lambda n1, n2: n1.get('name') == n2.get('name')
    )
    return GM.subgraph_is_isomorphic()

def write_subisomorphic_flag(Q, subgraphs):
    q_node_names = {Q.nodes[n].get("name") for n in Q.nodes}
    sub_flag = []

    for sub in subgraphs:
        sub_node_names = {sub.nodes[n].get("name") for n in sub.nodes}
        if sub_node_names.issubset(q_node_names):
            flag1 = is_subgraph_match(sub, Q)
        else:
            flag1 = False
        if q_node_names.issubset(sub_node_names):
            flag2 = is_subgraph_match(Q, sub)
        else:
            flag2 = False
        sub_flag.append([flag1, flag2])
    return sub_flag

def combine_graphs(graph_list):
    combined = nx.Graph()
    for g in graph_list:
        for node, data in g.nodes(data=True):
            if node not in combined:
                combined.add_node(node, **data)
        for u, v in g.edges():
            if not combined.has_edge(u, v):
                combined.add_edge(u, v)
    return combined

def subisomorphic_pricing(query_graph, subgraphs, vertex_price_list, edge_price_list):
    matched_graphs = []
    sub_flag = write_subisomorphic_flag(query_graph, subgraphs)

    for i, flag in enumerate(sub_flag):
        if flag[1]:
            price = calculate_query_price(query_graph, vertex_price_list, edge_price_list)
            matched_graphs.append(subgraphs[i])
            return [price, matched_graphs]

    fuzzy_candidates = [subgraphs[i] for i, flag in enumerate(sub_flag) if flag[0]]
    if not fuzzy_candidates:
        print("No answer.")
        return [0, []]

    combined_graph = combine_graphs(fuzzy_candidates)
    if is_subgraph_match(query_graph, combined_graph):
        price = calculate_query_price(query_graph, vertex_price_list, edge_price_list)
        return [price, fuzzy_candidates]
    else:
        print("No answer.")
        return [0, []]

def compute_frequency_and_success_rate(Q, full_transaction_history):
    vertex_freq = defaultdict(int)
    vertex_success = defaultdict(int)
    edge_freq = defaultdict(int)
    edge_success = defaultdict(int)

    query_nodes = set(Q.nodes())
    query_edges = set(Q.edges())

    # 注意：history 现在有4个元素 (graph, price, success, cost)
    for record in full_transaction_history:
        graph, price, success, _ = record # 忽略 cost
        
        matched_nodes = set(graph.nodes()).intersection(query_nodes)
        for node in matched_nodes:
            vertex_freq[node] += 1
            if success:
                vertex_success[node] += 1

        matched_edges = set()
        for (u, v) in graph.edges():
            if (u in query_nodes) and (v in query_nodes):
                if (u, v) in query_edges or (v, u) in query_edges:
                    matched_edges.add((u, v))

        for (u, v) in matched_edges:
            edge_freq[(u, v)] += 1
            if success:
                edge_success[(u, v)] += 1

    vertex_success_rate = {
        node: vertex_success[node] / vertex_freq[node]
        for node in vertex_freq
    }
    edge_success_rate = {
        edge: edge_success[edge] / edge_freq[edge]
        for edge in edge_freq
    }

    return vertex_freq, vertex_success_rate, edge_freq, edge_success_rate

def get_centrality_for_query(Q, centrality_file):
    with open(centrality_file, 'rb') as f:
        vertex_centrality_all, edge_centrality_all = pickle.load(f)

    vertex_centrality_q = {}
    for node in Q.nodes():
        if node in vertex_centrality_all:
            vertex_centrality_q[node] = vertex_centrality_all[node]
        else:
            vertex_centrality_q[node] = 0

    edge_centrality_q = {}
    for u, v in Q.edges():
        if (u, v) in edge_centrality_all:
            edge_centrality_q[(u, v)] = edge_centrality_all[(u, v)]
        elif (v, u) in edge_centrality_all:
            edge_centrality_q[(u, v)] = edge_centrality_all[(v, u)]
        else:
            edge_centrality_q[(u, v)] = 0

    return vertex_centrality_q, edge_centrality_q

def entropy_weight_method(data_dict):
    if not data_dict:
        return {}

    df = pd.DataFrame.from_dict(data_dict, orient='index')
    if (df.sum().sum() == 0):
        return {item: 0 for item in data_dict}
    
    norm_df = (df - df.min()) / (df.max() - df.min() + 1e-12)
    eps = 1e-12
    P = norm_df / (norm_df.sum(axis=0) + eps)
    entropy = -np.sum(P * np.log(P + eps), axis=0) / np.log(len(df))

    redundancy = 1 - entropy
    weights = redundancy / np.sum(redundancy)

    final_scores = norm_df @ weights
    weight_dict = dict(zip(df.index, final_scores))

    return weight_dict

def compute_vertex_entropy_weights(vertex_freq_dict, vertex_succ_dict):
    data = {}
    for node in set(vertex_freq_dict) | set(vertex_succ_dict):
        data[node] = {
            'freq': vertex_freq_dict.get(node, 0),
            'succ_rate': vertex_succ_dict.get(node, 0)
        }
    return entropy_weight_method(data)

def compute_edge_entropy_weights(edge_freq_dict, edge_succ_dict):
    data = {}
    for edge in set(edge_freq_dict) | set(edge_succ_dict):
        data[edge] = {
            'freq': edge_freq_dict.get(edge, 0),
            'succ_rate': edge_succ_dict.get(edge, 0)
        }
    return entropy_weight_method(data)


def update_prices_advanced(
    query_graph,
    vertex_price_list,
    edge_price_list,
    new_price,
    centrality_file,
    full_transaction_history,
    centrality_weight=0.2
):
    def normalize(d):
        total = sum(d.values()) or 1e-6
        return {k: v / total for k, v in d.items()}

    vertex_centrality_all, edge_centrality_all = get_centrality_for_query(query_graph, centrality_file)

    used_vertices = set()
    used_edges = set()
    total_cost = 0

    for node in query_graph.nodes:
        name = query_graph.nodes[node].get('name', node)
        if name in vertex_price_list:
            used_vertices.add(name)
            total_cost += vertex_price_list[name]['cost']

    for u, v in query_graph.edges:
        name_u = query_graph.nodes[u].get('name', u)
        name_v = query_graph.nodes[v].get('name', v)
        edge_key = tuple(sorted((name_u, name_v)))
        if edge_key in edge_price_list:
            used_edges.add(edge_key)
            total_cost += edge_price_list[edge_key]['weight']

    if total_cost >= new_price:
        adjustment_ratio = new_price / total_cost
        for name in used_vertices:
            cost = vertex_price_list[name]['cost']
            vertex_price_list[name]['price'] = round(cost * adjustment_ratio, 4)
        for edge in used_edges:
            weight = edge_price_list[edge]['weight']
            edge_price_list[edge]['price'] = round(weight * adjustment_ratio, 4)
        return vertex_price_list, edge_price_list

    vertex_freq_dict, vertex_succ_dict, edge_freq_dict, edge_succ_dict = compute_frequency_and_success_rate(
        query_graph, full_transaction_history
    )

    v_weights = compute_vertex_entropy_weights(
        {k: vertex_freq_dict.get(k, 0) for k in used_vertices},
        {k: vertex_succ_dict.get(k, 0) for k in used_vertices}
    )
    e_weights = compute_edge_entropy_weights(
        {k: edge_freq_dict.get(k, 0) for k in used_edges},
        {k: edge_succ_dict.get(k, 0) for k in used_edges}
    )

    v_center = {k: vertex_centrality_all.get(k, 0) for k in used_vertices}
    e_center = {k: edge_centrality_all.get(k, 0) for k in used_edges}

    v_weights_norm = normalize(v_weights)
    e_weights_norm = normalize(e_weights)
    v_center_norm = normalize(v_center)
    e_center_norm = normalize(e_center)

    v_combined = {
        k: (1 - centrality_weight) * v_weights_norm.get(k, 0) + centrality_weight * v_center_norm.get(k, 0)
        for k in used_vertices
    }
    e_combined = {
        k: (1 - centrality_weight) * e_weights_norm.get(k, 0) + centrality_weight * e_center_norm.get(k, 0)
        for k in used_edges
    }

    remaining = new_price - total_cost
    total_combined = sum(v_combined.values()) + sum(e_combined.values()) or 1e-6
    v_ratio = {k: v / total_combined for k, v in v_combined.items()}
    e_ratio = {k: v / total_combined for k, v in e_combined.items()}

    for name in used_vertices:
        base_cost = vertex_price_list[name]['cost']
        alloc = remaining * v_ratio.get(name, 0)
        vertex_price_list[name]['price'] = round(base_cost + alloc, 4)

    for edge in used_edges:
        base_cost = edge_price_list[edge]['weight']
        alloc = remaining * e_ratio.get(edge, 0)
        edge_price_list[edge]['price'] = round(base_cost + alloc, 4)

    return vertex_price_list, edge_price_list


def process_transactions(
    queries,
    subgraphs,
    vertex_price_list,
    edge_price_list,
    transaction_manager,
    full_transaction,
    plist_buyer,
    evaluator,
    start_time,
    xpricing=1,
    centrality_file=None,
    result_file_name=None
):
    # ca-AstroPh 特定的评估区间
    evaluation_intervals = {1000, 5000, 10000, 20000, 30000, 50000, 80000, 120000, 150000}
    results = {}
    etime = []
    start_time = time.time()
    for i, Q in enumerate(queries, start=1):
        # 1. 获取/生成买家预期价格
        found = False
        sig = graph_signature(Q)
        if sig in plist_buyer:
            expected_price = plist_buyer[sig]
            found = True
        else:
            expected_price = evaluator.generate_expected_price(Q)
            plist_buyer[sig] = expected_price
            found = False

        # 2. 【核心修改】计算当前查询图的成本，无论是否使用
        cost = computecost_G(Q, vertex_price_list, edge_price_list)

        # --- Pricing Section ---
        if xpricing == 0:
            # 子图匹配定价
            price, _ = subisomorphic_pricing(Q, subgraphs, vertex_price_list, edge_price_list)
            if price != 0:
                success = expected_price >= price
                print(f'第{i}次交易：{"成功" if success else "失败"} - Q: {Q}, 子图匹配定价， 价格: {price}, 预期价格: {expected_price}')
            else:
                continue
        else:
            record = transaction_manager.get_summary(Q)

            if record:
                # 重新计算一次 price (虽然理论上 new_price会覆盖，但保持逻辑一致性)
                price = calculate_query_price(Q, vertex_price_list, edge_price_list)

                sp_max = record['max_success_price']
                fp_min = record['min_fail_price']
                success_count = record['success_count']
                fail_count = record['fail_count']

                new_price = price

                if expected_price < price:
                    success = False
                    print(f'第{i}次交易：失败 - Q: {Q},历史记录定价, 价格: {price}, 预期价格: {expected_price}')
                    if success_count > 0:
                        new_price = (sp_max + price) / 2
                    else:
                        diff = price - cost
                        if diff > xpricing:
                            new_price = price - xpricing
                        elif diff > (xpricing / 2):
                            new_price = price - xpricing / 2
                        elif diff > 1:
                            new_price = price - 1
                        else:
                            new_price = cost + 1
                else:
                    success = True
                    print(f'第{i}次交易：成功 - Q: {Q},历史记录定价, 价格: {price}, 预期价格: {expected_price}')
                    if fail_count > 0 and fp_min != float('inf'):
                        new_price = (fp_min + price) / 2
                    else:
                        new_price = price + xpricing

                if not isinstance(new_price, (int, float)) or not (new_price < float('inf')):
                    print("Error: 非法 new_price，跳过 Q")
                    continue
                
                if new_price != price:
                    vertex_price_list, edge_price_list = update_prices_advanced(
                        query_graph=Q,
                        vertex_price_list=vertex_price_list,
                        edge_price_list=edge_price_list,
                        new_price=new_price,
                        centrality_file=centrality_file,
                        full_transaction_history=full_transaction.get_history()
                    )

                price = new_price

            else:
                # 无历史记录：先进行子图匹配定价
                price, _ = subisomorphic_pricing(Q, subgraphs, vertex_price_list, edge_price_list)
                if price != 0:
                    success = expected_price >= price
                    print(f'第{i}次交易：{"成功" if success else "失败"} - Q: {Q}, 子图匹配定价, 价格: {price}, 预期价格: {expected_price}')

                    new_price = expected_price if success else price
                    vertex_price_list, edge_price_list = update_prices_advanced(
                        query_graph=Q,
                        vertex_price_list=vertex_price_list,
                        edge_price_list=edge_price_list,
                        new_price=new_price,
                        centrality_file=centrality_file,
                        full_transaction_history=full_transaction.get_history()
                    )
                else:
                    continue

        # 3. 【核心修改】记录交易时，传入 cost
        transaction_manager.add_transaction(Q, price, success)
        full_transaction.add_transaction(Q, price, success, cost)

        # 4. 评估
        if i in evaluation_intervals:
            t = time.time() - start_time
            etime.append(t)
            # 实时保存中间结果
            #evaluator.evaluate_transactions(full_transaction.get_history(), plist_buyer, filename=result_file_name, etime=etime)

    # 最终评估
    t = time.time() - start_time
    etime.append(t)
    results = evaluator.evaluate_transactions(full_transaction.get_history(), plist_buyer, filename=result_file_name, etime=etime)
