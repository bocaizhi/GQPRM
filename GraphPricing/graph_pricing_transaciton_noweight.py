from output_evaluator import TransactionEvaluator
from networkx.algorithms import isomorphism
import networkx as nx
import numpy as np
import json
import random
import time

def graph_signature(G):
    """
    用于为无向图生成唯一的同构签名（无属性版本）。
    使用 Graph canonical labeling 来做图结构签名。
    """
    gm = nx.convert_node_labels_to_integers(G, label_attribute="old_label")
    return nx.weisfeiler_lehman_graph_hash(gm)

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
        """返回所有的简化记录（用于全局评估）"""
        return list(self.transactions.values())

# 完整交易记录类 (Full transaction logger)
class FullTransactionLogger:
    def __init__(self):
        self.history = []  # 存储完整交易记录: (Q, price, success, timestamp)
    
    def add_transaction(self, Q, price, success, timestamp=None):
        """
        记录一笔新的完整交易 (Record a new transaction)
        :param Q: 查询图 (networkx.Graph)
        :param price: 交易价格
        :param success: 是否交易成功 (True/False)
        :param timestamp: 可选交易时间戳，默认使用当前时间
        """
        self.history.append((Q.copy(), price, success))
    
    def get_transaction_history(self, Q):
        """
        获取与 Q 同构图的交易历史 (Retrieve transaction history for graphs isomorphic to Q)
        """
        from networkx.algorithms.isomorphism import GraphMatcher
        result = []
        for record in self.history:
            graph, price, success = record
            if nx.is_isomorphic(Q, graph):
                result.append(record)
        return result

    def get_history(self):
        """
        获取所有交易历史记录 (Retrieve all transaction history)
        :return: 交易历史的列表 (List of all transactions)
        """
        return self.history

# class FullTransactionLogger:
#     def __init__(self):
#         self.full_records = {}  # key: graph_signature -> list of full transaction dicts

#     def add_transaction(self, Q, price, success, timestamp=None):
#         """
#         添加一笔完整交易记录
#         :param Q: 查询图 (networkx.Graph)
#         :param price: 交易价格
#         :param success: 是否成功（True/False）
#         :param timestamp: 可选时间戳（若为 None 则使用当前时间）
#         """
#         key = graph_signature(Q)
#         if key not in self.full_records:
#             self.full_records[key] = []

#         transaction = {
#             'graph': Q.copy(),  # 存储当前图的拷贝
#             'price': price,
#             'success': success,
#             'timestamp': timestamp if timestamp else time.time()
#         }
#         self.full_records[key].append(transaction)

#     def get_transactions(self, Q):
#         """
#         获取某个查询图的所有交易记录
#         """
#         key = graph_signature(Q)
#         return self.full_records.get(key, [])

#     def get_all_transactions(self):
#         """
#         获取所有图的交易记录（按签名组织）
#         """
#         return self.full_records

#     def get_all_records_flattened(self):
#         """
#         获取所有交易记录的扁平化列表（适合做整体统计）
#         """
#         records = []
#         for tx_list in self.full_records.values():
#             records.extend(tx_list)
#         return records

def gequal(Q1, Q2):
    """
    检查两个无向查询图 Q1 和 Q2 是否等价（包括节点 'name' 和边 'weight' 属性）
    
    :param Q1: networkx.Graph 查询图 1
    :param Q2: networkx.Graph 查询图 2
    :return: 布尔值，表示两图是否结构与属性等价
    """
    if Q1.number_of_nodes() != Q2.number_of_nodes() or Q1.number_of_edges() != Q2.number_of_edges():
        return False

    node_match = isomorphism.categorical_node_match('name', None)
    edge_match = isomorphism.categorical_edge_match('weight', None)

    gm = isomorphism.GraphMatcher(Q1, Q2, node_match=node_match, edge_match=edge_match)
    return gm.is_isomorphic()

# def gequal(Q1, Q2):
#     """
#     检查两个查询图 Q1 和 Q2 是否等价 (Check if two query graphs Q1 and Q2 are equivalent)
#     - 先比较顶点和边的数量 (First compare vertex and edge count)
#     - 再进行图同构检查 (Then check for graph isomorphism)

#     :param Q1: networkx.DiGraph 查询图 1
#     :param Q2: networkx.DiGraph 查询图 2
#     :return: 布尔值，表示两图是否等价
#     """
#     if Q1.number_of_nodes() != Q2.number_of_nodes() or Q1.number_of_edges() != Q2.number_of_edges():
#         return False  # 如果顶点或边数量不同，则一定不同

#     # 使用 networkx 的 isomorphism 进行同构检测
#     gm = nx.algorithms.isomorphism.DiGraphMatcher(Q1, Q2)
#     return gm.is_isomorphic()

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

def is_subgraph_match(G1, G2):
    """
    判断 G1 是否是 G2 的子图，同构匹配顶点 'name' 属性，忽略边属性
    """
    GM = isomorphism.GraphMatcher(
        G2, G1,
        node_match=lambda n1, n2: n1.get('name') == n2.get('name')
    )
    return GM.subgraph_is_isomorphic()

def write_subisomorphic_flag(Q, subgraphs):
    """
    返回每个子图与 Q 的（Q ⊆ subgraph, subgraph ⊆ Q）同构关系标记
    """
    sub_flag = []
    for sub in subgraphs:
        flag1 = is_subgraph_match(sub, Q)  # sub ⊆ Q
        flag2 = is_subgraph_match(Q, sub)  # Q ⊆ sub
        sub_flag.append([flag1, flag2])
    return sub_flag

def combine_graphs(graph_list):
    """
    合并多个 networkx 图，节点按 name 匹配，边无属性合并
    """
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
    """
    基于无向图的子图同构定价函数。
    匹配顶点 'name' 属性，忽略边属性，按顶点+边价格累加。
    
    :param query_graph: 查询图（NetworkX 格式）
    :param subgraphs: 数据图集合（NetworkX 图列表）
    :param vertex_price_list: 顶点价格字典
    :param edge_price_list: 边价格字典
    :return: (价格, 匹配子图列表)
    """
    matched_graphs = []
    sub_flag = write_subisomorphic_flag(query_graph, subgraphs)

    # Step 1: 精确匹配 Q ⊆ subgraph
    for i, flag in enumerate(sub_flag):
        if flag[1]:  # Q 是 subgraph 的子图
            price = calculate_query_price(query_graph, vertex_price_list, edge_price_list)
            matched_graphs.append(subgraphs[i])
            return [price, matched_graphs]

    # Step 2: 模糊匹配 subgraph ⊆ Q
    fuzzy_candidates = [subgraphs[i] for i, flag in enumerate(sub_flag) if flag[0]]
    if not fuzzy_candidates:
        print("No answer.")
        return [0, []]

    # Step 3: 合并模糊匹配图，再次尝试匹配
    combined_graph = combine_graphs(fuzzy_candidates)
    if is_subgraph_match(query_graph, combined_graph):
        price = calculate_query_price(query_graph, vertex_price_list, edge_price_list)
        return [price, fuzzy_candidates]
    else:
        print("No answer.")
        return [0, []]


#原adjust_vertex_price
def update_prices_by_feedback(query_graph, vertex_price_list, edge_price_list, new_price):
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
    xpricing=1
):
    evaluation_intervals = {10000, 30000, 50000, 80000, 120000, 150000}
    results = {}

    for i, Q in enumerate(queries, start=1):
        expected_price = evaluator.generate_expected_price(Q)

        # 更新或添加预期价格记录
        found = False
        for p_b in plist_buyer:
            if gequal(Q, p_b[0]):
                p_b[1] = expected_price
                found = True
                break
        if not found:
            plist_buyer.append([Q, expected_price])

        # --- Pricing Section ---
        if xpricing == 0:
            # 子图匹配定价
            price, _ = subisomorphic_pricing(Q, subgraphs, vertex_price_list, edge_price_list)
        else:
            record = transaction_manager.get_summary(Q)

            if record:
                # 历史记录定价
                cost = computecost_G(Q, vertex_price_list, edge_price_list)
                price = calculate_query_price(Q, vertex_price_list, edge_price_list)

                sp_max = record['max_success_price']
                fp_min = record['min_fail_price']
                success_count = record['success_count']
                fail_count = record['fail_count']

                new_price = price

                if expected_price < price:
                    success = False
                    print(f'第{i}次交易：失败 - Q: {Q}, 价格: {price}, 预期价格: {expected_price}')
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
                    print(f'第{i}次交易：成功 - Q: {Q}, 价格: {price}, 预期价格: {expected_price}')
                    if fail_count > 0 and fp_min != float('inf'):
                        new_price = (fp_min + price) / 2
                    else:
                        new_price = price + xpricing

                if not isinstance(new_price, (int, float)) or not (new_price < float('inf')):
                    print("Error: 非法 new_price，跳过 Q")
                    continue

                if new_price != price:
                    vertex_price_list, edge_price_list = update_prices_by_feedback(
                        Q, vertex_price_list, edge_price_list, new_price
                    )

                price = new_price

            else:
                # 无历史，退回子图匹配定价
                price, _ = subisomorphic_pricing(Q, subgraphs, vertex_price_list, edge_price_list)
                print(subisomorphic_pricing(Q, subgraphs, vertex_price_list, edge_price_list))

        # 成功/失败判断
        success = expected_price >= price
        print(f'第{i}次交易：{"成功" if success else "失败"} - Q: {Q}, 价格: {price}, 预期价格: {expected_price}')

        # 记录交易
        transaction_manager.add_transaction(Q, price, success)
        full_transaction.add_transaction(Q, price, success)

        # 定期评估并保存结果
        if i in evaluation_intervals:
            e_time = time.time() - start_time
            metrics = evaluator.evaluate_transactions(full_transaction.get_history())
            results = {
                "success_ratio": metrics[0],
                "avg_regret": metrics[1],
                "avg_price_deviation": metrics[2],
                "avg_sdiff": metrics[3],
                "avg_diff": metrics[4],
                "avg_per_diff": metrics[5],
                "customer_avg_per_diff": metrics[6],
                "time": e_time
            }
            evaluator.save_evaluation_results(i, "TEST_x0_test_evaluation_results.txt", results)

    # 最终评估
    e_time = time.time() - start_time
    metrics = evaluator.evaluate_transactions(full_transaction.get_history())
    results = {
        "success_ratio": metrics[0],
        "avg_regret": metrics[1],
        "avg_price_deviation": metrics[2],
        "avg_sdiff": metrics[3],
        "avg_diff": metrics[4],
        "avg_per_diff": metrics[5],
        "customer_avg_per_diff": metrics[6],
        "time": e_time
    }
    print(results)
    evaluator.save_evaluation_results(i, "TEST_x0_test_evaluation_results.txt", results)


# def process_transactions(queries, subgraphs, vertex_price_list, edge_price_list, transaction_manager, full_transaction, plist_buyer, evaluator, start_time, xpricing=1):
#     evaluation_intervals = {10000, 30000, 50000, 80000, 120000, 150000}
#     results = {}

#     for i, Q in enumerate(queries, start=1):
#         expected_price = evaluator.generate_expected_price(Q)
#         price = calculate_query_price(Q, vertex_price_list, edge_price_list)
#         print('exp:', expected_price, 'price:', price)

#         #预期价格部分待优化
#         found = False
#         for p_b in plist_buyer:
#             if gequal(Q, p_b[0]):
#                 found = True
#                 p_b[1] = expected_price
#         if not found:
#             plist_buyer.append([Q, expected_price])

#         record = transaction_manager.get_summary(Q)
#         sp_max = float('-inf')
#         fp_min = float('inf')
#         success_count = 0
#         fail_count = 0

#         if record:
#             sp_max = record['max_success_price']
#             fp_min = record['min_fail_price']
#             success_count = record['success_count']
#             fail_count = record['fail_count']

#         if price is None:
#             continue

#         new_price = price

#         if expected_price < price:
#             success = False
#             print(f'第{i}次交易：失败 - Q: {Q}, 价格: {price}, 预期价格: {expected_price}')
#             if success_count > 0:
#                 new_price = (sp_max + price) / 2
#             else:
#                 cost = computecost_G(Q, vertex_price_list, edge_price_list)
#                 diff = price - cost
#                 if diff > xpricing:
#                     new_price = price - xpricing
#                 elif diff > (xpricing / 2):
#                     new_price = price - xpricing / 2
#                 elif diff > 1:
#                     new_price = price - 1
#                 else:
#                     new_price = cost + 1
#         else:
#             success = True
#             print(f'第{i}次交易：成功 - Q: {Q}, 价格: {price}, 预期价格: {expected_price}')
#             if fail_count > 0:
#                 if fp_min == float('inf'):
#                     print("Warning: fp_min 未正确更新，跳过该 Q")
#                     continue
#                 new_price = (fp_min + price) / 2
#             else:
#                 new_price = price + xpricing

#         if not isinstance(new_price, (int, float)) or new_price == float('inf') or new_price != new_price:
#             print("Error: 计算得到的 new_price 非法（inf 或 NaN），跳过该 Q")
#             continue

#         cost = computecost_G(Q, vertex_price_list, edge_price_list)

#         if new_price != price:
#             vertex_price_list, edge_price_list = update_prices_by_feedback(
#                 Q, vertex_price_list, edge_price_list, new_price
#             )

#         transaction_manager.add_transaction(Q, price, success)
#         full_transaction.add_transaction(Q, price, success)
#         # # 查看 Q1 的所有交易记录
#         # records_q1 = full_transaction.get_history()
#         # for record in records_q1:
#         #     print(record)
#         if i in evaluation_intervals:
#             e_time = time.time() - start_time
#             success_ratio, avg_regret, avg_price_deviation, avg_sdiff, avg_diff, avg_per_diff, customer_avg_per_diff = evaluator.evaluate_transactions( full_transaction.get_history())
#             results = {
#                 "success_ratio": success_ratio,
#                 "avg_regret": avg_regret,
#                 "avg_price_deviation": avg_price_deviation,
#                 "avg_sdiff": avg_sdiff,
#                 "avg_diff": avg_diff,
#                 "avg_per_diff": avg_per_diff,
#                 "customer_avg_per_diff": customer_avg_per_diff,
#                 "time": e_time
#             }
#             evaluator.save_evaluation_results(i, "TEST_x0_test_evaluation_results.txt", results)

#     e_time = time.time() - start_time
#     success_ratio, avg_regret, avg_price_deviation, avg_sdiff, avg_diff, avg_per_diff, customer_avg_per_diff = evaluator.evaluate_transactions( full_transaction.get_history())
#     results = {
#         "success_ratio": success_ratio,
#         "avg_regret": avg_regret,
#         "avg_price_deviation": avg_price_deviation,
#         "avg_sdiff": avg_sdiff,
#         "avg_diff": avg_diff,
#         "avg_per_diff": avg_per_diff,
#         "customer_avg_per_diff": customer_avg_per_diff,
#         "time": e_time
#     }
#     print(results)
#     evaluator.save_evaluation_results(i, "TEST_x0_test_evaluation_results.txt", results)

