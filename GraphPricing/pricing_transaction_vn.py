
from output_evaluator import TransactionEvaluator

import networkx as nx
import numpy as np
import json
import random
import time

def gequal(Q1, Q2):
    """
    检查两个查询图 Q1 和 Q2 是否等价 (Check if two query graphs Q1 and Q2 are equivalent)
    - 先比较顶点和边的数量 (First compare vertex and edge count)
    - 再进行图同构检查 (Then check for graph isomorphism)

    :param Q1: networkx.DiGraph 查询图 1
    :param Q2: networkx.DiGraph 查询图 2
    :return: 布尔值，表示两图是否等价
    """
    if Q1.number_of_nodes() != Q2.number_of_nodes() or Q1.number_of_edges() != Q2.number_of_edges():
        return False  # 如果顶点或边数量不同，则一定不同

    # 使用 networkx 的 isomorphism 进行同构检测
    gm = nx.algorithms.isomorphism.DiGraphMatcher(Q1, Q2)
    return gm.is_isomorphic()

# def gequal(Q1, Q2):
#     """
#     检查两个查询图 Q1 和 Q2 是否等价 (Check if two query graphs Q1 and Q2 are equivalent)
#     - 先比较顶点和边的数量 (First compare vertex and edge count)
#     - 再进行图同构检查 (Then check for graph isomorphism)
#     """
#     if Q1.vcount() != Q2.vcount() or Q1.ecount() != Q2.ecount():
#         return False  # 如果顶点或边数量不同，则一定不同 (If vertex or edge count differs, they are not equal)
    
#     # 使用图同构算法检查两图是否相同 (Use graph isomorphism check)
#     return Q1.isomorphic(Q2)

# def computecost_G(Q, vertex_price_list):
#     """
#     计算查询图 Q 的执行成本 (Compute the execution cost of query graph Q)
#     - 成本 = 查询中所有顶点的价格之和 (Cost = Sum of vertex prices in the query)
#     """
#     cost = sum(vertex_price_list[v] for v in Q.vs.indices)  # 遍历 Q 的所有顶点索引，获取对应价格
#     return cost

def computecost_G(Q, vertex_price_list):
    """
    计算查询图 Q 的执行成本 (Compute execution cost of query graph Q)
    
    :param Q: 查询图 (networkx.Graph)
    :param vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}
    :return: 查询图的成本总和
    """
    cost = sum(
        vertex_price_list[node]['cost']
        for node in Q.nodes if node in vertex_price_list
    )  # 计算所有匹配顶点的成本总和
    return cost

# def computecost_G(Q, vertex_price_list):
#     """
#     计算查询图 Q 的执行成本 (Compute execution cost of query graph Q)
    
#     :param Q: 查询图 (igraph.Graph)
#     :param vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}
#     :return: 查询图的成本总和
#     """
#     cost = sum(
#         vertex_price_list[node['name']]['cost']
#         for node in Q.vs if node['name'] in vertex_price_list
#     )  # 计算所有匹配顶点的成本总和
#     return cost

# 判断 G1 是否是 G2 的子图 (Check if G1 is a subgraph of G2)
def is_subisomorphic(G1, G2):
    if G1.is_multigraph() or G2.is_multigraph():
        return False  # 仅支持简单图匹配 (Only support simple graph matching)
    if G1.number_of_nodes() > G2.number_of_nodes():
        return False  # 如果 G1 节点数大于 G2，则不可能是子图 (If G1 has more nodes than G2, it cannot be a subgraph)
    return nx.algorithms.isomorphism.GraphMatcher(G2, G1).subgraph_is_isomorphic()

# 判断两个图是否完全相同
# Check if two graphs are exactly the same
def are_graphs_equal(G1, G2):
    if not isinstance(G1, nx.Graph) or not isinstance(G2, nx.Graph):
        raise TypeError(f"Expected networkx.Graph, but got {type(G1)} and {type(G2)}")
    #faster_could_be_isomorphic(G1, G2)快速排除不相同的图，提升代码效率
    print(type(G1),type(G2))
    return G1.number_of_nodes() == G2.number_of_nodes() and nx.is_isomorphic(G1, G2)

def compute_pricing(Q, vertex_price_list):
    """
    计算查询图 Q 的价格（遍历 vertex_price_list 匹配 Q 的顶点），若存在价格为 inf 的顶点则报错。

    :param Q: 查询图 (Query Graph) - NetworkX Graph
    :param vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}
    :return: Q 的价格总和
    :raises ValueError: 若某顶点价格为 inf
    """
    # 获取 Q 中所有顶点的 name 或 ID 集合，方便后续匹配
    Q_node_names = {Q.nodes[n].get('name', n) for n in Q.nodes}
    
    total_price = 0
    for node_name, price_info in vertex_price_list.items():
        if node_name in Q_node_names:
            price = price_info.get('price', 0)
            
            # if isinstance(price, float) and not price.is_integer() == True:
            #     raise ValueError(f"顶点 '{node_name}' 的价格为小数")
            total_price += price

    return total_price

# def compute_pricing(Q, vertex_price_list):
#     """
#     计算查询图 Q 的价格（顶点价格求和）

#     :param Q: 查询图 (Query Graph) - NetworkX Graph
#     :param vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}
#     :return: Q 的价格总和
#     """
#     return sum(
#         vertex_price_list.get(Q.nodes[node].get('name', node), {'price': 0})['price']
#         for node in Q.nodes
#     )

# def compute_pricing(Q, subgraph_list, vertex_price_list):
#     """
#     计算查询图 Q 的价格（顶点价格求和）

#     :param Q: 查询图 (Query Graph) - NetworkX Graph
#     :param subgraph_list: 可能匹配的子图列表 (List of NetworkX Graphs)
#     :param vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}
#     :return: Q 的价格总和或 None
#     """
#     for subG in subgraph_list:
#         if nx.is_isomorphic(Q, subG):  # 确保 Q 与某个子图匹配
#             return sum(
#                 vertex_price_list[node]['price']
#                 for node in Q.nodes if node in vertex_price_list
#             )  # 计算所有匹配顶点的价格总和
#     return None  # 如果没有匹配的子图，则返回 None

# # 计算查询图的定价 (Compute the pricing for a query graph)
# def compute_pricing(Q, subgraph_list, vertex_price_list):
#     """
#     计算查询图 Q 的价格（顶点价格求和）

#     :param Q: 查询图 (Query Graph)
#     :param subgraph_list: 可能匹配的子图列表 (List of subgraphs)
#     :param vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}
#     :return: Q 的价格总和或 None
#     """
#     for subG in subgraph_list:
#         if are_graphs_equal(Q, subG):  # 确保 Q 与某个子图匹配
#             return sum(
#                 vertex_price_list[node['name']]['price']
#                 for node in Q.vs if node['name'] in vertex_price_list
#             )  # 计算所有匹配顶点的价格总和
#     return None  # 如果没有匹配的子图，则返回 None

def adjust_vertex_price(Q, cost, new_price, vertex_price_list):
    """
    调整查询图 Q 中所有顶点的价格，确保结果不为 inf，若为 inf 则报错。

    :param Q: 查询图 (networkx.Graph)
    :param cost: 当前成本 (float)
    :param new_price: 新价格 (float)
    :param vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}
    :return: 更新后的 vertex_price_list
    """
    if cost <= 0:
        return vertex_price_list  # 避免除以零
    adjustment_ratio = new_price / cost
    for node in Q.nodes:
        name = Q.nodes[node].get('name', node)  # 获取顶点名

        if name in vertex_price_list:
            # old_price = vertex_price_list[name]['price']
            # new_node_price = old_price * adjustment_ratio
            old_cost = vertex_price_list[name]['cost']
            new_node_price = old_cost * adjustment_ratio

            vertex_price_list[name]['price'] = new_node_price

    return vertex_price_list

# def adjust_vertex_price(Q, cost, new_price, vertex_price_list):
#     """
#     调整查询图 Q 中所有顶点的价格 (Adjust the prices of all vertices in query graph Q)

#     参数:
#     - Q: 交易涉及的查询图 (networkx.Graph)
#     - cost: 当前计算的查询成本 (float)
#     - new_price: 交易后计算的新价格 (float)
#     - vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}

#     返回:
#     - 更新后的 vertex_price_list
#     """
#     if cost <= 0:
#         return vertex_price_list  # 避免除零错误 (Avoid division by zero)

#     # 计算价格调整比例 (Compute price adjustment ratio)
#     adjustment_ratio = new_price / cost

#     # 遍历查询图 Q 中的所有顶点 (Iterate through all vertices in Q)
#     for node in Q.nodes:
#         if node in vertex_price_list:
#             # 计算新的顶点价格 (Compute new vertex price)
#             vertex_price_list[node]['price'] *= adjustment_ratio

#     return vertex_price_list

# def adjust_vertex_price(Q, cost, new_price, vertex_price_list):
#     """
#     调整查询图 Q 中所有顶点的价格 (Adjust the prices of all vertices in query graph Q)

#     参数:
#     - Q: 交易涉及的查询图 (igraph.Graph)
#     - cost: 当前计算的查询成本 (float)
#     - new_price: 交易后计算的新价格 (float)
#     - vertex_price_list: 顶点价格字典 {name: {'cost': c, 'price': p}}

#     返回:
#     - 更新后的 vertex_price_list
#     """
#     if cost <= 0:
#         return vertex_price_list  # 避免除零错误 (Avoid division by zero)

#     # 计算价格调整比例 (Compute price adjustment ratio)
#     adjustment_ratio = new_price / cost

#     # 遍历查询图 Q 中的所有顶点 (Iterate through all vertices in Q)
#     for node in Q.vs:
#         node_name = node['name']
#         if node_name in vertex_price_list:
#             # 计算新的顶点价格 (Compute new vertex price)
#             vertex_price_list[node_name]['price'] *= adjustment_ratio

#     return vertex_price_list
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


# class TransactionManager:
#     def __init__(self):
#         self.history = []  # 存储交易历史 (Store transaction history: (Q, price, success))
    
#     def add_transaction(self, Q, price, success):
#         self.history.append((Q, price, success))  # 记录交易信息 (Record transaction details)
    
#     def get_transaction_history(self, Q):
#         """获取查询 Q 的历史交易记录 (Retrieve transaction history for Q)"""
#         return [t for t in self.history if t[0] == Q]


# 交易流程模拟 (Simulate the transaction process)
# 在每次交易中：
# 1.查询是否有交易历史
#       1）若无交易历史则进行图匹配
#       2）若存在交易历史则根据历史价格给出定价，并记录最高成交价和最低失败价
# 2.根据买家预期价格进行交易是否成功的判断
# 3.根据交易结果和历史高成交价和最低失败价动态更新顶点价格
def process_transactions(queries, subgraphs, vertex_price_list, transaction_manager, plist_buyer, evaluator, start_time, xpricing = 1):
    evaluation_intervals = {10000, 30000, 50000, 80000, 120000, 150000}
    results = {}
    
    for i, Q in enumerate(queries, start=1):
        expected_price = evaluator.generate_expected_price(Q)  # 生成买家预期价格 (Generate buyer expected price)
        price = compute_pricing(Q, vertex_price_list)  # 计算查询图 Q 的价格 (Compute price for query Q)
        print('exp:',expected_price,'price:',price)

        # 随机生成买家的心理预期价格 (Randomly generate buyer's expected price)
        #p_buyer = Q.vcount() * 5 + random.uniform(0, 10)

        # 更新 plist_buyer
        found = False
        for p_b in plist_buyer:
            if gequal(Q, p_b[0]):
                found = True
                p_b[1] = expected_price
        if not found:
            plist_buyer.append([Q, expected_price])

        #优化后的历史交易访问方式
        record = transaction_manager.get_summary(Q)
        sp_max = float('-inf')
        fp_min = float('inf')
        success_count = 0
        fail_count = 0

        if record:
            sp_max = record['max_success_price']
            fp_min = record['min_fail_price']
            success_count = record['success_count']
            fail_count = record['fail_count']

        # if price is None or expected_price < price:
        if price is None:
            continue
        # 默认初始化 new_price 避免未定义
        new_price = price

        if expected_price < price:
            success = False
            print(f'第{i}次交易：失败 - Q: {Q}, 价格: {price}, 预期价格: {expected_price}')
            if success_count > 0:
                new_price = (sp_max + price) / 2
            else:
                cost = computecost_G(Q, vertex_price_list)

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
            if fail_count > 0:
                if fp_min == float('inf'):
                    print("Warning: fp_min 未正确更新，跳过该 Q")
                    continue
                new_price = (fp_min + price) / 2
            else:
                new_price = price + xpricing
        # 💡 检查 new_price 是否为 inf 或 NaN
        if not isinstance(new_price, (int, float)) or new_price == float('inf') or new_price != new_price:
            print("Error: 计算得到的 new_price 非法（inf 或 NaN），跳过该 Q")
            continue

        cost = computecost_G(Q, vertex_price_list)
        
        # 更新顶点价格 (Update vertex price)
        cost = computecost_G(Q, vertex_price_list)
        if new_price != price:
            vertex_price_list = adjust_vertex_price(Q, cost, new_price, vertex_price_list)
        
        transaction_manager.add_transaction(Q, price, success)  # 记录交易 (Record transaction)
        
        if i in evaluation_intervals:
            e_time = time.time()- start_time
            #success_ratio, avg_regret, avg_price_deviation = evaluator.evaluate_transactions(transaction_manager.get_history())
            success_ratio, avg_regret, avg_price_deviation, avg_sdiff, avg_diff, avg_per_diff, customer_avg_per_diff = evaluator.evaluate_transactions(transaction_manager.get_history())
            results = {
                "success_ratio": success_ratio,
                "avg_regret": avg_regret,
                "avg_price_deviation": avg_price_deviation,
                "avg_sdiff": avg_sdiff,
                "avg_diff": avg_diff,
                "avg_per_diff": avg_per_diff,
                "customer_avg_per_diff": customer_avg_per_diff,
                "time":e_time
            }
            evaluator.save_evaluation_results(i, "TEST_x0_test_evaluation_results.txt", results)
    e_time = time.time()- start_time
    tran_his = transaction_manager.get_history()
    success_ratio, avg_regret, avg_price_deviation, avg_sdiff, avg_diff, avg_per_diff, customer_avg_per_diff = evaluator.evaluate_transactions(transaction_manager.get_history())
    results = {
        "success_ratio": success_ratio,
        "avg_regret": avg_regret,
        "avg_price_deviation": avg_price_deviation,
        "avg_sdiff": avg_sdiff,
        "avg_diff": avg_diff,
        "avg_per_diff": avg_per_diff,
        "customer_avg_per_diff": customer_avg_per_diff,
        "time":e_time
    }
    print(results)
    evaluator.save_evaluation_results(i, "TEST_x0_test_evaluation_results.txt", results)
# def process_transactions(queries, subgraphs, vertex_price_list, transaction_manager, plist_buyer, evaluator, xpricing = 1):
#     evaluation_intervals = {10000, 30000, 50000, 80000, 120000, 150000}
#     results = {}
    
#     for i, Q in enumerate(queries, start=1):
#         expected_price = evaluator.generate_expected_price(Q)  # 生成买家预期价格 (Generate buyer expected price)
#         price = compute_pricing(Q, vertex_price_list)  # 计算查询图 Q 的价格 (Compute price for query Q)
#         print('exp:',expected_price,'price:',price)
#         if price == float('inf'):
#             print("Error!!!!!!!!!!!")
#             break
#         # 随机生成买家的心理预期价格 (Randomly generate buyer's expected price)
#         #p_buyer = Q.vcount() * 5 + random.uniform(0, 10)

#         # 检查是否已有历史预期价格 (Check if history exists)
#         flag = False
#         for p_b in plist_buyer:
#             if gequal(Q, p_b[0]):  
#                 flag = True
#                 if p_b[1] != expected_price:
#                     p_b[1] = expected_price  # 更新预期价格 (Update expected price)
#         if not flag:
#             plist_buyer.append([Q, expected_price])

#         # 获取历史交易记录 (Retrieve historical transaction records)
#         history = transaction_manager.get_transaction_history(Q)
#         success_count = 0
#         fail_count = 0
#         sp_max = float('-inf')  # 最高成功交易价格 (Max successful price)
#         fp_min = float('inf')  # 最低失败交易价格 (Min failed price)

#         # 遍历历史交易，记录最高成交价和最低失败价 (Analyze historical transactions)
#         for record in history:
#             if gequal(Q, record[0]):  # 确保是相同的查询图 (Ensure same query graph)
#                 if record[2] == 1:  # 成功交易 (Successful transaction)
#                     sp_max = max(sp_max, record[1])
#                     success_count += 1
#                 else:  # 失败交易 (Failed transaction)
#                     fp_min = min(fp_min, record[1])
#                     fail_count += 1
#         if success_count > 0 and sp_max == float('-inf'):
#             print("sp_max Error!!!!!!!!!!!")
#             break
#         if fail_count > 0 and fp_min == float('inf'):
#             print("fp_min Error!!!!!!!!!!!")
#             break
#         # if price is None or expected_price < price:
#         if price is None:
#             continue
#         if expected_price < price:
#             success = False  # 交易失败 (Transaction fails if expected price is lower)
#             print(f'交易失败 - Q: {Q}, 价格: {price}, 预期价格: {expected_price}')
#             #transaction_manager.add_transaction(Q, price, success)
            
#             # 价格动态调整策略 (Price adjustment strategy)
#             if success_count > 0:  # 过往存在成功交易数据 (Successful transaction history exists)
#             #无需考虑历史失败价格，因为报价一定是考虑过历史价格的
#                 new_price = (sp_max + price) / 2  
#             else:  # 只有失败交易数据 (Only failure transactions exist)
#                 cost = computecost_G(Q, vertex_price_list) 
#                 if price - cost > xpricing:
#                     new_price = price - xpricing
#                 elif price - cost > (xpricing / 2):
#                     new_price = price - xpricing / 2
#                 elif price - cost > 1:
#                     new_price = price - 1
#                 else:
#                     new_price = cost + 1
#         else:
#             success = True  # 交易成功 (Transaction succeeds if expected price >= price)
#             print(f'交易成功 - Q: {Q}, 价格: {price}, 预期价格: {expected_price}')
#             #transaction_manager.add_transaction(Q, price, success)
            
#             # 价格动态调整策略 (Price adjustment strategy)
#             if fail_count > 0:  # 过往存在失败交易数据 (Failed transaction history exists)
#                 new_price = (fp_min + price) / 2  
            
#             else:  # 只有成功交易数据 (Only success transactions exist)
#                 new_price = price + xpricing
        
#         # 更新顶点价格 (Update vertex price)
#         cost = computecost_G(Q, vertex_price_list)
#         vertex_price_list = adjust_vertex_price(Q, cost, new_price, vertex_price_list)
        
#         transaction_manager.add_transaction(Q, price, success)  # 记录交易 (Record transaction)
        
#         if i in evaluation_intervals:
#             #success_ratio, avg_regret, avg_price_deviation = evaluator.evaluate_transactions(transaction_manager.get_history())
#             success_ratio, avg_regret, avg_price_deviation, avg_sdiff, avg_diff, avg_per_diff, customer_avg_per_diff = evaluator.evaluate_transactions(transaction_manager.get_history())
#             results = {
#                 "success_ratio": success_ratio,
#                 "avg_regret": avg_regret,
#                 "avg_price_deviation": avg_price_deviation,
#                 "avg_sdiff": avg_sdiff,
#                 "avg_diff": avg_diff,
#                 "avg_per_diff": avg_per_diff,
#                 "customer_avg_per_diff": customer_avg_per_diff
#             }
#     tran_his = transaction_manager.get_history()
#     success_ratio, avg_regret, avg_price_deviation, avg_sdiff, avg_diff, avg_per_diff, customer_avg_per_diff = evaluator.evaluate_transactions(transaction_manager.get_history())
#     results = {
#         "success_ratio": success_ratio,
#         "avg_regret": avg_regret,
#         "avg_price_deviation": avg_price_deviation,
#         "avg_sdiff": avg_sdiff,
#         "avg_diff": avg_diff,
#         "avg_per_diff": avg_per_diff,
#         "customer_avg_per_diff": customer_avg_per_diff
#     }
#     print(results)
#     evaluator.save_evaluation_results(i, "TEST_0_evaluation_results.txt", results)

# 示例数据 (Example data)
# graph1 = nx.Graph()
# graph1.add_edges_from([(1, 2), (2, 3), (3, 4)])  # 创建图 1 (Create graph 1)

# graph2 = nx.Graph()
# graph2.add_edges_from([(1, 2), (2, 3)])  # 创建查询图 2 (Create query graph 2)

# vertex_price_list = {1: 10, 2: 20, 3: 15, 4: 25}  # 定义节点价格 (Define vertex prices)
# queries = [graph2] * 150000  # 需要查询的图列表 (List of query graphs)
# subgraphs = [graph1]  # 可用的子图列表 (List of available subgraphs)
# buyer_price_list = {1: 12, 2: 18, 3: 14, 4: 22}  # 定义买家期望价格 (Define buyer expected prices)

# # 运行交易系统 (Run the transaction system)
# transaction_manager = TransactionManager()
# evaluator = TransactionEvaluator(buyer_price_list)
# process_transactions(queries, subgraphs, vertex_price_list, transaction_manager, evaluator)


# # 计算查询图的定价 (Compute the pricing for a query graph)
# def compute_pricing(Q, subgraph_list, vertex_price_list):
#     for subG in subgraph_list:
#         if are_graphs_equal(Q, subG):
#             return sum(vertex_price_list[node] for node in Q.nodes if node in vertex_price_list)  # 计算 Q 中所有节点的价格总和 (Sum up the prices of all nodes in Q)
#     return None  # 无匹配则返回 None (Return None if no match is found)

# def adjust_vertex_price(Q, cost, new_price, vertex_price_list):
#     """
#     调整查询图 Q 中所有顶点的价格 (Adjust the prices of all vertices in query graph Q)
    
#     参数:
#     Q: 交易涉及的查询图 (Query graph involved in the transaction)
#     cost: 当前计算的查询成本 (Current computed cost of the query)
#     new_price: 交易后计算的新价格 (New price calculated after the transaction)
#     vertex_price_list: 存储所有顶点价格的列表 (List storing prices of all vertices)

#     返回:
#     更新后的 vertex_price_list (Updated vertex price list)
#     """
#     # 计算每个顶点的价格调整比例 (Compute adjustment ratio for each vertex)
#     adjustment_ratio = new_price / cost if cost > 0 else 1

#     # 遍历查询图 Q 中的所有顶点 (Iterate through all vertices in Q)
#     for v in Q.vs.indices:
#         # 计算新的顶点价格 (Compute new vertex price)
#         vertex_price_list[v] *= adjustment_ratio

#     return vertex_price_list