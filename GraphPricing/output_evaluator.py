import random
import networkx as nx
import numpy as np
import time
import hashlib
# class TransactionEvaluator:
#     def __init__(self, buyer_price_list):
#         self.buyer_price_list = buyer_price_list  # 记录买方期望价格
    
#     def generate_expected_price(self, Q):
#         """
#         计算或获取查询图 Q 的预期价格。

#         :param Q: 查询图 (networkx.Graph)
#         :return: 预期价格
#         """
#         # 查找 Q 是否已在 buyer_price_list 中（基于图同构匹配）
#         for query, price in self.buyer_price_list:
#             if nx.is_isomorphic(query, Q):  # 使用 networkx 的 is_isomorphic() 进行图同构匹配
#                 return price

#         # 如果 Q 不在 buyer_price_list 中，则生成新价格并存入列表
#         expected_price = Q.number_of_nodes() * 5 + random.uniform(0, 15)
#         self.buyer_price_list.append([Q, expected_price])
#         return expected_price

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

class TransactionEvaluator:
    def __init__(self, buyer_price_dict):
        """
        初始化：接收基于图结构签名的买家期望价格字典。
        :param buyer_price_dict: dict[str -> float]，图结构签名 -> 期望价格
        """
        self.buyer_price_dict = buyer_price_dict

    def generate_expected_price(self, Q):
        """
        根据图签名生成或获取查询图 Q 的预期价格。
        :param Q: 查询图 (networkx.Graph)
        :return: 预期价格
        """
        sig = graph_signature(Q)

        if sig in self.buyer_price_dict:
            return self.buyer_price_dict[sig]

        # 如果该签名未出现过，则生成新价格并记录
        expected_price = Q.number_of_nodes() * 4 + random.uniform(0, 15)
        self.buyer_price_dict[sig] = expected_price
        return expected_price
        
    def evaluate_transactions(self, transaction_history, plist_buyer, filename=None, etime=None):
        """
        分段评估交易历史，在指定检查点及最后一次交易时保存评估结果和运行时间。

        参数:
        - transaction_history: list[(Q, price, success)]，交易历史
        - plist_buyer: dict[str -> float]，买方期望价格记录
        - filename: str，结果保存文件名
        - etime: list[float]，每个checkpoint及最终交易的运行时间（长度应与评估点数量一致）
        
        返回:
        - list[(index, metrics_dict, time)]，包含评估点、指标及对应运行时间
        """
        checkpoints = [10000, 30000, 50000, 80000, 120000]
        total_transactions = len(transaction_history)
        if total_transactions not in checkpoints:
            checkpoints.append(total_transactions)
        checkpoints_set = set(checkpoints)
        checkpoints_sorted = sorted(checkpoints)

        results_list = []

        success_count = 0
        regret_list = []
        price_deviation_list = []
        diff_list = []
        success_list = []

        total_regret = 0
        total_price_deviation = 0
        total_diff = 0
        total_per_diff = 0
        total_customer_per_diff = 0

        checkpoint_index = 0  # 用于匹配 etime 中的时间戳

        for idx, (Q, price, success) in enumerate(transaction_history, start=1):
            sig = graph_signature(Q)
            expected_price = plist_buyer.get(sig)
            if expected_price is None:
                expected_price = self.generate_expected_price(Q)
                plist_buyer[sig] = expected_price

            price_deviation = abs(expected_price - price)
            total_price_deviation += price_deviation
            price_deviation_list.append(price_deviation)

            regret = abs(price - expected_price) if success else expected_price
            total_regret += regret
            regret_list.append(regret)

            if success:
                success_count += 1
                diff = abs(price - expected_price)
                success_list.append([price, expected_price, diff])
                diff_list.append([diff, price, expected_price])
            else:
                diff_list.append([price_deviation, expected_price, 1])

            # 评估点或结束位置
            if idx in checkpoints_set:
                success_ratio = success_count / idx if idx > 0 else 0
                avg_regret = total_regret / idx if idx > 0 else 0
                avg_price_deviation = total_price_deviation / idx if idx > 0 else 0
                avg_sdiff = sum(d[0] for d in diff_list) / idx if idx > 0 else 0

                if success_count > 0:
                    total_diff = sum(l[2] for l in success_list)
                    total_per_diff = sum(l[2] / l[0] for l in success_list)
                    total_customer_per_diff = sum(l[2] / l[1] for l in success_list)

                    avg_diff = total_diff / success_count
                    avg_per_diff = total_per_diff / success_count
                    customer_avg_per_diff = total_customer_per_diff / success_count
                else:
                    avg_diff = avg_per_diff = customer_avg_per_diff = 0

                results = {
                    "total transaction": idx,
                    "success transaction": success_count,
                    "success_ratio": success_ratio,
                    "avg_regret": avg_regret,
                    "avg_price_deviation": avg_price_deviation,
                    "avg_sdiff": avg_sdiff,
                    "avg_diff": avg_diff,
                    "avg_per_diff": avg_per_diff,
                    "customer_avg_per_diff": customer_avg_per_diff,
                }

                # 加入运行时间
                if etime and checkpoint_index < len(etime):
                    elapsed_time = etime[checkpoint_index]
                    checkpoint_index += 1
                else:
                    elapsed_time = None  # 若无时间记录，设为 None

                results_list.append((idx, results, elapsed_time))

                # 保存结果
                if filename:
                    self.save_evaluation_results(idx, filename, results, elapsed_time)

        return results_list

    def save_evaluation_results(self, num, filename, results, elapsed_time=None):
        """
        保存评估结果到文件，包括评估时间（如有）

        参数:
        - num: 当前交易数
        - filename: 输出文件路径
        - results: 指标字典
        - elapsed_time: 当前时间戳（秒）
        """
        with open(filename, 'a') as f:
            f.write("\n=== Evaluation at Transaction #{} ===\n".format(num))
            if elapsed_time is not None:
                f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")


    # def evaluate_transactions(self, transaction_history,plist_buyer):
    #     """
    #     评估交易历史的各项指标

    #     参数:
    #     - transaction_history: 交易历史，格式为 [(Q, price, success), ...]

    #     返回:
    #     - success_ratio: 交易成功率
    #     - avg_regret: 平均后悔值
    #     - avg_price_deviation: 平均价格偏差
    #     - avg_sdiff: 价格偏差均值
    #     - avg_diff: 成功交易价格偏差均值
    #     - avg_per_diff: 成功交易价格偏差百分比
    #     - customer_avg_per_diff: 买方心理预期偏差均值
    #     """
    #     success_count = 0
    #     total_transactions = len(transaction_history)
    #     regret_list = []
    #     price_deviation_list = []
    #     diff_list = []
    #     success_list = []
        
    #     total_regret = 0
    #     total_price_deviation = 0
    #     total_price_deviation = np.clip(total_price_deviation, -1e10, 1e10)
    #     total_diff = 0
    #     total_per_diff = 0
    #     total_customer_per_diff = 0

    #     for Q, price, success in transaction_history:
    #         # expected_price = self.generate_expected_price(Q)
    #                 # 更新或添加预期价格记录
    #         found = False
    #         sig = graph_signature(Q)
    #         if sig in plist_buyer:
    #             expected_price = plist_buyer[sig]
    #             found = True
    #         else:
    #             expected_price = self.generate_expected_price(Q)
    #             plist_buyer[sig] = expected_price  # 新增
    #             found = False
    #         # 计算价格偏差
    #         price_deviation = abs(expected_price - price)
    #         #print(price,expected_price,price_deviation)
    #         total_price_deviation += price_deviation
    #         price_deviation_list.append(price_deviation)
            
    #         # 计算后悔值（成功时为交易差值，失败时为期望价格）
    #         regret = abs(price - expected_price) if success else expected_price
    #         total_regret += regret
    #         regret_list.append(regret)

    #         # 记录成功交易信息
    #         if success:
    #             success_count += 1
    #             diff = abs(price - expected_price)
    #             # print(price, expected_price, diff)
    #             success_list.append([price, expected_price, diff])
    #             diff_list.append([diff, price, expected_price])
    #         else:
    #             diff_list.append([price_deviation, expected_price, 1])  # 失败交易
    #     #print(price_deviation_list)
    #     #print(total_price_deviation)
    #     # 计算交易成功率
    #     success_ratio = success_count / total_transactions if total_transactions > 0 else 0
    #     #平均regret
    #     avg_regret = total_regret / total_transactions if total_transactions > 0 else 0
        
    #     avg_price_deviation = total_price_deviation / total_transactions if total_transactions > 0 else 0

    #     # 计算 avg_sdiff（所有交易的价格偏差均值）
    #     avg_sdiff = sum(d[0] for d in diff_list) / total_transactions if total_transactions > 0 else 0

    #     # 计算 avg_per_diff（成功交易的价格偏差百分比）
    #     if success_count > 0:
    #         total_diff = sum(l[2] for l in success_list)
    #         total_per_diff = sum(l[2] / l[0] for l in success_list)
    #         total_customer_per_diff = sum(l[2] / l[1] for l in success_list)

    #         avg_diff = total_diff / success_count
    #         avg_per_diff = total_per_diff / success_count
    #         customer_avg_per_diff = total_customer_per_diff / success_count
    #     else:
    #         avg_diff = avg_per_diff = customer_avg_per_diff = 0

    #     # 输出评估结果
    #     print("Number of transactions:", total_transactions)
    #     print("Number of successful transactions:", success_count)
    #     print("Success ratio:", success_ratio)
    #     print("Total regret:", total_regret)
    #     print("Total avg_sdiff:", avg_sdiff)
    #     print("Total avg_per_diff:", avg_per_diff)
    #     print("Total customer expected avg_per_diff:", customer_avg_per_diff)
    #     print("Success avg_diff:", avg_diff)
    #     print("Success avg_per_diff:", avg_per_diff)
    #     print("Success customer expected avg_per_diff:", customer_avg_per_diff)

    #     # 返回评估指标
    #     return success_ratio, avg_regret, avg_price_deviation, avg_sdiff, avg_diff, avg_per_diff, customer_avg_per_diff

    # def save_evaluation_results(self, num, filename, results):
    #     """
    #     将评估结果保存到 TXT 文件
        
    #     :param num: 交易总数
    #     :param filename: 输出文件名 (str)
    #     :param results: 评估结果字典 (dict)
    #     """
    #     with open(filename, 'a') as f:
    #         f.write("\n=== New Evaluation ===\n")  # 添加分隔符，方便查看新记录
    #         f.write(f"Total transactions: {num}\n")
    #         for key, value in results.items():
    #             f.write(f"{key}: {value}\n")  # 逐行写入 key-value


# import igraph
# import random

# # 交易评估类 (Transaction evaluation class)
# class TransactionEvaluator:
#     def __init__(self, buyer_price_list):
#         self.buyer_price_list = buyer_price_list  # 记录买方期望价格 (Store buyer expected prices)
    
#     def generate_expected_price(self, Q):
#         """
#         计算或获取查询图 Q 的预期价格。

#         :param Q: 查询图
#         :return: 预期价格
#         """
#         # 查找 Q 是否已在 buyer_price_list 中（基于图同构匹配）
#         for query, price in self.buyer_price_list:
#             if query.isomorphic(Q):  # 使用 isomorphic() 检查图是否同构
#                 return price

#         # 如果 Q 不在 buyer_price_list 中，则生成新价格并存入列表
#         # expected_price = Q.vcount() * 5 + random.uniform(0, 10)
#         # self.buyer_price_list.append([Q, expected_price])
#         # 如果 Q 不在 buyer_price_list 中，则生成新价格并存入列表
#         expected_price = Q.vcount() * 5 + random.uniform(0, 10)
#         self.buyer_price_list.append([Q, expected_price])
#         return expected_price

#     # def generate_expected_price(self, Q):
#     #     if Q not in self.buyer_price_list:
#     #         self.buyer_price_list[Q] = len(Q.vs) * 5 + random.uniform(0, 10)
#     #     return self.buyer_price_list[Q]
    
#     def evaluate_transactions(self, transaction_history):
#         success_count = 0
#         regret_list = []
#         price_deviation_list = []
#         total_transactions = len(transaction_history)
#         total_regret = 0
#         total_price_deviation = 0
        
#         for i, transaction in enumerate(transaction_history):
#             Q, price, success = transaction
#             expected_price = self.generate_expected_price(Q)
#             regret = abs(price - expected_price) if success else expected_price
#             price_deviation = abs(expected_price - price)
            
#             total_regret += regret
#             total_price_deviation += price_deviation
#             regret_list.append(regret)
#             price_deviation_list.append(price_deviation)
            
#             if success:
#                 success_count += 1
        
#         success_ratio = success_count / total_transactions if total_transactions > 0 else 0
#         avg_regret = total_regret / total_transactions if total_transactions > 0 else 0
#         avg_price_deviation = total_price_deviation / total_transactions if total_transactions > 0 else 0
        
#         return success_ratio, avg_regret, avg_price_deviation

#     def evaluate_transactions(self, transaction_history):
#         """
#         评估交易历史的各项指标 (Evaluate various metrics of transaction history)

#         参数:
#         - transaction_history: 交易历史，格式为 [(Q, price, success), ...]

#         返回:
#         - success_ratio: 交易成功率
#         - avg_regret: 平均后悔值
#         - avg_price_deviation: 平均价格偏差
#         """
#         success_count = 0
#         total_transactions = len(transaction_history)
        
#         regret_list = []
#         price_deviation_list = []
#         diff_list = []
#         success_list = []
        
#         total_regret = 0
#         total_price_deviation = 0
#         total_diff = 0
#         total_per_diff = 0
#         total_customer_per_diff = 0

#         for i, transaction in enumerate(transaction_history):
#             Q, price, success = transaction
#             expected_price = self.generate_expected_price(Q)

#             # 计算价格偏差
#             price_deviation = abs(expected_price - price)
#             total_price_deviation += price_deviation
#             price_deviation_list.append(price_deviation)

#             # 计算后悔值（成功时为交易差值，失败时为期望价格）
#             regret = abs(price - expected_price) if success else expected_price
#             total_regret += regret
#             regret_list.append(regret)

#             # 记录成功交易信息
#             if success:
#                 success_count += 1
#                 diff = abs(price - expected_price)
#                 success_list.append([price, expected_price, diff])
#                 diff_list.append([diff, price, expected_price])
#             else:
#                 diff_list.append([price_deviation, expected_price, 1])  # 失败交易

#         # 计算交易成功率
#         success_ratio = success_count / total_transactions if total_transactions > 0 else 0
#         avg_regret = total_regret / total_transactions if total_transactions > 0 else 0
#         avg_price_deviation = total_price_deviation / total_transactions if total_transactions > 0 else 0

#         # 计算 avg_sdiff（所有交易的价格偏差均值）
#         if total_transactions > 0:
#             avg_sdiff = sum(d[0] for d in diff_list) / total_transactions
#         else:
#             avg_sdiff = 0

#         # 计算 avg_per_diff（成功交易的价格偏差百分比）
#         if success_count > 0:
#             total_diff = sum(l[2] for l in success_list)
#             total_per_diff = sum(l[2] / l[0] for l in success_list)
#             total_customer_per_diff = sum(l[2] / l[1] for l in success_list)

#             avg_diff = total_diff / success_count
#             avg_per_diff = total_per_diff / success_count
#             customer_avg_per_diff = total_customer_per_diff / success_count
#         else:
#             avg_diff = avg_per_diff = customer_avg_per_diff = 0

#         # 输出评估结果
#         print("number of transactions:", total_transactions)
#         print("number of successful transactions:", success_count)
#         print("success ratio:", success_ratio)
#         print("sum regret:", total_regret)
#         print("total avg_sdiff:", avg_sdiff)
#         print("total avg_per_diff:", avg_per_diff)
#         print("total customer expected avg_per_diff:", customer_avg_per_diff)
#         print("success avg_diff:", avg_diff)
#         print("success avg_per_diff:", avg_per_diff)
#         print("success customer expected avg_per_diff:", customer_avg_per_diff)

#         # 返回评估指标
#         return success_ratio, avg_regret, avg_price_deviation, avg_sdiff, avg_diff, avg_per_diff, customer_avg_per_diff


#     def save_evaluation_results(self, num, filename, results):
#         """
#         将评估结果保存到 TXT 文件 (Save evaluation results to a TXT file)
        
#         参数:
#         - filename: 输出文件名 (str)
#         - results: 评估结果字典 (dict)
#         """
#         with open(filename, 'a') as f:
#             f.write("\n=== New Evaluation ===\n")  # 添加分隔符，方便查看新记录
#             f.write(f"total transation: {num}\n")
#             for key, value in results.items():
#                 f.write(f"{key}: {value}\n")  # 逐行写入 key-value

    # def save_evaluation_results(self, filename, results):
    #     with open(filename, 'w') as f:
    #         json.dump(results, f, indent=4)