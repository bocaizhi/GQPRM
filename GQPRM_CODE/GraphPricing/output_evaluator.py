import random
import networkx as nx
import numpy as np
import time
import hashlib

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

class TransactionEvaluator:
    def __init__(self, buyer_price_dict):
        """
        初始化：接收基于图结构签名的买家期望价格字典。
        :param buyer_price_dict: dict[str -> float]，图结构签名 -> 期望价格
        """
        self.buyer_price_dict = buyer_price_dict

    def generate_expected_price(self, Q):
        sig = graph_signature(Q)
        if sig in self.buyer_price_dict:
            return self.buyer_price_dict[sig]

        # ca-AstroPh 数据集的预期价格公式 (number_of_nodes * 5)
        expected_price = Q.number_of_nodes() * 5 + random.uniform(0, 15)
        self.buyer_price_dict[sig] = expected_price
        return expected_price
        
    def evaluate_transactions(self, transaction_history, plist_buyer, filename=None, etime=None):
        """
        分段评估交易历史，增加收益(Revenue)评估。
        """
        # ca-AstroPh 对应的评估区间
        checkpoints = [1000, 5000, 10000, 20000, 30000, 50000, 80000, 120000]
        total_transactions = len(transaction_history)
        if total_transactions not in checkpoints:
            checkpoints.append(total_transactions)
        checkpoints_set = set(checkpoints)

        results_list = []

        success_count = 0
        regret_list = []
        price_deviation_list = []
        diff_list = []
        success_list = []
        
        # 新增：收益相关变量
        total_revenue = 0

        total_regret = 0
        total_price_deviation = 0
        total_diff = 0
        total_per_diff = 0
        total_customer_per_diff = 0

        checkpoint_index = 0

        # history item: (Q, price, success, cost)
        for idx, (Q, price, success, cost) in enumerate(transaction_history, start=1):
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

            # --- 新增：计算收益 ---
            # 如果交易成功，收益 = 成交价 - 成本
            # 如果交易失败，收益 = 0
            current_revenue = (price - cost) if success else 0
            total_revenue += current_revenue
            # --------------------

            if success:
                success_count += 1
                diff = abs(price - expected_price)
                success_list.append([price, expected_price, diff])
                diff_list.append([diff, price, expected_price])
            else:
                diff_list.append([price_deviation, expected_price, 1])

            # 评估点
            if idx in checkpoints_set:
                success_ratio = success_count / idx if idx > 0 else 0
                avg_regret = total_regret / idx if idx > 0 else 0
                avg_price_deviation = total_price_deviation / idx if idx > 0 else 0
                avg_sdiff = sum(d[0] for d in diff_list) / idx if idx > 0 else 0
                
                # 新增：计算平均收益
                avg_revenue = total_revenue / idx if idx > 0 else 0

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
                    "total_revenue": total_revenue,        # 新增指标
                    "avg_revenue": avg_revenue,            # 新增指标
                    "avg_regret": avg_regret,
                    "avg_price_deviation": avg_price_deviation,
                    "avg_sdiff": avg_sdiff,
                    "avg_diff": avg_diff,
                    "avg_per_diff": avg_per_diff,
                    "customer_avg_per_diff": customer_avg_per_diff,
                }

                if etime and checkpoint_index < len(etime):
                    elapsed_time = etime[checkpoint_index]
                    checkpoint_index += 1
                else:
                    elapsed_time = None

                results_list.append((idx, results, elapsed_time))

                if filename:
                    self.save_evaluation_results(idx, filename, results, elapsed_time)

        return results_list

    def save_evaluation_results(self, num, filename, results, elapsed_time=None):
        with open(filename, 'a') as f:
            f.write("\n=== Evaluation at Transaction #{} ===\n".format(num))
            if elapsed_time is not None:
                f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
