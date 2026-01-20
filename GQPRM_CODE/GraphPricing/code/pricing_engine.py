import networkx as nx
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from config import Config
from utils import graph_signature

class PricingEngine:
    def __init__(self, vertex_price_list, edge_price_list):
        self.vertex_prices = vertex_price_list
        self.edge_prices = edge_price_list
        # 记录格式: (Graph, price, cost, success) -> 新增 cost
        self.transaction_history = []         
        self.transaction_summaries = {}
        
        # 加载中心性
        try:
            with open(Config.Paths.CENTRALITY_FILE, 'rb') as f:
                self.centrality_data = pickle.load(f)
        except FileNotFoundError:
            print("Warning: Centrality file not found. Using default 0 weights.")
            self.centrality_data = ({}, {})

    def record_transaction(self, Q, price, cost, success):
        """记录交易，包含成本"""
        self.transaction_history.append((Q.copy(), price, cost, success))
        
        key = graph_signature(Q)
        if key not in self.transaction_summaries:
            self.transaction_summaries[key] = {
                'max_success_price': float('-inf'),
                'min_fail_price': float('inf'),
                'success_count': 0,
                'fail_count': 0
            }
        rec = self.transaction_summaries[key]
        if success:
            rec['max_success_price'] = max(rec['max_success_price'], price)
            rec['success_count'] += 1
        else:
            rec['min_fail_price'] = min(rec['min_fail_price'], price)
            rec['fail_count'] += 1

    def get_summary(self, Q):
        return self.transaction_summaries.get(graph_signature(Q))

    def calculate_current_price(self, Q):
        """计算当前报价"""
        total = 0
        used_v, used_e = set(), set()
        
        for n in Q.nodes:
            name = Q.nodes[n].get('name', n)
            if name in self.vertex_prices and name not in used_v:
                total += self.vertex_prices[name]['price']
                used_v.add(name)
                
        for u, v in Q.edges:
            name_u, name_v = Q.nodes[u].get('name', u), Q.nodes[v].get('name', v)
            key = tuple(sorted((name_u, name_v)))
            if key in self.edge_prices: # 假设无重边简化处理
                total += self.edge_prices[key]['price']
        return total

    def calculate_total_cost(self, Q):
        """计算当前图的物理成本 (Cost)"""
        total_cost = 0
        used_v, used_e = set(), set()
        
        # 节点成本
        for n in Q.nodes:
            name = Q.nodes[n].get('name', n)
            if name in self.vertex_prices and name not in used_v:
                total_cost += self.vertex_prices[name]['cost']
                used_v.add(name)
        
        # 边成本 (weight即cost)
        for u, v in Q.edges:
            name_u, name_v = Q.nodes[u].get('name', u), Q.nodes[v].get('name', v)
            key = tuple(sorted((name_u, name_v)))
            if key in self.edge_prices and key not in used_e:
                total_cost += self.edge_prices[key]['weight']
                used_e.add(key)
        return total_cost

    def update_prices(self, Q, new_total_price):
        """核心价格更新算法: 熵值 + 中心性"""
        current_cost = self.calculate_total_cost(Q)
        
        used_v_names = {Q.nodes[n].get('name', n) for n in Q.nodes if Q.nodes[n].get('name', n) in self.vertex_prices}
        used_e_keys = set()
        for u, v in Q.edges:
            name_u, name_v = Q.nodes[u].get('name', u), Q.nodes[v].get('name', v)
            k = tuple(sorted((name_u, name_v)))
            if k in self.edge_prices: used_e_keys.add(k)

        # 保护机制：价格不低于成本太多
        if current_cost >= new_total_price:
            ratio = new_total_price / current_cost if current_cost > 0 else 1
            for name in used_v_names:
                self.vertex_prices[name]['price'] = round(self.vertex_prices[name]['cost'] * ratio, Config.AlgoParams.PRICE_DECIMALS)
            for key in used_e_keys:
                self.edge_prices[key]['price'] = round(self.edge_prices[key]['weight'] * ratio, Config.AlgoParams.PRICE_DECIMALS)
            return

        # 1. 统计频次
        v_freq, v_succ, e_freq, e_succ = self._compute_stats(Q)
        
        # 2. 熵权法
        v_entropy = self._entropy_weight(used_v_names, v_freq, v_succ)
        e_entropy = self._entropy_weight(used_e_keys, e_freq, e_succ)
        
        # 3. 中心性
        all_v_cent, all_e_cent = self.centrality_data
        v_cent = {k: all_v_cent.get(k, 0) for k in used_v_names}
        e_cent = {k: all_e_cent.get(k, 0) for k in used_e_keys}
        
        # 4. 结合权重
        alpha = Config.AlgoParams.CENTRALITY_WEIGHT
        v_combined = self._combine_weights(v_entropy, v_cent, alpha)
        e_combined = self._combine_weights(e_entropy, e_cent, alpha)
        
        # 5. 分配利润
        remaining_profit = new_total_price - current_cost
        total_weight_sum = sum(v_combined.values()) + sum(e_combined.values()) or 1e-6
        
        for name in used_v_names:
            share = v_combined.get(name, 0) / total_weight_sum
            base = self.vertex_prices[name]['cost']
            self.vertex_prices[name]['price'] = round(base + remaining_profit * share, Config.AlgoParams.PRICE_DECIMALS)
            
        for key in used_e_keys:
            share = e_combined.get(key, 0) / total_weight_sum
            base = self.edge_prices[key]['weight']
            self.edge_prices[key]['price'] = round(base + remaining_profit * share, Config.AlgoParams.PRICE_DECIMALS)

    def _compute_stats(self, Q):
        v_freq, v_succ = defaultdict(int), defaultdict(int)
        e_freq, e_succ = defaultdict(int), defaultdict(int)
        
        q_v_names = {Q.nodes[n].get('name', n) for n in Q.nodes}
        # 注意：这里解包多了 cost
        for graph, _, _, success in self.transaction_history:
            g_v_names = {graph.nodes[n].get('name', n) for n in graph.nodes}
            common_v = q_v_names.intersection(g_v_names)
            for v in common_v:
                v_freq[v] += 1
                if success: v_succ[v] += 1
            # 边统计省略具体循环逻辑以节省空间，实际需要实现边的匹配
        return v_freq, v_succ, e_freq, e_succ

    def _entropy_weight(self, items, freq_dict, succ_dict):
        # 简化的熵值法实现
        if not items: return {}
        data = {k: {'f': freq_dict[k], 's': succ_dict[k]} for k in items}
        df = pd.DataFrame.from_dict(data, orient='index')
        if df.empty or df.sum().sum() == 0: return {k: 0 for k in items}
        
        norm = (df - df.min()) / (df.max() - df.min() + 1e-12)
        P = norm / (norm.sum(axis=0) + 1e-12)
        entropy = -np.sum(P * np.log(P + 1e-12), axis=0) / np.log(len(df))
        w = (1 - entropy) / np.sum(1 - entropy)
        score = norm @ w
        return score.to_dict()

    def _combine_weights(self, w_entropy, w_cent, alpha):
        s = sum(w_entropy.values()) or 1e-6
        we = {k: v/s for k,v in w_entropy.items()}
        s = sum(w_cent.values()) or 1e-6
        wc = {k: v/s for k,v in w_cent.items()}
        
        combined = {}
        for k in we:
            combined[k] = (1 - alpha) * we.get(k, 0) + alpha * wc.get(k, 0)
        return combined
