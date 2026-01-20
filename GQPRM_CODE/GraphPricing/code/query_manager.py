import networkx as nx
import random
import pickle
import os
from config import Config

class QueryManager:
    def __init__(self, full_graph):
        self.graph = full_graph
        self.target_num = Config.QueryParams.GENERATE_NUM
        self.fixed_nodes = Config.QueryParams.FIXED_NODES
        self.fixed_edges = Config.QueryParams.FIXED_EDGES
        self.queries = []

    def _get_fixed_size_candidate(self, source_graph):
        """
        尝试生成一个严格满足 fixed_nodes 和 fixed_edges 的连通子图
        """
        # 1. 节点不够，无法生成
        if source_graph.number_of_nodes() < self.fixed_nodes:
            return None

        # 2. 随机游走/BFS 获取 N 个连通节点
        start_node = random.choice(list(source_graph.nodes()))
        selected_nodes = {start_node}
        queue = [start_node]
        
        # BFS 扩展
        while len(selected_nodes) < self.fixed_nodes and queue:
            curr = queue.pop(0)
            neighbors = list(source_graph.neighbors(curr))
            random.shuffle(neighbors)
            for nbr in neighbors:
                if nbr not in selected_nodes:
                    selected_nodes.add(nbr)
                    queue.append(nbr)
                    if len(selected_nodes) == self.fixed_nodes:
                        break
        
        if len(selected_nodes) < self.fixed_nodes:
            return None

        # 3. 提取诱导子图 (包含这些节点之间的所有边)
        subgraph = source_graph.subgraph(selected_nodes).copy()
        
        # 4. 调整边数
        current_edges = list(subgraph.edges())
        if len(current_edges) < self.fixed_edges:
            # 边数不足，放弃 (原图中就没这么多边)
            return None
        
        if len(current_edges) == self.fixed_edges:
            return subgraph if nx.is_connected(subgraph.to_undirected()) else None

        # 5. 边数过多，需要随机删边 (Pruning)
        # 必须确保删除后图依然连通 (使用无向连通性检查即可，因为是弱连通)
        random.shuffle(current_edges)
        temp_graph = subgraph.copy()
        
        edges_to_remove = len(current_edges) - self.fixed_edges
        removed_count = 0
        
        for u, v in current_edges:
            temp_graph.remove_edge(u, v)
            # 检查连通性 (将有向图转无向图检查连通分量)
            if nx.is_connected(temp_graph.to_undirected()):
                removed_count += 1
                if removed_count == edges_to_remove:
                    return temp_graph
            else:
                # 删掉会导致断开，加回来
                temp_graph.add_edge(u, v, **subgraph.edges[u, v])
        
        return None

    def generate_queries(self, subgraphs_pool):
        """从预加载的子图池中采样并修剪"""
        cache_file = Config.Paths.QUERY_CACHE_FILE
        # 尝试读取缓存
        if os.path.exists(cache_file):
            print(f"Loading queries from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                # 简单的校验缓存是否符合当前配置
                if len(cached_data) > 0:
                    sample = cached_data[0]
                    if (sample.number_of_nodes() == self.fixed_nodes and 
                        sample.number_of_edges() == self.fixed_edges):
                        self.queries = cached_data
                        print(f"Loaded {len(self.queries)} cached queries.")
                        return

        print(f"Generating queries: Nodes={self.fixed_nodes}, Edges={self.fixed_edges}...")
        attempts = 0
        max_attempts = Config.QueryParams.MAX_ATTEMPTS
        
        while len(self.queries) < self.target_num and attempts < max_attempts:
            attempts += 1
            # 随机选一个大的子图作为源
            source = random.choice(subgraphs_pool)
            
            candidate = self._get_fixed_size_candidate(source)
            if candidate:
                self.queries.append(candidate)
                if len(self.queries) % 100 == 0:
                    print(f"Generated {len(self.queries)}/{self.target_num}...")

        print(f"Generation complete. Total: {len(self.queries)}")
        
        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(self.queries, f)

    def get_queries(self):
        return self.queries
