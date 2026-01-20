import os

class Config:
    # ================= 文件路径配置 =================
    class Paths:
        # 请确保这些文件在同级目录下，或修改为绝对路径
        RDF_FILE = "./data/dbpedia/dbpedia.ttl"
        SUBGRAPH_FILE = "./data/dbpedia/subgraphs.pkl"
        CENTRALITY_FILE = "./data/dbpedia/my_graph_centrality.pkl"
        
        # 结果输出文件
        RESULT_FILE = "dbpedia_results.txt"
        # 查询缓存文件（避免每次重新生成）
        QUERY_CACHE_FILE = "queries_fixed_cache.pkl"

    # ================= 实验规模配置 (核心修改) =================
    class QueryParams:
        GENERATE_NUM = 1000      # 总共生成的查询数量
        
        # [修改点] 固定查询的规模
        FIXED_NODES = 3          # 目标顶点数
        FIXED_EDGES = 5          # 目标边数 (必须满足连通图要求: N-1 <= E <= N*(N-1)/2)
        
        MAX_ATTEMPTS = 50000     # 生成时的最大重试次数

    # ================= 初始参数 (保持原逻辑) =================
    class InitParams:
        # 卖家成本与初始价格区间
        VERTEX_COST_MIN = 1
        VERTEX_COST_MAX = 4
        VERTEX_PRICE_ADD_MIN = 1
        VERTEX_PRICE_ADD_MAX = 6
        
        EDGE_WEIGHT_MIN = 1
        EDGE_WEIGHT_MAX = 3
        EDGE_PRICE_ADD_MIN = 1
        EDGE_PRICE_ADD_MAX = 6

        # 买家预期价格参数: Expected = Nodes * A + Random(0, B)
        BUYER_NODE_MULTIPLIER = 4
        BUYER_RANDOM_FLUCTUATION = 15

    # ================= 算法核心参数 (保持原逻辑) =================
    class AlgoParams:
        X_PRICING = 8.0          # 调价步长
        CENTRALITY_WEIGHT = 0.2  # 熵值法与中心性的权重平衡 (alpha)
        PRICE_DECIMALS = 4       # 价格保留小数位

    # ================= 评估配置 =================
    class Evaluation:
        # 在这些交易次数节点记录评估结果
        CHECKPOINTS = [1000, 5000, 10000, 20000, 50000, 80000, 100000]
        VERBOSE = True  # 是否打印每笔交易详情
