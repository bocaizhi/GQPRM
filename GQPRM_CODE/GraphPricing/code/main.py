import time
from config import Config
from utils import convert_igraph_to_networkx, initialize_vertex_edge_price_list, initialize_buyer_price_dict, graph_signature
from data_loader import GraphDataLoader
from query_manager import QueryManager
from pricing_engine import PricingEngine
from evaluator import TransactionEvaluator

def main():
    print("=== System Starting ===")
    
    # 1. 加载数据
    loader = GraphDataLoader()
    loader.load_rdf_graph()
    loader.load_subgraphs()
    
    G = loader.get_graph()
    raw_subgraphs = loader.get_subgraphs()
    # 转换 igraph -> networkx
    subgraphs_pool = [convert_igraph_to_networkx(g) for g in raw_subgraphs]

    # 2. 生成固定规模查询
    qm = QueryManager(G)
    # 传入子图池进行筛选和剪枝
    qm.generate_queries(subgraphs_pool)
    queries = qm.get_queries()

    if not queries:
        print("Error: No queries generated. Please check constraints.")
        return

    # 3. 初始化价格
    print("Initializing prices...")
    v_prices, e_prices = initialize_vertex_edge_price_list(G)
    buyer_prices = initialize_buyer_price_dict(queries)

    # 4. 初始化引擎
    engine = PricingEngine(v_prices, e_prices)
    evaluator = TransactionEvaluator(buyer_prices)

    # 5. 开始模拟
    print("=== Starting Transactions ===")
    start_time = time.time()
    etime_list = []
    checkpoints = set(Config.Evaluation.CHECKPOINTS)
    
    for i, Q in enumerate(queries, start=1):
        # 5.1 获取买家预期
        sig = graph_signature(Q)
        expected = buyer_prices.get(sig) # 假设初始化时已覆盖

        # 5.2 计算成本 (新增)
        cost = engine.calculate_total_cost(Q)
        
        # 5.3 卖家报价逻辑
        summary = engine.get_summary(Q)
        current_price = engine.calculate_current_price(Q)
        
        if summary:
            # 动态调价
            price = current_price
            success = (expected >= price)
            
            xp = Config.AlgoParams.X_PRICING
            
            if not success: # 失败降价
                if summary['success_count'] > 0:
                    new_price = (summary['max_success_price'] + price) / 2
                else:
                    diff = price - cost
                    if diff > xp: new_price = price - xp
                    elif diff > xp/2: new_price = price - xp/2
                    else: new_price = max(price - 1, cost + 0.1) # 保证不低于成本太多
            else: # 成功涨价
                if summary['fail_count'] > 0 and summary['min_fail_price'] != float('inf'):
                    new_price = (summary['min_fail_price'] + price) / 2
                else:
                    new_price = price + xp
            
            # 执行更新
            if abs(new_price - price) > 1e-4:
                engine.update_prices(Q, new_price)
        else:
            # 冷启动
            price = current_price
            if price == 0: continue
            success = (expected >= price)
            new_price = expected if success else price
            engine.update_prices(Q, new_price)

        # 5.4 记录交易 (带 Cost)
        engine.record_transaction(Q, price, cost, success)

        # 5.5 记录时间
        if i in checkpoints:
            etime_list.append(time.time() - start_time)
            if Config.Evaluation.VERBOSE:
                print(f"#{i}: {'Success' if success else 'Fail'} | Price: {price:.2f} | Cost: {cost:.2f} | Exp: {expected:.2f}")

    # 6. 最终评估
    print("=== Processing Evaluation ===")
    evaluator.evaluate_transactions(engine.transaction_history, buyer_prices, etime_list)
    print("Done.")

if __name__ == '__main__':
    main()
