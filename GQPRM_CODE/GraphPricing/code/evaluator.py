import time
from utils import graph_signature
from config import Config

class TransactionEvaluator:
    def __init__(self, buyer_price_dict):
        self.buyer_price_dict = buyer_price_dict
        self.IP = Config.InitParams

    def generate_expected_price(self, Q):
        # 运行时生成未命中的买家预期
        expected_price = (Q.number_of_nodes() * self.IP.BUYER_NODE_MULTIPLIER + 
                          self.IP.BUYER_RANDOM_FLUCTUATION) # 简化
        return expected_price
        
    def evaluate_transactions(self, transaction_history, plist_buyer, etime_list):
        checkpoints_set = set(Config.Evaluation.CHECKPOINTS)
        if len(transaction_history) not in checkpoints_set:
            checkpoints_set.add(len(transaction_history))

        filename = Config.Paths.RESULT_FILE
        
        # 统计变量
        success_count = 0
        total_regret = 0
        total_price_deviation = 0
        total_revenue = 0 # [新增] 总收益

        checkpoint_idx = 0
        results_list = []

        # 遍历历史记录 (Q, price, cost, success)
        for idx, (Q, price, cost, success) in enumerate(transaction_history, start=1):
            sig = graph_signature(Q)
            expected_price = plist_buyer.get(sig, price) # 兜底

            # 3. 价格偏差 (Price Deviation)
            # 定义：报价与市场真实价值的距离
            price_dev = abs(expected_price - price)
            total_price_deviation += price_dev
            
            # 2. 遗憾值 (Regret)
            # 成功：|成交价 - 预期| (卖便宜了)
            # 失败：预期 (没赚到)
            if success:
                regret = abs(price - expected_price)
            else:
                regret = expected_price
            total_regret += regret

            # 5. 交易收益 (Revenue)
            # 成功：成交价 - 成本
            # 失败：0
            if success:
                revenue = price - cost
                success_count += 1
            else:
                revenue = 0
            total_revenue += revenue

            # 检查点输出
            if idx in checkpoints_set:
                self._process_checkpoint(idx, success_count, total_regret, 
                                         total_price_deviation, total_revenue,
                                         filename, etime_list, checkpoint_idx)
                checkpoint_idx += 1

    def _process_checkpoint(self, idx, success_count, total_regret, total_price_dev, 
                            total_revenue, filename, etime_list, cp_index):
        
        # 计算平均指标
        success_ratio = success_count / idx
        avg_regret = total_regret / idx
        avg_price_dev = total_price_dev / idx
        avg_revenue = total_revenue / idx # 平均单笔交易收益
        
        elapsed = etime_list[cp_index] if cp_index < len(etime_list) else 0

        results = {
            "Transaction Count": idx,
            "1. Success Ratio": f"{success_ratio:.4f}",
            "2. Avg Regret": f"{avg_regret:.4f}",
            "3. Avg Price Deviation": f"{avg_price_dev:.4f}",
            "4. Elapsed Time (s)": f"{elapsed:.2f}",
            "5. Avg Revenue": f"{avg_revenue:.4f}",
            "   Total Revenue": f"{total_revenue:.2f}"
        }

        self.save_results(idx, filename, results)

    def save_results(self, num, filename, results):
        with open(filename, 'a') as f:
            f.write(f"\n=== Evaluation at Transaction #{num} ===\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        print(f"Check point #{num} saved.")
