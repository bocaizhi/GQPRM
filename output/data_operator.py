import pandas as pd
import re
import os
from openpyxl import load_workbook
from openpyxl.workbook import Workbook

def parse_txt_to_excel(txt_file, excel_file, mode="w"):
    """
    将 txt 文件中的数据解析并写入 Excel 文件。
    mode="w"：覆盖写入（默认）
    mode="a"：追加写入（如果文件不存在则创建）
    """
    # 定义正则表达式来匹配数据块
    pattern = re.compile(
        r"=== Evaluation at Transaction #(\d+) ===\n"
        r"Elapsed Time: (\d+\.\d+) seconds\n"
        r"total transaction: (\d+)\n"
        r"success transaction: (\d+)\n"
        r"success_ratio: (\d+\.\d+)\n"
        r"avg_regret: (\d+\.\d+)\n"
        r"avg_price_deviation: (\d+\.\d+)\n"
        r"avg_sdiff: (\d+\.\d+)\n"
        r"avg_diff: (\d+\.\d+)\n"
        r"avg_per_diff: (\d+\.\d+)\n"
        r"customer_avg_per_diff: (\d+\.\d+)",
        re.MULTILINE
    )
    
    # 查找所有匹配的数据块
    matches = pattern.findall(txt_file)
    
    # 将匹配的数据转换为 DataFrame
    data = []
    for match in matches:
        data.append({
            "Transaction": int(match[0]),
            "Elapsed Time (s)": float(match[1]),
            "Total Transactions": int(match[2]),
            "Success Transactions": int(match[3]),
            "Success Ratio": float(match[4]),
            "Avg Regret": float(match[5]),
            "Avg Price Deviation": float(match[6]),
            "Avg Sdiff": float(match[7]),
            "Avg Diff": float(match[8]),
            "Avg Per Diff": float(match[9]),
            "Customer Avg Per Diff": float(match[10])
        })
    
    df = pd.DataFrame(data)
    
    if mode == "w":
        # 覆盖写入模式（默认）
        df.to_excel(excel_file, index=False)
    elif mode == "a":
        # 追加写入模式
        try:
            if os.path.exists(excel_file):
                # 加载现有 Excel 文件
                book = load_workbook(excel_file)
                
                # 检查是否有工作表（至少保留一个可见的工作表）
                if not any(sheet.sheet_state != "hidden" for sheet in book.worksheets):
                    raise ValueError("Excel 文件中没有可见的工作表，无法追加数据！")
                
                # 读取现有数据
                existing_df = pd.read_excel(excel_file)
                
                # 确保列名一致
                if list(existing_df.columns) != list(df.columns):
                    raise ValueError("Excel 文件中的列名与当前数据不匹配，无法追加！")
                
                # 追加数据
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # 使用 openpyxl 的 ExcelWriter 写入（避免权限问题）
                with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                    writer.book = book
                    combined_df.to_excel(writer, index=False, sheet_name=writer.sheets.keys().__iter__().__next__())
            else:
                # 如果文件不存在，直接写入
                df.to_excel(excel_file, index=False)
        except Exception as e:
            print(f"追加写入失败: {e}")
            # 如果追加失败，就覆盖写入
            df.to_excel(excel_file, index=False)
    else:
        raise ValueError("mode 必须是 'w'（覆盖写入）或 'a'（追加写入）")


# 示例使用
if __name__ == "__main__":
    # 假设 txt 文件内容已经读取到一个字符串中
    with open("DB_x0.txt", "r") as file:
        txt_content = file.read()
    
    # 覆盖写入（默认）
    parse_txt_to_excel(txt_content, "DB_x0.xlsx", mode="w")
    
    # 追加写入（如果需要）
    # parse_txt_to_excel(new_txt_content, "output.xlsx", mode="a")