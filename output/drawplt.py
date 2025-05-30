import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.weight":'bold'
    #"axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)

'''
def curve(precision, recall, f1, folder_path):
    x = list(range(len(acc)))
    # plt.plot(x, acc, label='Accuracy', color='blue', linestyle='-', marker='o', linewidth=2, markersize=6)
    plt.plot(x, precision, label='Precision', color='green', linestyle='--', marker='s', linewidth=2, markersize=6)
    plt.plot(x, recall, label='Recall', color='orange', linestyle='-.', marker='^', linewidth=2, markersize=6)
    plt.plot(x, f1, label='F1-Score', color='red', linestyle=':', marker='d', linewidth=2, markersize=6)

    plt.xlabel('Time Slice', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')  # 图例位置设置为左上角

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()  # 自动调整布局
    plt.savefig(f'{folder_path}/academic_plot.png', dpi=300)  # 保存为高分辨率图片
    plt.show()  # 显示图形
'''

# 读取Excel文件
#file_path = "output_txt_afver_regret.xlsx"  # 替换为你的Excel文件路径
file_path = "data_FB.xlsx"  # 替换为你的Excel文件路径
sheet_name = "time"       # 替换为你的工作表名称

# 使用pandas读取Excel文件
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 打印数据框以查看内容
print(df.head())

# 假设第一列是X轴，其他列是不同系列的Y轴数据
x = df.iloc[:, 0]  # 第一列作为X轴

# 初始化图表
# plt.figure(figsize=(10, 6))
plt.figure(figsize=(6, 5))

# 绘制每一列的折线图
for column in df.columns[1:]:
    plt.plot(x, df[column], label=column, marker='s', linewidth=2, markersize=6)

# 添加图例
# plt.legend()

plt.xlabel('Number of Transactions($\\times 10^3$)', fontsize=20,weight='bold')
# plt.legend()  # 图例位置设置为左上角
plt.legend(bbox_to_anchor=(0, 1.3),loc=2,ncol=3, frameon=False, fontsize=15) # , borderaxespad=0

# 添加轴标签和标题

# 交易成功率
#plt.ylabel('Success rate(%)', fontsize=20,weight='bold')
#平均遗憾
#平均价格差异
#plt.ylabel('Price', fontsize=20,weight='bold')
#平均运行时间
plt.ylabel('Time(ms)', fontsize=20,weight='bold')

#设置坐标轴刻度
my_x_ticks = np.arange(0, 130, 10)
plt.tick_params(labelsize=12)
plt.xticks(my_x_ticks)
# plt.title('Transaction success rate')
plt.tight_layout()  # 自动调整布局

'''
#平均遗憾
plt.ylabel('Price')
plt.title('Average regret', fontsize=12)

#平均价格差异
plt.ylabel('price')
plt.title('Average price difference(success)', fontsize=12)


#总平均差距比
plt.ylabel('percentage(%)')
plt.title('Average percentage of price difference(success)', fontsize=12)

'''
# 显示网格
#plt.grid(True)
# plt.figure(dpi=300) 
plt.savefig(f'figure\\time_FB_1.png', dpi=300)  # 保存为高分辨率图片
# 显示图表
#plt.savefig()
# plt.show()