import igraph as ig
import rdflib
import random
import math
import pickle
import time
import pandas as pd

# 创建一个列表,用于存放数据集中各子图的价格
v_plist = []
# 创建一个空的图列表,用于存放是query子图的图（符合条件的子图）
q_subgraph = []
#交易历史记录，[查询，是否成功（0/1），成功交易价格（不成功时为0）]
tran_history = []
#数据子图集
subgraphs = []
#测试查询集
Q_clu_list = []
#顾客心理预期价格列表
plist_buyer = []

Q_num = 50000
# 指定文件名
file_name = "11_50_x0_facebook.txt"
ti = 1

def add_unique_vertex(label,G):
    f1 = False 
    for name in G.vs['name']:
        if label == name:
            f1 = True
    if f1 == False:
        G.add_vertex(label)
        #print('add successfully')

# 随机选择一个边
def choose_random_edge(graph):
    return random.choice(graph.es)

#判断图g1是否为g2严格意义的子图
#若子图具有自环则直接判定为否
#无法应对含有自环的情况
def total_subisomorphic(g1,g2):
    has_loops = g1.is_loop()
    if True in has_loops:
        return False
    else:
        for sub_name in g1.vs['name']:
            if sub_name not in g2.vs['name']:
                return False
        #从结构上判断g1是否是g2的子图
        is_subgraph = g2.subisomorphic_vf2(g1,return_mapping_21=True)
        #print(is_subgraph)
        #若结构子图成立
        if is_subgraph[0]==True:
            fl=True
            #print('严格子图匹配中，正在匹配名称')
            edge_list = []
            for edge in g2.es:
                edge_list.append([g2.vs[edge.source]['name'],g2.vs[edge.target]['name']])
            for edge in g1.es:
                if [g1.vs[edge.source]['name'],g1.vs[edge.target]['name']] not in edge_list:
                    return False
            return True
        else:
            return False

#判断两图是否完全相同
def gequal(g1,g2):
    flag1 = total_subisomorphic(g1,g2)
    flag2 = total_subisomorphic(g2,g1)
    #print('1:',flag1,'2:',flag2)
    if flag1 == True and flag2 == True:
        return True
    else:
        return False

#第二部分 根据平均价值筛选最终符合条件的子图并计算query最终价格
#计算平均有效价格的函数,并返回平均有效价格最大的index
def MCP(pl,gl):
    maxmcp = 0
    maxindex = -1
    for i,subg in enumerate(gl):
        if subg.ecount()!=0:
            mcp = subg.ecount()/pl[i]
            #print(mcp)
            if maxmcp < mcp:
                maxmcp = mcp
                maxindex = i
    if maxmcp > 0:
        return maxindex
    else:
        if maxmcp == 0:
            return -1
        else:
            print('Error')
            return -2

def remove_edges(G,g):
    edges = []
    for E in G.es:
        for e in g.es:
            if G.vs[E.source]['name']==g.vs[e.source]['name'] and G.vs[E.target]['name']== g.vs[e.target]['name']:
                #print(G.vs[E.source]['name'],G.vs[E.target]['name'],E['label'])
                edges.append(e)
    g.delete_edges(edges)

def ges_eq(elist,G):
    g_num = G.ecount()
    if len(elist)!= g_num:
        return False
    else:
        f_list = [1]*g_num
        i=0
        for edge in G.es:
            sname = G.vs[edge.source]['name']
            tname = G.vs[edge.target]['name']
            for e in elist:
                if e[0] == sname and e[1] == tname:
                    f_list[i]=0
            if f_list[i] == 1:
                return False
            i += 1
    return True

def computeprice_G(G,v_plist):
    p = 0
    for v in v_plist:
        if v[0] in G.vs['name']:
            p += v[2]
            #print(v[0],v[2])
    return p

def computecost_G(G,v_plist):
    c = 0
    for v in v_plist:
        if v[0] in G.vs['name']:
            c += v[1]
            #print(v[0],v[1])
    return c

def write_subisomorphic_flag(Q,subgraphs):
    sub_flag = []
    i = 0
    for subgraph in subgraphs:
        flag1 = total_subisomorphic(subgraph,Q)
        #print(i,'f1 over')
        flag2 = total_subisomorphic(Q,subgraph)
        #print(i,'f2 over')
        sub_flag.append([flag1,flag2])
        i += 1
    return sub_flag

def combine_graph(G_list):
    G=ig.Graph(directed=True)
    combined_vertices = []
    combined_edges = []
    for g in G_list:
        combined_vertices = list(set(combined_vertices).union(set(g.vs["name"])))
        elist = []
        for edge in g.es:
            s_name = g.vs[edge.source]['name']
            t_name = g.vs[edge.target]['name']
            elist.append([s_name,t_name])
        # 将内部列表转换为元组以便于集合操作
        tuple_list1 = [tuple(item) for item in combined_edges]
        tuple_list2 = [tuple(item) for item in elist]
        # 取并集
        union_set = set(tuple_list1).union(set(tuple_list2))
        # 将结果转换回列表形式
        combined_edges = [list(item) for item in union_set]
    G.add_vertices(combined_vertices)
    for e in combined_edges:
        G.add_edge(e[0], e[1])
    return G

#无历史交易记录，进行子图同构定价
def subisomorphic_pricing(Q,subgraphs,v_plist):
    #已筛选边列表
    elist = []
    #答案列表
    anslist = []
    ansplist = []
    p=0
    #备选图列表备用完整版
    q_sub = []
    #记录Q与数据集图的子图关系
    sub_flag = write_subisomorphic_flag(Q,subgraphs)
    #print('sub_flag write successfully.',len(sub_flag))
    #筛选Q的候选子图，并将符合条件的图及其价格放入列表
    q_subgraph = []
    for i,flag in enumerate(sub_flag):
        if flag[1] == True:#有精确匹配图
            p = computeprice_G(Q,v_plist)
            anslist.append(subgraphs[i])
            return [p,anslist]
        else:
            if flag[0] == True:
                q_subgraph.append(subgraphs[i])
    #若无匹配图
    if len(q_subgraph) == 0:
        #进行模糊定价
        print('no answer')
        #apprp_index = appr_pricing(Q,subgraphs,plist,sub_flag)
        #if apprp_index == -1:
        return [0,anslist]
        #else:
            #apprprice = plist[apprp_index]
            #anslist.append(subgraphs[apprp_index])
            #return [apprprice,anslist]
    #有模糊匹配图
    #将q_subgraph所有图合并为一个图
    G_combined = ig.Graph(directed=True)
    G_combined = combine_graph(q_subgraph)
    #判断合并图是否包含Q
    flag1 = total_subisomorphic(Q,G_combined)
    #若包含则精确匹配
    if flag1 ==True:
        p = computeprice_G(Q,v_plist)
        return [p,q_subgraph]
    #Q若不包含则模糊匹配
    else:
        print('no answer')
        return [0,anslist]

def adjust_vertex_price(G,c,p,v_plist):
    for v in v_plist:
        if v[0] in G.vs['name']:
            p0 = v[2]
            v[2] = v[1]*p/c
            print('before:',p0,',after:',v[2])
    return v_plist

def find_G_in_list(plist_buyer, tra, diff_list):
    for p in plist_buyer:
        if gequal(tra[0],p[0]):
            diff = abs(tra[2]-p[1])
            diff_list.append([diff,tra[2],p[1]])
            #返回这条交易历史对应图的顾客预期价格
            return p[1]
    print(f"未找到目标值")
    return -1



facebook = pd.read_csv(
    r"E:\\gyz\\code\\gpricing\\data\\facebook_combined.txt.gz",
    sep=" ",
    names=["start_node", "end_node"],
    header=None,
)
print(type(facebook))

# 创建图对象G
G = ig.Graph(directed=True)
# 获取所有唯一的用户，作为节点
nodes = set(facebook['start_node']).union(set(facebook['end_node']))
nodes_id_to_index = {node: idx for idx, node in enumerate(nodes)}
# 添加边（基于 DataFrame 中的每一对用户）
edges = [('node'+str(nodes_id_to_index[row['start_node']]), 'node'+str(nodes_id_to_index[row['end_node']])) for _, row in facebook.iterrows()]
for edge in edges:
    if edge[0] != edge[1]:
        if G.vcount() == 0:
            G.add_vertex(edge[0])
            G.add_vertex(edge[1])
            G.add_edge(edge[0], edge[1])
        else:
            add_unique_vertex(edge[0],G)
            add_unique_vertex(edge[1],G)
            G.add_edge(edge[0], edge[1])


with open('subgraphs_facebook_ver4_cutto200_ename.pkl', 'rb') as f:
    subgraphs = pickle.load(f)
    print(len(subgraphs))

#测试query集生成，每个query点数<=5
Q_clu_list = []
max_num = 5
# 使用 clusters 函数找到强连通分量（strongly connected components）
#clusters = G.connected_components(mode='weak')
# 如果需要生成每个子图，可以使用 subgraph 函数
#subgraphs_cluster = [G.subgraph(cluster) for cluster in clusters]
G_num = len(subgraphs)
print(G_num)
for i in range(Q_num):
    edge_list = []
    Q = ig.Graph(directed=True)
    #query对应的连通图序号
    g_no = random.randint(0, G_num-1)
    #对应连通图的总边数
    g_no_num = subgraphs[g_no].ecount()
    #print('no.',g_no,'num=',g_no_num)
    #随机生成query的边数
    if g_no_num > 1 and g_no_num < max_num :
        edge_num = random.randint(1, g_no_num)
    else:
        if g_no_num >= max_num:
            edge_num = random.randint(1, max_num)
        else:
            edge_num = 1
    #print('no.',g_no,'num=',g_no_num,'edge_num=',edge_num)
    #生成不重复的边
    if g_no_num-edge_num-1 <= 0:
        num = 0
    else:
        num = random.randint(0, g_no_num-edge_num-1)
    for j in range(edge_num):
        #num = random.randint(0, g_no_num-1)
        #若重复则重新生成随机数
        #while num  in edge_list:
            #num = random.randint(0, g_no_num-1)
        edge = subgraphs[g_no].es[num]
        sname = subgraphs[g_no].vs[edge.source]['name']
        tname = subgraphs[g_no].vs[edge.target]['name']
        if sname != tname:
            if Q.vcount() == 0:
                Q.add_vertex(sname)
                Q.add_vertex(tname)
                Q.add_edge(sname, tname)
            else:
                add_unique_vertex(sname,Q)
                add_unique_vertex(tname,Q)
                Q.add_edge(sname, tname)
        #print(Q.ecount())
        #edge_list.append(num)
        num += 1
    clusters = Q.connected_components(mode='weak')
    subgraphs_cluster = [Q.subgraph(cluster) for cluster in clusters]
    #print(len(subgraphs_cluster))
    if len(subgraphs_cluster)>1:
        Q_clu_list.append(subgraphs_cluster[0])
    else:
        Q_clu_list.append(Q)

#随机价格生成器，用于初始化
#改为价格以顶点为单位,防止套利行为
v_plist = []
for i,vertex in enumerate(G.vs):
    name = vertex['name']
    c = random.randint(1,8)
    p = random.randint(c+1,15)
    #print(name,c,p)
    v_plist.append([name,c,p])

start_time = time.time()

def evaluation(j,tran_history,elapsed_time):
    #评估指标
    #交易成功率
    success_num = 0
    sum_num = len(tran_history)
    diff_list = []
    success_list = []
    regret_list = []
    sum_regret = 0
    for i,tra in enumerate(tran_history):
        p_b = find_G_in_list(plist_buyer, tra, diff_list)
        if p_b == -1:
            print(i,'not found')
        else:
            diff = abs(tra[2]-p_b)
            if tra[1] == 1:
                success_num += 1
                success_list.append([tra[2],p_b,diff])
                regret_list.append(diff)
                sum_regret = sum_regret +diff
            else:
                regret_list.append(p_b)
                sum_regret = sum_regret + p_b
    print('number of transaction:',sum_num)
    print('number of success transaction:',success_num)
    s_ratio = success_num / sum_num
    print('ratio:',s_ratio)
    print('sum_regret:',sum_regret)

    diff = 0
    per_diff_sum_s = 0
    cus_per_diff_sum_s = 0
    for di in diff_list:
        if di[1] != 0:
            diff += di[0]
            perdiff = di[0]/di[1]
            cus_perdiff = di[0]/di[2]
            per_diff_sum_s += perdiff
            cus_per_diff_sum_s += cus_perdiff
    avg_sdif = diff / sum_num
    print('总avg_sdiff:',avg_sdif)
    print('总的avg_per_diff',per_diff_sum_s/sum_num)
    print('总的心理预期avg_per_diff',cus_per_diff_sum_s /sum_num)
    diff_sum = 0
    per_diff_sum = 0
    cus_per_diff_sum = 0
    for l in success_list:
        diff_sum += l[2]
        #差异百分比= 差值/交易价格
        perdiff = l[2] / l[0]
        #cus_perdiff = 差值/顾客心理预期价格
        cus_perdiff = l[2] / l[1]
        per_diff_sum += perdiff
        cus_per_diff_sum += cus_perdiff
    n = len(success_list)
    print('成功avg_diff:',diff_sum/n)
    print('成功avg_per_diff',per_diff_sum/n)
    print('成功心理预期avg_per_diff',cus_per_diff_sum / n)

    f_name = str(ti) +str(int(j)) + "_test_x0_facebook.txt"
    # 打开文件进行写操作，如果文件不存在则创建它
    with open(f_name, "w") as file:
        # 写入变量的内容到文件
        file.write("number of transaction:" + str(sum_num)+"\n")
        file.write("number of success transaction:" + str(success_num)+"\n")
        file.write("success ratio:" + str(s_ratio)+"\n")
        file.write("avg_sdiff:"+str(avg_sdif)+"\n")
        file.write("avg_per_diff:"+str(per_diff_sum_s/sum_num)+"\n")
        file.write("sum customer avg_per_diff:"+str(cus_per_diff_sum_s /sum_num)+"\n")
        file.write("avg_diff:" + str(diff_sum/n)+"\n")
        file.write("avg_per_diff:" + str(per_diff_sum/n)+"\n")
        file.write("customer avg_per_diff:" + str(cus_per_diff_sum / n)+"\n")
        file.write("sum_regret:"+str(sum_regret)+"\n")
        file.write("time:"+str(elapsed_time)+"\n")
        file.write(str(sum_num)+"\n")
        file.write(str(success_num)+"\n")
        file.write(str(s_ratio)+"\n")
        file.write(str(avg_sdif)+"\n")
        file.write(str(per_diff_sum_s/sum_num)+"\n")
        file.write(str(cus_per_diff_sum_s /sum_num)+"\n")
        file.write(str(diff_sum/n)+"\n")
        file.write(str(per_diff_sum/n)+"\n")
        file.write(str(cus_per_diff_sum / n)+"\n")

    print(f"File '{f_name}' created and content written successfully.")


record_flag = []
re_time = []
#原始的，无价格调节模块
for i,Q in enumerate(Q_clu_list):
    if i == 10000 or i == 30000 or i == 50000 or i == 80000 or i == 120000:
        e_time = time.time()- start_time
        tra_len = len(tran_history)
        record_flag.append(tra_len)
        re_time.append(e_time)
        j = i/1000
        evaluation(j,tran_history,e_time)
    
    #进行子图匹配,得到匹配子图与价格
    print(i,'query no transaction history')
    tran_information = subisomorphic_pricing(Q,subgraphs,v_plist)
    #print('transaction information:',tran_information)
    if tran_information[0] > 0:
        p = tran_information[0]
        anslist = tran_information[1]
        #将query及其价格加入数据集，缩短相同query匹配时间
        if len(anslist) > 1:
            subgraphs.append(Q)
            #plist.append(p)
    else:
        #无匹配图
        print(i,'query no match')
        p = -1
    #交易模块
    #顾客心理预期价格
    if p > 0:
        #随机生成顾客心理预期价格
        p_buyer = Q.vcount() * 5 + random.uniform(0, 10)
        flag = False
        #顾客心理预期价格修正
        for p_b in plist_buyer:
            if gequal(Q,p_b[0]) == True:
                flag = True
                if p_b[1] != p_buyer:
                    p_b[1] = p_buyer
        #列表中无关于此查询的顾客心理预期值
        if flag == False:
            plist_buyer.append([Q,p_buyer])
        #print(p_buyer)
        #历史交易记录录入部分
        if p_buyer<p:
            tranflag = 0
            print('交易失败')
            tran_history.append([Q,0,p])
            #print(tran_history)    
        else:
            tranflag = 1
            print('交易成功')
            tran_history.append([Q,1,p])

end_time = time.time()  # 结束计时
elapsed_time = end_time - start_time  # 计算运行时间

#评估指标
#交易成功率
success_num = 0
sum_num = len(tran_history)
diff_list = []
success_list = []
regret_list = []
sum_regret = 0
for i,tra in enumerate(tran_history):
    p_b = find_G_in_list(plist_buyer, tra, diff_list)
    if p_b == -1:
        print(i,'not found')
    else:
        diff = abs(tra[2]-p_b)
        if tra[1] == 1:
            success_num += 1
            success_list.append([tra[2],p_b,diff])
            regret_list.append(diff)
            sum_regret = sum_regret +diff
        else:
            regret_list.append(p_b)
            sum_regret = sum_regret + p_b
print('number of transaction:',sum_num)
print('number of success transaction:',success_num)
s_ratio = success_num / sum_num
print('ratio:',s_ratio)
print('sum_regret:',sum_regret)

diff = 0
per_diff_sum_s = 0
cus_per_diff_sum_s = 0
for di in diff_list:
    if di[1] != 0:
        diff += di[0]
        perdiff = di[0]/di[1]
        cus_perdiff = di[0]/di[2]
        per_diff_sum_s += perdiff
        cus_per_diff_sum_s += cus_perdiff
avg_sdif = diff / sum_num
print('总avg_sdiff:',avg_sdif)
print('总的avg_per_diff',per_diff_sum_s/sum_num)
print('总的心理预期avg_per_diff',cus_per_diff_sum_s /sum_num)

diff_sum = 0
per_diff_sum = 0
cus_per_diff_sum = 0
for l in success_list:
    diff_sum += l[2]
    #差异百分比= 差值/交易价格
    perdiff = l[2] / l[0]
    #cus_perdiff = 差值/顾客心理预期价格
    cus_perdiff = l[2] / l[1]
    per_diff_sum += perdiff
    cus_per_diff_sum += cus_perdiff
n = len(success_list)
print('成功avg_diff:',diff_sum/n)
print('成功avg_per_diff',per_diff_sum/n)
print('成功心理预期avg_per_diff',cus_per_diff_sum / n)
print('运行时长：',elapsed_time)

# 打开文件进行写操作，如果文件不存在则创建它
with open(file_name, "w") as file:
    # 写入变量的内容到文件
    file.write("number of transaction:" + str(sum_num)+"\n")
    file.write("number of success transaction:" + str(success_num)+"\n")
    file.write("success ratio:" + str(s_ratio)+"\n")
    file.write("avg_sdiff:"+str(avg_sdif)+"\n")
    file.write("avg_per_diff:"+str(per_diff_sum_s/sum_num)+"\n")
    file.write("sum customer avg_per_diff:"+str(cus_per_diff_sum_s /sum_num)+"\n")
    file.write("avg_diff:" + str(diff_sum/n)+"\n")
    file.write("avg_per_diff:" + str(per_diff_sum/n)+"\n")
    file.write("customer avg_per_diff:" + str(cus_per_diff_sum / n)+"\n")
    file.write("sum_regret:"+str(sum_regret)+"\n")
    file.write("time:"+str(elapsed_time)+"\n")
    file.write(str(sum_num)+"\n")
    file.write(str(success_num)+"\n")
    file.write(str(s_ratio)+"\n")
    file.write(str(avg_sdif)+"\n")
    file.write(str(per_diff_sum_s/sum_num)+"\n")
    file.write(str(cus_per_diff_sum_s /sum_num)+"\n")
    file.write(str(diff_sum/n)+"\n")
    file.write(str(per_diff_sum/n)+"\n")
    file.write(str(cus_per_diff_sum / n)+"\n")

print(f"File '{file_name}' created and content written successfully.")

