"""07_datatype_graph.py —— 图/网络型数据：生成无标度网络 + 布局可视化。"""  # 模块注释
import numpy as np                                                    # 数值计算
import pandas as pd                                                   # 导出边表与点表
import matplotlib.pyplot as plt                                       # 绘图
from _common import RNG, DAT, save, dump_csv                          # 公共工具

N, M = 60, 2                                                          # 节点数 60，每新增一个节点连 2 条边（BA 模型）
edges = []                                                            # 初始化边列表
targets = [0, 1]                                                      # 初始核心：节点 0 与 1 已相连
edges.append((0, 1))                                                  # 记录核心边
for v in range(2, N):                                                 # 从节点 2 开始逐个加入网络（优先连接）
    deg = np.zeros(v)                                                 # 统计已有 v 个节点的度数
    for a, b in edges:                                                # 遍历已有边累加度数
        deg[a] += 1                                                   # 起点度数 +1
        deg[b] += 1                                                   # 终点度数 +1
    prob = (deg + 1) / (deg + 1).sum()                                # 连接概率 ∝ 度数（+1 平滑，避免零概率）
    chosen = RNG.choice(v, size=M, replace=False, p=prob)             # 按概率无放回抽取 M 个邻居
    for c in chosen:                                                  # 遍历抽中的邻居
        edges.append((int(c), v))                                     # 连一条新边（形成“富者更富”结构）
adj = np.zeros((N, N))                                                # 初始化邻接矩阵
for a, b in edges:                                                    # 用边表填充邻接矩阵
    adj[a, b] = adj[b, a] = 1                                         # 无向图：矩阵对称置 1
deg = adj.sum(axis=1)                                                 # 每个节点的度 = 该行之和

# ---------------------------------------------------------------- 力导向布局（Fruchterman–Reingold 简化版）
pos = RNG.uniform(0, 1, (N, 2))                                       # 随机初始化节点坐标
for it in range(120):                                                 # 迭代 120 轮让布局收敛
    delta = pos[:, None, :] - pos[None, :, :]                         # 任意两点之间的位移向量
    dist = np.sqrt((delta ** 2).sum(-1)) + 1e-9                       # 两点距离（加极小值防止除零）
    rep = (delta / dist[..., None]) / dist[..., None] ** 2            # 所有点对之间的斥力（库仑式，反比于距离平方）
    np.fill_diagonal(rep[:, :, 0], 0); np.fill_diagonal(rep[:, :, 1], 0)  # 自身与自身无斥力
    attr = -(delta / dist[..., None]) * dist[..., None] * adj[..., None]  # 相邻点之间的引力（沿连线方向）
    step = (rep.sum(1) + attr.sum(1)) * 0.002                         # 合力 × 步长 = 每轮位移
    pos = pos + np.clip(step, -0.02, 0.02)                            # 限制单步位移，避免震荡
pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0) + 1e-9)           # 归一化到 [0,1] 方便绘图

dump_csv(pd.DataFrame({"src": [a for a, _ in edges],                  # 边表：起点
                       "dst": [b for _, b in edges]}),                # 边表：终点
         "graph_edges.csv")                                           # 落盘边表
dump_csv(pd.DataFrame({"node": range(N), "degree": deg}),             # 点表：节点编号
         "graph_nodes.csv")                                           # 落盘点表（含度数）

# ---------------------------------------------------------------- 可视化
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))                    # 一行两列：网络图 / 度分布
for a, b in edges:                                                    # 逐条边画连线
    axes[0].plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],      # 连接两端点坐标
                 color="#B9BFC7", lw=0.8, zorder=1)                   # 灰色细线，画在节点下层
axes[0].scatter(pos[:, 0], pos[:, 1], s=20 + 9 * deg,                 # 节点大小随度数增大
                c=deg, cmap="RdPu", edgecolors="#711426", zorder=2)  # 颜色也表示度数，便于识别枢纽
axes[0].set_title("网络图（%d 节点 / %d 边，BA 无标度）" % (N, len(edges)))  # 左图标题
axes[0].set_xticks([]); axes[0].set_yticks([])                        # 隐藏刻度
axes[0].grid(False)                                                   # 关闭网格
axes[0].set_aspect("equal")                                           # 等比例显示，避免图形拉伸

vals, cnt = np.unique(deg, return_counts=True)                        # 统计度分布（度数 → 节点个数）
axes[1].loglog(vals, cnt, "o", color="#8A0C3C", ms=6)                 # 右图：双对数坐标下画度分布（近似直线）
axes[1].set_title("度分布（双对数：幂律特征）")                        # 右图标题
axes[1].set_xlabel("节点度 k")                                        # 右图横轴含义
axes[1].set_ylabel("度为 k 的节点数 P(k)")                            # 右图纵轴含义
plt.tight_layout()                                                    # 调整布局
save(fig, "fig_graph.png")                                            # 保存图片
