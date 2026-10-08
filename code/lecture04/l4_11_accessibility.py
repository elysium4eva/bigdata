"""l4_11_accessibility.py —— 色盲友好与可访问性：色板对照、灰度检查与对比度计算。

可访问性不是"锦上添花"：约 8% 的男性有不同程度的色觉障碍，
灰度打印也会抹掉颜色差异。脚本对比三种色板、做灰度模拟、
计算文字与背景的 WCAG 对比度，并把检查项列成清单。
"""
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 数据表
import matplotlib.pyplot as plt                                    # 对照图
import matplotlib.colors as mcolors                                # 颜色空间转换（转灰度）
from _common_l4 import OKABE, PALETTE, DAT, save, dump             # 复用公共工具与两套色板

counts = pd.read_csv(DAT / "l4_category_counts.csv")               # 类别计数（来自 l4_05）
labels = counts["category"].tolist()                               # 类别名称
values = counts["count"].tolist()                                  # 各类计数

# ---------------------------------------------------------------- 1) 三种色板
palettes = {                                                       # 三套待比较的色板
    "matplotlib 默认": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"],
    "深大主题色": PALETTE[:len(labels)],                            # 本课程的品牌色板
    "Okabe–Ito（色盲友好）": OKABE[:len(labels)],                   # 通用色盲友好色板
}                                                                 # 三套色板字典结束
def to_gray(hex_color):                                            # 把颜色转成灰度亮度
    """用 ITU-R BT.601 权重计算感知亮度（0—1）。"""                    # 函数约定
    r, g, b = mcolors.to_rgb(hex_color)                            # 转成 0—1 的 RGB
    return 0.299 * r + 0.587 * g + 0.114 * b                       # 加权求和得到亮度
def contrast_ratio(fg, bg):                                        # 计算 WCAG 对比度
    """返回前景色与背景色的对比度（1—21，4.5 以上为正文达标）。"""       # 函数约定
    l1, l2 = to_gray(fg), to_gray(bg)                              # 计算两者亮度
    hi, lo = max(l1, l2), min(l1, l2)                              # 亮者与暗者
    return (hi + 0.05) / (lo + 0.05)                               # WCAG 公式
report_rows = []                                                   # 可访问性检查表
for name, pal in palettes.items():                                 # 逐套色板评估
    grays = [to_gray(c) for c in pal[:len(labels)]]                 # 各色转灰度
    gaps = np.diff(np.sort(grays))                                 # 排序后相邻灰度差
    min_gap = float(gaps.min()) if len(gaps) else 0.0              # 最小灰度间距（越大越易区分）
    contrasts = [contrast_ratio(c, "#FFFFFF") for c in pal[:len(labels)]]   # 与白底的对比度
    report_rows.append({"色板": name,                                  # 色板名
                        "最小灰度间距": round(min_gap, 3),              # 灰度可分性
                        "与白底最低对比度": round(min(contrasts), 2),    # 对比度下限
                        "正文达标(≥4.5)": bool(min(contrasts) >= 4.5),   # WCAG 正文阈值
                        "可用作文字色": bool(min(contrasts) >= 4.5)})    # 是否适合当文字色
report = pd.DataFrame(report_rows)                                 # 汇总为表
dump(report, "l4_accessibility_report.csv")                        # 落盘检查表
print(report.to_string(index=False))                               # 终端打印

# ---------------------------------------------------------------- 2) 图：色板、灰度与对比度
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), gridspec_kw={"width_ratios": [1.7, 1.2, 1.1]})   # 三格：色板灰度对照 / 灰度亮度分布 / 对比度矩阵
y_pos = np.arange(len(palettes))[::-1]                             # 每套色板一行
for i, (name, pal) in enumerate(palettes.items()):                 # 第一格：色板色块 + 灰度块
    for j, c in enumerate(pal[:len(labels)]):                      # 逐色画方块
        axes[0].add_patch(plt.Rectangle((j, y_pos[i] + 0.35), 0.9, 0.3,   # 彩色方块
                                        color=c, transform=axes[0].transData))   # 颜色
        gray = to_gray(c)                                          # 该色的灰度
        axes[0].add_patch(plt.Rectangle((j, y_pos[i] - 0.05), 0.9, 0.3,   # 灰度方块
                                        color=(gray, gray, gray), transform=axes[0].transData))   # 灰度值
axes[0].set_xlim(-0.2, len(labels) + 0.2); axes[0].set_ylim(-0.3, len(palettes) - 0.3)   # 坐标范围
axes[0].set_xticks([j + 0.45 for j in range(len(labels))], labels, rotation=25, fontsize=8)   # 类别标签
axes[0].set_yticks(y_pos + 0.3, list(palettes.keys()), fontsize=8.6)   # 色板名称
axes[0].grid(False)                                                # 色块图不需要网格
axes[0].set_title("上排原色、下排灰度模拟：灰度分不开就会误导色盲读者")   # 子图标题
x = np.arange(len(labels))                                         # 横坐标
width = 0.26                                                       # 柱宽
for k, (name, pal) in enumerate(palettes.items()):                 # 第二格：灰度间距对比
    grays = [to_gray(c) for c in pal[:len(labels)]]                 # 各色灰度
    axes[1].bar(x + (k - 1) * width, grays, width=width,            # 分组柱
                color=pal[:len(labels)], alpha=0.9, label=name)     # 用本色填充
axes[1].set_xticks(x, labels, rotation=25, fontsize=8)              # 类别标签
axes[1].set_ylabel("感知亮度（0—1）")                               # 轴标签
axes[1].set_title("灰度亮度越分散越易区分")                          # 子图标题
axes[1].legend(fontsize=7.6)                                       # 图例
bg_list = ["#FFFFFF", "#F7F3F2", "#1E2630"]                        # 三种背景
fg_list = ["#8A0C3C", "#66717D", "#C9A227", "#2E6F8E"]             # 四种前景
matrix = np.array([[contrast_ratio(f, b) for b in bg_list] for f in fg_list])   # 对比度矩阵
im = axes[2].imshow(matrix, cmap="RdPu", vmin=1, vmax=21)          # 第三格：对比度热力图
axes[2].set_xticks(range(len(bg_list)), ["白", "浅底", "深底"], fontsize=8.6)   # 背景标签
axes[2].set_yticks(range(len(fg_list)), fg_list, fontsize=8.6)     # 前景标签
for i in range(matrix.shape[0]):                                   # 在格子写对比度数值
    for j in range(matrix.shape[1]):                               # 双层循环
        axes[2].text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8.6,   # 在对比度矩阵中写数值
                     color="white" if matrix[i, j] > 10 else "#1E2630")   # 数值标签
axes[2].grid(False)                                                # 热力图不需要网格
axes[2].set_title("WCAG 对比度（≥4.5 达标）")                        # 子图标题
plt.colorbar(im, ax=axes[2], fraction=0.046)                       # 颜色条
plt.tight_layout()                                                 # 调整三格
save(fig, "fig_palette_accessibility.png")                         # 保存：可访问性对照

# ---------------------------------------------------------------- 3) 检查清单
CHECKLIST = [                                                      # 可访问性自检清单
    "色板选色盲友好（Okabe–Ito / viridis 等），不要红绿对比",                      # 检查项一：色板选择
    "颜色之外再加冗余编码（形状、线型、图案、直接标注）",                                  # 检查项二：冗余编码
    "灰度打印后仍能区分（本页第二格检查）",                                         # 检查项三：灰度可区分
    "文字与背景对比度 ≥ 4.5（大字号 ≥ 3.0）",                                  # 检查项四：对比度阈值
    "字号：投影正文 ≥ 18pt、图表标注 ≥ 12pt",                                 # 检查项五：字号下限
    "不要只靠颜色传达涨跌或好坏（红绿组合最有问题）",                                    # 检查项六：不要只用颜色表达好坏
    "为每张图提供 alt 文本（一句话讲清结论）",                                     # 检查项七：提供 alt 文本
    "交互图提供键盘可达的筛选与缩放",                                            # 检查项八：键盘可达
]                                                                 # 清单结束
dump(pd.DataFrame({"检查项": CHECKLIST}), "l4_accessibility_checklist.csv")   # 落盘清单
print("检查清单条数:", len(CHECKLIST))                              # 打印清单规模
print("红绿对比的灰度差（示例）:", round(abs(to_gray("#8A0C3C") - to_gray("#2E6F8E")), 3))   # 示例数值
