"""10_datatype_structured_unstructured.py —— 结构化 / 半结构化 / 非结构化三类样本与可视化。"""  # 模块注释
import json                                                           # 标准库：生成半结构化的 JSON 数据
import numpy as np                                                    # 数值计算
import pandas as pd                                                   # 表格处理（结构化）
import matplotlib.pyplot as plt                                       # 绘图
from _common import DAT, save, dump_csv                               # 公共工具

# ---------------------------------------------------------------- 1) 结构化：定长字段的二维表
table = pd.DataFrame({                                                # 结构化数据 = 行 × 列，列有名称与类型
    "order_id": ["SO2026001", "SO2026002", "SO2026003", "SO2026004"],  # 订单号（字符串主键）
    "date": ["2026-03-02", "2026-03-02", "2026-03-03", "2026-03-03"],  # 下单日期（可解析为日期型）
    "amount": [1280.50, 640.00, 3980.75, 215.20],                      # 订单金额（浮点型）
    "paid": [True, True, False, True],                                 # 是否已付款（布尔型）
})                                                                    # 结构化样本表构造结束
dump_csv(table, "structured_orders.csv")                              # 落盘结构化样本

# ---------------------------------------------------------------- 2) 半结构化：可嵌套、字段可缺省的 JSON
records = [                                                           # 半结构化数据 = 自带“模式”的文本
    {"order_id": "SO2026001", "buyer": {"name": "林", "vip": True},   # 记录 1：含嵌套对象
     "items": [{"sku": "A-1", "qty": 2}, {"sku": "B-7", "qty": 1}]},  # 记录 1：含数组字段
    {"order_id": "SO2026002", "buyer": {"name": "陈"},                # 记录 2：缺少 vip 字段（模式不固定）
     "items": [{"sku": "C-3", "qty": 5}]},                            # 记录 2：数组长度也不同
    {"order_id": "SO2026003", "buyer": {"name": "王", "vip": False},  # 记录 3
     "items": [], "note": "待补开发票"},                               # 记录 3：多出一个 note 字段
]                                                                     # 半结构化记录列表结束
(DAT / "semistructured_orders.json").write_text(                      # 把半结构化数据写入仓库
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")  # 中文不转义 + 缩进便于阅读
print("[data]", DAT / "semistructured_orders.json")                   # 打印落盘位置

# ---------------------------------------------------------------- 3) 非结构化：无固定字段的自然语言/波形
free_text = ("本季度公司在东南亚市场的收入同比增长三成，主要来自跨境电商渠道；"  # 非结构化文本第 1 段
             "管理层预计下季度毛利率保持稳定，但需关注汇率波动与海运费上行带来的成本压力。")  # 第 2 段
(DAT / "unstructured_note.txt").write_text(free_text, encoding="utf-8")  # 落盘非结构化文本样本
wave_sample = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 400))          # 非结构化的另一形态：原始波形
np.save(DAT / "unstructured_wave.npy", wave_sample)                   # 保存为二进制数组样本
print("[data]", DAT / "unstructured_note.txt")                        # 打印落盘位置

# ---------------------------------------------------------------- 4) 排版辅助：手工按字数折行（比 wrap 更可控）
def wrap_cjk(s, per_line=24):                                         # 定义中文折行函数：每行多少字
    """按固定字数把长文本切成多行，避免自动排版把画布撑大。"""          # 函数说明
    return "\n".join(s[i:i + per_line] for i in range(0, len(s), per_line))  # 每 per_line 个字符插一个换行

# ---------------------------------------------------------------- 5) 可视化：三种形态并排展示
fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3))                   # 一行三列：表格 / JSON / 文本
fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.04,     # 手动留出边距，比 tight_layout 更可控
                    wspace=0.12)                                      # 子图水平间距
for ax in axes:                                                       # 统一处理三个子图
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)                              # 固定坐标范围，防止 tight bbox 被文字撑开
    ax.set_xticks([]); ax.set_yticks([])                              # 隐藏刻度
    ax.grid(False)                                                    # 关闭网格

t = axes[0].table(cellText=table.astype(str).values,                  # 左图：把 DataFrame 渲染成表格
                  colLabels=list(table.columns),                      # 表头取自列名
                  loc="center", cellLoc="center", bbox=[0.02, 0.42, 0.96, 0.5])  # 用 bbox 固定表格位置
t.auto_set_font_size(False); t.set_fontsize(8)                        # 关闭自动字号，手动设为 8pt
axes[0].set_title("结构化：定长字段的二维表 (CSV)", fontsize=12)        # 左图标题

axes[1].text(0.02, 0.96, wrap_cjk(json.dumps(records[0],              # 中图：展示 JSON 记录（含嵌套与数组）
               ensure_ascii=False), 26)[:520],                        # 折行后只取前 520 个字符
             family=["Consolas", "Microsoft YaHei"],                  # 等宽字体 + 中文字体逐字回退
             fontsize=8.2, va="top", ha="left")                       # 字号与左上角对齐方式
axes[1].set_title("半结构化：嵌套 JSON（模式可变）", fontsize=12)        # 中图标题
axes[1].text(0.02, 0.18, "字段可缺省、可新增：解析时需容错",            # 中图底部补一句要点
             fontsize=8.5, color="#711426", va="bottom", ha="left")   # 用深红色突出

axes[2].text(0.02, 0.96, wrap_cjk(free_text, 22), fontsize=9.5,       # 右图：展示无字段的自然语言文本
             va="top", ha="left")                                     # 手工折行后的多行文本
axes[2].plot(np.linspace(0, 1, 400), 0.22 + 0.12 * wave_sample,       # 右图下半部画一段原始波形
             color="#8A0C3C", lw=1.0)                                 # 表示“非结构化”也可以是信号
axes[2].text(0.02, 0.06, "文本 / 音频 / 图像：无字段可枚举",            # 右图底部补一句要点
             fontsize=8.5, color="#711426", va="bottom", ha="left")   # 用深红色突出
axes[2].set_title("非结构化：文本 / 音频 / 图像", fontsize=12)          # 右图标题
save(fig, "fig_structured.png")                                       # 保存图片（已固定坐标范围，不会异常放大）
