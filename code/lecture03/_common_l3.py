"""_common_l3.py —— Lecture 3（数据整备）示例的公共工具（每一行都有中文注释）。

固定随机种子保证样本与图片在每次运行时完全一致，这是课堂示例可复现的前提。
"""
import matplotlib                                                  # 先选绘图后端，再导入其余绘图模块
matplotlib.use("Agg")                                              # 无界面后端：没有显示器也能出图
import matplotlib.pyplot as plt                                    # 图形与坐标轴接口
import numpy as np                                                 # 数值数组与随机数
import pandas as pd                                                # 表格数据结构
from pathlib import Path                                           # 跨平台路径处理

SEED = 20260317                                                    # 所有脚本共用的固定随机种子
RNG = np.random.default_rng(SEED)                                  # 共享随机数发生器：结果可复现
ROOT = Path(__file__).resolve().parents[2]                         # …/website（code/lecture03 往上两级）
FIG = ROOT / "figures" / "lecture03"                               # Lecture 3 的图片输出目录
DAT = ROOT / "data" / "lecture03"                                  # Lecture 3 的样本数据输出目录
FIG.mkdir(parents=True, exist_ok=True)                             # 目录不存在则自动创建
DAT.mkdir(parents=True, exist_ok=True)                             # 目录不存在则自动创建

plt.rcParams["figure.dpi"] = 140                                   # 交互预览分辨率
plt.rcParams["savefig.dpi"] = 200                                  # 保存分辨率：投影时更清晰
plt.rcParams["savefig.bbox"] = "tight"                             # 保存时自动裁掉多余白边
plt.rcParams["axes.grid"] = True                                   # 打开浅网格，便于读数
plt.rcParams["grid.alpha"] = 0.25                                  # 网格透明度，避免喧宾夺主
plt.rcParams["axes.titlesize"] = 13                                # 子图标题字号（面向投影可读性）
plt.rcParams["axes.labelsize"] = 11                                # 坐标轴标签字号
plt.rcParams["xtick.labelsize"] = 10                               # x 轴刻度字号
plt.rcParams["ytick.labelsize"] = 10                               # y 轴刻度字号
plt.rcParams["legend.fontsize"] = 9.2                              # 图例字号
plt.rcParams["font.size"] = 10                                     # 其余文字默认字号


def save(fig, name):                                               # 保存一张图并释放内存
    """把图保存到 website/figures/lecture03 目录并关闭图形对象。"""    # 函数约定
    path = FIG / name                                              # 拼出完整输出路径
    fig.savefig(path, facecolor="white")                           # 白底保存，方便插入幻灯片
    plt.close(fig)                                                 # 关闭图形，释放内存
    print("[figure]", path)                                        # 打印产物路径，便于检查


def dump(df, name):                                                # 把样本表写入数据目录
    """把 DataFrame 写成 UTF-8 CSV 存到 website/data/lecture03。"""    # 函数约定
    path = DAT / name                                              # 拼出完整输出路径
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig 让 Excel 正确显示中文
    print("[data]", path)                                          # 打印产物路径，便于检查


def money_to_float(series):                                        # 把脏金额字符串转成数值
    """去掉货币符号、千分位与空格后转成浮点数。"""                       # 函数约定
    cleaned = (series.astype(str)                                  # 先统一成字符串才能用正则
               .str.replace(r"[^\d.\-]", "", regex=True)           # 只保留数字、小数点和负号
               .replace({"": np.nan}))                             # 空字符串转为真正的缺失值
    return pd.to_numeric(cleaned, errors="coerce")                 # 无法解析的也转成 NaN
