"""_common_l4.py —— Lecture 4（数据描述与可视化）示例的公共工具（每一行都有中文注释）。

统一字体、色板与出图参数，保证 13 张插图在课件里风格一致、字号可读；
固定随机种子让样本与图在每次运行时完全一致。
"""
import json                                                        # 读写 JSON（地理数据与元数据）
import matplotlib                                                  # 先选后端再导入其余绘图模块
matplotlib.use("Agg")                                              # 无界面后端：没有显示器也能出图
import matplotlib.pyplot as plt                                    # 图形与坐标轴接口
import numpy as np                                                 # 数值数组与随机数
import pandas as pd                                                # 表格数据结构
from pathlib import Path                                           # 跨平台路径处理

SEED = 20260417                                                    # 本讲共用的固定随机种子
RNG = np.random.default_rng(SEED)                                  # 共享随机数发生器：结果可复现
ROOT = Path(__file__).resolve().parents[2]                         # …/website（code/lecture04 往上两级）
FIG = ROOT / "figures" / "lecture04"                               # 图片输出目录
DAT = ROOT / "data" / "lecture04"                                  # 样本数据输出目录
INT = FIG / "interactive"                                          # plotly 交互 HTML 输出目录
for _d in (FIG, DAT, INT):                                         # 逐个确保目录存在
    _d.mkdir(parents=True, exist_ok=True)                          # 不存在则自动创建

# ---------------------------------------------------------------- 统一视觉规范
FONT = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]                # 中文字体优先级（Windows 自带前两个）
PALETTE = ["#8A0C3C", "#C9A227", "#66717D", "#2E6F8E", "#711426",  # 分类色板：深大荔枝红 + 金 + 灰 + 蓝
           "#A6426B", "#4F7942", "#B5651D"]                        # 再补四个可区分色
SEQ = "RdPu"                                                       # 连续色板（单色深浅，适合 choropleth）
DIV = "RdBu_r"                                                     # 发散色板（有中心基准的差异）
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",               # Okabe–Ito 色盲友好色板（8 色）
         "#0072B2", "#D55E00", "#CC79A7", "#000000"]               # 各色在灰度下仍可区分

plt.rcParams["font.sans-serif"] = FONT                             # 全局中文字体
plt.rcParams["axes.unicode_minus"] = False                         # 正常显示负号而不是方框
plt.rcParams["figure.dpi"] = 140                                   # 交互预览分辨率
plt.rcParams["savefig.dpi"] = 200                                  # 保存分辨率：投影更清晰
plt.rcParams["savefig.bbox"] = "tight"                             # 保存时裁掉多余白边
plt.rcParams["axes.grid"] = True                                   # 打开浅网格便于读数
plt.rcParams["grid.alpha"] = 0.25                                  # 网格透明度
plt.rcParams["axes.titlesize"] = 12.5                              # 子图标题字号
plt.rcParams["axes.labelsize"] = 11                                # 坐标轴标签字号
plt.rcParams["xtick.labelsize"] = 9.5                              # x 轴刻度字号
plt.rcParams["ytick.labelsize"] = 9.5                              # y 轴刻度字号
plt.rcParams["legend.fontsize"] = 9                                # 图例字号
plt.rcParams["figure.titlesize"] = 14                              # 总标题字号
plt.rcParams["axes.spines.top"] = False                            # 去掉上方边框，减少图表垃圾
plt.rcParams["axes.spines.right"] = False                          # 去掉右侧边框


def style_axes(ax, title=None, xlabel=None, ylabel=None):           # 统一设置单个子图的标题与轴标签
    """给一个子图套用统一风格：标题、轴标签、网格。"""                  # 函数约定
    if title:                                                      # 有标题才设置
        ax.set_title(title)                                        # 设置子图标题
    if xlabel:                                                     # 有横轴标签才设置
        ax.set_xlabel(xlabel)                                      # 设置横轴标签
    if ylabel:                                                     # 有纵轴标签才设置
        ax.set_ylabel(ylabel)                                      # 设置纵轴标签
    return ax                                                      # 返回同一对象便于链式调用


def annotated(ax, text, loc="upper left"):                          # 在图内加注释（图注自足性的关键）
    """在图内四角之一加一行说明文字。"""                                # 函数约定
    xy = {"upper left": (0.02, 0.96), "upper right": (0.98, 0.96),  # 上方两个位置
          "lower left": (0.02, 0.06), "lower right": (0.98, 0.06)}.get(loc, (0.02, 0.96))   # 下方两个位置，未知位置退回左上
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, va="top",   # 用轴坐标放置文字
            ha="left" if "left" in loc else "right", fontsize=8.6,  # 左/右对齐
            color="#1E2630", bbox=dict(boxstyle="round,pad=0.25",   # 加一个浅底框提高可读性
                                       fc="#F7F3F2", ec="#D6CDD0", lw=0.6))   # 底色与边框
    return ax                                                      # 返回同一对象


def save(fig, name):                                               # 保存图片并释放内存
    """把图保存到 website/figures/lecture04 并关闭图形对象。"""         # 函数约定
    path = FIG / name                                              # 拼出完整路径
    fig.savefig(path, facecolor="white")                           # 白底保存，便于插入课件
    plt.close(fig)                                                 # 关闭图形释放内存
    print("[figure]", path)                                        # 打印产物路径


def dump(df, name):                                                # 把样本表写入数据目录
    """把 DataFrame 写成 UTF-8 CSV 存到 website/data/lecture04。"""     # 函数约定
    path = DAT / name                                              # 拼出完整路径
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig 便于 Excel 打开
    print("[data]", path)                                          # 打印产物路径


def load_geojson(name):                                            # 读取（可选的）地理边界文件
    """从数据目录读取 GeoJSON；文件不存在时返回 None，由调用方退化处理。"""   # 函数约定
    path = DAT / name                                              # 拼接路径
    if not path.exists():                                          # 文件缺失
        print("[geojson] not found, fall back to a symbol map:", path)   # 提示将退化为气泡图
        return None                                                # 返回空值
    with open(path, "r", encoding="utf-8") as fh:                  # 以 UTF-8 读取
        return json.load(fh)                                       # 返回解析后的字典


def lorenz(values):                                                # 计算洛伦兹曲线（衡量不均衡）
    """返回累计人口占比与累计数值占比，用于绘制洛伦兹曲线与基尼系数。"""     # 函数约定
    x = np.sort(np.asarray(values, dtype=float))                   # 升序排列
    cum = np.cumsum(x) / x.sum()                                   # 累计占比
    return np.insert(np.arange(1, len(x) + 1) / len(x), 0, 0), np.insert(cum, 0, 0)   # 两端补 0
