"""_common.py —— 公共工具：中文字体、输出目录、保存函数（每一行都有注释）。"""  # 模块文档字符串，说明用途
import matplotlib                                                    # 导入 matplotlib 主包
matplotlib.use("Agg")                                                # 使用无界面后端，保证在无显示器环境也能出图
import matplotlib.pyplot as plt                                      # 导入绘图接口并简写为 plt
import numpy as np                                                   # 导入数值计算库并简写为 np
from pathlib import Path                                             # 导入跨平台路径工具

RNG = np.random.default_rng(20260917)                                # 固定随机种子，保证每次运行结果一致
FIG = Path(__file__).resolve().parents[1] / "figures"                # 图片输出目录：仓库下的 website/figures
DAT = Path(__file__).resolve().parents[1] / "data"                   # 数据输出目录：仓库下的 website/data
FIG.mkdir(parents=True, exist_ok=True)                               # 若图片目录不存在则创建（含父目录）
DAT.mkdir(parents=True, exist_ok=True)                               # 若数据目录不存在则创建（含父目录）

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # 中文字体优先级列表（Windows 自带前两个）
plt.rcParams["font.monospace"] = ["Consolas", "Microsoft YaHei", "DejaVu Sans Mono"]  # 等宽字体也补中文字体，避免缺字
plt.rcParams["axes.unicode_minus"] = False                           # 让坐标轴负号正常显示而不是方框
plt.rcParams["figure.dpi"] = 140                                     # 屏幕/保存的默认分辨率
plt.rcParams["savefig.dpi"] = 200                                    # 保存图片时的分辨率（课件里更清晰）
plt.rcParams["savefig.bbox"] = "tight"                               # 保存时自动裁掉多余空白
plt.rcParams["axes.grid"] = True                                     # 默认打开网格，便于读数
plt.rcParams["grid.alpha"] = 0.25                                    # 网格透明度，避免喧宾夺主
plt.rcParams["axes.titlesize"] = 13                                  # 子图标题字号（投影可读性优先）
plt.rcParams["axes.labelsize"] = 11                                  # 坐标轴标签字号
plt.rcParams["xtick.labelsize"] = 10                                 # x 轴刻度字号
plt.rcParams["ytick.labelsize"] = 10                                 # y 轴刻度字号
plt.rcParams["legend.fontsize"] = 9.2                                # 图例字号


def save(fig, name):                                                 # 定义保存函数：传入图形对象和文件名
    """保存图形到 FIG 目录并关闭，避免内存泄漏。"""                    # 函数说明
    path = FIG / name                                                # 拼接完整输出路径
    fig.savefig(path, facecolor="white")                             # 以白底保存为 PNG
    plt.close(fig)                                                   # 关闭图形，释放内存
    print("[figure]", path)                                          # 打印输出位置，便于检查


def dump_csv(df, name):                                              # 定义 CSV 落盘函数
    """把 DataFrame 写入 DAT 目录（UTF-8-SIG 便于 Excel 打开中文）。"""  # 函数说明
    path = DAT / name                                                # 拼接完整输出路径
    df.to_csv(path, index=False, encoding="utf-8-sig")               # 写出 CSV，不写行号，兼容 Excel
    print("[data]", path)                                            # 打印输出位置
