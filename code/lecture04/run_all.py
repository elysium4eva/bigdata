"""run_all.py —— 按顺序运行 Lecture 4 的全部示例，重建样本、插图与交互 HTML。

顺序重要：l4_01 产出调研样本，l4_03 与 l4_12 会读取它；
l4_05 产出城市与类别样本（地图与可访问性要用），l4_09 产出 A/B 数据（不确定性要用）。
运行一次即可复现课件里的全部 CSV、PNG 与 HTML。
"""
import runpy                                                       # 以脚本方式执行同级脚本
import sys                                                         # 调整模块搜索路径
import time                                                        # 统计每个示例的耗时
from pathlib import Path                                           # 可靠定位脚本目录

HERE = Path(__file__).resolve().parent                             # 当前脚本所在目录
sys.path.insert(0, str(HERE))                                      # 让 "import _common_l4" 生效
SCRIPTS = [                                                        # 运行顺序即讲课顺序
    "l4_01_docs_metadata.py",                                      # 文档清单体检 + 数据字典
    "l4_02_descriptive_stats.py",                                  # 中心趋势、离散与形状
    "l4_03_correlation_crosstab.py",                               # 相关、列联表与受限范围
    "l4_04_multivariate.py",                                       # 多变量：热力图与小多图
    "l4_05_chart_zoo.py",                                          # 图型动物园
    "l4_06_plotly_interactive.py",                                 # 交互图表与静态对照
    "l4_07_dashboard.py",                                          # Python 仪表板
    "l4_08_geo_maps.py",                                           # 地理信息：choropleth 与空间自相关
    "l4_09_misleading.py",                                         # 误导与修正对照
    "l4_10_uncertainty.py",                                        # 不确定性表达
    "l4_11_accessibility.py",                                      # 色板与可访问性
    "l4_12_report_pipeline.py",                                    # 报告流水线（表 + 图 + 图注）
    "l4_13_highdim.py",                                            # 高维可视化（降维投影 / 平行坐标 / 解释性）
]                                                                  # 运行清单到此结束
for name in SCRIPTS:                                               # 逐个执行示例
    started = time.time()                                          # 开始计时
    print(f"\n===== run {name} =====")                             # 日志标记
    runpy.run_path(str(HERE / name), run_name="__main__")           # 以主程序身份运行
    print(f"----- done {name} in {time.time() - started:.2f}s")     # 打印耗时
print("\nAll Lecture 4 examples finished: data/samples in website/data/lecture04, "   # 收尾提示（产物位置）
      "figures in website/figures/lecture04, interactive HTML in figures/lecture04/interactive")   # 收尾提示
