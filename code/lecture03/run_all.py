"""run_all.py —— 按顺序运行 Lecture 3 的全部示例，重建所有样本与图片。

顺序很重要：l3_02 产出的两期抽取会被 l3_05 用到，l3_01 产出的脏订单表也是 l3_05 的输入。
运行本脚本一次，即可复现课件里用到的全部 CSV 与 PNG。
"""
import runpy                                                       # 以"脚本方式"执行同级脚本
import sys                                                         # 调整模块搜索路径
import time                                                        # 统计每个示例的耗时
from pathlib import Path                                           # 可靠地定位脚本目录

HERE = Path(__file__).resolve().parent                             # 示例脚本所在目录
sys.path.insert(0, str(HERE))                                      # 让 "import _common_l3" 生效
SCRIPTS = [                                                        # 运行顺序即讲课顺序
    "l3_01_quality_profile.py",                                    # 质量仪表盘与评分卡
    "l3_02_join_concat.py",                                        # 纵向拼接与结构对齐
    "l3_03_join_types_validate.py",                                # 连接语义与键校验
    "l3_04_reshape_toolbox.py",                                    # 宽长表、透视与编码工具
    "l3_05_cleaning_pipeline.py",                                  # 清洗五步法
    "l3_06_outliers_iqr_z_iforest.py",                             # 三种识别规则与三种处置
    "l3_07_missing_mechanisms.py",                                 # MCAR/MAR/MNAR 模拟
    "l3_08_imputation_compare.py",                                 # 五种插补策略评分
    "l3_09_panel_balance.py",                                      # 面板结构与插值
    "l3_10_transform_assert.py",                                   # 变换与 schema 契约
]                                                                  # 运行清单到此结束
for name in SCRIPTS:                                               # 逐个执行示例
    started = time.time()                                          # 开始计时
    print(f"\n===== run {name} =====")                             # 日志中标记当前脚本
    runpy.run_path(str(HERE / name), run_name="__main__")           # 以主程序身份运行该脚本
    print(f"----- done {name} in {time.time() - started:.2f}s")     # 打印耗时
print("\nAll Lecture 3 examples finished: samples in website/data/lecture03, figures in website/figures/lecture03")   # 收尾提示
