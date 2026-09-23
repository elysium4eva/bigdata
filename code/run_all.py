"""run_all.py —— 一键运行全部数据类型示例：生成样本数据与可视化图片。"""  # 模块注释
import runpy                                                          # 用 runpy 以“脚本方式”执行其他 .py 文件
import sys                                                            # 用于修改模块搜索路径
import time                                                           # 用于统计每个脚本耗时
from pathlib import Path                                              # 路径工具

HERE = Path(__file__).resolve().parent                                # 当前脚本所在目录：website/code
sys.path.insert(0, str(HERE))                                         # 保证子脚本能 import _common
SCRIPTS = ["01_datatype_numeric.py",                                  # 数值型（连续/离散）
           "02_datatype_categorical.py",                              # 类别型（名义/有序）
           "03_datatype_text.py",                                     # 文本型
           "04_datatype_image.py",                                    # 图像型
           "05_datatype_audio.py",                                    # 音频型
           "06_datatype_video.py",                                    # 视频型
           "07_datatype_graph.py",                                    # 图 / 网络型
           "08_datatype_spatiotemporal.py",                           # 时空型
           "09_datastructure_cross_time_panel.py",                    # 截面 / 时序 / 面板
           "10_datatype_structured_unstructured.py"]                  # 结构化 / 半结构化 / 非结构化
for name in SCRIPTS:                                                  # 依次执行每个脚本（顺序即讲解顺序）
    t0 = time.time()                                                  # 记录起始时间
    print("\n===== run", name, "=====")                               # 打印当前脚本名，便于观察进度
    runpy.run_path(str(HERE / name), run_name="__main__")             # 以主程序身份运行该脚本
    print("----- done %s in %.2fs" % (name, time.time() - t0))        # 打印耗时
print("\n全部完成：样本在 website/data/，图片在 website/figures/")      # 收尾提示输出位置
