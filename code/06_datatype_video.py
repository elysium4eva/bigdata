"""06_datatype_video.py —— 视频型数据：合成帧序列 + 关键帧/时序信号可视化。"""  # 模块注释
import numpy as np                                                    # 数值计算
import matplotlib.pyplot as plt                                       # 绘图
from _common import DAT, save                                         # 公共工具（只用到数据目录与保存函数）

H = W = 48                                                            # 每帧分辨率 48×48
NF = 12                                                               # 共 12 帧（1 秒 12 fps 的示意）
frames, means = [], []                                                # 初始化帧列表与逐帧亮度均值
for k in range(NF):                                                   # 逐帧生成视频内容
    img = np.zeros((H, W))                                            # 全黑背景
    yy, xx = np.mgrid[0:H, 0:W]                                       # 像素坐标网格
    cy, cx = H / 2, 2 + k * (W - 4) / (NF - 1)                        # 小球中心：随时间从左向右移动
    ball = ((yy - cy) ** 2 + (xx - cx) ** 2) < 5 ** 2                 # 半径 5 的小球掩码
    img[ball] = 220                                                   # 小球像素设为亮色
    bar_x = (W * k / NF)                                              # 竖向光条位置：随时间移动
    img[:, int(bar_x):int(bar_x) + 3] += 60                           # 光条像素加亮
    frames.append(np.clip(img, 0, 255))                               # 收进帧序列
    means.append(float(img[ball].mean()))                             # 记录小球区域的平均亮度
frames = np.stack(frames)                                             # 堆叠为 (12, 48, 48) 的视频张量
np.save(DAT / "video_frames.npy", frames)                             # 保存帧张量样本
(DAT / "video_frames").mkdir(parents=True, exist_ok=True)             # 创建抽帧图片目录
for i in range(NF):                                                   # 逐帧导出 PNG，模拟“抽帧存盘”
    plt.imsave(DAT / "video_frames" / ("frame_%02d.png" % (i + 1)),   # 保存第 i 帧到 data/video_frames/
               frames[i], cmap="gray")                                # 用灰度配色保存
print("[data]", DAT / "video_frames.npy")                             # 打印样本位置

# ---------------------------------------------------------------- 可视化
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                   # 一行三列：关键帧 / 帧间差分 / 时序信号
axes[0].imshow(frames[0], cmap="gray")                                # 左图：第 1 帧
axes[0].set_title("关键帧 t=1（48×48 灰度）")                          # 左图标题
axes[0].set_xticks([]); axes[0].set_yticks([])                         # 隐藏刻度
axes[0].grid(False)                                                   # 关闭网格
axes[1].imshow(np.abs(frames[6] - frames[0]), cmap="inferno")          # 中图：第 7 帧与第 1 帧的差分（运动检测）
axes[1].set_title("帧间差分 |I(t=7) − I(t=1)|（运动检测）")              # 中图标题
axes[1].set_xticks([]); axes[1].set_yticks([])                         # 隐藏刻度
axes[1].grid(False)                                                   # 关闭网格
axes[2].plot(range(1, NF + 1), means, "-o", color="#8A0C3C", ms=4)     # 右图：逐帧亮度均值构成的时间序列
axes[2].set_title("逐帧信号：亮度均值随 t 变化")                        # 右图标题
axes[2].set_xlabel("帧序号 t")                                        # 右图横轴含义
axes[2].set_ylabel("平均亮度")                                         # 右图纵轴含义
plt.tight_layout()                                                    # 调整布局
save(fig, "fig_video.png")                                            # 保存图片
