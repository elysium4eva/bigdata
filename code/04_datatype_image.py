"""04_datatype_image.py —— 图像型数据：合成样本图片 + 像素级可视化。"""  # 模块注释
import numpy as np                                                    # 数值计算（图像就是矩阵）
import matplotlib.pyplot as plt                                       # 绘图
from _common import FIG, DAT, save                                    # 公共工具（需要直接写图片样本）

# ---------------------------------------------------------------- 1) 生成三类合成图像（32×32 灰度）
H = W = 32                                                            # 图像高度与宽度（32 像素，便于课堂展示）
yy, xx = np.mgrid[0:H, 0:W]                                           # 生成像素坐标网格
cy, cx = (H - 1) / 2, (W - 1) / 2                                      # 计算图像中心坐标
circle = (((yy - cy) ** 2 + (xx - cx) ** 2) < 9 ** 2).astype(float)    # 类别 0：半径 9 的实心圆
square = ((np.abs(yy - cy) <= 8) & (np.abs(xx - cx) <= 8)).astype(float)  # 类别 1：边长 17 的正方形
triangle = ((xx >= yy) & (xx <= W - 1 - yy)).astype(float)             # 类别 2：等腰三角形（上尖下宽）
protos = [circle, square, triangle]                                    # 三个类别的原型图案
labels = ["circle", "square", "triangle"]                              # 三个类别的名称

imgs, lab = [], []                                                     # 初始化样本列表与标签列表
for i, p in enumerate(protos):                                         # 遍历三个类别
    for k in range(4):                                                 # 每个类别生成 4 张样本
        noise = np.random.default_rng(100 + i * 10 + k).normal(0, 0.08, (H, W))  # 加入噪声模拟真实拍摄
        imgs.append(np.clip(p + noise, 0, 1))                          # 叠加噪声并裁剪到 [0,1]
        lab.append(i)                                                  # 记录该样本的类别编号
imgs = np.stack(imgs)                                                  # 堆叠成 (12, 32, 32) 的样本张量
np.save(DAT / "image_samples.npy", imgs)                               # 保存为 npy 样本（机器学习常用格式）
np.save(DAT / "image_labels.npy", np.array(lab))                       # 保存标签数组

grid = imgs.reshape(3, 4, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 4 * W)  # 拼成 3 行 4 列的马赛克大图
plt.imsave(FIG / "fig_image_montage_raw.png", grid, cmap="gray")       # 把马赛克另存为一张 PNG 样本

# ---------------------------------------------------------------- 2) 彩色图像与通道分解
rgb = np.zeros((H, W, 3))                                              # 初始化一张彩色图（RGB 三通道）
rgb[..., 0] = xx / (W - 1)                                             # 红通道：从左到右线性增加
rgb[..., 1] = yy / (H - 1)                                             # 绿通道：从上到下线性增加
rgb[..., 2] = 0.8 * circle                                             # 蓝通道：只在圆内为高亮

# ---------------------------------------------------------------- 3) 可视化
fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.9))                    # 一行四列：灰度样本/原色/通道/RGB 直方图
axes[0].imshow(grid, cmap="gray")                                      # 左一：灰度样本马赛克（3 类 × 4 张）
axes[0].set_title("灰度图像样本 32×32（3 类 × 4 张）")                   # 左一标题
axes[0].set_xticks([]); axes[0].set_yticks([])                          # 隐藏坐标刻度，更像“图片”
axes[0].grid(False)                                                    # 关闭网格
axes[1].imshow(rgb)                                                    # 左二：彩色图像
axes[1].set_title("彩色图像 = R/G/B 三个矩阵")                          # 左二标题
axes[1].set_xticks([]); axes[1].set_yticks([])                          # 隐藏刻度
axes[1].grid(False)                                                    # 关闭网格
axes[2].imshow(circle, cmap="gray")                                    # 左三：单个通道（灰度矩阵）
axes[2].set_title("单通道 I(x, y) ∈ [0, 255]")                         # 左三标题
axes[2].set_xticks([]); axes[2].set_yticks([])                          # 隐藏刻度
axes[2].grid(False)                                                    # 关闭网格
axes[3].hist((imgs * 255).ravel(), bins=32, color="#8A0C3C", alpha=0.85)  # 右一：像素灰度直方图
axes[3].set_title("像素强度分布")                                       # 右一标题
axes[3].set_xlabel("灰度值")                                           # 右一横轴含义
plt.tight_layout()                                                     # 调整布局
save(fig, "fig_image.png")                                             # 保存图片
