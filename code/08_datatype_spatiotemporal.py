"""08_datatype_spatiotemporal.py —— 时空数据：轨迹样本 + 热力图/速度曲线可视化。"""  # 模块注释
import numpy as np                                                    # 数值计算
import pandas as pd                                                   # 表格处理
import matplotlib.pyplot as plt                                       # 绘图
from _common import RNG, save, dump_csv                               # 公共工具

# ---------------------------------------------------------------- 1) 生成三条轨迹（深圳南山周边经纬度）
lat0, lon0 = 22.5333, 113.9300                                        # 起点：深圳南山区附近
tracks = ["粤B·A1234", "粤B·B5678", "粤B·C9012"]                       # 三个车辆/骑手编号
rows = []                                                             # 初始化逐点记录列表
for i, tid in enumerate(tracks):                                      # 逐条轨迹生成
    n = 120                                                           # 每条轨迹 120 个采样点（约 2 分钟）
    speed = RNG.normal(9.0 + 2 * i, 1.5, n).clip(2, 18)               # 速度 m/s：均值不同、带随机波动
    head = np.cumsum(RNG.normal(0, 0.12, n))                          # 航向角随机游走（模拟转弯）
    dlat = np.cos(head) * speed * 5 / 111000.0                        # 纬度增量：位移 / 每度约 111 km
    dlon = np.sin(head) * speed * 5 / (111000.0 * np.cos(np.radians(lat0)))  # 经度增量：需按纬度做余弦修正
    la = lat0 + np.cumsum(dlat) + RNG.normal(0, 1e-5, n)              # 累加得到纬度序列（加定位噪声）
    lo = lon0 + np.cumsum(dlon) + RNG.normal(0, 1e-5, n)              # 累加得到经度序列（加定位噪声）
    for k in range(n):                                                # 逐点写入记录
        rows.append({"track_id": tid, "t_sec": k * 5,                # 轨迹编号与时间戳（每 5 秒一个点）
                     "lat": round(float(la[k]), 6),                   # 纬度（保留 6 位小数，约 0.1 m）
                     "lon": round(float(lo[k]), 6),                   # 经度
                     "speed_mps": round(float(speed[k]), 2)})         # 瞬时速度
traj = pd.DataFrame(rows)                                             # 组装为长表（一行一个“点”）
dump_csv(traj, "spatiotemporal_tracks.csv")                           # 落盘时空样本

# ---------------------------------------------------------------- 2) 可视化
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9))                   # 一行三列：轨迹 / 密度热力 / 速度时序
for tid in tracks:                                                    # 逐条轨迹绘制
    sub = traj[traj["track_id"] == tid]                               # 取出该轨迹的子表
    axes[0].plot(sub["lon"], sub["lat"], "-", lw=1.4, label=tid)      # 左图：经纬度连线即轨迹
axes[0].set_title("轨迹（经纬度序列）")                                # 左图标题
axes[0].set_xlabel("经度 / °E")                                       # 左图横轴含义
axes[0].set_ylabel("纬度 / °N")                                       # 左图纵轴含义
axes[0].legend(fontsize=7)                                            # 显示车辆图例
axes[0].ticklabel_format(useOffset=False)                             # 关闭科学计数偏移，显示真实经纬度

hb = axes[1].hexbin(traj["lon"], traj["lat"], gridsize=18,            # 中图：六边形分箱热力图（空间密度）
                    cmap="magma", mincnt=1)                           # 只显示有点的格子
axes[1].set_title("空间密度热力图（hexbin）")                          # 中图标题
axes[1].set_xlabel("经度 / °E")                                       # 中图横轴含义
axes[1].ticklabel_format(useOffset=False)                             # 关闭科学计数偏移
axes[1].grid(False)                                                   # 热力图不需要网格

for tid in tracks:                                                    # 逐条轨迹绘制速度曲线
    sub = traj[traj["track_id"] == tid]                               # 取出该轨迹
    axes[2].plot(sub["t_sec"], sub["speed_mps"], lw=1.0,              # 右图：速度随时间的折线
                 label=tid)                                           # 加上图例标签
axes[2].set_title("速度随时间变化（时空→时序）")                        # 右图标题
axes[2].set_xlabel("时间 / 秒")                                       # 右图横轴含义
axes[2].set_ylabel("速度 / (米/秒)")                                  # 右图纵轴含义
plt.tight_layout()                                                    # 调整布局
save(fig, "fig_spatiotemporal.png")                                   # 保存图片
