"""05_datatype_audio.py —— 音频型数据：合成 WAV 样本 + 波形/频谱可视化。"""  # 模块注释
import wave                                                           # 标准库：写入 WAV 文件
import numpy as np                                                    # 数值计算
import matplotlib.pyplot as plt                                       # 绘图
from _common import DAT, save                                         # 公共工具

SR = 8000                                                             # 采样率 8 kHz（电话语音级别）
T = 1.0                                                               # 每个片段时长 1 秒
t = np.linspace(0, T, int(SR * T), endpoint=False)                    # 生成时间轴（每秒 8000 个采样点）
clips = {}                                                            # 用字典存放三个音频片段

clips["tone_440"] = 0.6 * np.sin(2 * np.pi * 440 * t)                 # 片段①：440 Hz 纯音（单一频率）
clips["chirp_200_2000"] = 0.6 * np.sin(2 * np.pi * (200 + 900 * t) * t)  # 片段②：扫频信号（频率随时间上升）
env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))                           # 片段③：3 Hz 幅度包络（模拟说话节奏）
clips["am_noise"] = env * np.random.default_rng(7).normal(0, 0.3, t.size)  # 片段③：被调制的噪声（类语音）

for name, x in clips.items():                                         # 遍历三个片段
    pcm = np.int16(np.clip(x, -1, 1) * 32767)                         # 浮点转 16 位整型 PCM（WAV 标准格式）
    path = DAT / ("audio_%s.wav" % name)                              # 拼接输出路径
    with wave.open(str(path), "wb") as w:                             # 以二进制写模式打开 WAV 文件
        w.setnchannels(1)                                             # 单声道
        w.setsampwidth(2)                                             # 每个采样 2 字节 = 16 bit
        w.setframerate(SR)                                            # 写入采样率
        w.writeframes(pcm.tobytes())                                  # 写入原始字节流
    print("[data]", path)                                             # 打印落盘位置

# ---------------------------------------------------------------- 可视化
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                   # 一行三列：波形 / 频谱 / 对比波形
axes[0].plot(t, clips["tone_440"], color="#8A0C3C", lw=0.8)           # 左图：纯音波形（时域）
axes[0].set_title("波形（时域）：440 Hz 纯音")                         # 左图标题
axes[0].set_xlabel("时间 / 秒")                                       # 左图横轴含义
axes[0].set_ylabel("振幅")                                            # 左图纵轴含义

axes[1].specgram(clips["chirp_200_2000"], NFFT=256, Fs=SR,           # 中图：短时傅里叶变换得到频谱图（时频）
                 cmap="magma", noverlap=128)                          # 设置重叠点数与配色
axes[1].set_title("频谱图（时频）：200→2000 Hz 扫频")                  # 中图标题
axes[1].set_xlabel("时间 / 秒")                                       # 中图横轴含义
axes[1].set_ylabel("频率 / Hz")                                       # 中图纵轴含义
axes[1].grid(False)                                                   # 频谱图不需要网格

axes[2].plot(t, clips["am_noise"], color="#66717D", lw=0.6)           # 右图：调幅噪声波形
axes[2].plot(t, env, color="#C9A227", lw=1.8, label="包络")           # 叠加包络线，展示“节奏”
axes[2].set_title("调幅噪声（模拟语音节奏）")                           # 右图标题
axes[2].set_xlabel("时间 / 秒")                                       # 右图横轴含义
axes[2].legend(fontsize=8)                                            # 显示图例
plt.tight_layout()                                                    # 调整布局
save(fig, "fig_audio.png")                                            # 保存图片
