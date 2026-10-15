"""06_datatype_video.py -- video data: synthetic frame sequence plus key frame and signal plots."""  # module comment
import numpy as np                                                    # numeric computation
import matplotlib.pyplot as plt                                       # plotting
from _common import DAT, save                                         # shared helpers, only the data folder and save

H = W = 48                                                            # frame resolution 48x48
NF = 12                                                               # 12 frames, like 1 second at 12 fps
frames, means = [], []                                                # initialise the frame list and per-frame mean brightness
for k in range(NF):                                                   # generate the video content frame by frame
    img = np.zeros((H, W))                                            # all black background
    yy, xx = np.mgrid[0:H, 0:W]                                       # pixel coordinate grid
    cy, cx = H / 2, 2 + k * (W - 4) / (NF - 1)                        # ball centre moves left to right over time
    ball = ((yy - cy) ** 2 + (xx - cx) ** 2) < 5 ** 2                 # mask of a ball with radius 5
    img[ball] = 220                                                   # set the ball pixels to a bright value
    bar_x = (W * k / NF)                                              # vertical light bar position moves over time
    img[:, int(bar_x):int(bar_x) + 3] += 60                           # brighten the light bar pixels
    frames.append(np.clip(img, 0, 255))                               # append to the frame sequence
    means.append(float(img[ball].mean()))                             # record the mean brightness inside the ball
frames = np.stack(frames)                                             # stack into a (12, 48, 48) video tensor
np.save(DAT / "video_frames.npy", frames)                             # save the frame tensor sample
(DAT / "video_frames").mkdir(parents=True, exist_ok=True)             # create the folder for extracted frames
for i in range(NF):                                                   # export each frame as PNG, like frame extraction
    plt.imsave(DAT / "video_frames" / ("frame_%02d.png" % (i + 1)),   # save frame i into data/video_frames/
               frames[i], cmap="gray")                                # save with the grey colour map
print("[data]", DAT / "video_frames.npy")                             # print the sample path

# ---------------------------------------------------------------- visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                   # one row, three columns: key frame / difference / signal
axes[0].imshow(frames[0], cmap="gray")                                # left: the first frame
axes[0].set_title("关键帧 t=1（48×48 灰度）")                          # left panel title
axes[0].set_xticks([]); axes[0].set_yticks([])                         # hide the ticks
axes[0].grid(False)                                                   # turn off the grid
axes[1].imshow(np.abs(frames[6] - frames[0]), cmap="inferno")          # middle: frame 7 minus frame 1, motion detection
axes[1].set_title("帧间差分 |I(t=7) − I(t=1)|（运动检测）")              # middle panel title
axes[1].set_xticks([]); axes[1].set_yticks([])                         # hide the ticks
axes[1].grid(False)                                                   # turn off the grid
axes[2].plot(range(1, NF + 1), means, "-o", color="#8A0C3C", ms=4)     # right: time series of per-frame mean brightness
axes[2].set_title("逐帧信号：亮度均值随 t 变化")                        # right panel title
axes[2].set_xlabel("帧序号 t")                                        # right panel x label
axes[2].set_ylabel("平均亮度")                                         # right panel y label
plt.tight_layout()                                                    # adjust the layout
save(fig, "fig_video.png")                                            # save the figure
