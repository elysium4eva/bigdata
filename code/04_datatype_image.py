"""04_datatype_image.py -- image data: synthetic sample images and pixel-level plots."""  # module comment
import numpy as np                                                    # numeric computation, an image is a matrix
import matplotlib.pyplot as plt                                       # plotting
from _common import FIG, DAT, save                                    # shared helpers, we write image samples directly

# ---------------------------------------------------------------- 1) build three classes of synthetic 32x32 grey images
H = W = 32                                                            # image height and width: 32 pixels for the lecture
yy, xx = np.mgrid[0:H, 0:W]                                           # build the pixel coordinate grid
cy, cx = (H - 1) / 2, (W - 1) / 2                                      # compute the image centre coordinates
circle = (((yy - cy) ** 2 + (xx - cx) ** 2) < 9 ** 2).astype(float)    # class 0: solid circle of radius 9
square = ((np.abs(yy - cy) <= 8) & (np.abs(xx - cx) <= 8)).astype(float)  # class 1: square with side 17
triangle = ((xx >= yy) & (xx <= W - 1 - yy)).astype(float)             # class 2: isosceles triangle, pointed top
protos = [circle, square, triangle]                                    # prototype pattern of each class
labels = ["circle", "square", "triangle"]                              # names of the three classes

imgs, lab = [], []                                                     # initialise the image and label lists
for i, p in enumerate(protos):                                         # loop over the three classes
    for k in range(4):                                                 # generate 4 samples per class
        noise = np.random.default_rng(100 + i * 10 + k).normal(0, 0.08, (H, W))  # add noise to mimic real capture
        imgs.append(np.clip(p + noise, 0, 1))                          # add the noise and clip into [0,1]
        lab.append(i)                                                  # record the class index of this sample
imgs = np.stack(imgs)                                                  # stack into a (12, 32, 32) tensor
np.save(DAT / "image_samples.npy", imgs)                               # save as npy, a common machine learning format
np.save(DAT / "image_labels.npy", np.array(lab))                       # save the label array

grid = imgs.reshape(3, 4, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 4 * W)  # tile into a 3 by 4 montage
plt.imsave(FIG / "fig_image_montage_raw.png", grid, cmap="gray")       # also save the montage as a PNG sample

# ---------------------------------------------------------------- 2) colour image and channel decomposition
rgb = np.zeros((H, W, 3))                                              # initialise a colour image with RGB channels
rgb[..., 0] = xx / (W - 1)                                             # red channel rises from left to right
rgb[..., 1] = yy / (H - 1)                                             # green channel rises from top to bottom
rgb[..., 2] = 0.8 * circle                                             # blue channel is bright only inside the circle

# ---------------------------------------------------------------- 3) visualization
fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.9))                    # one row, four columns: grey / colour / channel / histogram
axes[0].imshow(grid, cmap="gray")                                      # first: grey montage, 3 classes by 4 images
axes[0].set_title("灰度图像样本 32×32（3 类 × 4 张）")                   # first panel title
axes[0].set_xticks([]); axes[0].set_yticks([])                          # hide ticks so it looks like a picture
axes[0].grid(False)                                                    # turn off the grid
axes[1].imshow(rgb)                                                    # second: the colour image
axes[1].set_title("彩色图像 = R/G/B 三个矩阵")                          # second panel title
axes[1].set_xticks([]); axes[1].set_yticks([])                          # hide the ticks
axes[1].grid(False)                                                    # turn off the grid
axes[2].imshow(circle, cmap="gray")                                    # third: a single channel, a grey matrix
axes[2].set_title("单通道 I(x, y) ∈ [0, 255]")                         # third panel title
axes[2].set_xticks([]); axes[2].set_yticks([])                          # hide the ticks
axes[2].grid(False)                                                    # turn off the grid
axes[3].hist((imgs * 255).ravel(), bins=32, color="#8A0C3C", alpha=0.85)  # fourth: histogram of pixel intensities
axes[3].set_title("像素强度分布")                                       # fourth panel title
axes[3].set_xlabel("灰度值")                                           # fourth panel x label
plt.tight_layout()                                                     # adjust the layout
save(fig, "fig_image.png")                                             # save the figure
