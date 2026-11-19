"""l8_15_diffusion_letters.py -- a tiny DDPM that draws the letters A / B / S in the plane.

The training cloud is sampled analytically around three letter strokes (about 1440 points),
a 2 -> 192 -> 192 -> 192 -> 2 SiLU network learns to predict the noise epsilon given (x_t, t), and
80 reverse steps carry pure noise back onto the letters.  The script writes the panel
figure, 24 animation frames, the animated GIF, two tidy CSV tables, and copies the images
into the beamer folder lecture08-2026-szu/figures.

Run:  python l8_15_diffusion_letters.py       (about 65 s on a CPU laptop)
"""
import shutil                                                      # copy artefacts into the beamer folder
import time                                                        # measure the wall time of the run
from pathlib import Path                                           # portable output paths

import matplotlib                                                  # select the backend before pyplot
matplotlib.use("Agg")                                              # headless backend: no display needed
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numeric arrays and random numbers
import pandas as pd                                                # tidy CSV tables
import torch                                                       # the tiny diffusion network
import torch.nn as nn                                              # linear layers and activations
from matplotlib.transforms import Bbox                             # exact pixel box for the frames
from PIL import Image                                              # assemble the animated GIF
from scipy.spatial import cKDTree                                  # nearest-neighbour distance

SEED = 20261119                                                    # one seed for every random draw
np.random.seed(SEED)                                               # seed the numpy global generator
torch.manual_seed(SEED)                                            # seed the torch global generator
RNG = np.random.default_rng(SEED)                                  # explicit generator for the data

ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above this file)
FIG = ROOT / "figures" / "lecture08"                               # figure output folder
DAT = ROOT / "data" / "lecture08"                                  # data output folder
ANIM = FIG / "anim_diffusion"                                      # one PNG per sampling step
DECK = ROOT.parent / "lecture08-2026-szu" / "figures"              # the beamer copy target
for _d in (FIG, DAT, ANIM, DECK):                                  # make sure every folder exists
    _d.mkdir(parents=True, exist_ok=True)                          # create it when missing

# ---------------------------------------------------------------- house style
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK-capable font stack
plt.rcParams["axes.unicode_minus"] = False                          # draw real minus signs, not tofu boxes
plt.rcParams["savefig.dpi"] = 200                                   # high resolution for the slides
plt.rcParams["savefig.bbox"] = "tight"                              # trim white space around the panels
plt.rcParams["axes.grid"] = True                                    # light grid helps reading the canvas
plt.rcParams["grid.alpha"] = 0.22                                   # keep the grid subtle
plt.rcParams["axes.spines.top"] = False                             # drop the top spine (chart junk)
plt.rcParams["axes.spines.right"] = False                           # drop the right spine as well
plt.rcParams["axes.titlesize"] = 11.5                               # subplot title size
plt.rcParams["axes.labelsize"] = 10.0                               # axis label size
plt.rcParams["xtick.labelsize"] = 8.6                               # x tick label size
plt.rcParams["ytick.labelsize"] = 8.6                               # y tick label size
plt.rcParams["legend.fontsize"] = 8.6                               # legend text size

OKABE = {"A": "#E69F00", "B": "#56B4E9", "S": "#009E73"}            # one colour-blind safe colour per letter
INK = "#0072B2"                                                     # default colour of a single-colour cloud
GREY = "#8A939E"                                                    # colour of the reference cloud
XLIM, YLIM = (-3.15, 3.15), (-1.65, 1.65)                           # fixed canvas for every panel and frame

# ---------------------------------------------------------------- 1) letter geometry
NOISE = 0.035                                                       # stroke thickness of the sampled points


def seg(p, q, n):                                                   # sample a straight stroke segment
    """Return n points evenly spaced from p to q."""                     # short docstring
    t = np.linspace(0.0, 1.0, n)[:, None]                           # parameter running from 0 to 1
    a, b = np.asarray(p, float)[None, :], np.asarray(q, float)[None, :]   # endpoints as row vectors
    return a + t * (b - a)                                          # linear interpolation along the segment


def arc(cx, cy, rx, ry, a0, a1, n):                                 # sample an elliptical arc
    """Return n points along an ellipse arc, angles given in degrees."""   # short docstring
    a = np.deg2rad(np.linspace(a0, a1, n))                          # dense angles in radians
    return np.c_[cx + rx * np.cos(a), cy + ry * np.sin(a)]          # analytic ellipse, x and y side by side


def strokes(letter):                                                # analytic strokes of one letter
    """Return the dense polylines (as point arrays) that make up A, B or S."""   # short docstring
    if letter == "A":                                               # the letter A
        left = seg([-0.45, -1.0], [0.00, 1.0], 400)                 # left leg
        right = seg([0.45, -1.0], [0.00, 1.0], 400)                 # right leg
        bar = seg([-0.27, -0.18], [0.27, -0.18], 260)               # crossbar between the legs
        return [left, right, bar]                                   # three strokes in total
    if letter == "B":                                               # the letter B
        stem = seg([-0.42, -1.0], [-0.42, 1.0], 400)                # the vertical stem
        upper = arc(-0.42, 0.47, 0.40, 0.47, 90, -90, 300)          # upper bowl, right half of an ellipse
        lower = arc(-0.42, -0.50, 0.46, 0.50, 90, -90, 300)         # lower bowl, a little wider
        return [stem, upper, lower]                                 # three strokes in total
    top = arc(0.0, 0.45, 0.42, 0.42, 30, 210, 300)                  # upper hook of the S
    diag = seg([-0.364, 0.24], [0.364, -0.24], 200)                 # the middle diagonal
    bottom = arc(0.0, -0.45, 0.42, 0.42, 30, -150, 300)             # lower hook of the S
    return [top, diag, bottom]                                      # three strokes in total


def letter_cloud(letter, centre_x, n_points=480):                   # one letter as a noisy point cloud
    """Return n_points sampled along the strokes, shifted to centre_x."""   # short docstring
    polys = strokes(letter)                                         # the dense analytic polylines
    lens = np.array([np.hypot(*np.diff(p, axis=0).T).sum() for p in polys])   # polyline length per stroke
    alloc = np.maximum(4, np.round(n_points * lens / lens.sum()).astype(int))   # points proportional to length
    parts = []                                                      # sampled pieces, one per stroke
    for poly, k in zip(polys, alloc):                               # walk the strokes of this letter
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poly, axis=0).T))])   # cumulative length
        s = cum / cum[-1]                                           # normalised arc-length parameter
        u = np.linspace(0.0, 1.0, k)                                # k evenly spaced positions
        parts.append(np.c_[np.interp(u, s, poly[:, 0]),             # x coordinate along the stroke
                           np.interp(u, s, poly[:, 1])])            # y coordinate along the stroke
    P = np.vstack(parts)                                            # all strokes of the letter together
    P = P + RNG.normal(0.0, NOISE, P.shape)                         # small isotropic stroke noise
    P[:, 0] += centre_x                                             # shift the letter into its slot
    return P                                                        # the finished letter cloud


CENTRES = {"A": -2.05, "B": 0.0, "S": 2.05}                         # three letters side by side
clouds = {k: letter_cloud(k, v) for k, v in CENTRES.items()}        # 480 points per letter
LABELS = np.concatenate([np.full(len(clouds[k]), k) for k in CENTRES])   # letter label per point
DATA = np.vstack([clouds[k] for k in CENTRES])                      # the full training cloud, about 1440 x 2
PD = pd.DataFrame({"letter": LABELS,                                # the letter each point belongs to
                   "x": np.round(DATA[:, 0], 4),                    # x coordinate in canvas units
                   "y": np.round(DATA[:, 1], 4)})                   # y coordinate in canvas units
PD.to_csv(DAT / "l8_diffusion_points.csv", index=False, encoding="utf-8-sig")   # tidy training table
print("[data]", DAT / "l8_diffusion_points.csv", "| points:", len(PD))   # report the table

# ---------------------------------------------------------------- 2) a tiny epsilon network
SCALE = 2.0                                                         # divide the canvas by 2 before training
X0 = DATA / SCALE                                                   # data in roughly unit scale
Z0 = torch.tensor(X0, dtype=torch.float32)                          # the same cloud as a tensor
N = Z0.shape[0]                                                     # number of training points


class EpsMLP(nn.Module):                                            # the noise-prediction network
    """A 2 -> 192 -> 192 -> 192 -> 2 SiLU MLP with a sinusoidal time embedding."""   # short docstring
    def __init__(self, width=192, dim=32):                          # width of the trunk, size of the embedding
        super().__init__()                                          # initialise the module base class
        self.dim = dim                                              # number of embedding channels
        half = dim // 2                                             # half sine, half cosine
        f = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)   # frequency ladder
        self.register_buffer("freqs", f)                            # keep the ladder with the module
        self.net = nn.Sequential(nn.Linear(2 + dim, width), nn.SiLU(),   # input layer and activation
                                 nn.Linear(width, width), nn.SiLU(),  # hidden layer one
                                 nn.Linear(width, width), nn.SiLU(),  # hidden layer two
                                 nn.Linear(width, 2))               # epsilon output, two coordinates

    def embed(self, t):                                             # sinusoidal features of the time stamp
        ang = t[:, None] * self.freqs[None, :] * 2.0 * float(np.pi)   # angle per frequency channel
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)   # sine and cosine side by side

    def forward(self, x, t):                                        # predict epsilon from (x_t, t)
        return self.net(torch.cat([x, self.embed(t)], dim=1))       # concatenate the inputs and run the trunk


# ---------------------------------------------------------------- 3) the forward process
T = 80                                                              # number of diffusion steps
STEPS = 10000                                                       # Adam steps, full batch
betas = torch.linspace(0.001, 0.20, T)                              # linear beta schedule
alphas = 1.0 - betas                                                # alpha_t = 1 - beta_t
abar = torch.cumprod(alphas, dim=0)                                 # alpha bar, the cumulative product
print("[schedule] T=%d | beta %.3f..%.3f | alpha_bar(T)=%.1e"       # report the noise schedule
      % (T, float(betas[0]), float(betas[-1]), float(abar[-1])))    # the last value must be near zero

model = EpsMLP()                                                    # the network to be trained
opt = torch.optim.Adam(model.parameters(), lr=3.0e-3)               # Adam, a generous learning rate
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1.0e-4)   # cosine decay to 1e-4
t_start = time.time()                                               # start the wall clock
for step in range(STEPS):                                           # full-batch training loop
    idx = torch.randint(0, T, (N,))                                 # one random timestep per point
    eps = torch.randn_like(Z0)                                      # the noise the network must recover
    a = abar[idx].unsqueeze(1)                                      # alpha bar per point, column vector
    xt = a.sqrt() * Z0 + (1.0 - a).sqrt() * eps                     # the forward diffusion sample
    pred = model(xt, idx.float() / (T - 1))                         # predicted epsilon, time in [0, 1]
    loss = ((pred - eps) ** 2).mean()                               # the simple DDPM objective
    opt.zero_grad()                                                 # clear the previous gradients
    loss.backward()                                                 # back-propagate the loss
    opt.step()                                                      # one Adam update
    sched.step()                                                    # advance the learning-rate schedule
    if step % 500 == 0 or step == STEPS - 1:                        # report occasionally, not every step
        print("  step %4d | loss %.4f" % (step, float(loss.detach())))        # the running training loss
train_time = time.time() - t_start                                  # training wall time in seconds

# ---------------------------------------------------------------- 4) the reverse process
x = torch.randn(N, 2)                                               # start from pure Gaussian noise
states = [x.clone()]                                                # state 0 is x_T, diffusion time T
with torch.no_grad():                                               # no gradients while sampling
    for t in reversed(range(T)):                                    # walk the chain from T-1 down to 0
        tt = torch.full((N,), t / (T - 1))                          # normalised time for this step
        eps_hat = model(x, tt)                                      # the network's noise estimate
        beta_t, alpha_t, abar_t = betas[t], alphas[t], abar[t]      # the three schedule values
        mean = (x - (beta_t / (1.0 - abar_t).sqrt()) * eps_hat) / alpha_t.sqrt()   # posterior mean
        if t > 0:                                                   # every step but the last one
            var = beta_t * (1.0 - abar[t - 1]) / (1.0 - abar_t)      # DDPM posterior variance
            x = mean + var.sqrt() * torch.randn(N, 2)               # one stochastic reverse step
        else:                                                       # the final step
            x = mean                                                # deterministic finish at t = 0
        states.append(x.clone())                                    # keep the full cloud, index T - t

FRAMES = 24                                                         # number of saved states and animation frames
pick = np.round(np.linspace(0, T, FRAMES)).astype(int)              # evenly spaced state indices, step > 1
assert len(set(pick.tolist())) == FRAMES                            # the spacing guarantees unique indices
tree = cKDTree(DATA)                                                # training cloud for the metric
rows = []                                                           # one row per saved state
for k in pick:                                                      # walk the saved states in time order
    cloud = states[k].numpy() * SCALE                               # back to canvas units
    dist = tree.query(cloud)[0].mean()                              # mean distance to the nearest real point
    rows.append({"step": int(T - k), "mean_nn_distance": round(float(dist), 4)})   # tidy row
trace = pd.DataFrame(rows)                                          # the distance trace
trace.to_csv(DAT / "l8_diffusion_trace.csv", index=False, encoding="utf-8-sig")   # tidy trace table
nn_start, nn_final = float(trace["mean_nn_distance"].iloc[0]), float(trace["mean_nn_distance"].iloc[-1])   # quote
print("[data]", DAT / "l8_diffusion_trace.csv", "| NN start %.3f -> final %.3f" % (nn_start, nn_final))   # report


def letter_colours(cloud, labels=None):                             # colour points by the nearest letter
    """Return one palette colour per point, from true or nearest-letter labels."""   # short docstring
    lab = LABELS[tree.query(cloud)[1]] if labels is None else labels   # nearest training letter by default
    return [OKABE[k] for k in lab]                                  # map every label to its colour


def show(ax, cloud, title, colour=INK, size=7.0, alpha=0.82):       # draw one cloud in house style
    """Plot a point cloud with the fixed canvas and a short title."""    # short docstring
    ax.scatter(cloud[:, 0], cloud[:, 1], s=size, c=colour, alpha=alpha, linewidths=0)   # the points
    ax.set_xlim(*XLIM)                                              # fixed horizontal range
    ax.set_ylim(*YLIM)                                              # fixed vertical range
    ax.set_aspect("equal")                                          # one data unit in x equals one in y
    ax.set_title(title)                                             # the panel or frame label
    return ax                                                       # return the axes for chaining


# ---------------------------------------------------------------- 5) panel figure
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))                 # six panels in two rows
ax = axes[0, 0]                                                     # panel (a)
for name in CENTRES:                                                # one scatter call per letter
    c = clouds[name]                                                # that letter's points
    ax.scatter(c[:, 0], c[:, 1], s=6, color=OKABE[name], alpha=0.85, linewidths=0, label="字母 " + name)   # draw
ax.set_xlim(*XLIM)                                                  # fixed horizontal range
ax.set_ylim(*YLIM)                                                  # fixed vertical range
ax.set_aspect("equal")                                              # keep the letter shapes undistorted
ax.set_title("（a）训练数据：A、B、S")                                # panel title
ax.legend(loc="upper right", frameon=True, framealpha=0.9)          # colour key for the three letters
a_mid = float(abar[T // 2])                                         # alpha bar at t = T / 2
noisy = np.sqrt(a_mid) * X0 + np.sqrt(1.0 - a_mid) * RNG.normal(0.0, 1.0, X0.shape)   # forward noising
show(axes[0, 1], noisy * SCALE, "（b）前向加噪 t = %d（T/2）" % (T // 2), letter_colours(noisy * SCALE, LABELS), 7.0, 0.62)   # panel (b)
show(axes[0, 2], states[T - 20].numpy() * SCALE, "（c）反向采样 t = 20", letter_colours(states[T - 20].numpy() * SCALE))   # panel (c)
show(axes[1, 0], states[T - 10].numpy() * SCALE, "（d）反向采样 t = 10", letter_colours(states[T - 10].numpy() * SCALE))   # panel (d)
show(axes[1, 1], states[T - 3].numpy() * SCALE, "（e）反向采样 t = 3", letter_colours(states[T - 3].numpy() * SCALE))   # panel (e)
final = states[T].numpy() * SCALE                                   # the generated cloud at t = 0
show(axes[1, 2], final, "（f）生成结果 t = 0", letter_colours(final))   # panel (f)
axes[1, 2].scatter(DATA[:, 0], DATA[:, 1], s=5, facecolors="none", edgecolors=GREY, linewidths=0.4, alpha=0.55)   # reference
axes[1, 2].text(0.02, 0.04, "平均最近邻距离 %.3f" % nn_final, transform=axes[1, 2].transAxes, fontsize=9.0,   # corner note
                color="#1E2630", bbox=dict(boxstyle="round,pad=0.25", fc="#F7F3F2", ec="#D6CDD0", lw=0.6))   # note
for a_ in axes.ravel():                                             # every panel gets the same canvas
    a_.set_xlabel("x")                                              # horizontal axis label
    a_.set_ylabel("y")                                              # vertical axis label
fig.suptitle("高斯扩散模型（DDPM）从纯噪声采样出字母点云：T = %d 步，颜色为最近的训练字母" % T, fontsize=13.5)   # figure title
fig.tight_layout()                                                  # tidy the layout
fig.savefig(FIG / "fig_diffusion_letters.png", dpi=200, bbox_inches="tight", facecolor="white")   # save the panels
plt.close(fig)                                                      # release the figure
print("[figure]", FIG / "fig_diffusion_letters.png")                # report the figure

# ---------------------------------------------------------------- 6) animation frames
for stale in sorted(ANIM.glob("frame-*.png")):                      # old frames would break the numbering
    stale.unlink()                                                  # remove them before the new run
FRAME_IN = 3.2                                                      # 3.2 inch at 200 dpi is exactly 640 pixel
FULLBOX = Bbox.from_bounds(0.0, 0.0, FRAME_IN, FRAME_IN)            # a full-canvas box gives a fixed size
fig_a, ax_a = plt.subplots(figsize=(FRAME_IN, FRAME_IN))            # one reusable frame canvas
fig_a.subplots_adjust(left=0.155, right=0.965, bottom=0.135, top=0.865)   # identical margins in every frame
frame_paths = []                                                    # collect the written files in order
for i, k in enumerate(pick):                                        # one frame per saved state
    ax_a.clear()                                                    # reset the axes for the next step
    show(ax_a, states[k].numpy() * SCALE, "反向采样 t = %d" % (T - k), letter_colours(states[k].numpy() * SCALE), 6.0, 0.78)   # the current cloud
    ax_a.text(0.98, 0.04, "frame %02d / %02d" % (i, FRAMES - 1), transform=ax_a.transAxes, ha="right",   # corner counter
              fontsize=8.5, color="#4A5560")                        # frame counter in the corner
    ax_a.set_xlabel("x")                                            # horizontal axis label
    ax_a.set_ylabel("y")                                            # vertical axis label
    path = ANIM / ("frame-%d.png" % i)                              # contiguous numbering from 0
    fig_a.savefig(path, dpi=200, bbox_inches=FULLBOX, facecolor="white")   # exactly 640 x 640 pixel
    if Image.open(path).size != (640, 640):                         # guard against an unexpected canvas
        Image.open(path).resize((640, 640)).save(path)              # force the nominal frame size
    frame_paths.append(path)                                        # remember the file name
plt.close(fig_a)                                                    # release the frame canvas
print("[frames]", ANIM, "|", len(frame_paths), "frames at 640 x 640")   # report the frame count

# ---------------------------------------------------------------- 7) animated GIF and the deck copies
imgs = [Image.open(p).convert("RGB") for p in frame_paths]          # load every frame in order
gif = FIG / "fig_diffusion.gif"                                     # the animated file next to the figures
imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=120, loop=0, optimize=False)   # 120 ms/frame
print("[gif]", gif, "| frames:", len(imgs))                         # report the animated file
shutil.copy2(FIG / "fig_diffusion_letters.png", DECK / "fig_diffusion_letters.png")   # panel figure to the deck
shutil.copy2(gif, DECK / "fig_diffusion.gif")                       # the GIF too, the deck may use either
deck_anim = DECK / "anim_diffusion"                                 # the deck frame folder
deck_anim.mkdir(parents=True, exist_ok=True)                        # create it when missing
for p in frame_paths:                                               # copy every frame, same file names
    shutil.copy2(p, deck_anim / p.name)                             # keep the numbering identical
print("[copy]", DECK, "| 1 png + 1 gif +", len(frame_paths), "frames")   # report the copied artefacts

# ---------------------------------------------------------------- 8) summary
wall = time.time() - t_start                                        # total wall time of the whole run
print("[summary] T=%d steps | Adam steps=%d | NN start %.3f -> final %.3f | frames=%d | %.1fs"   # summary
      % (T, STEPS, nn_start, nn_final, len(frame_paths), wall))   # fill the summary
print("done: diffusion letters, panel figure, %d frames and the GIF" % len(frame_paths))   # closing log
