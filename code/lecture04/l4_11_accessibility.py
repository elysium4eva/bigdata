"""l4_11_accessibility.py -- colorblind-friendly and accessible design: palettes, grayscale checks, contrast.

Accessibility is not a nice-to-have: about 8% of men have some form of color vision deficiency,
and grayscale printing erases color differences. The script compares three palettes, simulates grayscale,
computes WCAG contrast between text and background, and writes the checks as a list.
"""
import numpy as np                                                 # numeric computation
import pandas as pd                                                # data tables
import matplotlib.pyplot as plt                                    # comparison figures
import matplotlib.colors as mcolors                                # color space conversion (to grayscale)
from _common_l4 import OKABE, PALETTE, DAT, save, dump             # reuse shared helpers and both palettes

counts = pd.read_csv(DAT / "l4_category_counts.csv")               # category counts (from l4_05)
labels = counts["category"].tolist()                               # category names
values = counts["count"].tolist()                                  # counts per category

# ---------------------------------------------------------------- 1) three palettes
palettes = {                                                       # the three palettes to compare
    "matplotlib 默认": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"],
    "深大主题色": PALETTE[:len(labels)],                            # the course brand palette
    "Okabe–Ito（色盲友好）": OKABE[:len(labels)],                   # the standard colorblind-friendly palette
}                                                                 # end of the palette dict
def to_gray(hex_color):                                            # convert a color to grayscale luminance
    """Compute perceived luminance with ITU-R BT.601 weights (0-1)."""   # function contract
    r, g, b = mcolors.to_rgb(hex_color)                            # convert to RGB in 0-1
    return 0.299 * r + 0.587 * g + 0.114 * b                       # weighted sum gives the luminance
def contrast_ratio(fg, bg):                                        # compute the WCAG contrast ratio
    """Return the contrast between foreground and background (1-21; 4.5 passes for body text)."""   # function contract
    l1, l2 = to_gray(fg), to_gray(bg)                              # luminance of both colors
    hi, lo = max(l1, l2), min(l1, l2)                              # the lighter and the darker one
    return (hi + 0.05) / (lo + 0.05)                               # WCAG formula
report_rows = []                                                   # accessibility check table
for name, pal in palettes.items():                                 # evaluate each palette
    grays = [to_gray(c) for c in pal[:len(labels)]]                 # convert each color to grayscale
    gaps = np.diff(np.sort(grays))                                 # sorted adjacent luminance gaps
    min_gap = float(gaps.min()) if len(gaps) else 0.0              # smallest luminance gap (larger is easier to tell apart)
    contrasts = [contrast_ratio(c, "#FFFFFF") for c in pal[:len(labels)]]   # contrast against a white background
    report_rows.append({"色板": name,                                  # palette name
                        "最小灰度间距": round(min_gap, 3),              # grayscale separability
                        "与白底最低对比度": round(min(contrasts), 2),    # contrast floor
                        "正文达标(≥4.5)": bool(min(contrasts) >= 4.5),   # WCAG body text threshold
                        "可用作文字色": bool(min(contrasts) >= 4.5)})    # suitable as a text color
report = pd.DataFrame(report_rows)                                 # collect into a table
dump(report, "l4_accessibility_report.csv")                        # write the check table
print(report.to_string(index=False))                               # print to the terminal

# ---------------------------------------------------------------- 2) figure: palette, grayscale and contrast
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), gridspec_kw={"width_ratios": [1.7, 1.2, 1.1]})   # three panels: palette vs grayscale / luminance spread / contrast matrix
y_pos = np.arange(len(palettes))[::-1]                             # one row per palette
for i, (name, pal) in enumerate(palettes.items()):                 # first panel: color swatches plus grayscale swatches
    for j, c in enumerate(pal[:len(labels)]):                      # draw every color square
        axes[0].add_patch(plt.Rectangle((j, y_pos[i] + 0.35), 0.9, 0.3,   # color swatch
                                        color=c, transform=axes[0].transData))   # color
        gray = to_gray(c)                                          # grayscale value of that color
        axes[0].add_patch(plt.Rectangle((j, y_pos[i] - 0.05), 0.9, 0.3,   # grayscale swatch
                                        color=(gray, gray, gray), transform=axes[0].transData))   # grayscale value
axes[0].set_xlim(-0.2, len(labels) + 0.2); axes[0].set_ylim(-0.3, len(palettes) - 0.3)   # coordinate range
axes[0].set_xticks([j + 0.45 for j in range(len(labels))], labels, rotation=25, fontsize=8)   # category labels
axes[0].set_yticks(y_pos + 0.3, list(palettes.keys()), fontsize=8.6)   # palette names
axes[0].grid(False)                                                # no grid needed for swatches
axes[0].set_title("上排原色、下排灰度模拟：灰度分不开就会误导色盲读者")   # subplot title
x = np.arange(len(labels))                                         # x coordinates
width = 0.26                                                       # bar width
for k, (name, pal) in enumerate(palettes.items()):                 # second panel: luminance gap comparison
    grays = [to_gray(c) for c in pal[:len(labels)]]                 # luminance of each color
    axes[1].bar(x + (k - 1) * width, grays, width=width,            # grouped bars
                color=pal[:len(labels)], alpha=0.9, label=name)     # fill with the color itself
axes[1].set_xticks(x, labels, rotation=25, fontsize=8)              # category labels
axes[1].set_ylabel("感知亮度（0—1）")                               # axis label
axes[1].set_title("灰度亮度越分散越易区分")                          # subplot title
axes[1].legend(fontsize=7.6)                                       # legend
bg_list = ["#FFFFFF", "#F7F3F2", "#1E2630"]                        # three backgrounds
fg_list = ["#8A0C3C", "#66717D", "#C9A227", "#2E6F8E"]             # four foregrounds
matrix = np.array([[contrast_ratio(f, b) for b in bg_list] for f in fg_list])   # contrast matrix
im = axes[2].imshow(matrix, cmap="RdPu", vmin=1, vmax=21)          # third panel: contrast heatmap
axes[2].set_xticks(range(len(bg_list)), ["白", "浅底", "深底"], fontsize=8.6)   # background labels
axes[2].set_yticks(range(len(fg_list)), fg_list, fontsize=8.6)     # foreground labels
for i in range(matrix.shape[0]):                                   # write the contrast values in the cells
    for j in range(matrix.shape[1]):                               # double loop
        axes[2].text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8.6,   # write each value in the contrast matrix
                     color="white" if matrix[i, j] > 10 else "#1E2630")   # value label
axes[2].grid(False)                                                # no grid needed on a heatmap
axes[2].set_title("WCAG 对比度（≥4.5 达标）")                        # subplot title
plt.colorbar(im, ax=axes[2], fraction=0.046)                       # color bar
plt.tight_layout()                                                 # adjust the three panels
save(fig, "fig_palette_accessibility.png")                         # save: accessibility comparison

# ---------------------------------------------------------------- 3) checklist
CHECKLIST = [                                                      # accessibility self-check list
    "色板选色盲友好（Okabe–Ito / viridis 等），不要红绿对比",                      # item 1: palette choice
    "颜色之外再加冗余编码（形状、线型、图案、直接标注）",                                  # item 2: redundant encoding
    "灰度打印后仍能区分（本页第二格检查）",                                         # item 3: still separable in grayscale
    "文字与背景对比度 ≥ 4.5（大字号 ≥ 3.0）",                                  # item 4: contrast threshold
    "字号：投影正文 ≥ 18pt、图表标注 ≥ 12pt",                                 # item 5: minimum font size
    "不要只靠颜色传达涨跌或好坏（红绿组合最有问题）",                                    # item 6: do not signal good or bad by color alone
    "为每张图提供 alt 文本（一句话讲清结论）",                                     # item 7: provide alt text
    "交互图提供键盘可达的筛选与缩放",                                            # item 8: keyboard-accessible filtering and zoom
]                                                                 # end of the list
dump(pd.DataFrame({"检查项": CHECKLIST}), "l4_accessibility_checklist.csv")   # write the checklist
print("检查清单条数:", len(CHECKLIST))                              # print the list size
print("红绿对比的灰度差（示例）:", round(abs(to_gray("#8A0C3C") - to_gray("#2E6F8E")), 3))   # example value
