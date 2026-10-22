"""render_heatmap.py -- draw a static treemap PNG from the stock-heatmap JSON.

Every line carries an English comment. The upstream repository renders an interactive Plotly
page; the courseware needs a static figure, so this script re-uses the same data payload and
lays it out with a squarified treemap (Bruls, Huizing & van Wijk, 2000) in matplotlib.
"""
import json                                                        # read the payload produced upstream
import sys                                                         # read the input path from argv
import matplotlib                                                  # choose a headless backend first
matplotlib.use("Agg")                                              # no display is available here
import matplotlib.pyplot as plt                                    # figure API
import matplotlib.patches as mpatches                              # rectangle patches
import matplotlib.colors as mcolors                                # colour maps and norms
import numpy as np                                                 # numeric helpers

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK-capable fonts
plt.rcParams["axes.unicode_minus"] = False                         # draw minus signs normally


def _ratio(row, length):                                           # worst aspect ratio of a row
    """Return the worst aspect ratio of a candidate row (lower is better)."""   # short docstring
    s = float(sum(row))                                            # total area of the row
    if s <= 0 or length <= 0:                                      # degenerate input
        return float("inf")                                        # never accept it
    return max(max(row) * length ** 2 / s ** 2, s ** 2 / (min(row) * length ** 2))   # the standard measure


def squarify(sizes, x, y, dx, dy):                                 # squarified treemap layout
    """Return rectangles [(x, y, w, h)] filling (x, y, dx, dy) proportionally to sizes."""   # short docstring
    if not sizes:                                                  # nothing to lay out
        return []                                                  # empty result
    total = float(sum(sizes))                                      # total area asked for
    if total <= 0 or dx <= 0 or dy <= 0:                           # degenerate input
        return [(x, y, dx, dy)] * len(sizes)                       # give everything the full box
    areas = [s * (dx * dy) / total for s in sizes]                  # scaled areas
    rects = []                                                     # the output rectangles
    while areas:                                                   # consume the list
        length = min(dx, dy)                                       # the shorter free side
        row = [areas[0]]                                           # start a new row
        rest = areas[1:]                                           # the remaining items
        while rest and _ratio(row + [rest[0]], length) <= _ratio(row, length):   # growing helps
            row.append(rest.pop(0))                                # keep growing the row
        area = sum(row)                                            # the row's total area
        if dx >= dy:                                               # lay the row out vertically
            w = area / dy if dy else 0                             # its width
            yy = y                                                 # cursor along y
            for r in row:                                          # one tile per item
                h = r / w if w else 0                              # its height
                rects.append((x, yy, w, h))                        # append the tile
                yy += h                                            # advance
            x += w                                                 # shrink the free box
            dx -= w                                                # horizontally
        else:                                                      # lay the row out horizontally
            h = area / dx if dx else 0                             # its height
            xx = x                                                 # cursor along x
            for r in row:                                          # one tile per item
                w = r / h if h else 0                              # its width
                rects.append((xx, y, w, h))                        # append the tile
                xx += w                                            # advance
            y += h                                                 # shrink the free box
            dy -= h                                                # vertically
        areas = rest                                               # continue with the rest
    return rects                                                   # all rectangles

def main(path):                                                     # build the two figures
    """Read the payload, draw the treemap and the distribution figure."""   # short docstring
    payload = json.load(open(path, encoding="utf-8"))               # the upstream payload
    root = payload["root"]                                          # the A-share root node
    PSEUDO = {"创业板", "北交所", "科创板", "沪市主板", "深市主板", "中小板"}   # buckets for unmapped stocks
    boards = [b for b in root.get("children", [])                     # boards with stocks
              if b.get("children") and b.get("n") not in PSEUDO]      # keep real industries only
    boards.sort(key=lambda b: -b.get("v", 0))                       # largest turnover first
    meta = payload.get("title_meta", {})                            # caption metadata
    date = payload.get("trading_date", "")                          # trading date
    status = payload.get("market_status", "")                       # market status

    # ------------------------------------------------------------ colour scale
    lo, hi = -0.05, 0.05                                            # clip the colour scale at 5%
    cmap = mcolors.LinearSegmentedColormap.from_list(               # green -> neutral -> red
        "szu_pnl", ["#0E7C4A", "#5DB37E", "#F4F1EC", "#E0715F", "#B3202C"])   # A-share convention
    norm = mcolors.Normalize(vmin=lo, vmax=hi, clip=True)           # symmetric around zero

    # ------------------------------------------------------------ layout
    fig = plt.figure(figsize=(13.2, 6.6))                           # 16:8 canvas for the slide
    ax = fig.add_axes([0.012, 0.045, 0.86, 0.88])                    # the treemap axes
    ax.set_xlim(0, 1)                                               # unit square
    ax.set_ylim(0, 1)                                               # unit square
    ax.axis("off")                                                  # no axes or ticks
    board_rects = squarify([b.get("v", 1) for b in boards], 0, 0, 1, 1)   # outer partition
    labelled = []                                                   # stock rectangles big enough to label
    for board, (bx, by, bw, bh) in zip(boards, board_rects):        # loop over the boards
        pad = 0.002                                                 # inner padding in axes units
        inner = (bx + pad, by + pad, max(bw - 2 * pad, 0), max(bh - 2 * pad, 0))   # inner box
        stocks = sorted(board["children"], key=lambda s: -s.get("v", 0))   # largest first
        srects = squarify([s.get("v", 1) for s in stocks], *inner)   # inner partition
        for stock, (sx, sy, sw, sh) in zip(stocks, srects):          # draw every stock
            g = stock.get("g")                                      # its percent change
            ax.add_patch(mpatches.Rectangle((sx, sy), sw, sh, fill=True,   # the coloured tile
                                            facecolor=cmap(norm(g if g is not None else 0.0)),
                                            edgecolor="white", linewidth=0.35))   # thin white grid
            if sw * sh > 0.0022:                                    # only large tiles get a name
                labelled.append((sw * sh, stock.get("n", ""), g, sx, sy, sw, sh))   # remember it
        ax.add_patch(mpatches.Rectangle((bx, by), bw, bh, fill=False,   # the board outline
                                        edgecolor="#2B2B2B", linewidth=1.1))   # dark frame
        gb = board.get("g", 0.0) or 0.0                             # the board's percent change
        if bw * bh > 0.0035:                                       # label only boards with room
            fs = 8.4 if bw * bh > 0.02 else 6.6                     # smaller type on small boards
            ax.text(bx + 0.004, by + bh - 0.013,                    # board label position
                    "%s %+.1f%%" % (board.get("n", ""), gb * 100),   # name and change
                    fontsize=fs, fontweight="bold", color="white",   # white on the tiles
                    bbox=dict(boxstyle="round,pad=0.16", fc="#2B2B2B", ec="none", alpha=0.80))   # dark pill
    labelled.sort(reverse=True)                                     # biggest tiles first
    for _, name, g, sx, sy, sw, sh in labelled[:16]:                # label the largest tiles only
        ax.text(sx + sw / 2, sy + sh / 2, name, ha="center", va="center",   # centred label
                fontsize=5.0, color="white" if abs(g or 0) > 0.012 else "#2B2B2B")   # readable on light tiles
    up_n = sum(1 for b in boards if (b.get("g") or 0) > 0)             # industries that rose
    ax.set_title("A股全市场热力树图 · %s 收盘 · %d 个申万一级行业（%d 涨 %d 跌）" % (date, len(boards), up_n, len(boards) - up_n),   # the figure title
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)   # left aligned
    ax.text(0.0, -0.028, "样本：成交额前 %d 只（全市场 %d 只）· 分组：%s · 数据源：腾讯行情 · 生成 %s" % (
                meta.get("n_shown", 0), meta.get("filtered_total", 0),
                meta.get("group_desc", ""), payload.get("generated_at", "")),
            transform=ax.transAxes, fontsize=7.2, color="#5A5A5A")   # the source line
    cax = fig.add_axes([0.885, 0.10, 0.016, 0.62])                   # the colour bar axes
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")   # draw it
    cb.set_label("涨跌幅 (%)", fontsize=9)                            # the colour bar label
    cb.set_ticks([-0.05, -0.025, 0, 0.025, 0.05])                    # five ticks
    cb.set_ticklabels(["-5", "-2.5", "0", "+2.5", "+5"])             # percent labels
    cb.ax.tick_params(labelsize=8)                                   # tick size
    leg = fig.add_axes([0.885, 0.025, 0.10, 0.05])                    # a small legend strip
    leg.axis("off")                                                  # no axes
    leg.text(0, 0.5, "涨红 跌绿", fontsize=8.5, color="#5A5A5A")       # the convention note
    fig.text(0.012, 0.975, "Lecture 4 · 数据描述与可视化", fontsize=8.5, color="#8a0c3c")   # course tag
    fig.savefig("fig_ashare_heatmap.png", dpi=200, facecolor="white")   # save the treemap
    plt.close(fig)                                                   # release the figure

    # ------------------------------------------------------------ distribution figure
    all_pct = [s.get("g", 0.0) or 0.0 for b in boards for s in b["children"]]   # every shown stock
    fig2, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))             # two panels
    names = [b.get("n", "") for b in boards][::-1]                   # board names, smallest last
    vals = [(b.get("g", 0.0) or 0.0) * 100 for b in boards][::-1]    # their percent changes
    cols = [cmap(norm(v / 100)) for v in vals]                        # colour by sign
    axes[0].barh(names, vals, color=cols, edgecolor="#2B2B2B", linewidth=0.5)   # the bars
    axes[0].axvline(0, color="#2B2B2B", lw=0.9)                      # zero reference line
    axes[0].set_title("各板块涨跌幅（加权）", fontsize=11)              # panel title
    axes[0].set_xlabel("涨跌幅 (%)")                                  # axis label
    for i, v in enumerate(vals):                                     # annotate each bar
        axes[0].text(v + (0.06 if v >= 0 else -0.06), i, "%+.2f%%" % v, va="center",   # value label
                     ha="left" if v >= 0 else "right", fontsize=8)   # aligned outside the bar
    axes[1].hist(np.array(all_pct) * 100, bins=40, color="#8C8C8C", alpha=0.85)   # the histogram
    axes[1].axvline(float(np.median(all_pct)) * 100, color="#B3202C", lw=1.4, ls="--",
                    label="中位数 %+.2f%%" % (float(np.median(all_pct)) * 100))   # the median
    axes[1].legend(fontsize=8, loc="upper left")                     # legend
    axes[1].set_title("个股涨跌幅分布（成交额前 %d 只）" % len(all_pct), fontsize=11)   # panel title
    axes[1].set_xlabel("涨跌幅 (%)")                                  # axis label
    axes[1].set_ylabel("只数")                                        # axis label
    for a in axes:                                                   # tidy both panels
        a.grid(alpha=0.22)                                           # light grid
        for sp in ("top", "right"):                                   # drop two spines
            a.spines[sp].set_visible(False)                          # for a cleaner look
    fig2.tight_layout()                                              # avoid overlaps
    fig2.savefig("fig_ashare_distribution.png", dpi=200, facecolor="white")   # save the second figure
    plt.close(fig2)                                                  # release it
    up = sum(1 for g in all_pct if g > 0)                             # how many rose
    down = sum(1 for g in all_pct if g < 0)                           # how many fell
    print("boards:", [(b.get("n"), round((b.get("g") or 0) * 100, 2)) for b in boards])   # report
    print("shown: %d | up %d | down %d | median %+.2f%%" % (
        len(all_pct), up, down, float(np.median(all_pct)) * 100))     # report the summary
    print("root: %s %+.2f%%" % (root.get("n"), (root.get("g") or 0) * 100))   # report the index


if __name__ == "__main__":                                          # run as a script
    main(sys.argv[1] if len(sys.argv) > 1 else "heatmap_sw.json")    # default to the JSON we built
