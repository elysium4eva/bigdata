"""10_datatype_structured_unstructured.py -- structured, semi-structured and unstructured samples and plots."""  # module comment
import json                                                           # standard library: build semi-structured JSON data
import numpy as np                                                    # numeric computation
import pandas as pd                                                   # table handling for structured data
import matplotlib.pyplot as plt                                       # plotting
from _common import DAT, save, dump_csv                               # shared helpers

# ---------------------------------------------------------------- 1) structured: a two-dimensional table of fixed fields
table = pd.DataFrame({                                                # structured data is rows by columns with names and types
    "order_id": ["SO2026001", "SO2026002", "SO2026003", "SO2026004"],  # order id, a string primary key
    "date": ["2026-03-02", "2026-03-02", "2026-03-03", "2026-03-03"],  # order date, parsable as a date type
    "amount": [1280.50, 640.00, 3980.75, 215.20],                      # order amount, a float
    "paid": [True, True, False, True],                                 # whether the order is paid, a boolean
})                                                                    # end of the structured sample table
dump_csv(table, "structured_orders.csv")                              # dump the structured samples

# ---------------------------------------------------------------- 2) semi-structured: nested JSON with optional fields
records = [                                                           # semi-structured data is text carrying its own schema
    {"order_id": "SO2026001", "buyer": {"name": "林", "vip": True},   # record 1: contains a nested object
     "items": [{"sku": "A-1", "qty": 2}, {"sku": "B-7", "qty": 1}]},  # record 1: contains an array field
    {"order_id": "SO2026002", "buyer": {"name": "陈"},                # record 2: the vip field is missing, the schema varies
     "items": [{"sku": "C-3", "qty": 5}]},                            # record 2: the array length differs too
    {"order_id": "SO2026003", "buyer": {"name": "王", "vip": False},  # record 3
     "items": [], "note": "待补开发票"},                               # record 3: one extra note field
]                                                                     # end of the semi-structured record list
(DAT / "semistructured_orders.json").write_text(                      # write the semi-structured data into the repo
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")  # keep Chinese unescaped and indent for readability
print("[data]", DAT / "semistructured_orders.json")                   # print the output path

# ---------------------------------------------------------------- 3) unstructured: free text and waveforms without fixed fields
free_text = ("本季度公司在东南亚市场的收入同比增长三成，主要来自跨境电商渠道；"  # first paragraph of the unstructured text
             "管理层预计下季度毛利率保持稳定，但需关注汇率波动与海运费上行带来的成本压力。")  # second paragraph
(DAT / "unstructured_note.txt").write_text(free_text, encoding="utf-8")  # dump the unstructured text sample
wave_sample = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 400))          # another unstructured form: a raw waveform
np.save(DAT / "unstructured_wave.npy", wave_sample)                   # save as a binary array sample
print("[data]", DAT / "unstructured_note.txt")                        # print the output path

# ---------------------------------------------------------------- 4) layout helper: wrap manually by character count
def wrap_cjk(s, per_line=24):                                         # define the CJK wrap helper: characters per line
    """Split long text into lines of a fixed character count so the canvas stays compact."""          # function docstring
    return "\n".join(s[i:i + per_line] for i in range(0, len(s), per_line))  # insert a newline every per_line characters

# ---------------------------------------------------------------- 5) visualization: show the three forms side by side
fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3))                   # one row, three columns: table / JSON / text
fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.04,     # set the margins by hand for more control
                    wspace=0.12)                                      # horizontal spacing between subplots
for ax in axes:                                                       # apply the same settings to all three panels
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)                              # fix the axes limits so text cannot expand them
    ax.set_xticks([]); ax.set_yticks([])                              # hide the ticks
    ax.grid(False)                                                    # turn off the grid

t = axes[0].table(cellText=table.astype(str).values,                  # left: render the DataFrame as a table
                  colLabels=list(table.columns),                      # column labels come from the column names
                  loc="center", cellLoc="center", bbox=[0.02, 0.42, 0.96, 0.5])  # fix the table position with a bbox
t.auto_set_font_size(False); t.set_fontsize(8)                        # disable auto font size and set 8pt
axes[0].set_title("结构化：定长字段的二维表 (CSV)", fontsize=12)        # left panel title

axes[1].text(0.02, 0.96, wrap_cjk(json.dumps(records[0],              # middle: show a JSON record with nesting and arrays
               ensure_ascii=False), 26)[:520],                        # keep only the first 520 characters after wrapping
             family=["Consolas", "Microsoft YaHei"],                  # monospace font plus a Chinese fallback
             fontsize=8.2, va="top", ha="left")                       # font size and top-left alignment
axes[1].set_title("半结构化：嵌套 JSON（模式可变）", fontsize=12)        # middle panel title
axes[1].text(0.02, 0.18, "字段可缺省、可新增：解析时需容错",            # add a key point at the bottom of the middle panel
             fontsize=8.5, color="#711426", va="bottom", ha="left")   # highlight in dark red

axes[2].text(0.02, 0.96, wrap_cjk(free_text, 22), fontsize=9.5,       # right: show free text without any fields
             va="top", ha="left")                                     # multi-line text from manual wrapping
axes[2].plot(np.linspace(0, 1, 400), 0.22 + 0.12 * wave_sample,       # draw a raw waveform in the lower right panel
             color="#8A0C3C", lw=1.0)                                 # unstructured data can also be a signal
axes[2].text(0.02, 0.06, "文本 / 音频 / 图像：无字段可枚举",            # add a key point at the bottom of the right panel
             fontsize=8.5, color="#711426", va="bottom", ha="left")   # highlight in dark red
axes[2].set_title("非结构化：文本 / 音频 / 图像", fontsize=12)          # right panel title
save(fig, "fig_structured.png")                                       # save the figure, the axes limits are already fixed
