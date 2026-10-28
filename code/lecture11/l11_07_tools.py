"""l11_07_tools.py -- function calling, argument validation and an agent loop.

Every line carries an English comment. The script registers three local tools, has the model emit a   # continuation of the call
tool call, validates and executes it, then runs a short plan-act-observe loop and draws the trace.   # continuation of the call
No network is involved: the "model" is the local n-gram chat model.     # continuation of the call
"""
import json                                                        # tool-call payloads
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l11 import (get_model, save, dump, style_axes,          # shared helpers
                         annotated, OKABE)                              # continuation of the call

model = get_model()                                                # the local chat model

# ---------------------------------------------------------------- 1) the tool registry
def calculator(expression):                                        # tool 1: arithmetic
    """Return the value of a simple arithmetic expression."""       # short docstring
    allowed = set("0123456789+-*/(). ")                            # a strict character whitelist
    if not set(expression) <= allowed:                             # refuse anything else
        raise ValueError("unsafe expression")                      # fail loudly
    return float(eval(expression, {"__builtins__": {}}, {}))        # evaluate in an empty namespace

def lookup_rate(currency):                                         # tool 2: an exchange rate
    """Return a fixed teaching rate for a currency pair."""         # short docstring
    table = {"USD": 7.10, "EUR": 7.75, "JPY": 0.048, "GBP": 9.05}    # fixed rates for the demo
    return table.get(currency.upper(), None)                       # None when unknown

def count_tokens(text):                                            # tool 3: cost estimation
    """Return a rough token count for a piece of text."""           # short docstring
    from _common_l11 import tokenize                               # local import keeps the top clean
    return len(tokenize(text))                                     # the count

TOOLS = {                                                          # name -> (function, required args)
    "calculator": (calculator, ["expression"]),                    # arithmetic
    "lookup_rate": (lookup_rate, ["currency"]),                    # exchange rate
    "count_tokens": (count_tokens, ["text"]),                      # token count
}                                                                  # end of the registry

# ---------------------------------------------------------------- 2) the model's tool call
def parse_call(payload):                                           # validate a tool call
    """Return (ok, name, args) after checking the registry and required keys."""   # short docstring
    try:                                                          # parsing may fail
        call = json.loads(payload)                                 # parse the JSON
    except Exception:                                             # not JSON
        return False, "", {}                                       # invalid
    name = call.get("tool", "")                                    # the requested tool
    args = call.get("args", {})                                    # its arguments
    if name not in TOOLS:                                          # unknown tool
        return False, name, args                                   # invalid
    required = TOOLS[name][1]                                      # the required keys
    if any(k not in args for k in required):                        # missing arguments
        return False, name, args                                   # invalid
    return True, name, args                                        # valid

# a tool call the model would emit (written out so the shape is visible)
call_json = json.dumps({"tool": "lookup_rate", "args": {"currency": "USD"}}, ensure_ascii=False)   # the payload
ok, name, args = parse_call(call_json)                              # validate it
result = TOOLS[name][0](**args) if ok else None                      # execute it
print("call valid:", ok, "| tool:", name, "| result:", result)        # report
# the dangerous version: the model asks for something outside the whitelist
bad_json = json.dumps({"tool": "calculator", "args": {"expression": "__import__('os')"}})   # an injection attempt
try:                                                               # the tool must refuse
    calculator("__import__('os')")                                  # direct call
    blocked = False                                                # it was not blocked
except Exception:                                                  # it was blocked
    blocked = True                                                 # record it
print("injection blocked by the whitelist:", blocked)                # report

# ---------------------------------------------------------------- 3) the agent loop
question = "把 120 美元换算成人民币，并估算这段文本的 token 数：跨境电商的物流时效"   # the task
trace = []                                                         # the trace rows
def step(kind, detail):                                            # append one trace row
    """Append one step to the agent trace."""                      # short docstring
    trace.append({"step": len(trace) + 1, "type": kind, "detail": detail})   # the record

step("思考", "任务需要两步：先查汇率，再做乘法")                        # plan
step("工具调用", "lookup_rate(currency='USD')")                       # call the rate tool
rate = lookup_rate("USD")                                          # observe
step("观察", "1 USD = %.2f CNY" % rate)                             # the observation
step("工具调用", "calculator(expression='120*7.10')")                 # call the calculator
amount = calculator("120*%.2f" % rate)                              # observe
step("观察", "120 美元 = %.1f 元" % amount)                          # the observation
step("思考", "还需要 token 数，调用第三个工具")                         # plan again
step("工具调用", "count_tokens(text='...')")                         # call the counter
n_tokens = count_tokens("跨境电商的物流时效")                          # observe
step("观察", "约 %d tokens" % n_tokens)                              # the observation
step("回答", "120 美元 = %.1f 元；文本约 %d tokens" % (amount, n_tokens))   # final answer
trail = pd.DataFrame(trace)                                        # assemble
dump(trail, "l11_agent_trace.csv")                                 # persist
print(trail.to_string(index=False))                                # report

# ---------------------------------------------------------------- 4) the figure
fig, ax = plt.subplots(figsize=(11.0, 3.8))                         # one wide panel
ax.axis("off")                                                     # a sketch, not a plot
colours = {"思考": OKABE[1], "工具调用": OKABE[4], "观察": OKABE[2], "回答": OKABE[5]}   # per-type colour
for i, row in trail.iterrows():                                    # loop over the steps
    y = 0.88 - i * 0.11                                             # vertical position
    ax.add_patch(plt.Rectangle((0.02, y - 0.035), 0.16, 0.07,           # draw the panel
                               fc=colours[row["type"]], alpha=0.85, ec="white"))   # the type chip
    ax.text(0.10, y, row["type"], ha="center", va="center", fontsize=8.0, color="white")   # the chip label
    ax.text(0.21, y, row["detail"], va="center", fontsize=8.6)       # the detail
ax.set_xlim(0, 1); ax.set_ylim(0, 1)                                # the sketch bounds
ax.set_title("一次 Agent 轨迹：思考 → 调用 → 观察 → 再思考")            # the title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_agent.png")                                         # save the figure
print("done: function calling, validation and the agent loop")        # closing log
