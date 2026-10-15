"""l5_11_rl_gridworld.py -- reinforcement learning on a small grid world.

Every line carries an English comment. The script implements tabular Q-learning
with an epsilon-greedy policy on a 6x6 grid with walls, then draws the learned
state values, the greedy policy, the return per episode and the temporal
difference error as training progresses.
"""
import numpy as np                                                 # arrays for the Q table and the grid
import pandas as pd                                                # tabular history of the training run
import matplotlib.pyplot as plt                                    # four-panel figure of the learned behaviour
from _common_l5 import RNG, save, dump, style_axes, OKABE          # shared helpers

# ---------------------------------------------------------------- 1) define the environment
SIZE = 6                                                           # the grid is six by six cells
START, GOAL = (0, 0), (SIZE - 1, SIZE - 1)                         # start in the bottom left, goal in the top right
WALLS = {(1, 1), (1, 2), (2, 3), (3, 1), (4, 4)}                   # a few blocked cells shape the optimal path
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]                       # up, down, left, right
ARROWS = {(-1, 0): "^", (1, 0): "v", (0, -1): "<", (0, 1): ">"}     # symbols used when drawing the policy

def step(state, action):                                           # one environment transition
    """Return the next state and the reward for taking an action."""   # short docstring
    r, c = state[0] + action[0], state[1] + action[1]              # candidate next cell
    if not (0 <= r < SIZE and 0 <= c < SIZE):                      # moving off the grid is not allowed
        return state, -1.0                                         # stay in place and pay the step cost
    if (r, c) in WALLS:                                            # moving into a wall is not allowed
        return state, -1.0                                         # stay in place and pay the step cost
    if (r, c) == GOAL:                                             # reaching the goal ends the episode
        return (r, c), 10.0                                        # terminal reward
    return (r, c), -0.5                                            # ordinary move: small negative reward encourages speed

# ---------------------------------------------------------------- 2) Q-learning with epsilon-greedy exploration
Q = np.zeros((SIZE, SIZE, len(ACTIONS)))                           # state-action values, one table per action
alpha, gamma = 0.35, 0.95                                          # learning rate and discount factor
eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995                   # exploration schedule
episodes = 600                                                     # number of training episodes
history = []                                                       # record returns, steps, epsilon and mean TD error
eps = eps_start                                                    # current exploration rate
rng = np.random.default_rng(20260506)                              # local generator for reproducible exploration
for episode in range(1, episodes + 1):                             # loop over the episodes
    state = START                                                  # every episode starts at the same cell
    total_reward, steps, td_errors = 0.0, 0, []                    # reset the episode statistics
    for _ in range(120):                                           # cap the episode length for safety
        if rng.random() < eps:                                     # explore with probability epsilon
            a = int(rng.integers(len(ACTIONS)))                    # pick a random action
        else:                                                      # otherwise exploit the current knowledge
            a = int(np.argmax(Q[state[0], state[1]]))              # greedy action from the Q table
        next_state, reward = step(state, ACTIONS[a])               # interact with the environment
        best_next = 0.0 if next_state == GOAL else float(np.max(Q[next_state[0], next_state[1]]))   # bootstrap target
        td_target = reward + gamma * best_next                     # the Q-learning target
        td_error = td_target - Q[state[0], state[1], a]            # temporal difference error
        Q[state[0], state[1], a] += alpha * td_error               # update the value estimate
        total_reward += reward                                     # accumulate the episodic return
        steps += 1                                                 # count the steps taken
        td_errors.append(abs(td_error))                            # record the absolute error
        state = next_state                                         # move to the next state
        if state == GOAL:                                          # stop when the goal is reached
            break                                                  # leave the inner loop
    eps = max(eps_end, eps * eps_decay)                            # decay the exploration rate
    history.append({"episode": episode, "return": round(total_reward, 2),   # episodic return
                    "steps": steps, "epsilon": round(eps, 4),       # steps taken and current epsilon
                    "mean_td_error": round(float(np.mean(td_errors)), 4)})   # average absolute TD error
hist = pd.DataFrame(history)                                       # assemble the training history
hist["return_ma20"] = hist["return"].rolling(20, min_periods=1).mean().round(2)   # twenty episode moving average
dump(hist, "l5_rl_returns.csv")                                    # persist the history
print("RL: last-20 mean return %.2f | steps %.0f | epsilon %.3f" % (   # report the final performance
    hist["return"].tail(20).mean(), hist["steps"].tail(20).mean(), eps))   # mean return, steps and epsilon

# ---------------------------------------------------------------- 3) extract the greedy policy and the value function
policy = np.argmax(Q, axis=2)                                      # best action per cell
value = np.max(Q, axis=2)                                          # state value under the greedy policy
path, state = [START], START                                       # reconstruct the greedy path from the start
for _ in range(40):                                                # follow the policy for at most forty steps
    if state == GOAL or state in WALLS:                            # stop at the goal or if stuck
        break                                                      # leave the loop
    a = ACTIONS[policy[state[0], state[1]]]                        # greedy action in the current cell
    nxt = (state[0] + a[0], state[1] + a[1])                       # candidate next cell
    if nxt in WALLS or not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE):   # refuse illegal moves
        break                                                      # stop when the policy would leave the grid
    state = nxt                                                    # move along the greedy path
    path.append(state)                                             # record the cell
print("greedy path length:", len(path), "| reaches goal:", path[-1] == GOAL)   # report whether the policy solves the task

# ---------------------------------------------------------------- 4) figure: value, policy, returns, TD error
fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.5))                # four panels in one row
im = axes[0].imshow(value, cmap="RdPu", origin="lower")            # state values as a heatmap
for (r, c) in WALLS:                                               # draw the walls explicitly
    axes[0].add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc="#66717D", ec="none"))   # grey blocked cell
for r in range(SIZE):                                              # annotate every free cell with its greedy action
    for c in range(SIZE):                                          # loop over the columns
        if (r, c) in WALLS:                                        # skip blocked cells
            continue                                               # nothing to draw
        axes[0].text(c, r, ARROWS[ACTIONS[policy[r, c]]], ha="center", va="center",   # arrow of the greedy action
                     fontsize=11, color="#1E2630" if value[r, c] < value.max() * 0.6 else "white")   # readable colour
axes[0].scatter([START[1]], [START[0]], s=90, marker="s", color=OKABE[2], zorder=3, label="start")   # mark the start
axes[0].scatter([GOAL[1]], [GOAL[0]], s=90, marker="*", color=OKABE[0], zorder=3, label="goal")   # mark the goal
axes[0].plot([c for _, c in path], [r for r, _ in path], color="#1E2630", lw=1.6, ls="--", zorder=2)   # greedy path
axes[0].set_xticks(range(SIZE)); axes[0].set_yticks(range(SIZE))    # integer grid ticks
axes[0].grid(False)                                                # the heatmap replaces the grid
axes[0].legend(fontsize=7.2, loc="lower right")                    # legend inside the panel
axes[0].set_title("Learned values and greedy policy")              # panel title
plt.colorbar(im, ax=axes[0], fraction=0.046, label="state value")   # colour bar for the values
axes[1].plot(hist["episode"], hist["return"], lw=0.9, color="#B9B3B6", alpha=0.8, label="episode return")   # raw returns
axes[1].plot(hist["episode"], hist["return_ma20"], lw=2.0, color=OKABE[5], label="20-episode mean")   # smoothed returns
style_axes(axes[1], "Return rises as the policy improves", "Episode", "Episodic return")   # shared style
axes[1].legend(fontsize=7.4, loc="lower right")                    # compact legend
axes[2].plot(hist["episode"], hist["steps"], lw=1.0, color=OKABE[2], alpha=0.85, label="steps per episode")   # steps
axes[2].plot(hist["episode"], hist["steps"].rolling(20, min_periods=1).mean(), lw=2.0,   # smoothed step count
             color=OKABE[0], label="20-episode mean")              # second line
axes[2].axhline(2 * (SIZE - 1), color="#66717D", ls="--", lw=1.0)   # shortest possible path length
style_axes(axes[2], "Fewer steps once the path is found", "Episode", "Steps to goal")   # shared style
axes[2].legend(fontsize=7.4, loc="upper right")                    # legend inside the panel
axes[3].plot(hist["episode"], hist["mean_td_error"], lw=1.0, color=OKABE[5], alpha=0.8, label="mean |TD error|")   # TD error
axes[3].plot(hist["episode"], hist["mean_td_error"].rolling(20, min_periods=1).mean(), lw=2.0,   # smoothed error
             color=OKABE[2], label="20-episode mean")              # second line
ax3b = axes[3].twinx()                                             # second axis for the exploration rate
ax3b.plot(hist["episode"], hist["epsilon"], lw=1.6, color="#66717D", ls=":", label="epsilon")   # epsilon decay
ax3b.set_ylabel("Epsilon", fontsize=9)                             # label the right axis
ax3b.grid(False)                                                   # avoid a doubled grid
style_axes(axes[3], "Values settle as exploration decays", "Episode", "Mean |TD error|")   # shared style
axes[3].legend(fontsize=6.8, loc="upper right")                    # left axis legend
ax3b.legend(fontsize=6.8, loc="center right")                      # right axis legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_rl_gridworld.png")                                  # save the reinforcement learning figure
