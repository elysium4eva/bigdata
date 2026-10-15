"""07_datatype_graph.py -- graph and network data: build a scale-free network and plot it."""  # module comment
import numpy as np                                                    # numeric computation
import pandas as pd                                                   # export the edge and node tables
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, DAT, save, dump_csv                          # shared helpers

N, M = 60, 2                                                          # 60 nodes, each new node links 2 edges, BA model
edges = []                                                            # initialise the edge list
targets = [0, 1]                                                      # seed core: nodes 0 and 1 are already linked
edges.append((0, 1))                                                  # record the core edge
for v in range(2, N):                                                 # add nodes from 2 upward with preferential attachment
    deg = np.zeros(v)                                                 # degrees of the v existing nodes
    for a, b in edges:                                                # accumulate degrees over the existing edges
        deg[a] += 1                                                   # increment the source degree
        deg[b] += 1                                                   # increment the target degree
    prob = (deg + 1) / (deg + 1).sum()                                # link probability rises with degree, plus 1 smoothing
    chosen = RNG.choice(v, size=M, replace=False, p=prob)             # draw M neighbours without replacement
    for c in chosen:                                                  # loop over the chosen neighbours
        edges.append((int(c), v))                                     # add a new edge, the rich get richer
adj = np.zeros((N, N))                                                # initialise the adjacency matrix
for a, b in edges:                                                    # fill the adjacency matrix from the edge list
    adj[a, b] = adj[b, a] = 1                                         # undirected graph: set the matrix symmetric to 1
deg = adj.sum(axis=1)                                                 # degree of a node equals its row sum

# ---------------------------------------------------------------- force-directed layout, simplified Fruchterman–Reingold
pos = RNG.uniform(0, 1, (N, 2))                                       # random initial node coordinates
for it in range(120):                                                 # run 120 iterations to settle the layout
    delta = pos[:, None, :] - pos[None, :, :]                         # displacement vectors between all point pairs
    dist = np.sqrt((delta ** 2).sum(-1)) + 1e-9                       # pairwise distances, tiny value avoids division by zero
    rep = (delta / dist[..., None]) / dist[..., None] ** 2            # repulsion between all pairs, inverse square of distance
    np.fill_diagonal(rep[:, :, 0], 0); np.fill_diagonal(rep[:, :, 1], 0)  # a point does not repel itself
    attr = -(delta / dist[..., None]) * dist[..., None] * adj[..., None]  # attraction between linked nodes along the edge
    step = (rep.sum(1) + attr.sum(1)) * 0.002                         # net force times step size gives the shift
    pos = pos + np.clip(step, -0.02, 0.02)                            # clip the single step to avoid oscillation
pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0) + 1e-9)           # rescale into [0,1] for plotting

dump_csv(pd.DataFrame({"src": [a for a, _ in edges],                  # edge table: source node
                       "dst": [b for _, b in edges]}),                # edge table: target node
         "graph_edges.csv")                                           # dump the edge table
dump_csv(pd.DataFrame({"node": range(N), "degree": deg}),             # node table: node index
         "graph_nodes.csv")                                           # dump the node table with degrees

# ---------------------------------------------------------------- visualization
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))                    # one row, two columns: network plot / degree distribution
for a, b in edges:                                                    # draw one line per edge
    axes[0].plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],      # connect the coordinates of both endpoints
                 color="#B9BFC7", lw=0.8, zorder=1)                   # thin grey lines, drawn below the nodes
axes[0].scatter(pos[:, 0], pos[:, 1], s=20 + 9 * deg,                 # node size grows with the degree
                c=deg, cmap="RdPu", edgecolors="#711426", zorder=2)  # colour also shows the degree, to spot hubs
axes[0].set_title("网络图（%d 节点 / %d 边，BA 无标度）" % (N, len(edges)))  # left panel title
axes[0].set_xticks([]); axes[0].set_yticks([])                        # hide the ticks
axes[0].grid(False)                                                   # turn off the grid
axes[0].set_aspect("equal")                                           # equal aspect so the graph is not stretched

vals, cnt = np.unique(deg, return_counts=True)                        # degree distribution: degree to node count
axes[1].loglog(vals, cnt, "o", color="#8A0C3C", ms=6)                 # right: degree distribution on log-log axes, nearly a line
axes[1].set_title("度分布（双对数：幂律特征）")                        # right panel title
axes[1].set_xlabel("节点度 k")                                        # right panel x label
axes[1].set_ylabel("度为 k 的节点数 P(k)")                            # right panel y label
plt.tight_layout()                                                    # adjust the layout
save(fig, "fig_graph.png")                                            # save the figure
