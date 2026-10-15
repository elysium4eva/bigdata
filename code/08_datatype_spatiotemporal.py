"""08_datatype_spatiotemporal.py -- spatiotemporal data: tracks plus density and speed plots."""  # module comment
import numpy as np                                                    # numeric computation
import pandas as pd                                                   # table handling
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, save, dump_csv                               # shared helpers

# ---------------------------------------------------------------- 1) generate three tracks near Nanshan, Shenzhen
lat0, lon0 = 22.5333, 113.9300                                        # start point near Nanshan district, Shenzhen
tracks = ["粤B·A1234", "粤B·B5678", "粤B·C9012"]                       # three vehicle or rider ids
rows = []                                                             # initialise the per-point record list
for i, tid in enumerate(tracks):                                      # generate one track at a time
    n = 120                                                           # 120 sample points per track, about 2 minutes
    speed = RNG.normal(9.0 + 2 * i, 1.5, n).clip(2, 18)               # speed in m/s: different means plus noise
    head = np.cumsum(RNG.normal(0, 0.12, n))                          # heading angle random walk, mimics turning
    dlat = np.cos(head) * speed * 5 / 111000.0                        # latitude step: distance over 111 km per degree
    dlon = np.sin(head) * speed * 5 / (111000.0 * np.cos(np.radians(lat0)))  # longitude step: needs the latitude cosine correction
    la = lat0 + np.cumsum(dlat) + RNG.normal(0, 1e-5, n)              # cumulative latitude series plus positioning noise
    lo = lon0 + np.cumsum(dlon) + RNG.normal(0, 1e-5, n)              # cumulative longitude series plus positioning noise
    for k in range(n):                                                # write one record per point
        rows.append({"track_id": tid, "t_sec": k * 5,                # track id and timestamp, one point every 5 seconds
                     "lat": round(float(la[k]), 6),                   # latitude, 6 decimals, about 0.1 m
                     "lon": round(float(lo[k]), 6),                   # longitude
                     "speed_mps": round(float(speed[k]), 2)})         # instantaneous speed
traj = pd.DataFrame(rows)                                             # assemble a long table, one row per point
dump_csv(traj, "spatiotemporal_tracks.csv")                           # dump the spatiotemporal samples

# ---------------------------------------------------------------- 2) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9))                   # one row, three columns: track / density / speed over time
for tid in tracks:                                                    # draw one track at a time
    sub = traj[traj["track_id"] == tid]                               # take the subset belonging to this track
    axes[0].plot(sub["lon"], sub["lat"], "-", lw=1.4, label=tid)      # left: the longitude and latitude polyline is the track
axes[0].set_title("轨迹（经纬度序列）")                                # left panel title
axes[0].set_xlabel("经度 / °E")                                       # left panel x label
axes[0].set_ylabel("纬度 / °N")                                       # left panel y label
axes[0].legend(fontsize=7)                                            # show the vehicle legend
axes[0].ticklabel_format(useOffset=False)                             # disable the scientific offset to show real coordinates

hb = axes[1].hexbin(traj["lon"], traj["lat"], gridsize=18,            # middle: hexbin heatmap of spatial density
                    cmap="magma", mincnt=1)                           # show only bins that contain points
axes[1].set_title("空间密度热力图（hexbin）")                          # middle panel title
axes[1].set_xlabel("经度 / °E")                                       # middle panel x label
axes[1].ticklabel_format(useOffset=False)                             # disable the scientific offset
axes[1].grid(False)                                                   # a heatmap needs no grid

for tid in tracks:                                                    # draw a speed curve for every track
    sub = traj[traj["track_id"] == tid]                               # take this track
    axes[2].plot(sub["t_sec"], sub["speed_mps"], lw=1.0,              # right: speed against time
                 label=tid)                                           # add the legend label
axes[2].set_title("速度随时间变化（时空→时序）")                        # right panel title
axes[2].set_xlabel("时间 / 秒")                                       # right panel x label
axes[2].set_ylabel("速度 / (米/秒)")                                  # right panel y label
plt.tight_layout()                                                    # adjust the layout
save(fig, "fig_spatiotemporal.png")                                   # save the figure
