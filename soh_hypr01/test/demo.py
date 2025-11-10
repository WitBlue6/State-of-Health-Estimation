from datasets import load_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ds = load_dataset("wangxiangyu0814/TravelUAV_data_json")
print(ds["train"].column_names)
wps = ds["train"]["waypoints"]

N = 5
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for i in range(N):
    traj = wps[i]
    x = [p[0] for p in traj]
    y = [p[1] for p in traj]
    z = [p[2] for p in traj]
    ax.plot(x, y, z, marker=".", linewidth=1, label=f"traj_{i}")

ax.legend()
plt.show()