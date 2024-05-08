import numpy as np
import matplotlib.pyplot as plt
import json

from pytypes import BBox
# this is mostly the work of ChatGPT

LO = 0
HI = 1
FILES = [
    f"./queries/normal/r{LO}{HI}/1.json",
    # f"./queries/normal/r{LO}{HI}/2.json",
    # f"./queries/normal/r{LO}{HI}/5.json",
    # f"./queries/normal/r{LO}{HI}/10.json",
    # f"./queries/normal/r{LO}{HI}/20.json",
]
# Create a plot
fig, ax = plt.subplots()
for file in FILES:
    with open (file, "r") as fp:
        data: list[BBox] = json.load(fp)
        for bbox in data:
            minx = bbox["minx"]
            miny = bbox["miny"]
            maxx = bbox["maxx"]
            maxy = bbox["maxy"]
            width = maxx - minx
            height = maxy - miny
            ax.add_patch(plt.Rectangle((minx, miny), width, height, fill=True, edgecolor=None))

# Set limits and labels
ax.set_xlim(LO, HI)
ax.set_ylim(LO, HI)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Bounding Box Visualization')

# Show the plot
# plt.grid(True)
# plt.show()
plt.savefig("visualizer.png")
