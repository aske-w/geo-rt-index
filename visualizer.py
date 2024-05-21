import numpy as np
import matplotlib.pyplot as plt
import json

from pytypes import BBox
# this is mostly the work of ChatGPT

LO = 0
HI = 1
DIST = "uniform"
FILES = [
    ("tab:red",f"./queries/{DIST}/r{LO}{HI}/20.json"),
    ("tab:green",f"./queries/{DIST}/r{LO}{HI}/10.json"),
    ("tab:blue",f"./queries/{DIST}/r{LO}{HI}/5.json"),
    ("tab:orange",f"./queries/{DIST}/r{LO}{HI}/2.json"),
    ("tab:gray",f"./queries/{DIST}/r{LO}{HI}/1.json"),
]
# Create a plot
fig, ax = plt.subplots()
for color, file in FILES:
    with open (file, "r") as fp:
        data: list[BBox] = json.load(fp)
        # data: list[BBox] = json.loads('[{"minx": 0.2541947203804815, "miny": 0.8318307966853772, "maxx": 0.6739473347137762, "maxy": 0.8556543528909372}]')
        # data: list[BBox] = json.loads('[{"minx": -0.49161055923903696, "miny": 0.6636615933707544, "maxx": 0.3478946694275524, "maxy": 0.7113087057818743}]')
        for bbox in data[:4]:
            minx = bbox["minx"]
            miny = bbox["miny"]
            maxx = bbox["maxx"]
            maxy = bbox["maxy"]
            width = maxx - minx
            height = maxy - miny
            ax.add_patch(plt.Rectangle((minx, miny), width, height, fill=True, color=color))

# Set limits and labels
ax.set_xlim(LO, HI)
ax.set_ylim(LO, HI)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'{DIST} [{LO},{HI}) Query Visualization')

# Show the plot
# plt.grid(True)
plt.show()
# plt.savefig("visualizer.png")
