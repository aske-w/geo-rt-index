import glob
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

import tqdm

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

DATAS = glob.glob(f"data/{DIST}/*_r{LO}{HI}.xy.pickle")
# Create a plot
fig, ax = plt.subplots()
for file in tqdm.tqdm(FILES, "Loading query files"):
    with open (file, "r") as fp:
        data: list[BBox] = json.load(fp)
        for bbox in data[:64]:
            minx = bbox["minx"]
            miny = bbox["miny"]
            maxx = bbox["maxx"]
            maxy = bbox["maxy"]
            width = maxx - minx
            height = maxy - miny
            ax.add_patch(plt.Rectangle((minx, miny), width, height, fill=True, color=color))

xs = []
ys = []
for data_file in tqdm.tqdm(DATAS[:1], "Loading point data"):
    with open(data_file, "rb") as fp:
        xy = pickle.load(fp)
        shuffle(xy)
        xs += xy[::50]
        ys += xy[1::50]

ax.plot(xs, ys,'ro')
# Set limits and labels
ax.set_xlim(LO, HI)
ax.set_ylim(LO, HI)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'{DIST} [{LO},{HI}) Query Visualization')

# Show the plot
# plt.grid(True)
# plt.show()
plt.savefig(f"visualizer_{DIST}_r{LO}{HI}.png")

