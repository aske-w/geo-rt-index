from math import sqrt
from pytypes import BBox
import matplotlib.pyplot as plt

LO = 0
HI = 1
NUM_QUERIES = 20
base = BBox((0.0, 0.0), (1/NUM_QUERIES, 1/NUM_QUERIES))

# (0,1^2)/(0,1 - 0,078)^2
neighbor_coverages = [
    sqrt(0.10), 
    sqrt(0.050),
    sqrt(0.025),
    sqrt(0.0125),
    0.00
]

for cover in neighbor_coverages:
    data: list[BBox] = [base]
    last = base
    SIDE_LENGTH = (1.0 / NUM_QUERIES)
    for _ in range(NUM_QUERIES - 1):
        minx = last.maxx - SIDE_LENGTH * (cover)
        miny = last.maxy - SIDE_LENGTH * (cover)
        maxx = minx + SIDE_LENGTH
        maxy = miny + SIDE_LENGTH
        this = BBox((minx, miny), (maxx, maxy))
        data.append(this)
        last = this
        
    # this is mostly the work of ChatGPT
    # Create a plot
    fig, ax = plt.subplots()
    for bbox in data:
        minx = bbox.minx
        miny = bbox.miny
        maxx = bbox.maxx
        maxy = bbox.maxy
        width = maxx - minx
        height = maxy - miny
        ax.add_patch(plt.Rectangle((minx, miny), width, height, fill=True))

    # Set limits and labels
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Query Visualization {cover ** 2:.4f}')

    # Show the plot
    # plt.grid(True)
    plt.show()
    # plt.savefig("visualizer.png")
