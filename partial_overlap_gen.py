from math import sqrt
from pytypes import BBox
import matplotlib.pyplot as plt

def get(overlap: float, num_queries = 20):
    base = BBox((0.0, 0.0), (1/num_queries, 1/num_queries))
    cover = sqrt(overlap)
    data: list[BBox] = [base]
    last = base
    SIDE_LENGTH = (1.0 / num_queries)
    for _ in range(num_queries - 1):
        minx = last.maxx - SIDE_LENGTH * (cover)
        miny = last.maxy - SIDE_LENGTH * (cover)
        maxx = minx + SIDE_LENGTH
        maxy = miny + SIDE_LENGTH
        this = BBox((minx, miny), (maxx, maxy))
        data.append(this)
        last = this
    return data



if __name__ == "__main__":
    # (0,1^2)/(0,1 - 0,078)^2
    neighbor_coverages = [
        0.50,
        0.25,
        0.10,
        0.05,
        0.025,
        0.0125,
        0.00625,
        0.0
    ]
    LO = 0
    HI = 1
    for square in neighbor_coverages:
        print(f"--- {square} ---")
        data = get(square)
        # print("-q", *d, sep=" ")
        # continue
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
        ax.set_title(f'Query Visualization {square}')

        # Show the plot
        # plt.grid(True)
        plt.show()
        # plt.savefig("visualizer.png")
