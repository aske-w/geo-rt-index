from math import sqrt
from random import random
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
    colors = [(0.5129427242976542, 0.7803967101144862, 0.6247630632950358), (0.0845748310231269, 0.2649508535284796, 0.8563803520565164), (0.5129427242976542, 0.3774246857002559, 0.2681407625526605), (0.9799673057777121, 0.3426768832554534, 0.30856558909358744), (0.4724148989992216, 0.05377498651204338, 0.08137208910641491), (0.4289161667403253, 0.44023832780587213, 0.7107943548983909), (0.7628205547449644, 0.0831341981759044, 0.25404624768335915), (0.6179255604559561, 0.18132857680889958, 0.13032159855736436), (0.3953471680093287, 0.8505479559826894, 0.8490298666386054), (0.5546588418125822, 0.053542327428452774, 0.16438268995614724), (0.13595078051285114, 0.5292161628027696, 0.6604701304491002), (0.30708464326257756, 0.8185505528049336, 0.9900950129520214), (0.8665854777143146, 0.7986887682182844, 0.18736729346400316), (0.2627234766137163, 0.5691818011584459, 0.5300192005500743), (0.8919385335776499, 0.898386273300495, 0.8919320506222183), (0.7743331318210733, 0.6624132894323017, 0.36732728155276695), (0.17937820618578415, 0.2100496181354523, 0.23104307261312784), (0.612416737228473, 0.9956841596317944, 0.11520682028543228), (0.5943902809803092, 0.3387602601434647, 0.4789044316984543), (0.5555817832370566, 0.46838225376842435, 0.9965377138383733)]
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
        for i, bbox in enumerate(data):
            minx = bbox.minx
            miny = bbox.miny
            maxx = bbox.maxx
            maxy = bbox.maxy
            width = maxx - minx
            height = maxy - miny
            ax.add_patch(plt.Rectangle((minx, miny), width, height, fill=True, color=colors[i]))

        # Set limits and labels
        ax.set_xlim(LO, HI)
        ax.set_ylim(LO, HI)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_title(f'Query Visualization {square}')

        # Show the plot
        # plt.grid(True)
        # plt.show()
        plt.savefig(f"visualizer_{square}.png",dpi=400,bbox_inches="tight")
