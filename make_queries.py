import json
import numpy as np
import os
from pytypes import BBox

LO = 0
HI = 1
BASE_SEED = 0xBADC0DE # https://en.wikipedia.org/wiki/Hexspeak
NUM_QUERIES = 512
SELECTIVITIES = [1,2,5,10,20]
np.random.seed(BASE_SEED)


def mk_query(selectivity: float, lo = 0, hi = 1):
    # raise NotImplementedError()
    assert(0 < selectivity and selectivity <= 1)
    area = selectivity
    # Generate random dimensions
    width = np.random.uniform(1, area)  # Limit the width to be less than or equal to the area
    height = area / width
    
    # Check if the dimensions produce the desired area
    if abs(width * height - area) < 1e-6:
        coinflip = np.random.binomial(1, 0.5) # there is a tendency to create wide (as opposed to tall) rectangles
        if coinflip == 1:
            temp = width
            width = height
            height = temp
        # return width, height
        x1 = np.random.uniform(lo, hi - width)
        y1 = np.random.uniform(lo, hi - height)
        x2 = x1 + width
        y2 = y1 + height
        return (x1, y1), (x2, y2) # minx, miny, maxx maxy semantics

os.makedirs("./data/queries/uniform", exist_ok=True)

for s in SELECTIVITIES:
    np.random.seed(BASE_SEED * s)
    selectivity_normalized = s / 100
    boxes = []
    for _ in range(NUM_QUERIES):
        # Generate random coordinates
        corner1, corner2 = mk_query(selectivity_normalized, LO, HI)
        boxes.append(BBox(corner1, corner2))
        width = corner2[0] - corner1[0]
        length = corner2[1] - corner1[1]

    with open(f"./data/queries/{s}.json", "w") as out_file:
        json.dump(boxes, out_file, default=lambda x: x.__dict__)

