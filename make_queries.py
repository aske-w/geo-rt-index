import json
import numpy as np
import os
from pytypes import BBox
from scipy.stats import norm

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

def mk_nq(selectivity, _loc = 0.5, _scale = 0.15):
    cdf = lambda x: norm.cdf(x, loc=_loc, scale=_scale)
    x,y = np.random.uniform(LO,HI,2)
    coverage = (1 - cdf(x))*(1 - cdf(y))
    print("coverage1", coverage)
    while(coverage <= selectivity):
        x,y = np.random.uniform(LO,HI,2)
        coverage = (1 - cdf(x))*(1 - cdf(y))
        print("coverage1", coverage)
    print("minx:", x, "miny",y)
    width = np.random.uniform(LO,HI-x)
    coverage = (cdf(x+width) - cdf(x)) * (1 - cdf(y))
    print("x width", width)
    print("coverage2", coverage)
    while(coverage <= selectivity):
        (cdf(x+width) - cdf(x)) * (1 - cdf(y))
        width = np.random.uniform(LO, HI-x)
        print("coverage2", coverage)
    
    x_prop = cdf(x+width) - cdf(x)
    y_max = norm.ppf((selectivity+x_prop*cdf(y))/x_prop, loc=_loc, scale=_scale)
    # y_prop = (selectivity / (cdf(x+width) - cdf(x))) + cdf(y)
    # x_prop * y_prop = s
    # (cdf(x+width) - cdf(x)) * (cdf(y+height)-cdf(y)) = s
    # (cdf(y+height)-cdf(y)) = s / (cdf(x+width) - cdf(x))
    # cdf(y+height) = (s / (cdf(x+width) - cdf(x))) + cdf(y)
    print("cdf(x+width)", cdf(x+width))
    print("cdf(x)", cdf(x))
    print("cdf(y)", cdf(y))
    print("x_prop", x_prop)
    print("y_max", y_max)
    # print("y_prop", y_prop)
    # print("selectivity / x_prop", selectivity / x_prop)
    # height = norm.ppf(selectivity / x_prop, loc=_loc, scale=_scale)
    # print("height", height)
    return (x, y), (x + width, y_max)

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

