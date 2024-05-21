import json
import numpy as np
import os
import tqdm
from pytypes import BBox
from scipy.stats import norm

LO = -1
HI = 1
BASE_SEED = 0xBADC0DE # https://en.wikipedia.org/wiki/Hexspeak
NUM_QUERIES = 512
SELECTIVITIES = [1,2,5,10,20]
np.random.seed(BASE_SEED)


def mk_query(selectivity: float):
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
        x1 = np.random.uniform(LO, HI - width*2)
        y1 = np.random.uniform(LO, HI - height*2)
        x2 = x1 + width*2
        y2 = y1 + height*2
        return (x1, y1), (x2, y2) # minx, miny, maxx maxy semantics

def mk_nq(selectivity, _loc = 0.5, _scale = 0.15):
    VERBOSE = False
    cdf = lambda x: norm.cdf(x, loc=_loc, scale=_scale)

    while True:
        # select an X and Y value and check it is possible to use them for a query of given selectivity
        x,y = np.random.uniform(LO,HI,2)
        coverage = (1 - cdf(x))*(1 - cdf(y))
        if VERBOSE: 
            print("coverage1", coverage)
        while(coverage <= selectivity):
            x,y = np.random.uniform(LO,min(HI,max(x,y)),2)
            coverage = (1 - cdf(x))*(1 - cdf(y))
            if VERBOSE: 
                print("coverage1", coverage)
        if VERBOSE: 
            print("minx:", x, "miny",y)

        # select a random width and also check it can be used given the selectivity
        LIMIT = 20
        count = 0
        width = np.random.uniform(LO,HI-x)
        coverage = (cdf(x+width) - cdf(x)) * (1 - cdf(y))
        if VERBOSE: 
            print("x width", width)
        if VERBOSE: 
            print("coverage2", coverage)
        while(coverage <= selectivity) and count < LIMIT:
            width = np.random.uniform(LO, HI-x)
            coverage = (cdf(x+width) - cdf(x)) * (1 - cdf(y))
            count += 1
            if VERBOSE: 
                print("coverage2", coverage)
        if count < LIMIT:
            break
    
    # figure out what y+height(=y_max) should by isolating it and using inverse CDF
    x_prop = cdf(x+width) - cdf(x)
    y_max_prop = (selectivity+x_prop*cdf(y))/x_prop
    y_max = norm.ppf(y_max_prop, loc=_loc, scale=_scale)

    if VERBOSE: 
        print("cdf(x+width)", cdf(x+width))
    if VERBOSE: 
        print("cdf(x)", cdf(x))
    if VERBOSE: 
        print("cdf(y)", cdf(y))
    if VERBOSE: 
        print("x_prop", x_prop)
    if VERBOSE: 
        print("y_max", y_max)
    return (x, y), (x + width, y_max)

os.makedirs(f"./queries/uniform/r{LO}{HI}", exist_ok=True)
os.makedirs(f"./queries/normal/r{LO}{HI}", exist_ok=True)

for s in SELECTIVITIES:
    np.random.seed(BASE_SEED * s)
    selectivity_normalized = s / 100
    print("selectivity", selectivity_normalized)
    boxes = []
    for x in tqdm.tqdm(range(NUM_QUERIES)):
        # Generate random coordinates
        corner1, corner2 = mk_query(selectivity_normalized)
        boxes.append(BBox(corner1, corner2))
        width = corner2[0] - corner1[0]
        length = corner2[1] - corner1[1]

    with open(f"./queries/uniform/r{LO}{HI}/{s}.json", "w") as out_file:
        json.dump(boxes, out_file, default=lambda x: x.__dict__)

