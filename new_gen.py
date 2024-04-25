from osgeo.ogr import DataSource, GetDriverByName, Geometry, Layer, wkbPoint
from osgeo import osr, ogr
import os
import numpy as np
import sys

import tqdm

DRIVER = "Parquet"
seed = int(sys.argv[1])
rng = None
rng_single = None
dist = sys.argv[2].strip()
match dist:
    case "uniform":
        rng = lambda: np.random.uniform(-1,1,(N,2))
        rng_single = lambda: np.random.uniform(0,1,2)
    case "normal":
        # rng = lambda: np.random.normal(0.5,0.15,(N,2))
        # rng_single = lambda: np.random.normal(0.5, 0.15, 2)
        rng = lambda: np.random.normal(0.0,0.3,(N,2))
        rng_single = lambda: np.random.normal(0.0, 0.3, 2)

np.random.seed(seed)
# P = 22 # 4m
P = 26 # 67m
N = 1 << P
DIR  = os.path.join("/Volumes/untitled/data", dist)

os.makedirs(DIR, exist_ok=True)
driver: ogr.Driver = GetDriverByName(DRIVER)

file = os.path.join(DIR, f"p{P}_s{seed}_r01.parquet")
while os.path.exists(file):
    seed = (seed + int(seed / 2)) % (1 << 15 )
    print("dub", seed)
    file = os.path.join(DIR, f"p{P}_s{seed}_r01.parquet")
    # os.remove(FILE)

ds: ogr.DataSource = driver.CreateDataSource(file)

srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

layer: ogr.Layer = ds.CreateLayer("data", srs, wkbPoint)
layer_def = layer.GetLayerDefn()

base_feature = ogr.Feature(layer_def)

nums = rng()
for f in tqdm.tqdm(nums):
    while f[0] < 0 or f[1] < 0 or f[0] > 1 or f[1] > 1:
        f = rng_single()

    point = Geometry(ogr.wkbPoint)
    point.AddPoint_2D(f[0], f[1])
    feature = base_feature.Clone()
    feature.SetGeometry(point)
    layer.CreateFeature(feature)
    feature = None

data_source = None