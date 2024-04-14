from osgeo.ogr import DataSource, GetDriverByName, Geometry, Layer, wkbPoint
from osgeo import osr, ogr
import os
import numpy as np
import sys

DRIVER = "Parquet"
seed = int(sys.argv[1])
P = 22 # 4m
# P = 26 # 67m
N = 1 << P

os.makedirs("./data", exist_ok=True)
driver: ogr.Driver = GetDriverByName(DRIVER)

file = f"./data/duniform_p{P}_s{seed}.parquet"
while os.path.exists(file):
    seed += 1
    file = f"./data/duniform_p{P}_s{seed}.parquet"
    # os.remove(FILE)

ds: ogr.DataSource = driver.CreateDataSource(file)

srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

layer: ogr.Layer = ds.CreateLayer("data", srs, wkbPoint)
layer_def = layer.GetLayerDefn()

base_feature = ogr.Feature(layer_def)

nums = np.random.uniform(0,1,(N, 2))
for f in nums:
    wkt = "POINT(%f %f)" % (f[0], f[1])
    point: Geometry = ogr.CreateGeometryFromWkt(wkt)
    feature = base_feature.Clone()
    feature.SetGeometry(point)
    layer.CreateFeature(feature)
    feature = None

data_source = None