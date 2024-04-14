from time import sleep
import cuspatial
import cudf
import cupy
import cuspatial.utils
import cuspatial.utils.column_utils
import geopandas as gp
import pandas as pd
import numpy as np
import cuda
import time
import pyarrow.parquet
import shapely
from shapely.geometry import *
from shapely import wkt, from_wkb
import pyarrow.parquet as pq
import pyarrow
from osgeo import ogr

from concurrent.futures import ThreadPoolExecutor

pool = ThreadPoolExecutor(2)

NUM = (1 << 25) + (3 * 1 << 23) + (1 << 22) # 62,914,560
# NUM = (1 << 26) # 67,108,864 - see error in bottom
# NUM = 10
np.random.seed(0)
cupy.random.seed(0)

p = pq.read_table("data/duniform_p22_s1337.parquet")

def destruct(g):
  return g[0].x, g[0].y

def work(batch):
  geom_col = batch["geometry"]
  result = [None] * batch.num_rows
  for i, geom in enumerate(geom_col):
    g = ogr.CreateGeometryFromWkb(geom.as_py())
    result[i] = (g.GetX())
    result[i] = (g.GetY())
  return result

t = time.perf_counter()
batches: list = p.to_batches(1 << 20)


xy = []
for result in pool.map(work, batches):
  xy += result

# for batch in batches:

print(f"converting to xy took: {time.perf_counter() - t:.3f} ms")
print(len(xy))
exit()

t = time.perf_counter()
df = gp.read_parquet("data/duniform_p22_s1337.parquet", ["geometry"])["geometry"]
print(df.size)
print(f"generating {NUM:,} xy points took: {time.perf_counter() - t:.3f} ms")
print("loaded parquet file")

# t = time.perf_counter()
# xy = []
# for g in df:
#     xy.append(g.x)
#     xy.append(g.y)
# print(f"converting to xy took: {time.perf_counter() - t:.3f} ms")

t = time.perf_counter()
points = cuspatial.from_geopandas(df)
print(f"converting to cuspatial took: {time.perf_counter() - t:.3f} ms")
del df

# t = time.perf_counter()
# with open("data/duniform_p26_s1278.pickle", "wb") as f:
#     pickle.dump(points, f)
# print(f"pickling took: {time.perf_counter() - t:.3f} ms")
exit()
# def worker(slice):
#     return cuspatial.GeoSeries(slice)

# points_fut = pool.submit(worker, df[:(1<<25)])
# points2_fut = pool.submit(worker, df[(1<<25):])
# # points = cuspatial.from_geopandas(df[:(1<<25)])
# # points2 = cuspatial.from_geopandas(df[(1<<25):])
# points = points_fut.result()
# points.append(points2_fut.result())
del df
print(f"generating {NUM:,} xy points took: {time.perf_counter() - t:.3f} ms")
print("loaded geoseries")
print(points.head())

# t = time.perf_counter()
# x_points = (cupy.random.random(NUM) - 0.5) * 360
# y_points = (cupy.random.random(NUM) - 0.5) * 180
# print(f"generating {NUM:,} xy points took: {time.perf_counter() - t:.3f} ms")
# sleep(3)


# t = time.perf_counter()
# xy = cudf.DataFrame({"x": x_points, "y": y_points}).interleave_columns()
# print(f"interleave_columns took: {time.perf_counter() - t:.3f} ms")
# sleep(3)


# t = time.perf_counter()
# points = cuspatial.GeoSeries.from_points_xy(xy)
# del xy, x_points, y_points
# print(f"Creating points from interleaved columns took: {time.perf_counter() - t:.3f} ms")
# print(points.head())
# exit()

input()
hit_start = time.perf_counter()
hit = cuspatial.points_in_spatial_window(points, 0, 1, 0, 1) # beware that it is minx, maxx, miny, maxy
print("hit took ", time.perf_counter() - hit_start)
print(hit.head()) # lmao
input()

# bboxes_dict = {
#     "minx": [-20.],
#     "maxx": [20.],
#     "miny": [-10.],
#     "maxy": [10.],
# }
# bboxes = cudf.DataFrame(bboxes_dict)
# print(bboxes)


# scale = 5
# max_depth = 7
# max_size = 125
# print(44)
# point_indices, quadtree = cuspatial.quadtree_on_points(points,
#                                                        x_points.min(),
#                                                        x_points.max(),
#                                                        y_points.min(),
#                                                        y_points.max(),
#                                                        scale,
#                                                        max_depth,
#                                                        max_size)

# # print(point_indices.head())
# print(point_indices)
# # host_dataframe = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
# # gpu_dataframe = cuspatial.from_geopandas(host_dataframe)

# # polygons = gpu_dataframe['geometry']
# # poly_bboxes = cuspatial.polygon_bounding_boxes(
# #     polygons
# # )

# # print(poly_bboxes)

# print(5)
# intersections = cuspatial.join_quadtree_and_bounding_boxes(
#     quadtree,
#     bboxes,
#     x_points.min(),
#     x_points.max(),
#     y_points.min(),
#     y_points.max(),
#     scale,
#     max_depth
# )
# print(6)
# print(intersections)

"""
Traceback (most recent call last):
  File "/home/aske/dev/py_cuspatial/main.py", line 44, in <module>
    hit = cuspatial.points_in_spatial_window(points, 140, 180, 50, 60)
  File "/usr/local/lib/python3.10/dist-packages/cuspatial/core/spatial/filtering.py", line 52, in points_in_spatial_window
    ys = as_column(points.points.y)
  File "/usr/local/lib/python3.10/dist-packages/cuspatial/core/geoseries.py", line 236, in y
    return self.xy[1::2].reset_index(drop=True)
  File "/usr/local/lib/python3.10/dist-packages/cuspatial/core/geoseries.py", line 240, in xy
    features = self._get_current_features(self._type)
  File "/usr/local/lib/python3.10/dist-packages/cuspatial/core/geoseries.py", line 253, in _get_current_features
    existing_features = self._col.take(existing_indices._column)
  File "/usr/local/lib/python3.10/dist-packages/cudf/core/column/column.py", line 830, in take
    return libcudf.copying.gather([self], indices, nullify=nullify)[
  File "/usr/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "copying.pyx", line 151, in cudf._lib.copying.gather
  File "copying.pyx", line 48, in cudf._lib.pylibcudf.copying.gather
  File "copying.pyx", line 75, in cudf._lib.pylibcudf.copying.gather
MemoryError: std::bad_alloc: out_of_memory: CUDA error at: /__w/cudf/cudf/python/cudf/build/cp310-cp310-manylinux_2_28_x86_64/_deps/rmm-src/include/rmm/mr/device/cuda_memory_resource.hpp:69: cudaErrorMemoryAllocation out of memory
"""