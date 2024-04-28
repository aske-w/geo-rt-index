import cuspatial
import cuspatial.utils
import cuspatial.utils.column_utils
import time
import pyarrow.parquet as pq
import multiprocessing as mp
from osgeo import ogr
from concurrent.futures import ProcessPoolExecutor # ThreadPoolExecutor locks GIL
import argparse
import gc

BATCH_SIZE = 1 << 21
# -n 5 -q 0.5 0.5 1 1 -q 0.3 0.3 0.8 0.8 -q 0.1 0.1 0.6 0.6 -q 0 0 0.5 0.5 /home/aske/dev/geo-rt-index/data/duniform_p26_s13573.parquet

# arg parsing from ChatGPT
parser = argparse.ArgumentParser(description='Cuspatial runner')
parser.add_argument('-n', type=int, help='Number of repetitions', required=True)
parser.add_argument('-q', nargs=4, type=float, action='append', help='Bounding box (minx, miny, maxx, maxy)', required=True)
parser.add_argument('file', nargs='+', help='File paths')
args = parser.parse_args()
n = args.n
queries = args.q
files = args.file

pool = ProcessPoolExecutor(mp.cpu_count())

def work(batch):
  geom_col = batch["geometry"]
  result = []
  for geom in geom_col:
    g = ogr.CreateGeometryFromWkb(geom.as_py())
    result.append(g.GetX())
    result.append(g.GetY())
  return result

t = time.perf_counter()
xy = []
for file in files:
  # print("file:", file)
  data = pq.read_table(file)
  batches: list = data.to_batches(BATCH_SIZE) # 1m
  for result in pool.map(work, batches):
    xy += result

print(f"load + convert: {time.perf_counter() - t:.3f}s.")
pool.shutdown()

for i in range(n):
  from_xy_time = 0.0
  query_time = 0.0

  t = time.perf_counter()
  points = cuspatial.GeoSeries.from_points_xy(xy)
  from_xy_time += time.perf_counter() - t
  for query in queries:
    minx = query[0]
    miny = query[1]
    maxx = query[2]
    maxy = query[3]

    t = time.perf_counter()
    cuspatial.points_in_spatial_window(points, minx, maxx, miny, maxy) # beware that it is minx, maxx, miny, maxy
    query_time += time.perf_counter() - t

    gc.collect()

  points = None
  print(f"from_points_xy: {from_xy_time:.3f}s.")
  print(f"queries took: {query_time:.3f}s.")

exit()

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
  File "/home/aske/dev/py_cuspatial/cuspatial.py", line 44, in <module>
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