from time import sleep
import cuspatial
import cudf
import cupy
import geopandas as gp
import pandas as pd
import numpy as np
import cuda
import time
import shapely
from shapely.geometry import *
from shapely import wkt



print(cuda._version.get_versions())
print(cupy.is_available())

NUM = (1 << 25) + (3 * 1 << 23) + (1 << 22) # 62,914,560
# NUM = (1 << 26) # 67,108,864 - see error in bottom
# NUM = 10
np.random.seed(0)
cupy.random.seed(0)


t = time.perf_counter()
x_points = (cupy.random.random(NUM) - 0.5) * 360
y_points = (cupy.random.random(NUM) - 0.5) * 180
print(f"generating {NUM:,} xy points took: {time.perf_counter() - t:.3f} ms")
# sleep(3)


t = time.perf_counter()
xy = cudf.DataFrame({"x": x_points, "y": y_points}).interleave_columns()
print(f"interleave_columns took: {time.perf_counter() - t:.3f} ms")
# sleep(3)


t = time.perf_counter()
points = cuspatial.GeoSeries.from_points_xy(xy)
del xy, x_points, y_points
print(f"Creating points from interleaved columns took: {time.perf_counter() - t:.3f} ms")


hit_start = time.perf_counter()
hit = cuspatial.points_in_spatial_window(points, 140, 180, 50, 60)
print("hit took ", time.perf_counter() - hit_start)
print(hit.head()) # lmao
sleep(3)
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