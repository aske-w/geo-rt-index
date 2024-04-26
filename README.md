# geo-rt-index

## How to add OptiX CMake module

1. Download and install OptiX from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Add the following CMake cache variables
   - `<optix install directory>/SDK/CMake` to CMAKE_INSTALL_PREFIX
   - `<optix install directory>/SDK/CMake` to CMAKE_MODULE_PATH
   - `<optix install directory>/include` to OptiX_INCLUDE
   - `<optix install directory>` to OptiX_INSTALL_DIR

## GDAL SQL

```bash
ogrinfo -dialect sqlite -sql "select count(*) from duniform_n10 where (ST_X(geometry) <= 0.1)" duniform_n10.parquet
```

## pbzip2

### Compression

```
pbzip2 -1zkvf -m2000
```

