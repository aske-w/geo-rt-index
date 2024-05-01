#! /bin/bash

sudo apt update
sudo apt install -y -V ca-certificates lsb-release wget libc6 libproj-dev swig python3-pip pbzip2 g++-10
# shellcheck disable=SC2019,SC2018
wget "https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb"
sudo apt install -y -V "./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb"
sudo apt update
sudo apt install -y libarrow-dev libparquet-dev

# shellcheck disable=SC2016
{
    echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$HOME/.local/lib"'
    echo 'export LIBRARY_PATH="${LIBRARY_PATH}:$HOME/.local/lib"'  
    echo 'export INCLUDE_PATH="${INCLUDE_PATH}:$HOME/.local/include"' 
    echo 'export PATH="/home/ucloud/.local/bin:$PATH"'
    echo 'export PYTHONPATH="/home/ucloud/.local/lib/python3/dist-packages:$PYTHONPATH"'
} >> ~/.profile
# shellcheck disable=SC1090
source ~/.profile

pip3.10 install cmake "numpy<1.25"
sudo apt install -y cmake-curses-gui
pip3.10 install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11==24.4.* cuspatial-cu11==24.4.* cuproj-cu11==24.4.*

cd ~ || exit 1
git clone https://github.com/OSGeo/gdal.git
mkdir -p gdal/build
cd gdal/build || exit 1
git checkout v3.8.5
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_INSTALL_PREFIX=~/.local \
    -DBUILD_APPS=OFF \
    -DBUILD_GMOCK=OFF \
    -DBUILD_TESTING=OFF \
    -DENABLE_IPO=ON \
    -DCMAKE_CXX_FLAGS="-march=native"
make -j40 install

cd ~ || exit 1
git clone https://github.com/aske-w/geo-rt-index.git
cd geo-rt-index || exit 1
mkdir -p build/release
cmake .. -DCMAKE_INSTALL_PREFIX:PATH="/home/ucloud/.local;/usr/local;/home/ucloud/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/SDK/CMake" \
    -DCMAKE_MODULE_PATH=/home/ucloud/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/SDK/CMake \
    -DOptiX_INSTALL_DIR:PATH=/home/ucloud/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64 \
    -DOptiX_INCLUDE:PATH=/home/ucloud/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/include \
    -DCUDA_CUDA_LIBRARY=/usr/lib/x86_64-linux-gnu/libcuda.so \
    -DBIN2C=/usr/bin/bin2c \
    -DUSE_DEBUG=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON \
    -DDEBUG_PRINT:BOOL=OFF \
    -DCUDA_USE_STATIC_CUDA_RUNTIME:BOOL=ON \
    -DCMAKE_CXX_FLAGS:STRING="-I/usr/include -march=native -std=c++20 -I/home/ucloud/.local/include" \
    -DCMAKE_CUDA_FLAGS:STRING="-I/home/ucloud/.local/include" \
    -DUSE_MEASURE_TIME:BOOL=ON \
    -DVERIFICATION_MODE:BOOL=ON \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
    -DBUILD_TESTS=OFF