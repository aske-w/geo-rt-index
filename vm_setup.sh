#! /bin/bash

sudo apt update
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt install -y -V python3.10 python3.10-distutils python3.10-dev python-is-python3
# sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8  2
# sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
# sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2
# echo 1 | sudo update-alternatives --config python
# echo 1 | sudo update-alternatives --config python3

sudo apt install -y -V ca-certificates lsb-release wget libc6 libproj-dev swig 
# shellcheck disable=SC2019,SC2018
wget "https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb"
sudo apt install -y -V "./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb"
sudo apt update
sudo apt install -y libarrow-dev libparquet-dev

sudo apt install python3-pip
# curl https://bootstrap.pypa.io/get-pip.py -O
# python3.10 get-pip.py
# # shellcheck disable=SC2016
# {
#     echo 'export PATH="/home/ucloud/.local/bin:$PATH"'
#     echo 'export PYTHONPATH="/home/ucloud/.local/lib/python3/dist-packages:$PYTHONPATH"'
# } >> ~/.profile
# # shellcheck disable=SC1090
# source ~/.profile

pip3.10 install cmake "numpy<1.25"
sudo apt install -y cmake-curses-gui
pip3.10 install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11==24.4.* cuspatial-cu11==24.4.* cuproj-cu11==24.4.*

git clone https://github.com/OSGeo/gdal.git
mkdir -p gdal/build
cd gdal/build || exit 1
git checkout v3.8.5
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_INSTALL_PREFIX=~/.local \
    -DBUILD_APPS=OFF \
    -DBUILD_GMOCK=OFF \
    -DBUILD_TESTING=OFF
make -j40 install
# shellcheck disable=SC2016
{
    echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$HOME/.local/lib"'
    echo 'export LIBRARY_PATH="${LD_LIBRARY_PATH}:$HOME/.local/lib"'  
    echo 'export INCLUDE_PATH="${LD_LIBRARY_PATH}:$HOME/.local/include"' 
} >> ~/.profile
# shellcheck disable=SC1090
source ~/.profile
cd ~ || exit 1
git clone https://github.com/aske-w/geo-rt-index.git