#!/bin/bash

/Applications/CMake.app/Contents/bin/cmake \
-DCMAKE_CXX_COMPILER=/usr/bin/g++ \
-DCMAKE_C_COMPILER=/usr/bin/gcc \
-DCMAKE_Fortran_COMPILER=/usr/local/bin/gfortran \
-DCMAKE_BUILD_TYPE=Release \
-DMADLIB_INSTALL_EXTENDED_API=ON \
-DMADLIB_BUILD_STATIC_LIBRARY=ON \
-DMADLIB_BUILD_SHARED_LIBRARY=OFF \
-DCMAKE_INSTALL_PREFIX=/Users/armandyam/opt/madlib/MeshAdapt.8.09.2017/install \
/Users/armandyam/opt/madlib/MeshAdapt.8.09.2017/
