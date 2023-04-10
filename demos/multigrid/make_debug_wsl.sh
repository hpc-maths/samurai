#!/bin/bash

PROJECT_PATH=/mnt/c/Users/pierr/Documents/Source/Repos/samurai_mg

$CONDA_PREFIX/bin/mpicxx -DFMT_SHARED -I$PROJECT_PATH/include -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O0 -ffunction-sections -pipe -isystem $CONDA_PREFIX/include -g -MD -MT demos/Coarsening/CMakeFiles/multigrid.dir/main.cpp.o -MF CMakeFiles/multigrid.dir/main.cpp.o.d -o CMakeFiles/multigrid.dir/main.cpp.o -c $PROJECT_PATH/demos/multigrid/main.cpp

$CONDA_PREFIX/bin/mpicxx -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O0 -ffunction-sections -pipe -isystem $CONDA_PREFIX/include -g -Wl,-O0 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,$CONDA_PREFIX/lib -Wl,-rpath-link,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib CMakeFiles/multigrid.dir/main.cpp.o -o multigrid  -lpetsc -lpugixml -lhdf5 $CONDA_PREFIX/lib/libfmt.so.9.1.0
