#!/bin/bash

set -exu

bash --version

EXCLUDE=(
    "demos/apple"
    "demos/FiniteVolume/BZ"
    "demos/FiniteVolume/stokes_2d.cpp"
    "demos/FiniteVolume/heat_1d.cpp"
    "demos/highorder"
    "demos/fromobj"
    "demos/multigrid"
    "demos/LBM"
    "demos/Weno"
    "include/samurai/petsc"
)

N=4

EXCLUDE_FROM_FIND=`echo ${EXCLUDE[@]} | sed "s/ /|/g"`

DIR=(
    "demos"
    "include"
)

# i=0
for d in ${DIR[@]}; do
    FILES=`find $d -type f \( -name "*.hpp" -o -name "*.cpp" \) | grep -Ev $EXCLUDE_FROM_FIND`
    for FILE in $FILES; do
        # ((i=i%N)); ((i++==0)) && wait
        echo ------------$FILE------------
        clang-tidy --warnings-as-errors='1' $FILE -- -I$CONDA_PREFIX/include -Iinclude --std=c++17
    done
done
