#!/bin/bash

mkdir build_conda
conda build build_conda geometric-smote
dirs=`ls -l "$PWD/build_conda"`
for d in build_conda/*/ ; do
    for file in $d*
    do
        if [[ -f $file ]]; then
            anaconda upload --user AlgoWit $file
        fi
    done
done
rm -r build_conda