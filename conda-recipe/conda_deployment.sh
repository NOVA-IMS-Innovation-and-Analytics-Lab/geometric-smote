#!/bin/bash

mkdir build_conda
conda config --set anaconda_upload yes
conda build --output-folder ./build_conda --user gdouzas geometric-smote
rm -r build_conda
