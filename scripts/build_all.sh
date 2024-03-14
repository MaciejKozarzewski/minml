#!/bin/bash
cd ..
echo "$PWD"
mkdir build
cmake -B build/cpu/Release -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_OPENCL=OFF
cmake --build build/cpu/Release
cmake -B build/cpu/Debug -DCMAKE_BUILD_TYPE=Debug -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_OPENCL=OFF
cmake --build build/cpu/Debug

cmake -B build/cuda/Release -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=ON -DBUILD_WITH_OPENCL=OFF
cmake --build build/cuda/Release
cmake -B build/cuda/Debug -DCMAKE_BUILD_TYPE=Debug -DBUILD_WITH_CUDA=ON -DBUILD_WITH_OPENCL=OFF
cmake --build build/cuda/Debug

cmake -B build/opencl/Release -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_OPENCL=ON
cmake --build build/opencl/Release
cmake -B build/opencl/Debug -DCMAKE_BUILD_TYPE=Debug -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_OPENCL=ON
cmake --build build/opencl/Debug
