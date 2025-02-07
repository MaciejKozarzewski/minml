mkdir ../build

cd ..
cmake -B build/cpu/Release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_OPENCL=OFF
cmake -B build/cuda/Release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=ON -DBUILD_WITH_OPENCL=OFF
cmake -B build/opencl/Release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_OPENCL=ON

cmake -B build/cpu/Debug -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_OPENCL=OFF
cmake -B build/cuda/Debug -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DBUILD_WITH_CUDA=ON -DBUILD_WITH_OPENCL=OFF
cmake -B build/opencl/Debug -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_OPENCL=ON

call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"
nvcc -ccbin cl -m64 -arch=all -std=c++17 -O3 -DUSE_CUDA -DIN_THE_DLL -DNDEBUG -I"include/" -default-stream per-thread -DUSE_CUDA -DBUILDING_DLL -shared -o "build/cudaml.dll" "src/backend/cuda/cuda_context.cpp" "src/backend/cuda/cuda_memory.cu" "src/backend/cuda/cuda_properties.cpp" "src/backend/cuda/cudnn_conv.cpp" "src/backend/cuda/gemms.cpp" "src/backend/cuda/utils.cpp" "src/backend/cuda/kernels/activations.cu" "src/backend/cuda/kernels/attention.cu" "src/backend/cuda/kernels/conversion.cu" "src/backend/cuda/kernels/depthwise_conv.cu" "src/backend/cuda/kernels/global_pooling.cu" "src/backend/cuda/kernels/normalization.cu" "src/backend/cuda/kernels/quantized.cu" "src/backend/cuda/kernels/softmax.cu" "src/backend/cuda/kernels/training.cu" "src/backend/cuda/kernels/winograd_nonfused.cu" cudart.lib cublas.lib

cmake --build build/cpu/Release
cmake --build build/cuda/Release
cmake --build build/opencl/Release

cmake --build build/cpu/Debug
cmake --build build/cuda/Debug
cmake --build build/opencl/Debug