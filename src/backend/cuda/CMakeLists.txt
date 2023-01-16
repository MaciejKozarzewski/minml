include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 14)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(cuda_backend OBJECT kernels/activations.cu
				kernels/batchnorm.cu
				kernels/conversion.cu
				kernels/implicit_gemm_conv.cu
				kernels/training.cu
				kernels/winograd_fused.cu
				kernels/winograd_nonfused.cu
				cuda_context.cpp
				cuda_memory.cu
				cuda_properties.cpp
				gemms.cpp
				misc.cpp
				utils.cpp)

set_target_properties(cuda_backend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(cuda_backend PRIVATE USE_CUDA)
target_include_directories(cuda_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "/usr/local/cuda/include")

target_link_libraries(MinML PRIVATE cuda_backend)
target_link_libraries(MinML PUBLIC cudart cublas)

