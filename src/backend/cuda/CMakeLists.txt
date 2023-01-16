include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

add_library(cuda_backend SHARED kernels/activations.cu
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

set_target_properties(cuda_backend PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#set_target_properties(cuda_backend PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_property(TARGET cuda_backend PROPERTY CUDA_ARCHITECTURES all)
target_compile_definitions(cuda_backend PRIVATE USE_CUDA)
target_include_directories(cuda_backend PUBLIC "${PROJECT_SOURCE_DIR}/include" "/usr/local/cuda/include")

target_link_libraries(MinML PRIVATE cuda_backend)

