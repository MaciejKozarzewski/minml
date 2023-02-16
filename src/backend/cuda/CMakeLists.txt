include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

set(codesrc 	kernels/activations.cu
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
		utils.cpp)

if (WIN32)
	set(CUDA_BACKEND_NAME "cuda_backend")
	add_library(${CUDA_BACKEND_NAME} SHARED ${codesrc})
	set_target_properties(${CUDA_BACKEND_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
	set(CUDA_BACKEND_NAME "cuda_backend_static")
	add_library(${CUDA_BACKEND_NAME} STATIC ${codesrc})
endif()


set_target_properties(${CUDA_BACKEND_NAME} PROPERTIES CUDA_STANDARD 14)
set_target_properties(${CUDA_BACKEND_NAME} PROPERTIES CUDA_STANDARD_REQUIRED YES)
set_target_properties(${CUDA_BACKEND_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CUDA_BACKEND_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_property(TARGET ${CUDA_BACKEND_NAME} PROPERTY CUDA_ARCHITECTURES all)
target_compile_definitions(${CUDA_BACKEND_NAME} PRIVATE USE_CUDA)
target_include_directories(${CUDA_BACKEND_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include" "/usr/local/cuda/include")

target_link_libraries(MinML PRIVATE ${CUDA_BACKEND_NAME})

