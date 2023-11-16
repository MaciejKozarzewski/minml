if (WIN32)
	add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/bin/cudaml.dll" COMMAND "scripts/windows_cuda_build.bat")
	target_link_libraries(${LibName} PUBLIC "${CMAKE_BINARY_DIR}/bin/cudaml.dll")
else()
	find_package(CUDAToolkit)
	enable_language(CUDA)

	set(codesrc 	kernels/activations.cu
		kernels/batchnorm.cu
		kernels/conversion.cu
		kernels/training.cu
		kernels/winograd_nonfused.cu
		cuda_context.cpp
		cuda_memory.cu
		cuda_properties.cpp
		cudnn_conv.cpp
		gemms.cpp
		utils.cpp)

	set(CUDA_BACKEND_NAME "cuda_backend_static")
	add_library(${CUDA_BACKEND_NAME} STATIC ${codesrc})

	target_link_libraries(${LibName} PUBLIC ${CUDA_BACKEND_NAME})
	target_link_directories(${CUDA_BACKEND_NAME} PUBLIC ${CUDAToolkit_LIBRARY_DIR})
	target_link_libraries(${CUDA_BACKEND_NAME} PUBLIC cudart cublas)

	if(BUILD_WITH_CUDNN)
		target_compile_definitions(${CUDA_BACKEND_NAME} PUBLIC USE_CUDNN)
		target_link_libraries(${CUDA_BACKEND_NAME} PUBLIC cudnn_ops_infer cudnn_cnn_infer cublasLt)
	endif()

	set_target_properties(${CUDA_BACKEND_NAME} PROPERTIES CUDA_STANDARD 14)
	set_target_properties(${CUDA_BACKEND_NAME} PROPERTIES CUDA_STANDARD_REQUIRED YES)
	set_target_properties(${CUDA_BACKEND_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(${CUDA_BACKEND_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

	set_property(TARGET ${CUDA_BACKEND_NAME} PROPERTY CUDA_ARCHITECTURES all)
	target_compile_definitions(${CUDA_BACKEND_NAME} PRIVATE USE_CUDA)
	target_include_directories(${CUDA_BACKEND_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include" ${CUDAToolkit_INCLUDE_DIRS})
endif()

