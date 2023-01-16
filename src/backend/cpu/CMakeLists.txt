target_sources(MinML PRIVATE 	batchnorm.cpp
				cpu_memory.cpp
				cpu_properties.cpp
				dispatcher.cpp
				gemms.cpp
				misc.cpp
				training.cpp
				utils.cpp)
								
set(kernelsrc 	kernels/activations.cpp
		kernels/add_bias_act.cpp
		kernels/conv_explicit_gemm.cpp
		kernels/conv_implicit_gemm.cpp
		kernels/conv_winograd_fused.cpp
		kernels/conv_winograd_nonfused.cpp
		kernels/conversion.cpp
		kernels/global_pooling.cpp)
				
if(DYNAMIC_ARCH)
	target_compile_definitions(MinML PRIVATE DYNAMIC_ARCH=1)
	add_library(base_backend OBJECT ${kernelsrc})
	target_compile_definitions(base_backend PRIVATE COMPILE_COMMON_CODE)
	target_compile_definitions(base_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(base_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(base_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")
	
	add_library(sse2_backend OBJECT ${kernelsrc})
	target_compile_options(sse2_backend PRIVATE -msse2)
	target_compile_definitions(sse2_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(sse2_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(sse2_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")
	
	add_library(sse41_backend OBJECT ${kernelsrc})
	target_compile_options(sse41_backend PRIVATE -msse4.1)
	target_compile_definitions(sse41_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(sse41_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(sse41_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")
	
	add_library(avx_backend OBJECT ${kernelsrc})
	target_compile_options(avx_backend PRIVATE -mavx -mf16c)
	target_compile_definitions(avx_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(avx_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(avx_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")
	
	add_library(avx2_backend OBJECT ${kernelsrc})
	target_compile_options(avx2_backend PRIVATE -mavx2 -mfma -mf16c)
	target_compile_definitions(avx2_backend PRIVATE CPU_BACKEND DYNAMIC_ARCH=1)
	set_target_properties(avx2_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_include_directories(avx2_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")
	
	target_link_libraries(MinML PRIVATE base_backend sse2_backend sse41_backend avx_backend avx2_backend)
	
else()
	target_compile_definitions(MinML PRIVATE DYNAMIC_ARCH=0)
	target_sources(MinML PRIVATE ${kernelsrc})
	target_compile_options(MinML PRIVATE -march=native)
#	target_compile_options(MinML PRIVATE -fsanitize=address -fsanitize=undefined)
endif()
