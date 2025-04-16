target_sources(${LibName} PRIVATE attention.cpp
				conv_im2row.cpp
				cpu_context.cpp
				cpu_event.cpp
				cpu_memory.cpp
				cpu_properties.cpp
				cpu_x86.cpp
				def_misc_kernels.cpp
				dispatcher.cpp
				fp16.cpp
				normalization.cpp
				training.cpp
				utils.cpp)
				
add_library(misc_backend OBJECT avx_misc_kernels.cpp)
								
set_target_properties(misc_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(misc_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")

set_source_files_properties(avx_misc_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx;-mf16c")
target_link_libraries(${LibName} PRIVATE misc_backend)
								
add_subdirectory("gemm")
add_subdirectory("kernels")
add_subdirectory("winograd")
