add_library(kernels_backend OBJECT avx_kernels.cpp
								def_kernels.cpp
								pooling_and_scaling.cpp
								sse2_kernels.cpp)
								
set_target_properties(kernels_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(kernels_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")

set_source_files_properties(sse2_kernels.cpp PROPERTIES COMPILE_OPTIONS "-msse2")
set_source_files_properties(avx_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx;-mf16c")

target_link_libraries(${LibName} PRIVATE kernels_backend)