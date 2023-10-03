
add_library(winograd_backend OBJECT avx_winograd_kernels.cpp
								avx2_winograd_kernels.cpp
								avx512f_winograd_kernels.cpp
								def_winograd_kernels.cpp
								sse2_winograd_kernels.cpp
								winograd_runtime.cpp)
								
set_target_properties(winograd_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(winograd_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")

set_source_files_properties(sse2_winograd_kernels.cpp PROPERTIES COMPILE_OPTIONS "-msse2")
set_source_files_properties(avx_winograd_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
set_source_files_properties(avx2_winograd_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(avx512f_winograd_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma;-mf16c")

target_link_libraries(${LibName} PRIVATE winograd_backend)