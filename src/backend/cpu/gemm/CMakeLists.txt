
add_library(gemm_backend OBJECT avx_gemm_kernels.cpp
								avx2_gemm_kernels.cpp
								def_gemm_kernels.cpp
								gemm_runtime.cpp
								gemm.cpp
								sse2_gemm_kernels.cpp)
								
set_target_properties(gemm_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(gemm_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")

set_source_files_properties(sse2_gemm_kernels.cpp PROPERTIES COMPILE_OPTIONS "-msse2")
set_source_files_properties(avx_gemm_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
set_source_files_properties(avx2_gemm_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")

target_link_libraries(MinML PRIVATE gemm_backend)