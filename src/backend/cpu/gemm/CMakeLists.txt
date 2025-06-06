add_library(gemm_backend OBJECT avx_kernels.cpp
								avx2_kernels.cpp
								avx512f_kernels.cpp
								def_kernels.cpp
								fused_conv_block_runtime.cpp
								gemm_runtime.cpp
								gemm.cpp
								mha_runtime.cpp
								sse2_kernels.cpp)
								
set_target_properties(gemm_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(gemm_backend PUBLIC PUBLIC "${PROJECT_SOURCE_DIR}/include")

set_source_files_properties(sse2_kernels.cpp PROPERTIES COMPILE_OPTIONS "-msse2")
set_source_files_properties(avx_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
set_source_files_properties(avx2_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(avx512f_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma;-mf16c")

target_link_libraries(${LibName} PRIVATE gemm_backend)