target_sources(MinML PRIVATE 	batchnorm.cpp
				ComputeConfig.cpp
				cpu_context.cpp
				cpu_memory.cpp
				cpu_properties.cpp
				cpu_x86.cpp
				dispatcher.cpp
				openblas_gemm.cpp
				training.cpp
				utils.cpp)
								
add_subdirectory("kernels")
add_subdirectory("gemm")
