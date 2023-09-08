target_sources(${LibName} PRIVATE 	batchnorm.cpp
				ComputeConfig.cpp
				conv_im2row.cpp
				cpu_context.cpp
				cpu_memory.cpp
				cpu_properties.cpp
				cpu_x86.cpp
				dispatcher.cpp
				fp16.cpp
				openblas_gemm.cpp
				training.cpp
				utils.cpp)
								
add_subdirectory("kernels")
add_subdirectory("gemm")
add_subdirectory("winograd")
