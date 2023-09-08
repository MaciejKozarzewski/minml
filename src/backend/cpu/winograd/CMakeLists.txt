target_sources(${LibName} PRIVATE 	avx_winograd_kernels.cpp
				avx2_winograd_kernels.cpp
				def_winograd_kernels.cpp
				sse2_winograd_kernels.cpp)