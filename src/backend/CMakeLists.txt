add_subdirectory("cpu")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_WITH_CUDA)
	add_subdirectory("cuda")
else()
	target_sources(MinML PRIVATE 	cuda_dummy_backend.cpp)
endif()


