set(codesrc 	activations.cpp
				attention.cpp
				conversion.cpp
				depthwise_conv.cpp
				gemms.cpp
				global_pooling.cpp
				kernel_table.cpp
				normalization.cpp
				opencl_context.cpp
				opencl_event.cpp
				opencl_memory.cpp
				opencl_properties.cpp
				training.cpp
				utils.cpp
				winograd_nonfused.cpp)

set(OPENCL_BACKEND_NAME "opencl_backend_static")
add_library(${OPENCL_BACKEND_NAME} STATIC ${codesrc})

target_link_libraries(${LibName} PUBLIC ${OPENCL_BACKEND_NAME})
target_link_libraries(${OPENCL_BACKEND_NAME} PUBLIC OpenCL clblast)

set_target_properties(${OPENCL_BACKEND_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_compile_definitions(${OPENCL_BACKEND_NAME} PRIVATE USE_OPENCL)
target_include_directories(${OPENCL_BACKEND_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")


