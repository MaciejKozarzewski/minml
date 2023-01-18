add_executable(MinML_test	test_launcher.cpp
				backend/test_batchnorm.cpp
				backend/test_conv2D.cpp
				backend/test_gemm_conv.cpp
				backend/test_gemms.cpp
				backend/test_nonlinearity.cpp
				backend/test_tensor_op.cpp
				backend/test_winograd.cpp
				core/test_Shape.cpp
				core/test_Tensor.cpp
				utils/test_json.cpp
				utils/test_serialization.cpp
				utils/test_string_util.cpp
				utils/test_time_util.cpp)
				
set_target_properties(MinML_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(MinML_test PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(MinML_test PUBLIC gtest MinML)
