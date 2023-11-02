/*
 * selfcheck.cpp
 *
 *  Created on: Jun 12, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/selfcheck.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/DataType.hpp>
#include <minml/core/Device.hpp>
#include <minml/core/math.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/layers/Layer.hpp>

#include <minml/backend/cpu_backend.h>
#include <minml/backend/cuda_backend.h>
#include <minml/backend/opencl_backend.h>

#include <vector>
#include <memory>
#include <iostream>

namespace
{
	using namespace ml;

	int square(int x) noexcept
	{
		return x * x;
	}

	struct GemmTestConfig
	{
			int M, N, K;
			GemmTestConfig(int m, int n, int k) noexcept :
					M(m),
					N(n),
					K(k)
			{
			}
	};

	std::vector<DataType> get_datatypes_list(Device device)
	{
		std::vector<DataType> result;
		if (device.supportsType(DataType::FLOAT32))
			result.push_back(DataType::FLOAT32);
		if (device.supportsType(DataType::FLOAT16))
			result.push_back(DataType::FLOAT16);
		return result;
	}

	std::vector<GemmTestConfig> get_matrix_dimensions(Device device)
	{
		std::vector<GemmTestConfig> result;
		if (device.isCPU())
		{
			switch (device.cpuSimdLevel())
			{
				case CpuSimd::NONE:
				case CpuSimd::SSE:
					result.emplace_back(4, 4, 77);
					break;
				case CpuSimd::SSE2:
				case CpuSimd::SSE3:
				case CpuSimd::SSSE3:
				case CpuSimd::SSE41:
				case CpuSimd::SSE42:
					result.emplace_back(8, 4, 77);
					result.emplace_back(4, 4, 77);
					break;
				case CpuSimd::AVX:
					result.emplace_back(10, 8, 77);
					result.emplace_back(8, 8, 77);
					break;
				case CpuSimd::AVX2:
					result.emplace_back(6, 16, 77);
					result.emplace_back(24, 4, 77);
					break;
				case CpuSimd::AVX512F:
				case CpuSimd::AVX512VL_BW_DQ:
					break;
			}
		}
		if (device.isCUDA())
			result.emplace_back(123, 46, 78);
		return result;
	}
	std::vector<std::pair<char, char>> get_ops_list()
	{
		std::vector<std::pair<char, char>> result;
		result.emplace_back('N', 'N');
		result.emplace_back('N', 'T');
		result.emplace_back('T', 'N');
		result.emplace_back('T', 'T');
		return result;
	}
	Tensor create_test_tensor(const Shape &shape, Device device, DataType dtype)
	{
		Tensor result(shape, dtype, device);
		testing::initForTest(result, 0.0);
		return result;
	}

	std::vector<int> get_winograd_transform_sizes(Device device, int filter_size)
	{
		assert(filter_size == 3 || filter_size == 5);
		switch (device.type())
		{
			case DeviceType::CPU:
				return (filter_size == 3) ? std::vector<int>( { 2, 4, 5 }) : std::vector<int>( { 2 });
			case DeviceType::CUDA:
				return (filter_size == 3) ? std::vector<int>( { 2, 4 }) : std::vector<int>( { 2 });
			default:
				return std::vector<int>();
		}
	}
}

namespace ml
{
	int checkDeviceDetection(Device device)
	{
		try
		{
			std::cout << "Starting checkDeviceDetection(" << device.toString() << ")" << std::endl;
			switch (device.type())
			{
				case DeviceType::CPU:
				{
					cpu_print_device_features();
					std::cout << "Supports fp32 : " << (cpu_supports_type(DTYPE_FLOAT32) ? "YES" : "NO") << '\n';
					std::cout << "Supports fp16 : " << (cpu_supports_type(DTYPE_FLOAT16) ? "YES" : "NO") << '\n';
					return 0;
				}
				case DeviceType::CUDA:
				{
					cuda_print_device_features(device.index());
					std::cout << "Supports fp32 : " << (cuda_supports_type(device.index(), DTYPE_FLOAT32) ? "YES" : "NO") << '\n';
					std::cout << "Supports fp16 : " << (cuda_supports_type(device.index(), DTYPE_FLOAT16) ? "YES" : "NO") << '\n';
					return 0;
				}
				case DeviceType::OPENCL:
				{
					opencl_print_device_features(device.index());
					std::cout << "Supports fp32 : " << (opencl_supports_type(device.index(), DTYPE_FLOAT32) ? "YES" : "NO") << '\n';
					std::cout << "Supports fp16 : " << (opencl_supports_type(device.index(), DTYPE_FLOAT16) ? "YES" : "NO") << '\n';
					return 0;
				}
				default:
					std::cout << "Invalid device '" << device.toString() << '\n';
					return 1;
			}
		} catch (std::exception &e)
		{
			std::cout << "Caught exception '" << e.what() << "'" << std::endl;
			return 1;
		}
	}

	int checkWinogradTransforms(Device device)
	{
		const std::vector<DataType> dtypes = get_datatypes_list(device);
		const std::vector<int> filter_sizes( { 3, 5 });

		constexpr int batch_size = 7;
		constexpr int height = 12;
		constexpr int width = 13;
		constexpr int filters_in = 45;
		constexpr int filters_out = 23;

		try
		{
			std::cout << "Starting checkWinogradTransforms(" << device.toString() << ")" << std::endl;
			Context context(device);
			std::cout << "Created context" << std::endl;

			for (auto dt = dtypes.begin(); dt < dtypes.end(); dt++)
				for (auto fs = filter_sizes.begin(); fs < filter_sizes.end(); fs++)
				{
					const std::vector<int> transform_sizes = get_winograd_transform_sizes(device, *fs);
					for (auto ts = transform_sizes.begin(); ts < transform_sizes.end(); ts++)
					{
						std::cout << "Checking Winograd transforms with : tile size=" << *ts << ", filter size=" << *fs << ", data type = " << *dt
								<< std::endl;

						const int tile_size = *ts + *fs - 1;
						const int tiles_h = (height + *ts - 1) / (*ts);
						const int tiles_w = (width + *ts - 1) / (*ts);

						/*
						 *  weight transform
						 */
						const Shape weight_shape( { filters_out, *fs, *fs, filters_in });
						const Shape weight_matrices_shape( { square(tile_size), filters_out, filters_in });
						std::cout << "  Created shapes " << weight_shape.toString() << ", " << weight_matrices_shape.toString() << std::endl;

						const Tensor weights = create_test_tensor(weight_shape, device, *dt);
						std::cout << "  Allocated weights tensor=" << weights.info() << std::endl;
						Tensor weights_matrices = create_test_tensor(weight_matrices_shape, device, *dt);
						std::cout << "  Allocated weight matrices=" << weights_matrices.info() << std::endl;

						std::cout << "  Running weight transform" << std::endl;
						winogradWeightTransform(context, weights, weights_matrices, false);
						std::cout << "    WORKS" << std::endl;

						/*
						 *  input transform
						 */
						const Shape input_shape( { batch_size, height, width, filters_in });
						const Shape input_matrices_shape( { square(tile_size), batch_size * tiles_h * tiles_w, filters_in });
						std::cout << "  Created shapes " << input_shape.toString() << ", " << input_matrices_shape.toString() << std::endl;

						const Tensor input = create_test_tensor(input_shape, device, *dt);
						std::cout << "  Allocated input tensor=" << input.info() << std::endl;
						Tensor input_matrices = create_test_tensor(input_matrices_shape, device, *dt);
						std::cout << "  Allocated input matrices=" << input_matrices.info() << std::endl;

						std::cout << "  Running input transform" << std::endl;
						winogradInputTransform(context, weight_shape, input, input_matrices);
						std::cout << "    WORKS" << std::endl;

						/*
						 *  output transform
						 */
						const Shape output_shape( { batch_size, height, width, filters_out });
						const Shape output_matrices_shape( { square(tile_size), batch_size * tiles_h * tiles_w, filters_out });
						const Shape bias_shape( { filters_out });
						std::cout << "  Created shapes " << output_shape.toString() << ", " << output_matrices_shape.toString() << ", "
								<< bias_shape.toString() << std::endl;

						Tensor output = create_test_tensor(output_shape, device, *dt);
						std::cout << "  Allocated output tensor=" << output.info() << std::endl;
						const Tensor output_matrices = create_test_tensor(output_matrices_shape, device, *dt);
						std::cout << "  Allocated output matrices=" << output_matrices.info() << std::endl;
						const Tensor bias = create_test_tensor(bias_shape, device, *dt);
						std::cout << "  Allocated bias=" << bias.info() << std::endl;

						std::cout << "  Running output transform" << std::endl;
						winogradOutputTransform(context, weight_shape, output_matrices, output, bias, output, ActivationType::RELU);
						std::cout << "    WORKS" << std::endl;
					}
				}

		} catch (std::exception &e)
		{
			std::cout << "Caught exception '" << e.what() << "'" << std::endl;
			return 1;
		}
		return 0;
	}

	int checkMatrixMultiplication(Device device)
	{
		const std::vector<DataType> dtypes = get_datatypes_list(device);
		const std::vector<GemmTestConfig> configs = get_matrix_dimensions(device);
		const std::vector<std::pair<char, char>> ops = get_ops_list();

		try
		{
			std::cout << "Starting checkMatrixMultiplication(" << device.toString() << ")" << std::endl;
			Context context(device);
			std::cout << "Created context" << std::endl;

			for (auto dt = dtypes.begin(); dt < dtypes.end(); dt++)
				for (auto cfg = configs.begin(); cfg < configs.end(); cfg++)
					for (auto op = ops.begin(); op < ops.end(); op++)
					{
						std::cout << "Checking GEMM_" << op->first << op->second << " with : M=" << cfg->M << ", N=" << cfg->N << ", K=" << cfg->K
								<< ", data type = " << toString(*dt) << ", compute type=" << toString(DataType::FLOAT32) << std::endl;
						const Shape shape_a = (op->first == 'N') ? Shape( { cfg->M, cfg->K }) : Shape( { cfg->K, cfg->M });
						const Shape shape_b = (op->second == 'N') ? Shape( { cfg->K, cfg->N }) : Shape( { cfg->N, cfg->K });
						const Shape shape_c( { cfg->M, cfg->N });
						std::cout << "  Created shapes " << shape_a.toString() << ", " << shape_b.toString() << ", " << shape_c.toString()
								<< std::endl;

						const Tensor A = create_test_tensor(shape_a, device, *dt);
						std::cout << "  Allocated tensor A=" << A.info() << std::endl;
						const Tensor B = create_test_tensor(shape_b, device, *dt);
						std::cout << "  Allocated tensor B=" << B.info() << std::endl;
						Tensor C = create_test_tensor(shape_c, device, *dt);
						std::cout << "  Allocated tensor C=" << C.info() << std::endl;

						const float alpha = 1.1f;
						const float beta = 0.1f;

						std::cout << "  Running matrix multiplication" << std::endl;
						gemm(context, op->first, op->second, C, A, B, alpha, beta);
						std::cout << "    WORKS" << std::endl;
					}
		} catch (std::exception &e)
		{
			std::cout << "Caught exception '" << e.what() << "'" << std::endl;
			return 1;
		}
		return 0;
	}

	int checkActivationFuncion(Device device)
	{
		const std::vector<DataType> dtypes = get_datatypes_list(device);

		try
		{
			std::cout << "Starting checkActivationFuncion(" << device.toString() << ")" << std::endl;
			Context context(device);
			std::cout << "Created context" << std::endl;

			constexpr int batch_size = 123;

			const Shape shape_3( { batch_size, 3 });
			const Shape shape_N( { batch_size, 234 });
			std::cout << "Created shapes " << shape_3.toString() << ", " << shape_N.toString() << std::endl;
			for (auto dt = dtypes.begin(); dt < dtypes.end(); dt++)
			{
				std::cout << "Checking activation functions with : data type=" << *dt << std::endl;

				const Tensor input_3 = create_test_tensor(shape_3, device, *dt);
				std::cout << "  Allocated input tensor=" << input_3.info() << std::endl;
				Tensor output_3 = create_test_tensor(shape_3, device, *dt);
				std::cout << "  Allocated output tensor=" << output_3.info() << std::endl;

				std::cout << "  Running softmax with 3 channels" << std::endl;
				activationForward(context, output_3, input_3, ActivationType::SOFTMAX);
				std::cout << "    WORKS" << std::endl;

				const Tensor input_N = create_test_tensor(shape_N, device, *dt);
				std::cout << "  Allocated input tensor=" << input_N.info() << std::endl;
				Tensor output_N = create_test_tensor(shape_N, device, *dt);
				std::cout << "  Allocated output tensor=" << output_N.info() << std::endl;

				std::cout << "  Running softmax with multiple channels" << std::endl;
				activationForward(context, output_N, input_N, ActivationType::SOFTMAX);
				std::cout << "    WORKS" << std::endl;
			}

		} catch (std::exception &e)
		{
			std::cout << "Caught exception '" << e.what() << "'" << std::endl;
			return 1;
		}
		return 0;
	}

} /* namespace ml */
