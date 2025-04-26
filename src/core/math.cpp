/*
 * math.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/layers/Conv2D.hpp>
#include <minml/utils/time_util.hpp>

#include <minml/backend/cpu_backend.h>
#include <minml/backend/cuda_backend.h>
#include <minml/backend/opencl_backend.h>

namespace
{
	using namespace ml;
//#define USE_TIMING

#ifdef USE_TIMING
#define SYNC() context.synchronize()

	struct Timer
	{
			std::string m_name;
			double m_start = 0.0;
			double m_total_time = 0.0;
			int m_count = 0;
			bool m_init = false;

			Timer(const std::string &name) :
					m_name(name)
			{
			}
			~Timer()
			{
				if (m_count > 0)
				{
					double time = m_total_time / m_count;
					char unit = ' ';
					if (time < 1.0e-3)
					{
						time *= 1.0e6;
						unit = 'u';
					}
					else
					{
						if (time < 1.0)
						{
							time *= 1.0e3;
							unit = 'm';
						}
					}
					std::cout << m_name << " : " << m_total_time << "s : " << time << " " << unit << "s (" << m_count << ")\n";
				}
			}
			void start() noexcept
			{
				m_start = getTime();
			}
			void stop() noexcept
			{
				if (m_init)
				{
					m_total_time += getTime() - m_start;
					m_count++;
				}
				else
					m_init = true;
			}
	};
#else
#define SYNC()
	struct Timer
	{
			Timer(const std::string &name)
			{
			}
			void start() noexcept
			{
			}
			void stop() noexcept
			{
			}
	};
#endif

	struct TimerGuard
	{
			Timer &t;
			TimerGuard(Timer &timer) :
					t(timer)
			{
				t.start();
			}
			~TimerGuard()
			{
				t.stop();
			}
	};

	int integer_sqrt(int i) noexcept
	{
		int result = 1;
		while (result * result != i)
			result++;
		return result;
	}
	int get_winograd_tile(const Shape &weight_shape, const Shape &matrices_shape) noexcept
	{
		assert(weight_shape[1] == weight_shape[2]); // only square filters
		return integer_sqrt(matrices_shape.firstDim()) - (weight_shape[1] - 1);
	}

	mlDataType_t get(DataType dtype) noexcept
	{
		return static_cast<mlDataType_t>(dtype);
	}
	mlActivationType_t get(ActivationType act) noexcept
	{
		return static_cast<mlActivationType_t>(act);
	}
	mlContext_t get(const Context &context) noexcept
	{
		return context.backend();
	}
	mlShape_t get(const Shape &shape) noexcept
	{
		mlShape_t result;
		result.rank = shape.rank();
		for (int i = 0; i < shape.rank(); i++)
			result.dim[i] = shape[i];
		for (int i = shape.rank(); i < Shape::max_dimension; i++)
			result.dim[i] = 0;
		return result;
	}
	mlShape_t get_shape(const Tensor &tensor) noexcept
	{
		return get(tensor.shape());
	}
	mlQuantizationData_t get(const AffineTransform &transform) noexcept
	{
		mlQuantizationData_t result;
		result.scale = transform.scale();
		result.shift = transform.shift();
		return result;
	}
	mlTensor_t get(const Tensor &tensor) noexcept
	{
		mlTensor_t result;
		result.data = const_cast<void*>(tensor.data());
		result.dtype = get(tensor.dtype());
		result.rank = tensor.rank();
		for (int i = 0; i < tensor.rank(); i++)
			result.dim[i] = tensor.dim(i);
		return result;
	}
	mlTensor_t get(Tensor &tensor) noexcept
	{
		mlTensor_t result;
		result.data = tensor.data();
		result.dtype = get(tensor.dtype());
		result.rank = tensor.rank();
		for (int i = 0; i < tensor.rank(); i++)
			result.dim[i] = tensor.dim(i);
		for (int i = tensor.rank(); i < 6; i++)
			result.dim[i] = 0;
		return result;
	}

	std::vector<mlTensor_t> get(const std::vector<Tensor> &list)
	{
		std::vector<mlTensor_t> result(list.size());
		for (size_t i = 0; i < list.size(); i++)
			result[i] = get(list[i]);
		return result;
	}
}

namespace ml
{
	void unpackInput(const Context &context, Tensor &dst, const Tensor &src)
	{
		static Timer timer("unpackInput");
		TimerGuard tg(timer);

		assert(src.dtype() == DataType::INT32);
		assert(dst.dim(0) == src.dim(0));
		assert(dst.dim(1) == src.dim(1));
		assert(dst.dim(2) == src.dim(2));

		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_unpack_input(get(context), get_shape(dst), get(dst.dtype()), dst.data(), src.data());
				break;
			case DeviceType::CUDA:
				cuda_unpack_input(get(context), get_shape(dst), get(dst.dtype()), dst.data(), src.data());
				break;
			case DeviceType::OPENCL:
				opencl_unpack_input(get(context), get_shape(dst), get(dst.dtype()), dst.data(), src.data());
				break;
		}SYNC();
	}
	void convertTensor(const Context &context, Tensor &dst, const Tensor &src)
	{
		assert(dst.volume() == src.volume());
		convertType(context, dst.data(), dst.dtype(), src.data(), src.dtype(), src.volume());
	}
	void convertType(const Context &context, void *dst, DataType dst_dtype, const void *src, DataType src_dtype, int elements)
	{
		static Timer timer("convertType");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_convert_type(get(context), dst, get(dst_dtype), src, get(src_dtype), elements);
				break;
			case DeviceType::CUDA:
				cuda_convert_type(get(context), dst, get(dst_dtype), src, get(src_dtype), elements);
				break;
			case DeviceType::OPENCL:
				opencl_convert_type(get(context), dst, get(dst_dtype), src, get(src_dtype), elements);
				break;
		}SYNC();
	}
	void transpose_021(const Context &context, const Tensor &input, Tensor &output)
	{
		assert(input.rank() == 3 && output.rank() == 3);
		assert(input.dtype() == output.dtype());
		assert(input.dim(0) == output.dim(0));
		assert(input.dim(1) == output.dim(2));
		assert(input.dim(2) == output.dim(1));

		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_transpose_021(get(context), get(input.dtype()), get_shape(input), input.data(), output.data());
				break;
			case DeviceType::CUDA:
				cuda_transpose_021(get(context), get(input.dtype()), get_shape(input), input.data(), output.data());
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}
	}

	void winogradWeightTransform(const Context &context, const Tensor &weights, Tensor &matrices, bool invert)
	{
		static Timer timer("winogradWeightTransform");
		TimerGuard tg(timer);

		const int tile_size = get_winograd_tile(weights.shape(), matrices.shape());
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_weight_transform(get(context), tile_size, get(weights.dtype()), get_shape(weights), weights.data(), matrices.data(),
						invert);
				break;
			case DeviceType::CUDA:
				cuda_winograd_weight_transform(get(context), tile_size, get(weights), get(matrices), invert);
				break;
			case DeviceType::OPENCL:
				opencl_winograd_weight_transform(get(context), tile_size, get(weights.dtype()), get_shape(weights), weights.data(), matrices.data(),
						invert);
				break;
		}SYNC();
	}
	void winogradInputTransform(const Context &context, const Shape &weight_shape, const Tensor &input, Tensor &matrices)
	{
		static Timer timer("winogradInputTransform");
		TimerGuard tg(timer);

		const int tile_size = get_winograd_tile(weight_shape, matrices.shape());
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_input_transform(get(context), tile_size, get(input.dtype()), get(weight_shape), get_shape(input), input.data(),
						matrices.data());
				break;
			case DeviceType::CUDA:
				cuda_winograd_input_transform(get(context), tile_size, get(input), get(matrices));
				break;
			case DeviceType::OPENCL:
				opencl_winograd_input_transform(get(context), tile_size, get(input.dtype()), get(weight_shape), get_shape(input), input.data(),
						matrices.data());
				break;
		}SYNC();
	}
	void winogradOutputTransform(const Context &context, const Shape &weight_shape, const Tensor &matrices, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act, float beta)
	{
		static Timer timer("winogradOutputTransform");
		TimerGuard tg(timer);

		const int tile_size = get_winograd_tile(weight_shape, matrices.shape());
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_output_transform(get(context), tile_size, get(output.dtype()), get(weight_shape), get_shape(output), matrices.data(),
						output.data(), bias.data(), add.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_winograd_output_transform(get(context), tile_size, get(matrices), get(bias), get(add), get(output), get(act));
				break;
			case DeviceType::OPENCL:
				opencl_winograd_output_transform(get(context), tile_size, get(output.dtype()), get(weight_shape), get_shape(output), matrices.data(),
						output.data(), bias.data(), add.data(), get(act));
				break;
		}SYNC();
	}
	void winogradGradientTransform(const Context &context, const Shape &weight_shape, const Tensor &gradient, Tensor &matrices)
	{
		const int tile_size = get_winograd_tile(weight_shape, matrices.shape());
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_gradient_transform(get(context), tile_size, get(gradient.dtype()), get(weight_shape), get_shape(gradient),
						gradient.data(), matrices.data());
				break;
			case DeviceType::CUDA:
				cuda_winograd_gradient_transform(get(context), tile_size, get(gradient), get(matrices));
				break;
			case DeviceType::OPENCL:
				opencl_winograd_gradient_transform(get(context), tile_size, get(gradient.dtype()), get(weight_shape), get_shape(gradient),
						gradient.data(), matrices.data());
				break;
		}SYNC();
	}
	void winogradUpdateTransform(const Context &context, const Tensor &matrices, Tensor &update)
	{
		const int tile_size = get_winograd_tile(update.shape(), matrices.shape());
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_winograd_update_transform(get(context), tile_size, get(matrices.dtype()), get_shape(update), matrices.data(), update.data());
				break;
			case DeviceType::CUDA:
				cuda_winograd_update_transform(get(context), tile_size, get(matrices), get(update));
				break;
			case DeviceType::OPENCL:
				opencl_winograd_update_transform(get(context), tile_size, get(matrices.dtype()), get_shape(update), matrices.data(), update.data());
				break;
		}SYNC();
	}

	void im2row(const Context &context, Tensor &output, const Tensor &input, int kernel_size, bool invert, const void *padding)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_im2row(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), kernel_size, invert, padding);
				break;
			case DeviceType::CUDA:
				cuda_im2row(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), kernel_size, invert, padding);
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}SYNC();
	}
	void depthToSpace(const Context &context, const Tensor &input, Tensor &output, float beta)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_depth_to_space(get(context), get(input.dtype()), get_shape(input), input.data(), get_shape(output), output.data());
				break;
			case DeviceType::CUDA:
				cuda_depth_to_space(get(context), get(input.dtype()), get_shape(input), input.data(), get_shape(output), output.data());
				break;
			case DeviceType::OPENCL:
//				opencl_depth_to_space(get(context), get(input.dtype()), get_shape(input), input.data(), get_shape(output), output.data());
				break;
		}
	}
	void spaceToDepth(const Context &context, const Tensor &input, Tensor &output, float beta)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_space_to_depth(get(context), get(input.dtype()), get_shape(input), input.data(), get_shape(output), output.data());
				break;
			case DeviceType::CUDA:
				cuda_space_to_depth(get(context), get(input.dtype()), get_shape(input), input.data(), get_shape(output), output.data());
				break;
			case DeviceType::OPENCL:
//				opencl_space_to_depth(get(context), get(input.dtype()), get_shape(input), input.data(), get_shape(output), output.data());
				break;
		}
	}

	void convolutionImplicitGemmForward(const Context &context, const Tensor &input, const Tensor &weights, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_convolution_implicit_gemm_forward(get(context), get(weights.dtype()), get_shape(input), get_shape(weights), input.data(),
						weights.data(), output.data(), bias.data(), add.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_convolution_implicit_gemm_forward(get(context), get(weights.dtype()), get_shape(input), get_shape(weights), input.data(),
						weights.data(), output.data(), bias.data(), add.data(), get(act));
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}
	}
	void fusedConvBlockForward(const Context &context, const Tensor &input, const Tensor &dwconv_weights, const Tensor &dwconv_bias,
			const Tensor &first_conv_weights, const Tensor &first_conv_bias, const Tensor &second_conv_weights, const Tensor &second_conv_bias,
			Tensor &output)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_fused_conv_block_forward(get(context), get(input), get(dwconv_weights), get(dwconv_bias), get(first_conv_weights),
						get(first_conv_bias), get(second_conv_weights), get(second_conv_bias), get(output));
				break;
			case DeviceType::CUDA:
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}
	}

	void depthwiseConvForward(const Context &context, float alpha, const Tensor &input, const Tensor &weights, float beta, Tensor &output,
			const Tensor &bias)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
			{
//				const int batch = input.dim(0);
//				const int height = input.dim(1);
//				const int width = input.dim(2);
//				const int filters = input.dim(3);
//
//				const int kernel_height = weights.dim(0);
//				const int kernel_width = weights.dim(1);
//
//				const int pad_h = -(kernel_height - 1) / 2;
//				const int pad_w = -(kernel_width - 1) / 2;
//
//				output.zeroall();
//				for (int b = 0; b < batch; b++)
//					for (int f = 0; f < filters; f++)
//						for (int h = 0; h < height; h++)
//							for (int w = 0; w < width; w++)
//							{
//								float tmp = 0.0f;
//								for (int i = 0; i < kernel_height; i++)
//									for (int j = 0; j < kernel_width; j++)
//										if ((pad_h + h + i) >= 0 and (pad_h + h + i) < height and (pad_w + w + j) >= 0 and (pad_w + w + j) < width)
//											tmp += weights.get( { i, j, f }) * input.get( { b, pad_h + h + i, pad_w + w + j, f });
//								if (not bias.isEmpty())
//									tmp += bias.get( { f });
//								tmp *= alpha;
//								if (beta != 0.0f)
//									tmp += beta * output.get( { b, h, w, f });
//								output.at( { b, h, w, f }) = tmp;
//							}
				break;
			}
			case DeviceType::CUDA:
				cuda_depthwise_conv_forward(get(context), alpha, get(input), get(weights), get(bias), beta, get(output));
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}SYNC();
	}
	void depthwiseConvBackward(const Context &context, float alpha, const Tensor &gradient_next, const Tensor &weights, float beta,
			Tensor &gradient_prev)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_depthwise_conv_backward(get(context), get_shape(gradient_prev), get_shape(weights), gradient_next.data(), weights.data(),
						gradient_prev.data());
				break;
			case DeviceType::CUDA:
				cuda_depthwise_conv_backward(get(context), alpha, get(gradient_next), get(weights), beta, get(gradient_prev));
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}SYNC();
	}
	void depthwiseConvUpdate(const Context &context, float alpha, const Tensor &input, const Tensor &gradient_next, float beta,
			Tensor &weights_update)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_depthwise_conv_update(get(context), get_shape(input), get_shape(weights_update), input.data(), gradient_next.data(),
						weights_update.data());
				break;
			case DeviceType::CUDA:
				cuda_depthwise_conv_update(get(context), alpha, get(input), get(gradient_next), beta, get(weights_update));
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}SYNC();
	}

	void averagePoolingForward(const Context &context, float alpha, const Tensor &input, float beta, Tensor &output, int size)
	{
		static Timer timer("global_average_pooling");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_average_pooling_forward(get(context), alpha, get(input), beta, get(output), size);
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void averagePoolingBackward(const Context &context, float alpha, const Tensor &gradient_next, float beta, Tensor &gradient_prev, int size)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_average_pooling_backward(get(context), alpha, get(gradient_next), beta, get(gradient_prev), size);
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}

	void globalAveragePoolingForward(const Context &context, float alpha, const Tensor &input, float beta, Tensor &output)
	{
		static Timer timer("global_average_pooling");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_global_average_pooling_forward(get(context), alpha, get(input), beta, get(output));
				break;
			case DeviceType::CUDA:
				cuda_global_average_pooling_forward(get(context), alpha, get(input), beta, get(output));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void globalAveragePoolingBackward(const Context &context, float alpha, const Tensor &gradient_next, float beta, Tensor &gradient_prev)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_global_average_pooling_backward(get(context), alpha, get(gradient_next), beta, get(gradient_prev));
				break;
			case DeviceType::CUDA:
				cuda_global_average_pooling_backward(get(context), alpha, get(gradient_next), beta, get(gradient_prev));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void channelScalingForward(const Context &context, float alpha, const Tensor &input, const Tensor &scales, float beta, Tensor &output)
	{
		static Timer timer("channel_scaling");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_channel_scaling_forward(get(context), alpha, get(input), get(scales), beta, get(output));
				break;
			case DeviceType::CUDA:
				cuda_channel_scaling_forward(get(context), alpha, get(input), get(scales), beta, get(output));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void channelScalingBackward(const Context &context, float alpha, const Tensor &gradient_next, const Tensor &input, const Tensor &scales,
			float beta_input, Tensor &gradient_prev, float beta_scales, Tensor &gradient_scales)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_channel_scaling_backward(get(context), alpha, get(gradient_next), get(input), get(scales), beta_input, get(gradient_prev),
						beta_scales, get(gradient_scales));
				break;
			case DeviceType::CUDA:
				cuda_channel_scaling_backward(get(context), alpha, get(gradient_next), get(input), get(scales), beta_input, get(gradient_prev),
						beta_scales, get(gradient_scales));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void channelAveragePoolingForward(const Context &context, float alpha, const Tensor &input, float beta, Tensor &output)
	{
		static Timer timer("global_average_pooling");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
//					cpu_global_average_pooling_forward(get(context), alpha, get(input), beta, get(output));
				break;
			case DeviceType::CUDA:
				cuda_channel_average_pooling_forward(get(context), alpha, get(input), beta, get(output));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void channelAveragePoolingBackward(const Context &context, float alpha, const Tensor &gradient_next, float beta, Tensor &gradient_prev)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
//					cpu_global_average_pooling_backward(get(context), alpha, get(gradient_next), beta, get(gradient_prev));
				break;
			case DeviceType::CUDA:
				cuda_channel_average_pooling_backward(get(context), alpha, get(gradient_next), beta, get(gradient_prev));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void spatialScalingForward(const Context &context, float alpha, const Tensor &input, const Tensor &scales, float beta, Tensor &output)
	{
		static Timer timer("channel_scaling");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
//					cpu_channel_scaling_forward(get(context), alpha, get(input), get(scales), beta, get(output));
				break;
			case DeviceType::CUDA:
				cuda_spatial_scaling_forward(get(context), alpha, get(input), get(scales), beta, get(output));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void spatialScalingBackward(const Context &context, float alpha, const Tensor &gradient_next, const Tensor &input, const Tensor &scales,
			float beta_input, Tensor &gradient_prev, float beta_scales, Tensor &gradient_scales)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
//					cpu_channel_scaling_backward(get(context), alpha, get(gradient_next), get(input), get(scales), beta_input, get(gradient_prev),
//							beta_scales, get(gradient_scales));
				break;
			case DeviceType::CUDA:
				cuda_spatial_scaling_backward(get(context), alpha, get(gradient_next), get(input), get(scales), beta_input, get(gradient_prev),
						beta_scales, get(gradient_scales));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}

	void gemm(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
	{
		static Timer timer("gemm");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_gemm(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB, alpha, beta);
				break;
			case DeviceType::CUDA:
				cuda_gemm_v2(get(context), opA, opB, alpha, get(A), get(B), beta, get(C));
				break;
			case DeviceType::OPENCL:
				opencl_gemm(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB, alpha,
						beta);
				break;
		}SYNC();
	}
	void gemmBatched(const Context &context, char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, float alpha, float beta)
	{
		static Timer timer("gemmBatched");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_gemm_batched(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB,
						alpha, beta);
				break;
			case DeviceType::CUDA:
				cuda_gemm_batched_v2(get(context), opA, opB, alpha, get(A), get(B), beta, get(C));
				break;
			case DeviceType::OPENCL:
				opencl_gemm_batched(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB,
						alpha, beta);
				break;
		}SYNC();
	}

	void gemm_ex(const Context &context, Tensor &D, float alpha, char opA, const Tensor &A, char opB, const Tensor &B, float beta, const Tensor &C,
			const Tensor &bias, ActivationType act)
	{
		static Timer timer("gemm_ex");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_gemm_ex(get(context), get(C.dtype()), get_shape(D), D.data(), alpha, opA, get_shape(A), A.data(), opB, get_shape(B), B.data(),
						beta, get_shape(C), C.data(), bias.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_gemm_ex(get(context), get(C.dtype()), get_shape(D), D.data(), alpha, opA, get_shape(A), A.data(), opB, get_shape(B), B.data(),
						beta, get_shape(C), C.data(), bias.data(), get(act));
				break;
			case DeviceType::OPENCL:
				opencl_gemm_ex(get(context), get(C.dtype()), get_shape(D), D.data(), alpha, opA, get_shape(A), A.data(), opB, get_shape(B), B.data(),
						beta, get_shape(C), C.data(), bias.data(), get(act));
				break;
		}SYNC();
	}

	void addBiasAct(const Context &context, float alpha, const Tensor &input, const Tensor &bias, float beta, Tensor &output, ActivationType act)
	{
		static Timer timer("addBiasAct");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_add_bias_act(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), bias.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_add_bias_act(get(context), alpha, get(input), get(bias), beta, get(output), get(act));
				break;
			case DeviceType::OPENCL:
				opencl_add_bias_act(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), bias.data(), get(act));
				break;
		}SYNC();
	}

	void batchnormInference(const Context &context, float alpha, const Tensor &input, const Tensor &weights, const Tensor &bias,
			const Tensor &avg_var, float beta, Tensor &output, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
//				cpu_batchnorm_inference(get(context), get(input.dtype()), get_shape(input), input.data(), output.data(), weights.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_inference(get(context), alpha, get(input), get(weights), get(bias), get(avg_var), beta, get(output), get(act));
				break;
			case DeviceType::OPENCL:
//				opencl_batchnorm_inference(get(context), get(input.dtype()), get_shape(input), input.data(), output.data(), weights.data(), get(act));
				break;
		}SYNC();
	}
	void batchnormForward(const Context &context, float alpha, const Tensor &input, const Tensor &weights, const Tensor &bias, float beta,
			Tensor &output, Tensor &running_stats, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
//				cpu_batchnorm_forward(get(context), get_shape(input), input.data(), output.data(), weights.data(), running_stats.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_forward(get(context), alpha, get(input), get(weights), get(bias), beta, get(output), get(running_stats), get(act));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void batchnormBackward(const Context &context, float alpha, const Tensor &input, const Tensor &output, Tensor &gradient_next,
			const Tensor &weights, const Tensor &bias, float beta_prev, Tensor &gradient_prev, float beta_update, Tensor &weights_update,
			Tensor &bias_update, const Tensor &running_stats, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
//				cpu_batchnorm_backward(get(context), get_shape(input), input.data(), output.data(), gradient_prev.data(), gradient_next.data(),
//						weights.data(), weights_update.data(), running_stats.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_backward(get(context), alpha, get(input), get(gradient_next), get(weights), get(bias), get(running_stats), beta_prev,
						get(gradient_prev), beta_update, get(weights_update), get(bias_update), get(act));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void batchnormUpdate(const Context &context, const Tensor &running_stats, Tensor &weights)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
//				cpu_batchnorm_update(get(context), shape, running_stat.data(), weights.data(), use_gamma, use_beta);
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_update(get(context), get(running_stats), get(weights));
				break;
			case DeviceType::OPENCL:
//				opencl_batchnorm_update(get(context), shape, running_stat.data(), weights.data(), use_gamma, use_beta);
				break;
		}SYNC();
	}
	void foldBatchnorm(const Context &context, Tensor &layer_weights, Tensor &layer_bias, const Tensor &bn_weights, const Tensor &bn_bias,
			const Tensor &bn_avg_var)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_fold_batchnorm(get(context), get(layer_weights), get(layer_bias), get(bn_weights), get(bn_bias), get(bn_avg_var));
				break;
			case DeviceType::CUDA:
				cuda_fold_batchnorm(get(context), get(layer_weights), get(layer_bias), get(bn_weights), get(bn_bias), get(bn_avg_var));
				break;
			case DeviceType::OPENCL:
//				opencl_fold_batchnorm(get(context), get_shape(layer_weights), layer_weights.data(), layer_bias.data(), batchnorm_weights.data());
				break;
		}
		context.synchronize();
	}

	void layernormForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias, const Tensor &ext)
	{
		static Timer timer("layernormForward");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_layernorm_forward(get(context), get_shape(input), get(input.dtype()), input.data(), output.data(), weights.data(), bias.data(),
						ext.data());
				break;
			case DeviceType::CUDA:
				cuda_layernorm_forward(get(context), get_shape(input), get(input.dtype()), input.data(), output.data(), weights.data(), bias.data(),
						ext.data());
				break;
			case DeviceType::OPENCL:
				opencl_layernorm_forward(get(context), get_shape(input), get(input.dtype()), input.data(), output.data(), weights.data(), bias.data(),
						ext.data());
				break;
		}SYNC();
	}
	void layernormBackward(const Context &context, const Tensor &input, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weights,
			Tensor &weights_update, Tensor &bias_update, float beta)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_layernorm_backward(get(context), get_shape(input), input.data(), gradient_prev.data(), gradient_next.data(), weights.data(),
						weights_update.data(), bias_update.data());
				break;
			case DeviceType::CUDA:
				cuda_layernorm_backward(get(context), get_shape(input), input.data(), gradient_prev.data(), gradient_next.data(), weights.data(),
						weights_update.data(), bias_update.data());
				break;
			case DeviceType::OPENCL:
				opencl_layernorm_backward(get(context), get_shape(input), input.data(), gradient_prev.data(), gradient_next.data(), weights.data(),
						weights_update.data(), bias_update.data());
				break;
		}
	}

	void rmsnormForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights)
	{
		static Timer timer("rmsnormForward");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_rmsnorm_forward(get(context), get_shape(input), get(input.dtype()), input.data(), output.data(), weights.data());
				break;
			case DeviceType::CUDA:
				cuda_rmsnorm_forward(get(context), get_shape(input), get(input.dtype()), input.data(), output.data(), weights.data());
				break;
			case DeviceType::OPENCL:
				opencl_rmsnorm_forward(get(context), get_shape(input), get(input.dtype()), input.data(), output.data(), weights.data());
				break;
		}SYNC();
	}
	void rmsnormBackward(const Context &context, const Tensor &input, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &weights,
			Tensor &weights_update, float beta)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_rmsnorm_backward(get(context), get_shape(input), input.data(), gradient_prev.data(), gradient_next.data(), weights.data(),
						weights_update.data());
				break;
			case DeviceType::CUDA:
				cuda_rmsnorm_backward(get(context), get_shape(input), input.data(), gradient_prev.data(), gradient_next.data(), weights.data(),
						weights_update.data());
				break;
			case DeviceType::OPENCL:
				opencl_rmsnorm_backward(get(context), get_shape(input), input.data(), gradient_prev.data(), gradient_next.data(), weights.data(),
						weights_update.data());
				break;
		}
	}

	int multiHeadAttentionGetWorkspaceSize(const Context &context, const Shape &inputShape, const Shape &weightsShape, int num_heads, bool training)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				return cpu_multi_head_attention_get_workspace_size(get(inputShape), get(weightsShape), num_heads, training);
			case DeviceType::CUDA:
				return cuda_multi_head_attention_get_workspace_size(get(inputShape), get(weightsShape), num_heads, training);
			case DeviceType::OPENCL:
				return opencl_multi_head_attention_get_workspace_size(get(inputShape), get(weightsShape), num_heads, training);
			default:
				return 0;
		}
	}
	void multiHeadAttentionForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias,
			const Tensor &mask, Tensor &workspace, Tensor &backwardData, int num_heads, bool symmetric)
	{
		static Timer timer("multiHeadAttentionForward");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_multi_head_attention_forward(get(context), get_shape(input), get_shape(weights), get_shape(bias), get(input.dtype()),
						input.data(), output.data(), weights.data(), bias.data(), mask.data(), workspace.data(), backwardData.data(), num_heads,
						symmetric);
				break;
			case DeviceType::CUDA:
				cuda_multi_head_attention_forward(get(context), get_shape(input), get_shape(weights), get_shape(bias), get(input.dtype()),
						input.data(), output.data(), weights.data(), bias.data(), mask.data(), workspace.data(), backwardData.data(), num_heads,
						symmetric);
				break;
			case DeviceType::OPENCL:
				opencl_multi_head_attention_forward(get(context), get_shape(input), get_shape(weights), get_shape(bias), get(input.dtype()),
						input.data(), output.data(), weights.data(), bias.data(), mask.data(), workspace.data(), backwardData.data(), num_heads,
						symmetric);
				break;
		}SYNC();
	}
	void multiHeadAttentionBackward(const Context &context, const Tensor &input, const Tensor &weights, const Tensor &bias, const Tensor &mask,
			Tensor &gradient_prev, Tensor &gradient_next, Tensor &weights_update, Tensor &bias_update, Tensor &mask_update, Tensor &workspace,
			Tensor &backwardData, int num_heads, bool symmetric, float beta)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_multi_head_attention_backward(get(context), get_shape(input), get_shape(weights), get_shape(bias), input.data(), weights.data(),
						bias.data(), mask.data(), gradient_prev.data(), gradient_next.data(), weights_update.data(), bias_update.data(),
						workspace.data(), backwardData.data(), num_heads, symmetric);
				break;
			case DeviceType::CUDA:
				cuda_multi_head_attention_backward(get(context), get_shape(input), get_shape(weights), get_shape(bias), input.data(), weights.data(),
						bias.data(), mask.data(), gradient_prev.data(), gradient_next.data(), weights_update.data(), bias_update.data(),
						mask_update.data(), workspace.data(), backwardData.data(), num_heads, symmetric);
				break;
			case DeviceType::OPENCL:
				opencl_multi_head_attention_backward(get(context), get_shape(input), get_shape(weights), get_shape(bias), input.data(),
						weights.data(), bias.data(), mask.data(), gradient_prev.data(), gradient_next.data(), weights_update.data(),
						bias_update.data(), workspace.data(), backwardData.data(), num_heads, symmetric);
				break;
		}
	}

	void windowPartitioning(const Context &context, const Tensor &input, Tensor &output, const Shape &offset)
	{
		static Timer timer("windowPartitioning");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
//				cpu_window_partitioning(get(context), get(input.dtype()), get_shape(input), get_shape(output), input.data(), output.data(),
//						get(offset));
				break;
			case DeviceType::CUDA:
				cuda_window_partitioning(get(context), get(input.dtype()), get_shape(input), get_shape(output), input.data(), output.data(),
						get(offset));
				break;
			case DeviceType::OPENCL:
//				opencl_window_partitioning(get(context), get(input.dtype()), get_shape(input), get_shape(output), input.data(), output.data(),
//										get(offset));
				break;
		}SYNC();
	}
	void windowMerging(const Context &context, const Tensor &input, Tensor &output, const Shape &offset)
	{
		static Timer timer("windowMerging");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
//				cpu_window_merging(get(context), get(input.dtype()), get_shape(input), get_shape(output), input.data(), output.data(),
//						get(offset));
				break;
			case DeviceType::CUDA:
				cuda_window_merging(get(context), get(input.dtype()), get_shape(input), get_shape(output), input.data(), output.data(), get(offset));
				break;
			case DeviceType::OPENCL:
//				opencl_window_merging(get(context), get(input.dtype()), get_shape(input), get_shape(output), input.data(), output.data(),
//										get(offset));
				break;
		}SYNC();
	}

	void activationForward(const Context &context, float alpha, const Tensor &input, float beta, Tensor &output, ActivationType act)
	{
		static Timer timer("activationForward");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_activation_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_activation_forward(get(context), alpha, get(input), beta, get(output), get(act));
				break;
			case DeviceType::OPENCL:
				opencl_activation_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), get(act));
				break;
		}SYNC();
	}
	void activationBackward(const Context &context, float alpha, const Tensor &gradient_next, const Tensor &output, float beta, Tensor &gradient_prev,
			ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_activation_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), output.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_activation_backward(get(context), alpha, get(gradient_next), get(output), beta, get(gradient_prev), get(act));
				break;
			case DeviceType::OPENCL:
				opencl_activation_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), output.data(),
						get(act));
				break;
		}SYNC();
	}
	void softmaxForward(const Context &context, Tensor &output, const Tensor &input)
	{
		static Timer timer("softmaxForward");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_softmax_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data());
				break;
			case DeviceType::CUDA:
				cuda_softmax_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data());
				break;
			case DeviceType::OPENCL:
				opencl_softmax_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data());
				break;
		}SYNC();
	}
	void fusedBiasActCopyBackward(const Context &context, Tensor &gradient_next, const Tensor &output, float beta_prev, Tensor &gradient_prev,
			float beta_bias_update, Tensor &bias_update, ActivationType act)
	{
		static Timer timer("fusedBiasActCopyBackward");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_fused_act_bias_copy_backward(get(context), get(gradient_next), get(output), beta_prev, get(gradient_prev), beta_bias_update,
						get(bias_update), get(act));
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}

	void emulateLowPrecision(const Context &context, Tensor &dst, const Tensor &src, DataType dtype, AffineTransform transform)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_emulate_low_precision(get(context), get_shape(dst), get(dtype), dst.data(), src.data(), get(transform));
				break;
			case DeviceType::CUDA:
				cuda_emulate_low_precision(get(context), get_shape(dst), get(dtype), dst.data(), src.data(), get(transform));
				break;
			case DeviceType::OPENCL:
				opencl_emulate_low_precision(get(context), get_shape(dst), get(dtype), dst.data(), src.data(), get(transform));
				break;
		}
	}
	void sumOverFirstDim(const Context &context, float alpha, const Tensor &src, float beta, Tensor &dst)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
//				cpu_sum_over_first_dim(get(context), get_shape(src), dst.data(), src.data(), beta);
				break;
			case DeviceType::CUDA:
				cuda_sum_over_first_dim(get(context), alpha, get(src), beta, get(dst));
				break;
			case DeviceType::OPENCL:
//				opencl_sum_over_first_dim(get(context), get_shape(src), dst.data(), src.data(), beta);
				break;
		}SYNC();
	}
	void multiplyTensors(const Context &context, Tensor &dst, const Tensor &lhs, const Tensor &rhs)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_multiply_tensors(get(context), get(dst.dtype()), get_shape(dst), dst.data(), lhs.data(), rhs.data());
				break;
			case DeviceType::CUDA:
				cuda_multiply_tensors(get(context), get(dst.dtype()), get_shape(dst), dst.data(), lhs.data(), rhs.data());
				break;
			case DeviceType::OPENCL:
				opencl_multiply_tensors(get(context), get(dst.dtype()), get_shape(dst), dst.data(), lhs.data(), rhs.data());
				break;
		}
	}
	void addTensors(const Context &context, Tensor &dst, const Tensor &src1, const Tensor &src2)
	{
		addTensors(context, 0.0f, dst, 1.0f, src1, 1.0f, src2);
	}
	void addTensors(const Context &context, float beta, Tensor &dst, float alpha1, const Tensor &src1, float alpha2, const Tensor &src2)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_add_tensors(get(context), get(dst.dtype()), get_shape(dst), beta, dst.data(), alpha1, src1.data(), alpha2, src2.data());
				break;
			case DeviceType::CUDA:
				cuda_add_tensors(get(context), get(dst.dtype()), get_shape(dst), beta, dst.data(), alpha1, src1.data(), alpha2, src2.data());
				break;
			case DeviceType::OPENCL:
				opencl_add_tensors(get(context), get(dst.dtype()), get_shape(dst), beta, dst.data(), alpha1, src1.data(), alpha2, src2.data());
				break;
		}SYNC();
	}

	float meanSquaredLoss(const Context &context, const Tensor &output, const Tensor &target, const Tensor &mask)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				return cpu_mean_squared_loss(get(context), get_shape(output), output.data(), target.data(), mask.data());
			case DeviceType::CUDA:
				return cuda_mean_squared_loss(get(context), get(output), get(target), get(mask));
			case DeviceType::OPENCL:
				return opencl_mean_squared_loss(get(context), get_shape(output), output.data(), target.data(), mask.data());
		}
		return 0.0f;
	}
	float crossEntropyLoss(const Context &context, const Tensor &output, const Tensor &target, const Tensor &mask)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				return cpu_cross_entropy_loss(get(context), get_shape(output), output.data(), target.data(), mask.data());
			case DeviceType::CUDA:
				return cuda_cross_entropy_loss(get(context), get(output), get(target), get(mask));
			case DeviceType::OPENCL:
				return opencl_cross_entropy_loss(get(context), get_shape(output), output.data(), target.data(), mask.data());
		}
		return 0.0f;
	}
	void meanSquaredGradient(const Context &context, float alpha, const Tensor &output, const Tensor &target, const Tensor &mask, float beta,
			Tensor &gradient)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_mean_squared_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), mask.data(), alpha);
				break;
			case DeviceType::CUDA:
				cuda_mean_squared_gradient(get(context), alpha, get(output), get(target), get(mask), beta, get(gradient));
				break;
			case DeviceType::OPENCL:
//				opencl_mean_squared_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), mask.data(), weight);
				break;
		}SYNC();
	}
	void crossEntropyGradient(const Context &context, float alpha, const Tensor &output, const Tensor &target, const Tensor &mask, float beta,
			Tensor &gradient)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_cross_entropy_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), mask.data(), alpha);
				break;
			case DeviceType::CUDA:
				cuda_cross_entropy_gradient(get(context), alpha, get(output), get(target), get(mask), beta, get(gradient));
				break;
			case DeviceType::OPENCL:
//				opencl_cross_entropy_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), mask.data(), weight);
				break;
		}SYNC();
	}
	void radamOptimize(const Context &context, float scale, const std::vector<Tensor> &gradients, std::vector<Tensor> &weights,
			std::vector<Tensor> &momentums, std::vector<Tensor> &variances, std::vector<Tensor> &weights_copy, float learning_rate, float beta1,
			float beta2, int step, float weight_decay)
	{
		assert(gradients.size() == weights.size());
		assert(gradients.size() == momentums.size());
		assert(gradients.size() == variances.size());
		std::vector<mlTensor_t> _gradients = get(gradients);
		std::vector<mlTensor_t> _weights = get(weights);
		std::vector<mlTensor_t> _momentums = get(momentums);
		std::vector<mlTensor_t> _variances = get(variances);
		std::vector<mlTensor_t> _weights_copy = get(weights_copy);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_fused_radam_optimize(get(context), scale, _gradients.data(), _weights.data(), _momentums.data(), _variances.data(),
						weights_copy.empty() ? nullptr : _weights_copy.data(), learning_rate, beta1, beta2, step, _gradients.size(), weight_decay);
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void lionOptimize(const Context &context, float scale, const std::vector<Tensor> &gradients, std::vector<Tensor> &weights,
			std::vector<Tensor> &momentums, std::vector<Tensor> &weights_copy, float learning_rate, float beta1, float beta2, int step,
			float weight_decay)
	{
		assert(gradients.size() == weights.size());
		assert(gradients.size() == momentums.size());
		std::vector<mlTensor_t> _gradients = get(gradients);
		std::vector<mlTensor_t> _weights = get(weights);
		std::vector<mlTensor_t> _momentums = get(momentums);
		std::vector<mlTensor_t> _weights_copy = get(weights_copy);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_fused_lion_optimize(get(context), scale, _gradients.data(), _weights.data(), _momentums.data(),
						weights_copy.empty() ? nullptr : _weights_copy.data(), learning_rate, beta1, beta2, step, _gradients.size(), weight_decay);
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	std::vector<int> isNanOrInf(const Context &context, const std::vector<Tensor> &tensors)
	{
		std::vector<int> result(tensors.size(), 0);
		std::vector<mlTensor_t> _tensors = get(tensors);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_fused_is_nan_or_inf(get(context), _tensors.data(), result.data(), _tensors.size());
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
		return result;
	}
	void l2Regularization(const Context &context, std::vector<Tensor> &gradients, const std::vector<Tensor> &params, float scale)
	{
		assert(gradients.size() == params.size());
		std::vector<mlTensor_t> _gradients = get(gradients);
		std::vector<mlTensor_t> _params = get(params);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}

	/*
	 * quantization
	 */
	void dequantize(const Context &context, const Tensor &input, Tensor &output, AffineTransform transform)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				break;
			case DeviceType::OPENCL:
				break;
		}
	}
	void quantized_depthwise_conv_forward(const Context &context, const Tensor &input, const Tensor &weights, const Tensor &scales,
			const Tensor &bias, Tensor &output, AffineTransform output_transform, int padding_value)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_quantized_depthwise_conv_forward(get(context), get(output.dtype()), get_shape(input), get_shape(weights), input.data(),
						weights.data(), scales.data(), bias.data(), output.data(), get(output_transform), padding_value);
				break;
			case DeviceType::OPENCL:
				break;
		}
	}
	void quantized_scale_shift_act(const Context &context, Tensor &output, AffineTransform output_transform, const Tensor &input,
			const Tensor &scales, const Tensor &bias, ActivationType act, const Tensor &ext, AffineTransform ext_transform)
	{
		assert(output.dtype() == DataType::FLOAT32 || output.dtype() == DataType::INT8);
		assert(input.dtype() == DataType::INT32);
		assert(scales.dtype() == DataType::FLOAT32);
		assert(bias.dtype() == DataType::FLOAT32);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_quantized_scale_shift_act(get(context), get(output.dtype()), get_shape(output), output.data(), get(output_transform),
						input.data(), scales.data(), bias.data(), get(act), ext.data(), get(ext_transform));
				break;
			case DeviceType::OPENCL:
				break;
		}
	}
	void transpose(const Context &context, Tensor &output, const Tensor &input, std::initializer_list<int> ordering)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_transpose(get(context), get(input.dtype()), get_shape(output), get_shape(input), output.data(), input.data(), ordering.begin());
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	std::array<int, 3> explicit_gemm_workspace(const Shape &inputShape, const Shape &outputShape, const Shape &weightShape)
	{
		int forward_workspace = inputShape.volumeWithoutLastDim() * weightShape.volumeWithoutFirstDim();
		int backward_workspace = weightShape.volume() + outputShape.volumeWithoutLastDim() * weightShape.volumeWithoutLastDim();
		int update_workspace = inputShape.volumeWithoutLastDim() * weightShape.volumeWithoutFirstDim();
		return std::array<int, 3> { forward_workspace, backward_workspace, update_workspace };
	}
	void explicit_gemm_forward(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, const Tensor &bias,
			Tensor &workspace, ActivationType activation, const Tensor &add)
	{
		assert(same_device(context, input, output, weights, bias, workspace));
		if (add.isEmpty() == false)
		{
			assert(same_device(context, add));
			assert(same_shape(output, add));
		}

		assert(weights.dim(1) == weights.dim(2));
		const int kernel_size = weights.dim(1);

		Tensor input_matrix;
		if (kernel_size == 1)
			input_matrix = input.view( { input.shape().volumeWithoutLastDim(), input.lastDim() });
		else
		{
			input_matrix = workspace.view( { input.shape().volumeWithoutLastDim(), weights.shape().volumeWithoutFirstDim() });
			im2row(context, input_matrix, input, kernel_size, false, nullptr);
		}

		Tensor output_matrix = output.view( { output.shape().volumeWithoutLastDim(), output.lastDim() });
		Tensor weight_matrix = weights.view( { weights.firstDim(), weights.shape().volumeWithoutFirstDim() });
		const float beta = add.isEmpty() ? 0.0f : 1.0f;
		const Tensor ext = add.isEmpty() ? output_matrix : add.view(output_matrix.shape());
		gemm_ex(context, output_matrix, 1.0f, 'n', input_matrix, 't', weight_matrix, beta, ext, bias, activation);
		SYNC();
	}
	void explicit_gemm_backward(const Context &context, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &output, const Tensor &weights,
			Tensor &workspace, float beta)
	{
		assert(same_device(context, gradient_prev, gradient_next, weights, workspace));
		assert(weights.dim(1) == weights.dim(2));
		const int kernel_size = weights.dim(1);

		Tensor gradient_prev_matrix = gradient_prev.view( { gradient_prev.shape().volumeWithoutLastDim(), gradient_prev.lastDim() });
		if (kernel_size == 1)
		{
			Tensor weight_matrix = weights.view( { weights.firstDim(), weights.shape().volumeWithoutFirstDim() });
			Tensor gradient_next_matrix = gradient_next.view(
					{ gradient_next.shape().volumeWithoutLastDim(), weights.shape().volumeWithoutLastDim() });

			gemm(context, 'n', 'n', gradient_prev_matrix, gradient_next_matrix, weight_matrix, 1, beta);
		}
		else
		{
			Tensor inv_weight = workspace.view( { weights.dim(3), kernel_size, kernel_size, weights.dim(0) });
			transpose(context, inv_weight, weights, { 3, 1, 2, 0 });
			Tensor weight_matrix = inv_weight.view( { inv_weight.firstDim(), inv_weight.shape().volumeWithoutFirstDim() });

			Tensor gradient_next_matrix = workspace.view( { gradient_next.shape().volumeWithoutLastDim(), weights.shape().volumeWithoutLastDim() },
					inv_weight.volume());
			im2row(context, gradient_next_matrix, gradient_next, kernel_size, true, nullptr);

			gemm(context, 'n', 't', gradient_prev_matrix, gradient_next_matrix, weight_matrix, 1, beta);
		}SYNC();
	}
	void explicit_gemm_update(const Context &context, const Tensor &input, const Tensor &gradient_next, Tensor &weight_update, Tensor &workspace)
	{
		assert(same_device(context, input, gradient_next, weight_update, workspace));
		assert(weight_update.dim(1) == weight_update.dim(2));
		const int kernel_size = weight_update.dim(1);

		Tensor input_matrix;
		if (kernel_size == 1)
			input_matrix = input.view( { input.shape().volumeWithoutLastDim(), input.lastDim() });
		else
		{
			input_matrix = workspace.view( { input.shape().volumeWithoutLastDim(), weight_update.shape().volumeWithoutFirstDim() });
			im2row(context, input_matrix, input, kernel_size, false, nullptr);
		}
		Tensor weight_update_matrix = weight_update.view( { weight_update.firstDim(), weight_update.shape().volumeWithoutFirstDim() });
		Tensor gradient_matrix = gradient_next.view( { gradient_next.shape().volumeWithoutLastDim(), gradient_next.lastDim() });
		gemm(context, 't', 'n', weight_update_matrix, gradient_matrix, input_matrix, 1, 0);
		SYNC();
	}

} /* namespace ml */

