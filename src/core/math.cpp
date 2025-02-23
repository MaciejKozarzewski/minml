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
		for (int i = shape.rank(); i < 4; i++)
			result.dim[i] = 0;
		return result;
	}
	mlShape_t get_shape(const Tensor &tensor) noexcept
	{
		return get(tensor.shape());
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
				cuda_winograd_weight_transform(get(context), tile_size, get(weights.dtype()), get_shape(weights), weights.data(), matrices.data(),
						invert);
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
				cuda_winograd_input_transform(get(context), tile_size, get(input.dtype()), get(weight_shape), get_shape(input), input.data(),
						matrices.data());
				break;
			case DeviceType::OPENCL:
				opencl_winograd_input_transform(get(context), tile_size, get(input.dtype()), get(weight_shape), get_shape(input), input.data(),
						matrices.data());
				break;
		}SYNC();
	}
	void winogradOutputTransform(const Context &context, const Shape &weight_shape, const Tensor &matrices, Tensor &output, const Tensor &bias,
			const Tensor &add, ActivationType act)
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
				cuda_winograd_output_transform(get(context), tile_size, get(output.dtype()), get(weight_shape), get_shape(output), matrices.data(),
						output.data(), bias.data(), add.data(), get(act));
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
				cuda_winograd_gradient_transform(get(context), tile_size, get(gradient.dtype()), get(weight_shape), get_shape(gradient),
						gradient.data(), matrices.data());
				break;
			case DeviceType::OPENCL:
				opencl_winograd_gradient_transform(get(context), tile_size, get(gradient.dtype()), get(weight_shape), get_shape(gradient),
						gradient.data(), matrices.data());
				break;
		}
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
				cuda_winograd_update_transform(get(context), tile_size, get(matrices.dtype()), get_shape(update), matrices.data(), update.data());
				break;
			case DeviceType::OPENCL:
				opencl_winograd_update_transform(get(context), tile_size, get(matrices.dtype()), get_shape(update), matrices.data(), update.data());
				break;
		}
	}

	void im2row(const Context &context, const Shape &weight_shape, const Tensor &input, Tensor &matrix)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_im2row(get(context), get(input.dtype()), get(weight_shape), get_shape(input), input.data(), matrix.data());
				break;
			case DeviceType::CUDA:
				// TODO
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}
	}
	void depthToSpace(const Context &context, const Tensor &input, Tensor &output)
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
	void spaceToDepth(const Context &context, const Tensor &input, Tensor &output)
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

	void depthwiseConvForward(const Context &context, const Tensor &input, const Tensor &weights, Tensor &output, const Tensor &bias)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_depthwise_conv_forward(get(context), get(input.dtype()), get_shape(input), get_shape(weights), input.data(), weights.data(),
						bias.data(), output.data());
				break;
			case DeviceType::CUDA:
				cuda_depthwise_conv_forward(get(context), get(input.dtype()), get_shape(input), get_shape(weights), input.data(), weights.data(),
						bias.data(), output.data());
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}
	}
	void depthwiseConvBackward(const Context &context, const Tensor &gradient_next, const Tensor &weights, Tensor &gradient_prev)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_depthwise_conv_backward(get(context), get_shape(gradient_prev), get_shape(weights), gradient_next.data(), weights.data(),
						gradient_prev.data());
				break;
			case DeviceType::CUDA:
				cuda_depthwise_conv_backward(get(context), get_shape(gradient_prev), get_shape(weights), gradient_next.data(), weights.data(),
						gradient_prev.data());
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}
	}
	void depthwiseConvUpdate(const Context &context, const Tensor &input, const Tensor &gradient_next, Tensor &weights_update)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_depthwise_conv_update(get(context), get_shape(input), get_shape(weights_update), input.data(), gradient_next.data(),
						weights_update.data());
				break;
			case DeviceType::CUDA:
				cuda_depthwise_conv_update(get(context), get_shape(input), get_shape(weights_update), input.data(), gradient_next.data(),
						weights_update.data());
				break;
			case DeviceType::OPENCL:
				// TODO
				break;
		}
	}

	void globalAvgAndMaxPoolingForward(const Context &context, const Tensor &input, Tensor &output)
	{
		static Timer timer("global_pooling");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_global_avg_and_max_pooling_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data());
				break;
			case DeviceType::CUDA:
				cuda_global_avg_and_max_pooling_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data());
				break;
			case DeviceType::OPENCL:
				opencl_global_avg_and_max_pooling_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data());
				break;
		}SYNC();
	}
	void globalAvgAndMaxPoolingBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &input,
			const Tensor &output)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_global_avg_and_max_pooling_backward(get(context), get_shape(input), gradient_prev.data(), gradient_next.data(), input.data(),
						output.data());
				break;
			case DeviceType::CUDA:
				cuda_global_avg_and_max_pooling_backward(get(context), get_shape(input), gradient_prev.data(), gradient_next.data(), input.data(),
						output.data());
				break;
			case DeviceType::OPENCL:
				opencl_global_avg_and_max_pooling_backward(get(context), get_shape(input), gradient_prev.data(), gradient_next.data(), input.data(),
						output.data());
				break;
		}
	}
	void globalBroadcastingForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &bias, ActivationType act)
	{
		static Timer timer("global_broadcasting");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_global_broadcasting_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), bias.data(),
						get(act));
				break;
			case DeviceType::CUDA:
				cuda_global_broadcasting_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), bias.data(),
						get(act));
				break;
			case DeviceType::OPENCL:
				opencl_global_broadcasting_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), bias.data(),
						get(act));
				break;
		}SYNC();
	}
	void globalBroadcastingBackward(const Context &context, Tensor &gradient_prev, Tensor &gradient_next, const Tensor &output, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_global_broadcasting_backward(get(context), get_shape(output), gradient_prev.data(), gradient_next.data(), output.data(),
						get(act));
				break;
			case DeviceType::CUDA:
				cuda_global_broadcasting_backward(get(context), get_shape(output), gradient_prev.data(), gradient_next.data(), output.data(),
						get(act));
				break;
			case DeviceType::OPENCL:
				opencl_global_broadcasting_backward(get(context), get_shape(output), gradient_prev.data(), gradient_next.data(), output.data(),
						get(act));
				break;
		}
	}

	void globalAveragePoolingForward(const Context &context, const Tensor &input, Tensor &output)
	{
		static Timer timer("global_average_pooling");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_global_average_pooling_forward(get(context), get(input.dtype()), get(output.dtype()), get_shape(input), output.data(),
						input.data(), 1.0f, 0.0f);
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void globalAveragePoolingBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_global_average_pooling_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data());
				break;
			case DeviceType::OPENCL:
				break;
		}
	}
	void channelScalingForward(const Context &context, const Tensor &input, Tensor &output, const Tensor &scales)
	{
		static Timer timer("channel_scaling");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_channel_scaling_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), scales.data());
				break;
			case DeviceType::OPENCL:
				break;
		}SYNC();
	}
	void channelScalingBackward(const Context &context, Tensor &gradient_prev_0, Tensor &gradient_prev_1, const Tensor &gradient_next,
			const Tensor &input_0, const Tensor &input_1)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				break;
			case DeviceType::CUDA:
				cuda_channel_scaling_backward(get(context), get_shape(gradient_prev_0), gradient_prev_0.data(), gradient_prev_1.data(),
						gradient_next.data(), input_0.data(), input_1.data());
				break;
			case DeviceType::OPENCL:
				break;
		}
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
				cuda_gemm(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB, alpha,
						beta);
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
				cuda_gemm_batched(get(context), get(C.dtype()), get_shape(C), C.data(), get_shape(A), A.data(), get_shape(B), B.data(), opA, opB,
						alpha, beta);
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

	void addBiasAct(const Context &context, Tensor &output, const Tensor &input, const Tensor &bias, ActivationType act)
	{
		static Timer timer("addBiasAct");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_add_bias_act(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), bias.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_add_bias_act(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), bias.data(), get(act));
				break;
			case DeviceType::OPENCL:
				opencl_add_bias_act(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), bias.data(), get(act));
				break;
		}SYNC();
	}

	void batchnormInference(const Context &context, const Tensor &input, Tensor &output, const Tensor &weights, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_batchnorm_inference(get(context), get_shape(input), input.data(), output.data(), weights.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_inference(get(context), get_shape(input), input.data(), output.data(), weights.data(), get(act));
				break;
			case DeviceType::OPENCL:
				opencl_batchnorm_inference(get(context), get_shape(input), input.data(), output.data(), weights.data(), get(act));
				break;
		}
	}
	void batchnormForward(const Context &context, const Tensor &input, Tensor &output, Tensor &weights, Tensor &running_stats, int running_stat_idx,
			ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_batchnorm_forward(get(context), get_shape(input), input.data(), output.data(), weights.data(), running_stats.data(),
						running_stat_idx, get(act));
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_forward(get(context), get_shape(input), input.data(), output.data(), weights.data(), running_stats.data(),
						running_stat_idx, get(act));
				break;
			case DeviceType::OPENCL:
				opencl_batchnorm_forward(get(context), get_shape(input), input.data(), output.data(), weights.data(), running_stats.data(),
						running_stat_idx, get(act));
				break;
		}
	}
	void batchnormBackward(const Context &context, const Tensor &input, const Tensor &output, Tensor &gradient_prev, Tensor &gradient_next,
			const Tensor &weights, Tensor &weights_update, const Tensor &running_stats, int running_stat_idx, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_batchnorm_backward(get(context), get_shape(input), input.data(), output.data(), gradient_prev.data(), gradient_next.data(),
						weights.data(), weights_update.data(), running_stats.data(), running_stat_idx, get(act));
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_backward(get(context), get_shape(input), input.data(), output.data(), gradient_prev.data(), gradient_next.data(),
						weights.data(), weights_update.data(), running_stats.data(), running_stat_idx, get(act));
				break;
			case DeviceType::OPENCL:
				opencl_batchnorm_backward(get(context), get_shape(input), input.data(), output.data(), gradient_prev.data(), gradient_next.data(),
						weights.data(), weights_update.data(), running_stats.data(), running_stat_idx, get(act));
				break;
		}
	}
	void batchnormUpdate(const Context &context, const Tensor &running_stat, int stats_to_average, Tensor &weights, bool use_gamma, bool use_beta)
	{
		mlShape_t shape = get_shape(running_stat);
		shape.dim[0] = stats_to_average;
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_batchnorm_update(get(context), shape, running_stat.data(), weights.data(), use_gamma, use_beta);
				break;
			case DeviceType::CUDA:
				cuda_batchnorm_update(get(context), shape, running_stat.data(), weights.data(), use_gamma, use_beta);
				break;
			case DeviceType::OPENCL:
				opencl_batchnorm_update(get(context), shape, running_stat.data(), weights.data(), use_gamma, use_beta);
				break;
		}
	}
	void foldBatchnorm(const Context &context, Tensor &layer_weights, Tensor &layer_bias, const Tensor &batchnorm_weights)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_fold_batchnorm(get(context), get_shape(layer_weights), layer_weights.data(), layer_bias.data(), batchnorm_weights.data());
				break;
			case DeviceType::CUDA:
				cuda_fold_batchnorm(get(context), get_shape(layer_weights), layer_weights.data(), layer_bias.data(), batchnorm_weights.data());
				break;
			case DeviceType::OPENCL:
				opencl_fold_batchnorm(get(context), get_shape(layer_weights), layer_weights.data(), layer_bias.data(), batchnorm_weights.data());
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
			Tensor &weights_update, Tensor &bias_update)
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
			Tensor &weights_update)
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
			Tensor &backwardData, int num_heads, bool symmetric)
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

	void activationForward(const Context &context, Tensor &output, const Tensor &input, ActivationType act)
	{
		static Timer timer("activationForward");
		TimerGuard tg(timer);
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_activation_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_activation_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), get(act));
				break;
			case DeviceType::OPENCL:
				opencl_activation_forward(get(context), get(input.dtype()), get_shape(input), output.data(), input.data(), get(act));
				break;
		}SYNC();
	}
	void activationBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &output, ActivationType act)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_activation_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), output.data(), get(act));
				break;
			case DeviceType::CUDA:
				cuda_activation_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), output.data(), get(act));
				break;
			case DeviceType::OPENCL:
				opencl_activation_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), output.data(),
						get(act));
				break;
		}
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
	void geluBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &input)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_gelu_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), input.data());
				break;
			case DeviceType::CUDA:
				cuda_gelu_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), input.data());
				break;
			case DeviceType::OPENCL:
				opencl_gelu_backward(get(context), get_shape(gradient_prev), gradient_prev.data(), gradient_next.data(), input.data());
				break;
		}
	}

	void emulateLowPrecision(const Context &context, Tensor &dst, const Tensor &src)
	{
		if (dst.dtype() != DataType::FLOAT32 or src.dtype() != DataType::FLOAT32)
			return;
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_emulate_low_precision(get(context), get_shape(dst), dst.data(), src.data());
				break;
			case DeviceType::CUDA:
				cuda_emulate_low_precision(get(context), get_shape(dst), dst.data(), src.data());
				break;
			case DeviceType::OPENCL:
				opencl_emulate_low_precision(get(context), get_shape(dst), dst.data(), src.data());
				break;
		}
	}
	void sumOverFirstDim(const Context &context, Tensor &dst, const Tensor &src, float beta)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_sum_over_first_dim(get(context), get_shape(src), dst.data(), src.data(), beta);
				break;
			case DeviceType::CUDA:
				cuda_sum_over_first_dim(get(context), get_shape(src), dst.data(), src.data(), beta);
				break;
			case DeviceType::OPENCL:
				opencl_sum_over_first_dim(get(context), get_shape(src), dst.data(), src.data(), beta);
				break;
		}
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
		addTensors(context, dst, 1.0f, src1, 1.0f, src2);
	}
	void addTensors(const Context &context, Tensor &dst, float alpha1, const Tensor &src1, float alpha2, const Tensor &src2)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_add_tensors(get(context), get(dst.dtype()), get_shape(dst), dst.data(), alpha1, src1.data(), alpha2, src2.data());
				break;
			case DeviceType::CUDA:
				cuda_add_tensors(get(context), get(dst.dtype()), get_shape(dst), dst.data(), alpha1, src1.data(), alpha2, src2.data());
				break;
			case DeviceType::OPENCL:
				opencl_add_tensors(get(context), get(dst.dtype()), get_shape(dst), dst.data(), alpha1, src1.data(), alpha2, src2.data());
				break;
		}
	}
	float meanSquaredLoss(const Context &context, const Tensor &output, const Tensor &target)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				return cpu_mean_squared_loss(get(context), get_shape(output), output.data(), target.data());
			case DeviceType::CUDA:
				return cuda_mean_squared_loss(get(context), get_shape(output), output.data(), target.data());
			case DeviceType::OPENCL:
				return opencl_mean_squared_loss(get(context), get_shape(output), output.data(), target.data());
		}
		return 0.0f;
	}
	void meanSquaredGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target, float weight)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_mean_squared_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
			case DeviceType::CUDA:
				cuda_mean_squared_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
			case DeviceType::OPENCL:
				opencl_mean_squared_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
		}
	}
	float crossEntropyLoss(const Context &context, const Tensor &output, const Tensor &target)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				return cpu_cross_entropy_loss(get(context), get_shape(output), output.data(), target.data());
			case DeviceType::CUDA:
				return cuda_cross_entropy_loss(get(context), get_shape(output), output.data(), target.data());
			case DeviceType::OPENCL:
				return opencl_cross_entropy_loss(get(context), get_shape(output), output.data(), target.data());
		}
		return 0.0f;
	}
	void crossEntropyGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target, float weight)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_cross_entropy_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
			case DeviceType::CUDA:
				cuda_cross_entropy_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
			case DeviceType::OPENCL:
				opencl_cross_entropy_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
		}
	}
	float valueHeadLoss(const Context &context, const Tensor &output, const Tensor &target)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				return cpu_value_head_loss(get(context), get_shape(output), output.data(), target.data());
			case DeviceType::CUDA:
				return cuda_value_head_loss(get(context), get_shape(output), output.data(), target.data());
			case DeviceType::OPENCL:
				return opencl_value_head_loss(get(context), get_shape(output), output.data(), target.data());
		}
		return 0.0f;
	}
	void valueHeadGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target, float weight)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_value_head_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
			case DeviceType::CUDA:
				cuda_value_head_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
			case DeviceType::OPENCL:
				opencl_value_head_gradient(get(context), get_shape(output), gradient.data(), output.data(), target.data(), weight);
				break;
		}
	}

	void radamOptimize(const Context &context, Tensor &weight, const Tensor &update, Tensor &momentum, Tensor &variance, float learning_rate,
			float beta1, float beta2, int step, float weight_decay)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_radam_optimize(get(context), get_shape(weight), weight.data(), update.data(), momentum.data(), variance.data(), learning_rate,
						beta1, beta2, step, weight_decay);
				break;
			case DeviceType::CUDA:
				cuda_radam_optimize(get(context), get_shape(weight), weight.data(), update.data(), momentum.data(), variance.data(), learning_rate,
						beta1, beta2, step, weight_decay);
				break;
			case DeviceType::OPENCL:
				opencl_radam_optimize(get(context), get_shape(weight), weight.data(), update.data(), momentum.data(), variance.data(), learning_rate,
						beta1, beta2, step, weight_decay);
				break;
		}
	}

	void l2Regularization(const Context &context, Tensor &gradient, const Tensor &param, float coefficient, float offset)
	{
		switch (context.device().type())
		{
			case DeviceType::CPU:
				cpu_l2_regularization(get(context), get_shape(gradient), gradient.data(), param.data(), coefficient, offset);
				break;
			case DeviceType::CUDA:
				cuda_l2_regularization(get(context), get_shape(gradient), gradient.data(), param.data(), coefficient, offset);
				break;
			case DeviceType::OPENCL:
				opencl_l2_regularization(get(context), get_shape(gradient), gradient.data(), param.data(), coefficient, offset);
				break;
		}
	}

}

