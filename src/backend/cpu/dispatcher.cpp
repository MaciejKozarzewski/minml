/*
 * dispatcher.cpp
 *
 *  Created on: Jan 15, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "misc_kernels.hpp"
#include "utils.hpp"
#include "indexers.hpp"
#include "fp16.hpp"

#include <functional>
#include <iostream>
#include <cstring>

namespace
{

	using conversion_function = std::function<void(void*, const void*, size_t)>;

	conversion_function get_conversion_function_fp32_to_fp16(ml::mlContext_t context)
	{
		if (ml::cpu::Context::getSimdLevel(context) >= ml::cpu::SimdLevel::AVX and ml::cpu::has_hardware_fp16_conversion())
			return ml::cpu::avx_kernel_convert_fp32_to_fp16;
		else
			return ml::cpu::def_kernel_convert_fp32_to_fp16;
	}
	conversion_function get_conversion_function_fp16_to_fp32(ml::mlContext_t context)
	{
		if (ml::cpu::Context::getSimdLevel(context) >= ml::cpu::SimdLevel::AVX and ml::cpu::has_hardware_fp16_conversion())
			return ml::cpu::avx_kernel_convert_fp16_to_fp32;
		else
			return ml::cpu::def_kernel_convert_fp16_to_fp32;
	}
	conversion_function get_conversion_function_fp64_to_fp16(ml::mlContext_t context)
	{
		return ml::cpu::def_kernel_convert_fp64_to_fp16;
	}
	conversion_function get_conversion_function_fp16_to_fp64(ml::mlContext_t context)
	{
		return ml::cpu::def_kernel_convert_fp16_to_fp64;
	}
	conversion_function get_conversion_function_fp64_to_fp32(ml::mlContext_t context)
	{
		return ml::cpu::def_kernel_convert_fp64_to_fp32;
	}
	conversion_function get_conversion_function_fp32_to_fp64(ml::mlContext_t context)
	{
		return ml::cpu::def_kernel_convert_fp32_to_fp64;
	}

	conversion_function get_conversion_function(ml::mlContext_t context, ml::mlDataType_t src, ml::mlDataType_t dst)
	{
		switch (src)
		{
			default:
				throw std::logic_error("unsupported conversion");
			case ml::DTYPE_FLOAT16:
			{
				switch (dst)
				{
					default:
						throw std::logic_error("unsupported conversion");
					case ml::DTYPE_FLOAT32:
						return get_conversion_function_fp16_to_fp32(context);
					case ml::DTYPE_FLOAT64:
						return get_conversion_function_fp16_to_fp64(context);
				}
			}
			case ml::DTYPE_FLOAT32:
			{
				switch (dst)
				{
					default:
						throw std::logic_error("unsupported conversion");
					case ml::DTYPE_FLOAT16:
						return get_conversion_function_fp32_to_fp16(context);
					case ml::DTYPE_FLOAT64:
						return get_conversion_function_fp32_to_fp64(context);
				}
			}
			case ml::DTYPE_FLOAT64:
			{
				switch (dst)
				{
					default:
						throw std::logic_error("unsupported conversion");
					case ml::DTYPE_FLOAT16:
						return get_conversion_function_fp64_to_fp16(context);
					case ml::DTYPE_FLOAT64:
						return get_conversion_function_fp64_to_fp32(context);
				}
			}
		}
	}

	template<typename T>
	T one_or_zero(bool b) noexcept
	{
		return b ? static_cast<T>(1) : static_cast<T>(0);
	}
	template<>
	ml::cpu::float16 one_or_zero(bool b) noexcept
	{
		return b ? ml::cpu::float16(0x3c00) : ml::cpu::float16(0x0000);
	}

	template<typename T>
	void kernel_unpack_input(T *dst, const uint32_t *src, int first_dim, int last_dim)
	{
		const int stride = (last_dim + 31) / 32;
		for (int i = 0; i < first_dim; i++, dst += last_dim)
			for (int j = 0; j < last_dim; j++)
			{
				const int int_idx = j / 32;
				const int bit_idx = j % 32;
				dst[j] = one_or_zero<T>((src[i * stride + int_idx] >> bit_idx) & 1u);
			}
	}

	int get_patch_size(int smaller, int larger) noexcept
	{
		assert(smaller <= larger);
		for (int i = 1;; i++)
		{
			const int tmp = (larger + i - 1) / i;
			if (tmp == smaller)
				return i;
			if (tmp < smaller)
				break;
		}
		return 0;
	}
}

namespace ml
{

	void cpu_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		switch (dst_dtype)
		{
			case DTYPE_FLOAT16:
				kernel_unpack_input(getPointer<cpu::float16>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_FLOAT32:
				kernel_unpack_input(getPointer<float>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_FLOAT64:
				kernel_unpack_input(getPointer<double>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_UINT8:
				kernel_unpack_input(getPointer<uint8_t>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_INT8:
				kernel_unpack_input(getPointer<int8_t>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_INT16:
				kernel_unpack_input(getPointer<int16_t>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_INT32:
				kernel_unpack_input(getPointer<int32_t>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			default:
				break;
		}
	}
	void cpu_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		if (dst_dtype == src_dtype)
		{ // same type
			if (dst != src)
				std::memcpy(dst, src, size_of(dst_dtype) * elements); // different locations, can just copy memory
			return;
		}

		const conversion_function func = get_conversion_function(context, src_dtype, dst_dtype);
		func(dst, src, elements);
	}
	void cpu_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
	}
	void cpu_space_to_depth(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, const void *input, mlShape_t output_shape, void *output)
	{
		const int batch_size = get_first_dim(input_shape);
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int patch_size_h = get_patch_size(output_shape.dim[1], input_shape.dim[1]);
		const int patch_size_w = get_patch_size(output_shape.dim[2], input_shape.dim[2]);
		assert(patch_size_h != 0 && patch_size_w != 0);
		const int channels_in = get_last_dim(input_shape);
		const int channels_out = get_last_dim(output_shape);
		assert(channels_in * patch_size_h * patch_size_w == channels_out);

		const uint8_t *input_ptr = reinterpret_cast<const uint8_t*>(input);
		uint8_t *output_ptr = reinterpret_cast<uint8_t*>(output);
		Indexer<4> input_indexer(batch_size, height, width, channels_in);

		const int block_size = size_of(dtype) * channels_in;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h += patch_size_h)
				for (int w = 0; w < width; w += patch_size_w)
					for (int x = 0; x < patch_size_h; x++)
						for (int y = 0; y < patch_size_w; y++)
						{
							if ((h + x) < height and (w + y) < width)
								std::memcpy(output_ptr, input_ptr + size_of(dtype) * input_indexer.at(b, h + x, w + y, 0), block_size);
							else
								std::memset(output_ptr, 0, block_size);
							output_ptr += block_size;
						}
	}
	void cpu_depth_to_space(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, const void *input, mlShape_t output_shape, void *output)
	{
		const int batch_size = get_first_dim(input_shape);
		const int height = output_shape.dim[1];
		const int width = output_shape.dim[2];
		const int patch_size_h = get_patch_size(input_shape.dim[1], output_shape.dim[1]);
		const int patch_size_w = get_patch_size(input_shape.dim[2], output_shape.dim[2]);
		assert(patch_size_h != 0 && patch_size_w != 0);
		const int channels_in = get_last_dim(input_shape);
		const int channels_out = get_last_dim(output_shape);
		assert(channels_out * patch_size_h * patch_size_w == channels_in);

		const uint8_t *input_ptr = reinterpret_cast<const uint8_t*>(input);
		uint8_t *output_ptr = reinterpret_cast<uint8_t*>(output);
		Indexer<4> output_indexer(batch_size, height, width, channels_out);

		const int block_size = size_of(dtype) * channels_out;
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h += patch_size_h)
				for (int w = 0; w < width; w += patch_size_w)
					for (int x = 0; x < patch_size_h; x++)
						for (int y = 0; y < patch_size_w; y++)
						{
							if ((h + x) < height and (w + y) < width)
								std::memcpy(output_ptr + size_of(dtype) * output_indexer.at(b, h + x, w + y, 0), input_ptr, block_size);
							input_ptr += block_size;
						}
	}

	void cpu_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
	}
	void cpu_depthwise_conv_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape, const void *input,
			const void *weights, const void *bias, void *output)
	{
	}
	void cpu_depthwise_conv_backward(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *gradient_next,
			const void *weights, void *gradient_prev)
	{
	}
	void cpu_depthwise_conv_update(mlContext_t context, mlShape_t input_shape, mlShape_t weights_shape, const void *input, const void *gradient_next,
			void *weights_update)
	{

	}

	void cpu_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				assert(cpu_supports_type(DTYPE_FLOAT16));
				cpu::avx_kernel_global_avg_and_max_pooling_forward_fp16(context, shape, input, output);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (cpu::getSimdSupport() >= cpu::SimdLevel::AVX)
					cpu::avx_kernel_global_avg_and_max_pooling_forward_fp32(context, shape, input, output);
				else
					cpu::def_kernel_global_avg_and_max_pooling_forward_fp32(context, shape, input, output);
				break;
			}
			default:
				break;
		}
	}
	void cpu_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
			const void *input, const void *output)
	{
		cpu::def_kernel_global_avg_and_max_pooling_backward(context, shape, gradient_prev, gradient_next, input, output);
	}
	void cpu_global_broadcasting_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *bias,
			mlActivationType_t act)
	{
		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				assert(cpu_supports_type(DTYPE_FLOAT16));
				cpu::avx_kernel_global_broadcasting_forward_fp16(context, shape, output, input, bias, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (cpu::getSimdSupport() >= cpu::SimdLevel::AVX)
					cpu::avx_kernel_global_broadcasting_forward_fp32(context, shape, output, input, bias, act);
				else
					cpu::def_kernel_global_broadcasting_forward_fp32(context, shape, output, input, bias, act);
				break;
			}
			default:
				break;
		}
	}
	void cpu_global_broadcasting_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		cpu::def_kernel_global_broadcasting_backward(context, shape, gradient_prev, gradient_next, output, act);
	}

	void cpu_global_average_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
		const int batch_size = shape.dim[0];
		const int hw = shape.dim[1] * shape.dim[2];
		const int channels = shape.dim[3];

		for (int b = 0; b < batch_size; b++)
		{

		}
	}
	void cpu_global_average_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next)
	{
	}
	void cpu_channel_scaling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *scales)
	{
	}
	void cpu_channel_scaling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev_0, void *gradient_prev_1, const void *gradient_next,
			const void *input_0, const void *input_1)
	{
	}

	void cpu_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, const void *bias,
			mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				assert(cpu_supports_type(DTYPE_FLOAT16));
				cpu::avx_kernel_add_bias_act_fp16(output, input, bias, first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				cpu::def_kernel_add_bias_act_fp32(output, input, bias, first_dim, last_dim, act);
				break;
			}
			default:
				break;
		}
	}

	void cpu_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, mlActivationType_t act)
	{
		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				assert(cpu_supports_type(DTYPE_FLOAT16));
				cpu::avx_kernel_activation_forward_fp16(output, input, volume(shape), act);
				break;
			}
			case DTYPE_FLOAT32:
				cpu::def_kernel_activation_forward_fp32(output, input, volume(shape), act);
				break;
			default:
				break;
		}
	}
	void cpu_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		cpu::def_kernel_activation_backward_fp32(gradient_prev, gradient_next, nullptr, output, volume(shape), act);
	}
	void cpu_fused_bias_and_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, void *gradient_next, const void *output,
			void *bias_gradient, mlActivationType_t act, float beta_prev, float beta_bias)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		assert(cpu::Context::getWorkspaceSize(context) >= last_dim * sizeof(float));

		float *tmp_ptr = cpu::Context::getWorkspace<float>(context);
		float *prev_ptr = getPointer<float>(gradient_prev);
		float *next_ptr = getPointer<float>(gradient_next);
		const float *output_ptr = getPointer<float>(output);
		float *bias_gradient_ptr = getPointer<float>(bias_gradient);

		for (int j = 0; j < last_dim; j++)
			tmp_ptr[j] = 0.0f;

		for (int i = 0; i < first_dim; i++)
		{
			switch (act)
			{
				case ACTIVATION_SIGMOID:
					for (int j = 0; j < last_dim; j++)
						next_ptr[i] *= output_ptr[j] * (1.0f - output_ptr[j]);
					break;
				case ACTIVATION_TANH:
					for (int j = 0; j < last_dim; j++)
						next_ptr[i] *= (1.0f + output_ptr[j]) * (1.0f - output_ptr[j]);
					break;
				case ACTIVATION_RELU:
					for (int j = 0; j < last_dim; j++)
						next_ptr[i] = (output_ptr[j] == 0.0f) ? 0.0f : next_ptr[j];
					break;
				case ACTIVATION_EXP:
					for (int j = 0; j < last_dim; j++)
						next_ptr[i] *= output_ptr[j];
					break;
				default:
					break;
			}
			for (int j = 0; j < last_dim; j++)
				tmp_ptr[j] += next_ptr[i * last_dim + j];

			if (prev_ptr != nullptr)
			{
				if (beta_prev != 0.0f)
					for (int j = 0; j < last_dim; j++)
						prev_ptr[j] = next_ptr[i * last_dim + j];
				else
					for (int j = 0; j < last_dim; j++)
						prev_ptr[j] = beta_prev * prev_ptr[j] + next_ptr[i * last_dim + j];
			}
		}

		if (beta_bias == 0.0f)
			for (int j = 0; j < last_dim; j++)
				bias_gradient_ptr[j] = tmp_ptr[j];
		else
			for (int j = 0; j < last_dim; j++)
				bias_gradient_ptr[j] = bias_gradient_ptr[j] * beta_bias + tmp_ptr[j];
	}

	void cpu_softmax_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
		const int first_dim = get_first_dim(shape);
		const int last_dim = get_last_dim(shape);

		assert(cpu::Context::getWorkspaceSize(context) >= sizeof(float) * last_dim);
		float *workspace = cpu::Context::getWorkspace<float>(context);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				assert(cpu_supports_type(DTYPE_FLOAT16));
				if (last_dim == 3)
					cpu::avx_kernel_softmax_3_channels_fp16(output, input, first_dim);
				else
					cpu::avx_kernel_softmax_fp16(output, input, first_dim, last_dim, workspace);
				break;
			}
			case DTYPE_FLOAT32:
			{
				if (last_dim == 3)
					cpu::def_kernel_softmax_3_channels_fp32(output, input, first_dim);
				else
					cpu::def_kernel_softmax_fp32(output, input, first_dim, last_dim, workspace);
				break;
			}
			default:
				break;
		}
	}
} /* namespace avocado */

