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
			case ml::DTYPE_FLOAT16:
			{
				switch (dst)
				{
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
					case ml::DTYPE_FLOAT16:
						return get_conversion_function_fp64_to_fp16(context);
					case ml::DTYPE_FLOAT64:
						return get_conversion_function_fp64_to_fp32(context);
				}
			}
		}
	}

	template<typename T>
	T one_or_zero(bool b) noexcept;

	template<>
	double one_or_zero(bool b) noexcept
	{
		return b ? 1.0 : 0.0;
	}
	template<>
	float one_or_zero(bool b) noexcept
	{
		return b ? 1.0f : 0.0f;
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

	void cpu_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
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
		if (act == ACTIVATION_SOFTMAX)
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
		else
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
	}
	void cpu_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		cpu::def_kernel_activation_backward_fp32(gradient_prev, gradient_next, output, volume(shape), act);
	}

} /* namespace avocado */

