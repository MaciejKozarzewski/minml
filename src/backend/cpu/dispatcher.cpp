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

	template<typename T>
	T one_or_zero(bool b) noexcept;

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
		assert(last_dim <= 32);
		for (int i = 0; i < first_dim; i++, dst += last_dim)
		{
			uint32_t mask = src[i];
			for (int j = 0; j < last_dim; j++, mask >>= 1)
				dst[j] = one_or_zero<T>(mask & 1u);
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

		if (dst_dtype == DTYPE_FLOAT16 and src_dtype == DTYPE_FLOAT32)
		{
			static const conversion_function func = get_conversion_function_fp32_to_fp16(context);
			func(dst, src, elements);
		}
		if (dst_dtype == DTYPE_FLOAT32 and src_dtype == DTYPE_FLOAT16)
		{
			static const conversion_function func = get_conversion_function_fp16_to_fp32(context);
			func(dst, src, elements);
		}
	}
	void cpu_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
	}

	void cpu_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
	}

	// implemented in 'global_pooling.cpp'
	void cpu_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output,
			void *max_indices)
	{
	}
	void cpu_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
			const void *max_indices)
	{
	}

	void cpu_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias, mlActivationType_t act)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
			{
				const cpu::SimdLevel simd_level = cpu::Context::getSimdLevel(context);
				if (simd_level >= cpu::SimdLevel::AVX and cpu::has_hardware_fp16_conversion())
					cpu::avx_kernel_add_bias_act_fp16(input, bias, first_dim, last_dim, act);
				else
					cpu::def_kernel_add_bias_act_fp16(input, bias, first_dim, last_dim, act);
				break;
			}
			case DTYPE_FLOAT32:
			{
				cpu::def_kernel_add_bias_act_fp32(input, bias, first_dim, last_dim, act);
				break;
			}
			default:
				break;
		}
	}

	void cpu_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, mlActivationType_t act)
	{
		const cpu::SimdLevel simd_level = cpu::Context::getSimdLevel(context);
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
					if (simd_level >= cpu::SimdLevel::AVX and cpu::has_hardware_fp16_conversion())
					{
						if (last_dim == 3)
							cpu::avx_kernel_softmax_3_channels_fp16(output, input, first_dim);
						else
							cpu::avx_kernel_softmax_fp16(output, input, first_dim, last_dim, workspace);
					}
					else
					{
						if (last_dim == 3)
							cpu::def_kernel_softmax_3_channels_fp16(output, input, first_dim);
						else
							cpu::def_kernel_softmax_fp16(output, input, first_dim, last_dim, workspace);
					}
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
					if (simd_level >= cpu::SimdLevel::AVX and cpu::has_hardware_fp16_conversion())
						cpu::avx_kernel_activation_forward_fp16(output, input, volume(shape), act);
					else
						cpu::def_kernel_activation_forward_fp16(output, input, volume(shape), act);
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

