/*
 * dispatcher.cpp
 *
 *  Created on: Jan 15, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "kernel_definitions.hpp"

#include <functional>

namespace
{

#if DYNAMIC_ARCH

#  define CONCAT_IMPL(a, b) a##b
#  define CONCAT(a, b) CONCAT_IMPL(a, b)

	static const int simd_level = static_cast<int>(ml::cpu::getSimdSupport());

	template<typename RetType, typename ... Args>
	std::function<RetType(Args...)> deduce_type(RetType (*)(Args...))
	{
		return std::function<RetType(Args...)>();
	}

#  define CREATE_TABLE(function_name) static decltype(deduce_type(SIMD_NAMESPACE::function_name)) CONCAT(function_name, _simd_dispatch_table)[16] = { \
		NAMESPACE_NO_SIMD::function_name, /*  0 - NONE (scalar code) */	\
		NAMESPACE_NO_SIMD::function_name, /*  1 - SSE     */	\
		NAMESPACE_SSE2::function_name, /*  2 - SSE2    */	\
		NAMESPACE_SSE2::function_name, /*  3 - SSE3    */	\
		NAMESPACE_SSE2::function_name, /*  4 - SSSE3   */	\
		NAMESPACE_SSE41::function_name, /*  5 - SSE41   */	\
		NAMESPACE_SSE41::function_name, /*  6 - SSE42   */	\
		NAMESPACE_AVX::function_name, /*  7 - AVX     */	\
		NAMESPACE_AVX2::function_name, /*  8 - AVX2    */	\
		nullptr, /*  9 - AVX512F */	\
		nullptr, /* 10 - */	\
		nullptr, /* 11 - */	\
		nullptr, /* 12 - */	\
		nullptr, /* 13 - */	\
		nullptr, /* 14 - */	\
		nullptr} /* 15 - */	\

#  define DISPATCH_AND_CALL(function_name) CONCAT(function_name, _simd_dispatch_table)[simd_level]
#else
#  define CREATE_TABLE(function_name)
#  define DISPATCH_AND_CALL(function_name) SIMD_NAMESPACE::function_name
#endif

}

namespace ml
{

	void cpu_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
		CREATE_TABLE(cpu_kernel_unpack_input);
		DISPATCH_AND_CALL(cpu_kernel_unpack_input)(context, shape, dst_dtype, dst, src);
	}
	void cpu_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		CREATE_TABLE(cpu_kernel_convert_type);
		DISPATCH_AND_CALL(cpu_kernel_convert_type)(context, dst, dst_dtype, src, src_dtype, elements);
	}
	void cpu_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
		CREATE_TABLE(cpu_kernel_transpose_021);
		DISPATCH_AND_CALL(cpu_kernel_transpose_021)(context, dtype, shape, input, output);
	}

	void cpu_winograd_weight_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *weights,
			void *matrices, bool invert, bool low_precision)
	{
		CREATE_TABLE(cpu_kernel_winograd_weight_transform);
		DISPATCH_AND_CALL(cpu_kernel_winograd_weight_transform)(context, tile_size, dtype, weight_shape, weights, matrices, invert, low_precision);
	}
	void cpu_winograd_input_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape,
			const void *input, void *matrices)
	{
		CREATE_TABLE(cpu_kernel_winograd_input_transform);
		DISPATCH_AND_CALL(cpu_kernel_winograd_input_transform)(context, tile_size, dtype, weight_shape, input_shape, input, matrices);
	}
	void cpu_winograd_output_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape,
			const void *matrices, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		CREATE_TABLE(cpu_kernel_winograd_output_transform);
		DISPATCH_AND_CALL(cpu_kernel_winograd_output_transform)(context, tile_size, dtype, weight_shape, output_shape, matrices, output, bias, add,
				act);
	}
	void cpu_winograd_gradient_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
			const void *gradient, void *matrices)
	{
		CREATE_TABLE(cpu_kernel_winograd_gradient_transform);
		DISPATCH_AND_CALL(cpu_kernel_winograd_gradient_transform)(context, tile_size, dtype, weight_shape, gradient_shape, gradient, matrices);
	}
	void cpu_winograd_update_transform(mlContext_t context, int tile_size, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices,
			void *update)
	{
		CREATE_TABLE(cpu_kernel_winograd_update_transform);
		DISPATCH_AND_CALL(cpu_kernel_winograd_update_transform)(context, tile_size, dtype, weight_shape, matrices, update);
	}

	void cpu_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
		CREATE_TABLE(cpu_kernel_convolution_implicit_gemm_forward);
		DISPATCH_AND_CALL(cpu_kernel_convolution_implicit_gemm_forward)(context, dtype, input_shape, weights_shape, input, weights, output, bias, add,
				act);
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
		CREATE_TABLE(cpu_kernel_add_bias_act);
		DISPATCH_AND_CALL(cpu_kernel_add_bias_act)(context, dtype, shape, input, bias, act);
	}

	void cpu_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input, mlActivationType_t act)
	{
		CREATE_TABLE(cpu_kernel_activation_forward);
		DISPATCH_AND_CALL(cpu_kernel_activation_forward)(context, dtype, shape, output, input, act);
	}
	void cpu_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		CREATE_TABLE(cpu_kernel_activation_backward);
		DISPATCH_AND_CALL(cpu_kernel_activation_backward)(context, shape, gradient_prev, gradient_next, output, act);
	}

} /* namespace avocado */

