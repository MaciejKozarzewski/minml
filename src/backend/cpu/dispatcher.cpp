/*
 * dispatcher.cpp
 *
 *  Created on: Jan 15, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "kernel_definitions.hpp"

namespace ml
{

	void cpu_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_unpack_input(context, shape, dst_dtype, dst, src);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_unpack_input(context, shape, dst_dtype, dst, src);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_unpack_input(context, shape, dst_dtype, dst, src);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_unpack_input(context, shape, dst_dtype, dst, src);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_unpack_input(context, shape, dst_dtype, dst, src);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_unpack_input(context, shape, dst_dtype, dst, src);
#endif
	}
	void cpu_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_convert_type(context, dst, dst_dtype, src, src_dtype, elements);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_convert_type(context, dst, dst_dtype, src, src_dtype, elements);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_convert_type(context, dst, dst_dtype, src, src_dtype, elements);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_convert_type(context, dst, dst_dtype, src, src_dtype, elements);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_convert_type(context, dst, dst_dtype, src, src_dtype, elements);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_convert_type(context, dst, dst_dtype, src, src_dtype, elements);
#endif
	}

	void cpu_winograd_weight_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, const void *weights, void *matrices,
			bool invert, bool low_precision)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_winograd_weight_transform(context, dtype, weight_shape, weights, matrices, invert, low_precision);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_winograd_weight_transform(context, dtype, weight_shape, weights, matrices, invert, low_precision);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_winograd_weight_transform(context, dtype, weight_shape, weights, matrices, invert, low_precision);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_winograd_weight_transform(context, dtype, weight_shape, weights, matrices, invert, low_precision);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_winograd_weight_transform(context, dtype, weight_shape, weights, matrices, invert, low_precision);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_winograd_weight_transform(context, dtype, weight_shape, weights, matrices, invert, low_precision);
#endif
	}
	void cpu_winograd_input_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t input_shape, const void *input,
			void *matrices)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_winograd_input_transform(context, dtype, weight_shape, input_shape, input, matrices);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_winograd_input_transform(context, dtype, weight_shape, input_shape, input, matrices);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_winograd_input_transform(context, dtype, weight_shape, input_shape, input, matrices);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_winograd_input_transform(context, dtype, weight_shape, input_shape, input, matrices);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_winograd_input_transform(context, dtype, weight_shape, input_shape, input, matrices);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_winograd_input_transform(context, dtype, weight_shape, input_shape, input, matrices);
#endif
	}
	void cpu_winograd_output_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t output_shape, const void *matrices,
			void *output, const void *bias, const void *add, mlActivationType_t act)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_winograd_output_transform(context, dtype, weight_shape, output_shape, matrices, output, bias, add, act);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_winograd_output_transform(context, dtype, weight_shape, output_shape, matrices, output, bias, add, act);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_winograd_output_transform(context, dtype, weight_shape, output_shape, matrices, output, bias, add, act);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_winograd_output_transform(context, dtype, weight_shape, output_shape, matrices, output, bias, add, act);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_winograd_output_transform(context, dtype, weight_shape, output_shape, matrices, output, bias, add, act);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_winograd_output_transform(context, dtype, weight_shape, output_shape, matrices, output, bias, add, act);
#endif
	}
	void cpu_winograd_gradient_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, mlShape_t gradient_shape,
			const void *gradient, void *matrices)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_winograd_gradient_transform(context, dtype, weight_shape, gradient_shape, gradient, matrices);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_winograd_gradient_transform(context, dtype, weight_shape, gradient_shape, gradient, matrices);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_winograd_gradient_transform(context, dtype, weight_shape, gradient_shape, gradient, matrices);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_winograd_gradient_transform(context, dtype, weight_shape, gradient_shape, gradient, matrices);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_winograd_gradient_transform(context, dtype, weight_shape, gradient_shape, gradient, matrices);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_winograd_gradient_transform(context, dtype, weight_shape, gradient_shape, gradient, matrices);
#endif
	}
	void cpu_winograd_update_transform(mlContext_t context, mlDataType_t dtype, mlShape_t weight_shape, const void *matrices, void *update)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_winograd_update_transform(context, dtype, weight_shape, matrices, update);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_winograd_update_transform(context, dtype, weight_shape, matrices, update);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_winograd_update_transform(context, dtype, weight_shape, matrices, update);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_winograd_update_transform(context, dtype, weight_shape, matrices, update);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_winograd_update_transform(context, dtype, weight_shape, matrices, update);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_winograd_update_transform(context, dtype, weight_shape, matrices, update);
#endif
	}

	void cpu_convolution_implicit_gemm_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_convolution_implicit_gemm_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_convolution_implicit_gemm_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_convolution_implicit_gemm_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_convolution_implicit_gemm_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_convolution_implicit_gemm_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_convolution_implicit_gemm_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add,
				act);
#endif
	}

	void cpu_convolution_fused_winograd_forward(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, mlShape_t weights_shape,
			const void *input, const void *weights, void *output, const void *bias, const void *add, mlActivationType_t act)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_convolution_fused_winograd_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_convolution_fused_winograd_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_convolution_fused_winograd_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_convolution_fused_winograd_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_convolution_fused_winograd_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add, act);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_convolution_fused_winograd_forward(context, dtype, input_shape, weights_shape, input, weights, output, bias, add,
				act);
#endif
	}

	// implemented in 'global_pooling.cpp'
	void cpu_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output,
			void *max_indices)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_global_avg_and_max_pooling_forward(context, dtype, shape, input, output, max_indices);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_global_avg_and_max_pooling_forward(context, dtype, shape, input, output, max_indices);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_global_avg_and_max_pooling_forward(context, dtype, shape, input, output, max_indices);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_global_avg_and_max_pooling_forward(context, dtype, shape, input, output, max_indices);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_global_avg_and_max_pooling_forward(context, dtype, shape, input, output, max_indices);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_global_avg_and_max_pooling_forward(context, dtype, shape, input, output, max_indices);
#endif
	}
	void cpu_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
			const void *max_indices)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_global_avg_and_max_pooling_backward(context, shape, gradient_prev, gradient_next, max_indices);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_global_avg_and_max_pooling_backward(context, shape, gradient_prev, gradient_next, max_indices);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_global_avg_and_max_pooling_backward(context, shape, gradient_prev, gradient_next, max_indices);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_global_avg_and_max_pooling_backward(context, shape, gradient_prev, gradient_next, max_indices);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_global_avg_and_max_pooling_backward(context, shape, gradient_prev, gradient_next, max_indices);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_global_avg_and_max_pooling_backward(context, shape, gradient_prev, gradient_next, max_indices);
#endif
	}

//	void cpu_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
//			const void *B, char opA, char opB, float alpha, float beta)
//	{
//#if DYNAMIC_ARCH
//		switch (cpu::Context::getSimdLevel(context))
//		{
//			case cpu::SimdLevel::AVX2:
//				return ns_avx2::cpu_kernel_gemm(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//			case cpu::SimdLevel::AVX:
//				return ns_avx::cpu_kernel_gemm(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//			case cpu::SimdLevel::SSE41:
//				return ns_sse41::cpu_kernel_gemm(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//			case cpu::SimdLevel::SSE2:
//				return ns_sse2::cpu_kernel_gemm(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//			case cpu::SimdLevel::NONE:
//				return ns_none::cpu_kernel_gemm(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//		}
//#else
//		SIMD_NAMESPACE::cpu_kernel_gemm(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//#endif
//	}
//	void cpu_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A,
//			mlShape_t shape_B, const void *B, char opA, char opB, float alpha, float beta)
//	{
//#if DYNAMIC_ARCH
//		switch (cpu::Context::getSimdLevel(context))
//		{
//			case cpu::SimdLevel::AVX2:
//				return ns_avx2::cpu_kernel_gemm_batched(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//			case cpu::SimdLevel::AVX:
//				return ns_avx::cpu_kernel_gemm_batched(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//			case cpu::SimdLevel::SSE41:
//				return ns_sse41::cpu_kernel_gemm_batched(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//			case cpu::SimdLevel::SSE2:
//				return ns_sse2::cpu_kernel_gemm_batched(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//			case cpu::SimdLevel::NONE:
//				return ns_none::cpu_kernel_gemm_batched(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//		}
//#else
//		SIMD_NAMESPACE::cpu_kernel_gemm_batched(context, dtype, shape_C, C, shape_A, A, shape_B, B, opA, opB, alpha, beta);
//#endif
//	}

	void cpu_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias, mlActivationType_t act)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_add_bias_act(context, dtype, shape, input, bias, act);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_add_bias_act(context, dtype, shape, input, bias, act);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_add_bias_act(context, dtype, shape, input, bias, act);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_add_bias_act(context, dtype, shape, input, bias, act);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_add_bias_act(context, dtype, shape, input, bias, act);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_add_bias_act(context, dtype, shape, input, bias, act);
#endif
	}

	void cpu_activation_forward_in_place(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, mlActivationType_t act)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_activation_forward_in_place(context, dtype, shape, input, act);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_activation_forward_in_place(context, dtype, shape, input, act);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_activation_forward_in_place(context, dtype, shape, input, act);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_activation_forward_in_place(context, dtype, shape, input, act);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_activation_forward_in_place(context, dtype, shape, input, act);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_activation_forward_in_place(context, dtype, shape, input, act);
#endif
	}
	void cpu_activation_backward_in_place(mlContext_t context, mlShape_t shape, void *gradient, const void *output, mlActivationType_t act)
	{
#if DYNAMIC_ARCH
		switch (cpu::Context::getSimdLevel(context))
		{
			case cpu::SimdLevel::AVX2:
				return ns_avx2::cpu_kernel_activation_backward_in_place(context, shape, gradient, output, act);
			case cpu::SimdLevel::AVX:
				return ns_avx::cpu_kernel_activation_backward_in_place(context, shape, gradient, output, act);
			case cpu::SimdLevel::SSE41:
				return ns_sse41::cpu_kernel_activation_backward_in_place(context, shape, gradient, output, act);
			case cpu::SimdLevel::SSE2:
				return ns_sse2::cpu_kernel_activation_backward_in_place(context, shape, gradient, output, act);
			case cpu::SimdLevel::NONE:
				return ns_none::cpu_kernel_activation_backward_in_place(context, shape, gradient, output, act);
		}
#else
		SIMD_NAMESPACE::cpu_kernel_activation_backward_in_place(context, shape, gradient, output, act);
#endif
	}

} /* namespace avocado */

