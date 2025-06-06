/*
 * misc_kernels.hpp
 *
 *  Created on: Oct 20, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_MISC_KERNELS_HPP_
#define BACKEND_CPU_MISC_KERNELS_HPP_

#include <minml/backend/backend_types.h>

#include <cstddef>

namespace ml
{
	namespace cpu
	{
		/*
		 * Default kernels (non-vectorized)
		 */
		void def_kernel_convert_fp32_to_fp16(void *dst, const void *src, size_t elements);
		void def_kernel_convert_fp16_to_fp32(void *dst, const void *src, size_t elements);
		void def_kernel_convert_fp64_to_fp16(void *dst, const void *src, size_t elements);
		void def_kernel_convert_fp16_to_fp64(void *dst, const void *src, size_t elements);
		void def_kernel_convert_fp64_to_fp32(void *dst, const void *src, size_t elements);
		void def_kernel_convert_fp32_to_fp64(void *dst, const void *src, size_t elements);

		void def_kernel_softmax_3_channels_fp32(void *dst, const void *src, int first_dim);
		void def_kernel_softmax_fp32(void *dst, const void *src, int first_dim, int last_dim, void *workspace);
		void def_kernel_softmax_fp16(void *dst, const void *src, int first_dim, int last_dim, void *workspace);

		void def_kernel_activation_forward_fp32(void *dst, const void *src, size_t elements, mlActivationType_t activation);

		void def_kernel_activation_backward_fp32(void *gradient_prev, const void *gradient_next, const void *input, const void *output,
				size_t elements, mlActivationType_t activation);

		void def_kernel_add_bias_act_fp32(void *output, const void *input, const void *bias, int first_dim, int last_dim, mlActivationType_t act);

		/*
		 * AVX kernels
		 */
		void avx_kernel_convert_fp32_to_fp16(void *dst, const void *src, size_t elements);
		void avx_kernel_convert_fp16_to_fp32(void *dst, const void *src, size_t elements);

		void avx_kernel_softmax_3_channels_fp16(void *dst, const void *src, int first_dim);
		void avx_kernel_softmax_fp16(void *dst, const void *src, int first_dim, int last_dim, void *workspace);

		void avx_kernel_activation_forward_fp16(void *dst, const void *src, size_t elements, mlActivationType_t activation);

		void avx_kernel_add_bias_act_fp16(void *output, const void *input, const void *bias, int first_dim, int last_dim, mlActivationType_t act);

	} /* namespace cpu */
} /* namespace ml */

#endif /* BACKEND_CPU_MISC_KERNELS_HPP_ */
