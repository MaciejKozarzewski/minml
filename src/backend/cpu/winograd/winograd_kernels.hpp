/*
 * winograd_kernels.hpp
 *
 *  Created on: Jun 26, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_WINOGRAD_WINOGRAD_KERNELS_HPP_
#define BACKEND_CPU_WINOGRAD_WINOGRAD_KERNELS_HPP_

namespace ml
{
	/*
	 * scalar kernels
	 */

	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP32
	 */
	void winograd_weight_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_input_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu);
	void winograd_gradient_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_update_transform_4x4_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters);
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP16
	 */
	void winograd_weight_transform_4x4_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_input_transform_4x4_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_4x4_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu);
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_weight_transform_5x5_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_input_transform_5x5_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_5x5_3x3_def_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu);
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP16
	 */
	void winograd_weight_transform_5x5_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_input_transform_5x5_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_5x5_3x3_def_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu);
	/*
	 * Transforms for 5x5 kernel and 2x2 tile size in FP32
	 */
	void winograd_weight_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_input_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu);
	void winograd_gradient_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_update_transform_2x2_5x5_def_fp32(const void *src[], void *dst[], void *workspace, int filters);

	/*
	 * SSE2 kernels
	 */

	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP32
	 */
	void winograd_input_transform_4x4_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_4x4_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu);
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_input_transform_5x5_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_5x5_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu);

	/*
	 * AVX kernels
	 */

	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP32
	 */
	void winograd_input_transform_4x4_3x3_avx_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_4x4_3x3_avx_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu);
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP16
	 */
	void winograd_input_transform_4x4_3x3_avx_fp16(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_4x4_3x3_avx_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu);
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_input_transform_5x5_3x3_avx_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_5x5_3x3_avx_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu);
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP16
	 */
	void winograd_input_transform_5x5_3x3_avx_fp16(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_5x5_3x3_avx_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu);

	/*
	 * AVX2+FMA3 kernels
	 */

	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP32
	 */
	void winograd_input_transform_4x4_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_4x4_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu);
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP16
	 */
	void winograd_input_transform_4x4_3x3_avx2_fma_fp16(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_4x4_3x3_avx2_fma_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu);
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_input_transform_5x5_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_5x5_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu);
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP16
	 */
	void winograd_input_transform_5x5_3x3_avx2_fma_fp16(const void *src[], void *dst[], void *workspace, int filters);
	void winograd_output_transform_5x5_3x3_avx2_fma_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu);

} /* namespace ml */

#endif /* BACKEND_CPU_WINOGRAD_WINOGRAD_KERNELS_HPP_ */
