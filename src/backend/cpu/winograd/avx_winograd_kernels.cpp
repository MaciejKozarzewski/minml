/*
 * avx_winograd_kernels.cpp
 *
 *  Created on: Jun 26, 2023
 *      Author: Maciej Kozarzewski
 */

#include "winograd_kernels.hpp"
#include "../assembly_macros.hpp"

namespace ml
{

	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP32
	 */
	void winograd_input_transform_4x4_3x3_avx_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
	}
	void winograd_output_transform_4x4_3x3_avx_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu)
	{
	}
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP16
	 */
	void winograd_input_transform_4x4_3x3_avx_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
	}
	void winograd_output_transform_4x4_3x3_avx_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu)
	{
	}
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_input_transform_5x5_3x3_avx_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
	}
	void winograd_output_transform_5x5_3x3_avx_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu)
	{
	}
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP16
	 */
	void winograd_input_transform_5x5_3x3_avx_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
	}
	void winograd_output_transform_5x5_3x3_avx_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[], const void *bias,
			bool use_relu)
	{
	}

} /* namespace ml */

