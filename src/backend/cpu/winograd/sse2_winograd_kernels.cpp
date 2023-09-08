/*
 * sse2_winograd_kernels.cpp
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
	void winograd_input_transform_4x4_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
	}
	void winograd_output_transform_4x4_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
	}
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_input_transform_5x5_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
	}
	void winograd_output_transform_5x5_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
	}
	/*
	 * Transforms for 5x5 kernel and 2x2 tile size in FP32
	 */
	void winograd_input_transform_2x2_5x5_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
	}
	void winograd_output_transform_2x2_5x5_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
	}

} /* namespace ml */

