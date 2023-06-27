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

	void winograd_input_transform_5x5_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters);

} /* namespace ml */

#endif /* BACKEND_CPU_WINOGRAD_WINOGRAD_KERNELS_HPP_ */
