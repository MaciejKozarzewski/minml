/*
 * gemm_traits.hpp
 *
 *  Created on: May 11, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_GEMM_GEMM_TRAITS_HPP_
#define BACKEND_CPU_KERNELS_GEMM_GEMM_TRAITS_HPP_

#include "utilities.hpp"

namespace ml
{
	enum class Use
	{
		MATRIX_A,
		MATRIX_B,
		MATRIX_C,
		MATRIX_D
	};

	struct PackingTraits
	{
//			StackVector<Size2D, 8> available_sizes;
	};

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_GEMM_GEMM_TRAITS_HPP_ */
