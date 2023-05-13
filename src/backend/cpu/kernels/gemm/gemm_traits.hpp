/*
 * gemm_traits.hpp
 *
 *  Created on: May 11, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_GEMM_GEMM_TRAITS_HPP_
#define BACKEND_CPU_KERNELS_GEMM_GEMM_TRAITS_HPP_

#include <minml/backend/backend_types.h>
#include "Fragment.hpp"
#include "Matrix.hpp"
#include "utilities.hpp"

#include <functional>
#include <cassert>

namespace ml
{
	enum class Use
	{
		MATRIX_A,
		MATRIX_B,
		MATRIX_C,
		MATRIX_D
	};

	class Packing
	{
			using packing_function = std::function<void(Fragment &, const Matrix &, const Position2D &, MatrixOp )>;
			packing_function m_function;
			int m_min_width = 0;
			int m_max_width = std::numeric_limits<int>::max();
			mlDataType_t m_src_dtype = DTYPE_UNKNOWN;
			mlDataType_t m_dst_dtype = DTYPE_UNKNOWN;
		public:
			Packing() noexcept = default;

			bool can_pack(const Fragment &dst, const Matrix &src)
			{
				return m_min_width <= dst.stride() and dst.stride() <= m_max_width and dst.dtype() == m_dst_dtype and src.dtype() == m_src_dtype;
			}

			void operator()(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) const noexcept
			{
				assert(can_pack(dst, src));
				m_function(dst, src, src_pos, src_op);
			}
	};

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_GEMM_GEMM_TRAITS_HPP_ */
