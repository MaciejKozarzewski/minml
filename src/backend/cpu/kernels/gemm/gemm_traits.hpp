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
#include <limits>
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
			Packing(packing_function f, int min_width, int max_width, mlDataType_t src_dtype, mlDataType_t dst_dtype) noexcept :
					m_function(f),
					m_min_width(min_width),
					m_max_width(max_width),
					m_src_dtype(src_dtype),
					m_dst_dtype(dst_dtype)
			{
			}
			bool can_work_with(const Fragment &dst, const Matrix &src) const noexcept
			{
				return m_min_width <= dst.stride() and dst.stride() <= m_max_width and dst.dtype() == m_dst_dtype and src.dtype() == m_src_dtype;
			}
			void operator()(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) const noexcept
			{
				assert(can_work_with(dst, src));
				m_function(dst, src, src_pos, src_op);
			}
	};
	class Unpacking
	{
			using unpacking_function = std::function<void( Matrix &, const Position2D &,const Fragment & )>;
			unpacking_function m_function;
			mlDataType_t m_src_dtype = DTYPE_UNKNOWN;
			mlDataType_t m_dst_dtype = DTYPE_UNKNOWN;
		public:
			Unpacking() noexcept = default;
			Unpacking(unpacking_function f, mlDataType_t src_dtype, mlDataType_t dst_dtype) noexcept :
					m_function(f),
					m_src_dtype(src_dtype),
					m_dst_dtype(dst_dtype)
			{
			}
			bool can_work_with(const Matrix &dst, const Fragment &src) const noexcept
			{
				return dst.dtype() == m_dst_dtype and src.dtype() == m_src_dtype;
			}
			void operator()(Matrix &dst, const Position2D &dst_pos, const Fragment &src) const noexcept
			{
				assert(can_work_with(dst, src));
				m_function(dst, dst_pos, src);
			}
	};

	class MicroKernel
	{
			using kernel_function = std::function<void(Fragment &, const void *, const Fragment &, const Fragment &, const void *,
					const Fragment &)>;
			kernel_function m_function;
			int m_A_width = 0;
			int m_B_width = 0;
			mlDataType_t m_D_dtype = DTYPE_UNKNOWN;
			mlDataType_t m_A_dtype = DTYPE_UNKNOWN;
			mlDataType_t m_B_dtype = DTYPE_UNKNOWN;
			mlDataType_t m_C_dtype = DTYPE_UNKNOWN;
		public:
			MicroKernel() noexcept = default;
			MicroKernel(kernel_function f, int A_width, int B_width, mlDataType_t D_dtype, mlDataType_t A_dtype, mlDataType_t B_dtype,
					mlDataType_t C_dtype) noexcept :
					m_function(f),
					m_A_width(A_width),
					m_B_width(B_width),
					m_D_dtype(D_dtype),
					m_A_dtype(A_dtype),
					m_B_dtype(B_dtype),
					m_C_dtype(C_dtype)
			{
			}
			bool can_work_with(const Fragment &D, const Fragment &A, const Fragment &B, const Fragment &C) const noexcept
			{
				return m_A_width == A.stride() and m_B_width == B.stride() and A.dtype() == m_A_dtype and B.dtype() == m_B_dtype
						and C.dtype() == m_C_dtype and D.dtype() == m_D_dtype;
			}
			void operator()(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
					const Fragment &C) const noexcept
			{
				assert(can_work_with(D, A, B, C));
				m_function(D, alpha_ptr, A, B, beta_ptr, C);
			}
	};

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_GEMM_GEMM_TRAITS_HPP_ */
