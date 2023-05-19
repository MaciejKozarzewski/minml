/*
 * Fragment.hpp
 *
 *  Created on: May 9, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_GEMM_FRAGMENT_HPP_
#define BACKEND_CPU_KERNELS_GEMM_FRAGMENT_HPP_

#include <minml/backend/backend_types.h>
#include "utilities.hpp"

#include <cassert>

namespace ml
{
	class Fragment
	{
			void *m_data = nullptr;
			Size2D m_size;
			int m_stride = 0;
			mlDataType_t m_dtype = DTYPE_UNKNOWN;
			bool m_is_packed = false;
		public:
			Fragment() noexcept = default;
			Fragment(mlDataType_t dtype, int stride) noexcept :
					m_stride(stride),
					m_dtype(dtype)
			{
			}
			Fragment(const void *ptr, mlDataType_t dtype, int stride) noexcept :
					Fragment(const_cast<void*>(ptr), dtype, stride)
			{
			}
			Fragment(void *ptr, mlDataType_t dtype, int stride) noexcept :
					m_data(ptr),
					m_stride(stride),
					m_dtype(dtype)
			{
				assert(ptr != nullptr);
			}
			Size2D size() const noexcept
			{
				return m_size;
			}
			int rows() const noexcept
			{
				return size().rows;
			}
			int columns() const noexcept
			{
				return size().columns;
			}
			int stride() const noexcept
			{
				return m_stride;
			}
			mlDataType_t dtype() const noexcept
			{
				return m_dtype;
			}
			bool is_packed() const noexcept
			{
				return m_is_packed;
			}
			template<typename T = void>
			const T* data() const noexcept
			{
				return reinterpret_cast<const T*>(m_data);
			}
			template<typename T = void>
			T* data() noexcept
			{
				return reinterpret_cast<T*>(m_data);
			}

			template<typename T>
			const T& at(int row, int col) const noexcept
			{
				return data<T>()[offset_at(row, col)];
			}
			template<typename T>
			T& at(int row, int col) noexcept
			{
				return data<T>()[offset_at(row, col)];
			}
			template<typename T>
			void setall(T value) noexcept
			{
				for (int i = 0; i < rows() * columns(); i++)
					data<T>()[i] = value;
			}

			int offset_at(int row, int col) const noexcept
			{
				assert(0 <= row && row < rows());
				assert(0 <= col && col < columns());
				return row * stride() + col;
			}

			void set_size(Size2D size, int stride) noexcept
			{
				m_is_packed = false;
				m_size = size;
				m_stride = stride;
			}
			void mark_as_packed_with_size(Size2D size) noexcept
			{
				m_size = size;
				m_is_packed = true;
			}
	};

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_GEMM_FRAGMENT_HPP_ */
