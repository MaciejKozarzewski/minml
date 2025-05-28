/*
 * TensorFragment.hpp
 *
 *  Created on: Feb 10, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_TENSORFRAGMENT_HPP_
#define BACKEND_CPU_KERNELS_TENSORFRAGMENT_HPP_

#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include <cstdint>
#include <iostream>
#include <cassert>

namespace ml
{
	class TensorFragment
	{
			void *m_data = nullptr;
			int m_rows = 0;
			int m_columns = 0;
			int m_stride = 0;
			mlDataType_t m_dtype = DTYPE_UNKNOWN;
		public:
			TensorFragment() noexcept = default;
			TensorFragment(const void *ptr, mlDataType_t dtype, int rows, int columns, int stride) noexcept :
					TensorFragment(const_cast<void*>(ptr), dtype, rows, columns, stride)
			{
			}
			TensorFragment(void *ptr, mlDataType_t dtype, int rows, int columns, int stride) noexcept :
					m_data(ptr),
					m_rows(rows),
					m_columns(columns),
					m_stride(stride),
					m_dtype(dtype)
			{
				assert(ptr != nullptr);
			}
			int rows() const noexcept
			{
				return m_rows;
			}
			int columns() const noexcept
			{
				return m_columns;
			}
			int stride() const noexcept
			{
				return m_stride;
			}
			mlDataType_t dtype() const noexcept
			{
				return m_dtype;
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

			bool is_partial() const noexcept
			{
				assert(columns() <= stride());
				return columns() < stride();
			}
			bool is_fp16() const noexcept
			{
				return dtype() == DTYPE_FLOAT16;
			}
			bool is_fp32() const noexcept
			{
				return dtype() == DTYPE_FLOAT32;
			}
			bool is_fp64() const noexcept
			{
				return dtype() == DTYPE_FLOAT64;
			}
			uint64_t stride_in_bytes() const noexcept
			{
				return stride() * size_of(dtype());
			}

			void print_info() const
			{
				std::cout << "TensorFragment (" << m_data << ") of size (" << m_rows << " x " << m_columns << ") with stride " << m_stride << '\n';
			}
	};

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_TENSORFRAGMENT_HPP_ */
