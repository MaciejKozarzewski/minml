/*
 * Matrix.hpp
 *
 *  Created on: May 11, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_GEMM_MATRIX_HPP_
#define BACKEND_CPU_KERNELS_GEMM_MATRIX_HPP_

#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include <cinttypes>
#include <cassert>

namespace ml
{
	class Matrix
	{
			void *m_data = nullptr;
			mlDataType_t m_dtype = DTYPE_UNKNOWN;
			int m_rows = 0;
			int m_columns = 0;
			int m_stride = 0; // stride within the same column
		public:
			Matrix() noexcept = default;
			Matrix(const void *ptr, mlDataType_t dtype, int rows, int columns, int stride) noexcept :
					Matrix(const_cast<void*>(ptr), dtype, rows, columns, stride)
			{
			}
			Matrix(void *ptr, mlDataType_t dtype, int rows, int columns, int stride) noexcept :
					m_data(ptr),
					m_dtype(dtype),
					m_rows(rows),
					m_columns(columns),
					m_stride(stride)
			{
				assert(rows > 0);
				assert(columns > 0);
				assert(stride >= 0);
			}
			mlDataType_t dtype() const noexcept
			{
				return m_dtype;
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
			const void* data() const noexcept
			{
				return m_data;
			}
			void* data() noexcept
			{
				return m_data;
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
			/*
			 * Returns pointer to the element at given row and column
			 */
			const void* pointer_at(int row, int column) const noexcept
			{
				return reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(m_data) + offset_at(row, column));
			}
			void* pointer_at(int row, int column) noexcept
			{
				return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(m_data) + offset_at(row, column));
			}

			template<typename T>
			const T& at(int row, int column) const noexcept
			{
				return *reinterpret_cast<const T*>(pointer_at(row, column));
			}
			template<typename T>
			T& at(int row, int column) noexcept
			{
				return *reinterpret_cast<const T*>(pointer_at(row, column));
			}

			/*
			 * Returns index of an element at given row and column
			 */
			int index_at(int row, int column) const noexcept
			{
				assert((stride() == 0) or (0 <= row && row < rows()));
				assert(0 <= column && column < columns());
				return row * stride() + column;
			}
			/*
			 * Returns offset in bytes of an element at given row and column
			 */
			int offset_at(int row, int column) const noexcept
			{
				return size_of(dtype()) * index_at(row, column);
			}

			bool hasEqualShapeAs(const Matrix &other) const noexcept
			{
				return this->rows() == other.rows() and this->columns() == other.columns();
			}
	};

	class BatchedMatrix
	{
			void *m_data = nullptr;
			mlDataType_t m_dtype = DTYPE_UNKNOWN;
			int m_batch_size = 0;
			int m_rows = 0;
			int m_columns = 0;
			int m_stride = 0; // stride within the same column
		public:
			BatchedMatrix() noexcept = default;
			BatchedMatrix(const void *ptr, mlDataType_t dtype, int batch_size, int rows, int columns, int stride) noexcept :
					BatchedMatrix(const_cast<void*>(ptr), dtype, batch_size, rows, columns, stride)
			{
			}
			BatchedMatrix(void *ptr, mlDataType_t dtype, int batch_size, int rows, int columns, int stride) noexcept :
					m_data(ptr),
					m_dtype(dtype),
					m_batch_size(batch_size),
					m_rows(rows),
					m_columns(columns),
					m_stride(stride)
			{
				assert(batch_size > 0);
				assert(rows > 0);
				assert(columns > 0);
				assert(stride >= 0);
			}
			mlDataType_t dtype() const noexcept
			{
				return m_dtype;
			}
			int batch_size() const noexcept
			{
				return m_batch_size;
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
			const void* data() const noexcept
			{
				return m_data;
			}
			void* data() noexcept
			{
				return m_data;
			}
			/*
			 * Returns pointer to the element at given row and column
			 */
			const void* pointer_at(int batch, int row, int column) const noexcept
			{
				return reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(m_data) + offset_at(batch, row, column));
			}
			void* pointer_at(int batch, int row, int column) noexcept
			{
				return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(m_data) + offset_at(batch, row, column));
			}

			template<typename T>
			const T& at(int batch, int row, int column) const noexcept
			{
				return *reinterpret_cast<const T*>(pointer_at(batch, row, column));
			}
			template<typename T>
			T& at(int batch, int row, int column) noexcept
			{
				return *reinterpret_cast<const T*>(pointer_at(batch, row, column));
			}

			/*
			 * Returns index of an element at given row and column
			 */
			int index_at(int batch, int row, int column) const noexcept
			{
				assert((stride() == 0) or (0 <= row && row < rows()));
				assert(0 <= column && column < columns());
				return batch * rows() * stride() + row * stride() + column;
			}
			/*
			 * Returns offset in bytes of an element at given row and column
			 */
			int offset_at(int batch, int row, int column) const noexcept
			{
				return size_of(dtype()) * index_at(batch, row, column);
			}

			bool hasEqualShapeAs(const BatchedMatrix &other) const noexcept
			{
				return this->batch_size() == other.batch_size() and this->rows() == other.rows() and this->columns() == other.columns();
			}
	};

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_GEMM_MATRIX_HPP_ */
