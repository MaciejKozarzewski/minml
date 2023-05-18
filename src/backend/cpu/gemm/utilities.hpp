/*
 * utilities.hpp
 *
 *  Created on: May 11, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_GEMM_UTILITIES_HPP_
#define BACKEND_CPU_KERNELS_GEMM_UTILITIES_HPP_

#include <initializer_list>
#include <cassert>

namespace ml
{

	template<typename T, int N>
	class StackVector
	{
			T m_data[N];
			int m_size = 0;
		public:
			StackVector() noexcept = default;
			StackVector(std::initializer_list<T> list) :
					m_size(list.size())
			{
				for (int i = 0; i < m_size; i++)
					m_data[i] = list.begin()[i];
			}
			bool contains(const T &value) const noexcept
			{
				for (int i = 0; i < size(); i++)
					if (m_data[i] == value)
						return true;
				return false;
			}
			int find(const T &value) const noexcept
			{
				for (int i = 0; i < size(); i++)
					if (m_data[i] == value)
						return i;
				return -1;
			}
			void clear() noexcept
			{
				m_size = 0;
			}
			void add(const T &value) noexcept
			{
				assert(m_size < capacity());
				m_data[m_size++] = value;
			}
			template<int M>
			void add(const StackVector<T, M> &other) noexcept
			{
				assert(size() + M <= N);
				for (int i = 0; i < other.size(); i++)
					add(other[i]);
			}
			void remove(const T &value) noexcept
			{
				for (int i = 0; i < size(); i++)
					if (m_data[i] == value)
					{
						m_data[i] = m_data[--m_size]; // move the element to remove to the end and decrement size
						return;
					}
			}
			void remove(int index) noexcept
			{
				assert(0 <= index && index < m_size);
				m_data[index] = m_data[--m_size]; // move the element to remove to the end and decrement size
			}
			int size() const noexcept
			{
				return m_size;
			}
			int capacity() const noexcept
			{
				return N;
			}
			const T& operator[](int index) const noexcept
			{
				assert(0 <= index && index < size());
				return m_data[index];
			}
			T& operator[](int index) noexcept
			{
				assert(0 <= index && index < size());
				return m_data[index];
			}
			T* begin() noexcept
			{
				return m_data;
			}
			T* end() noexcept
			{
				return begin() + size();
			}
			const T* begin() const noexcept
			{
				return m_data;
			}
			const T* end() const noexcept
			{
				return begin() + size();
			}
	};

	enum class MatrixOp
	{
		NORMAL,
		TRANSPOSE
	};

	struct Size2D
	{
			int rows = 0;
			int columns = 0;
			Size2D() noexcept = default;
			Size2D(int r, int c) noexcept :
					rows(r),
					columns(c)
			{
			}
			friend bool operator==(const Size2D &lhs, const Size2D &rhs) noexcept
			{
				return lhs.rows == rhs.rows and lhs.columns == rhs.columns;
			}
			friend bool operator!=(const Size2D &lhs, const Size2D &rhs) noexcept
			{
				return not (lhs == rhs);
			}
	};

	struct Position2D
	{
			int row = 0;
			int column = 0;
			Position2D() noexcept = default;
			Position2D(int r, int c) noexcept :
					row(r),
					column(c)
			{
			}
	};
	struct Position4D
	{
			int n = 0;
			int h = 0;
			int w = 0;
			int c = 0;
			Position4D() noexcept = default;
			Position4D(int _n, int _h, int _w, int _c) noexcept :
					n(_n),
					h(_h),
					w(_w),
					c(_c)
			{
			}
	};

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_GEMM_UTILITIES_HPP_ */
