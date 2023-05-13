/*
 * Tile.hpp
 *
 *  Created on: May 9, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_TILES_TILE_HPP_
#define BACKEND_CPU_TILES_TILE_HPP_

#include "../vectors/register_type.hpp"
#include "../vectors/vectors.hpp"

namespace SIMD_NAMESPACE
{

	template<typename T, RegisterType RT, int Rows, int Columns>
	class Tile
	{
			Vector<T, RT> m_data[Rows * ((Columns + vector_size<T, RT>() - 1) / vector_size<T, RT>())];
		public:
			void zeroall() noexcept
			{
				for (int i = 0; i < rows(); i++)
					m_data[i] = Vector<T, RT>::zero();
			}
			void fill(const T &x) noexcept
			{
				for (int i = 0; i < rows(); i++)
					m_data[i] = x;
			}
			template<typename U>
			void load(const U *ptr, int stride, int rows_to_load = rows(), int columns_to_load = columns()) noexcept
			{
				if (rows_to_load >= rows())
				{
					if (columns_to_load >= columns())
					{
						for (int i = 0; i < rows(); i++)
							m_data[i].load(ptr + i * stride, columns());
					}
					else
					{
						for (int i = 0; i < rows(); i++)
							m_data[i].load(ptr + i * stride, columns_to_load);
					}
				}
				else
				{
					if (columns_to_load >= columns())
					{
						for (int i = 0; i < rows_to_load; i++)
							m_data[i].load(ptr + i * stride, columns());
					}
					else
					{
						for (int i = 0; i < rows_to_load; i++)
							m_data[i].load(ptr + i * stride, columns_to_load);
					}
				}
			}
			template<typename U>
			void store(U *ptr, int stride, int rows_to_store = rows(), int columns_to_store = columns()) const noexcept
			{
				if (rows_to_store >= rows())
				{
					if (columns_to_store >= columns())
					{
						for (int i = 0; i < rows(); i++)
							m_data[i].store(ptr + i * stride, columns());
					}
					else
					{
						for (int i = 0; i < rows(); i++)
							m_data[i].store(ptr + i * stride, columns_to_store);
					}
				}
				else
				{
					if (columns_to_store >= columns())
					{
						for (int i = 0; i < rows_to_store; i++)
							m_data[i].store(ptr + i * stride, columns());
					}
					else
					{
						for (int i = 0; i < rows_to_store; i++)
							m_data[i].store(ptr + i * stride, columns_to_store);
					}
				}
			}

			const Vector<T, RT>& operator[](int idx) const noexcept
			{
				return get_row(idx);
			}
			Vector<T, RT>& operator[](int idx) noexcept
			{
				return get_row(idx);
			}

			const Vector<T, RT>& get_row(int idx) const noexcept
			{
				assert(0 <= idx && idx < rows());
				return m_data[idx];
			}
			Vector<T, RT>& get_row(int idx) noexcept
			{
				assert(0 <= idx && idx < rows());
				return m_data[idx];
			}

			constexpr int rows() const noexcept
			{
				return Rows;
			}
			constexpr int columns() const noexcept
			{
				return vector_size<T, RT>();
			}
	};

} /* SIMD_NAMESPACE */

#endif /* BACKEND_CPU_TILES_TILE_HPP_ */
