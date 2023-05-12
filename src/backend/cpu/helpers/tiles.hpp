/*
 * tiles.hpp
 *
 *  Created on: May 9, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_HELPERS_TILES_HPP_
#define BACKEND_CPU_HELPERS_TILES_HPP_

#include "../vectors/vectors.hpp"

#include <cassert>

//template<typename T, RegisterType RT, int Rows, int Columns>
//class Tile
//{
//	public:
//};
//
//template<RegisterType RT, int Rows>
//class Tile<float, RT, Rows, vector_size<float, RT>()>
//{
//		SIMD_NAMESPACE::Vector<float, RT> data[Rows];
//	public:
//		void load(const float *ptr, int stride, int rows_to_load, int columns_to_load) noexcept
//		{
//			if (rows_to_load >= rows())
//			{
//				if (columns_to_load >= columns())
//				{
//					for (int i = 0; i < rows(); i++)
//						data[i].load(ptr + i * stride, columns());
//				}
//				else
//				{
//					for (int i = 0; i < rows(); i++)
//						data[i].load(ptr + i * stride, columns_to_load);
//				}
//			}
//			else
//			{
//				if (columns_to_load >= columns())
//				{
//					for (int i = 0; i < rows_to_load; i++)
//						data[i].load(ptr + i * stride, columns());
//				}
//				else
//				{
//					for (int i = 0; i < rows_to_load; i++)
//						data[i].load(ptr + i * stride, columns_to_load);
//				}
//			}
//		}
//		constexpr int rows() const noexcept
//		{
//			return Rows;
//		}
//		constexpr int columns() const noexcept
//		{
//			return vector_size<float, RT>();
//		}
//};

#endif /* BACKEND_CPU_HELPERS_TILES_HPP_ */
