/*
 * lines_and_tiles.cuh
 *
 *  Created on: Jan 8, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_HELPERS_LINES_AND_TILES_CUH_
#define BACKEND_CUDA_HELPERS_LINES_AND_TILES_CUH_

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cassert>

namespace hidden_internal
{
	template<typename T, int N>
	struct Storage
	{
	};
	template<typename T>
	struct Storage<T, 1>
	{
			T x0;
	};
	template<typename T>
	struct __builtin_align__(2 * sizeof(T)) Storage<T, 2>
	{
			T x0, x1;
	};
	template<typename T>
	struct Storage<T, 3>
	{
			T x0, x1, x2;
	};
	template<typename T>
	struct __builtin_align__(4 * sizeof(T)) Storage<T, 4>
	{
			T x0, x1, x2, x3;
	};
	template<typename T>
	struct Storage<T, 5>
	{
			T x0, x1, x2, x3, x4;
	};
	template<typename T>
	struct Storage<T, 6>
	{
			T x0, x1, x2, x3, x4, x5;
	};
	template<typename T>
	struct Storage<T, 7>
	{
			T x0, x1, x2, x3, x4, x5, x6;
	};
	template<typename T>
	struct Storage<T, 9>
	{
			T x0, x1, x2, x3, x4, x5, x6, x7, x8;
	};
	template<typename T>
	struct Storage<T, 10>
	{
			T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;
	};
	template<typename T>
	struct __builtin_align__(4 * sizeof(T)) Storage<T, 8>
	{
			T x0, x1, x2, x3, x4, x5, x6, x7;
	};
	template<typename T>
	struct __builtin_align__(4 * sizeof(T)) Storage<T, 16>
	{
			T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
	};
}

template<typename T, int N>
struct Line: hidden_internal::Storage<T, N>
{
		__host__ __device__ constexpr int size() const
		{
			return N;
		}
		__host__ __device__ T& operator[](int index)
		{
			assert(0 <= index && index < size());
			return reinterpret_cast<T*>(this)[index];
		}
		__host__ __device__ T operator[](int index) const
		{
			assert(0 <= index && index < size());
			return reinterpret_cast<const T*>(this)[index];
		}
		__host__ __device__ void fill(T value)
		{
			for (int i = 0; i < size(); i++)
				reinterpret_cast<T*>(this)[i] = value;
		}
};

template<typename T, int Rows, int Cols>
struct Tile: hidden_internal::Storage<Line<T, Cols>, Rows>
{
		__host__ __device__ constexpr int rows() const
		{
			return Rows;
		}
		__host__ __device__ constexpr int columns() const
		{
			return Cols;
		}
		__host__ __device__ constexpr int size() const
		{
			return rows() * columns();
		}
		__host__ __device__ T& at(int row, int col)
		{
			return reinterpret_cast<T*>(this)[row * columns() + col];
		}
		__host__ __device__ T at(int row, int col) const
		{
			return reinterpret_cast<const T*>(this)[row * columns() + col];
		}
		__host__ __device__ Line<T, Cols>& get_row(int index)
		{
			assert(0 <= index && index < rows());
			return reinterpret_cast<Line<T, Cols>*>(this)[index];
		}
		__host__ __device__ Line<T, Cols> get_row(int index) const
		{
			assert(0 <= index && index < rows());
			return reinterpret_cast<const Line<T, Cols>*>(this)[index];
		}
		__host__ __device__ void fill(T value)
		{
			for (int i = 0; i < rows(); i++)
				get_row(i).fill(value);
		}
		__host__ __device__ T& operator[](int index)
		{
			assert(0 <= index && index < size());
			return reinterpret_cast<T*>(this)[index];
		}
		__host__ __device__ T operator[](int index) const
		{
			assert(0 <= index && index < size());
			return reinterpret_cast<const T*>(this)[index];
		}
};

#endif /* BACKEND_CUDA_HELPERS_LINES_AND_TILES_CUH_ */
