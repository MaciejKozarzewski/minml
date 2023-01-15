/*
 * accumulators.cuh
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_HELPERS_ACCUMULATORS_CUH_
#define BACKEND_CUDA_HELPERS_ACCUMULATORS_CUH_

#include "lines_and_tiles.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

template<typename T>
__device__ void mul_add(Line<T, 4> &acc, const T &lhs, const Line<T, 4> &rhs)
{
	acc.x0 += lhs * rhs.x0;
	acc.x1 += lhs * rhs.x1;
	acc.x2 += lhs * rhs.x2;
	acc.x3 += lhs * rhs.x3;
}
template<typename T>
__device__ void mul_add(Line<T, 8> &acc, const T &lhs, const Line<T, 8> &rhs)
{
	acc.x0 = __fmaf_rn(lhs, rhs.x0, acc.x0);
	acc.x1 = __fmaf_rn(lhs, rhs.x1, acc.x1);
	acc.x2 = __fmaf_rn(lhs, rhs.x2, acc.x2);
	acc.x3 = __fmaf_rn(lhs, rhs.x3, acc.x3);
	acc.x4 = __fmaf_rn(lhs, rhs.x4, acc.x4);
	acc.x5 = __fmaf_rn(lhs, rhs.x5, acc.x5);
	acc.x6 = __fmaf_rn(lhs, rhs.x6, acc.x6);
	acc.x7 = __fmaf_rn(lhs, rhs.x7, acc.x7);
//	acc.x0 += lhs * rhs.x0;
//	acc.x1 += lhs * rhs.x1;
//	acc.x2 += lhs * rhs.x2;
//	acc.x3 += lhs * rhs.x3;
//	acc.x4 += lhs * rhs.x4;
//	acc.x5 += lhs * rhs.x5;
//	acc.x6 += lhs * rhs.x6;
//	acc.x7 += lhs * rhs.x7;
}

template<typename T>
__device__ void outer_product(Tile<T, 4, 4> &acc, const Line<T, 4> &lhs, const Line<T, 4> &rhs)
{
	mul_add(acc.x0, lhs.x0, rhs);
	mul_add(acc.x1, lhs.x1, rhs);
	mul_add(acc.x2, lhs.x2, rhs);
	mul_add(acc.x3, lhs.x3, rhs);
}
template<typename T>
__device__ void outer_product(Tile<T, 4, 8> &acc, const Line<T, 4> &lhs, const Line<T, 8> &rhs)
{
	mul_add(acc.x0, lhs.x0, rhs);
	mul_add(acc.x1, lhs.x1, rhs);
	mul_add(acc.x2, lhs.x2, rhs);
	mul_add(acc.x3, lhs.x3, rhs);
}
template<typename T>
__device__ void outer_product(Tile<T, 8, 4> &acc, const Line<T, 8> &lhs, const Line<T, 4> &rhs)
{
	mul_add(acc.x0, lhs.x0, rhs);
	mul_add(acc.x1, lhs.x1, rhs);
	mul_add(acc.x2, lhs.x2, rhs);
	mul_add(acc.x3, lhs.x3, rhs);
	mul_add(acc.x4, lhs.x4, rhs);
	mul_add(acc.x5, lhs.x5, rhs);
	mul_add(acc.x6, lhs.x6, rhs);
	mul_add(acc.x7, lhs.x7, rhs);
}
template<typename T>
__device__ void outer_product(Tile<T, 8, 8> &acc, const Line<T, 8> &lhs, const Line<T, 8> &rhs)
{
	mul_add(acc.x0, lhs.x0, rhs);
	mul_add(acc.x1, lhs.x1, rhs);
	mul_add(acc.x2, lhs.x2, rhs);
	mul_add(acc.x3, lhs.x3, rhs);
	mul_add(acc.x4, lhs.x4, rhs);
	mul_add(acc.x5, lhs.x5, rhs);
	mul_add(acc.x6, lhs.x6, rhs);
	mul_add(acc.x7, lhs.x7, rhs);
}

#endif /* BACKEND_CUDA_HELPERS_ACCUMULATORS_CUH_ */
