/*
 * predicate_vector.cuh
 *
 *  Created on: Feb 12, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_KERNELS_GEMM_PREDICATE_VECTOR_CUH_
#define BACKEND_CUDA_KERNELS_GEMM_PREDICATE_VECTOR_CUH_

#include <cinttypes>

template<int N = 32>
class PredicateVector
{
		static constexpr int bytes = (N + 7) / 8;
		static constexpr int elements = (bytes + (int) sizeof(uint32_t) - 1) / (int) sizeof(uint32_t);

		uint32_t data[elements];
	public:
		class iterator
		{

		};
};

#endif /* BACKEND_CUDA_KERNELS_GEMM_PREDICATE_VECTOR_CUH_ */
