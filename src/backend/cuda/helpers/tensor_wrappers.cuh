/*
 * tensor_wrappers.cuh
 *
 *  Created on: Jan 7, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_HELPERS_TENSOR_WRAPPERS_CUH_
#define BACKEND_CUDA_HELPERS_TENSOR_WRAPPERS_CUH_

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "indexers.cuh"

#include "../vectors/vectors.cuh"

#include <cassert>

template<typename T, class IndexerType>
HOST_DEVICE vectors::Vector<T> load_vector(const T *__restrict__ ptr, const IndexerType &indexer, int x0)
{
	return vectors::Vector<T>(ptr + indexer.at(x0), indexer.last_dim() - x0);
}
template<typename T, class IndexerType>
HOST_DEVICE vectors::Vector<T> load_vector(const T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1)
{
	return vectors::Vector<T>(ptr + indexer.at(x0, x1), indexer.last_dim() - x1);
}
template<typename T, class IndexerType>
HOST_DEVICE vectors::Vector<T> load_vector(const T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1, int x2)
{
	return vectors::Vector<T>(ptr + indexer.at(x0, x1, x2), indexer.last_dim() - x2);
}
template<typename T, class IndexerType>
HOST_DEVICE vectors::Vector<T> load_vector(const T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1, int x2, int x3)
{
	return vectors::Vector<T>(ptr + indexer.at(x0, x1, x2, x3), indexer.last_dim() - x3);
}

template<typename T, class IndexerType>
HOST_DEVICE void store_vector(vectors::Vector<T> value, T *__restrict__ ptr, const IndexerType &indexer, int x0)
{
	value.store(ptr + indexer.at(x0), indexer.last_dim() - x0);
}
template<typename T, class IndexerType>
HOST_DEVICE void store_vector(vectors::Vector<T> value, T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1)
{
	value.store(ptr + indexer.at(x0, x1), indexer.last_dim() - x1);
}
template<typename T, class IndexerType>
HOST_DEVICE void store_vector(vectors::Vector<T> value, T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1, int x2)
{
	value.store(ptr + indexer.at(x0, x1, x2), indexer.last_dim() - x2);
}
template<typename T, class IndexerType>
HOST_DEVICE void store_vector(vectors::Vector<T> value, T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1, int x2, int x3)
{
	value.store(ptr + indexer.at(x0, x1, x2, x3), indexer.last_dim() - x3);
}

template<int Rank, typename T, class IndexerType = Indexer<Rank>>
class TensorWrapper
{
		T *__restrict__ ptr;
	public:
		IndexerType indexer;

		HOST_DEVICE TensorWrapper() // @suppress("Class members should be properly initialized")
		{
		}
		HOST_DEVICE TensorWrapper(T *__restrict__ ptr, IndexerType ind) :
				ptr(ptr),
				indexer(ind)
		{
			assert(ind.rank() == rank());
		}
		template<class ... Dims>
		HOST_DEVICE TensorWrapper(T *__restrict__ ptr, Dims ... dims) :
				ptr(ptr),
				indexer(dims...)
		{
		}
		template<class ... Dims>
		HOST_DEVICE T* pointer_at(Dims ... dims)
		{
			return ptr + indexer.at(dims...);
		}
		template<class ... Dims>
		HOST_DEVICE const T* pointer_at(Dims ... dims) const
		{
			return ptr + indexer.at(dims...);
		}
		template<class ... Dims>
		HOST_DEVICE vectors::Vector<T> load(Dims ... dims) const
		{
			return load_vector<T>(ptr, indexer, dims...);
		}
		template<class ... Dims>
		HOST_DEVICE void store(vectors::Vector<T> value, Dims ... dims)
		{
			store_vector<T>(value, ptr, indexer, dims...);
		}
		HOST_DEVICE constexpr int rank() const
		{
			return Rank;
		}
};

template<int Rank, typename T, class IndexerType = Indexer<Rank>>
class ConstTensorWrapper
{
		const T *__restrict__ ptr;
	public:
		IndexerType indexer;

		HOST_DEVICE ConstTensorWrapper() // @suppress("Class members should be properly initialized")
		{
		}
		HOST_DEVICE ConstTensorWrapper(T *__restrict__ ptr, IndexerType ind) :
				ptr(ptr),
				indexer(ind)
		{
			assert(ind.rank() == rank());
		}
		template<class ... Dims>
		HOST_DEVICE ConstTensorWrapper(const T *__restrict__ ptr, Dims ... dims) :
				ptr(ptr),
				indexer(dims...)
		{
		}
		template<class ... Dims>
		HOST_DEVICE const T* pointer_at(Dims ... dims) const
		{
			return ptr + indexer.at(dims...);
		}
		template<class ... Dims>
		HOST_DEVICE vectors::Vector<T> load(Dims ... dims) const
		{
			return load_vector<T>(ptr, indexer, dims...);
		}
		HOST_DEVICE constexpr int rank() const
		{
			return Rank;
		}
};

#endif /* BACKEND_CUDA_HELPERS_TENSOR_WRAPPERS_CUH_ */
