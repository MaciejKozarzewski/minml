/*
 * tensor_wrappers.cuh
 *
 *  Created on: Jan 7, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_HELPERS_TENSOR_WRAPPERS_HPP_
#define BACKEND_CPU_HELPERS_TENSOR_WRAPPERS_HPP_

#include "../vectors/vectors.hpp"

#include <cassert>
#include "indexers.hpp"

template<typename T, class IndexerType>
SIMD_NAMESPACE::Vector<T> load_vector(const T *__restrict__ ptr, const IndexerType &indexer, int x0)
{
	return SIMD_NAMESPACE::Vector<T>(ptr + indexer.at(x0), indexer.last_dim() - x0);
}
template<typename T, class IndexerType>
SIMD_NAMESPACE::Vector<T> load_vector(const T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1)
{
	return SIMD_NAMESPACE::Vector<T>(ptr + indexer.at(x0, x1), indexer.last_dim() - x1);
}
template<typename T, class IndexerType>
SIMD_NAMESPACE::Vector<T> load_vector(const T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1, int x2)
{
	return SIMD_NAMESPACE::Vector<T>(ptr + indexer.at(x0, x1, x2), indexer.last_dim() - x2);
}
template<typename T, class IndexerType>
SIMD_NAMESPACE::Vector<T> load_vector(const T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1, int x2, int x3)
{
	return SIMD_NAMESPACE::Vector<T>(ptr + indexer.at(x0, x1, x2, x3), indexer.last_dim() - x3);
}

template<typename T, class IndexerType>
void store_vector(SIMD_NAMESPACE::Vector<T> value, T *__restrict__ ptr, const IndexerType &indexer, int x0)
{
	value.store(ptr + indexer.at(x0), indexer.last_dim() - x0);
}
template<typename T, class IndexerType>
void store_vector(SIMD_NAMESPACE::Vector<T> value, T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1)
{
	value.store(ptr + indexer.at(x0, x1), indexer.last_dim() - x1);
}
template<typename T, class IndexerType>
void store_vector(SIMD_NAMESPACE::Vector<T> value, T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1, int x2)
{
	value.store(ptr + indexer.at(x0, x1, x2), indexer.last_dim() - x2);
}
template<typename T, class IndexerType>
void store_vector(SIMD_NAMESPACE::Vector<T> value, T *__restrict__ ptr, const IndexerType &indexer, int x0, int x1, int x2, int x3)
{
	value.store(ptr + indexer.at(x0, x1, x2, x3), indexer.last_dim() - x3);
}

template<int Rank, typename T, class IndexerType = Indexer<Rank>>
class TensorWrapper
{
		T *__restrict__ ptr;
	public:
		IndexerType indexer;

		TensorWrapper() // @suppress("Class members should be properly initialized")
		{
		}
		TensorWrapper(T *__restrict__ ptr, IndexerType ind) :
				ptr(ptr),
				indexer(ind)
		{
			assert(ind.rank() == rank());
		}
		template<class ... Dims>
		TensorWrapper(T *__restrict__ ptr, Dims ... dims) :
				ptr(ptr),
				indexer(dims...)
		{
		}
		template<class ... Dims>
		SIMD_NAMESPACE::Vector<T> load(Dims ... dims) const
		{
			return load_vector<T>(ptr, indexer, dims...);
		}
		template<class ... Dims>
		void store(SIMD_NAMESPACE::Vector<T> value, Dims ... dims)
		{
			store_vector<T>(value, ptr, indexer, dims...);
		}
		T* data() noexcept
		{
			return ptr;
		}
		const T* data() const noexcept
		{
			return ptr;
		}
		constexpr int rank() const
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

		ConstTensorWrapper() // @suppress("Class members should be properly initialized")
		{
		}
		ConstTensorWrapper(T *__restrict__ ptr, IndexerType ind) :
				ptr(ptr),
				indexer(ind)
		{
			assert(ind.rank() == rank());
		}
		template<class ... Dims>
		ConstTensorWrapper(const T *__restrict__ ptr, Dims ... dims) :
				ptr(ptr),
				indexer(dims...)
		{
		}
		template<class ... Dims>
		SIMD_NAMESPACE::Vector<T> load(Dims ... dims) const
		{
			return load_vector<T>(ptr, indexer, dims...);
		}
		const T* data() const noexcept
		{
			return ptr;
		}
		constexpr int rank() const
		{
			return Rank;
		}
};

#endif /* BACKEND_CPU_HELPERS_TENSOR_WRAPPERS_HPP_ */
