/*
 * indexers.cuh
 *
 *  Created on: Jan 7, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_HELPERS_INDEXERS_CUH_
#define BACKEND_CUDA_HELPERS_INDEXERS_CUH_

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cassert>

template<int Rank>
class Indexer
{
	public:
		__host__ __device__ Indexer()
		{
		}
		__host__ __device__ int last_dim() const
		{
			return 0;
		}
		__host__ __device__ int at() const
		{
			return 0;
		}
		__host__ __device__ constexpr int rank() const
		{
			return Rank;
		}
};

template<>
class Indexer<1>
{
	private:
		int length;
	public:
		__host__ __device__ Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		__host__ __device__ Indexer(int dim0) :
				length(dim0)
		{
		}
		__host__ __device__ int last_dim() const
		{
			return length;
		}
		__host__ __device__ int at(int x0) const
		{
			assert(0 <= x0 && x0 < length);
			return x0;
		}
		__host__ __device__ constexpr int rank() const
		{
			return 1;
		}
};

template<>
class Indexer<2>
{
	private:
		int stride0;
#ifndef NDEBUG
		int d0, d1;
#endif
	public:
		__host__ __device__ Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		__host__ __device__ Indexer(int dim0, int dim1) :
				stride0(dim1)
		{
#ifndef NDEBUG
			d0 = dim0;
			d1 = dim1;
#endif
		}
		__host__ __device__ int last_dim() const
		{
			return stride0;
		}
		__host__ __device__ int at(int x0, int x1) const
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			return x0 * stride0 + x1;
		}
		__host__ __device__ constexpr int rank() const
		{
			return 2;
		}
};

template<>
class Indexer<3>
{
	private:
		int stride0, stride1;
#ifndef NDEBUG
		int d0, d1, d2;
#endif
	public:
		__host__ __device__ Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		__host__ __device__ Indexer(int dim0, int dim1, int dim2) :
				stride0(dim1 * dim2),
				stride1(dim2)
		{
#ifndef NDEBUG
			d0 = dim0;
			d1 = dim1;
			d2 = dim2;
#endif
		}
		__host__ __device__ int last_dim() const
		{
			return stride1;
		}
		__host__ __device__ int at(int x0, int x1, int x2) const
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			assert(0 <= x2 && x2 < d2);
			return x0 * stride0 + x1 * stride1 + x2;
		}
		__host__ __device__ constexpr int rank() const
		{
			return 3;
		}
};

template<>
class Indexer<4>
{
	private:
		int stride0, stride1, stride2;
#ifndef NDEBUG
		int d0, d1, d2, d3;
#endif
	public:
		__host__ __device__ Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		__host__ __device__ Indexer(int dim0, int dim1, int dim2, int dim3) :
				stride0(dim1 * dim2 * dim3),
				stride1(dim2 * dim3),
				stride2(dim3)
		{
#ifndef NDEBUG
			d0 = dim0;
			d1 = dim1;
			d2 = dim2;
			d3 = dim3;
#endif
		}
		__host__ __device__ int last_dim() const
		{
			return stride2;
		}
		__host__ __device__ int at(int x0, int x1, int x2, int x3) const
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			assert(0 <= x2 && x2 < d2);
			assert(0 <= x3 && x3 < d3);
			return x0 * stride0 + x1 * stride1 + x2 * stride2 + x3;
		}
		__host__ __device__ constexpr int rank() const
		{
			return 4;
		}
};

template<>
class Indexer<5>
{
	private:
		int stride0, stride1, stride2, stride3;
#ifndef NDEBUG
		int d0, d1, d2, d3, d4;
#endif
	public:
		__host__ __device__ Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		__host__ __device__ Indexer(int dim0, int dim1, int dim2, int dim3, int dim4) :
				stride0(dim1 * dim2 * dim3 * dim4),
				stride1(dim2 * dim3 * dim4),
				stride2(dim3 * dim4),
				stride3(dim4)
		{
#ifndef NDEBUG
			d0 = dim0;
			d1 = dim1;
			d2 = dim2;
			d3 = dim3;
			d4 = dim4;
#endif
		}
		__host__ __device__ int last_dim() const
		{
			return stride3;
		}
		__host__ __device__ int at(int x0, int x1, int x2, int x3, int x4) const
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			assert(0 <= x2 && x2 < d2);
			assert(0 <= x3 && x3 < d3);
			assert(0 <= x4 && x4 < d4);
			return x0 * stride0 + x1 * stride1 + x2 * stride2 + x3 * stride3 + x4;
		}
		__host__ __device__ constexpr int rank() const
		{
			return 5;
		}
};

template<>
class Indexer<6>
{
	private:
		int stride0, stride1, stride2, stride3, stride4;
#ifndef NDEBUG
		int d0, d1, d2, d3, d4, d5;
#endif
	public:
		__host__ __device__ Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		__host__ __device__ Indexer(int dim0, int dim1, int dim2, int dim3, int dim4, int dim5) :
				stride0(dim1 * dim2 * dim3 * dim4 * dim5),
				stride1(dim2 * dim3 * dim4 * dim5),
				stride2(dim3 * dim4 * dim5),
				stride3(dim4 * dim5),
				stride4(dim5)
		{
#ifndef NDEBUG
			d0 = dim0;
			d1 = dim1;
			d2 = dim2;
			d3 = dim3;
			d4 = dim4;
			d5 = dim5;
#endif
		}
		__host__ __device__ int last_dim() const
		{
			return stride4;
		}
		__host__ __device__ int at(int x0, int x1, int x2, int x3, int x4, int x5) const
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			assert(0 <= x2 && x2 < d2);
			assert(0 <= x3 && x3 < d3);
			assert(0 <= x4 && x4 < d4);
			assert(0 <= x5 && x5 < d5);
			return x0 * stride0 + x1 * stride1 + x2 * stride2 + x3 * stride3 + x4 * stride4 + x5;
		}
		__host__ __device__ constexpr int rank() const
		{
			return 6;
		}
};

#endif /* BACKEND_CUDA_HELPERS_INDEXERS_CUH_ */
