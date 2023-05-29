/*
 * indexers.hpp
 *
 *  Created on: Jan 7, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_HELPERS_INDEXERS_HPP_
#define BACKEND_CPU_HELPERS_INDEXERS_HPP_

#include <cassert>

template<int Rank>
class Indexer
{
	public:
		Indexer()
		{
		}
		int last_dim() const
		{
			return 0;
		}
		int at() const
		{
			return 0;
		}
		constexpr int rank() const
		{
			return Rank;
		}
};

template<>
class Indexer<1>
{
	public:
		int length;
	public:
		Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		Indexer(int dim0) :
				length(dim0)
		{
		}
		int last_dim() const
		{
			return length;
		}
		int at(int x0) const
		{
			assert(0 <= x0 && x0 < length);
			return x0;
		}
		constexpr int rank() const
		{
			return 1;
		}
};

template<>
class Indexer<2>
{
	public:
		int stride0;
#ifndef NDEBUG
		int d0, d1;
#endif
	public:
		Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		Indexer(int dim0, int dim1) :
				stride0(dim1)
		{
#ifndef NDEBUG
			d0 = dim0;
			d1 = dim1;
#endif
		}
		int last_dim() const
		{
			return stride0;
		}
		int at(int x0, int x1) const
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			return x0 * stride0 + x1;
		}
		constexpr int rank() const
		{
			return 2;
		}
};

template<>
class Indexer<3>
{
	public:
		int stride0, stride1;
#ifndef NDEBUG
		int d0, d1, d2;
#endif
	public:
		Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		Indexer(int dim0, int dim1, int dim2) :
				stride0(dim1 * dim2),
				stride1(dim2)
		{
#ifndef NDEBUG
			d0 = dim0;
			d1 = dim1;
			d2 = dim2;
#endif
		}
		int last_dim() const
		{
			return stride1;
		}
		int at(int x0, int x1, int x2) const
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			assert(0 <= x2 && x2 < d2);
			return x0 * stride0 + x1 * stride1 + x2;
		}
		constexpr int rank() const
		{
			return 3;
		}
};

template<>
class Indexer<4>
{
	public:
		int stride0, stride1, stride2;
#ifndef NDEBUG
		int d0, d1, d2, d3;
#endif
	public:
		Indexer() // @suppress("Class members should be properly initialized")
		{
		}
		Indexer(int dim0, int dim1, int dim2, int dim3) :
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
		int last_dim() const
		{
			return stride2;
		}
		int at(int x0, int x1, int x2, int x3) const
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			assert(0 <= x2 && x2 < d2);
			assert(0 <= x3 && x3 < d3);
			return x0 * stride0 + x1 * stride1 + x2 * stride2 + x3;
		}
		constexpr int rank() const
		{
			return 4;
		}
};

#endif /* BACKEND_CPU_HELPERS_INDEXERS_HPP_ */
