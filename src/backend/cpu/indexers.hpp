/*
 * indexers.hpp
 *
 *  Created on: Jan 7, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_INDEXERS_HPP_
#define BACKEND_CPU_INDEXERS_HPP_

#include <cassert>

template<int Rank>
class Indexer
{
	public:
		int last_dim() const noexcept
		{
			return 0;
		}
		int at() const noexcept
		{
			return 0;
		}
		constexpr int rank() const noexcept
		{
			return Rank;
		}
};

template<>
class Indexer<1>
{
	public:
		int length = 0;
	public:
		Indexer(int dim0) noexcept :
				length(dim0)
		{
		}
		int last_dim() const noexcept
		{
			return length;
		}
		int at(int x0) const noexcept
		{
			assert(0 <= x0 && x0 < length);
			return x0;
		}
		constexpr int rank() const noexcept
		{
			return 1;
		}
};

template<>
class Indexer<2>
{
	public:
		int stride0 = 0;
#ifndef NDEBUG
		int d0 = 0, d1 = 0;
#endif
	public:
		Indexer(int dim0, int dim1) noexcept :
				stride0(dim1)
		{
#ifndef NDEBUG
			d0 = dim0;
			d1 = dim1;
#endif
		}
		int last_dim() const noexcept
		{
			return stride0;
		}
		int at(int x0, int x1) const noexcept
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			return x0 * stride0 + x1;
		}
		constexpr int rank() const noexcept
		{
			return 2;
		}
};

template<>
class Indexer<3>
{
	public:
		int stride0 = 0, stride1 = 0;
#ifndef NDEBUG
		int d0 = 0, d1 = 0, d2 = 0;
#endif
	public:
		Indexer(int dim0, int dim1, int dim2) noexcept :
				stride0(dim1 * dim2),
				stride1(dim2)
		{
#ifndef NDEBUG
			d0 = dim0;
			d1 = dim1;
			d2 = dim2;
#endif
		}
		int last_dim() const noexcept
		{
			return stride1;
		}
		int at(int x0, int x1, int x2) const noexcept
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			assert(0 <= x2 && x2 < d2);
			return x0 * stride0 + x1 * stride1 + x2;
		}
		constexpr int rank() const noexcept
		{
			return 3;
		}
};

template<>
class Indexer<4>
{
	public:
		int stride0 = 0, stride1 = 0, stride2 = 0;
#ifndef NDEBUG
		int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#endif
	public:
		Indexer(int dim0, int dim1, int dim2, int dim3) noexcept :
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
		int last_dim() const noexcept
		{
			return stride2;
		}
		int at(int x0, int x1, int x2, int x3) const noexcept
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			assert(0 <= x2 && x2 < d2);
			assert(0 <= x3 && x3 < d3);
			return x0 * stride0 + x1 * stride1 + x2 * stride2 + x3;
		}
		constexpr int rank() const noexcept
		{
			return 4;
		}
};

template<>
class Indexer<5>
{
	public:
		int stride0 = 0, stride1 = 0, stride2 = 0, stride3 = 0;
#ifndef NDEBUG
		int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0;
#endif
	public:
		Indexer(int dim0, int dim1, int dim2, int dim3, int dim4) noexcept :
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
		int last_dim() const noexcept
		{
			return stride3;
		}
		int at(int x0, int x1, int x2, int x3, int x4) const noexcept
		{
			assert(0 <= x0 && x0 < d0);
			assert(0 <= x1 && x1 < d1);
			assert(0 <= x2 && x2 < d2);
			assert(0 <= x3 && x3 < d3);
			assert(0 <= x4 && x4 < d4);
			return x0 * stride0 + x1 * stride1 + x2 * stride2 + x3 * stride3 + x4;
		}
		constexpr int rank() const noexcept
		{
			return 5;
		}
};

#endif /* BACKEND_CPU_INDEXERS_HPP_ */
