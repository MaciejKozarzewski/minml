/*
 * predicate_vector.cuh
 *
 *  Created on: Feb 12, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_KERNELS_GEMM_PREDICATE_VECTOR_CUH_
#define BACKEND_CUDA_KERNELS_GEMM_PREDICATE_VECTOR_CUH_

#include "../vectors/vectors.cuh"
#include <cinttypes>

template<int N = 32>
class PredicateVector
{
		static constexpr int elements = (N + 31) / 32;
		static constexpr uint32_t ones = 0xFFFFFFFFu;
		static constexpr uint32_t zeros = 0x00000000u;

		uint32_t data[elements];

		HOST_DEVICE void computeStorageOffset(int &element_index, int &bit_index, int idx) const
		{
			assert(idx < N);
			element_index = idx / 32;
			bit_index = idx % 32;
		}
	public:
		class Iterator
		{
				PredicateVector &vec;
				int index;
			public:
				HOST_DEVICE Iterator(Iterator const &it) :
						vec(it.vec),
						index(it.index)
				{
				}
				HOST_DEVICE Iterator(PredicateVector &vec, int start = 0) :
						vec(vec),
						index(start)
				{
				}
				HOST_DEVICE Iterator& operator++()
				{
					++index;
					return *this;
				}
				HOST_DEVICE Iterator& operator+=(int offset)
				{
					index += offset;
					return *this;
				}
				HOST_DEVICE Iterator& operator--()
				{
					--index;
					return *this;
				}
				HOST_DEVICE Iterator& operator-=(int offset)
				{
					index -= offset;
					return *this;
				}
				HOST_DEVICE Iterator operator++(int)
				{
					Iterator ret(*this);
					ret.index++;
					return ret;
				}
				HOST_DEVICE Iterator operator--(int)
				{
					Iterator ret(*this);
					ret.index--;
					return ret;
				}
				HOST_DEVICE Iterator operator+(int offset)
				{
					Iterator ret(*this);
					ret.index += offset;
					return ret;
				}
				HOST_DEVICE Iterator operator-(int offset)
				{
					Iterator ret(*this);
					ret.index -= offset;
					return ret;
				}
				HOST_DEVICE bool operator==(Iterator const &it) const
				{
					return index == it.index;
				}
				HOST_DEVICE bool operator!=(Iterator const &it) const
				{
					return index != it.index;
				}
				HOST_DEVICE bool get()
				{
					return vec.at(index);
				}
				HOST_DEVICE void set(bool value = true)
				{
					vec.set(index, value);
				}
		};

		class ConstIterator
		{
				const PredicateVector &vec;
				int index;
			public:
				HOST_DEVICE ConstIterator(const ConstIterator &other) :
						vec(other.vec),
						index(other.index)
				{
				}
				HOST_DEVICE ConstIterator(const PredicateVector &vec, int start = 0) :
						vec(vec),
						index(start)
				{
				}
				HOST_DEVICE ConstIterator& operator++()
				{
					++index;
					return *this;
				}
				HOST_DEVICE ConstIterator& operator+=(int offset)
				{
					index += offset;
					return *this;
				}
				HOST_DEVICE ConstIterator& operator--()
				{
					--index;
					return *this;
				}
				HOST_DEVICE ConstIterator& operator-=(int offset)
				{
					index -= offset;
					return *this;
				}
				HOST_DEVICE ConstIterator operator++(int)
				{
					ConstIterator ret(*this);
					ret.index++;
					return ret;
				}
				HOST_DEVICE ConstIterator operator--(int)
				{
					ConstIterator ret(*this);
					ret.index--;
					return ret;
				}
				HOST_DEVICE ConstIterator operator+(int offset)
				{
					ConstIterator ret(*this);
					ret.index += offset;
					return ret;
				}
				HOST_DEVICE ConstIterator operator-(int offset)
				{
					ConstIterator ret(*this);
					ret.index -= offset;
					return ret;
				}
				HOST_DEVICE bool operator==(ConstIterator const &it) const
				{
					return index == it.index;
				}
				HOST_DEVICE bool operator!=(ConstIterator const &it) const
				{
					return index != it.index;
				}
				HOST_DEVICE bool get()
				{
					return vec.at(index);
				}
		};

	public:
		HOST_DEVICE PredicateVector(bool value = true)
		{
			fill(value);
		}

		HOST_DEVICE void fill(bool value = true)
		{
			const uint32_t item = (value ? ones : zeros);
#pragma unroll
			for (int i = 0; i < elements; i++)
				data[i] = item;
		}
		HOST_DEVICE void clear()
		{
#pragma unroll
			for (int i = 0; i < elements; i++)
				data[i] = zeros;
		}
		HOST_DEVICE void enable()
		{
#pragma unroll
			for (int i = 0; i < elements; i++)
				data[i] = ones;
		}
		HOST_DEVICE bool operator[](int idx) const
		{
			return at(idx);
		}
		HOST_DEVICE bool at(int idx) const
		{
			int element_index, bit_index;
			computeStorageOffset(element_index, bit_index, idx);
			return (data[element_index] >> bit_index) & 1;
		}
		HOST_DEVICE void set(int idx, bool value = true)
		{
			int element_index, bit_index;
			computeStorageOffset(element_index, bit_index, idx);

			const uint32_t disable_mask = ~(1u << bit_index);
			const uint32_t enable_mask = (uint32_t) value << bit_index;

			data[element_index] = (data[element_index] & disable_mask) | enable_mask;
		}

		HOST_DEVICE PredicateVector& operator&=(const PredicateVector &other)
		{
#pragma unroll
			for (int i = 0; i < elements; ++i)
				data[i] &= other.data[i];
			return *this;
		}
		HOST_DEVICE PredicateVector& operator|=(const PredicateVector &other)
		{
#pragma unroll
			for (int i = 0; i < elements; ++i)
				data[i] |= other.data[i];
			return *this;
		}
		HOST_DEVICE bool is_zero() const
		{
			uint32_t result = zeros;
			for (int i = 0; i < elements; i++)
				result |= data[i];
			return result == zeros;
		}

		HOST_DEVICE Iterator begin()
		{
			return Iterator(*this);
		}
		HOST_DEVICE Iterator end()
		{
			return Iterator(*this, N);
		}
		HOST_DEVICE ConstIterator const_begin() const
		{
			return ConstIterator(*this);
		}
		HOST_DEVICE ConstIterator const_end() const
		{
			return ConstIterator(*this, N);
		}
};

#endif /* BACKEND_CUDA_KERNELS_GEMM_PREDICATE_VECTOR_CUH_ */
