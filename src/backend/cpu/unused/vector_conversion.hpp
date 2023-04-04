/*
 * vector_conversion.hpp
 *
 *  Created on: Jan 22, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_VECTORS_VECTOR_CONVERSION_HPP_
#define BACKEND_CPU_VECTORS_VECTOR_CONVERSION_HPP_

#include <cstring>

#include "vectors.hpp"
#include "types.hpp"

namespace SIMD_NAMESPACE
{
	namespace internal
	{
		template<typename DstT, typename SrcT>
		struct VectorConverterBase
		{
				static constexpr int length = std::min(Vector<DstT>::length, Vector<SrcT>::length); // number of elements that will be converted in one step
		};
	} /* namespace internal */

	template<typename DstT, typename SrcT>
	struct VectorConverter: public internal::VectorConverterBase<DstT, SrcT>
	{
			Vector<DstT> operator()(const Vector<SrcT> &x) const noexcept
			{
			}
	};

	template<typename T>
	struct VectorConverter<T, T> : public internal::VectorConverterBase<T, T>
	{
			Vector<T> operator()(const Vector<T> &x) const noexcept
			{
				return x;
			}
	};

	/*
	 * Conversion bfloat16 -> float
	 */
	template<>
	struct VectorConverter<float, bfloat16> : public internal::VectorConverterBase<float, bfloat16>
	{
			Vector<float> operator()(const Vector<bfloat16> &x) const noexcept
			{
				return static_cast<Vector<float>>(x);
			}
	};

	/*
	 * Conversion float16 -> float
	 */
	template<>
	struct VectorConverter<float, float16> : public internal::VectorConverterBase<float, float16>
	{
			Vector<float> operator()(const Vector<float16> &x) const noexcept
			{
				return static_cast<Vector<float>>(x);
			}
	};

	/*
	 * Conversion float16 -> bfloat16
	 */
	template<>
	struct VectorConverter<bfloat16, float16> : public internal::VectorConverterBase<bfloat16, float16>
	{
			Vector<bfloat16> operator()(const Vector<float16> &x) const noexcept
			{
				return Vector<bfloat16>(static_cast<Vector<float>>(x));
			}
	};

	/*
	 * Conversion float -> bfloat16
	 */
	template<>
	struct VectorConverter<bfloat16, float> : public internal::VectorConverterBase<bfloat16, float>
	{
			Vector<bfloat16> operator()(const Vector<float> &x) const noexcept
			{
				return Vector<bfloat16>(x);
			}
	};

	/*
	 * Conversion bfloat16 -> float16
	 */
	template<>
	struct VectorConverter<float16, bfloat16> : public internal::VectorConverterBase<float16, bfloat16>
	{
			Vector<float16> operator()(const Vector<bfloat16> &x) const noexcept
			{
				return Vector<float16>(static_cast<Vector<float>>(x));
			}
	};
	/*
	 * Conversion float -> float16
	 */
	template<>
	struct VectorConverter<float16, float> : public internal::VectorConverterBase<float16, float>
	{
			Vector<float16> operator()(const Vector<float> &x) const noexcept
			{
				return Vector<float16>(x);
			}
	};
}

#endif /* BACKEND_CPU_VECTORS_VECTOR_CONVERSION_HPP_ */
