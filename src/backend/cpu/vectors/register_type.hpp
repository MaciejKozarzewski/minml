/*
 * RegisterType.hpp
 *
 *  Created on: Mar 31, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef VECTORS_REGISTER_TYPE_HPP_
#define VECTORS_REGISTER_TYPE_HPP_

#include "vector_macros.hpp"

typedef int RegisterType;

constexpr static RegisterType SCALAR = 0;
constexpr static RegisterType XMM = 1;
constexpr static RegisterType YMM = 2;
constexpr static RegisterType ZMM = 3;
constexpr static RegisterType TMM = 4;

#if SUPPORTS_AVX
constexpr static RegisterType AUTO = YMM;
#elif SUPPORTS_SSE2
constexpr static RegisterType AUTO = XMM;
#else
constexpr static RegisterType AUTO = SCALAR;
#endif

template<typename T, RegisterType RT>
static constexpr int vector_size() noexcept
{
	switch (RT)
	{
		case SCALAR:
			return 1;
		case XMM:
			return 128 / (sizeof(T) * 8);
		case YMM:
			return 256 / (sizeof(T) * 8);
		case ZMM:
			return 512 / (sizeof(T) * 8);
		case TMM:
			return 0; // TODO add support for AMX instructions
		default:
			return 0;
	}
}

#endif /* VECTORS_REGISTER_TYPE_HPP_ */
