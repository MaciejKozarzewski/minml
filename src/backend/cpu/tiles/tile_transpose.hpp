/*
 * vector_transpose.hpp
 *
 *  Created on: May 9, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_VECTORS_VECTOR_TRANSPOSE_HPP_
#define BACKEND_CPU_VECTORS_VECTOR_TRANSPOSE_HPP_

#include "Tile.hpp"

#include <array>
#include <x86intrin.h>

namespace SIMD_NAMESPACE
{

	template<typename T, RegisterType RT, int Rows, int Columns>
	struct Transpose;

#if COMPILED_WITH_SSE2
	template<>
	struct Transpose<float, XMM, 4, 4>
	{
			Tile<float, XMM, 4, 4> operator()(Tile<float, XMM, 4, 4> &x) const noexcept
			{
				Tile<float, XMM, 4, 4> tmp;
				tmp[0] = _mm_unpacklo_ps(x[0], x[1]);
				tmp[1] = _mm_unpackhi_ps(x[0], x[1]);
				tmp[2] = _mm_unpacklo_ps(x[2], x[3]);
				tmp[3] = _mm_unpackhi_ps(x[2], x[3]);

				x[0] = _mm_movelh_ps(tmp[0], tmp[2]);
				x[1] = _mm_movehl_ps(tmp[2], tmp[0]);
				x[2] = _mm_movelh_ps(tmp[1], tmp[3]);
				x[3] = _mm_movehl_ps(tmp[3], tmp[1]);

				return x;
			}
	};
#endif

#if COMPILED_WITH_AVX
	template<>
	struct Transpose<float, YMM, 8, 8>
	{
			Tile<float, YMM, 8, 8> operator()(Tile<float, YMM, 8, 8> &x) const noexcept
			{
				Tile<float, YMM, 8, 8> tmp;
				tmp[0] = _mm256_unpacklo_ps(x[0], x[1]);
				tmp[1] = _mm256_unpackhi_ps(x[0], x[1]);
				tmp[2] = _mm256_unpacklo_ps(x[2], x[3]);
				tmp[3] = _mm256_unpackhi_ps(x[2], x[3]);
				tmp[4] = _mm256_unpacklo_ps(x[4], x[5]);
				tmp[5] = _mm256_unpackhi_ps(x[4], x[5]);
				tmp[6] = _mm256_unpacklo_ps(x[6], x[7]);
				tmp[7] = _mm256_unpackhi_ps(x[6], x[7]);

				x[0] = _mm256_shuffle_ps(tmp[0], tmp[2], _MM_SHUFFLE(1,0,1,0));
				x[1] = _mm256_shuffle_ps(tmp[0], tmp[2], _MM_SHUFFLE(3,2,3,2));
				x[2] = _mm256_shuffle_ps(tmp[1], tmp[3], _MM_SHUFFLE(1,0,1,0));
				x[3] = _mm256_shuffle_ps(tmp[1], tmp[3], _MM_SHUFFLE(3,2,3,2));
				x[4] = _mm256_shuffle_ps(tmp[4], tmp[6], _MM_SHUFFLE(1,0,1,0));
				x[5] = _mm256_shuffle_ps(tmp[4], tmp[6], _MM_SHUFFLE(3,2,3,2));
				x[6] = _mm256_shuffle_ps(tmp[5], tmp[7], _MM_SHUFFLE(1,0,1,0));
				x[7] = _mm256_shuffle_ps(tmp[5], tmp[7], _MM_SHUFFLE(3,2,3,2));

				tmp[0] = _mm256_permute2f128_ps(x[0], x[4], 0x20);
				tmp[1] = _mm256_permute2f128_ps(x[1], x[5], 0x20);
				tmp[2] = _mm256_permute2f128_ps(x[2], x[6], 0x20);
				tmp[3] = _mm256_permute2f128_ps(x[3], x[7], 0x20);
				tmp[4] = _mm256_permute2f128_ps(x[0], x[4], 0x31);
				tmp[5] = _mm256_permute2f128_ps(x[1], x[5], 0x31);
				tmp[6] = _mm256_permute2f128_ps(x[2], x[6], 0x31);
				tmp[7] = _mm256_permute2f128_ps(x[3], x[7], 0x31);
				return tmp;
			}
	};
#endif

}

#endif /* BACKEND_CPU_VECTORS_VECTOR_TRANSPOSE_HPP_ */
