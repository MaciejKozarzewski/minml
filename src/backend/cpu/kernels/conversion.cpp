/*
 * conversion.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>

#include "../vectors/vectors.hpp"

#include <type_traits>
#include <iostream>

namespace
{
	using namespace SIMD_NAMESPACE;

	template<typename T, typename U>
	void kernel_convert(T *dst, const U *src, const int elements)
	{
		for (int i = 0; i < elements; i += Vector<float>::size())
		{
			const int processed_elements = std::min(Vector<float>::size(), elements - i);
			const Vector<float> tmp(src + i, processed_elements);
			tmp.partial_store(dst + i, processed_elements);
		}
	}

	template<typename T>
	T one_or_zero(bool b) noexcept;

	template<>
	float one_or_zero(bool b) noexcept
	{
		return b ? 1.0f : 0.0f;
	}
	template<>
	float16 one_or_zero(bool b) noexcept
	{
		return b ? float16 { 0x3c00 } : float16 { 0x0000 };
	}

	template<typename T>
	void kernel_unpack_input(T *dst, const uint32_t *src, int first_dim, int last_dim)
	{
		for (int i = 0; i < first_dim; i++, dst += last_dim)
		{
			uint32_t mask = src[i];
			for (int j = 0; j < last_dim; j++, mask >>= 1)
				dst[j] = one_or_zero<T>(mask & 1u);
		}
	}

	struct TileSize
	{
			int rows, columns;
	};

	template<typename T>
	constexpr TileSize get_tile_size() noexcept
	{
		if constexpr (std::is_same<T, float>::value)
		{
#if SUPPORTS_AVX
			return TileSize( { 8, 8 });
#elif SUPPORTS_SSE2
			return TileSize( { 4, 4 });
#else
			return TileSize( { 1, 1 });
#endif
		}
		if constexpr (std::is_same<T, uint16_t>::value)
		{
#if SUPPORTS_AVX2
			return TileSize( { 8, 16});
#elif SUPPORTS_SSE2
			return TileSize( { 8, 8 });
#else
			return TileSize( { 1, 1});
#endif
		}
		return TileSize( { 1, 1 });
	}

	template<typename T>
	void def_transpose(const T *src, int src_stride, TileSize size, T *dst, int dst_stride)
	{
		for (int i = 0; i < size.rows; i++)
			for (int j = 0; j < size.columns; j++)
				dst[j * dst_stride + i] = src[i * src_stride + j];
	}
	template<typename T>
	void kernel_transpose_021(T *dst, const T *src, int dim0, int dim1, int dim2)
	{
		constexpr TileSize tile_size = get_tile_size<T>();

		for (int i = 0; i < dim0; i++)
		{
			const int tmp_dim1 = dim1 - dim1 % tile_size.rows;
			const int tmp_dim2 = dim2 - dim2 % tile_size.columns;

			// first loop over bulk of full tiles
			for (int j = 0; j < tmp_dim1; j += tile_size.rows)
				for (int k = 0; k < tmp_dim2; k += tile_size.columns)
				{
					const int src_stride = dim2;
					const int dst_stride = dim1;
					const T *src_ptr = src + j * src_stride + k;
					T *dst_ptr = dst + k * dst_stride + j;

					if constexpr (std::is_same<T, float>::value)
					{
#if SUPPORTS_AVX
						__m256 tile[8];
						for (int i = 0; i < 8; i++)
							tile[i] = _mm256_loadu_ps(src_ptr + i * src_stride);

						__m256 tmp[8];
						tmp[0] = _mm256_unpacklo_ps(tile[0], tile[1]);
						tmp[1] = _mm256_unpackhi_ps(tile[0], tile[1]);
						tmp[2] = _mm256_unpacklo_ps(tile[2], tile[3]);
						tmp[3] = _mm256_unpackhi_ps(tile[2], tile[3]);
						tmp[4] = _mm256_unpacklo_ps(tile[4], tile[5]);
						tmp[5] = _mm256_unpackhi_ps(tile[4], tile[5]);
						tmp[6] = _mm256_unpacklo_ps(tile[6], tile[7]);
						tmp[7] = _mm256_unpackhi_ps(tile[6], tile[7]);

						tile[0] = _mm256_shuffle_ps(tmp[0], tmp[2], _MM_SHUFFLE(1,0,1,0));
						tile[1] = _mm256_shuffle_ps(tmp[0], tmp[2], _MM_SHUFFLE(3,2,3,2));
						tile[2] = _mm256_shuffle_ps(tmp[1], tmp[3], _MM_SHUFFLE(1,0,1,0));
						tile[3] = _mm256_shuffle_ps(tmp[1], tmp[3], _MM_SHUFFLE(3,2,3,2));
						tile[4] = _mm256_shuffle_ps(tmp[4], tmp[6], _MM_SHUFFLE(1,0,1,0));
						tile[5] = _mm256_shuffle_ps(tmp[4], tmp[6], _MM_SHUFFLE(3,2,3,2));
						tile[6] = _mm256_shuffle_ps(tmp[5], tmp[7], _MM_SHUFFLE(1,0,1,0));
						tile[7] = _mm256_shuffle_ps(tmp[5], tmp[7], _MM_SHUFFLE(3,2,3,2));

						tmp[0] = _mm256_permute2f128_ps(tile[0], tile[4], 0x20);
						tmp[1] = _mm256_permute2f128_ps(tile[1], tile[5], 0x20);
						tmp[2] = _mm256_permute2f128_ps(tile[2], tile[6], 0x20);
						tmp[3] = _mm256_permute2f128_ps(tile[3], tile[7], 0x20);
						tmp[4] = _mm256_permute2f128_ps(tile[0], tile[4], 0x31);
						tmp[5] = _mm256_permute2f128_ps(tile[1], tile[5], 0x31);
						tmp[6] = _mm256_permute2f128_ps(tile[2], tile[6], 0x31);
						tmp[7] = _mm256_permute2f128_ps(tile[3], tile[7], 0x31);

						for (int i = 0; i < 8; i++)
							_mm256_storeu_ps(dst_ptr + i * dst_stride, tmp[i]);
#elif SUPPORTS_SSE2
						__m128 tile[4];
						for (int i = 0; i < 4; i++)
							tile[i] = _mm_loadu_ps(src_ptr + i * src_stride);
						_MM_TRANSPOSE4_PS(tile[0], tile[1], tile[2], tile[3]);
						for (int i = 0; i < 4; i++)
							_mm_storeu_ps(dst_ptr + i * dst_stride, tile[i]);
#endif
					}
					if constexpr (std::is_same<T, uint16_t>::value)
					{
#if SUPPORTS_AVX2
						__m256i tile[8];
						for (int i = 0; i < 8; i++)
						tile[i] = _mm256_loadu_si256((const __m256i*) (src_ptr + i * src_stride));

						__m256i tmp[8];
						tmp[0] = _mm256_unpacklo_epi16(tile[0], tile[1]);
						tmp[1] = _mm256_unpackhi_epi16(tile[0], tile[1]);
						tmp[2] = _mm256_unpacklo_epi16(tile[2], tile[3]);
						tmp[3] = _mm256_unpackhi_epi16(tile[2], tile[3]);
						tmp[4] = _mm256_unpacklo_epi16(tile[4], tile[5]);
						tmp[5] = _mm256_unpackhi_epi16(tile[4], tile[5]);
						tmp[6] = _mm256_unpacklo_epi16(tile[6], tile[7]);
						tmp[7] = _mm256_unpackhi_epi16(tile[6], tile[7]);

						tile[0] = _mm256_unpacklo_epi32(tmp[0], tmp[2]);
						tile[1] = _mm256_unpackhi_epi32(tmp[0], tmp[2]);
						tile[2] = _mm256_unpacklo_epi32(tmp[1], tmp[3]);
						tile[3] = _mm256_unpackhi_epi32(tmp[1], tmp[3]);
						tile[4] = _mm256_unpacklo_epi32(tmp[4], tmp[6]);
						tile[5] = _mm256_unpackhi_epi32(tmp[4], tmp[6]);
						tile[6] = _mm256_unpacklo_epi32(tmp[5], tmp[7]);
						tile[7] = _mm256_unpackhi_epi32(tmp[5], tmp[7]);

						tmp[0] = _mm256_unpacklo_epi64(tile[0], tile[4]);
						tmp[1] = _mm256_unpackhi_epi64(tile[0], tile[4]);
						tmp[2] = _mm256_unpacklo_epi64(tile[1], tile[5]);
						tmp[3] = _mm256_unpackhi_epi64(tile[1], tile[5]);
						tmp[4] = _mm256_unpacklo_epi64(tile[2], tile[6]);
						tmp[5] = _mm256_unpackhi_epi64(tile[2], tile[6]);
						tmp[6] = _mm256_unpacklo_epi64(tile[3], tile[7]);
						tmp[7] = _mm256_unpackhi_epi64(tile[3], tile[7]);

						for (int i = 0; i < 8; i++)
							_mm_storeu_si128((__m128i*) (dst_ptr + i * dst_stride), _mm256_castsi256_si128(tmp[i]));
						for (int i = 0; i < 8; i++)
							_mm_storeu_si128((__m128i*) (dst_ptr + (8 + i) * dst_stride), _mm256_extracti128_si256(tmp[i], 1));
#elif SUPPORTS_SSE2
						__m128i tile[8];
						for (int i = 0; i < 8; i++)
							tile[i] = _mm_loadu_si128((const __m128i*) (src_ptr + i * src_stride));

						__m128i tmp[8];
						tmp[0] = _mm_unpacklo_epi16(tile[0], tile[1]);
						tmp[1] = _mm_unpackhi_epi16(tile[0], tile[1]);
						tmp[2] = _mm_unpacklo_epi16(tile[2], tile[3]);
						tmp[3] = _mm_unpackhi_epi16(tile[2], tile[3]);
						tmp[4] = _mm_unpacklo_epi16(tile[4], tile[5]);
						tmp[5] = _mm_unpackhi_epi16(tile[4], tile[5]);
						tmp[6] = _mm_unpacklo_epi16(tile[6], tile[7]);
						tmp[7] = _mm_unpackhi_epi16(tile[6], tile[7]);

						tile[0] = _mm_unpacklo_epi32(tmp[0], tmp[2]);
						tile[1] = _mm_unpackhi_epi32(tmp[0], tmp[2]);
						tile[2] = _mm_unpacklo_epi32(tmp[1], tmp[3]);
						tile[3] = _mm_unpackhi_epi32(tmp[1], tmp[3]);
						tile[4] = _mm_unpacklo_epi32(tmp[4], tmp[6]);
						tile[5] = _mm_unpackhi_epi32(tmp[4], tmp[6]);
						tile[6] = _mm_unpacklo_epi32(tmp[5], tmp[7]);
						tile[7] = _mm_unpackhi_epi32(tmp[5], tmp[7]);

						tmp[0] = _mm_unpacklo_epi64(tile[0], tile[4]);
						tmp[1] = _mm_unpackhi_epi64(tile[0], tile[4]);
						tmp[2] = _mm_unpacklo_epi64(tile[1], tile[5]);
						tmp[3] = _mm_unpackhi_epi64(tile[1], tile[5]);
						tmp[4] = _mm_unpacklo_epi64(tile[2], tile[6]);
						tmp[5] = _mm_unpackhi_epi64(tile[2], tile[6]);
						tmp[6] = _mm_unpacklo_epi64(tile[3], tile[7]);
						tmp[7] = _mm_unpackhi_epi64(tile[3], tile[7]);

						for (int i = 0; i < 8; i++)
							_mm_storeu_si128((__m128i*) (dst_ptr + i * dst_stride), tmp[i]);
#endif
					}
				}
			def_transpose(src + tmp_dim2, dim2, TileSize( { tmp_dim1, dim2 - tmp_dim2 }), dst + tmp_dim2 * dim1, dim1); // rightmost panel (without lower right corner)
			def_transpose(src + tmp_dim1 * dim2, dim2, TileSize( { dim1 - tmp_dim1, dim2 }), dst + tmp_dim1, dim1); // bottom panel

			src += dim1 * dim2;
			dst += dim1 * dim2;
		}
	}
}

namespace SIMD_NAMESPACE
{
	using namespace ml;

	void cpu_kernel_unpack_input(mlContext_t context, mlShape_t shape, mlDataType_t dst_dtype, void *dst, const void *src)
	{
		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);
		assert(last_dim <= 32);
		switch (dst_dtype)
		{
			case DTYPE_FLOAT16:
				kernel_unpack_input(getPointer<float16>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			case DTYPE_FLOAT32:
				kernel_unpack_input(getPointer<float>(dst), getPointer<uint32_t>(src), first_dim, last_dim);
				break;
			default:
				break;
		}
	}
	void cpu_kernel_convert_type(mlContext_t context, void *dst, mlDataType_t dst_dtype, const void *src, mlDataType_t src_dtype, int elements)
	{
		if (dst_dtype == src_dtype and dst != src)
		{ // same type, different locations, can just copy memory
			std::memcpy(dst, src, size_of(dst_dtype) * elements);
			return;
		}

		if (dst_dtype == DTYPE_FLOAT16 and src_dtype == DTYPE_FLOAT32)
		{
			assert(dst != src);
			kernel_convert(getPointer<float16>(dst), getPointer<float>(src), elements);
			return;
		}
		if (dst_dtype == DTYPE_FLOAT32 and src_dtype == DTYPE_FLOAT16)
		{

			assert(dst != src);
			kernel_convert(getPointer<float>(dst), getPointer<float16>(src), elements);
			return;
		}
	}
	void cpu_kernel_transpose_021(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
		assert(input != output);

		switch (dtype)
		{
			case DTYPE_FLOAT16:
				kernel_transpose_021<uint16_t>(getPointer<uint16_t>(output), getPointer<uint16_t>(input), shape.dim[0], shape.dim[1], shape.dim[2]);
				break;
			case DTYPE_FLOAT32:
			case DTYPE_INT32:
				kernel_transpose_021<float>(getPointer<float>(output), getPointer<float>(input), shape.dim[0], shape.dim[1], shape.dim[2]);
				break;
			default:
				break;
		}
	}
}

