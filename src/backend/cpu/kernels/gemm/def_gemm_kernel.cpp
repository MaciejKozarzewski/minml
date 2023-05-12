/*
 * def_gemm_kernel.cpp
 *
 *  Created on: May 6, 2023
 *      Author: Maciej Kozarzewski
 */

#include "Fragment.hpp"
#include "Matrix.hpp"
#include "gemm_kernels.hpp"
#include "utilities.hpp"
#include "../../utils.hpp"
#include "../../vectors/types.hpp"

#include <x86intrin.h>
#include <cinttypes>
#include <cassert>

namespace
{
	uint32_t as_uint(const float x) noexcept
	{
		return *(uint32_t*) &x;
	}
	float as_float(const uint32_t x) noexcept
	{
		return *(float*) &x;
	}
	float fp16_to_fp32(const uint16_t x) noexcept
	{
		uint32_t exponent = x & 0x7C00; // '0 11111 0000000000'
		uint32_t mantissa = x & 0x03FF; // '0 00000 1111111111'

		const uint32_t sign = (x & 0x8000) << 16; // '1 00000 0000000000'
		if (exponent == 0x7C00)
		{
			exponent = 0x3FC00; // +/- Inf or +/- NaN (it's 0x7F800000 >> 13)
			mantissa |= ((mantissa != 0) << 9); // set first bit of the mantissa in case of NaN, but preserve other bits
		}
		else
		{
			if (exponent != 0) // normalized
				exponent += (112 << 10);
			else
			{
				if (mantissa != 0)
				{ // denormalized
					const uint32_t v = as_uint((float) mantissa) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
					exponent = (v - 24) << 10;
					mantissa = (mantissa << (137 - v)) & 0x03FF;
				}
			}
		}
		return as_float(sign | ((exponent | mantissa) << 13));
	}
	uint16_t fp32_to_fp16(const float x) noexcept
	{
		const uint32_t original = as_uint(x);
		const uint32_t rounded = original + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
		uint32_t exponent = (rounded & 0x7F800000) >> 23; // exponent
		uint32_t mantissa = rounded & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding

		const uint32_t sign = (original & 0x80000000) >> 16;

		if ((original & 0x7FFFFFFF) > 0x7F800000)
		{ // check NaN
			exponent = 0x7C00;
			mantissa = ((original & 0x007FFFFF) >> 13) | 0x200; // set first mantissa bit but preserve others
		}
		else
		{
			if (exponent > 142)
			{ // +/- Inf
				exponent = 0x7C00;
				mantissa = 0;
			}
			else
			{
				if (exponent > 112)
				{ // normalized
					exponent = ((exponent - 112) << 10) & 0x7C00;
					mantissa >>= 13;
				}
				else
				{ // denormalized
					mantissa += 0x007FF000; // TODO figure out why it is here
					mantissa >>= std::min(125u - exponent, 31u);
					mantissa = (mantissa + 1) >> 1;
					exponent = 0;
				}
			}
		}
		return sign | exponent | mantissa;
	}

	float bf16_to_fp32(const uint16_t x) noexcept
	{
		return as_float(static_cast<uint32_t>(x) << 16);
	}
	uint16_t fp32_to_bf16(const float x) noexcept
	{
		return as_uint(x) >> 16;
	}

	using namespace ml;
	template<typename SrcT, typename DstT>
	DstT convert(SrcT x) noexcept
	{
		return static_cast<DstT>(x);
	}
	template<>
	float16 convert(float x) noexcept
	{
		return float16 { fp32_to_fp16(x) };
	}
	template<>
	bfloat16 convert(float x) noexcept
	{
		return bfloat16 { fp32_to_bf16(x) };
	}
	template<>
	float convert(float16 x) noexcept
	{
		return fp16_to_fp32(x.m_data);
	}
	template<>
	float convert(bfloat16 x) noexcept
	{
		return bf16_to_fp32(x.m_data);
	}

	template<typename T>
	void kernel_gemm_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		const int M = A.columns();
		const int N = B.columns();
		const int K = A.rows();

		std::unique_ptr<float[]> acc = std::make_unique<float[]>(M * N);
		for (int i = 0; i < M * N; i++)
			acc[i] = 0.0f;

		for (int k = 0; k < K; k++)
			for (int m = 0; m < M; m++)
			{
				const float tmp = A.at<float>(k, m);
				for (int n = 0; n < N; n++)
					acc[m * N + n] += tmp * B.at<float>(k, n);
			}

		assert(alpha_ptr != nullptr);
		const float alpha = reinterpret_cast<const float*>(alpha_ptr)[0];
		if (alpha != 1.0f)
		{
			for (int i = 0; i < M * N; i++)
				acc[i] *= alpha;
		}

		assert(D.rows() == M);
		assert(D.columns() == N);
		assert(C.size() == D.size());
		assert(beta_ptr != nullptr);
		const float beta = reinterpret_cast<const float*>(beta_ptr)[0];
		if (beta == 0.0f)
		{
			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					D.at<T>(m, n) = convert<float, T>(acc[m * N + n]);
		}
		else
		{

			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					D.at<T>(m, n) = convert<float, T>(beta * convert<T, float>(C.at<T>(m, n)) + acc[m * N + n]);
		}
	}

	template<typename SrcT, typename DstT>
	void kernel_pack(Fragment &dst, const Matrix &src, int src_row, int src_col, MatrixOp src_op) noexcept
	{
		const int M = dst.columns();
		const int K = dst.rows();

		const SrcT *src_ptr = reinterpret_cast<const SrcT*>(src.pointer_at(src_row, src_col));
		DstT *dst_ptr = dst.data<DstT>();

		if (src_op == MatrixOp::NORMAL)
		{ // pack K x M fragment
			const int max_rows = std::min(K, src.rows() - src_row);
			const int max_cols = std::min(M, src.columns() - src_col);
			for (int k = 0; k < max_rows; k++)
			{
				for (int m = 0; m < max_cols; m++)
					dst_ptr[m] = convert<SrcT, DstT>(src_ptr[m]);
				src_ptr += src.stride();
				dst_ptr += dst.stride();
			}
		}
		else
		{ // pack M x K fragment
			const int max_rows = std::min(M, src.rows() - src_row);
			const int max_cols = std::min(K, src.columns() - src_col);
			for (int k = 0; k < max_cols; k++)
			{
				for (int m = 0; m < max_rows; m++)
					dst_ptr[m] = convert<SrcT, DstT>(src_ptr[m * src.stride()]);
				src_ptr += 1;
				dst_ptr += dst.stride();
			}
		}
	}

	template<typename SrcT, typename DstT>
	void kernel_unpack(Matrix &dst, int dst_row, int dst_col, const Fragment &src) noexcept
	{
		const int M = src.columns();
		const int K = src.rows();

		const SrcT *src_ptr = src.data<SrcT>();
		DstT *dst_ptr = reinterpret_cast<DstT*>(dst.pointer_at(dst_row, dst_col));

		const int max_rows = std::min(K, dst.rows() - dst_row);
		const int max_cols = std::min(M, dst.columns() - dst_col);
		for (int k = 0; k < max_rows; k++)
		{
			for (int m = 0; m < max_cols; m++)
				dst_ptr[m] = convert<SrcT, DstT>(src_ptr[m]);
			src_ptr += src.stride();
			dst_ptr += dst.stride();
		}
	}
}

namespace ml
{

	/*
	 * Computes D = alpha * A * B + beta * C
	 * C and D may point to the same object
	 */
	void gemm_def_MxN_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C) noexcept
	{
		kernel_gemm_fp32<float>(D, alpha_ptr, A, B, beta_ptr, C);
	}
	void gemm_def_MxN_fp16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		kernel_gemm_fp32<float16>(D, alpha_ptr, A, B, beta_ptr, C);
	}
	void gemm_def_MxN_bf16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		kernel_gemm_fp32<bfloat16>(D, alpha_ptr, A, B, beta_ptr, C);
	}

	void pack_def_MxK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		kernel_pack<float, float>(dst, src, src_pos.row, src_pos.column, src_op);
	}
	void pack_def_MxK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		kernel_pack<float16, float>(dst, src, src_pos.row, src_pos.column, src_op);
	}
	void pack_def_MxK_fp16(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		kernel_pack<float16, float16>(dst, src, src_pos.row, src_pos.column, src_op);
	}
	void pack_def_MxK_bf16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		kernel_pack<bfloat16, float>(dst, src, src_pos.row, src_pos.column, src_op);
	}
	void pack_def_MxK_bf16(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		kernel_pack<bfloat16, bfloat16>(dst, src, src_pos.row, src_pos.column, src_op);
	}

	void unpack_def_MxK_fp32(Matrix &dst, const Position2D &dst_pos, const Fragment &src) noexcept
	{
		kernel_unpack<float, float>(dst, dst_pos.row, dst_pos.column, src);
	}
	void unpack_def_MxK_fp16(Matrix &dst, const Position2D &dst_pos, const Fragment &src) noexcept
	{
		kernel_unpack<float16, float16>(dst, dst_pos.row, dst_pos.column, src);
	}
	void unpack_def_MxK_bf16(Matrix &dst, const Position2D &dst_pos, const Fragment &src) noexcept
	{
		kernel_unpack<bfloat16, bfloat16>(dst, dst_pos.row, dst_pos.column, src);
	}

} /* namespace ml */

