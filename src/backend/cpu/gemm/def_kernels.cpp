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
#include "../utils.hpp"
#include "../fp16.hpp"

#include <cinttypes>
#include <cstring>
#include <cassert>
#include <iostream>
#include <cmath>

#ifndef M_LN2
#define M_LN2 0.69314718056
#endif

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
//		uint64_t y;
//		asm volatile(
//				"movzw %[x], %%r14 \n\t"
//				"vmovq %%r14, %%xmm0 \n\t"
//				"vcvtph2ps %%xmm0, %%xmm0 \n\t"
//				"vmovq %%xmm0, %%r14 \n\t"
//				"movq %%r14, %[y] \n\t"
//				: // outputs
//				[y] "=m"(y)
//				:// inputs
//				[x] "m"(x)
//				:// clobbers
//				"cc", "%ymm0", "%r14");
//		return as_float(y);

//		return _cvtsh_ss(x);

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
//		uint64_t tmp_x = as_uint(x);
//		uint64_t y;
//		asm volatile(
//				"movq %[tmp_x], %%r14 \n\t"
//				"vmovq %%r14, %%xmm0 \n\t"
//				"vcvtps2ph $(0x03), %%xmm0, %%xmm0 \n\t"
//				"vmovq %%xmm0, %%r14 \n\t"
//				"movq %%r14, %[y] \n\t"
//				: // outputs
//				[y] "=m"(y)
//				:// inputs
//				[tmp_x] "m"(tmp_x)
//				:// clobbers
//				"cc", "%ymm0", "%r14");
//		return y;

//		return _cvtss_sh(x, (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC));

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

	using namespace ml;
	using namespace ml::cpu;

	template<typename SrcT, typename DstT>
	DstT convert(SrcT x) noexcept
	{
		return static_cast<DstT>(x);
	}
	template<>
	float16 convert(float x) noexcept
	{
		return cpu::convert_fp32_to_fp16(x);
	}
	template<>
	float convert(float16 x) noexcept
	{
		return cpu::convert_fp16_to_fp32(x);
	}
	template<typename T>
	T relu(T x) noexcept
	{
		return (x > static_cast<T>(0)) ? x : static_cast<T>(0);
	}

	float as_float(const int32_t x) noexcept
	{
		return *(float*) &x;
	}
	float fast_exp(float x) noexcept
	{
		// maximum relative error = 0.628981%
		static constexpr float a = (1 << 22) / float(M_LN2);
		static constexpr int32_t b = 127 * (1 << 23) - 139160;

		const int32_t r = static_cast<int32_t>(a * x);
		const float s = as_float(b + r);
		const float t = as_float(b - r);
		return s / t;
	}
	float fast_sigmoid(float x) noexcept
	{
		// maximum relative error = 0.628656%
		static constexpr float a = (1 << 22) / float(M_LN2);
		static constexpr int32_t b = 127 * (1 << 23) - 139002;
		const int32_t r = static_cast<int32_t>(a * x);
		const float s = as_float(b + r);
		const float t = as_float(b - r);
		return s / (s + t);
	}
	float fast_tanh(float x) noexcept
	{
		// maximum relative error = 4.049%
		if (std::fabs(x) < 0.347f)
			return x * (1.0f - 0.0924f * x);
		else
		{
			static constexpr float a = (1 << 23) / float(M_LN2);
			static constexpr int32_t b = 127 * (1 << 23);
			const int32_t r = static_cast<int32_t>(a * x);
			const float s = as_float(b + r);
			const float t = as_float(b - r);
			return (s - t) / (s + t);
		}
	}
	float fast_gelu(float x) noexcept
	{
		static constexpr float a = (1 << 22) * (1.6849f / float(M_LN2));
		static constexpr int32_t b = 127 * (1 << 23) - 329698;
		const int32_t r = static_cast<int32_t>(a * x);
		const float s = as_float(b + r);
		const float t = as_float(b - r);
		return (s * x) / (s + t);
	}

	/*
	 * computes D = optional_relu(alpha * op(A) * op(B) + beta * C + broadcast(bias))
	 */
	template<typename DT, typename AT, typename BT, typename CT, typename ComputeType>
	void kernel_gemm(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
		assert(A.is_packed() && A.data() != nullptr);
		assert(B.is_packed() && B.data() != nullptr);
		assert(A.rows() == B.rows());
		const int M = A.columns();
		const int N = B.columns();
		const int K = A.rows();

		std::unique_ptr<ComputeType[]> acc = std::make_unique<ComputeType[]>(M * N);
		for (int i = 0; i < M * N; i++)
			acc[i] = 0.0f;

		for (int k = 0; k < K; k++)
			for (int m = 0; m < M; m++)
			{
				const ComputeType tmp = convert<AT, ComputeType>(A.at<AT>(k, m));
				for (int n = 0; n < N; n++)
					acc[m * N + n] += tmp * convert<BT, ComputeType>(B.at<BT>(k, n));
			}

		assert(alpha.is_packed());
		assert(alpha.is_fp32() || alpha.is_fp64());
		if (alpha.rows() == 1 and alpha.columns() == 1)
		{
			for (int i = 0; i < M * N; i++)
				acc[i] *= alpha.at<ComputeType>(0, 0);
		}
		if (alpha.rows() == M and alpha.columns() == 1)
		{
			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					acc[m * N + n] *= alpha.at<ComputeType>(m, 0);
		}

		if (bias.is_packed())
		{
			assert(bias.data() != nullptr);
			assert(bias.rows() == 1);
			assert(bias.columns() == N);
			assert(bias.dtype() == B.dtype());
			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					acc[m * N + n] += convert<BT, ComputeType>(bias.at<BT>(0, n));
		}

		assert(beta_ptr != nullptr);
		const ComputeType beta = reinterpret_cast<const ComputeType*>(beta_ptr)[0];
		if (beta != 0.0f)
		{
			assert(C.size() == D.size());
			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					acc[m * N + n] += beta * convert<DT, ComputeType>(C.at<DT>(m, n));
		}
		if (use_relu)
		{
			for (int i = 0; i < M * N; i++)
				acc[i] = relu(acc[i]);
		}

		assert(D.rows() == M);
		assert(D.columns() == N);
		for (int m = 0; m < M; m++)
			for (int n = 0; n < N; n++)
				D.at<DT>(m, n) = convert<ComputeType, DT>(acc[m * N + n]);
	}

	template<typename SrcT, typename DstT>
	void kernel_pack(Fragment &dst, const Matrix &src, int src_row, int src_col, MatrixOp src_op) noexcept
	{
		const int M = dst.columns();
		const int K = dst.rows();

		if (M != dst.stride())
			std::memset(dst.data(), 0, size_of(dst.dtype()) * K * dst.stride()); // zero-fill the edges

		const SrcT *src_ptr = reinterpret_cast<const SrcT*>(src.pointer_at(src_row, src_col));
		DstT *dst_ptr = dst.data<DstT>();

		if (src_op == MatrixOp::NORMAL)
		{ // pack K x M fragment
			for (int k = 0; k < K; k++)
			{
				for (int m = 0; m < M; m++)
					dst_ptr[m] = convert<SrcT, DstT>(src_ptr[m]);
				src_ptr += src.stride();
				dst_ptr += dst.stride();
			}
		}
		else
		{ // pack M x K fragment
			for (int m = 0; m < M; m++)
			{
				for (int k = 0; k < K; k++)
					dst_ptr[k * dst.stride()] = convert<SrcT, DstT>(src_ptr[k]);
				src_ptr += src.stride();
				dst_ptr += 1;
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

	template<typename CT, typename AT, typename BT>
	void kernel_mha(Fragment &temp, const void *alpha_ptr, const Fragment &A, const Fragment &B, const Fragment &bias, Fragment &softmax_sum) noexcept
	{
		assert(A.is_packed() && A.data() != nullptr);
		assert(B.is_packed() && B.data() != nullptr);
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
				const float tmp = convert<AT, float>(A.at<AT>(k, m));
				for (int n = 0; n < N; n++)
					acc[m * N + n] += tmp * convert<BT, float>(B.at<BT>(k, n));
			}

		assert(alpha_ptr != nullptr);
		const float alpha = reinterpret_cast<const float*>(alpha_ptr)[0];

		assert(bias.is_packed());
		assert(bias.data() != nullptr);
		assert(bias.rows() == M);
		assert(bias.columns() == N);
		assert(bias.dtype() == B.dtype());
		assert(temp.columns() == M);
		for (int m = 0; m < M; m++)
		{
			float sum = softmax_sum.is_packed() ? softmax_sum.at<CT>(m, 0) : 0.0f;
			for (int n = 0; n < N; n++)
			{
				const float b = bias.is_packed() ? convert<BT, float>(bias.at<BT>(m, n)) : 0.0f;
				const float tmp = fast_exp(acc[m * N + n] * alpha + b);
				temp.at<CT>(n, m) = convert<float, CT>(tmp);
				sum += tmp;
			}
			if (softmax_sum.is_packed())
				softmax_sum.at<CT>(m, 0) = sum;
		}
	}
	template<typename T>
	void kernel_softmax(Fragment &temp, Fragment &softmax_sum) noexcept
	{
		if (softmax_sum.is_packed())
		{
			for (int n = 0; n < temp.rows(); n++)
				for (int m = 0; m < temp.columns(); m++)
				{
					softmax_sum.data<T>()[m] += temp.data<T>()[n * temp.columns() + m];
//				T sum = softmax_sum.at<T>(m, 0);

					{
//					const T t = fast_exp(temp.at<T>(n, m));
//					temp.at<T>(n, m) = t;
//					sum += t;
//					sum += temp.at<T>(n, m);
					}
//				softmax_sum.at<T>(m, 0) = sum;
				}
		}
	}

	template<typename AT, typename BT, typename CT>
	void kernel_depthwise_conv(Fragment &C, const Fragment &alpha, const Fragment &A, const Fragment &B) noexcept
	{
		assert(A.is_packed() && A.data() != nullptr);
		assert(B.is_packed() && B.data() != nullptr);
		assert(A.rows() == B.rows());

		const int N = B.columns();
		assert(A.columns() % N == 0);
		const int M = A.columns() / N;
		const int K = A.rows();

		std::unique_ptr<float[]> acc = std::make_unique<float[]>(M * N);
		for (int i = 0; i < M * N; i++)
			acc[i] = 0.0f;

		for (int k = 0; k < K; k++)
			for (int m = 0; m < M; m++)
			{
				for (int n = 0; n < N; n++)
					acc[m * N + n] += convert<AT, float>(A.at<AT>(k, m * N + n)) * convert<BT, float>(B.at<BT>(k, n));
			}

		assert(alpha.is_packed());
		assert(alpha.is_fp32());
		if (alpha.rows() == 1 and alpha.columns() == 1)
		{
			for (int i = 0; i < M * N; i++)
				acc[i] *= alpha.at<float>(0, 0);
		}
		if (alpha.rows() == 1 and alpha.columns() == N)
		{
			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					acc[m * N + n] *= alpha.at<float>(0, n);
		}

		assert(C.rows() == M);
		assert(C.columns() == N);
		for (int m = 0; m < M; m++)
			for (int n = 0; n < N; n++)
				C.at<CT>(m, n) = convert<float, CT>(acc[m * N + n]);
	}
	template<typename AT, typename BT, typename CT>
	void kernel_depthwise_conv_v2(Fragment &C, const Fragment &A, const Fragment &B, const Fragment &bias) noexcept
	{
		assert(A.is_packed() && A.data() != nullptr);
		assert(B.is_packed() && B.data() != nullptr);

		const int channels = B.columns();
		const int kernel_size = std::sqrt(B.rows());
		const int inputs = A.columns() / channels;
		const int outputs = C.rows();

		std::unique_ptr<float[]> acc = std::make_unique<float[]>(outputs * channels);
		for (int i = 0; i < outputs * channels; i++)
			acc[i] = 0.0f;

		for (int h = 0; h < kernel_size; h++)
			for (int out = 0; out < outputs; out++)
				for (int w = 0; w < kernel_size; w++)
				{
					for (int c = 0; c < channels; c++)
						acc[out * channels + c] += convert<AT, float>(A.at<AT>(h, (out + w) * channels + c))
								* convert<BT, float>(B.at<BT>(h * kernel_size + w, c));
				}

		if (bias.is_packed())
		{
			assert(bias.data() != nullptr);
			assert(bias.rows() == 1);
			assert(bias.columns() == channels);
			assert(bias.dtype() == B.dtype());
			for (int out = 0; out < outputs; out++)
				for (int c = 0; c < channels; c++)
					acc[out * channels + c] += convert<BT, float>(bias.at<BT>(0, c));
		}

		for (int out = 0; out < outputs; out++)
			for (int c = 0; c < channels; c++)
				C.at<CT>(out, c) = convert<float, CT>(acc[out * channels + c]);
	}
}

namespace ml
{
	/*
	 * Computes D = alpha * A * B + beta * C
	 * C and D may point to the same object
	 */
	void gemm_def_MxN(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
		if (A.is_fp64() and B.is_fp64() and C.is_fp64() and D.is_fp64())
		{
			kernel_gemm<double, double, double, double, double>(D, alpha, A, B, beta_ptr, C, bias, use_relu);
			return;
		}
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(C.is_fp32() || C.is_fp16());
		assert(D.is_fp32() || D.is_fp16());
		if (C.is_fp32())
		{
			if (D.is_fp32())
				kernel_gemm<float, float, float, float, float>(D, alpha, A, B, beta_ptr, C, bias, use_relu);
			else
				kernel_gemm<float16, float, float, float, float>(D, alpha, A, B, beta_ptr, C, bias, use_relu);
		}
		else
		{
			if (D.is_fp32())
				kernel_gemm<float, float, float, float16, float>(D, alpha, A, B, beta_ptr, C, bias, use_relu);
			else
				kernel_gemm<float16, float, float, float16, float>(D, alpha, A, B, beta_ptr, C, bias, use_relu);
		}
	}

	void pack_def_MxK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		if (dst.is_fp64() and src.is_fp64())
		{
			kernel_pack<double, double>(dst, src, src_pos.row, src_pos.column, src_op);
			return;
		}
		assert(dst.is_fp32() || dst.is_fp16());
		assert(src.is_fp32() || src.is_fp16());
		if (src.is_fp32())
		{
			if (dst.is_fp32())
				kernel_pack<float, float>(dst, src, src_pos.row, src_pos.column, src_op);
			else
				kernel_pack<float, float16>(dst, src, src_pos.row, src_pos.column, src_op);
		}
		else
		{
			if (dst.is_fp32())
				kernel_pack<float16, float>(dst, src, src_pos.row, src_pos.column, src_op);
			else
				kernel_pack<float16, float16>(dst, src, src_pos.row, src_pos.column, src_op);
		}
	}
	void unpack_def_MxK(Matrix &dst, const Position2D &dst_pos, const Fragment &src) noexcept
	{
		if (dst.is_fp64() and src.is_fp64())
		{
			kernel_unpack<double, double>(dst, dst_pos.row, dst_pos.column, src);
			return;
		}
		assert(dst.is_fp32() || dst.is_fp16());
		assert(src.is_fp32() || src.is_fp16());
		if (src.is_fp32())
		{
			if (dst.is_fp32())
				kernel_unpack<float, float>(dst, dst_pos.row, dst_pos.column, src);
			else
				kernel_unpack<float, float16>(dst, dst_pos.row, dst_pos.column, src);
		}
		else
		{
			if (dst.is_fp32())
				kernel_unpack<float16, float>(dst, dst_pos.row, dst_pos.column, src);
			else
				kernel_unpack<float16, float16>(dst, dst_pos.row, dst_pos.column, src);
		}
	}
	// multi-head attention kernel
	void mha_qk_def_MxN(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept
	{
		kernel_mha<float, float, float>(temp, alpha_ptr, Q, K, bias, softmax_sum);
	}
	void mha_softmax_def_MxN(Fragment &temp, Fragment &softmax_sum) noexcept
	{
		assert(temp.is_fp32());
		if (softmax_sum.is_packed())
		{
			assert(softmax_sum.is_fp32());
		}
		kernel_softmax<float>(temp, softmax_sum);
	}
	// batched depthwise convolution kernel
	void depthwise_conv_def_MxN(Fragment &C, const Fragment &alpha, const Fragment &A, const Fragment &B) noexcept
	{
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(C.is_fp32() || C.is_fp16());
		if (C.is_fp32())
			kernel_depthwise_conv<float, float, float>(C, alpha, A, B);
		else
			kernel_depthwise_conv<float, float, float16>(C, alpha, A, B);
	}
	void depthwise_conv_def_MxN_v2(Fragment &C, const Fragment &A, const Fragment &B, const Fragment &bias) noexcept
	{
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(C.is_fp32() || C.is_fp16());
		if (C.is_fp32())
			kernel_depthwise_conv_v2<float, float, float>(C, A, B, bias);
		else
			kernel_depthwise_conv_v2<float, float, float16>(C, A, B, bias);
	}
} /* namespace ml */

