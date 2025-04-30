/*
 * gemm_kernels.hpp
 *
 *  Created on: May 10, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_GEMM_GEMM_KERNELS_HPP_
#define BACKEND_CPU_KERNELS_GEMM_GEMM_KERNELS_HPP_

#include <vector>

namespace ml
{
	class Fragment;
	class Matrix;
	class BatchedMatrix;
	struct Position2D;
	enum class MatrixOp;
	class GemmRuntime;
}

namespace ml
{
	/*
	 * default kernels
	 */
	void gemm_def_MxN(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept;
	void pack_def_MxK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void unpack_def_MxK(Matrix &dst, const Position2D &dst_pos, const Fragment &src) noexcept;
	// multi-head attention (MHA) kernel
	void mha_qk_def_MxN(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept;
	void mha_softmax_def_MxN(Fragment &temp, Fragment &softmax_sum) noexcept;
	void mha_pack_bias_def(Fragment &dst, const BatchedMatrix &src, int head, int height, int width, int range) noexcept;
	// batched depthwise convolution kernel
	void depthwise_conv_def_MxN(Fragment &C, const Fragment &alpha, const Fragment &A, const Fragment &B) noexcept;
	void depthwise_conv_def_MxN_v2(Fragment &C, const Fragment &A, const Fragment &B, const Fragment &bias) noexcept;

	/*
	 * SSE2 kernels
	 */
	void gemm_sse2_4x8(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept;
	void pack_sse2_4xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_sse2_8xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	// multi-head attention (MHA) kernel
	void mha_qk_sse2_4x8(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept;
	void mha_pack_bias_sse2(Fragment &dst, const BatchedMatrix &src, int head, int height, int width, int range) noexcept;
	// batched depthwise convolution kernel
	void depthwise_conv_sse2_4x8(Fragment &C, const Fragment &alpha, const Fragment &A, const Fragment &B) noexcept;

	/*
	 * AVX kernels
	 */
	void gemm_avx_10x8(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept;
	void gemm_avx_8x8(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept;
	void pack_avx_10xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_avx_8xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	// multi-head attention (MHA) kernel
	void mha_qk_avx_10x8(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept;
	void mha_pack_bias_avx(Fragment &dst, const BatchedMatrix &src, int head, int height, int width, int range) noexcept;
	// batched depthwise convolution kernel
	void depthwise_conv_avx_10x8(Fragment &C, const Fragment &alpha, const Fragment &A, const Fragment &B) noexcept;

	/*
	 * AVX2 kernels
	 */

	void gemm_avx2_12x8(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept;
	void gemm_avx2_6x16(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept;
	void pack_avx2_12xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_avx2_6xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_avx2_16xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	// multi-head attention (MHA) kernel
	void mha_qk_avx2_12x8(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept;
	void mha_softmax_avx2_12x8(Fragment &temp, Fragment &softmax_sum) noexcept;

	// batched depthwise convolution kernel
	void depthwise_conv_avx2_12x8(Matrix &output, const Matrix &input, const Matrix &weights, const Matrix &bias, const int *args,
			void *workspace) noexcept;
	void fused_conv_block_stage_1_avx2_12x8(Fragment &temp, const Fragment &A, const Fragment &B, const Fragment &bias) noexcept;
	void quantize_avx2_8xK(Fragment &dst, const Fragment &src, const Fragment &scales) noexcept;

	void intgemm_avx2_12x8(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept;

	/*
	 * AVX512 kernels
	 */
	void gemm_avx512f_24x16(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept;
	void pack_avx512f_24xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_avx512f_16xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	// multi-head attention (MHA) kernel
	void mha_qk_avx512f_24x16(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept;
	void mha_pack_bias_avx512(Fragment &dst, const BatchedMatrix &src, int head, int height, int width, int range) noexcept;
	// batched depthwise convolution kernel
	void depthwise_conv_avx512f_24x16(Fragment &C, const Fragment &alpha, const Fragment &A, const Fragment &B) noexcept;

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_GEMM_GEMM_KERNELS_HPP_ */
