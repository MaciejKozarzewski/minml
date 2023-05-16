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
	struct Position2D;
	enum class MatrixOp;
	class MicroKernel;
	class Packing;
	class Unpacking;
}

namespace ml
{
	/*
	 * default kernels
	 */
	std::vector<MicroKernel> get_def_microkernels();
	std::vector<Packing> get_def_packing();
	std::vector<Unpacking> get_def_unpacking();
	void gemm_def_MxN_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C) noexcept;
	void gemm_def_MxN_fp16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;
	void gemm_def_MxN_bf16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;
	void pack_def_MxK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_def_MxK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_def_MxK_fp16(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_def_MxK_bf16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_def_MxK_bf16(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void unpack_def_MxK_fp32(Matrix &dst, const Position2D &dst_pos, const Fragment &src) noexcept;
	void unpack_def_MxK_fp16(Matrix &dst, const Position2D &dst_pos, const Fragment &src) noexcept;
	void unpack_def_MxK_bf16(Matrix &dst, const Position2D &dst_pos, const Fragment &src) noexcept;

	/*
	 * SSE2 kernels
	 */
	std::vector<MicroKernel> get_sse2_microkernels();
	std::vector<Packing> get_sse2_packing();
	void gemm_sse2_8x4_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;
	void gemm_sse2_4x4_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;
	void pack_sse2_8xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_sse2_4xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;

	/*
	 * AVX kernels
	 */
	std::vector<MicroKernel> get_avx_microkernels();
	std::vector<Packing> get_avx_packing();
	void gemm_avx_8x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C) noexcept;
	void gemm_avx_10x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C) noexcept;
	void pack_avx_8xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;

	/*
	 * AVX2 kernels
	 */
	std::vector<MicroKernel> get_avx2_fma_microkernels();
	std::vector<Packing> get_avx2_fma_packing();
	void gemm_avx2_fma_6x16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;
	void gemm_avx2_fma_6x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;
	void gemm_avx2_fma_12x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;
	void gemm_avx2_fma_24x4_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;
	void gemm_avx2_fma_6x16_fp16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;

	void gemm_avx2_fma_5x16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;
	void gemm_avx2_fma_6x16_fp16(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept;

	void pack_avx2_fma_6xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_avx2_fma_16xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_avx2_fma_24xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;

	void pack_avx2_fma_6xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_avx2_fma_16xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;
	void pack_avx2_fma_24xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept;

} /* namespace ml */

#endif /* BACKEND_CPU_KERNELS_GEMM_GEMM_KERNELS_HPP_ */
