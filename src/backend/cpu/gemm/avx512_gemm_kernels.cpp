/*
 * avx512_gemm_kernels.cpp
 *
 *  Created on: Sep 24, 2023
 *      Author: Maciej Kozarzewski
 */

#include "Fragment.hpp"
#include "Matrix.hpp"
#include "gemm_kernels.hpp"
#include "../utils.hpp"
#include "../vectors/vectors.hpp"

#include <x86intrin.h>
#include <cinttypes>
#include <cassert>

#include "../assembly_macros.hpp"

#define ZERO_ACCUMULATORS()\
	vxorps(zmm8, zmm8, zmm8)\
	vxorps(zmm9, zmm9, zmm9)\
	vxorps(zmm10, zmm10, zmm10)\
	vxorps(zmm11, zmm11, zmm11)\
	vxorps(zmm12, zmm12, zmm12)\
	vxorps(zmm13, zmm13, zmm13)\
	vxorps(zmm14, zmm14, zmm14)\
	vxorps(zmm15, zmm15, zmm15)\
	vxorps(zmm16, zmm16, zmm16)\
	vxorps(zmm17, zmm17, zmm17)\
	vxorps(zmm18, zmm18, zmm18)\
	vxorps(zmm19, zmm19, zmm19)\
	vxorps(zmm20, zmm20, zmm20)\
	vxorps(zmm21, zmm21, zmm21)\
	vxorps(zmm22, zmm22, zmm22)\
	vxorps(zmm23, zmm23, zmm23)\
	vxorps(zmm24, zmm24, zmm24)\
	vxorps(zmm25, zmm25, zmm25)\
	vxorps(zmm26, zmm26, zmm26)\
	vxorps(zmm27, zmm27, zmm27)\
	vxorps(zmm28, zmm28, zmm28)\
	vxorps(zmm29, zmm29, zmm29)\
	vxorps(zmm30, zmm30, zmm30)\
	vxorps(zmm31, zmm31, zmm31)

#define SUB_KERNEL_24xFP32_16xFP32(n) \
	vmovaps(mem(rbx, n*16*4), zmm6)\
	vpermilps(imm(0xB1), zmm6, zmm7)\
	vbroadcastsd(mem(rax, (24*n+0)*4), zmm0)\
	vbroadcastsd(mem(rax, (24*n+2)*4), zmm1)\
	vbroadcastsd(mem(rax, (24*n+4)*4), zmm2)\
	vbroadcastsd(mem(rax, (24*n+6)*4), zmm3)\
	vbroadcastsd(mem(rax, (24*n+8)*4), zmm4)\
	vbroadcastsd(mem(rax, (24*n+10)*4), zmm5)\
	vfmadd231ps(zmm0, zmm6, zmm8)\
	vfmadd231ps(zmm0, zmm7, zmm9)\
	vfmadd231ps(zmm1, zmm6, zmm10)\
	vfmadd231ps(zmm1, zmm7, zmm11)\
	vfmadd231ps(zmm2, zmm6, zmm12)\
	vfmadd231ps(zmm2, zmm7, zmm13)\
	vfmadd231ps(zmm3, zmm6, zmm14)\
	vfmadd231ps(zmm3, zmm7, zmm15)\
	vfmadd231ps(zmm4, zmm6, zmm16)\
	vfmadd231ps(zmm4, zmm7, zmm17)\
	vfmadd231ps(zmm5, zmm6, zmm18)\
	vfmadd231ps(zmm5, zmm7, zmm19)\
	vbroadcastsd(mem(rax, (24*n+12)*4), zmm0)\
	vbroadcastsd(mem(rax, (24*n+14)*4), zmm1)\
	vbroadcastsd(mem(rax, (24*n+16)*4), zmm2)\
	vbroadcastsd(mem(rax, (24*n+18)*4), zmm3)\
	vbroadcastsd(mem(rax, (24*n+20)*4), zmm4)\
	vbroadcastsd(mem(rax, (24*n+22)*4), zmm5)\
	vfmadd231ps(zmm0, zmm6, zmm20)\
	vfmadd231ps(zmm0, zmm7, zmm21)\
	vfmadd231ps(zmm1, zmm6, zmm22)\
	vfmadd231ps(zmm1, zmm7, zmm23)\
	vfmadd231ps(zmm2, zmm6, zmm24)\
	vfmadd231ps(zmm2, zmm7, zmm25)\
	vfmadd231ps(zmm3, zmm6, zmm26)\
	vfmadd231ps(zmm3, zmm7, zmm27)\
	vfmadd231ps(zmm4, zmm6, zmm28)\
	vfmadd231ps(zmm4, zmm7, zmm29)\
	vfmadd231ps(zmm5, zmm6, zmm30)\
	vfmadd231ps(zmm5, zmm7, zmm31)

#define LOAD_ADD_3x16xFP32(beta, reg0, reg1, reg2)\
	vmovups(mem(rcx), zmm1)\
	vmovups(mem(rcx, r14, 1), zmm2)\
	vmovups(mem(rcx, r14, 2), zmm3)\
	vfmadd231ps(zmm1, beta, reg0)\
	vfmadd231ps(zmm2, beta, reg1)\
	vfmadd231ps(zmm3, beta, reg2)\
	add(r15, rcx)

#define LOAD_ADD_16xFP16(beta, reg)\
	vmovups(mem(rcx), ymm2)\
	vcvtph2ps(ymm2, zmm2)\
	vfmadd231ps(zmm2, beta, reg)\
	add(r14, rcx)

#define STORE_3x16xFP32(reg0, reg1, reg2)\
	vmovups(reg0, mem(rcx))\
	vmovups(reg1, mem(rcx, r14, 1))\
	vmovups(reg2, mem(rcx, r14, 2))\
	add(r15, rcx)

#define PERMUTE_AND_SCALE_6x16xFP32(reg0, reg1, reg2, reg3, reg4, reg5)\
	movq(imm(0x5555), rdx)\
	kmovw(edx, k(1))\
	movq(imm(0xAAAA), rdx)\
	kmovw(edx, k(2))\
	vpermilps(imm(0xB1), reg1, reg1)\
	vpermilps(imm(0xB1), reg3, reg3)\
	vpermilps(imm(0xB1), reg5, reg5)\
	vblendmps(reg0, reg1, zmm2 mask_k(1))\
	vblendmps(reg0, reg1, reg1 mask_k(2))\
	vblendmps(reg2, reg3, zmm3 mask_k(1))\
	vblendmps(reg2, reg3, reg3 mask_k(2))\
	vblendmps(reg4, reg5, zmm4 mask_k(1))\
	vblendmps(reg4, reg5, reg5 mask_k(2))\
	vmulps(zmm1, zmm2, reg0)\
	vmulps(zmm1, reg1, reg1)\
	vmulps(zmm1, zmm3, reg2)\
	vmulps(zmm1, reg3, reg3)\
	vmulps(zmm1, zmm4, reg4)\
	vmulps(zmm1, reg5, reg5)

#define CONVERT_ACCUMULATORS_TO_FP16()\
	vcvtps2ph(imm(0x03), zmm8, ymm8)\
	vcvtps2ph(imm(0x03), zmm9, ymm9)\
	vcvtps2ph(imm(0x03), zmm10, ymm10)\
	vcvtps2ph(imm(0x03), zmm11, ymm11)\
	vcvtps2ph(imm(0x03), zmm12, ymm12)\
	vcvtps2ph(imm(0x03), zmm13, ymm13)\
	vcvtps2ph(imm(0x03), zmm14, ymm14)\
	vcvtps2ph(imm(0x03), zmm15, ymm15)\
	vcvtps2ph(imm(0x03), zmm16, ymm16)\
	vcvtps2ph(imm(0x03), zmm17, ymm17)\
	vcvtps2ph(imm(0x03), zmm18, ymm18)\
	vcvtps2ph(imm(0x03), zmm19, ymm19)\
	vcvtps2ph(imm(0x03), zmm20, ymm20)\
	vcvtps2ph(imm(0x03), zmm21, ymm21)\
	vcvtps2ph(imm(0x03), zmm22, ymm22)\
	vcvtps2ph(imm(0x03), zmm23, ymm23)\
	vcvtps2ph(imm(0x03), zmm24, ymm24)\
	vcvtps2ph(imm(0x03), zmm25, ymm25)\
	vcvtps2ph(imm(0x03), zmm26, ymm26)\
	vcvtps2ph(imm(0x03), zmm27, ymm27)\
	vcvtps2ph(imm(0x03), zmm28, ymm28)\
	vcvtps2ph(imm(0x03), zmm29, ymm29)\
	vcvtps2ph(imm(0x03), zmm30, ymm30)\
	vcvtps2ph(imm(0x03), zmm31, ymm31)

#define LOAD_ADD_3x16xFP16(beta, reg0, reg1, reg2)\
	vmovups(mem(rcx), ymm1)\
	vmovups(mem(rcx, r14, 1), ymm2)\
	vmovups(mem(rcx, r14, 2), ymm3)\
	vcvtph2ps(ymm1, zmm1)\
	vcvtph2ps(ymm2, zmm2)\
	vcvtph2ps(ymm3, zmm3)\
	vfmadd231ps(zmm1, beta, reg0)\
	vfmadd231ps(zmm2, beta, reg1)\
	vfmadd231ps(zmm3, beta, reg2)\
	add(r15, rcx)

#define STORE_3x16xFP16(reg0, reg1, reg2)\
	vmovups(reg0, mem(rcx))\
	vmovups(reg1, mem(rcx, r14, 1))\
	vmovups(reg2, mem(rcx, r14, 2))\
	add(r15, rcx)

#define RELU_24x16xFP32()\
	vxorps(zmm0, zmm0, zmm0)\
	vmaxps(zmm0, zmm8, zmm8)\
	vmaxps(zmm0, zmm9, zmm9)\
	vmaxps(zmm0, zmm10, zmm10)\
	vmaxps(zmm0, zmm11, zmm11)\
	vmaxps(zmm0, zmm12, zmm12)\
	vmaxps(zmm0, zmm13, zmm13)\
	vmaxps(zmm0, zmm14, zmm14)\
	vmaxps(zmm0, zmm15, zmm15)\
	vmaxps(zmm0, zmm16, zmm16)\
	vmaxps(zmm0, zmm17, zmm17)\
	vmaxps(zmm0, zmm18, zmm18)\
	vmaxps(zmm0, zmm19, zmm19)\
	vmaxps(zmm0, zmm20, zmm20)\
	vmaxps(zmm0, zmm21, zmm21)\
	vmaxps(zmm0, zmm22, zmm22)\
	vmaxps(zmm0, zmm23, zmm23)\
	vmaxps(zmm0, zmm24, zmm24)\
	vmaxps(zmm0, zmm25, zmm25)\
	vmaxps(zmm0, zmm26, zmm26)\
	vmaxps(zmm0, zmm27, zmm27)\
	vmaxps(zmm0, zmm28, zmm28)\
	vmaxps(zmm0, zmm29, zmm29)\
	vmaxps(zmm0, zmm30, zmm30)\
	vmaxps(zmm0, zmm31, zmm31)

namespace ml
{
	void gemm_avx512_24x16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			bool use_relu) noexcept
	{
		assert(A.dtype() == DTYPE_FLOAT32);
		assert(B.dtype() == DTYPE_FLOAT32);
		assert(C.dtype() == DTYPE_FLOAT32);
		assert(D.dtype() == DTYPE_FLOAT32);
		assert(A.rows() == B.rows());
		assert(A.stride() == 24);
		assert(B.stride() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), register_size<ZMM>()));

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float *C_ptr = C.data<float>();
		float *D_ptr = D.data<float>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float);
		const uint64_t D_stride = D.stride() * sizeof(float);
		const uint64_t flag_relu = use_relu;

		begin_asm()

		movq(var(A_ptr), rax)
		movq(var(B_ptr), rbx)
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14) // load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_24xFP32_16xFP32(0)
		SUB_KERNEL_24xFP32_16xFP32(1)
		SUB_KERNEL_24xFP32_16xFP32(2)
		SUB_KERNEL_24xFP32_16xFP32(3)

		add(imm(4*24*4), rax)// 4 iterations x 24 elements x 4 bytes
		add(imm(4*16*4), rbx)// 4 iterations x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_24xFP32_16xFP32(0)
		add(imm(1*24*4), rax)// 1 iteration x 24 elements x 4 bytes
		add(imm(1*16*4), rbx)// 1 iteration x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rax), zmm1)
		vbroadcastss(mem(rbx), zmm0)

		// permute back to original layout
		PERMUTE_AND_SCALE_6x16xFP32(zmm8, zmm9, zmm10, zmm11, zmm12, zmm13)
		PERMUTE_AND_SCALE_6x16xFP32(zmm14, zmm15, zmm16, zmm17, zmm18, zmm19)
		PERMUTE_AND_SCALE_6x16xFP32(zmm20, zmm21, zmm22, zmm23, zmm24, zmm25)
		PERMUTE_AND_SCALE_6x16xFP32(zmm26, zmm27, zmm28, zmm29, zmm30, zmm31)

		vxorps(zmm1, zmm1, zmm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(APPLY_RELU)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)

		LOAD_ADD_3x16xFP32(zmm0, zmm8, zmm9, zmm10)
		LOAD_ADD_3x16xFP32(zmm0, zmm11, zmm12, zmm13)
		LOAD_ADD_3x16xFP32(zmm0, zmm14, zmm15, zmm16)
		LOAD_ADD_3x16xFP32(zmm0, zmm17, zmm18, zmm19)
		LOAD_ADD_3x16xFP32(zmm0, zmm20, zmm21, zmm22)
		LOAD_ADD_3x16xFP32(zmm0, zmm23, zmm24, zmm25)
		LOAD_ADD_3x16xFP32(zmm0, zmm26, zmm27, zmm28)
		LOAD_ADD_3x16xFP32(zmm0, zmm29, zmm30, zmm31)

		label(APPLY_RELU)
		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(STORE_D)
		// apply ReLU case
		RELU_24x16xFP32()

		label(STORE_D)
		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*D_stride)

		STORE_3x16xFP32(zmm8, zmm9, zmm10)
		STORE_3x16xFP32(zmm11, zmm12, zmm13)
		STORE_3x16xFP32(zmm14, zmm15, zmm16)
		STORE_3x16xFP32(zmm17, zmm18, zmm19)
		STORE_3x16xFP32(zmm20, zmm21, zmm22)
		STORE_3x16xFP32(zmm23, zmm24, zmm25)
		STORE_3x16xFP32(zmm26, zmm27, zmm28)
		STORE_3x16xFP32(zmm29, zmm30, zmm31)

		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[A_ptr] "m"(A_ptr),
				[B_ptr] "m"(B_ptr),
				[C_ptr] "m"(C_ptr),
				[D_ptr] "m"(D_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[C_stride] "m"(C_stride),
				[D_stride] "m"(D_stride),
				[alpha_ptr] "m"(alpha_ptr),
				[beta_ptr] "m"(beta_ptr),
				[flag_relu] "m"(flag_relu)
				:// clobbers
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%rdx", "%r14", "%r15")
	}
	void gemm_avx512_24x16_fp32_fp16(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C, bool use_relu) noexcept
	{
		assert(A.dtype() == DTYPE_FLOAT32);
		assert(B.dtype() == DTYPE_FLOAT32);
		assert(C.dtype() == DTYPE_FLOAT16);
		assert(D.dtype() == DTYPE_FLOAT16);
		assert(A.rows() == B.rows());
		assert(A.stride() == 24);
		assert(B.stride() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), register_size<ZMM>()));

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float16 *C_ptr = C.data<float16>();
		float16 *D_ptr = D.data<float16>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float16);
		const uint64_t D_stride = D.stride() * sizeof(float16);
		const uint64_t flag_relu = use_relu;

		begin_asm()

		movq(var(A_ptr), rax)
		movq(var(B_ptr), rbx)
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14) // load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_24xFP32_16xFP32(0)
		SUB_KERNEL_24xFP32_16xFP32(1)
		SUB_KERNEL_24xFP32_16xFP32(2)
		SUB_KERNEL_24xFP32_16xFP32(3)

		add(imm(4*24*4), rax)// 4 iterations x 24 elements x 4 bytes
		add(imm(4*16*4), rbx)// 4 iterations x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_24xFP32_16xFP32(0)
		add(imm(1*24*4), rax)// 1 iteration x 24 elements x 4 bytes
		add(imm(1*16*4), rbx)// 1 iteration x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rax), zmm1)
		vbroadcastss(mem(rbx), zmm0)

		// permute back to original layout
		PERMUTE_AND_SCALE_6x16xFP32(zmm8, zmm9, zmm10, zmm11, zmm12, zmm13)
		PERMUTE_AND_SCALE_6x16xFP32(zmm14, zmm15, zmm16, zmm17, zmm18, zmm19)
		PERMUTE_AND_SCALE_6x16xFP32(zmm20, zmm21, zmm22, zmm23, zmm24, zmm25)
		PERMUTE_AND_SCALE_6x16xFP32(zmm26, zmm27, zmm28, zmm29, zmm30, zmm31)

		vxorps(zmm1, zmm1, zmm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(APPLY_RELU)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)

		LOAD_ADD_3x16xFP16(zmm0, zmm8, zmm9, zmm10)
		LOAD_ADD_3x16xFP16(zmm0, zmm11, zmm12, zmm13)
		LOAD_ADD_3x16xFP16(zmm0, zmm14, zmm15, zmm16)
		LOAD_ADD_3x16xFP16(zmm0, zmm17, zmm18, zmm19)
		LOAD_ADD_3x16xFP16(zmm0, zmm20, zmm21, zmm22)
		LOAD_ADD_3x16xFP16(zmm0, zmm23, zmm24, zmm25)
		LOAD_ADD_3x16xFP16(zmm0, zmm26, zmm27, zmm28)
		LOAD_ADD_3x16xFP16(zmm0, zmm29, zmm30, zmm31)

		label(APPLY_RELU)
		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(STORE_D)
		// apply ReLU case
		RELU_24x16xFP32()

		label(STORE_D)
		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*D_stride)

		CONVERT_ACCUMULATORS_TO_FP16()

		STORE_3x16xFP16(ymm8, ymm9, ymm10)
		STORE_3x16xFP16(ymm11, ymm12, ymm13)
		STORE_3x16xFP16(ymm14, ymm15, ymm16)
		STORE_3x16xFP16(ymm17, ymm18, ymm19)
		STORE_3x16xFP16(ymm20, ymm21, ymm22)
		STORE_3x16xFP16(ymm23, ymm24, ymm25)
		STORE_3x16xFP16(ymm26, ymm27, ymm28)
		STORE_3x16xFP16(ymm29, ymm30, ymm31)

		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[A_ptr] "m"(A_ptr),
				[B_ptr] "m"(B_ptr),
				[C_ptr] "m"(C_ptr),
				[D_ptr] "m"(D_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[C_stride] "m"(C_stride),
				[D_stride] "m"(D_stride),
				[alpha_ptr] "m"(alpha_ptr),
				[beta_ptr] "m"(beta_ptr),
				[flag_relu] "m"(flag_relu)
				:// clobbers
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%rdx", "%r14", "%r15")
	}

	void pack_avx512_24xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 24);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

		const uint64_t k_iter = dst.rows() / 16;
		const uint64_t k_left = dst.rows() % 16;
		const uint64_t src_stride = src.stride() * sizeof(float);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			vmovups(mem(rax, 0*4), zmm0)
			vmovups(mem(rax, 16*4), ymm1)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm2)
			vmovups(mem(rax, 16*4), ymm3)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm4)
			vmovups(mem(rax, 16*4), ymm5)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm6)
			vmovups(mem(rax, 16*4), ymm7)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm8)
			vmovups(mem(rax, 16*4), ymm9)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm10)
			vmovups(mem(rax, 16*4), ymm11)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm12)
			vmovups(mem(rax, 16*4), ymm13)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm14)
			vmovups(mem(rax, 16*4), ymm15)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm16)
			vmovups(mem(rax, 16*4), ymm17)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm18)
			vmovups(mem(rax, 16*4), ymm19)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm20)
			vmovups(mem(rax, 16*4), ymm21)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm22)
			vmovups(mem(rax, 16*4), ymm23)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm24)
			vmovups(mem(rax, 16*4), ymm25)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm26)
			vmovups(mem(rax, 16*4), ymm27)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm28)
			vmovups(mem(rax, 16*4), ymm29)
			add(r12, rax)// add stride to src pointer
			vmovups(mem(rax, 0*4), zmm30)
			vmovups(mem(rax, 16*4), ymm31)
			add(r12, rax)// add stride to src pointer

			vmovups(zmm0, mem(rbx, (0*24+0)*4))
			vmovaps(ymm1, mem(rbx, (0*24+16)*4))
			vmovups(zmm2, mem(rbx, (1*24+0)*4))
			vmovaps(ymm3, mem(rbx, (1*24+16)*4))
			vmovups(zmm4, mem(rbx, (2*24+0)*4))
			vmovaps(ymm5, mem(rbx, (2*24+16)*4))
			vmovups(zmm6, mem(rbx, (3*24+0)*4))
			vmovaps(ymm7, mem(rbx, (3*24+16)*4))
			vmovups(zmm8, mem(rbx, (4*24+0)*4))
			vmovaps(ymm9, mem(rbx, (4*24+16)*4))
			vmovups(zmm10, mem(rbx, (5*24+0)*4))
			vmovaps(ymm11, mem(rbx, (5*24+16)*4))
			vmovups(zmm12, mem(rbx, (6*24+0)*4))
			vmovaps(ymm13, mem(rbx, (6*24+16)*4))
			vmovups(zmm14, mem(rbx, (7*24+0)*4))
			vmovaps(ymm15, mem(rbx, (7*24+16)*4))
			vmovups(zmm16, mem(rbx, (8*24+0)*4))
			vmovaps(ymm17, mem(rbx, (8*24+16)*4))
			vmovups(zmm18, mem(rbx, (9*24+0)*4))
			vmovaps(ymm19, mem(rbx, (9*24+16)*4))
			vmovups(zmm20, mem(rbx, (10*24+0)*4))
			vmovaps(ymm21, mem(rbx, (10*24+16)*4))
			vmovups(zmm22, mem(rbx, (11*24+0)*4))
			vmovaps(ymm23, mem(rbx, (11*24+16)*4))
			vmovups(zmm24, mem(rbx, (12*24+0)*4))
			vmovaps(ymm25, mem(rbx, (12*24+16)*4))
			vmovups(zmm26, mem(rbx, (13*24+0)*4))
			vmovaps(ymm27, mem(rbx, (13*24+16)*4))
			vmovups(zmm28, mem(rbx, (14*24+0)*4))
			vmovaps(ymm29, mem(rbx, (14*24+16)*4))
			vmovups(zmm30, mem(rbx, (15*24+0)*4))
			vmovaps(ymm31, mem(rbx, (15*24+16)*4))

			add(imm(4*16*24), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			vmovups(mem(rax, 0), zmm0)
			vmovups(mem(rax, 16*4), ymm1)
			add(r12, rax)// add stride to src pointer
			vmovups(zmm0, mem(rbx, 0))
			vmovaps(ymm1, mem(rbx, 16*4))
			add(imm(4*1*24), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED1)

			label(EPILOGUE)
			vzeroupper()

			end_asm(:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
					"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx",
					"%r12", "%r14")
		}
		else
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			// first 16x16 tile
			movq(rax, r13)// tmp src pointer is in r13

			vmovups(mem(r13), zmm0)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm1)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm2)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm3)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm4)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm5)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm6)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm7)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm8)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm9)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm10)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm11)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm12)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm13)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm14)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm15)
			add(r12, r13)// add stride to src pointer

			// transpose 16x16
			// first shuffle
			vunpcklps(zmm1, zmm0, zmm16)
			vunpckhps(zmm1, zmm0, zmm17)
			vunpcklps(zmm3, zmm2, zmm18)
			vunpckhps(zmm3, zmm2, zmm19)
			vunpcklps(zmm5, zmm4, zmm20)
			vunpckhps(zmm5, zmm4, zmm21)
			vunpcklps(zmm7, zmm6, zmm22)
			vunpckhps(zmm7, zmm6, zmm23)
			vunpcklps(zmm9, zmm8, zmm24)
			vunpckhps(zmm9, zmm8, zmm25)
			vunpcklps(zmm11, zmm10, zmm26)
			vunpckhps(zmm11, zmm10, zmm27)
			vunpcklps(zmm13, zmm12, zmm28)
			vunpckhps(zmm13, zmm12, zmm29)
			vunpcklps(zmm15, zmm14, zmm30)
			vunpckhps(zmm15, zmm14, zmm31)

			// second shuffle
			vunpcklpd(zmm18, zmm16, zmm0)
			vunpckhpd(zmm18, zmm16, zmm1)
			vunpcklpd(zmm19, zmm17, zmm2)
			vunpckhpd(zmm19, zmm17, zmm3)
			vunpcklpd(zmm22, zmm20, zmm4)
			vunpckhpd(zmm22, zmm20, zmm5)
			vunpcklpd(zmm23, zmm21, zmm6)
			vunpckhpd(zmm23, zmm21, zmm7)
			vunpcklpd(zmm26, zmm24, zmm8)
			vunpckhpd(zmm26, zmm24, zmm9)
			vunpcklpd(zmm27, zmm25, zmm10)
			vunpckhpd(zmm27, zmm25, zmm11)
			vunpcklpd(zmm30, zmm28, zmm12)
			vunpckhpd(zmm30, zmm28, zmm13)
			vunpcklpd(zmm31, zmm29, zmm14)
			vunpckhpd(zmm31, zmm29, zmm15)

			// third shuffle
			vshuff32x4(imm(0x88), zmm4, zmm0, zmm16)
			vshuff32x4(imm(0x88), zmm5, zmm1, zmm17)
			vshuff32x4(imm(0x88), zmm6, zmm2, zmm18)
			vshuff32x4(imm(0x88), zmm7, zmm3, zmm19)
			vshuff32x4(imm(0xDD), zmm4, zmm0, zmm20)
			vshuff32x4(imm(0xDD), zmm5, zmm1, zmm21)
			vshuff32x4(imm(0xDD), zmm6, zmm2, zmm22)
			vshuff32x4(imm(0xDD), zmm7, zmm3, zmm23)
			vshuff32x4(imm(0x88), zmm12, zmm8, zmm24)
			vshuff32x4(imm(0x88), zmm13, zmm9, zmm25)
			vshuff32x4(imm(0x88), zmm14, zmm10, zmm26)
			vshuff32x4(imm(0x88), zmm15, zmm11, zmm27)
			vshuff32x4(imm(0xDD), zmm12, zmm8, zmm28)
			vshuff32x4(imm(0xDD), zmm13, zmm9, zmm29)
			vshuff32x4(imm(0xDD), zmm14, zmm10, zmm30)
			vshuff32x4(imm(0xDD), zmm15, zmm11, zmm31)

			// fourth shuffle
			vshuff32x4(imm(0x88), zmm24, zmm16, zmm0)
			vshuff32x4(imm(0x88), zmm25, zmm17, zmm1)
			vshuff32x4(imm(0x88), zmm26, zmm18, zmm2)
			vshuff32x4(imm(0x88), zmm27, zmm19, zmm3)
			vshuff32x4(imm(0x88), zmm28, zmm20, zmm4)
			vshuff32x4(imm(0x88), zmm29, zmm21, zmm5)
			vshuff32x4(imm(0x88), zmm30, zmm22, zmm6)
			vshuff32x4(imm(0x88), zmm31, zmm23, zmm7)
			vshuff32x4(imm(0xDD), zmm24, zmm16, zmm8)
			vshuff32x4(imm(0xDD), zmm25, zmm17, zmm9)
			vshuff32x4(imm(0xDD), zmm26, zmm18, zmm10)
			vshuff32x4(imm(0xDD), zmm27, zmm19, zmm11)
			vshuff32x4(imm(0xDD), zmm28, zmm20, zmm12)
			vshuff32x4(imm(0xDD), zmm29, zmm21, zmm13)
			vshuff32x4(imm(0xDD), zmm30, zmm22, zmm14)
			vshuff32x4(imm(0xDD), zmm31, zmm23, zmm15)

			vmovups(zmm0, mem(rbx, 0*24*4))
			vmovups(zmm1, mem(rbx, 1*24*4))
			vmovups(zmm2, mem(rbx, 2*24*4))
			vmovups(zmm3, mem(rbx, 3*24*4))
			vmovups(zmm4, mem(rbx, 4*24*4))
			vmovups(zmm5, mem(rbx, 5*24*4))
			vmovups(zmm6, mem(rbx, 6*24*4))
			vmovups(zmm7, mem(rbx, 7*24*4))
			vmovups(zmm8, mem(rbx, 8*24*4))
			vmovups(zmm9, mem(rbx, 9*24*4))
			vmovups(zmm10, mem(rbx, 10*24*4))
			vmovups(zmm11, mem(rbx, 11*24*4))
			vmovups(zmm12, mem(rbx, 12*24*4))
			vmovups(zmm13, mem(rbx, 13*24*4))
			vmovups(zmm14, mem(rbx, 14*24*4))
			vmovups(zmm15, mem(rbx, 15*24*4))

			// second 8x16 tile
			vmovups(mem(r13), zmm0)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm1)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm2)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm3)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm4)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm5)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm6)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), zmm7)
			add(r12, r13)// add stride to src pointer

			// transpose 16x16
			// first shuffle
			vunpcklps(zmm1, zmm0, zmm16)
			vunpckhps(zmm1, zmm0, zmm17)
			vunpcklps(zmm3, zmm2, zmm18)
			vunpckhps(zmm3, zmm2, zmm19)
			vunpcklps(zmm5, zmm4, zmm20)
			vunpckhps(zmm5, zmm4, zmm21)
			vunpcklps(zmm7, zmm6, zmm22)
			vunpckhps(zmm7, zmm6, zmm23)

			// second shuffle
			vunpcklpd(zmm18, zmm16, zmm0)
			vunpckhpd(zmm18, zmm16, zmm1)
			vunpcklpd(zmm19, zmm17, zmm2)
			vunpckhpd(zmm19, zmm17, zmm3)
			vunpcklpd(zmm22, zmm20, zmm4)
			vunpckhpd(zmm22, zmm20, zmm5)
			vunpcklpd(zmm23, zmm21, zmm6)
			vunpckhpd(zmm23, zmm21, zmm7)

			// third shuffle
			vshuff32x4(imm(0x88), zmm4, zmm0, zmm16)
			vshuff32x4(imm(0x88), zmm5, zmm1, zmm17)
			vshuff32x4(imm(0x88), zmm6, zmm2, zmm18)
			vshuff32x4(imm(0x88), zmm7, zmm3, zmm19)
			vshuff32x4(imm(0xDD), zmm4, zmm0, zmm20)
			vshuff32x4(imm(0xDD), zmm5, zmm1, zmm21)
			vshuff32x4(imm(0xDD), zmm6, zmm2, zmm22)
			vshuff32x4(imm(0xDD), zmm7, zmm3, zmm23)

			vextractf32x8(imm(0x0), zmm16, ymm0)
			vextractf32x8(imm(0x1), zmm16, ymm1)
			vextractf32x8(imm(0x0), zmm17, ymm2)
			vextractf32x8(imm(0x1), zmm17, ymm3)
			vextractf32x8(imm(0x0), zmm18, ymm4)
			vextractf32x8(imm(0x1), zmm18, ymm5)
			vextractf32x8(imm(0x0), zmm19, ymm6)
			vextractf32x8(imm(0x1), zmm19, ymm7)
			vextractf32x8(imm(0x0), zmm20, ymm8)
			vextractf32x8(imm(0x1), zmm20, ymm9)
			vextractf32x8(imm(0x0), zmm21, ymm10)
			vextractf32x8(imm(0x1), zmm21, ymm11)
			vextractf32x8(imm(0x0), zmm22, ymm12)
			vextractf32x8(imm(0x1), zmm22, ymm13)
			vextractf32x8(imm(0x0), zmm23, ymm14)
			vextractf32x8(imm(0x1), zmm23, ymm15)

			vmovaps(ymm0, mem(rbx, (0*24+16)*4))
			vmovaps(ymm1, mem(rbx, (1*24+16)*4))
			vmovaps(ymm2, mem(rbx, (2*24+16)*4))
			vmovaps(ymm3, mem(rbx, (3*24+16)*4))
			vmovaps(ymm4, mem(rbx, (4*24+16)*4))
			vmovaps(ymm5, mem(rbx, (5*24+16)*4))
			vmovaps(ymm6, mem(rbx, (6*24+16)*4))
			vmovaps(ymm7, mem(rbx, (7*24+16)*4))
			vmovaps(ymm8, mem(rbx, (8*24+16)*4))
			vmovaps(ymm9, mem(rbx, (9*24+16)*4))
			vmovaps(ymm10, mem(rbx, (10*24+16)*4))
			vmovaps(ymm11, mem(rbx, (11*24+16)*4))
			vmovaps(ymm12, mem(rbx, (12*24+16)*4))
			vmovaps(ymm13, mem(rbx, (13*24+16)*4))
			vmovaps(ymm14, mem(rbx, (14*24+16)*4))
			vmovaps(ymm15, mem(rbx, (15*24+16)*4))

			add(imm(4*16), rax)// add stride to src pointer
			add(imm(4*24*16), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(rax, r13)// tmp src pointer is in r13

			vmovss(mem(r13), xmm0)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm1)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm2)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm3)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm4)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm5)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm6)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm7)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm8)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm9)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm10)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm11)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm12)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm13)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm14)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm15)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm16)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm17)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm18)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm19)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm20)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm21)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm22)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm23)
			add(r12, r13)// add stride to src pointer

			vmovss(xmm0, mem(rbx, 4*0))
			vmovss(xmm1, mem(rbx, 4*1))
			vmovss(xmm2, mem(rbx, 4*2))
			vmovss(xmm3, mem(rbx, 4*3))
			vmovss(xmm4, mem(rbx, 4*4))
			vmovss(xmm5, mem(rbx, 4*5))
			vmovss(xmm6, mem(rbx, 4*6))
			vmovss(xmm7, mem(rbx, 4*7))
			vmovss(xmm8, mem(rbx, 4*8))
			vmovss(xmm9, mem(rbx, 4*9))
			vmovss(xmm10, mem(rbx, 4*10))
			vmovss(xmm11, mem(rbx, 4*11))
			vmovss(xmm12, mem(rbx, 4*12))
			vmovss(xmm13, mem(rbx, 4*13))
			vmovss(xmm14, mem(rbx, 4*14))
			vmovss(xmm15, mem(rbx, 4*15))
			vmovss(xmm16, mem(rbx, 4*16))
			vmovss(xmm17, mem(rbx, 4*17))
			vmovss(xmm18, mem(rbx, 4*18))
			vmovss(xmm19, mem(rbx, 4*19))
			vmovss(xmm20, mem(rbx, 4*20))
			vmovss(xmm21, mem(rbx, 4*21))
			vmovss(xmm22, mem(rbx, 4*22))
			vmovss(xmm23, mem(rbx, 4*23))

			add(imm(4*1), rax)// add stride to src pointer
			add(imm(4*24*1), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED1)

			label(EPILOGUE)
			vzeroupper()
			end_asm(:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
					"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx",
					"%r12", "%r13", "%r14")
		}
	}
//	void pack_avx512_24xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
//	{
//		assert(dst.stride() == 12);
//		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));
//
//		uint64_t k_iter = dst.rows() / 8;
//		uint64_t k_left = dst.rows() % 8;
//		const uint64_t src_stride = src.stride() * sizeof(float16);
//		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
//		void *dst_ptr = dst.data();
//
//		if (src_op == MatrixOp::NORMAL)
//		{
//			asm volatile(
//					"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
//					"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
//					"movq %[src_stride], %%r12 \n\t"// src stride is in r12
//
//					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
//					"test %%r14, %%r14 \n\t"
//					"je FINALLOOP%= \n\t"
//
//					"UNROLLED8%=: \n\t"
//					"vmovups 0x00(%%rax), %%xmm0 \n\t"
//					"vmovsd 0x10(%%rax), %%xmm1 \n\t"
//					"add %%r12, %%rax \n\t"// add stride to src pointer
//					"vmovups 0x00(%%rax), %%xmm2 \n\t"
//					"vmovsd 0x10(%%rax), %%xmm3 \n\t"
//					"add %%r12, %%rax \n\t"// add stride to src pointer
//					"vmovups 0x00(%%rax), %%xmm4 \n\t"
//					"vmovsd 0x10(%%rax), %%xmm5 \n\t"
//					"add %%r12, %%rax \n\t"// add stride to src pointer
//					"vmovups 0x00(%%rax), %%xmm6 \n\t"
//					"vmovsd 0x10(%%rax), %%xmm7 \n\t"
//					"add %%r12, %%rax \n\t"// add stride to src pointer
//					"vmovups 0x00(%%rax), %%xmm8 \n\t"
//					"vmovsd 0x10(%%rax), %%xmm9 \n\t"
//					"add %%r12, %%rax \n\t"// add stride to src pointer
//					"vmovups 0x00(%%rax), %%xmm10 \n\t"
//					"vmovsd 0x10(%%rax), %%xmm11 \n\t"
//					"add %%r12, %%rax \n\t"// add stride to src pointer
//					"vmovups 0x00(%%rax), %%xmm12 \n\t"
//					"vmovsd 0x10(%%rax), %%xmm13 \n\t"
//					"add %%r12, %%rax \n\t"// add stride to src pointer
//					"vmovups 0x00(%%rax), %%xmm14 \n\t"
//					"vmovsd 0x10(%%rax), %%xmm15 \n\t"
//					"add %%r12, %%rax \n\t"// add stride to src pointer
//
//					"vcvtph2ps %%xmm0, %%zmm0 \n\t"
//					"vcvtph2ps %%xmm1, %%xmm1 \n\t"
//					"vcvtph2ps %%xmm2, %%zmm2 \n\t"
//					"vcvtph2ps %%xmm3, %%xmm3 \n\t"
//					"vcvtph2ps %%xmm4, %%zmm4 \n\t"
//					"vcvtph2ps %%xmm5, %%xmm5 \n\t"
//					"vcvtph2ps %%xmm6, %%zmm6 \n\t"
//					"vcvtph2ps %%xmm7, %%xmm7 \n\t"
//					"vcvtph2ps %%xmm8, %%zmm8 \n\t"
//					"vcvtph2ps %%xmm9, %%xmm9 \n\t"
//					"vcvtph2ps %%xmm10, %%zmm10 \n\t"
//					"vcvtph2ps %%xmm11, %%xmm11 \n\t"
//					"vcvtph2ps %%xmm12, %%zmm12 \n\t"
//					"vcvtph2ps %%xmm13, %%xmm13 \n\t"
//					"vcvtph2ps %%xmm14, %%zmm14 \n\t"
//					"vcvtph2ps %%xmm15, %%xmm15 \n\t"
//
//					"vmovups %%zmm0, ((0*12+0)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm1, ((0*12+8)*4)(%%rbx) \n\t"
//					"vmovups %%zmm2, ((1*12+0)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm3, ((1*12+8)*4)(%%rbx) \n\t"
//					"vmovups %%zmm4, ((2*12+0)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm5, ((2*12+8)*4)(%%rbx) \n\t"
//					"vmovups %%zmm6, ((3*12+0)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm7, ((3*12+8)*4)(%%rbx) \n\t"
//					"vmovups %%zmm8, ((4*12+0)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm9, ((4*12+8)*4)(%%rbx) \n\t"
//					"vmovups %%zmm10, ((5*12+0)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm11, ((5*12+8)*4)(%%rbx) \n\t"
//					"vmovups %%zmm12, ((6*12+0)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm13, ((6*12+8)*4)(%%rbx) \n\t"
//					"vmovups %%zmm14, ((7*12+0)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm15, ((7*12+8)*4)(%%rbx) \n\t"
//
//					"add $(4*8*12), %%rbx \n\t"// add stride to dst pointer
//
//					"dec %%r14 \n\t"
//					"jne UNROLLED8%= \n\t"
//
//					"FINALLOOP%=: \n\t"
//					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
//					"test %%r14, %%r14 \n\t"
//					"je EPILOGUE%= \n\t"
//
//					"UNROLLED1%=: \n\t"
//					"vmovups 0x00(%%rax), %%xmm0 \n\t"
//					"vmovsd 0x10(%%rax), %%xmm1 \n\t"
//					"add %%r12, %%rax \n\t"// add stride to src pointer
//					"vcvtph2ps %%xmm0, %%zmm0 \n\t"
//					"vcvtph2ps %%xmm1, %%xmm1 \n\t"
//					"vmovups %%zmm0, 0x00(%%rbx) \n\t"
//					"vmovaps %%xmm1, 0x20(%%rbx) \n\t"
//					"add $(4*1*12), %%rbx \n\t"// add stride to dst pointer
//
//					"dec %%r14 \n\t"
//					"jne UNROLLED1%= \n\t"
//
//					"EPILOGUE%=: \n\t"
//					"vzeroupper \n\t"
//
//					:// outputs
//					:// inputs
//					[src_ptr] "m"(src_ptr),
//					[dst_ptr] "m"(dst_ptr),
//					[k_iter] "m"(k_iter),
//					[k_left] "m"(k_left),
//					[src_stride] "m"(src_stride)
//					:// clobbers
//					"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
//					"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx",
//					"%r12", "%r14");
//		}
//		else
//		{
//			asm volatile(
//					"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
//					"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
//					"movq %[src_stride], %%r12 \n\t"// src stride is in r12
//
//					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
//					"test %%r14, %%r14 \n\t"
//					"je FINALLOOP%= \n\t"
//
//					"UNROLLED8%=: \n\t"
//					// first 8x8 tile
//					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13
//
//					"vmovups 0x0(%%r13), %%xmm0 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm1 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm2 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm3 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm4 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm5 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm6 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm7 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//
//					"vcvtph2ps %%xmm0, %%zmm0 \n\t"
//					"vcvtph2ps %%xmm1, %%zmm1 \n\t"
//					"vcvtph2ps %%xmm2, %%zmm2 \n\t"
//					"vcvtph2ps %%xmm3, %%zmm3 \n\t"
//					"vcvtph2ps %%xmm4, %%zmm4 \n\t"
//					"vcvtph2ps %%xmm5, %%zmm5 \n\t"
//					"vcvtph2ps %%xmm6, %%zmm6 \n\t"
//					"vcvtph2ps %%xmm7, %%zmm7 \n\t"
//
//					// transpose 8x8
//					// first shuffle
//					"vunpcklps %%zmm1, %%zmm0, %%zmm8 \n\t"
//					"vunpckhps %%zmm1, %%zmm0, %%zmm9 \n\t"
//					"vunpcklps %%zmm3, %%zmm2, %%zmm10 \n\t"
//					"vunpckhps %%zmm3, %%zmm2, %%zmm11 \n\t"
//					"vunpcklps %%zmm5, %%zmm4, %%zmm12 \n\t"
//					"vunpckhps %%zmm5, %%zmm4, %%zmm13 \n\t"
//					"vunpcklps %%zmm7, %%zmm6, %%zmm14 \n\t"
//					"vunpckhps %%zmm7, %%zmm6, %%zmm15 \n\t"
//
//					// second shuffle
//					"vunpcklpd %%zmm10, %%zmm8, %%zmm0 \n\t"
//					"vunpckhpd %%zmm10, %%zmm8, %%zmm1 \n\t"
//					"vunpcklpd %%zmm11, %%zmm9, %%zmm2 \n\t"
//					"vunpckhpd %%zmm11, %%zmm9, %%zmm3 \n\t"
//					"vunpcklpd %%zmm14, %%zmm12, %%zmm4 \n\t"
//					"vunpckhpd %%zmm14, %%zmm12, %%zmm5 \n\t"
//					"vunpcklpd %%zmm15, %%zmm13, %%zmm6 \n\t"
//					"vunpckhpd %%zmm15, %%zmm13, %%zmm7 \n\t"
//
//					// third shuffle
//					"vperm2f128 $0x20, %%zmm4, %%zmm0, %%zmm8 \n\t"
//					"vperm2f128 $0x20, %%zmm5, %%zmm1, %%zmm9 \n\t"
//					"vperm2f128 $0x20, %%zmm6, %%zmm2, %%zmm10 \n\t"
//					"vperm2f128 $0x20, %%zmm7, %%zmm3, %%zmm11 \n\t"
//					"vperm2f128 $0x31, %%zmm4, %%zmm0, %%zmm12 \n\t"
//					"vperm2f128 $0x31, %%zmm5, %%zmm1, %%zmm13 \n\t"
//					"vperm2f128 $0x31, %%zmm6, %%zmm2, %%zmm14 \n\t"
//					"vperm2f128 $0x31, %%zmm7, %%zmm3, %%zmm15 \n\t"
//
//					"vmovups %%zmm8, (0*12*4)(%%rbx) \n\t"
//					"vmovups %%zmm9, (1*12*4)(%%rbx) \n\t"
//					"vmovups %%zmm10, (2*12*4)(%%rbx) \n\t"
//					"vmovups %%zmm11, (3*12*4)(%%rbx) \n\t"
//					"vmovups %%zmm12, (4*12*4)(%%rbx) \n\t"
//					"vmovups %%zmm13, (5*12*4)(%%rbx) \n\t"
//					"vmovups %%zmm14, (6*12*4)(%%rbx) \n\t"
//					"vmovups %%zmm15, (7*12*4)(%%rbx) \n\t"
//
//					// second 4x8 tile
//					"vmovups 0x0(%%r13), %%xmm0 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm1 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm2 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%xmm3 \n\t"
//
//					"vcvtph2ps %%xmm0, %%zmm0 \n\t"
//					"vcvtph2ps %%xmm1, %%zmm1 \n\t"
//					"vcvtph2ps %%xmm2, %%zmm2 \n\t"
//					"vcvtph2ps %%xmm3, %%zmm3 \n\t"
//
//					// transpose 4x8
//					// first shuffle
//					"vunpcklps %%zmm1, %%zmm0, %%zmm8 \n\t"
//					"vunpckhps %%zmm1, %%zmm0, %%zmm9 \n\t"
//					"vunpcklps %%zmm3, %%zmm2, %%zmm10 \n\t"
//					"vunpckhps %%zmm3, %%zmm2, %%zmm11 \n\t"
//
//					// second shuffle
//					"vunpcklpd %%zmm10, %%zmm8, %%zmm0 \n\t"
//					"vunpckhpd %%zmm10, %%zmm8, %%zmm1 \n\t"
//					"vunpcklpd %%zmm11, %%zmm9, %%zmm2 \n\t"
//					"vunpckhpd %%zmm11, %%zmm9, %%zmm3 \n\t"
//
//					// third shuffle
//					"vextractf128 $0x1, %%zmm0, %%xmm4 \n\t"
//					"vextractf128 $0x1, %%zmm1, %%xmm5 \n\t"
//					"vextractf128 $0x1, %%zmm2, %%xmm6 \n\t"
//					"vextractf128 $0x1, %%zmm3, %%xmm7 \n\t"
//
//					"vmovaps %%xmm0, ((0*12+8)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm1, ((1*12+8)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm2, ((2*12+8)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm3, ((3*12+8)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm4, ((4*12+8)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm5, ((5*12+8)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm6, ((6*12+8)*4)(%%rbx) \n\t"
//					"vmovaps %%xmm7, ((7*12+8)*4)(%%rbx) \n\t"
//
//					"add $(2*8), %%rax \n\t"// add stride to src pointer
//					"add $(4*12*8), %%rbx \n\t"// add stride to dst pointer
//
//					"dec %%r14 \n\t"
//					"jne UNROLLED8%= \n\t"
//
//					"FINALLOOP%=: \n\t"
//					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
//					"test %%r14, %%r14 \n\t"
//					"je EPILOGUE%= \n\t"
//
//					"UNROLLED1%=: \n\t"
//					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13
//
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm0 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm1 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm2 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm3 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm4 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm5 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm6 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm7 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm8 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm9 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm10 \n\t"
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"movzw 0x0(%%r13), %%rcx \n\t"
//					"vmovq %%rcx, %%xmm11 \n\t"
//
//					"vcvtph2ps %%xmm0, %%xmm0 \n\t"
//					"vcvtph2ps %%xmm1, %%xmm1 \n\t"
//					"vcvtph2ps %%xmm2, %%xmm2 \n\t"
//					"vcvtph2ps %%xmm3, %%xmm3 \n\t"
//					"vcvtph2ps %%xmm4, %%xmm4 \n\t"
//					"vcvtph2ps %%xmm5, %%xmm5 \n\t"
//					"vcvtph2ps %%xmm6, %%xmm6 \n\t"
//					"vcvtph2ps %%xmm7, %%xmm7 \n\t"
//					"vcvtph2ps %%xmm8, %%xmm8 \n\t"
//					"vcvtph2ps %%xmm9, %%xmm9 \n\t"
//					"vcvtph2ps %%xmm10, %%xmm10 \n\t"
//					"vcvtph2ps %%xmm11, %%xmm11 \n\t"
//
//					"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
//					"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
//					"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
//					"vmovss %%xmm3, (4*3)(%%rbx) \n\t"
//					"vmovss %%xmm4, (4*4)(%%rbx) \n\t"
//					"vmovss %%xmm5, (4*5)(%%rbx) \n\t"
//					"vmovss %%xmm6, (4*6)(%%rbx) \n\t"
//					"vmovss %%xmm7, (4*7)(%%rbx) \n\t"
//					"vmovss %%xmm8, (4*8)(%%rbx) \n\t"
//					"vmovss %%xmm9, (4*9)(%%rbx) \n\t"
//					"vmovss %%xmm10, (4*10)(%%rbx) \n\t"
//					"vmovss %%xmm11, (4*11)(%%rbx) \n\t"
//
//					"add $(2*1), %%rax \n\t"// add stride to src pointer
//					"add $(4*12*1), %%rbx \n\t"// add stride to dst pointer
//
//					"dec %%r14 \n\t"
//					"jne UNROLLED1%= \n\t"
//
//					"EPILOGUE%=: \n\t"
//					"vzeroupper \n\t"
//
//					:// outputs
//					:// inputs
//					[src_ptr] "m"(src_ptr),
//					[dst_ptr] "m"(dst_ptr),
//					[k_iter] "m"(k_iter),
//					[k_left] "m"(k_left),
//					[src_stride] "m"(src_stride)
//					:// clobbers
//					"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
//					"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx",
//					"%r12", "%r13", "%r14");
//		}
//	}

} /* namespace ml */

