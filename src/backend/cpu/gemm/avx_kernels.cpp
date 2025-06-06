/*
 * avx_gemm_kernels.cpp
 *
 *  Created on: May 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include "gemm_kernels.hpp"
#include "Fragment.hpp"
#include "Matrix.hpp"
#include "../utils.hpp"
#include "../fp16.hpp"

#include <cinttypes>
#include <cassert>

#include "../src/backend/cpu/assembly_macros.hpp"
#include "common_operations.hpp"

#define ZERO_ACCUMULATORS()\
	vxorps(ymm6, ymm6, ymm6)\
	vxorps(ymm7, ymm7, ymm7)\
	vxorps(ymm8, ymm8, ymm8)\
	vxorps(ymm9, ymm9, ymm9)\
	vxorps(ymm10, ymm10, ymm10)\
	vxorps(ymm11, ymm11, ymm11)\
	vxorps(ymm12, ymm12, ymm12)\
	vxorps(ymm13, ymm13, ymm13)\
	vxorps(ymm14, ymm14, ymm14)\
	vxorps(ymm15, ymm15, ymm15)\


#define SUB_KERNEL_10xFP32_8xFP32(n) \
	vmovaps(mem(rbx, n*8*4), ymm0)\
	vbroadcastss(mem(rax, (10*n+0)*4), ymm1)\
	vbroadcastss(mem(rax, (10*n+1)*4), ymm2)\
	vbroadcastss(mem(rax, (10*n+2)*4), ymm3)\
	vbroadcastss(mem(rax, (10*n+3)*4), ymm4)\
	vbroadcastss(mem(rax, (10*n+4)*4), ymm5)\
	vmulps(ymm1, ymm0, ymm1)\
	vmulps(ymm2, ymm0, ymm2)\
	vmulps(ymm3, ymm0, ymm3)\
	vmulps(ymm4, ymm0, ymm4)\
	vmulps(ymm5, ymm0, ymm5)\
	vaddps(ymm1, ymm6, ymm6)\
	vaddps(ymm2, ymm7, ymm7)\
	vaddps(ymm3, ymm8, ymm8)\
	vaddps(ymm4, ymm9, ymm9)\
	vaddps(ymm5, ymm10, ymm10)\
	vbroadcastss(mem(rax, (10*n+5)*4), ymm1)\
	vbroadcastss(mem(rax, (10*n+6)*4), ymm2)\
	vbroadcastss(mem(rax, (10*n+7)*4), ymm3)\
	vbroadcastss(mem(rax, (10*n+8)*4), ymm4)\
	vbroadcastss(mem(rax, (10*n+9)*4), ymm5)\
	vmulps(ymm1, ymm0, ymm1)\
	vmulps(ymm2, ymm0, ymm2)\
	vmulps(ymm3, ymm0, ymm3)\
	vmulps(ymm4, ymm0, ymm4)\
	vmulps(ymm5, ymm0, ymm5)\
	vaddps(ymm1, ymm11, ymm11)\
	vaddps(ymm2, ymm12, ymm12)\
	vaddps(ymm3, ymm13, ymm13)\
	vaddps(ymm4, ymm14, ymm14)\
	vaddps(ymm5, ymm15, ymm15)
#define SUB_KERNEL_8xFP32_8xFP32(n) \
	vmovaps(mem(rbx, n*8*4), ymm0)\
	vbroadcastss(mem(rax, (8*n+0)*4), ymm1)\
	vbroadcastss(mem(rax, (8*n+1)*4), ymm2)\
	vmulps(ymm1, ymm0, ymm1)\
	vmulps(ymm2, ymm0, ymm2)\
	vbroadcastss(mem(rax, (8*n+2)*4), ymm3)\
	vbroadcastss(mem(rax, (8*n+3)*4), ymm4)\
	vmulps(ymm3, ymm0, ymm3)\
	vmulps(ymm4, ymm0, ymm4)\
	vbroadcastss(mem(rax, (8*n+4)*4), ymm5)\
	vbroadcastss(mem(rax, (8*n+5)*4), ymm6)\
	vaddps(ymm1, ymm8, ymm8)\
	vaddps(ymm2, ymm9, ymm9)\
	vbroadcastss(mem(rax, (8*n+6)*4), ymm1)\
	vbroadcastss(mem(rax, (8*n+7)*4), ymm2)\
	vmulps(ymm5, ymm0, ymm5)\
	vmulps(ymm6, ymm0, ymm6)\
	vmulps(ymm1, ymm0, ymm1)\
	vmulps(ymm2, ymm0, ymm2)\
	vaddps(ymm3, ymm10, ymm10)\
	vaddps(ymm4, ymm11, ymm11)\
	vaddps(ymm5, ymm12, ymm12)\
	vaddps(ymm6, ymm13, ymm13)\
	vaddps(ymm1, ymm14, ymm14)\
	vaddps(ymm2, ymm15, ymm15)

#define SCALE_ACCUMULATORS_BY(reg)\
	vmulps(reg, ymm6, ymm6) \
	vmulps(reg, ymm7, ymm7) \
	vmulps(reg, ymm8, ymm8) \
	vmulps(reg, ymm9, ymm9) \
	vmulps(reg, ymm10, ymm10) \
	vmulps(reg, ymm11, ymm11) \
	vmulps(reg, ymm12, ymm12) \
	vmulps(reg, ymm13, ymm13) \
	vmulps(reg, ymm14, ymm14) \
	vmulps(reg, ymm15, ymm15)

#define SCALE_ACCUMULATORS_1x1() \
	vbroadcastss(mem(rax), ymm0) \
	vmulps(ymm0, ymm6, ymm6) \
	vmulps(ymm0, ymm7, ymm7) \
	vmulps(ymm0, ymm8, ymm8) \
	vmulps(ymm0, ymm9, ymm9) \
	vmulps(ymm0, ymm10, ymm10) \
	vmulps(ymm0, ymm11, ymm11) \
	vmulps(ymm0, ymm12, ymm12) \
	vmulps(ymm0, ymm13, ymm13) \
	vmulps(ymm0, ymm14, ymm14) \
	vmulps(ymm0, ymm15, ymm15)
#define SCALE_ACCUMULATORS_10x1() \
	vbroadcastss(mem(rax, 0*4), ymm0) \
	vbroadcastss(mem(rax, 1*4), ymm1) \
	vbroadcastss(mem(rax, 2*4), ymm2) \
	vbroadcastss(mem(rax, 3*4), ymm3) \
	vbroadcastss(mem(rax, 4*4), ymm4) \
	vbroadcastss(mem(rax, 5*4), ymm5) \
	vmulps(ymm0, ymm6, ymm6) \
	vmulps(ymm1, ymm7, ymm7) \
	vmulps(ymm2, ymm8, ymm8) \
	vmulps(ymm3, ymm9, ymm9) \
	vmulps(ymm4, ymm10, ymm10) \
	vmulps(ymm5, ymm11, ymm11) \
	vbroadcastss(mem(rax, 6*4), ymm0) \
	vbroadcastss(mem(rax, 7*4), ymm1) \
	vbroadcastss(mem(rax, 8*4), ymm2) \
	vbroadcastss(mem(rax, 9*4), ymm3) \
	vmulps(ymm0, ymm12, ymm12) \
	vmulps(ymm1, ymm13, ymm13) \
	vmulps(ymm2, ymm14, ymm14) \
	vmulps(ymm3, ymm15, ymm15)
#define SCALE_ACCUMULATORS_8x1() \
	vbroadcastss(mem(rax, 0*4), ymm0) \
	vbroadcastss(mem(rax, 1*4), ymm1) \
	vbroadcastss(mem(rax, 2*4), ymm2) \
	vbroadcastss(mem(rax, 3*4), ymm3) \
	vbroadcastss(mem(rax, 4*4), ymm4) \
	vbroadcastss(mem(rax, 5*4), ymm5) \
	vbroadcastss(mem(rax, 6*4), ymm6) \
	vbroadcastss(mem(rax, 7*4), ymm7) \
	vmulps(ymm0, ymm8, ymm8) \
	vmulps(ymm1, ymm9, ymm9) \
	vmulps(ymm2, ymm10, ymm10) \
	vmulps(ymm3, ymm11, ymm11) \
	vmulps(ymm4, ymm12, ymm12) \
	vmulps(ymm5, ymm13, ymm13) \
	vmulps(ymm6, ymm14, ymm14) \
	vmulps(ymm7, ymm15, ymm15)

#define ADD_BIAS_10x8xFP32(reg)\
	vaddps(reg, ymm6, ymm6) \
	vaddps(reg, ymm7, ymm7) \
	vaddps(reg, ymm8, ymm8) \
	vaddps(reg, ymm9, ymm9) \
	vaddps(reg, ymm10, ymm10) \
	vaddps(reg, ymm11, ymm11) \
	vaddps(reg, ymm12, ymm12) \
	vaddps(reg, ymm13, ymm13) \
	vaddps(reg, ymm14, ymm14) \
	vaddps(reg, ymm15, ymm15)

#define LOAD_5x8xFP32()\
	vmovups(mem(rcx), ymm1)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm2)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm3)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm4)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm5)\
	add(r14, rcx)

#define LOAD_5x8xFP16()\
	vmovups(mem(rcx), xmm1)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm2)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm3)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm4)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm5)\
	add(r14, rcx)

#define SCALE_5x8xFP32_BY_BETA() \
	vmulps(ymm1, ymm0, ymm1) \
	vmulps(ymm2, ymm0, ymm2) \
	vmulps(ymm3, ymm0, ymm3) \
	vmulps(ymm4, ymm0, ymm4) \
	vmulps(ymm5, ymm0, ymm5)

#define ADD_5x8xFP32_TO_ACCUMULATORS(acc0, acc1, acc2, acc3, acc4) \
	vaddps(ymm1, acc0, acc0) \
	vaddps(ymm2, acc1, acc1) \
	vaddps(ymm3, acc2, acc2) \
	vaddps(ymm4, acc3, acc3) \
	vaddps(ymm5, acc4, acc4)

#define ADD_BIAS_8x8xFP32(reg)\
	vaddps(reg, ymm8, ymm8) \
	vaddps(reg, ymm9, ymm9) \
	vaddps(reg, ymm10, ymm10) \
	vaddps(reg, ymm11, ymm11) \
	vaddps(reg, ymm12, ymm12) \
	vaddps(reg, ymm13, ymm13) \
	vaddps(reg, ymm14, ymm14) \
	vaddps(reg, ymm15, ymm15)

#define LOAD_4x8xFP32()\
	vmovups(mem(rcx), ymm1)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm2)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm3)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm4)\
	add(r14, rcx)

#define LOAD_4x8xFP16()\
	vmovups(mem(rcx), xmm1)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm2)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm3)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm4)\
	add(r14, rcx)

#define SCALE_4x8xFP32_BY_BETA() \
	vmulps(ymm1, ymm0, ymm1) \
	vmulps(ymm2, ymm0, ymm2) \
	vmulps(ymm3, ymm0, ymm3) \
	vmulps(ymm4, ymm0, ymm4)

#define ADD_4x8xFP32_TO_ACCUMULATORS(acc0, acc1, acc2, acc3) \
	vaddps(ymm1, acc0, acc0) \
	vaddps(ymm2, acc1, acc1) \
	vaddps(ymm3, acc2, acc2) \
	vaddps(ymm4, acc3, acc3)

#define LOAD_ADD_16xFP16(beta, reg1, reg2, stride)\
	vmovups(mem(rcx, 0*stride), xmm2)\
	vmovups(mem(rcx, 1*stride), xmm3)\
	vcvtph2ps(xmm2, ymm2)\
	vcvtph2ps(xmm3, ymm3)\
	vmulps(reg1, beta, reg1)\
	vmulps(reg2, beta, reg2)\
	vaddps(reg1, ymm2, reg1)\
	vaddps(reg1, ymm3, reg2)\
	add(r14, rcx)

#define STORE_5x8xFP32(reg1, reg2, reg3, reg4, reg5)\
	vmovups(reg1, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg2, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg3, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg4, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg5, mem(rcx))\
	add(r14, rcx)
#define STORE_4x8xFP32(reg1, reg2, reg3, reg4)\
	vmovups(reg1, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg2, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg3, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg4, mem(rcx))\
	add(r14, rcx)

#define STORE_5x8xFP16(reg0, reg1, reg2, reg3, reg4)\
	vmovups(reg0, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg1, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg2, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg3, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg4, mem(rcx))\
	add(r14, rcx)

#define CONVERT_5x8xFP16_TO_5x8xFP32() \
	vcvtph2ps(xmm1, ymm1) \
	vcvtph2ps(xmm2, ymm2) \
	vcvtph2ps(xmm3, ymm3) \
	vcvtph2ps(xmm4, ymm4) \
	vcvtph2ps(xmm5, ymm5)

#define CONVERT_ACCUMULATORS_TO_FP16() \
	vcvtps2ph(imm(0x03), ymm6, xmm6) \
	vcvtps2ph(imm(0x03), ymm7, xmm7) \
	vcvtps2ph(imm(0x03), ymm8, xmm8) \
	vcvtps2ph(imm(0x03), ymm9, xmm9) \
	vcvtps2ph(imm(0x03), ymm10, xmm10) \
	vcvtps2ph(imm(0x03), ymm11, xmm11) \
	vcvtps2ph(imm(0x03), ymm12, xmm12) \
	vcvtps2ph(imm(0x03), ymm13, xmm13) \
	vcvtps2ph(imm(0x03), ymm14, xmm14) \
	vcvtps2ph(imm(0x03), ymm15, xmm15)

#define LOAD_1x10xFP32(reg0, reg1)\
	vmovups(mem(rax), ymm(reg0))\
	vmovsd (mem(rax, 8*4), xmm(reg1))\
	add(r12, rax)

#define STORE_8x10xFP32() \
	vmovups(ymm0, mem(rbx, (0*10+0)*4)) \
	vmovsd (xmm1, mem(rbx, (0*10+8)*4)) \
	vmovups(ymm2, mem(rbx, (1*10+0)*4)) \
	vmovsd (xmm3, mem(rbx, (1*10+8)*4)) \
	vmovups(ymm4, mem(rbx, (2*10+0)*4)) \
	vmovsd (xmm5, mem(rbx, (2*10+8)*4)) \
	vmovups(ymm6, mem(rbx, (3*10+0)*4)) \
	vmovsd (xmm7, mem(rbx, (3*10+8)*4)) \
	vmovups(ymm8, mem(rbx, (4*10+0)*4)) \
	vmovsd (xmm9, mem(rbx, (4*10+8)*4)) \
	vmovups(ymm10, mem(rbx, (5*10+0)*4)) \
	vmovsd (xmm11, mem(rbx, (5*10+8)*4)) \
	vmovups(ymm12, mem(rbx, (6*10+0)*4)) \
	vmovsd (xmm13, mem(rbx, (6*10+8)*4)) \
	vmovups(ymm14, mem(rbx, (7*10+0)*4)) \
	vmovsd (xmm15, mem(rbx, (7*10+8)*4))

#define LOAD_1x10xFP16(reg0, reg1)\
	vmovups(mem(rax), xmm(reg0))\
	vmovss (mem(rax, 8*2), xmm(reg1))\
	add(r12, rax)\
	vcvtph2ps(xmm(reg0), ymm(reg0))\
	vcvtph2ps(xmm(reg1), xmm(reg1))

#define LOAD_2x10xFP16(reg0, reg1, reg2, reg3)\
	vmovups(mem(rax), xmm(reg0))\
	vmovss (mem(rax, 8*2), xmm(reg1))\
	vmovups(mem(rax, r15, 1), xmm(reg2))\
	vmovss (mem(rax, r15, 1, 8*2), xmm(reg3))\
	add(r12, rax)\
	vcvtph2ps(xmm(reg0), ymm(reg0))\
	vcvtph2ps(xmm(reg1), xmm(reg1))\
	vcvtph2ps(xmm(reg2), ymm(reg2))\
	vcvtph2ps(xmm(reg3), xmm(reg3))

#define STORE_1x10xFP32(n, reg0, reg1)\
	vmovups(ymm(reg0), (4*(n*10+0))(rbx))\
	vmovsd (xmm(reg1), (4*(n*10+8))(rbx))

#define RELU_10x8xFP32()\
	vxorps(ymm0, ymm0, ymm0)\
	vmaxps(ymm0, ymm6, ymm6)\
	vmaxps(ymm0, ymm7, ymm7)\
	vmaxps(ymm0, ymm8, ymm8)\
	vmaxps(ymm0, ymm9, ymm9)\
	vmaxps(ymm0, ymm10, ymm10)\
	vmaxps(ymm0, ymm11, ymm11)\
	vmaxps(ymm0, ymm12, ymm12)\
	vmaxps(ymm0, ymm13, ymm13)\
	vmaxps(ymm0, ymm14, ymm14)\
	vmaxps(ymm0, ymm15, ymm15)

#define LOAD_ADD_BIAS_10x8xFP32() \
	vmovaps(mem(rbx), ymm0) \
	vmovaps(mem(rbx, r14, 1), ymm1) \
	vmovaps(mem(rbx, r14, 2), ymm2) \
	vmovaps(mem(rbx, r13, 1), ymm3) \
	vmovaps(mem(rbx, r15, 1), ymm4) \
	add(r15, rbx) \
	add(r14, rbx) \
	vaddps(ymm0, ymm6, ymm6) \
	vaddps(ymm1, ymm7, ymm7) \
	vaddps(ymm2, ymm8, ymm8) \
	vaddps(ymm3, ymm9, ymm9) \
	vaddps(ymm4, ymm10, ymm10) \
	vmovaps(mem(rbx), ymm0) \
	vmovaps(mem(rbx, r14, 1), ymm1) \
	vmovaps(mem(rbx, r14, 2), ymm2) \
	vmovaps(mem(rbx, r13, 1), ymm3) \
	vmovaps(mem(rbx, r15, 1), ymm4) \
	vaddps(ymm0, ymm11, ymm11) \
	vaddps(ymm1, ymm12, ymm12) \
	vaddps(ymm2, ymm13, ymm13) \
	vaddps(ymm3, ymm14, ymm14) \
	vaddps(ymm4, ymm15, ymm15)
#define EXP_FIRST_STAGE_10x8xFP32() \
	movq(imm(0x4ab8aa3b4ab8aa3b), r14) \
	vmovq(r14, xmm0) \
	vpermpd(imm(0), ymm0, ymm0) \
	vmulps(ymm0, ymm6, ymm6) \
	vmulps(ymm0, ymm7, ymm7) \
	vmulps(ymm0, ymm8, ymm8) \
	vmulps(ymm0, ymm9, ymm9) \
	vmulps(ymm0, ymm10, ymm10) \
	vmulps(ymm0, ymm11, ymm11) \
	vmulps(ymm0, ymm12, ymm12) \
	vmulps(ymm0, ymm13, ymm13) \
	vmulps(ymm0, ymm14, ymm14) \
	vmulps(ymm0, ymm15, ymm15) \
	vcvtps2dq(ymm6, ymm6) \
	vcvtps2dq(ymm7, ymm7) \
	vcvtps2dq(ymm8, ymm8) \
	vcvtps2dq(ymm9, ymm9) \
	vcvtps2dq(ymm10, ymm10) \
	vcvtps2dq(ymm11, ymm11) \
	vcvtps2dq(ymm12, ymm12) \
	vcvtps2dq(ymm13, ymm13) \
	vcvtps2dq(ymm14, ymm14) \
	vcvtps2dq(ymm15, ymm15) \
	movq(imm(0x3f7de0683f7de068), r14) \
	vmovq(r14, xmm0) \
	vpermpd(imm(0), ymm0, ymm0)
#define EXP_SECOND_STAGE_1x8xFP32(r) \
	vextractf128(imm(1), ymm(r), xmm1) \
	vpsubd(xmm(r), xmm0, xmm2) \
	vpsubd(xmm1, xmm0, xmm3) \
	vinsertf128(imm(1), xmm3, ymm2, ymm2) \
	vrcpps(ymm2, ymm2) \
	vpaddd(xmm(r), xmm0, xmm(r)) \
	vpaddd(xmm1, xmm0, xmm1) \
	vinsertf128(imm(1), xmm1, ymm(r), ymm(r)) \
	vmulps(ymm(r), ymm2, ymm(r))

namespace ml
{
	using namespace ml::cpu;

	void gemm_avx_10x8(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(C.is_fp32() || C.is_fp16());
		assert(D.is_fp32() || D.is_fp16());
		assert(A.rows() == B.rows());
		assert(A.stride() == 10);
		assert(B.stride() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha.is_packed());
		assert(alpha.is_fp32());
		assert(cpu::is_aligned(A.data(), 32));
		assert(cpu::is_aligned(B.data(), 32));
		assert(beta_ptr != nullptr);
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 32));
		}

		const void *A_ptr = A.data();
		const void *B_ptr = B.data();
		const void *C_ptr = C.data();
		void *D_ptr = D.data();
		const void *bias_ptr = bias.is_packed() ? bias.data() : nullptr;
		const void *alpha_ptr = alpha.data();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride_in_bytes();
		const uint64_t D_stride = D.stride_in_bytes();
		const uint64_t flag_relu = use_relu;
		const uint64_t cd_in_fp16 = C.is_fp16() | (D.is_fp16() << 1);
		const uint64_t scalar_alpha = alpha.rows() == 1;

		begin_asm()
		movq(var(A_ptr), rax) // lhs pointer is in rax
		movq(var(B_ptr), rbx)// rhs pointer is in rbx
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_10xFP32_8xFP32(0)
		SUB_KERNEL_10xFP32_8xFP32(1)
		SUB_KERNEL_10xFP32_8xFP32(2)
		SUB_KERNEL_10xFP32_8xFP32(3)

		add(imm(4*10*4), rax)
		add(imm(4*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_10xFP32_8xFP32(0)
		add(imm(1*10*4), rax)
		add(imm(1*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(scalar_alpha), r14)
		test(r14, r14)
		je(COLUMN_ALPHA)
		SCALE_ACCUMULATORS_1x1()
		jmp(AFTER_ALPHA_SCALING)

		label(COLUMN_ALPHA)
		SCALE_ACCUMULATORS_10x1()
		label(AFTER_ALPHA_SCALING)

		// load address of bias pointer
		movq(var(bias_ptr), rax)
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax), ymm2)// load bias
		ADD_BIAS_10x8xFP32(ymm2)
		label(AFTER_BIAS)

		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rbx), ymm0)
		vxorps(ymm1, ymm1, ymm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(AFTER_LOAD_C)// if not loading C, jump to ReLU

		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx

		movq(var(cd_in_fp16), r11)// load fp16 flags
		and_(imm(0x1), r11)// if set
		test(r11, r11)
		je(C_IN_FP32)

		LOAD_5x8xFP16()// C in fp16 path
		CONVERT_5x8xFP16_TO_5x8xFP32()
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm6, ymm7, ymm8, ymm9, ymm10)
		LOAD_5x8xFP16()
		CONVERT_5x8xFP16_TO_5x8xFP32()
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm11, ymm12, ymm13, ymm14, ymm15)
		jmp(AFTER_LOAD_C)

		label(C_IN_FP32)
		LOAD_5x8xFP32()// C in fp32 path
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm6, ymm7, ymm8, ymm9, ymm10)
		LOAD_5x8xFP32()
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm11, ymm12, ymm13, ymm14, ymm15)
		label(AFTER_LOAD_C)

		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(AFTER_RELU)
		RELU_10x8xFP32()
		label(AFTER_RELU)

		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx

		movq(var(cd_in_fp16), r11)// load fp16 flags
		and_(imm(0x2), r11)// if set
		test(r11, r11)
		je(D_IN_FP32)

		CONVERT_ACCUMULATORS_TO_FP16()// D in fp16 path
		STORE_5x8xFP16(xmm6, xmm7, xmm8, xmm9, xmm10)
		STORE_5x8xFP16(xmm11, xmm12, xmm13, xmm14, xmm15)
		jmp(END)

		label(D_IN_FP32)
		STORE_5x8xFP32(ymm6, ymm7, ymm8, ymm9, ymm10)// D in fp32 path
		STORE_5x8xFP32(ymm11, ymm12, ymm13, ymm14, ymm15)
		label(END)

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
				[flag_relu] "m"(flag_relu),
				[bias_ptr] "m"(bias_ptr),
				[cd_in_fp16] "m"(cd_in_fp16),
				[scalar_alpha] "m"(scalar_alpha)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r11", "%r14")
	}
	void gemm_avx_8x8(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(C.is_fp32() || C.is_fp16());
		assert(D.is_fp32() || D.is_fp16());
		assert(A.rows() == B.rows());
		assert(A.stride() == 8);
		assert(B.stride() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha.is_packed());
		assert(alpha.is_fp32());
		assert(cpu::is_aligned(A.data(), 32));
		assert(cpu::is_aligned(B.data(), 32));
		assert(beta_ptr != nullptr);
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 32));
		}

		const void *A_ptr = A.data();
		const void *B_ptr = B.data();
		const void *C_ptr = C.data();
		void *D_ptr = D.data();
		const void *bias_ptr = bias.is_packed() ? bias.data() : nullptr;
		const void *alpha_ptr = alpha.data();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride_in_bytes();
		const uint64_t D_stride = D.stride_in_bytes();
		const uint64_t flag_relu = use_relu;
		const uint64_t cd_in_fp16 = C.is_fp16() | (D.is_fp16() << 1);
		const uint64_t scalar_alpha = alpha.rows() == 1;

		begin_asm()
		movq(var(A_ptr), rax) // lhs pointer is in rax
		movq(var(B_ptr), rbx)// rhs pointer is in rbx
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_8xFP32_8xFP32(0)
		SUB_KERNEL_8xFP32_8xFP32(1)
		SUB_KERNEL_8xFP32_8xFP32(2)
		SUB_KERNEL_8xFP32_8xFP32(3)

		add(imm(4*8*4), rax)
		add(imm(4*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_8xFP32_8xFP32(0)
		add(imm(1*8*4), rax)
		add(imm(1*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(scalar_alpha), r14)
		test(r14, r14)
		je(COLUMN_ALPHA)
//		SCALE_ACCUMULATORS_1x1()
		jmp(AFTER_ALPHA_SCALING)

		label(COLUMN_ALPHA)
		SCALE_ACCUMULATORS_8x1()
		label(AFTER_ALPHA_SCALING)

		// load address of bias pointer
		movq(var(bias_ptr), rax)
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax), ymm2)// load bias
		ADD_BIAS_10x8xFP32(ymm2)
		label(AFTER_BIAS)

		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rbx), ymm0)
		vxorps(ymm1, ymm1, ymm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(AFTER_LOAD_C)// if not loading C, jump to ReLU

		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx

		movq(var(cd_in_fp16), r11)// load fp16 flags
		and_(imm(0x1), r11)// if set
		test(r11, r11)
		je(C_IN_FP32)

		LOAD_5x8xFP16()// C in fp16 path
		CONVERT_5x8xFP16_TO_5x8xFP32()
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm6, ymm7, ymm8, ymm9, ymm10)
		LOAD_5x8xFP16()
		CONVERT_5x8xFP16_TO_5x8xFP32()
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm11, ymm12, ymm13, ymm14, ymm15)
		jmp(AFTER_LOAD_C)

		label(C_IN_FP32)
//		LOAD_4x8xFP32()
//		SCALE_4x8xFP32_BY_BETA()
//		ADD_4x8xFP32_TO_ACCUMULATORS(ymm8, ymm9, ymm10, ymm1)
//		LOAD_4x8xFP32()
//		SCALE_4x8xFP32_BY_BETA()
//		ADD_4x8xFP32_TO_ACCUMULATORS(ymm12, ymm13, ymm14, ymm15)
		label(AFTER_LOAD_C)

		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(AFTER_RELU)
		RELU_10x8xFP32()
		label(AFTER_RELU)

		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx

		movq(var(cd_in_fp16), r11)// load fp16 flags
		and_(imm(0x2), r11)// if set
		test(r11, r11)
		je(D_IN_FP32)

		CONVERT_ACCUMULATORS_TO_FP16()// D in fp16 path
		STORE_5x8xFP16(xmm6, xmm7, xmm8, xmm9, xmm10)
		STORE_5x8xFP16(xmm11, xmm12, xmm13, xmm14, xmm15)
		jmp(END)

		label(D_IN_FP32)
		STORE_4x8xFP32(ymm8, ymm9, ymm10, ymm11)// D in fp32 path
		STORE_4x8xFP32(ymm12, ymm13, ymm14, ymm15)
		label(END)

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
				[flag_relu] "m"(flag_relu),
				[bias_ptr] "m"(bias_ptr),
				[cd_in_fp16] "m"(cd_in_fp16),
				[scalar_alpha] "m"(scalar_alpha)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r11", "%r14")
	}

	void pack_avx_10xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		if (dst.is_partial())
		{
			pack_def_MxK(dst, src, src_pos, src_op);
			return;
		}
		assert(src.is_fp32() || src.is_fp16());
		assert(dst.is_fp32());
		assert(dst.stride() == 10);
		assert(ml::cpu::is_aligned(dst.data(), 32));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = dst.rows() % 8;
		const uint64_t src_stride = src.stride_in_bytes();
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			if (src.is_fp32())
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				vmovups(mem(rax), ymm0)
				vmovsd (mem(rax, 8*4), xmm1)
				vmovups(mem(rax, r12, 1), ymm2)
				vmovsd (mem(rax, r12, 1, 8*4), xmm3)
				vmovups(mem(rax, r12, 2), ymm4)
				vmovsd (mem(rax, r12, 2, 8*4), xmm5)
				vmovups(mem(rax, r13, 1), ymm6)
				vmovsd (mem(rax, r13, 1, 8*4), xmm7)
				add(r15, rax)
				vmovups(mem(rax), ymm8)
				vmovsd (mem(rax, 8*4), xmm9)
				vmovups(mem(rax, r12, 1), ymm10)
				vmovsd (mem(rax, r12, 1, 8*4), xmm11)
				vmovups(mem(rax, r12, 2), ymm12)
				vmovsd (mem(rax, r12, 2, 8*4), xmm13)
				vmovups(mem(rax, r13, 1), ymm14)
				vmovsd (mem(rax, r13, 1, 8*4), xmm15)
				add(r15, rax)

				STORE_8x10xFP32()

				add(imm(8*10*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				vmovups(mem(rax), ymm0)
				vmovsd (mem(rax, 8*4), xmm1)
				add(r12, rax)
				vmovups(ymm0, mem(rbx))
				vmovsd (xmm1, mem(rbx, 8*4))
				add(imm(1*10*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(
						:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
						"%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r15)
				sal(imm(1), r12)

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				LOAD_2x10xFP16(0, 1, 2, 3)
				LOAD_2x10xFP16(4, 5, 6, 7)
				LOAD_2x10xFP16(8, 9, 10, 11)
				LOAD_2x10xFP16(12, 13, 14, 15)

				STORE_1x10xFP32(0, 0, 1)
				STORE_1x10xFP32(1, 2, 3)
				STORE_1x10xFP32(2, 4, 5)
				STORE_1x10xFP32(3, 6, 7)
				STORE_1x10xFP32(4, 8, 9)
				STORE_1x10xFP32(5, 10, 11)
				STORE_1x10xFP32(6, 12, 13)
				STORE_1x10xFP32(7, 14, 15)

				add(imm(4*8*10), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				sar(imm(1), r12)// divide stride by 2, effectively reverting to reading single row at the time
				label(UNROLLED1)
				LOAD_1x10xFP16(0, 1)
				STORE_1x10xFP32(0, 0, 1)
				add(imm(4*1*10), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(
						:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
						"%r12", "%r14", "%r15")
			}
		}
		else
		{
			if (src.is_fp32())
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				// load 8x8 tile
				movq(rax, rcx)// tmp src pointer is in rcx
				vmovups(mem(rcx), ymm0)
				vmovups(mem(rcx, r12, 1), ymm1)
				vmovups(mem(rcx, r12, 2), ymm2)
				vmovups(mem(rcx, r13, 1), ymm3)
				add(r15, rcx)
				vmovups(mem(rcx), ymm4)
				vmovups(mem(rcx, r12, 1), ymm5)
				vmovups(mem(rcx, r12, 2), ymm6)
				vmovups(mem(rcx, r13, 1), ymm7)
				add(r15, rcx)

				AVX_8x8_TRANSPOSE()

				vmovups(ymm8, mem(rbx, 4*(0*10+0)))
				vmovups(ymm9, mem(rbx, 4*(1*10+0)))
				vmovups(ymm10, mem(rbx, 4*(2*10+0)))
				vmovups(ymm11, mem(rbx, 4*(3*10+0)))
				vmovups(ymm12, mem(rbx, 4*(4*10+0)))
				vmovups(ymm13, mem(rbx, 4*(5*10+0)))
				vmovups(ymm14, mem(rbx, 4*(6*10+0)))
				vmovups(ymm15, mem(rbx, 4*(7*10+0)))

				// rows 8-9
				vmovups(mem(rcx), ymm0)
				vmovups(mem(rcx, r12, 1), ymm1)

				vunpcklps(ymm1, ymm0, ymm4)
				vunpckhps(ymm1, ymm0, ymm5)

				vextractf128(imm(0x1), ymm4, xmm6)// e4 f4 e5 f5
				vextractf128(imm(0x1), ymm5, xmm7)// e6 f6 e7 f7

				vmovlpd(xmm4, mem(rbx, 4*(0*10+8)))
				vmovhpd(xmm4, mem(rbx, 4*(1*10+8)))
				vmovlpd(xmm5, mem(rbx, 4*(2*10+8)))
				vmovhpd(xmm5, mem(rbx, 4*(3*10+8)))
				vmovlpd(xmm6, mem(rbx, 4*(4*10+8)))
				vmovhpd(xmm6, mem(rbx, 4*(5*10+8)))
				vmovlpd(xmm7, mem(rbx, 4*(6*10+8)))
				vmovhpd(xmm7, mem(rbx, 4*(7*10+8)))

				add(imm(4*8), rax)// add stride to src pointer
				add(imm(4*8*10), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)// tmp src pointer is in rcx
				vmovss(mem(rcx), xmm0)
				vmovss(mem(rcx, r12, 1), xmm1)
				vmovss(mem(rcx, r12, 2), xmm2)
				vmovss(mem(rcx, r13, 1), xmm3)
				add(r15, rcx)
				vmovss(mem(rcx), xmm4)
				vmovss(mem(rcx, r12, 1), xmm5)
				vmovss(mem(rcx, r12, 2), xmm6)
				vmovss(mem(rcx, r13, 1), xmm7)
				add(r15, rcx)
				vmovss(mem(rcx), xmm8)
				vmovss(mem(rcx, r12, 1), xmm9)

				vmovss(xmm0, mem(rbx, 0*4))
				vmovss(xmm1, mem(rbx, 1*4))
				vmovss(xmm2, mem(rbx, 2*4))
				vmovss(xmm3, mem(rbx, 3*4))
				vmovss(xmm4, mem(rbx, 4*4))
				vmovss(xmm5, mem(rbx, 5*4))
				vmovss(xmm6, mem(rbx, 6*4))
				vmovss(xmm7, mem(rbx, 7*4))
				vmovss(xmm8, mem(rbx, 8*4))
				vmovss(xmm9, mem(rbx, 9*4))

				add(imm(4*1), rax)// add stride to src pointer
				add(imm(4*10*1), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(
						:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
						"%r12", "%r13", "%r14", "%r15")
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

				label(UNROLLED8)
				// first 8x8 tile
				movq(rax, r13)// tmp src pointer is in r13
				// rows 0-7
				vmovups(mem(r13), xmm0)
				add(r12, r13)// add stride to src pointer
				vmovups(mem(r13), xmm1)
				add(r12, r13)// add stride to src pointer
				vmovups(mem(r13), xmm2)
				add(r12, r13)// add stride to src pointer
				vmovups(mem(r13), xmm3)
				add(r12, r13)// add stride to src pointer
				vmovups(mem(r13), xmm4)
				add(r12, r13)// add stride to src pointer
				vmovups(mem(r13), xmm5)
				add(r12, r13)// add stride to src pointer
				vmovups(mem(r13), xmm6)
				add(r12, r13)// add stride to src pointer
				vmovups(mem(r13), xmm7)
				add(r12, r13)// add stride to src pointer

				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, ymm1)
				vcvtph2ps(xmm2, ymm2)
				vcvtph2ps(xmm3, ymm3)
				vcvtph2ps(xmm4, ymm4)
				vcvtph2ps(xmm5, ymm5)
				vcvtph2ps(xmm6, ymm6)
				vcvtph2ps(xmm7, ymm7)

				AVX_8x8_TRANSPOSE()

				vmovups(ymm8, mem(rbx,4*(0*10+0)))
				vmovups(ymm9, mem(rbx,4*(1*10+0)))
				vmovups(ymm10, mem(rbx,4*(2*10+0)))
				vmovups(ymm11, mem(rbx,4*(3*10+0)))
				vmovups(ymm12, mem(rbx,4*(4*10+0)))
				vmovups(ymm13, mem(rbx,4*(5*10+0)))
				vmovups(ymm14, mem(rbx,4*(6*10+0)))
				vmovups(ymm15, mem(rbx,4*(7*10+0)))

				// rows 8-9
				vmovups(mem(r13), xmm0)
				add(r12, r13)
				vmovups(mem(r13), xmm1)
				add(r12, r13)

				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, ymm1)

				vunpcklps(ymm1, ymm0, ymm4)
				vunpckhps(ymm1, ymm0, ymm5)

				vextractf128(imm(0x1), ymm4, xmm6)// e4 f4 e5 f5
				vextractf128(imm(0x1), ymm5, xmm7)// e6 f6 e7 f7

				vmovlpd(xmm4, mem(rbx, 4*(0*10+8)))
				vmovhpd(xmm4, mem(rbx, 4*(1*10+8)))
				vmovlpd(xmm5, mem(rbx, 4*(2*10+8)))
				vmovhpd(xmm5, mem(rbx, 4*(3*10+8)))
				vmovlpd(xmm6, mem(rbx, 4*(4*10+8)))
				vmovhpd(xmm6, mem(rbx, 4*(5*10+8)))
				vmovlpd(xmm7, mem(rbx, 4*(6*10+8)))
				vmovhpd(xmm7, mem(rbx, 4*(7*10+8)))

				add(imm(2*8), rax)// add stride to src pointer
				add(imm(4*8*10), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, r13)// tmp src pointer is in r13

				movzw(mem(r13), rcx)
				vmovq(rcx, xmm0)
				add(r12, r13)// add stride to src pointer
				movzw(mem(r13), rcx)
				vmovq(rcx, xmm1)
				add(r12, r13)// add stride to src pointer
				movzw(mem(r13), rcx)
				vmovq(rcx, xmm2)
				add(r12, r13)// add stride to src pointer
				movzw(mem(r13), rcx)
				vmovq(rcx, xmm3)
				add(r12, r13)// add stride to src pointer
				movzw(mem(r13), rcx)
				vmovq(rcx, xmm4)
				add(r12, r13)// add stride to src pointer
				movzw(mem(r13), rcx)
				vmovq(rcx, xmm5)
				add(r12, r13)// add stride to src pointer
				movzw(mem(r13), rcx)
				vmovq(rcx, xmm6)
				add(r12, r13)// add stride to src pointer
				movzw(mem(r13), rcx)
				vmovq(rcx, xmm7)
				add(r12, r13)// add stride to src pointer
				movzw(mem(r13), rcx)
				vmovq(rcx, xmm8)
				add(r12, r13)// add stride to src pointer
				movzw(mem(r13), rcx)
				vmovq(rcx, xmm9)
				add(r12, r13)// add stride to src pointer

				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, ymm1)
				vcvtph2ps(xmm2, ymm2)
				vcvtph2ps(xmm3, ymm3)
				vcvtph2ps(xmm4, ymm4)
				vcvtph2ps(xmm5, ymm5)
				vcvtph2ps(xmm6, ymm6)
				vcvtph2ps(xmm7, ymm7)
				vcvtph2ps(xmm8, ymm8)
				vcvtph2ps(xmm9, ymm9)

				vmovss(xmm0, mem(rbx, 0*4))
				vmovss(xmm1, mem(rbx, 1*4))
				vmovss(xmm2, mem(rbx, 2*4))
				vmovss(xmm3, mem(rbx, 3*4))
				vmovss(xmm4, mem(rbx, 4*4))
				vmovss(xmm5, mem(rbx, 5*4))
				vmovss(xmm6, mem(rbx, 6*4))
				vmovss(xmm7, mem(rbx, 7*4))
				vmovss(xmm8, mem(rbx, 8*4))
				vmovss(xmm9, mem(rbx, 9*4))

				add(imm(2*1), rax)// add stride to src pointer
				add(imm(4*10*1), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(
						:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
						"%r12", "%r13", "%r14", "%r15")
			}
		}
	}
	void pack_avx_8xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		if (dst.is_partial())
		{
			pack_def_MxK(dst, src, src_pos, src_op);
			return;
		}
		assert(src.is_fp32() || src.is_fp16());
		assert(dst.is_fp32());
		assert(dst.stride() == 8);
		assert(ml::cpu::is_aligned(dst.data(), 32));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = dst.rows() % 8;
		const uint64_t src_stride = src.stride_in_bytes();
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			if (src.is_fp32())
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				vmovups(mem(rax), ymm0)
				vmovups(mem(rax, r12, 1), ymm1)
				vmovups(mem(rax, r12, 2), ymm2)
				vmovups(mem(rax, r13, 1), ymm3)
				add(r15, rax)
				vmovups(mem(rax), ymm4)
				vmovups(mem(rax, r12, 1), ymm5)
				vmovups(mem(rax, r12, 2), ymm6)
				vmovups(mem(rax, r13, 1), ymm7)
				add(r15, rax)
				vmovaps(ymm0, mem(rbx, (0*8)*4))
				vmovups(ymm1, mem(rbx, (1*8)*4))
				vmovaps(ymm2, mem(rbx, (2*8)*4))
				vmovups(ymm3, mem(rbx, (3*8)*4))
				vmovaps(ymm4, mem(rbx, (4*8)*4))
				vmovups(ymm5, mem(rbx, (5*8)*4))
				vmovaps(ymm6, mem(rbx, (6*8)*4))
				vmovups(ymm7, mem(rbx, (7*8)*4))
				add(imm(4*8*8), rbx)// add stride to dst pointer
				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				vmovups(mem(rax), ymm0)
				add(r12, rax)// add stride to src pointer
				vmovaps(ymm0, mem(rbx))
				add(imm(4*8*1), rbx)// add stride to dst pointer
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
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
						"%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				asm volatile(
						"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
						"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
						"movq %[src_stride], %%r12 \n\t"// src stride is in r12

						"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
						"test %%r14, %%r14 \n\t"
						"je FINALLOOP%= \n\t"

						"UNROLLED8%=: \n\t"
						"vmovups 0x00(%%rax), %%xmm0 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%xmm1 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%xmm2 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%xmm3 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%xmm4 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%xmm5 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%xmm6 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%xmm7 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer

						"vcvtph2ps %%xmm0, %%ymm0 \n\t"
						"vcvtph2ps %%xmm1, %%ymm1 \n\t"
						"vcvtph2ps %%xmm2, %%ymm2 \n\t"
						"vcvtph2ps %%xmm3, %%ymm3 \n\t"
						"vcvtph2ps %%xmm4, %%ymm4 \n\t"
						"vcvtph2ps %%xmm5, %%ymm5 \n\t"
						"vcvtph2ps %%xmm6, %%ymm6 \n\t"
						"vcvtph2ps %%xmm7, %%ymm7 \n\t"

						"vmovaps %%ymm0, 0x000(%%rbx) \n\t"
						"vmovaps %%ymm1, 0x020(%%rbx) \n\t"
						"vmovaps %%ymm2, 0x040(%%rbx) \n\t"
						"vmovaps %%ymm3, 0x060(%%rbx) \n\t"
						"vmovaps %%ymm4, 0x080(%%rbx) \n\t"
						"vmovaps %%ymm5, 0x0A0(%%rbx) \n\t"
						"vmovaps %%ymm6, 0x0C0(%%rbx) \n\t"
						"vmovaps %%ymm7, 0x0E0(%%rbx) \n\t"

						"add $(4*8*8), %%rbx \n\t"// add stride to dst pointer

						"dec %%r14 \n\t"
						"jne UNROLLED8%= \n\t"

						"FINALLOOP%=: \n\t"
						"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
						"test %%r14, %%r14 \n\t"
						"je EPILOGUE%= \n\t"

						"UNROLLED1%=: \n\t"
						"vmovups 0x00(%%rax), %%xmm0 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vcvtph2ps %%xmm0, %%ymm0 \n\t"
						"vmovaps %%ymm0, 0x00(%%rbx) \n\t"
						"add $(4*1*8), %%rbx \n\t"// add stride to dst pointer

						"dec %%r14 \n\t"
						"jne UNROLLED1%= \n\t"

						"EPILOGUE%=: \n\t"
						"vzeroupper \n\t"

						:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
						"%r12", "%r14");
			}
		}
		else
		{
			if (src.is_fp32())
			{
				asm volatile(
						"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
						"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
						"movq %[src_stride], %%r12 \n\t"// src stride is in r12

						"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
						"test %%r14, %%r14 \n\t"
						"je FINALLOOP%= \n\t"

						"UNROLLED8%=: \n\t"
						// first 8x8 tile
						"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

						"vmovups 0x0(%%r13), %%ymm0 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%ymm1 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%ymm2 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%ymm3 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%ymm4 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%ymm5 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%ymm6 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%ymm7 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer

						// transpose 8x8
						// first shuffle
						"vunpcklps %%ymm1, %%ymm0, %%ymm8 \n\t"
						"vunpckhps %%ymm1, %%ymm0, %%ymm9 \n\t"
						"vunpcklps %%ymm3, %%ymm2, %%ymm10 \n\t"
						"vunpckhps %%ymm3, %%ymm2, %%ymm11 \n\t"
						"vunpcklps %%ymm5, %%ymm4, %%ymm12 \n\t"
						"vunpckhps %%ymm5, %%ymm4, %%ymm13 \n\t"
						"vunpcklps %%ymm7, %%ymm6, %%ymm14 \n\t"
						"vunpckhps %%ymm7, %%ymm6, %%ymm15 \n\t"

						// second shuffle
						"vunpcklpd %%ymm10, %%ymm8, %%ymm0 \n\t"
						"vunpckhpd %%ymm10, %%ymm8, %%ymm1 \n\t"
						"vunpcklpd %%ymm11, %%ymm9, %%ymm2 \n\t"
						"vunpckhpd %%ymm11, %%ymm9, %%ymm3 \n\t"
						"vunpcklpd %%ymm14, %%ymm12, %%ymm4 \n\t"
						"vunpckhpd %%ymm14, %%ymm12, %%ymm5 \n\t"
						"vunpcklpd %%ymm15, %%ymm13, %%ymm6 \n\t"
						"vunpckhpd %%ymm15, %%ymm13, %%ymm7 \n\t"

						// third shuffle
						"vperm2f128 $0x20, %%ymm4, %%ymm0, %%ymm8 \n\t"
						"vperm2f128 $0x20, %%ymm5, %%ymm1, %%ymm9 \n\t"
						"vperm2f128 $0x20, %%ymm6, %%ymm2, %%ymm10 \n\t"
						"vperm2f128 $0x20, %%ymm7, %%ymm3, %%ymm11 \n\t"
						"vperm2f128 $0x31, %%ymm4, %%ymm0, %%ymm12 \n\t"
						"vperm2f128 $0x31, %%ymm5, %%ymm1, %%ymm13 \n\t"
						"vperm2f128 $0x31, %%ymm6, %%ymm2, %%ymm14 \n\t"
						"vperm2f128 $0x31, %%ymm7, %%ymm3, %%ymm15 \n\t"

						"vmovaps %%ymm8, (4*8*0)(%%rbx) \n\t"
						"vmovaps %%ymm9, (4*8*1)(%%rbx) \n\t"
						"vmovaps %%ymm10, (4*8*2)(%%rbx) \n\t"
						"vmovaps %%ymm11, (4*8*3)(%%rbx) \n\t"
						"vmovaps %%ymm12, (4*8*4)(%%rbx) \n\t"
						"vmovaps %%ymm13, (4*8*5)(%%rbx) \n\t"
						"vmovaps %%ymm14, (4*8*6)(%%rbx) \n\t"
						"vmovaps %%ymm15, (4*8*7)(%%rbx) \n\t"

						"add $(4*8), %%rax \n\t"// add stride to src pointer
						"add $(4*8*8), %%rbx \n\t"// add stride to dst pointer

						"dec %%r14 \n\t"
						"jne UNROLLED8%= \n\t"

						"FINALLOOP%=: \n\t"
						"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
						"test %%r14, %%r14 \n\t"
						"je EPILOGUE%= \n\t"

						"UNROLLED1%=: \n\t"
						"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

						"vmovss 0x0(%%r13), %%xmm0 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm1 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm2 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm3 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm4 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm5 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm6 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm7 \n\t"

						"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
						"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
						"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
						"vmovss %%xmm3, (4*3)(%%rbx) \n\t"
						"vmovss %%xmm4, (4*4)(%%rbx) \n\t"
						"vmovss %%xmm5, (4*5)(%%rbx) \n\t"
						"vmovss %%xmm6, (4*6)(%%rbx) \n\t"
						"vmovss %%xmm7, (4*7)(%%rbx) \n\t"

						"add $(4*1), %%rax \n\t"// add stride to src pointer
						"add $(4*8*1), %%rbx \n\t"// add stride to dst pointer

						"dec %%r14 \n\t"
						"jne UNROLLED1%= \n\t"

						"EPILOGUE%=: \n\t"
						"vzeroupper \n\t"

						:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
						"%r12", "%r13", "%r14");
			}
			else
			{
				asm volatile(
						"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
						"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
						"movq %[src_stride], %%r12 \n\t"// src stride is in r12

						"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
						"test %%r14, %%r14 \n\t"
						"je FINALLOOP%= \n\t"

						"UNROLLED8%=: \n\t"
						// first 8x8 tile
						"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

						"vmovups 0x0(%%r13), %%xmm0 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%xmm1 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%xmm2 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%xmm3 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%xmm4 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%xmm5 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%xmm6 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovups 0x0(%%r13), %%xmm7 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer

						"vcvtph2ps %%xmm0, %%ymm0 \n\t"
						"vcvtph2ps %%xmm1, %%ymm1 \n\t"
						"vcvtph2ps %%xmm2, %%ymm2 \n\t"
						"vcvtph2ps %%xmm3, %%ymm3 \n\t"
						"vcvtph2ps %%xmm4, %%ymm4 \n\t"
						"vcvtph2ps %%xmm5, %%ymm5 \n\t"
						"vcvtph2ps %%xmm6, %%ymm6 \n\t"
						"vcvtph2ps %%xmm7, %%ymm7 \n\t"

						// transpose 8x8
						// first shuffle
						"vunpcklps %%ymm1, %%ymm0, %%ymm8 \n\t"
						"vunpckhps %%ymm1, %%ymm0, %%ymm9 \n\t"
						"vunpcklps %%ymm3, %%ymm2, %%ymm10 \n\t"
						"vunpckhps %%ymm3, %%ymm2, %%ymm11 \n\t"
						"vunpcklps %%ymm5, %%ymm4, %%ymm12 \n\t"
						"vunpckhps %%ymm5, %%ymm4, %%ymm13 \n\t"
						"vunpcklps %%ymm7, %%ymm6, %%ymm14 \n\t"
						"vunpckhps %%ymm7, %%ymm6, %%ymm15 \n\t"

						// second shuffle
						"vunpcklpd %%ymm10, %%ymm8, %%ymm0 \n\t"
						"vunpckhpd %%ymm10, %%ymm8, %%ymm1 \n\t"
						"vunpcklpd %%ymm11, %%ymm9, %%ymm2 \n\t"
						"vunpckhpd %%ymm11, %%ymm9, %%ymm3 \n\t"
						"vunpcklpd %%ymm14, %%ymm12, %%ymm4 \n\t"
						"vunpckhpd %%ymm14, %%ymm12, %%ymm5 \n\t"
						"vunpcklpd %%ymm15, %%ymm13, %%ymm6 \n\t"
						"vunpckhpd %%ymm15, %%ymm13, %%ymm7 \n\t"

						// third shuffle
						"vperm2f128 $0x20, %%ymm4, %%ymm0, %%ymm8 \n\t"
						"vperm2f128 $0x20, %%ymm5, %%ymm1, %%ymm9 \n\t"
						"vperm2f128 $0x20, %%ymm6, %%ymm2, %%ymm10 \n\t"
						"vperm2f128 $0x20, %%ymm7, %%ymm3, %%ymm11 \n\t"
						"vperm2f128 $0x31, %%ymm4, %%ymm0, %%ymm12 \n\t"
						"vperm2f128 $0x31, %%ymm5, %%ymm1, %%ymm13 \n\t"
						"vperm2f128 $0x31, %%ymm6, %%ymm2, %%ymm14 \n\t"
						"vperm2f128 $0x31, %%ymm7, %%ymm3, %%ymm15 \n\t"

						"vmovaps %%ymm8, (4*8*0)(%%rbx) \n\t"
						"vmovaps %%ymm9, (4*8*1)(%%rbx) \n\t"
						"vmovaps %%ymm10, (4*8*2)(%%rbx) \n\t"
						"vmovaps %%ymm11, (4*8*3)(%%rbx) \n\t"
						"vmovaps %%ymm12, (4*8*4)(%%rbx) \n\t"
						"vmovaps %%ymm13, (4*8*5)(%%rbx) \n\t"
						"vmovaps %%ymm14, (4*8*6)(%%rbx) \n\t"
						"vmovaps %%ymm15, (4*8*7)(%%rbx) \n\t"

						"add $(2*8), %%rax \n\t"// add stride to src pointer
						"add $(4*8*8), %%rbx \n\t"// add stride to dst pointer

						"dec %%r14 \n\t"
						"jne UNROLLED8%= \n\t"

						"FINALLOOP%=: \n\t"
						"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
						"test %%r14, %%r14 \n\t"
						"je EPILOGUE%= \n\t"

						"UNROLLED1%=: \n\t"
						"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

						"vmovss 0x0(%%r13), %%xmm0 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm1 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm2 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm3 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm4 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm5 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm6 \n\t"
						"add %%r12, %%r13 \n\t"// add stride to src pointer
						"vmovss 0x0(%%r13), %%xmm7 \n\t"

						"vcvtph2ps %%xmm0, %%xmm0 \n\t"
						"vcvtph2ps %%xmm1, %%xmm1 \n\t"
						"vcvtph2ps %%xmm2, %%xmm2 \n\t"
						"vcvtph2ps %%xmm3, %%xmm3 \n\t"
						"vcvtph2ps %%xmm4, %%xmm4 \n\t"
						"vcvtph2ps %%xmm5, %%xmm5 \n\t"
						"vcvtph2ps %%xmm6, %%xmm6 \n\t"
						"vcvtph2ps %%xmm7, %%xmm7 \n\t"

						"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
						"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
						"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
						"vmovss %%xmm3, (4*3)(%%rbx) \n\t"
						"vmovss %%xmm4, (4*4)(%%rbx) \n\t"
						"vmovss %%xmm5, (4*5)(%%rbx) \n\t"
						"vmovss %%xmm6, (4*6)(%%rbx) \n\t"
						"vmovss %%xmm7, (4*7)(%%rbx) \n\t"

						"add $(2*1), %%rax \n\t"// add stride to src pointer
						"add $(4*8*1), %%rbx \n\t"// add stride to dst pointer

						"dec %%r14 \n\t"
						"jne UNROLLED1%= \n\t"

						"EPILOGUE%=: \n\t"
						"vzeroupper \n\t"

						:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
						"%r12", "%r13", "%r14");
			}
		}
	}

// multi-head attention (MHA) kernel
	void mha_qk_avx_10x8(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept
	{
		assert(temp.is_fp32());
		assert(Q.is_fp32());
		assert(K.is_fp32());
		assert(Q.rows() == K.rows());
		assert(Q.stride() == 10);
		assert(K.stride() == 8);
		assert(temp.columns() == Q.columns());
		assert(temp.rows() == K.columns());
		assert(temp.stride() == 10);

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(Q.data(), 32));
		assert(cpu::is_aligned(K.data(), 32));

		const float *Q_ptr = Q.data<float>();
		const float *K_ptr = K.data<float>();
		float *temp_ptr = temp.data<float>();
		const float *bias_ptr = bias.is_packed() ? bias.data<float>() : nullptr;
		if (bias.is_packed())
		{
			assert(bias.is_fp32());
			assert(cpu::is_aligned(bias.data(), 16));
		}
		float *softmax_ptr = softmax_sum.is_packed() ? softmax_sum.data<float>() : nullptr;
		if (softmax_sum.is_packed())
		{
			assert(softmax_sum.is_fp32());
			assert(softmax_sum.rows() >= temp.columns());
			assert(cpu::is_aligned(softmax_sum.data(), 32));
		}

		uint64_t k_iter = Q.rows() / 4;
		uint64_t k_left = Q.rows() % 4;
		const uint64_t bias_stride = bias.stride() * sizeof(float);
		assert(bias_stride % 32 == 0);

		begin_asm()
		movq(var(Q_ptr), rax) // lhs pointer is in rax
		movq(var(K_ptr), rbx)// rhs pointer is in rbx
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_10xFP32_8xFP32(0)
		SUB_KERNEL_10xFP32_8xFP32(1)
		SUB_KERNEL_10xFP32_8xFP32(2)
		SUB_KERNEL_10xFP32_8xFP32(3)

		add(imm(4*10*4), rax)
		add(imm(4*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_10xFP32_8xFP32(0)
		add(imm(1*10*4), rax)
		add(imm(1*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		vbroadcastss(mem(rax), ymm0)
		SCALE_ACCUMULATORS_BY(ymm0)

		movq(var(bias_ptr), rbx)// load address of bias pointer
		test(rbx, rbx)
		je(AFTER_BIAS)
		movq(var(bias_stride), r14)// load address of bias stride into r14
		movq(r14, r13)
		sal(imm(1), r13)// r13 = stride * 2
		add(r14, r13)// r13 == stride * 3
		movq(r14, r15)
		sal(imm(2), r15)// r15 = stride * 4
		LOAD_ADD_BIAS_10x8xFP32()
		label(AFTER_BIAS)

		EXP_FIRST_STAGE_10x8xFP32()
		EXP_SECOND_STAGE_1x8xFP32(6)
		EXP_SECOND_STAGE_1x8xFP32(7)
		EXP_SECOND_STAGE_1x8xFP32(8)
		EXP_SECOND_STAGE_1x8xFP32(9)
		EXP_SECOND_STAGE_1x8xFP32(10)
		EXP_SECOND_STAGE_1x8xFP32(11)
		EXP_SECOND_STAGE_1x8xFP32(12)
		EXP_SECOND_STAGE_1x8xFP32(13)
		EXP_SECOND_STAGE_1x8xFP32(14)
		EXP_SECOND_STAGE_1x8xFP32(15)

		movq(var(temp_ptr), rbx)// temp pointer is in rbx
		movq(var(softmax_ptr), rcx)// softmax sum pointer is in rcx

		// store 2x8 tile
		vunpcklps(ymm7, ymm6, ymm0)
		vunpckhps(ymm7, ymm6, ymm1)
		vextractf128(imm(0x1), ymm0, xmm2)// e4 f4 e5 f5
		vextractf128(imm(0x1), ymm1, xmm3)// e6 f6 e7 f7

		vmovlpd(xmm0, mem(rbx, 4*(0*10+0)))
		vmovhpd(xmm0, mem(rbx, 4*(1*10+0)))
		vmovlpd(xmm1, mem(rbx, 4*(2*10+0)))
		vmovhpd(xmm1, mem(rbx, 4*(3*10+0)))
		vmovlpd(xmm2, mem(rbx, 4*(4*10+0)))
		vmovhpd(xmm2, mem(rbx, 4*(5*10+0)))
		vmovlpd(xmm3, mem(rbx, 4*(6*10+0)))
		vmovhpd(xmm3, mem(rbx, 4*(7*10+0)))

		test(rcx, rcx)
		je(SKIP_REDUCTION_1)
		// sum first 2 elements
		vmovsd(mem(rcx), xmm4)// load previous sum
		vaddps(ymm1, ymm0, ymm0)
		vaddps(ymm3, ymm2, ymm2)
		vaddps(ymm2, ymm0, ymm0)
		vmovhlps(xmm0, xmm1, xmm1)// copy 3rd and 4th elements into 1st and 2nd to be added together in next line
		vaddps(xmm1, xmm0, xmm0)
		vaddps(xmm4, xmm0, xmm0)
		vmovsd(xmm0, mem(rcx))// store updated sum 2xfp32
		label(SKIP_REDUCTION_1)

		// store 8x8 tile
		AVX_8x8_TRANSPOSE_INV()

		vmovups(ymm0, mem(rbx, 4*(0*10+2)))
		vmovups(ymm1, mem(rbx, 4*(1*10+2)))
		vmovups(ymm2, mem(rbx, 4*(2*10+2)))
		vmovups(ymm3, mem(rbx, 4*(3*10+2)))
		vmovups(ymm4, mem(rbx, 4*(4*10+2)))
		vmovups(ymm5, mem(rbx, 4*(5*10+2)))
		vmovups(ymm6, mem(rbx, 4*(6*10+2)))
		vmovups(ymm7, mem(rbx, 4*(7*10+2)))

		test(rcx, rcx)
		je(SKIP_REDUCTION_2)
		vmovups(mem(rcx, 2*4), ymm15)// load previous sum
		// sum all accumulators and place result in the first one (ymm8)
		vaddps(ymm1, ymm0, ymm0)
		vaddps(ymm3, ymm2, ymm2)
		vaddps(ymm5, ymm4, ymm4)
		vaddps(ymm7, ymm6, ymm6)
		vaddps(ymm4, ymm0, ymm0)
		vaddps(ymm6, ymm2, ymm2)
		vaddps(ymm2, ymm0, ymm0)
		vaddps(ymm15, ymm0, ymm0)// add current sum
		vmovups(ymm0, mem(rcx, 2*4))// store updated sum 8xfp32
		label(SKIP_REDUCTION_2)

		vzeroupper()

		end_asm(:// outputs
				:// inputs
				[Q_ptr] "m"(Q_ptr),
				[K_ptr] "m"(K_ptr),
				[temp_ptr] "m"(temp_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[bias_stride] "m"(bias_stride),
				[alpha_ptr] "m"(alpha_ptr),
				[bias_ptr] "m"(bias_ptr),
				[softmax_ptr] "m"(softmax_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r13", "%r14", "%r15")
	}

	// batched depthwise convolution kernel

#define DWCONV_MAIN_LOOP_FP32() \
	vmovups(mem(rax, 0*8*4), ymm8) \
	vmovups(mem(rax, 1*8*4), ymm9) \
	vmovups(mem(rax, 2*8*4), ymm10) \
	vmovups(mem(rax, 3*8*4), ymm11) \
	vmovups(mem(rbx, 0*8*4), ymm12) \
	vmovups(mem(rbx, 1*8*4), ymm13) \
	vmovups(mem(rbx, 2*8*4), ymm14) \
	vmovups(mem(rbx, 3*8*4), ymm15) \
	vmulps(ymm8, ymm12, ymm8) \
	vmulps(ymm9, ymm13, ymm9) \
	vmulps(ymm10, ymm14, ymm10) \
	vmulps(ymm11, ymm15, ymm11) \
	vaddps(ymm8, ymm0, ymm0) \
	vaddps(ymm9, ymm1, ymm1) \
	vaddps(ymm10, ymm2, ymm2) \
	vaddps(ymm11, ymm3, ymm3) \
	vmovups(mem(rax, 4*8*4), ymm8) \
	vmovups(mem(rax, 5*8*4), ymm9) \
	vmovups(mem(rax, 6*8*4), ymm10) \
	vmovups(mem(rax, 7*8*4), ymm11) \
	vmovups(mem(rbx, 4*8*4), ymm12) \
	vmovups(mem(rbx, 5*8*4), ymm13) \
	vmovups(mem(rbx, 6*8*4), ymm14) \
	vmovups(mem(rbx, 7*8*4), ymm15) \
	vmulps(ymm8, ymm12, ymm8) \
	vmulps(ymm9, ymm13, ymm9) \
	vmulps(ymm10, ymm14, ymm10) \
	vmulps(ymm11, ymm15, ymm11) \
	vaddps(ymm8, ymm4, ymm4) \
	vaddps(ymm9, ymm5, ymm5) \
	vaddps(ymm10, ymm6, ymm6) \
	vaddps(ymm11, ymm7, ymm7)
#define DWCONV_MAIN_LOOP_FP16() \
	vmovups(mem(rax, 0*8*2), xmm8) \
	vmovups(mem(rax, 1*8*2), xmm9) \
	vmovups(mem(rax, 2*8*2), xmm10) \
	vmovups(mem(rax, 3*8*2), xmm11) \
	vmovups(mem(rbx, 0*8*2), xmm12) \
	vmovups(mem(rbx, 1*8*2), xmm13) \
	vmovups(mem(rbx, 2*8*2), xmm14) \
	vmovups(mem(rbx, 3*8*2), xmm15) \
	vcvtph2ps(xmm8, ymm8) \
	vcvtph2ps(xmm9, ymm9) \
	vcvtph2ps(xmm10, ymm10) \
	vcvtph2ps(xmm11, ymm11) \
	vcvtph2ps(xmm12, ymm12) \
	vcvtph2ps(xmm13, ymm13) \
	vcvtph2ps(xmm14, ymm14) \
	vcvtph2ps(xmm15, ymm15) \
	vmulps(ymm8, ymm12, ymm8) \
	vmulps(ymm9, ymm13, ymm9) \
	vmulps(ymm10, ymm14, ymm10) \
	vmulps(ymm11, ymm15, ymm11) \
	vaddps(ymm8, ymm0, ymm0) \
	vaddps(ymm9, ymm1, ymm1) \
	vaddps(ymm10, ymm2, ymm2) \
	vaddps(ymm11, ymm3, ymm3) \
	vmovups(mem(rax, 4*8*2), xmm8) \
	vmovups(mem(rax, 5*8*2), xmm9) \
	vmovups(mem(rax, 6*8*2), xmm10) \
	vmovups(mem(rax, 7*8*2), xmm11) \
	vmovups(mem(rbx, 4*8*2), xmm12) \
	vmovups(mem(rbx, 5*8*2), xmm13) \
	vmovups(mem(rbx, 6*8*2), xmm14) \
	vmovups(mem(rbx, 7*8*2), xmm15) \
	vcvtph2ps(xmm8, ymm8) \
	vcvtph2ps(xmm9, ymm9) \
	vcvtph2ps(xmm10, ymm10) \
	vcvtph2ps(xmm11, ymm11) \
	vcvtph2ps(xmm12, ymm12) \
	vcvtph2ps(xmm13, ymm13) \
	vcvtph2ps(xmm14, ymm14) \
	vcvtph2ps(xmm15, ymm15) \
	vmulps(ymm8, ymm12, ymm8) \
	vmulps(ymm9, ymm13, ymm9) \
	vmulps(ymm10, ymm14, ymm10) \
	vmulps(ymm11, ymm15, ymm11) \
	vaddps(ymm8, ymm4, ymm4) \
	vaddps(ymm9, ymm5, ymm5) \
	vaddps(ymm10, ymm6, ymm6) \
	vaddps(ymm11, ymm7, ymm7)

#define DWCONV_LOAD_BIAS_FP32() \
	vmovups(mem(rdx, 0*8*4), ymm0) \
	vmovups(mem(rdx, 1*8*4), ymm1) \
	vmovups(mem(rdx, 2*8*4), ymm2) \
	vmovups(mem(rdx, 3*8*4), ymm3) \
	vmovups(mem(rdx, 4*8*4), ymm4) \
	vmovups(mem(rdx, 5*8*4), ymm5) \
	vmovups(mem(rdx, 6*8*4), ymm6) \
	vmovups(mem(rdx, 7*8*4), ymm7)
#define DWCONV_LOAD_BIAS_FP16() \
	vmovups(mem(rdx, 0*8*2), xmm0) \
	vmovups(mem(rdx, 1*8*2), xmm1) \
	vmovups(mem(rdx, 2*8*2), xmm2) \
	vmovups(mem(rdx, 3*8*2), xmm3) \
	vmovups(mem(rdx, 4*8*2), xmm4) \
	vmovups(mem(rdx, 5*8*2), xmm5) \
	vmovups(mem(rdx, 6*8*2), xmm6) \
	vmovups(mem(rdx, 7*8*2), xmm7) \
	vcvtph2ps(xmm0, ymm0) \
	vcvtph2ps(xmm1, ymm1) \
	vcvtph2ps(xmm2, ymm2) \
	vcvtph2ps(xmm3, ymm3) \
	vcvtph2ps(xmm4, ymm4) \
	vcvtph2ps(xmm5, ymm5) \
	vcvtph2ps(xmm6, ymm6) \
	vcvtph2ps(xmm7, ymm7)

#define DWCONV_STORE_OUTPUT_FP32() \
	vmovups(ymm0, mem(rcx, 0*8*4)) \
	vmovups(ymm1, mem(rcx, 1*8*4)) \
	vmovups(ymm2, mem(rcx, 2*8*4)) \
	vmovups(ymm3, mem(rcx, 3*8*4)) \
	vmovups(ymm4, mem(rcx, 4*8*4)) \
	vmovups(ymm5, mem(rcx, 5*8*4)) \
	vmovups(ymm6, mem(rcx, 6*8*4)) \
	vmovups(ymm7, mem(rcx, 7*8*4))
#define DWCONV_STORE_OUTPUT_FP16() \
	vcvtps2ph(imm(0x03), ymm0, xmm0) \
	vcvtps2ph(imm(0x03), ymm1, xmm1) \
	vcvtps2ph(imm(0x03), ymm2, xmm2) \
	vcvtps2ph(imm(0x03), ymm3, xmm3) \
	vcvtps2ph(imm(0x03), ymm4, xmm4) \
	vcvtps2ph(imm(0x03), ymm5, xmm5) \
	vcvtps2ph(imm(0x03), ymm6, xmm6) \
	vcvtps2ph(imm(0x03), ymm7, xmm7) \
	vmovups(xmm0, mem(rcx, 0*8*2)) \
	vmovups(xmm1, mem(rcx, 1*8*2)) \
	vmovups(xmm2, mem(rcx, 2*8*2)) \
	vmovups(xmm3, mem(rcx, 3*8*2)) \
	vmovups(xmm4, mem(rcx, 4*8*2)) \
	vmovups(xmm5, mem(rcx, 5*8*2)) \
	vmovups(xmm6, mem(rcx, 6*8*2)) \
	vmovups(xmm7, mem(rcx, 7*8*2)) \

	void depthwise_conv_avx_10x8(Matrix &output, const Matrix &input, const Matrix &weights, const Matrix &bias, const int *args,
			void *workspace) noexcept
	{
		assert(args != nullptr);
		const int batch_size = args[0];
		const int height = args[1];
		const int width = args[2];
		const uint64_t channels = args[3];
		const int kernel_height = args[4];
		const int kernel_width = args[5];

		const int padding_h = (kernel_height - 1) / 2;
		const int padding_w = (kernel_width - 1) / 2;

		assert(output.is_fp32() || output.is_fp16());
		assert(input.is_fp32() || input.is_fp16());
		assert(weights.is_fp32() || weights.is_fp16());
		assert(bias.is_fp32() || bias.is_fp16());
		assert(channels % 64 == 0);

		const uint8_t *b_ptr = reinterpret_cast<const uint8_t*>(bias.data());

		const uint64_t stride_w = input.stride_in_bytes();

		for (int i = 0; i < output.rows(); i++)
		{
			uint8_t *output_ptr = reinterpret_cast<uint8_t*>(output.data()) + i * output.stride_in_bytes();

			const int origin_b = reinterpret_cast<int*>(workspace)[3 * i + 0];
			const int origin_h = reinterpret_cast<int*>(workspace)[3 * i + 1];
			const int origin_w = reinterpret_cast<int*>(workspace)[3 * i + 2];

			const int kh0 = padding_h - std::min(origin_h, padding_h);
			const int kh1 = padding_h + std::min(height - 1 - origin_h, padding_h);

			const int kw0 = padding_w - std::min(origin_w, padding_w);
			const int kw1 = padding_w + std::min(width - 1 - origin_w, padding_w);

			const uint64_t elements_h = 1 + kh1 - kh0;
			const uint64_t elements_w = 1 + kw1 - kw0;

			const int h0 = origin_h + kh0 - padding_h;
			const int w0 = origin_w + kw0 - padding_w;

			const uint64_t in_stride_h = (width - elements_w) * stride_w;
			const uint64_t w_stride_h = (kernel_width - elements_w) * stride_w;

			const uint8_t *in_ptr = reinterpret_cast<const uint8_t*>(input.data()) + ((origin_b * height + h0) * width + w0) * stride_w;
			const uint8_t *w_ptr = reinterpret_cast<const uint8_t*>(weights.data()) + (kh0 * kernel_width + kw0) * stride_w;

			const uint64_t in_stride_back = elements_h * width * stride_w;
			const uint64_t w_stride_back = elements_h * kernel_width * stride_w;

			const uint64_t c_iter = channels / 64;

			if (input.is_fp32())
			{
				begin_asm()
				movq(var(input_ptr), rax)
				movq(var(weights_ptr), rbx)
				movq(var(output_ptr), rcx)
				movq(var(bias_ptr), rdx)

				movq(var(stride_w), r11)

				movq(var(in_stride_h), r12)
				movq(var(w_stride_h), r13)

				movq(var(in_stride_back), r14)
				movq(var(w_stride_back), r15)

				movq(var(c_iter), r8)
				label(CHANNELLOOP)

				DWCONV_LOAD_BIAS_FP32()

				movq(var(elements_h), r9)
				label(VERTICALLOOP)

				movq(var(elements_w), r10)
				label(HORIZONTALLOOP)

				DWCONV_MAIN_LOOP_FP32()

				add(r11, rax) // add channels
				add(r11, rbx)// add channels
				dec(r10)
				jne(HORIZONTALLOOP)

				add(r12, rax)// add channels
				add(r13, rbx)// add channels
				dec(r9)
				jne(VERTICALLOOP)

				sub(r14, rax)
				sub(r15, rbx)
				add(imm(8*8*4), rax)
				add(imm(8*8*4), rbx)

				DWCONV_STORE_OUTPUT_FP32()

				add(imm(8*8*4), rcx)
				add(imm(8*8*4), rdx)
				dec(r8)
				jne(CHANNELLOOP)

				vzeroupper()

				end_asm(
						:// outputs
						:// inputs
						[input_ptr] "m"(in_ptr),
						[weights_ptr] "m"(w_ptr),
						[output_ptr] "m"(output_ptr),
						[bias_ptr] "m"(b_ptr),
						[stride_w] "m"(stride_w),
						[in_stride_h] "m"(in_stride_h),
						[w_stride_h] "m"(w_stride_h),
						[in_stride_back] "m"(in_stride_back),
						[w_stride_back] "m"(w_stride_back),
						[c_iter] "m"(c_iter),
						[elements_h] "m"(elements_h),
						[elements_w] "m"(elements_w)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
						"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				if (output.is_fp16())
				{
					begin_asm()
					movq(var(input_ptr), rax)
					movq(var(weights_ptr), rbx)
					movq(var(output_ptr), rcx)
					movq(var(bias_ptr), rdx)

					movq(var(stride_w), r11)

					movq(var(in_stride_h), r12)
					movq(var(w_stride_h), r13)

					movq(var(in_stride_back), r14)
					movq(var(w_stride_back), r15)

					movq(var(c_iter), r8)
					label(CHANNELLOOP)

					DWCONV_LOAD_BIAS_FP16()

					movq(var(elements_h), r9)
					label(VERTICALLOOP)

					movq(var(elements_w), r10)
					label(HORIZONTALLOOP)

					DWCONV_MAIN_LOOP_FP16()

					add(r11, rax) // add channels
					add(r11, rbx)// add channels
					dec(r10)
					jne(HORIZONTALLOOP)

					add(r12, rax)// add channels
					add(r13, rbx)// add channels
					dec(r9)
					jne(VERTICALLOOP)

					sub(r14, rax)
					sub(r15, rbx)
					add(imm(8*8*2), rax)
					add(imm(8*8*2), rbx)

					DWCONV_STORE_OUTPUT_FP16()

					add(imm(8*8*2), rcx)
					add(imm(8*8*2), rdx)
					dec(r8)
					jne(CHANNELLOOP)

					vzeroupper()

					end_asm(
							:// outputs
							:// inputs
							[input_ptr] "m"(in_ptr),
							[weights_ptr] "m"(w_ptr),
							[output_ptr] "m"(output_ptr),
							[bias_ptr] "m"(b_ptr),
							[stride_w] "m"(stride_w),
							[in_stride_h] "m"(in_stride_h),
							[w_stride_h] "m"(w_stride_h),
							[in_stride_back] "m"(in_stride_back),
							[w_stride_back] "m"(w_stride_back),
							[c_iter] "m"(c_iter),
							[elements_h] "m"(elements_h),
							[elements_w] "m"(elements_w)
							:// clobbers
							"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
							"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
				}
				else
				{
					begin_asm()
					movq(var(input_ptr), rax)
					movq(var(weights_ptr), rbx)
					movq(var(output_ptr), rcx)
					movq(var(bias_ptr), rdx)

					movq(var(stride_w), r11)

					movq(var(in_stride_h), r12)
					movq(var(w_stride_h), r13)

					movq(var(in_stride_back), r14)
					movq(var(w_stride_back), r15)

					movq(var(c_iter), r8)
					label(CHANNELLOOP)

					DWCONV_LOAD_BIAS_FP16()

					movq(var(elements_h), r9)
					label(VERTICALLOOP)

					movq(var(elements_w), r10)
					label(HORIZONTALLOOP)

					DWCONV_MAIN_LOOP_FP16()

					add(r11, rax) // add channels
					add(r11, rbx)// add channels
					dec(r10)
					jne(HORIZONTALLOOP)

					add(r12, rax)// add channels
					add(r13, rbx)// add channels
					dec(r9)
					jne(VERTICALLOOP)

					sub(r14, rax)
					sub(r15, rbx)
					add(imm(8*8*2), rax)
					add(imm(8*8*2), rbx)

					DWCONV_STORE_OUTPUT_FP32()

					add(imm(8*8*4), rcx)
					add(imm(8*8*2), rdx)
					dec(r8)
					jne(CHANNELLOOP)

					vzeroupper()

					end_asm(
							:// outputs
							:// inputs
							[input_ptr] "m"(in_ptr),
							[weights_ptr] "m"(w_ptr),
							[output_ptr] "m"(output_ptr),
							[bias_ptr] "m"(b_ptr),
							[stride_w] "m"(stride_w),
							[in_stride_h] "m"(in_stride_h),
							[w_stride_h] "m"(w_stride_h),
							[in_stride_back] "m"(in_stride_back),
							[w_stride_back] "m"(w_stride_back),
							[c_iter] "m"(c_iter),
							[elements_h] "m"(elements_h),
							[elements_w] "m"(elements_w)
							:// clobbers
							"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
							"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
				}
			}
		}
	}
	void fused_conv_block_stage_1_avx_10x8(Fragment &temp, const Fragment &A, const Fragment &B, const Fragment &bias) noexcept
	{
		assert(temp.is_fp32());
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(bias.is_fp32());

		assert(temp.is_packed());
		assert(A.is_packed());
		assert(B.is_packed());
		assert(bias.is_packed());

		assert(cpu::is_aligned(A.data(), 32));
		assert(cpu::is_aligned(B.data(), 32));
		assert(cpu::is_aligned(bias.data(), 32));

		assert(A.rows() == B.rows());
		assert(A.stride() == 10);
		assert(B.stride() == 8);
		assert(temp.columns() == A.columns());
		assert(temp.rows() == B.columns());

		const void *A_ptr = A.data();
		const void *B_ptr = B.data();
		void *temp_ptr = temp.data();
		const void *bias_ptr = bias.data();

		uint64_t k_iter = A.rows() / 4;
		uint64_t k_left = A.rows() % 4;

		begin_asm()
		movq(var(A_ptr), rax)
		movq(var(B_ptr), rbx)
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14) // load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_10xFP32_8xFP32(0)
		SUB_KERNEL_10xFP32_8xFP32(1)
		SUB_KERNEL_10xFP32_8xFP32(2)
		SUB_KERNEL_10xFP32_8xFP32(3)

		add(imm(4*10*4), rax)// 4 iterations x 10 elements x 4 bytes
		add(imm(4*8*4), rbx)// 4 iterations x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_10xFP32_8xFP32(0)
		add(imm(1*10*4), rax)// 1 iteration x 10 elements x 4 bytes
		add(imm(1*8*4), rbx)// 1 iteration x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(bias_ptr), rax)// load address of bias pointer
		vmovaps(mem(rax), ymm2)// load bias value
		ADD_BIAS_10x8xFP32(ymm2)
		RELU_10x8xFP32()

		// transpose and store into packed fragment of D
		movq(var(temp_ptr), rcx)// temp pointer is in rcx
		vunpcklps(ymm7, ymm6, ymm0)
		vunpckhps(ymm7, ymm6, ymm1)
		vextractf128(imm(0x1), ymm0, xmm2)// e4 f4 e5 f5
		vextractf128(imm(0x1), ymm1, xmm3)// e6 f6 e7 f7

		vmovlpd(xmm0, mem(rcx, 4*(0*10+0)))
		vmovhpd(xmm0, mem(rcx, 4*(1*10+0)))
		vmovlpd(xmm1, mem(rcx, 4*(2*10+0)))
		vmovhpd(xmm1, mem(rcx, 4*(3*10+0)))
		vmovlpd(xmm2, mem(rcx, 4*(4*10+0)))
		vmovhpd(xmm2, mem(rcx, 4*(5*10+0)))
		vmovlpd(xmm3, mem(rcx, 4*(6*10+0)))
		vmovhpd(xmm3, mem(rcx, 4*(7*10+0)))

		AVX_8x8_TRANSPOSE_INV()
		vmovups(ymm0, mem(rcx, (0*10+2)*4))
		vmovups(ymm1, mem(rcx, (1*10+2)*4))
		vmovups(ymm2, mem(rcx, (2*10+2)*4))
		vmovups(ymm3, mem(rcx, (3*10+2)*4))
		vmovups(ymm4, mem(rcx, (4*10+2)*4))
		vmovups(ymm5, mem(rcx, (5*10+2)*4))
		vmovups(ymm6, mem(rcx, (6*10+2)*4))
		vmovups(ymm7, mem(rcx, (7*10+2)*4))
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[A_ptr] "m"(A_ptr),
				[B_ptr] "m"(B_ptr),
				[temp_ptr] "m"(temp_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[bias_ptr] "m"(bias_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r11", "%r13", "%r14", "%r15")
	}

} /* namespace ml */

