/*
 * avx2_gemm_kernels.cpp
 *
 *  Created on: May 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include "Fragment.hpp"
#include "Matrix.hpp"
#include "gemm_kernels.hpp"
#include "../utils.hpp"
#include "../fp16.hpp"

#include <cinttypes>
#include <cassert>

#include "../assembly_macros.hpp"
#include "common_operations.hpp"

#define ZERO_ACCUMULATORS()\
	vxorps(ymm4, ymm4, ymm4)\
	vxorps(ymm5, ymm5, ymm5)\
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


#define SUB_KERNEL_12xFP32_8xFP32(n) \
	vmovaps(mem(rbx, n*8*4), ymm2)\
	vpermilps(imm(0xB1), ymm2, ymm3)\
	vbroadcastsd(mem(rax, (12*n+0)*4), ymm0)\
	vbroadcastsd(mem(rax, (12*n+2)*4), ymm1)\
	vfmadd231ps(ymm0, ymm2, ymm4)\
	vfmadd231ps(ymm0, ymm3, ymm5)\
	vfmadd231ps(ymm1, ymm2, ymm6)\
	vfmadd231ps(ymm1, ymm3, ymm7)\
	vbroadcastsd(mem(rax, (12*n+4)*4), ymm0)\
	vbroadcastsd(mem(rax, (12*n+6)*4), ymm1)\
	vfmadd231ps(ymm0, ymm2, ymm8)\
	vfmadd231ps(ymm0, ymm3, ymm9)\
	vfmadd231ps(ymm1, ymm2, ymm10)\
	vfmadd231ps(ymm1, ymm3, ymm11)\
	vbroadcastsd(mem(rax, (12*n+8)*4), ymm0)\
	vbroadcastsd(mem(rax, (12*n+10)*4), ymm1)\
	vfmadd231ps(ymm0, ymm2, ymm12)\
	vfmadd231ps(ymm0, ymm3, ymm13)\
	vfmadd231ps(ymm1, ymm2, ymm14)\
	vfmadd231ps(ymm1, ymm3, ymm15)

#define ADD_BIAS_12x8xFP32(reg)\
	vaddps(reg, ymm4, ymm4) \
	vaddps(reg, ymm5, ymm5) \
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

#define LOAD_ADD_3x8xFP32(beta, reg0, reg1, reg2)\
	vmovups(mem(rcx), ymm1)\
	vmovups(mem(rcx, r14, 1), ymm2)\
	vmovups(mem(rcx, r14, 2), ymm3)\
	vfmadd231ps(ymm1, beta, reg0)\
	vfmadd231ps(ymm2, beta, reg1)\
	vfmadd231ps(ymm3, beta, reg2)\
	add(r15, rcx)

#define LOAD_ADD_8xFP16(beta, reg)\
	vmovups(mem(rcx), xmm2)\
	vcvtph2ps(xmm2, ymm2)\
	vfmadd231ps(ymm2, beta, reg)\
	add(r14, rcx)

#define STORE_3x8xFP32(reg0, reg1, reg2)\
	vmovups(reg0, mem(rcx))\
	vmovups(reg1, mem(rcx, r14, 1))\
	vmovups(reg2, mem(rcx, r14, 2))\
	add(r15, rcx)
#define STORE_4x8xFP32(reg0, reg1, reg2, reg3)\
	vmovups(reg0, mem(rcx)) \
	vmovups(reg1, mem(rcx, r14, 1)) \
	vmovups(reg2, mem(rcx, r14, 2)) \
	vmovups(reg3, mem(rcx, r13, 1)) \
	add(r15, rcx)

#define PERMUTE_AND_SCALE_4x8xFP32(reg0, reg1, reg2, reg3)\
	vpermilps(imm(0xB1), reg1, reg1)\
	vpermilps(imm(0xB1), reg3, reg3)\
	vblendps(imm(0x55), reg0, reg1, ymm2)\
	vblendps(imm(0xAA), reg0, reg1, reg1)\
	vblendps(imm(0x55), reg2, reg3, ymm3)\
	vblendps(imm(0xAA), reg2, reg3, reg3)\
	vmulps(ymm1, ymm2, reg0)\
	vmulps(ymm1, reg1, reg1)\
	vmulps(ymm1, ymm3, reg2)\
	vmulps(ymm1, reg3, reg3)

#define CONVERT_ACCUMULATORS_TO_FP16()\
	vcvtps2ph(imm(0x03), ymm4, xmm4)\
	vcvtps2ph(imm(0x03), ymm5, xmm5)\
	vcvtps2ph(imm(0x03), ymm6, xmm6)\
	vcvtps2ph(imm(0x03), ymm7, xmm7)\
	vcvtps2ph(imm(0x03), ymm8, xmm8)\
	vcvtps2ph(imm(0x03), ymm9, xmm9)\
	vcvtps2ph(imm(0x03), ymm10, xmm10)\
	vcvtps2ph(imm(0x03), ymm11, xmm11)\
	vcvtps2ph(imm(0x03), ymm12, xmm12)\
	vcvtps2ph(imm(0x03), ymm13, xmm13)\
	vcvtps2ph(imm(0x03), ymm14, xmm14)\
	vcvtps2ph(imm(0x03), ymm15, xmm15)

#define LOAD_ADD_3x8xFP16(beta, reg0, reg1, reg2)\
	vmovups(mem(rcx), xmm1)\
	vmovups(mem(rcx, r14, 1), xmm2)\
	vmovups(mem(rcx, r14, 2), xmm3)\
	vcvtph2ps(xmm1, ymm1)\
	vcvtph2ps(xmm2, ymm2)\
	vcvtph2ps(xmm3, ymm3)\
	vfmadd231ps(ymm1, beta, reg0)\
	vfmadd231ps(ymm2, beta, reg1)\
	vfmadd231ps(ymm3, beta, reg2)\
	add(r15, rcx)

#define STORE_3x8xFP16(reg0, reg1, reg2)\
	vmovups(reg0, mem(rcx))\
	vmovups(reg1, mem(rcx, r14, 1))\
	vmovups(reg2, mem(rcx, r14, 2))\
	add(r15, rcx)
#define STORE_4x8xFP16(reg0, reg1, reg2, reg3)\
	vmovups(reg0, mem(rcx)) \
	vmovups(reg1, mem(rcx, r14, 1)) \
	vmovups(reg2, mem(rcx, r14, 2)) \
	vmovups(reg3, mem(rcx, r13, 1)) \
	add(r15, rcx)

#define RELU_12x8xFP32()\
	vxorps(ymm0, ymm0, ymm0)\
	vmaxps(ymm0, ymm4, ymm4)\
	vmaxps(ymm0, ymm5, ymm5)\
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

/*
 * MHA helper macros
 */
#define ADD_BIAS_4x8xFP32(reg0, reg1, reg2, reg3) \
	vmovaps(mem(rbx), ymm0) \
	vmovaps(mem(rbx, r14, 1), ymm1) \
	vmovaps(mem(rbx, r14, 2), ymm2) \
	vmovaps(mem(rbx, r13, 1), ymm3) \
	add(r15, rbx) \
	vaddps(ymm0, reg0, reg0) \
	vaddps(ymm1, reg1, reg1) \
	vaddps(ymm2, reg2, reg2) \
	vaddps(ymm3, reg3, reg3)
#define EXP_FIRST_STAGE_12x8xFP32() \
	movq(imm(0x4ab8aa3b4ab8aa3b), r14) \
	vmovq(r14, xmm0) \
	vpermpd(imm(0), ymm0, ymm0) \
	vmulps(ymm0, ymm4, ymm4) \
	vmulps(ymm0, ymm5, ymm5) \
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
	vcvtps2dq(ymm4, ymm4) \
	vcvtps2dq(ymm5, ymm5) \
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
#define EXP_SECOND_STAGE_3x8xFP32(reg0, reg1, reg2) \
	vpsubd(reg0, ymm0, ymm1) \
	vpsubd(reg1, ymm0, ymm2) \
	vpsubd(reg2, ymm0, ymm3) \
	vrcpps(ymm1, ymm1) \
	vrcpps(ymm2, ymm2) \
	vrcpps(ymm3, ymm3) \
	vpaddd(reg0, ymm0, reg0) \
	vpaddd(reg1, ymm0, reg1) \
	vpaddd(reg2, ymm0, reg2) \
	vmulps(ymm1, reg0, reg0) \
	vmulps(ymm2, reg1, reg1) \
	vmulps(ymm3, reg2, reg2)
#define PERMUTE_4x8xFP32(reg0, reg1, reg2, reg3) \
	vpermilps(imm(0xB1), reg1, reg1) \
	vpermilps(imm(0xB1), reg3, reg3) \
	vblendps(imm(0x55), reg0, reg1, ymm2) \
	vblendps(imm(0xAA), reg0, reg1, reg1) \
	vblendps(imm(0x55), reg2, reg3, ymm3) \
	vblendps(imm(0xAA), reg2, reg3, reg3) \
	vmovaps(ymm2, reg0) \
	vmovaps(ymm3, reg2)
#define REDUCE_SUM() \
	vaddps(ymm4, ymm0, ymm0) \
	vaddps(ymm5, ymm1, ymm1) \
	vaddps(ymm6, ymm2, ymm2) \
	vaddps(ymm7, ymm3, ymm3) \
	vaddps(ymm1, ymm0, ymm0) \
	vaddps(ymm3, ymm2, ymm2) \
	vaddps(ymm2, ymm0, ymm0)

namespace ml
{
	using namespace ml::cpu;

	void gemm_avx2_fma_12x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
		assert(A.dtype() == DTYPE_FLOAT32);
		assert(B.dtype() == DTYPE_FLOAT32);
		assert(C.dtype() == DTYPE_FLOAT32);
		assert(D.dtype() == DTYPE_FLOAT32);
		assert(A.rows() == B.rows());
		assert(A.stride() == 12);
		assert(B.stride() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(A.data(), 32));
		assert(cpu::is_aligned(B.data(), 32));
		assert(beta_ptr != nullptr);
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 32));
		}

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float *C_ptr = C.data<float>();
		float *D_ptr = D.data<float>();
		const float *bias_ptr = bias.is_packed() ? bias.data<float>() : nullptr;

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
		SUB_KERNEL_12xFP32_8xFP32(0)
		SUB_KERNEL_12xFP32_8xFP32(1)
		SUB_KERNEL_12xFP32_8xFP32(2)
		SUB_KERNEL_12xFP32_8xFP32(3)

		add(imm(4*12*4), rax)// 4 iterations x 12 elements x 4 bytes
		add(imm(4*8*4), rbx)// 4 iterations x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_12xFP32_8xFP32(0)
		add(imm(1*12*4), rax)// 1 iteration x 12 elements x 4 bytes
		add(imm(1*8*4), rbx)// 1 iteration x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rax), ymm1)
		vbroadcastss(mem(rbx), ymm0)
		// permute back to original layout
		PERMUTE_AND_SCALE_4x8xFP32(ymm4, ymm5, ymm6, ymm7)
		PERMUTE_AND_SCALE_4x8xFP32(ymm8, ymm9, ymm10, ymm11)
		PERMUTE_AND_SCALE_4x8xFP32(ymm12, ymm13, ymm14, ymm15)

		// load address of bias pointer
		movq(var(bias_ptr), rax)
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax), ymm2)// load bias
		ADD_BIAS_12x8xFP32(ymm2)
		label(AFTER_BIAS)

		vxorps(ymm1, ymm1, ymm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(AFTER_LOAD_C)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)

		LOAD_ADD_3x8xFP32(ymm0, ymm4, ymm5, ymm6)
		LOAD_ADD_3x8xFP32(ymm0, ymm7, ymm8, ymm9)
		LOAD_ADD_3x8xFP32(ymm0, ymm10, ymm11, ymm12)
		LOAD_ADD_3x8xFP32(ymm0, ymm13, ymm14, ymm15)
		label(AFTER_LOAD_C)

		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(AFTER_RELU)
		RELU_12x8xFP32()
		label(AFTER_RELU)

		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r13)// r13 = r14
		sal(imm(1), r13)// r13 = 2 * r14 (2 * stride)
		add(r14, r13)// r13 = 2 * r14 + r14 = 3 * r14 (3*D_stride)
		movq(r14, r15)// r15 = r14
		sal(imm(2), r15)// r15 = 4 * r14 (4 * stride)

		STORE_4x8xFP32(ymm4, ymm5, ymm6, ymm7)
		STORE_4x8xFP32(ymm8, ymm9, ymm10, ymm11)
		STORE_4x8xFP32(ymm12, ymm13, ymm14, ymm15)

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
				[bias_ptr] "m"(bias_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r13", "%r14", "%r15")
	}
	void gemm_avx2_fma_12x8_fp32_fp16(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C, const Fragment &bias, bool use_relu) noexcept
	{
		assert(A.dtype() == DTYPE_FLOAT32);
		assert(B.dtype() == DTYPE_FLOAT32);
		assert(C.dtype() == DTYPE_FLOAT16);
		assert(D.dtype() == DTYPE_FLOAT16);
		assert(A.rows() == B.rows());
		assert(A.stride() == 12);
		assert(B.stride() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(A.data(), 32));
		assert(cpu::is_aligned(B.data(), 32));
		assert(beta_ptr != nullptr);
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 32));
		}

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float16 *C_ptr = C.data<float16>();
		float16 *D_ptr = D.data<float16>();
		const float *bias_ptr = bias.is_packed() ? bias.data<float>() : nullptr;

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
		SUB_KERNEL_12xFP32_8xFP32(0)
		SUB_KERNEL_12xFP32_8xFP32(1)
		SUB_KERNEL_12xFP32_8xFP32(2)
		SUB_KERNEL_12xFP32_8xFP32(3)

		add(imm(4*12*4), rax)// 4 iterations x 12 elements x 4 bytes
		add(imm(4*8*4), rbx)// 4 iterations x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_12xFP32_8xFP32(0)
		add(imm(1*12*4), rax)// 1 iteration x 12 elements x 4 bytes
		add(imm(1*8*4), rbx)// 1 iteration x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rax), ymm1)
		vbroadcastss(mem(rbx), ymm0)
		// permute back to original layout
		PERMUTE_AND_SCALE_4x8xFP32(ymm4, ymm5, ymm6, ymm7)
		PERMUTE_AND_SCALE_4x8xFP32(ymm8, ymm9, ymm10, ymm11)
		PERMUTE_AND_SCALE_4x8xFP32(ymm12, ymm13, ymm14, ymm15)

		movq(var(bias_ptr), rax)// load address of bias pointer
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax), ymm2)// load bias
		ADD_BIAS_12x8xFP32(ymm2)
		label(AFTER_BIAS)

		vxorps(ymm1, ymm1, ymm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(AFTER_LOAD_C)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)

		LOAD_ADD_3x8xFP16(ymm0, ymm4, ymm5, ymm6)
		LOAD_ADD_3x8xFP16(ymm0, ymm7, ymm8, ymm9)
		LOAD_ADD_3x8xFP16(ymm0, ymm10, ymm11, ymm12)
		LOAD_ADD_3x8xFP16(ymm0, ymm13, ymm14, ymm15)
		label(AFTER_LOAD_C)

		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(AFTER_RELU)
		RELU_12x8xFP32()
		label(AFTER_RELU)

		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r13)// r13 = r14
		sal(imm(1), r13)// r13 = 2 * r14 (2 * stride)
		add(r14, r13)// r13 = 2 * r14 + r14 = 3 * r14 (3*D_stride)
		movq(r14, r15)// r15 = r14
		sal(imm(2), r15)// r15 = 4 * r14 (4 * stride)

		CONVERT_ACCUMULATORS_TO_FP16()

		STORE_4x8xFP16(xmm4, xmm5, xmm6, xmm7)
		STORE_4x8xFP16(xmm8, xmm9, xmm10, xmm11)
		STORE_4x8xFP16(xmm12, xmm13, xmm14, xmm15)

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
				[bias_ptr] "m"(bias_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r13", "%r14", "%r15")
	}

	void pack_avx2_fma_12xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 12);
		assert(ml::cpu::is_aligned(dst.data(), 32));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = dst.rows() % 8;
		const uint64_t src_stride = src.stride() * sizeof(float);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
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
			vmovups(mem(rax, 8*4), xmm1)
			vmovups(mem(rax, r12, 1), ymm2)
			vmovups(mem(rax, r12, 1, 8*4), xmm3)
			vmovups(mem(rax, r12, 2), ymm4)
			vmovups(mem(rax, r12, 2, 8*4), xmm5)
			vmovups(mem(rax, r13, 1), ymm6)
			vmovups(mem(rax, r13, 1, 8*4), xmm7)
			add(r15, rax)
			vmovups(mem(rax), ymm8)
			vmovups(mem(rax, 8*4), xmm9)
			vmovups(mem(rax, r12, 1), ymm10)
			vmovups(mem(rax, r12, 1, 8*4), xmm11)
			vmovups(mem(rax, r12, 2), ymm12)
			vmovups(mem(rax, r12, 2, 8*4), xmm13)
			vmovups(mem(rax, r13, 1), ymm14)
			vmovups(mem(rax, r13, 1, 8*4), xmm15)
			add(r15, rax)

			vmovaps(ymm0, mem(rbx, (0*12+0)*4))
			vmovaps(xmm1, mem(rbx, (0*12+8)*4))
			vmovups(ymm2, mem(rbx, (1*12+0)*4))
			vmovaps(xmm3, mem(rbx, (1*12+8)*4))
			vmovaps(ymm4, mem(rbx, (2*12+0)*4))
			vmovaps(xmm5, mem(rbx, (2*12+8)*4))
			vmovups(ymm6, mem(rbx, (3*12+0)*4))
			vmovaps(xmm7, mem(rbx, (3*12+8)*4))
			vmovaps(ymm8, mem(rbx, (4*12+0)*4))
			vmovaps(xmm9, mem(rbx, (4*12+8)*4))
			vmovups(ymm10, mem(rbx, (5*12+0)*4))
			vmovaps(xmm11, mem(rbx, (5*12+8)*4))
			vmovaps(ymm12, mem(rbx, (6*12+0)*4))
			vmovaps(xmm13, mem(rbx, (6*12+8)*4))
			vmovups(ymm14, mem(rbx, (7*12+0)*4))
			vmovaps(xmm15, mem(rbx, (7*12+8)*4))

			add(imm(4*8*12), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED8)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			vmovups(mem(rax), ymm0)
			vmovups(mem(rax, 8*4), xmm1)
			add(r12, rax)// add stride to src pointer
			vmovups(ymm0, mem(rbx))
			vmovaps(xmm1, mem(rbx, 8*4))
			add(imm(4*1*12), rbx)// add stride to dst pointer

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
					"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
					"%r12", "%r13", "%r14", "%r15")
		}
		else
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
			// first 8x8 tile
			movq(rax, rcx)// tmp src pointer is in rcx

			// load 8x8 tile
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

			vmovaps(ymm8, mem(rbx, (0*12+0)*4))
			vmovups(ymm9, mem(rbx, (1*12+0)*4))
			vmovaps(ymm10, mem(rbx, (2*12+0)*4))
			vmovups(ymm11, mem(rbx, (3*12+0)*4))
			vmovaps(ymm12, mem(rbx, (4*12+0)*4))
			vmovups(ymm13, mem(rbx, (5*12+0)*4))
			vmovaps(ymm14, mem(rbx, (6*12+0)*4))
			vmovups(ymm15, mem(rbx, (7*12+0)*4))

			// load 4x8 tile
			vmovups(mem(rcx), ymm4)
			vmovups(mem(rcx, r12, 1), ymm5)
			vmovups(mem(rcx, r12, 2), ymm6)
			vmovups(mem(rcx, r13, 1), ymm7)

			AVX_4x8_TRANSPOSE()

			vmovaps(xmm4, mem(rbx, (0*12+8)*4))
			vmovaps(xmm5, mem(rbx, (1*12+8)*4))
			vmovaps(xmm6, mem(rbx, (2*12+8)*4))
			vmovaps(xmm7, mem(rbx, (3*12+8)*4))
			vmovaps(xmm0, mem(rbx, (4*12+8)*4))
			vmovaps(xmm1, mem(rbx, (5*12+8)*4))
			vmovaps(xmm2, mem(rbx, (6*12+8)*4))
			vmovaps(xmm3, mem(rbx, (7*12+8)*4))

			add(imm(4*8), rax)// add stride to src pointer
			add(imm(4*12*8), rbx)// add stride to dst pointer

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
			vmovss(mem(rcx, r12, 2), xmm10)
			vmovss(mem(rcx, r13, 1), xmm11)

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

			add(imm(4*1), rax)// add stride to src pointer
			add(imm(4*12*1), rbx)// add stride to dst pointer

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
	}
	void pack_avx2_fma_12xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 12);
		assert(cpu::is_aligned(dst.data(), 32));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = dst.rows() % 8;
		const uint64_t src_stride = src.stride() * sizeof(float16);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
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
			vmovups(mem(rax), xmm0)
			vmovsd(mem(rax, 8*2), xmm1)
			vmovups(mem(rax, r12, 1), xmm2)
			vmovsd(mem(rax, r12, 1, 8*2), xmm3)
			vmovups(mem(rax, r12, 2), xmm4)
			vmovsd(mem(rax, r12, 2, 8*2), xmm5)
			vmovups(mem(rax, r13, 1), xmm6)
			vmovsd(mem(rax, r13, 1, 8*2), xmm7)
			add(r15, rax)
			vmovups(mem(rax), xmm8)
			vmovsd(mem(rax, 8*2), xmm9)
			vmovups(mem(rax, r12, 1), xmm10)
			vmovsd(mem(rax, r12, 1, 8*2), xmm11)
			vmovups(mem(rax, r12, 2), xmm12)
			vmovsd(mem(rax, r12, 2, 8*2), xmm13)
			vmovups(mem(rax, r13, 1), xmm14)
			vmovsd(mem(rax, r13, 1, 8*2), xmm15)
			add(r15, rax)

			vcvtph2ps(xmm0, ymm0)
			vcvtph2ps(xmm1, xmm1)
			vcvtph2ps(xmm2, ymm2)
			vcvtph2ps(xmm3, xmm3)
			vcvtph2ps(xmm4, ymm4)
			vcvtph2ps(xmm5, xmm5)
			vcvtph2ps(xmm6, ymm6)
			vcvtph2ps(xmm7, xmm7)
			vcvtph2ps(xmm8, ymm8)
			vcvtph2ps(xmm9, xmm9)
			vcvtph2ps(xmm10, ymm10)
			vcvtph2ps(xmm11, xmm11)
			vcvtph2ps(xmm12, ymm12)
			vcvtph2ps(xmm13, xmm13)
			vcvtph2ps(xmm14, ymm14)
			vcvtph2ps(xmm15, xmm15)

			vmovaps(ymm0, mem(rbx, (0*12+0)*4))
			vmovaps(xmm1, mem(rbx, (0*12+8)*4))
			vmovups(ymm2, mem(rbx, (1*12+0)*4))
			vmovaps(xmm3, mem(rbx, (1*12+8)*4))
			vmovaps(ymm4, mem(rbx, (2*12+0)*4))
			vmovaps(xmm5, mem(rbx, (2*12+8)*4))
			vmovups(ymm6, mem(rbx, (3*12+0)*4))
			vmovaps(xmm7, mem(rbx, (3*12+8)*4))
			vmovaps(ymm8, mem(rbx, (4*12+0)*4))
			vmovaps(xmm9, mem(rbx, (4*12+8)*4))
			vmovups(ymm10, mem(rbx, (5*12+0)*4))
			vmovaps(xmm11, mem(rbx, (5*12+8)*4))
			vmovaps(ymm12, mem(rbx, (6*12+0)*4))
			vmovaps(xmm13, mem(rbx, (6*12+8)*4))
			vmovups(ymm14, mem(rbx, (7*12+0)*4))
			vmovaps(xmm15, mem(rbx, (7*12+8)*4))

			add(imm(4*8*12), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED8)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			vmovups(mem(rax), xmm0)
			vmovsd(mem(rax, 8*2), xmm1)
			add(r12, rax)// add stride to src pointer
			vcvtph2ps(xmm0, ymm0)
			vcvtph2ps(xmm1, xmm1)

			vmovups(ymm0, mem(rbx))
			vmovaps(xmm1, mem(rbx, 8*4))
			add(imm(4*1*12), rbx)// add stride to dst pointer

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
					"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
					"%r12", "%r13", "%r14", "%r15")
		}
		else
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
			// first 8x8 tile
			movq(rax, rcx)// tmp src pointer is in rcx

			// load 8x8 tile
			vmovups(mem(rcx), xmm0)
			vmovups(mem(rcx, r12, 1), xmm1)
			vmovups(mem(rcx, r12, 2), xmm2)
			vmovups(mem(rcx, r13, 1), xmm3)
			add(r15, rcx)
			vmovups(mem(rcx), xmm4)
			vmovups(mem(rcx, r12, 1), xmm5)
			vmovups(mem(rcx, r12, 2), xmm6)
			vmovups(mem(rcx, r13, 1), xmm7)
			add(r15, rcx)

			vcvtph2ps(xmm0, ymm0)
			vcvtph2ps(xmm1, ymm1)
			vcvtph2ps(xmm2, ymm2)
			vcvtph2ps(xmm3, ymm3)
			vcvtph2ps(xmm4, ymm4)
			vcvtph2ps(xmm5, ymm5)
			vcvtph2ps(xmm6, ymm6)
			vcvtph2ps(xmm7, ymm7)

			AVX_8x8_TRANSPOSE()

			vmovaps(ymm8, mem(rbx, (0*12+0)*4))
			vmovups(ymm9, mem(rbx, (1*12+0)*4))
			vmovaps(ymm10, mem(rbx, (2*12+0)*4))
			vmovups(ymm11, mem(rbx, (3*12+0)*4))
			vmovaps(ymm12, mem(rbx, (4*12+0)*4))
			vmovups(ymm13, mem(rbx, (5*12+0)*4))
			vmovaps(ymm14, mem(rbx, (6*12+0)*4))
			vmovups(ymm15, mem(rbx, (7*12+0)*4))

			// load 4x8 tile
			vmovups(mem(rcx), xmm4)
			vmovups(mem(rcx, r12, 1), xmm5)
			vmovups(mem(rcx, r12, 2), xmm6)
			vmovups(mem(rcx, r13, 1), xmm7)

			vcvtph2ps(xmm4, ymm4)
			vcvtph2ps(xmm5, ymm5)
			vcvtph2ps(xmm6, ymm6)
			vcvtph2ps(xmm7, ymm7)

			AVX_4x8_TRANSPOSE()

			vmovaps(xmm4, mem(rbx, (0*12+8)*4))
			vmovaps(xmm5, mem(rbx, (1*12+8)*4))
			vmovaps(xmm6, mem(rbx, (2*12+8)*4))
			vmovaps(xmm7, mem(rbx, (3*12+8)*4))
			vmovaps(xmm0, mem(rbx, (4*12+8)*4))
			vmovaps(xmm1, mem(rbx, (5*12+8)*4))
			vmovaps(xmm2, mem(rbx, (6*12+8)*4))
			vmovaps(xmm3, mem(rbx, (7*12+8)*4))

			add(imm(8*2), rax)// add stride to src pointer
			add(imm(12*8*4), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED8)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(rax, rcx)// tmp src pointer is in rcx

			movzw(mem(rcx), r8)
			movzw(mem(rcx, r12, 1), r9)
			movzw(mem(rcx, r12, 2), r10)
			movzw(mem(rcx, r13, 1), r11)
			vmovq(r8, xmm0)
			vmovq(r9, xmm1)
			vmovq(r10, xmm2)
			vmovq(r11, xmm3)
			add(r15, rcx)
			movzw(mem(rcx), r8)
			movzw(mem(rcx, r12, 1), r9)
			movzw(mem(rcx, r12, 2), r10)
			movzw(mem(rcx, r13, 1), r11)
			vmovq(r8, xmm4)
			vmovq(r9, xmm5)
			vmovq(r10, xmm6)
			vmovq(r11, xmm7)
			add(r15, rcx)
			movzw(mem(rcx), r8)
			movzw(mem(rcx, r12, 1), r9)
			movzw(mem(rcx, r12, 2), r10)
			movzw(mem(rcx, r13, 1), r11)
			vmovq(r8, xmm8)
			vmovq(r9, xmm9)
			vmovq(r10, xmm10)
			vmovq(r11, xmm11)

			vcvtph2ps(xmm0, xmm0)
			vcvtph2ps(xmm1, xmm1)
			vcvtph2ps(xmm2, xmm2)
			vcvtph2ps(xmm3, xmm3)
			vcvtph2ps(xmm4, xmm4)
			vcvtph2ps(xmm5, xmm5)
			vcvtph2ps(xmm6, xmm6)
			vcvtph2ps(xmm7, xmm7)
			vcvtph2ps(xmm8, xmm8)
			vcvtph2ps(xmm9, xmm9)
			vcvtph2ps(xmm10, xmm10)
			vcvtph2ps(xmm11, xmm11)

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

			add(imm(1*2), rax)// add stride to src pointer
			add(imm(12*1*4), rbx)// add stride to dst pointer

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
					"%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
		}
	}

	// multi-head attention (MHA) kernel
	void mha_qk_avx2_fma_12x8_fp32(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept
	{
		assert(Q.rows() == K.rows());
		assert(Q.stride() == 12);
		assert(K.stride() == 8);
		assert(temp.columns() == Q.columns());
		assert(temp.rows() == K.columns());
		assert(temp.stride() == 12);

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(Q.data(), 32));
		assert(cpu::is_aligned(K.data(), 32));
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 32));
		}
		if (softmax_sum.is_packed())
		{
			assert(cpu::is_aligned(softmax_sum.data(), 32));
		}

		const float *Q_ptr = Q.data<float>();
		const float *K_ptr = K.data<float>();
		float *temp_ptr = temp.data<float>();
		const float *bias_ptr = bias.data<float>();
		float *softmax_ptr = softmax_sum.data<float>();

		uint64_t k_iter = Q.rows() / 4;
		uint64_t k_left = Q.rows() % 4;
		const uint64_t bias_stride = bias.stride() * sizeof(float);

		begin_asm()
		movq(var(Q_ptr), rax) // lhs pointer is in rax
		movq(var(K_ptr), rbx)// rhs pointer is in rbx
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_12xFP32_8xFP32(0)
		SUB_KERNEL_12xFP32_8xFP32(1)
		SUB_KERNEL_12xFP32_8xFP32(2)
		SUB_KERNEL_12xFP32_8xFP32(3)

		add(imm(4*12*4), rax)
		add(imm(4*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_12xFP32_8xFP32(0)
		add(imm(1*12*4), rax)
		add(imm(1*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		vbroadcastss(mem(rax), ymm1)
		PERMUTE_AND_SCALE_4x8xFP32(ymm4, ymm5, ymm6, ymm7)
		PERMUTE_AND_SCALE_4x8xFP32(ymm8, ymm9, ymm10, ymm11)
		PERMUTE_AND_SCALE_4x8xFP32(ymm12, ymm13, ymm14, ymm15)

		movq(var(bias_ptr), rbx)// load address of bias pointer
		movq(var(bias_stride), r14)// load address of bias stride into r14
		movq(r14, r13)
		sal(imm(1), r13)// r13 = stride * 2
		add(r14, r13)// r13 == stride * 3
		movq(r14, r15)
		sal(imm(2), r15)// r15 = stride * 4
		ADD_BIAS_4x8xFP32(ymm4, ymm5, ymm6, ymm7)
		ADD_BIAS_4x8xFP32(ymm8, ymm9, ymm10, ymm11)
		ADD_BIAS_4x8xFP32(ymm12, ymm13, ymm14, ymm15)

		EXP_FIRST_STAGE_12x8xFP32()
		EXP_SECOND_STAGE_3x8xFP32(ymm4, ymm5, ymm6)
		EXP_SECOND_STAGE_3x8xFP32(ymm7, ymm8, ymm9)
		EXP_SECOND_STAGE_3x8xFP32(ymm10, ymm11, ymm12)
		EXP_SECOND_STAGE_3x8xFP32(ymm13, ymm14, ymm15)

		movq(var(temp_ptr), rbx)// temp pointer is in rbx
		movq(var(softmax_ptr), rcx)// softmax sum pointer is in rcx

		AVX_4x8_TRANSPOSE()
		vmovups(xmm4, mem(rbx, (0*12+0)*4))
		vmovups(xmm5, mem(rbx, (1*12+0)*4))
		vmovups(xmm6, mem(rbx, (2*12+0)*4))
		vmovups(xmm7, mem(rbx, (3*12+0)*4))
		vmovups(xmm0, mem(rbx, (4*12+0)*4))
		vmovups(xmm1, mem(rbx, (5*12+0)*4))
		vmovups(xmm2, mem(rbx, (6*12+0)*4))
		vmovups(xmm3, mem(rbx, (7*12+0)*4))

		REDUCE_SUM()// sum registers ymm0-ymm7 and place result in ymm0
		vmovaps(mem(rcx), xmm1)// load previous sum
		vaddps(xmm1, xmm0, xmm0)// add current sum
		vmovaps(xmm0, mem(rcx))

		AVX_8x8_TRANSPOSE_INV()
		vmovups(ymm0, mem(rbx, (0*12+4)*4))
		vmovups(ymm1, mem(rbx, (1*12+4)*4))
		vmovups(ymm2, mem(rbx, (2*12+4)*4))
		vmovups(ymm3, mem(rbx, (3*12+4)*4))
		vmovups(ymm4, mem(rbx, (4*12+4)*4))
		vmovups(ymm5, mem(rbx, (5*12+4)*4))
		vmovups(ymm6, mem(rbx, (6*12+4)*4))
		vmovups(ymm7, mem(rbx, (7*12+4)*4))

		REDUCE_SUM()// sum registers ymm0-ymm7 and place result in ymm0
		vmovups(mem(rcx, 4*4), ymm1)// load previous sum
		vaddps(ymm1, ymm0, ymm0)// add current sum
		vmovups(ymm0, mem(rcx, 4*4))

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
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%r12", "%r13", "%r14", "%r15")
	}
} /* namespace ml */

