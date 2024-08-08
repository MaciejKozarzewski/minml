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
		assert(cpu::is_aligned(bias.data(), 32));

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

		movq(var(bias_ptr), rax)// load address of bias pointer
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax), ymm2)// load bias
		ADD_BIAS_12x8xFP32(ymm2)
		label(AFTER_BIAS)

		vxorps(ymm1, ymm1, ymm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(APPLY_RELU)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)

		LOAD_ADD_3x8xFP32(ymm0, ymm4, ymm5, ymm6)
		LOAD_ADD_3x8xFP32(ymm0, ymm7, ymm8, ymm9)
		LOAD_ADD_3x8xFP32(ymm0, ymm10, ymm11, ymm12)
		LOAD_ADD_3x8xFP32(ymm0, ymm13, ymm14, ymm15)

		label(APPLY_RELU)
		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(STORE_D)
		// apply ReLU case
		RELU_12x8xFP32()

		label(STORE_D)
		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*D_stride)

		STORE_3x8xFP32(ymm4, ymm5, ymm6)
		STORE_3x8xFP32(ymm7, ymm8, ymm9)
		STORE_3x8xFP32(ymm10, ymm11, ymm12)
		STORE_3x8xFP32(ymm13, ymm14, ymm15)

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
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14", "%r15")
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
		assert(cpu::is_aligned(bias.data(), 32));

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

		vxorps(ymm1, ymm1, ymm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(APPLY_RELU)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)

		LOAD_ADD_3x8xFP16(ymm0, ymm4, ymm5, ymm6)
		LOAD_ADD_3x8xFP16(ymm0, ymm7, ymm8, ymm9)
		LOAD_ADD_3x8xFP16(ymm0, ymm10, ymm11, ymm12)
		LOAD_ADD_3x8xFP16(ymm0, ymm13, ymm14, ymm15)

		movq(var(bias_ptr), rax)// load address of bias pointer
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax), ymm2)// load bias
		ADD_BIAS_12x8xFP32(ymm2)
		label(AFTER_BIAS)

		label(APPLY_RELU)
		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(STORE_D)
		// apply ReLU case
		RELU_12x8xFP32()

		label(STORE_D)
		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*D_stride)

		CONVERT_ACCUMULATORS_TO_FP16()

		STORE_3x8xFP16(xmm4, xmm5, xmm6)
		STORE_3x8xFP16(xmm7, xmm8, xmm9)
		STORE_3x8xFP16(xmm10, xmm11, xmm12)
		STORE_3x8xFP16(xmm13, xmm14, xmm15)

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
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14", "%r15")
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
			asm volatile(
					"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
					"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
					"movq %[src_stride], %%r12 \n\t"// src stride is in r12

					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je FINALLOOP%= \n\t"

					"UNROLLED8%=: \n\t"
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"vmovups 0x20(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm2 \n\t"
					"vmovups 0x20(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm4 \n\t"
					"vmovups 0x20(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm6 \n\t"
					"vmovups 0x20(%%rax), %%xmm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm8 \n\t"
					"vmovups 0x20(%%rax), %%xmm9 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm10 \n\t"
					"vmovups 0x20(%%rax), %%xmm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm12 \n\t"
					"vmovups 0x20(%%rax), %%xmm13 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm14 \n\t"
					"vmovups 0x20(%%rax), %%xmm15 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vmovups %%ymm0, ((0*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm1, ((0*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm2, ((1*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm3, ((1*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm4, ((2*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm5, ((2*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm6, ((3*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm7, ((3*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm8, ((4*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm9, ((4*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm10, ((5*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm11, ((5*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm12, ((6*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm13, ((6*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm14, ((7*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm15, ((7*12+8)*4)(%%rbx) \n\t"

					"add $(4*8*12), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"vmovups 0x20(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups %%ymm0, 0x00(%%rbx) \n\t"
					"vmovaps %%xmm1, 0x20(%%rbx) \n\t"
					"add $(4*1*12), %%rbx \n\t"// add stride to dst pointer

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

					"vmovups %%ymm8, (0*12*4)(%%rbx) \n\t"
					"vmovups %%ymm9, (1*12*4)(%%rbx) \n\t"
					"vmovups %%ymm10, (2*12*4)(%%rbx) \n\t"
					"vmovups %%ymm11, (3*12*4)(%%rbx) \n\t"
					"vmovups %%ymm12, (4*12*4)(%%rbx) \n\t"
					"vmovups %%ymm13, (5*12*4)(%%rbx) \n\t"
					"vmovups %%ymm14, (6*12*4)(%%rbx) \n\t"
					"vmovups %%ymm15, (7*12*4)(%%rbx) \n\t"

					// second 4x8 tile
					"vmovups 0x0(%%r13), %%ymm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm3 \n\t"

					// transpose 4x8
					// first shuffle
					"vunpcklps %%ymm1, %%ymm0, %%ymm8 \n\t"
					"vunpckhps %%ymm1, %%ymm0, %%ymm9 \n\t"
					"vunpcklps %%ymm3, %%ymm2, %%ymm10 \n\t"
					"vunpckhps %%ymm3, %%ymm2, %%ymm11 \n\t"

					// second shuffle
					"vunpcklpd %%ymm10, %%ymm8, %%ymm0 \n\t"
					"vunpckhpd %%ymm10, %%ymm8, %%ymm1 \n\t"
					"vunpcklpd %%ymm11, %%ymm9, %%ymm2 \n\t"
					"vunpckhpd %%ymm11, %%ymm9, %%ymm3 \n\t"

					// third shuffle
					"vextractf128 $0x1, %%ymm0, %%xmm4 \n\t"
					"vextractf128 $0x1, %%ymm1, %%xmm5 \n\t"
					"vextractf128 $0x1, %%ymm2, %%xmm6 \n\t"
					"vextractf128 $0x1, %%ymm3, %%xmm7 \n\t"

					"vmovaps %%xmm0, ((0*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm1, ((1*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm2, ((2*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm3, ((3*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm4, ((4*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm5, ((5*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm6, ((6*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm7, ((7*12+8)*4)(%%rbx) \n\t"

					"add $(4*8), %%rax \n\t"// add stride to src pointer
					"add $(4*12*8), %%rbx \n\t"// add stride to dst pointer

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
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm8 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm9 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm10 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm11 \n\t"

					"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
					"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
					"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
					"vmovss %%xmm3, (4*3)(%%rbx) \n\t"
					"vmovss %%xmm4, (4*4)(%%rbx) \n\t"
					"vmovss %%xmm5, (4*5)(%%rbx) \n\t"
					"vmovss %%xmm6, (4*6)(%%rbx) \n\t"
					"vmovss %%xmm7, (4*7)(%%rbx) \n\t"
					"vmovss %%xmm8, (4*8)(%%rbx) \n\t"
					"vmovss %%xmm9, (4*9)(%%rbx) \n\t"
					"vmovss %%xmm10, (4*10)(%%rbx) \n\t"
					"vmovss %%xmm11, (4*11)(%%rbx) \n\t"

					"add $(4*1), %%rax \n\t"// add stride to src pointer
					"add $(4*12*1), %%rbx \n\t"// add stride to dst pointer

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
			asm volatile(
					"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
					"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
					"movq %[src_stride], %%r12 \n\t"// src stride is in r12

					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je FINALLOOP%= \n\t"

					"UNROLLED8%=: \n\t"
					"vmovups 0x00(%%rax), %%xmm0 \n\t"
					"vmovsd 0x10(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm2 \n\t"
					"vmovsd 0x10(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm4 \n\t"
					"vmovsd 0x10(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm6 \n\t"
					"vmovsd 0x10(%%rax), %%xmm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm8 \n\t"
					"vmovsd 0x10(%%rax), %%xmm9 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm10 \n\t"
					"vmovsd 0x10(%%rax), %%xmm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm12 \n\t"
					"vmovsd 0x10(%%rax), %%xmm13 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm14 \n\t"
					"vmovsd 0x10(%%rax), %%xmm15 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%xmm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%xmm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%xmm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%xmm7 \n\t"
					"vcvtph2ps %%xmm8, %%ymm8 \n\t"
					"vcvtph2ps %%xmm9, %%xmm9 \n\t"
					"vcvtph2ps %%xmm10, %%ymm10 \n\t"
					"vcvtph2ps %%xmm11, %%xmm11 \n\t"
					"vcvtph2ps %%xmm12, %%ymm12 \n\t"
					"vcvtph2ps %%xmm13, %%xmm13 \n\t"
					"vcvtph2ps %%xmm14, %%ymm14 \n\t"
					"vcvtph2ps %%xmm15, %%xmm15 \n\t"

					"vmovups %%ymm0, ((0*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm1, ((0*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm2, ((1*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm3, ((1*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm4, ((2*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm5, ((2*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm6, ((3*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm7, ((3*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm8, ((4*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm9, ((4*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm10, ((5*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm11, ((5*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm12, ((6*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm13, ((6*12+8)*4)(%%rbx) \n\t"
					"vmovups %%ymm14, ((7*12+0)*4)(%%rbx) \n\t"
					"vmovaps %%xmm15, ((7*12+8)*4)(%%rbx) \n\t"

					"add $(4*8*12), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%xmm0 \n\t"
					"vmovsd 0x10(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%xmm1 \n\t"
					"vmovups %%ymm0, 0x00(%%rbx) \n\t"
					"vmovaps %%xmm1, 0x20(%%rbx) \n\t"
					"add $(4*1*12), %%rbx \n\t"// add stride to dst pointer

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

					"vmovups %%ymm8, (0*12*4)(%%rbx) \n\t"
					"vmovups %%ymm9, (1*12*4)(%%rbx) \n\t"
					"vmovups %%ymm10, (2*12*4)(%%rbx) \n\t"
					"vmovups %%ymm11, (3*12*4)(%%rbx) \n\t"
					"vmovups %%ymm12, (4*12*4)(%%rbx) \n\t"
					"vmovups %%ymm13, (5*12*4)(%%rbx) \n\t"
					"vmovups %%ymm14, (6*12*4)(%%rbx) \n\t"
					"vmovups %%ymm15, (7*12*4)(%%rbx) \n\t"

					// second 4x8 tile
					"vmovups 0x0(%%r13), %%xmm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm3 \n\t"

					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"

					// transpose 4x8
					// first shuffle
					"vunpcklps %%ymm1, %%ymm0, %%ymm8 \n\t"
					"vunpckhps %%ymm1, %%ymm0, %%ymm9 \n\t"
					"vunpcklps %%ymm3, %%ymm2, %%ymm10 \n\t"
					"vunpckhps %%ymm3, %%ymm2, %%ymm11 \n\t"

					// second shuffle
					"vunpcklpd %%ymm10, %%ymm8, %%ymm0 \n\t"
					"vunpckhpd %%ymm10, %%ymm8, %%ymm1 \n\t"
					"vunpcklpd %%ymm11, %%ymm9, %%ymm2 \n\t"
					"vunpckhpd %%ymm11, %%ymm9, %%ymm3 \n\t"

					// third shuffle
					"vextractf128 $0x1, %%ymm0, %%xmm4 \n\t"
					"vextractf128 $0x1, %%ymm1, %%xmm5 \n\t"
					"vextractf128 $0x1, %%ymm2, %%xmm6 \n\t"
					"vextractf128 $0x1, %%ymm3, %%xmm7 \n\t"

					"vmovaps %%xmm0, ((0*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm1, ((1*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm2, ((2*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm3, ((3*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm4, ((4*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm5, ((5*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm6, ((6*12+8)*4)(%%rbx) \n\t"
					"vmovaps %%xmm7, ((7*12+8)*4)(%%rbx) \n\t"

					"add $(2*8), %%rax \n\t"// add stride to src pointer
					"add $(4*12*8), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm3 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm4 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm5 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm6 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm7 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm8 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm9 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm10 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm11 \n\t"

					"vcvtph2ps %%xmm0, %%xmm0 \n\t"
					"vcvtph2ps %%xmm1, %%xmm1 \n\t"
					"vcvtph2ps %%xmm2, %%xmm2 \n\t"
					"vcvtph2ps %%xmm3, %%xmm3 \n\t"
					"vcvtph2ps %%xmm4, %%xmm4 \n\t"
					"vcvtph2ps %%xmm5, %%xmm5 \n\t"
					"vcvtph2ps %%xmm6, %%xmm6 \n\t"
					"vcvtph2ps %%xmm7, %%xmm7 \n\t"
					"vcvtph2ps %%xmm8, %%xmm8 \n\t"
					"vcvtph2ps %%xmm9, %%xmm9 \n\t"
					"vcvtph2ps %%xmm10, %%xmm10 \n\t"
					"vcvtph2ps %%xmm11, %%xmm11 \n\t"

					"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
					"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
					"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
					"vmovss %%xmm3, (4*3)(%%rbx) \n\t"
					"vmovss %%xmm4, (4*4)(%%rbx) \n\t"
					"vmovss %%xmm5, (4*5)(%%rbx) \n\t"
					"vmovss %%xmm6, (4*6)(%%rbx) \n\t"
					"vmovss %%xmm7, (4*7)(%%rbx) \n\t"
					"vmovss %%xmm8, (4*8)(%%rbx) \n\t"
					"vmovss %%xmm9, (4*9)(%%rbx) \n\t"
					"vmovss %%xmm10, (4*10)(%%rbx) \n\t"
					"vmovss %%xmm11, (4*11)(%%rbx) \n\t"

					"add $(2*1), %%rax \n\t"// add stride to src pointer
					"add $(4*12*1), %%rbx \n\t"// add stride to dst pointer

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

} /* namespace ml */

