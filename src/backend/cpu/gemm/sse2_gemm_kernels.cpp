/*
 * sse2_gemm_kernels.cpp
 *
 *  Created on: May 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include "Fragment.hpp"
#include "Matrix.hpp"
#include "gemm_kernels.hpp"
#include "../utils.hpp"

#include <cinttypes>
#include <cassert>

#include "../assembly_macros.hpp"

namespace ml
{
#define ZERO_ACCUMULATORS()\
	xorps(xmm8, xmm8)\
	xorps(xmm9, xmm9)\
	xorps(xmm10, xmm10)\
	xorps(xmm11, xmm11)\
	xorps(xmm12, xmm12)\
	xorps(xmm13, xmm13)\
	xorps(xmm14, xmm14)\
	xorps(xmm15, xmm15)\

#define SUB_KERNEL_4xFP32_8xFP32(n) \
	movaps(mem(rax, n*4*4), xmm0)\
	movaps(mem(rbx, (2*n+0)*4*4), xmm6)\
	movaps(mem(rbx, (2*n+1)*4*4), xmm7)\
	pshufd(imm(0x00), xmm0, xmm2)\
	pshufd(imm(0x55), xmm0, xmm4)\
	movaps(xmm2, xmm3)\
	movaps(xmm4, xmm5)\
	mulps(xmm6, xmm2)\
	mulps(xmm7, xmm3)\
	mulps(xmm6, xmm4)\
	mulps(xmm7, xmm5)\
	addps(xmm2, xmm8)\
	addps(xmm3, xmm9)\
	addps(xmm4, xmm10)\
	addps(xmm5, xmm11)\
	pshufd(imm(0xAA), xmm0, xmm2)\
	pshufd(imm(0xFF), xmm0, xmm4)\
	movaps(xmm2, xmm3)\
	movaps(xmm4, xmm5)\
	mulps(xmm6, xmm2)\
	mulps(xmm7, xmm3)\
	mulps(xmm6, xmm4)\
	mulps(xmm7, xmm5)\
	addps(xmm2, xmm12)\
	addps(xmm3, xmm13)\
	addps(xmm4, xmm14)\
	addps(xmm5, xmm15)

#define SCALE_ACCUMULATORS_BY(reg)\
	mulps(reg, xmm8)\
	mulps(reg, xmm9)\
	mulps(reg, xmm10)\
	mulps(reg, xmm11)\
	mulps(reg, xmm12)\
	mulps(reg, xmm13)\
	mulps(reg, xmm14)\
	mulps(reg, xmm15)

#define LOAD_ADD_2x8xFP32(beta, reg00, reg01, reg10, reg11)\
	movups(mem(rcx, 0*4*4), xmm4)\
	movups(mem(rcx, 1*4*4), xmm5)\
	add(r14, rcx)\
	movups(mem(rcx, 0*4*4), xmm6)\
	movups(mem(rcx, 1*4*4), xmm7)\
	add(r14, rcx)\
	mulps(beta, xmm4)\
	mulps(beta, xmm5)\
	mulps(beta, xmm6)\
	mulps(beta, xmm7)\
	addps(xmm4, reg00)\
	addps(xmm5, reg01)\
	addps(xmm6, reg10)\
	addps(xmm7, reg11)

#define ADD_BIAS_4x8xFP32(b1, b2)\
	addps(b1, xmm8)\
	addps(b2, xmm9)\
	addps(b1, xmm10)\
	addps(b2, xmm11)\
	addps(b1, xmm12)\
	addps(b2, xmm13)\
	addps(b1, xmm14)\
	addps(b2, xmm15)

#define STORE_2x8xFP32(reg00, reg01, reg10, reg11)\
	movups(reg00, mem(rcx, 0*4*4))\
	movups(reg01, mem(rcx, 1*4*4))\
	add(r14, rcx)\
	movups(reg10, mem(rcx, 0*4*4))\
	movups(reg11, mem(rcx, 1*4*4))\
	add(r14, rcx)\

#define RELU_4x8xFP32()\
	xorps(xmm0, xmm0)\
	maxps(xmm0, xmm8)\
	maxps(xmm0, xmm9)\
	maxps(xmm0, xmm10)\
	maxps(xmm0, xmm11)\
	maxps(xmm0, xmm12)\
	maxps(xmm0, xmm13)\
	maxps(xmm0, xmm14)\
	maxps(xmm0, xmm15)

	void gemm_sse2_4x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.stride() == 4);
		assert(B.stride() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(A.data(), 16));
		assert(cpu::is_aligned(B.data(), 16));
		assert(beta_ptr != nullptr);

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
		movq(var(A_ptr), rax) // A pointer is in rax
		movq(var(B_ptr), rbx)// B pointer is in rbx
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_4xFP32_8xFP32(0)
		SUB_KERNEL_4xFP32_8xFP32(1)
		SUB_KERNEL_4xFP32_8xFP32(2)
		SUB_KERNEL_4xFP32_8xFP32(3)

		add(imm(4*4*4), rax)// 4 rows x 4 floats x 4 bytes
		add(imm(4*8*4), rbx)// 4 rows x 8 floats x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_4xFP32_8xFP32(0)

		add(imm(1*4*4), rax)// 1 row x 4 floats x 4 bytes
		add(imm(1*8*4), rbx)// 1 row x 8 floats x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(beta_ptr), rbx)// load address of beta
		movss(mem(rax), xmm0)
		movss(mem(rbx), xmm1)
		pshufd(imm(0), xmm0, xmm0)
		pshufd(imm(0), xmm1, xmm1)

		SCALE_ACCUMULATORS_BY(xmm0)

		movq(var(bias_ptr), rax)// load address of bias pointer
		test(rax, rax)
		je(AFTER_BIAS)
		movaps(mem(rax, 0*4*4), xmm2)// load bias
		movaps(mem(rax, 1*4*4), xmm3)// load bias
		ADD_BIAS_4x8xFP32(xmm2, xmm3)

		label(AFTER_BIAS)
		// load destination pointer and stride
		xorps(xmm0, xmm0)
		ucomiss(xmm1, xmm0)// set ZF if beta == 0.
		je(APPLY_RELU)
		// beta != 0 case
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(var(C_stride), r14)// C stride is r14

		LOAD_ADD_2x8xFP32(xmm1, xmm8, xmm9, xmm10, xmm11)
		LOAD_ADD_2x8xFP32(xmm1, xmm12, xmm13, xmm14, xmm15)

		label(APPLY_RELU)
		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(STORE_D)
		RELU_4x8xFP32()

		label(STORE_D)
		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx

		STORE_2x8xFP32(xmm8, xmm9, xmm10, xmm11)
		STORE_2x8xFP32(xmm12, xmm13, xmm14, xmm15)

		end_asm(:// outputs
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
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%r14")

	}

	void pack_sse2_4xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 4);
		assert(ml::cpu::is_aligned(dst.data(), 16));

		const uint64_t src_stride = src.stride() * sizeof(float);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();
		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = dst.rows() % 8;

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
					"movups 0x00(%%rax), %%xmm0 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm2 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm4 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm6 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"movaps %%xmm0, 0x00(%%rbx) \n\t"
					"movaps %%xmm1, 0x10(%%rbx) \n\t"
					"movaps %%xmm2, 0x20(%%rbx) \n\t"
					"movaps %%xmm3, 0x30(%%rbx) \n\t"
					"movaps %%xmm4, 0x40(%%rbx) \n\t"
					"movaps %%xmm5, 0x50(%%rbx) \n\t"
					"movaps %%xmm6, 0x60(%%rbx) \n\t"
					"movaps %%xmm7, 0x70(%%rbx) \n\t"

					"add $(4*8*4), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"movups 0x00(%%rax), %%xmm0 \n\t"
					"movaps %%xmm0, 0x00(%%rbx) \n\t"

					"add %%r12, %%rax \n\t"// add stride to src pointer
					"add $(4*1*4), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED1%= \n\t"

					"EPILOGUE%=: \n\t"

					:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
					"%rax", "%rbx", "%r12", "%r14");
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

					"UNROLLED4%=: \n\t"
					// first 8x8 tile
					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

					"movups 0x00(%%r13), %%xmm0 \n\t"
					"movups 0x10(%%r13), %%xmm10 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x00(%%r13), %%xmm1 \n\t"
					"movups 0x10(%%r13), %%xmm11 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x00(%%r13), %%xmm2 \n\t"
					"movups 0x10(%%r13), %%xmm12 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x00(%%r13), %%xmm3 \n\t"
					"movups 0x10(%%r13), %%xmm13 \n\t"

					"movaps %%xmm0, %%xmm4 \n\t"
					"movaps %%xmm2, %%xmm5 \n\t"
					"movaps %%xmm10, %%xmm14 \n\t"
					"movaps %%xmm12, %%xmm15 \n\t"

					// transpose 4x8
					// first shuffle
					"unpcklps %%xmm1, %%xmm4 \n\t"
					"unpcklps %%xmm3, %%xmm5 \n\t"
					"unpckhps %%xmm1, %%xmm0 \n\t"
					"unpckhps %%xmm3, %%xmm2 \n\t"
					"unpcklps %%xmm11, %%xmm14 \n\t"
					"unpcklps %%xmm13, %%xmm15 \n\t"
					"unpckhps %%xmm11, %%xmm10 \n\t"
					"unpckhps %%xmm13, %%xmm12 \n\t"

					"movaps %%xmm4, %%xmm1 \n\t"
					"movaps %%xmm0, %%xmm3 \n\t"
					"movaps %%xmm14, %%xmm11 \n\t"
					"movaps %%xmm10, %%xmm13 \n\t"

					// second shuffle
					"movlhps %%xmm5, %%xmm4 \n\t"
					"movlhps %%xmm2, %%xmm3 \n\t"
					"movhlps %%xmm1, %%xmm5 \n\t"
					"movhlps %%xmm0, %%xmm2 \n\t"
					"movlhps %%xmm15, %%xmm14 \n\t"
					"movlhps %%xmm12, %%xmm13 \n\t"
					"movhlps %%xmm11, %%xmm15 \n\t"
					"movhlps %%xmm10, %%xmm12 \n\t"

					"movaps %%xmm4, (4*4*0)(%%rbx) \n\t"
					"movaps %%xmm5, (4*4*1)(%%rbx) \n\t"
					"movaps %%xmm3, (4*4*2)(%%rbx) \n\t"
					"movaps %%xmm2, (4*4*3)(%%rbx) \n\t"
					"movaps %%xmm14, (4*4*4)(%%rbx) \n\t"
					"movaps %%xmm15, (4*4*5)(%%rbx) \n\t"
					"movaps %%xmm13, (4*4*6)(%%rbx) \n\t"
					"movaps %%xmm12, (4*4*7)(%%rbx) \n\t"

					"add $(4*8), %%rax \n\t"// add stride to src pointer
					"add $(4*8*4), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED4%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

					"movss 0x0(%%r13), %%xmm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm3 \n\t"

					"movss %%xmm0, (4*0)(%%rbx) \n\t"
					"movss %%xmm1, (4*1)(%%rbx) \n\t"
					"movss %%xmm2, (4*2)(%%rbx) \n\t"
					"movss %%xmm3, (4*3)(%%rbx) \n\t"

					"add $(4*1), %%rax \n\t"// add stride to src pointer
					"add $(4*4*1), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED1%= \n\t"

					"EPILOGUE%=: \n\t"

					:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
					"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx",
					"%r12", "%r13", "%r14");
		}
	}
	void pack_sse2_8xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 8);
		assert(ml::cpu::is_aligned(dst.data(), 16));

		const uint64_t src_stride = src.stride() * sizeof(float);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			uint64_t k_iter = dst.rows() / 8;
			uint64_t k_left = dst.rows() % 8;
			asm volatile(
					"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
					"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
					"movq %[src_stride], %%r12 \n\t"// src stride is in r12

					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je FINALLOOP%= \n\t"

					"UNROLLED8%=: \n\t"
					"movups 0x00(%%rax), %%xmm0 \n\t"
					"movups 0x10(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm2 \n\t"
					"movups 0x10(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm4 \n\t"
					"movups 0x10(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm6 \n\t"
					"movups 0x10(%%rax), %%xmm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm8 \n\t"
					"movups 0x10(%%rax), %%xmm9 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm10 \n\t"
					"movups 0x10(%%rax), %%xmm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm12 \n\t"
					"movups 0x10(%%rax), %%xmm13 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups 0x00(%%rax), %%xmm14 \n\t"
					"movups 0x10(%%rax), %%xmm15 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"movaps %%xmm0, 0x00(%%rbx) \n\t"
					"movaps %%xmm1, 0x10(%%rbx) \n\t"
					"movaps %%xmm2, 0x20(%%rbx) \n\t"
					"movaps %%xmm3, 0x30(%%rbx) \n\t"
					"movaps %%xmm4, 0x40(%%rbx) \n\t"
					"movaps %%xmm5, 0x50(%%rbx) \n\t"
					"movaps %%xmm6, 0x60(%%rbx) \n\t"
					"movaps %%xmm7, 0x70(%%rbx) \n\t"
					"movaps %%xmm8, 0x80(%%rbx) \n\t"
					"movaps %%xmm9, 0x90(%%rbx) \n\t"
					"movaps %%xmm10, 0xA0(%%rbx) \n\t"
					"movaps %%xmm11, 0xB0(%%rbx) \n\t"
					"movaps %%xmm12, 0xC0(%%rbx) \n\t"
					"movaps %%xmm13, 0xD0(%%rbx) \n\t"
					"movaps %%xmm14, 0xE0(%%rbx) \n\t"
					"movaps %%xmm15, 0xF0(%%rbx) \n\t"

					"add $(4*8*8), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"movups 0x00(%%rax), %%xmm0 \n\t"
					"movups 0x10(%%rax), %%xmm1 \n\t"
					"movaps %%xmm0, 0x00(%%rbx) \n\t"
					"movaps %%xmm1, 0x10(%%rbx) \n\t"

					"add %%r12, %%rax \n\t"// add stride to src pointer
					"add $(4*1*8), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED1%= \n\t"

					"EPILOGUE%=: \n\t"

					:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
					"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
					"%r12", "%r14");
		}
		else
		{
			uint64_t k_iter = dst.rows() / 4;
			uint64_t k_left = dst.rows() % 4;
			asm volatile(
					"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
					"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
					"movq %[src_stride], %%r12 \n\t"// src stride is in r12

					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je FINALLOOP%= \n\t"

					"UNROLLED4%=: \n\t"
					// first 8x8 tile
					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

					"movups 0x0(%%r13), %%xmm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x0(%%r13), %%xmm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x0(%%r13), %%xmm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x0(%%r13), %%xmm3 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x0(%%r13), %%xmm10 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x0(%%r13), %%xmm11 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x0(%%r13), %%xmm12 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movups 0x0(%%r13), %%xmm13 \n\t"

					"movaps %%xmm0, %%xmm4 \n\t"
					"movaps %%xmm2, %%xmm5 \n\t"
					"movaps %%xmm10, %%xmm14 \n\t"
					"movaps %%xmm12, %%xmm15 \n\t"

					// transpose 8x4
					// first shuffle
					"unpcklps %%xmm1, %%xmm4 \n\t"
					"unpcklps %%xmm3, %%xmm5 \n\t"
					"unpckhps %%xmm1, %%xmm0 \n\t"
					"unpckhps %%xmm3, %%xmm2 \n\t"
					"unpcklps %%xmm11, %%xmm14 \n\t"
					"unpcklps %%xmm13, %%xmm15 \n\t"
					"unpckhps %%xmm11, %%xmm10 \n\t"
					"unpckhps %%xmm13, %%xmm12 \n\t"

					"movaps %%xmm4, %%xmm1 \n\t"
					"movaps %%xmm0, %%xmm3 \n\t"
					"movaps %%xmm14, %%xmm11 \n\t"
					"movaps %%xmm10, %%xmm13 \n\t"

					// second shuffle
					"movlhps %%xmm5, %%xmm4 \n\t"
					"movlhps %%xmm2, %%xmm3 \n\t"
					"movhlps %%xmm1, %%xmm5 \n\t"
					"movhlps %%xmm0, %%xmm2 \n\t"
					"movlhps %%xmm15, %%xmm14 \n\t"
					"movlhps %%xmm12, %%xmm13 \n\t"
					"movhlps %%xmm11, %%xmm15 \n\t"
					"movhlps %%xmm10, %%xmm12 \n\t"

					"movaps %%xmm4,  (4*(8*0+0))(%%rbx) \n\t"
					"movaps %%xmm14, (4*(8*0+4))(%%rbx) \n\t"
					"movaps %%xmm5,  (4*(8*1+0))(%%rbx) \n\t"
					"movaps %%xmm15, (4*(8*1+4))(%%rbx) \n\t"
					"movaps %%xmm3,  (4*(8*2+0))(%%rbx) \n\t"
					"movaps %%xmm13, (4*(8*2+4))(%%rbx) \n\t"
					"movaps %%xmm2,  (4*(8*3+0))(%%rbx) \n\t"
					"movaps %%xmm12, (4*(8*3+4))(%%rbx) \n\t"

					"add $(4*4), %%rax \n\t"// add stride to src pointer
					"add $(4*4*8), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED4%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

					"movss 0x0(%%r13), %%xmm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm3 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm4 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm5 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm6 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movss 0x0(%%r13), %%xmm7 \n\t"

					"movss %%xmm0, (4*0)(%%rbx) \n\t"
					"movss %%xmm1, (4*1)(%%rbx) \n\t"
					"movss %%xmm2, (4*2)(%%rbx) \n\t"
					"movss %%xmm3, (4*3)(%%rbx) \n\t"
					"movss %%xmm4, (4*4)(%%rbx) \n\t"
					"movss %%xmm5, (4*5)(%%rbx) \n\t"
					"movss %%xmm6, (4*6)(%%rbx) \n\t"
					"movss %%xmm7, (4*7)(%%rbx) \n\t"

					"add $(4*1), %%rax \n\t"// add stride to src pointer
					"add $(4*8*1), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED1%= \n\t"

					"EPILOGUE%=: \n\t"

					:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
					"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx",
					"%r12", "%r13", "%r14");
		}
	}

} /* namespace ml */

