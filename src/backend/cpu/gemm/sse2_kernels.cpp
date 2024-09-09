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

#define LOAD_8x4xFP32(ptr, stride, stride_x2, reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7)\
	movups(mem(ptr), reg0)\
	movups(mem(ptr, stride, 1), reg1)\
	add(stride_x2, ptr)\
	movups(mem(ptr), reg2)\
	movups(mem(ptr, stride, 1), reg3)\
	add(stride_x2, ptr)\
	movups(mem(ptr), reg4)\
	movups(mem(ptr, stride, 1), reg5)\
	add(stride_x2, ptr)\
	movups(mem(ptr), reg6)\
	movups(mem(ptr, stride, 1), reg7)\
	add(stride_x2, ptr)
#define STORE_8x4xFP32(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7)\
	movaps(reg0, mem(rbx, 0*4*4))\
	movaps(reg1, mem(rbx, 1*4*4))\
	movaps(reg2, mem(rbx, 2*4*4))\
	movaps(reg3, mem(rbx, 3*4*4))\
	movaps(reg4, mem(rbx, 4*4*4))\
	movaps(reg5, mem(rbx, 5*4*4))\
	movaps(reg6, mem(rbx, 6*4*4))\
	movaps(reg7, mem(rbx, 7*4*4))\


#define LOAD_4x8xFP32(ptr, stride, stride_x2, reg00, reg01, reg10, reg11, reg20, reg21, reg30, reg31)\
	movups(mem(ptr), reg00)\
	movups(mem(ptr, 4*4), reg01)\
	movups(mem(ptr, stride, 1), reg10)\
	movups(mem(ptr, stride, 1, 4*4), reg11)\
	add(stride_x2, ptr)\
	movups(mem(ptr), reg20)\
	movups(mem(ptr, 4*4), reg21)\
	movups(mem(ptr, stride, 1), reg30)\
	movups(mem(ptr, stride, 1, 4*4), reg31)\
	add(stride_x2, ptr)
#define STORE_4x8xFP32(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7)\
	movaps(reg0, mem(rbx, 0*4*4))\
	movaps(reg1, mem(rbx, 1*4*4))\
	movaps(reg2, mem(rbx, 2*4*4))\
	movaps(reg3, mem(rbx, 3*4*4))\
	movaps(reg4, mem(rbx, 4*4*4))\
	movaps(reg5, mem(rbx, 5*4*4))\
	movaps(reg6, mem(rbx, 6*4*4))\
	movaps(reg7, mem(rbx, 7*4*4))\


#define TRANSPOSE_4x4xFP32(reg0, reg1, reg2, reg3, a, b)\
	movaps(reg0, a)\
	movaps(reg2, b)\
	unpcklps(reg1, reg0)\
	unpckhps(reg1, a)\
	unpcklps(reg3, b)\
	movaps(b, reg1)\
	movhlps(reg0, reg1)\
	movlhps(b, reg0)\
	movaps(reg2, b)\
	unpckhps(reg3, b)\
	movaps(a, reg2)\
	movaps(b, reg3)\
	movlhps(reg3, reg2)\
	movhlps(a, reg3)

#define LOAD_ADD_BIAS_4x8xFP32(ptr, stride)\
	movaps(mem(ptr, 0*4*4), xmm0)\
	movaps(mem(ptr, 1*4*4), xmm1)\
	movaps(mem(ptr, stride, 1, 0*4*4), xmm2)\
	movaps(mem(ptr, stride, 1, 1*4*4), xmm3)\
	add(stride, ptr)\
	add(stride, ptr)\
	movaps(mem(ptr, 0*4*4), xmm4)\
	movaps(mem(ptr, 1*4*4), xmm5)\
	movaps(mem(ptr, stride, 1, 0*4*4), xmm6)\
	movaps(mem(ptr, stride, 1, 1*4*4), xmm7)\
	addps(xmm0, xmm8)\
	addps(xmm1, xmm9)\
	addps(xmm2, xmm10)\
	addps(xmm3, xmm11)\
	addps(xmm4, xmm12)\
	addps(xmm5, xmm13)\
	addps(xmm6, xmm14)\
	addps(xmm7, xmm15)

// a = (1 << 22) / float(M_LN2) = 0x4ab8aa3b
// b = 127 * (1 << 23) - 139160 = 0x3f7de068
#define SETUP_EXP_CONSTANTS(a, b)\
	movq(imm(0x4ab8aa3b4ab8aa3b), r14)\
	movq(imm(0x3f7de0683f7de068), r15)\
	movq(r14, a)\
	movq(r15, b)\
	movlhps(a, a)\
	movlhps(b, b)
#define EXP_4x4xFP32(x0, x1, x2, x3, a, b, tmp0, tmp1, tmp2, tmp3)\
	mulps(a, x0)\
	mulps(a, x1)\
	mulps(a, x2)\
	mulps(a, x3)\
	cvtps2dq(x0, x0)\
	cvtps2dq(x1, x1)\
	cvtps2dq(x2, x2)\
	cvtps2dq(x3, x3)\
	movaps(b, tmp0)\
	movaps(b, tmp1)\
	movaps(b, tmp2)\
	movaps(b, tmp3)\
	psubd(x0, tmp0)\
	psubd(x1, tmp1)\
	psubd(x2, tmp2)\
	psubd(x3, tmp3)\
	rcpps(tmp0, tmp0)\
	rcpps(tmp1, tmp1)\
	rcpps(tmp2, tmp2)\
	rcpps(tmp3, tmp3)\
	paddd(b, x0)\
	paddd(b, x1)\
	paddd(b, x2)\
	paddd(b, x3)\
	mulps(tmp0, x0)\
	mulps(tmp1, x1)\
	mulps(tmp2, x2)\
	mulps(tmp3, x3)

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
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 16));
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

		// load address of bias pointer
		movq(var(bias_ptr), rax)
		test(rax, rax)
		je(AFTER_BIAS)
		movaps(mem(rax, 0*4*4), xmm2)// load bias
		movaps(mem(rax, 1*4*4), xmm3)// load bias
		ADD_BIAS_4x8xFP32(xmm2, xmm3)
		label(AFTER_BIAS)

		// load destination pointer and stride
		xorps(xmm0, xmm0)
		ucomiss(xmm1, xmm0)// set ZF if beta == 0.
		je(AFTER_LOAD_C)
		// beta != 0 case
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(var(C_stride), r14)// C stride is r14
		LOAD_ADD_2x8xFP32(xmm1, xmm8, xmm9, xmm10, xmm11)
		LOAD_ADD_2x8xFP32(xmm1, xmm12, xmm13, xmm14, xmm15)
		label(AFTER_LOAD_C)

		// load flag if to use relu
		movq(var(flag_relu), r14)
		test(r14, r14)
		je(AFTER_RELU)
		RELU_4x8xFP32()
		label(AFTER_RELU)

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
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12
			movq(r12, r15)
			sal(imm(1), r15)

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED8)
			LOAD_8x4xFP32(rax, r12, r15, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7)
			STORE_8x4xFP32(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7)
			add(imm(4*8*4), rbx)

			dec(r14)
			jne(UNROLLED8)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movups(mem(rax), xmm0)
			add(r12, rax)
			movaps(xmm0, mem(rbx))
			add(imm(4*1*4), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED1)

			label(EPILOGUE)

			end_asm(:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
					"%rax", "%rbx", "%r12", "%r14", "%r15")
		}
		else
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12
			movq(r12, r15)
			sal(imm(1), r15)

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)
			label(UNROLLED4)
			movq(rax, r13)
			LOAD_4x8xFP32(r13, r12, r15, xmm0, xmm10, xmm1, xmm11, xmm2, xmm12, xmm3, xmm13)
			TRANSPOSE_4x4xFP32(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5)
			TRANSPOSE_4x4xFP32(xmm10, xmm11, xmm12, xmm13, xmm4, xmm5)
			STORE_8x4xFP32(xmm0, xmm1, xmm2, xmm3, xmm10, xmm11, xmm12, xmm13)

			add(imm(4*8), rax)// add stride to src pointer
			add(imm(4*8*4), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED4)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(rax, r13)

			movss(mem(r13), xmm0)
			movss(mem(r13, r12, 1), xmm1)
			add(r15, r13)
			movss(mem(r13), xmm2)
			movss(mem(r13, r12, 1), xmm3)

			movss(xmm0, mem(rbx, 4*0))
			movss(xmm1, mem(rbx, 4*1))
			movss(xmm2, mem(rbx, 4*2))
			movss(xmm3, mem(rbx, 4*3))

			add(imm(4*1), rax)
			add(imm(4*1*4), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED1)

			label(EPILOGUE)

			end_asm(:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
					"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx",
					"%r12", "%r13", "%r14", "%r15")
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
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12
			movq(r12, r15)// copy stride to r15
			sal(imm(1), r15)// multiply stride by 2

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED8)
			LOAD_4x8xFP32(rax, r12, r15, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7)
			LOAD_4x8xFP32(rax, r12, r15, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15)

			STORE_4x8xFP32(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7)
			add(imm(4*8*4), rbx)// add stride to dst pointer
			STORE_4x8xFP32(xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15)
			add(imm(4*8*4), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED8)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movups(mem(rax, 0*4*4), xmm0)
			movups(mem(rax, 1*4*4), xmm1)
			movaps(xmm0, mem(rbx, 0*4*4))
			movaps(xmm1, mem(rbx, 1*4*4))

			add(r12, rax)// add stride to src pointer
			add(imm(4*1*8), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED1)

			label(EPILOGUE)

			end_asm(:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
					"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
					"%r12", "%r14", "%r15")
		}
		else
		{
			uint64_t k_iter = dst.rows() / 4;
			uint64_t k_left = dst.rows() % 4;

			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12
			movq(r12, r15)// copy stride to r15
			sal(imm(1), r15)// multiply stride by 2

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED4)
			movq(rax, r13)
			LOAD_8x4xFP32(r13, r12, r15, xmm0, xmm1, xmm2, xmm3, xmm10, xmm11, xmm12, xmm13)
			TRANSPOSE_4x4xFP32(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5)
			TRANSPOSE_4x4xFP32(xmm10, xmm11, xmm12, xmm13, xmm4, xmm5)
			STORE_4x8xFP32(xmm0, xmm10, xmm1, xmm11, xmm2, xmm12, xmm3, xmm13)

			add(imm(4*4), rax)// add stride to src pointer
			add(imm(4*4*8), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED4)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(rax, r13)// tmp src pointer is in r13

			movss(mem(r13), xmm0)
			movss(mem(r13, r12, 1), xmm1)
			add(r15, r13)// add 2*stride to src pointer
			movss(mem(r13), xmm2)
			movss(mem(r13, r12, 1), xmm3)
			add(r15, r13)// add 2*stride to src pointer
			movss(mem(r13), xmm4)
			movss(mem(r13, r12, 1), xmm5)
			add(r15, r13)// add 2*stride to src pointer
			movss(mem(r13), xmm6)
			movss(mem(r13, r12, 1), xmm7)

			movss(xmm0, mem(rbx, 4*0))
			movss(xmm1, mem(rbx, 4*1))
			movss(xmm2, mem(rbx, 4*2))
			movss(xmm3, mem(rbx, 4*3))
			movss(xmm4, mem(rbx, 4*4))
			movss(xmm5, mem(rbx, 4*5))
			movss(xmm6, mem(rbx, 4*6))
			movss(xmm7, mem(rbx, 4*7))

			add(imm(4*1), rax)// add stride to src pointer
			add(imm(4*8*1), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED1)

			label(EPILOGUE)

			end_asm(:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left),
					[src_stride] "m"(src_stride)
					:// clobbers
					"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
					"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx",
					"%r12", "%r13", "%r14", "%r15")
		}
	}

	// multi-head attention (MHA) kernel
	void mha_qk_sse2_4x8_fp32(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept
	{
		assert(Q.rows() == K.rows());
		assert(Q.stride() == 4);
		assert(K.stride() == 8);
		assert(temp.columns() == Q.columns());
		assert(temp.rows() == K.columns());
		assert(temp.stride() == 4);

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(Q.data(), 16));
		assert(cpu::is_aligned(K.data(), 16));
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 16));
		}
		if (softmax_sum.is_packed())
		{
			assert(cpu::is_aligned(softmax_sum.data(), 16));
		}

		const float *Q_ptr = Q.data<float>();
		const float *K_ptr = K.data<float>();
		float *temp_ptr = temp.data<float>();
		const float *bias_ptr = bias.is_packed() ? bias.data<float>() : nullptr;
		float *softmax_ptr = softmax_sum.is_packed() ? softmax_sum.data<float>() : nullptr;

		uint64_t k_iter = Q.rows() / 4;
		uint64_t k_left = Q.rows() % 4;
		const uint64_t bias_stride = bias.stride() * sizeof(float);

		begin_asm()
		movq(var(Q_ptr), rax) // Q pointer is in rax
		movq(var(K_ptr), rbx)// K pointer is in rbx
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
		movss(mem(rax), xmm0)
		pshufd(imm(0), xmm0, xmm0)
		SCALE_ACCUMULATORS_BY(xmm0)

		movq(var(bias_ptr), rax)// load address of bias pointer
		test(rax, rax)
		je(AFTER_BIAS)
		movq(var(bias_stride), r14)// load address of bias stride into r14
		LOAD_ADD_BIAS_4x8xFP32(rax, r14)
		label(AFTER_BIAS)

		SETUP_EXP_CONSTANTS(xmm0, xmm1)
		EXP_4x4xFP32(xmm8, xmm10, xmm12, xmm14, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5)
		EXP_4x4xFP32(xmm9, xmm11, xmm13, xmm15, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5)

		TRANSPOSE_4x4xFP32(xmm8, xmm10, xmm12, xmm14, xmm0, xmm1)
		TRANSPOSE_4x4xFP32(xmm9, xmm11, xmm13, xmm15, xmm0, xmm1)

		movq(var(temp_ptr), rbx)// temp pointer is in rbx
		STORE_8x4xFP32(xmm8, xmm10, xmm12, xmm14, xmm9, xmm11, xmm13, xmm15)

		movq(var(softmax_ptr), rbx)// softmax sum pointer is in rbx
		movaps(mem(rbx), xmm0)// load previous sum
		// sum all accumulators and place result in the first one (xmm8)
		addps(xmm9, xmm8)
		addps(xmm11, xmm10)
		addps(xmm13, xmm12)
		addps(xmm15, xmm14)
		addps(xmm10, xmm8)
		addps(xmm14, xmm12)
		addps(xmm12, xmm8)
		addps(xmm8, xmm0)// add current sum
		movaps(xmm0, mem(rbx))

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
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%r14", "%r15")
	}
} /* namespace ml */

