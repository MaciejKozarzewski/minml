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
#include "../vectors/vectors.hpp"

#include <x86intrin.h>
#include <cinttypes>
#include <cassert>

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
	movaps(mem(rax), xmm0)\
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
	mulps(reg, xmm8) \
	mulps(reg, xmm9) \
	mulps(reg, xmm10) \
	mulps(reg, xmm11) \
	mulps(reg, xmm12) \
	mulps(reg, xmm13) \
	mulps(reg, xmm14) \
	mulps(reg, xmm15)

#define LOAD_ADD_2x8xFP32(beta, reg00, reg01, reg10, reg11, stride)\
	movups(mem(rcx, 0*stride), xmm4)\
	movups(mem(rcx, 1*stride), xmm5)\
	add(r14, rcx)\
	movups(mem(rcx, 2*stride), xmm6)\
	movups(mem(rcx, 3*stride), xmm7)\
	add(r14, rcx)\
	mulps(beta, xmm4)\
	mulps(beta, xmm5)\
	mulps(beta, xmm6)\
	mulps(beta, xmm7)\
	addps(xmm4, reg00)\
	addps(xmm5, reg01)\
	addps(xmm6, reg10)\
	addps(xmm7, reg11)


#define STORE_2x8xFP32(reg00, reg01, reg10, reg11, stride)\
	movups(mem(rcx, 0*stride), reg00)\
	movups(mem(rcx, 1*stride), reg01)\
	add(r14, rcx)\
	movups(mem(rcx, 2*stride), reg10)\
	movups(mem(rcx, 3*stride), reg11)\
	add(r14, rcx)\


	void gemm_sse2_4x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 4);
		assert(B.columns() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(A.data(), register_size<XMM>()));
		assert(cpu::is_aligned(B.data(), register_size<XMM>()));
		assert(beta_ptr != nullptr);

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float *C_ptr = C.data<float>();
		float *D_ptr = D.data<float>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float);
		const uint64_t D_stride = D.stride() * sizeof(float);

		asm volatile(
				"movq %[A_ptr], %%rax \n\t" // A pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// B pointer is in rbx

				// Set accumulators to zero.
				"xorps %%xmm8, %%xmm8 \n\t"
				"xorps %%xmm9, %%xmm9 \n\t"
				"xorps %%xmm10, %%xmm10 \n\t"
				"xorps %%xmm11, %%xmm11 \n\t"
				"xorps %%xmm12, %%xmm12 \n\t"
				"xorps %%xmm13, %%xmm13 \n\t"
				"xorps %%xmm14, %%xmm14 \n\t"
				"xorps %%xmm15, %%xmm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"movaps 0x00(%%rax), %%xmm0 \n\t"// a0 a1 a2 a3
				"movaps 0x00(%%rbx), %%xmm6 \n\t"// b0 b1 b2 b3
				"movaps 0x10(%%rbx), %%xmm7 \n\t"// b4 b5 b6 b7

				"pshufd $0x00, %%xmm0, %%xmm2 \n\t"// a0 a0 a0 a0
				"pshufd $0x55, %%xmm0, %%xmm4 \n\t"// a1 a1 a1 a1
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm8 \n\t"
				"addps %%xmm3, %%xmm9 \n\t"
				"addps %%xmm4, %%xmm10 \n\t"
				"addps %%xmm5, %%xmm11 \n\t"

				"pshufd $0xAA, %%xmm0, %%xmm2 \n\t"// a2 a2 a2 a2
				"pshufd $0xFF, %%xmm0, %%xmm4 \n\t"// a3 a3 a3 a3
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm12 \n\t"
				"addps %%xmm3, %%xmm13 \n\t"
				"addps %%xmm4, %%xmm14 \n\t"
				"addps %%xmm5, %%xmm15 \n\t"

				// iteration 1
				"movaps 0x10(%%rax), %%xmm0 \n\t"// a0 a1 a2 a3
				"movaps 0x20(%%rbx), %%xmm6 \n\t"// b0 b1 b2 b3
				"movaps 0x30(%%rbx), %%xmm7 \n\t"// b4 b5 b6 b7

				"pshufd $0x00, %%xmm0, %%xmm2 \n\t"// a0 a0 a0 a0
				"pshufd $0x55, %%xmm0, %%xmm4 \n\t"// a1 a1 a1 a1
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm8 \n\t"
				"addps %%xmm3, %%xmm9 \n\t"
				"addps %%xmm4, %%xmm10 \n\t"
				"addps %%xmm5, %%xmm11 \n\t"

				"pshufd $0xAA, %%xmm0, %%xmm2 \n\t"// a2 a2 a2 a2
				"pshufd $0xFF, %%xmm0, %%xmm4 \n\t"// a3 a3 a3 a3
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm12 \n\t"
				"addps %%xmm3, %%xmm13 \n\t"
				"addps %%xmm4, %%xmm14 \n\t"
				"addps %%xmm5, %%xmm15 \n\t"

				// iteration 2
				"movaps 0x20(%%rax), %%xmm0 \n\t"// a0 a1 a2 a3
				"movaps 0x40(%%rbx), %%xmm6 \n\t"// b0 b1 b2 b3
				"movaps 0x50(%%rbx), %%xmm7 \n\t"// b4 b5 b6 b7

				"pshufd $0x00, %%xmm0, %%xmm2 \n\t"// a0 a0 a0 a0
				"pshufd $0x55, %%xmm0, %%xmm4 \n\t"// a1 a1 a1 a1
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm8 \n\t"
				"addps %%xmm3, %%xmm9 \n\t"
				"addps %%xmm4, %%xmm10 \n\t"
				"addps %%xmm5, %%xmm11 \n\t"

				"pshufd $0xAA, %%xmm0, %%xmm2 \n\t"// a2 a2 a2 a2
				"pshufd $0xFF, %%xmm0, %%xmm4 \n\t"// a3 a3 a3 a3
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm12 \n\t"
				"addps %%xmm3, %%xmm13 \n\t"
				"addps %%xmm4, %%xmm14 \n\t"
				"addps %%xmm5, %%xmm15 \n\t"

				// iteration 3
				"movaps 0x30(%%rax), %%xmm0 \n\t"// a0 a1 a2 a3
				"movaps 0x60(%%rbx), %%xmm6 \n\t"// b0 b1 b2 b3
				"movaps 0x70(%%rbx), %%xmm7 \n\t"// b4 b5 b6 b7

				"pshufd $0x00, %%xmm0, %%xmm2 \n\t"// a0 a0 a0 a0
				"pshufd $0x55, %%xmm0, %%xmm4 \n\t"// a1 a1 a1 a1
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm8 \n\t"
				"addps %%xmm3, %%xmm9 \n\t"
				"addps %%xmm4, %%xmm10 \n\t"
				"addps %%xmm5, %%xmm11 \n\t"

				"pshufd $0xAA, %%xmm0, %%xmm2 \n\t"// a2 a2 a2 a2
				"pshufd $0xFF, %%xmm0, %%xmm4 \n\t"// a3 a3 a3 a3
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm12 \n\t"
				"addps %%xmm3, %%xmm13 \n\t"
				"addps %%xmm4, %%xmm14 \n\t"
				"addps %%xmm5, %%xmm15 \n\t"

				"add $0x40, %%rax \n\t"
				"add $0x80, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"movaps 0x00(%%rax), %%xmm0 \n\t"// a0 a1 a2 a3
				"movaps 0x00(%%rbx), %%xmm6 \n\t"// b0 b1 b2 b3
				"movaps 0x10(%%rbx), %%xmm7 \n\t"// b4 b5 b6 b7

				"pshufd $0x00, %%xmm0, %%xmm2 \n\t"// a0 a0 a0 a0
				"pshufd $0x55, %%xmm0, %%xmm4 \n\t"// a1 a1 a1 a1
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm8 \n\t"
				"addps %%xmm3, %%xmm9 \n\t"
				"addps %%xmm4, %%xmm10 \n\t"
				"addps %%xmm5, %%xmm11 \n\t"

				"pshufd $0xAA, %%xmm0, %%xmm2 \n\t"// a2 a2 a2 a2
				"pshufd $0xFF, %%xmm0, %%xmm4 \n\t"// a3 a3 a3 a3
				"movaps %%xmm2, %%xmm3 \n\t"
				"movaps %%xmm4, %%xmm5 \n\t"

				"mulps %%xmm6, %%xmm2 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"addps %%xmm2, %%xmm12 \n\t"
				"addps %%xmm3, %%xmm13 \n\t"
				"addps %%xmm4, %%xmm14 \n\t"
				"addps %%xmm5, %%xmm15 \n\t"

				"add $0x10, %%rax \n\t"
				"add $0x20, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"movss 0x00(%%rax), %%xmm0 \n\t"
				"movss 0x00(%%rbx), %%xmm1 \n\t"
				"pshufd $0x00, %%xmm0, %%xmm0 \n\t"
				"pshufd $0x00, %%xmm1, %%xmm1 \n\t"

				// scale by alpha
				"mulps %%xmm0, %%xmm8 \n\t"
				"mulps %%xmm0, %%xmm9 \n\t"
				"mulps %%xmm0, %%xmm10 \n\t"
				"mulps %%xmm0, %%xmm11 \n\t"
				"mulps %%xmm0, %%xmm12 \n\t"
				"mulps %%xmm0, %%xmm13 \n\t"
				"mulps %%xmm0, %%xmm14 \n\t"
				"mulps %%xmm0, %%xmm15 \n\t"

				// load destination pointer and stride

				"xorps %%xmm0, %%xmm0 \n\t"
				"ucomiss %%xmm1, %%xmm0 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// beta != 0 case
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx
				"movq %[C_stride], %%r14 \n\t"// C stride is r14

				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"movups 0x10(%%rcx), %%xmm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"movups 0x10(%%rcx), %%xmm7 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"mulps %%xmm1, %%xmm4 \n\t"
				"mulps %%xmm1, %%xmm5 \n\t"
				"mulps %%xmm1, %%xmm6 \n\t"
				"mulps %%xmm1, %%xmm7 \n\t"

				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"movups 0x10(%%rcx), %%xmm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"movups 0x10(%%rcx), %%xmm7 \n\t"

				"mulps %%xmm1, %%xmm4 \n\t"
				"mulps %%xmm1, %%xmm5 \n\t"
				"mulps %%xmm1, %%xmm6 \n\t"
				"mulps %%xmm1, %%xmm7 \n\t"

				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				"BETAZERO%=: \n\t"
				// beta == 0 case
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx
				"movq %[D_stride], %%r14 \n\t"// D stride is r14

				"movups %%xmm8, 0x00(%%rcx) \n\t"
				"movups %%xmm9, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm10, 0x00(%%rcx) \n\t"
				"movups %%xmm11, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm12, 0x00(%%rcx) \n\t"
				"movups %%xmm13, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm14, 0x00(%%rcx) \n\t"
				"movups %%xmm15, 0x10(%%rcx) \n\t"

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
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%r14");
	}
	void gemm_sse2_8x4_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 8);
		assert(B.columns() == 4);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(A.data(), register_size<XMM>()));
		assert(cpu::is_aligned(B.data(), register_size<XMM>()));
		assert(beta_ptr != nullptr);

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float *C_ptr = C.data<float>();
		float *D_ptr = D.data<float>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float);
		const uint64_t D_stride = D.stride() * sizeof(float);

		asm volatile(
				"movq %[A_ptr], %%rax \n\t" // A pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// B pointer is in rbx

				// Set accumulators to zero.
				"xorps %%xmm8, %%xmm8 \n\t"
				"xorps %%xmm9, %%xmm9 \n\t"
				"xorps %%xmm10, %%xmm10 \n\t"
				"xorps %%xmm11, %%xmm11 \n\t"
				"xorps %%xmm12, %%xmm12 \n\t"
				"xorps %%xmm13, %%xmm13 \n\t"
				"xorps %%xmm14, %%xmm14 \n\t"
				"xorps %%xmm15, %%xmm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"movaps 0x00(%%rax), %%xmm0 \n\t"
				"movaps 0x10(%%rax), %%xmm1 \n\t"
				"movaps 0x00(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				// iteration 1
				"movaps 0x20(%%rax), %%xmm0 \n\t"
				"movaps 0x30(%%rax), %%xmm1 \n\t"
				"movaps 0x10(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				// iteration 2
				"movaps 0x40(%%rax), %%xmm0 \n\t"
				"movaps 0x50(%%rax), %%xmm1 \n\t"
				"movaps 0x20(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				// iteration 3
				"movaps 0x60(%%rax), %%xmm0 \n\t"
				"movaps 0x70(%%rax), %%xmm1 \n\t"
				"movaps 0x30(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				"add $0x80, %%rax \n\t"
				"add $0x40, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"movaps 0x00(%%rax), %%xmm0 \n\t"
				"movaps 0x10(%%rax), %%xmm1 \n\t"
				"movaps 0x00(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				"add $0x20, %%rax \n\t"
				"add $0x10, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"movss 0x00(%%rax), %%xmm0 \n\t"
				"movss 0x00(%%rbx), %%xmm1 \n\t"
				"pshufd $0x00, %%xmm0, %%xmm0 \n\t"
				"pshufd $0x00, %%xmm1, %%xmm1 \n\t"

				// scale by alpha
				"mulps %%xmm0, %%xmm8 \n\t"
				"mulps %%xmm0, %%xmm9 \n\t"
				"mulps %%xmm0, %%xmm10 \n\t"
				"mulps %%xmm0, %%xmm11 \n\t"
				"mulps %%xmm0, %%xmm12 \n\t"
				"mulps %%xmm0, %%xmm13 \n\t"
				"mulps %%xmm0, %%xmm14 \n\t"
				"mulps %%xmm0, %%xmm15 \n\t"

				// load destination pointer and stride

				"xorps %%xmm0, %%xmm0 \n\t"
				"ucomiss %%xmm1, %%xmm0 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// beta != 0 case
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx
				"movq %[C_stride], %%r14 \n\t"// C stride is r14

				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm7 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"mulps %%xmm1, %%xmm4 \n\t"
				"mulps %%xmm1, %%xmm5 \n\t"
				"mulps %%xmm1, %%xmm6 \n\t"
				"mulps %%xmm1, %%xmm7 \n\t"

				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm7 \n\t"

				"mulps %%xmm1, %%xmm4 \n\t"
				"mulps %%xmm1, %%xmm5 \n\t"
				"mulps %%xmm1, %%xmm6 \n\t"
				"mulps %%xmm1, %%xmm7 \n\t"

				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				"BETAZERO%=: \n\t"
				// beta == 0 case
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx
				"movq %[D_stride], %%r14 \n\t"// D stride is r14

				"movups %%xmm8, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm9, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm10, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm11, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm12, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm13, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm14, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm15, 0x00(%%rcx) \n\t"

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
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%r14");
	}
	void gemm_sse2_4x4_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 4);
		assert(B.columns() == 4);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(A.data(), register_size<XMM>()));
		assert(cpu::is_aligned(B.data(), register_size<XMM>()));
		assert(beta_ptr != nullptr);

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float *C_ptr = C.data<float>();
		float *D_ptr = D.data<float>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float);
		const uint64_t D_stride = D.stride() * sizeof(float);

		asm volatile(
				"movq %[A_ptr], %%rax \n\t" // A pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// B pointer is in rbx

				// Set accumulators to zero.
				"xorps %%xmm12, %%xmm12 \n\t"
				"xorps %%xmm13, %%xmm13 \n\t"
				"xorps %%xmm14, %%xmm14 \n\t"
				"xorps %%xmm15, %%xmm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"movaps 0x00(%%rax), %%xmm0 \n\t"
				"movaps 0x00(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				// iteration 1
				"movaps 0x10(%%rax), %%xmm0 \n\t"
				"movaps 0x10(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				// iteration 2
				"movaps 0x20(%%rax), %%xmm0 \n\t"
				"movaps 0x20(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				// iteration 3
				"movaps 0x30(%%rax), %%xmm0 \n\t"
				"movaps 0x30(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				"add $0x40, %%rax \n\t"
				"add $0x40, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"movaps 0x00(%%rax), %%xmm0 \n\t"
				"movaps 0x00(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				"add $0x10, %%rax \n\t"
				"add $0x10, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"movss 0x00(%%rax), %%xmm0 \n\t"
				"movss 0x00(%%rbx), %%xmm1 \n\t"
				"pshufd $0x00, %%xmm0, %%xmm0 \n\t"
				"pshufd $0x00, %%xmm1, %%xmm1 \n\t"

				// scale by alpha
				"mulps %%xmm0, %%xmm12 \n\t"
				"mulps %%xmm0, %%xmm13 \n\t"
				"mulps %%xmm0, %%xmm14 \n\t"
				"mulps %%xmm0, %%xmm15 \n\t"

				"xorps %%xmm0, %%xmm0 \n\t"
				"ucomiss %%xmm1, %%xmm0 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// optionally load C fragment and scale by beta
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx
				"movq %[C_stride], %%r14 \n\t"// C stride is r14

				"movups 0x0(%%rcx), %%xmm4 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x0(%%rcx), %%xmm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x0(%%rcx), %%xmm6 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups 0x0(%%rcx), %%xmm7 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"mulps %%xmm1, %%xmm4 \n\t"
				"mulps %%xmm1, %%xmm5 \n\t"
				"mulps %%xmm1, %%xmm6 \n\t"
				"mulps %%xmm1, %%xmm7 \n\t"

				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				"BETAZERO%=: \n\t"
				// store final result into D fragment
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx
				"movq %[D_stride], %%r14 \n\t"// D stride is r14

				"movups %%xmm12, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm13, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm14, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm15, 0x00(%%rcx) \n\t"

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
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%r14");
	}

	void pack_sse2_4xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 4);
		assert(ml::cpu::is_aligned(dst.data(), register_size<XMM>()));

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
			uint64_t k_iter = dst.rows() / 8;
			uint64_t k_left = dst.rows() % 8;
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
		assert(ml::cpu::is_aligned(dst.data(), register_size<XMM>()));

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

