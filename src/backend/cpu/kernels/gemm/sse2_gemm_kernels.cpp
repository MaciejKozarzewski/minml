/*
 * sse2_gemm_kernels.cpp
 *
 *  Created on: May 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include "Fragment.hpp"
#include "gemm_kernels.hpp"
#include "../../utils.hpp"
#include "../../vectors/vectors.hpp"

#include <x86intrin.h>
#include <cinttypes>
#include <cassert>

namespace ml
{

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
				"pxor %%xmm8, %%xmm8 \n\t"
				"pxor %%xmm9, %%xmm9 \n\t"
				"pxor %%xmm10, %%xmm10 \n\t"
				"pxor %%xmm11, %%xmm11 \n\t"
				"pxor %%xmm12, %%xmm12 \n\t"
				"pxor %%xmm13, %%xmm13 \n\t"
				"pxor %%xmm14, %%xmm14 \n\t"
				"pxor %%xmm15, %%xmm15 \n\t"

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

				"pxor %%xmm0, %%xmm0 \n\t"
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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14");
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
				"pxor %%xmm12, %%xmm12 \n\t"
				"pxor %%xmm13, %%xmm13 \n\t"
				"pxor %%xmm14, %%xmm14 \n\t"
				"pxor %%xmm15, %%xmm15 \n\t"

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

				"pxor %%xmm0, %%xmm0 \n\t"
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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14");
	}

	void gemm_sse2_8x4_fp32(int M, int N, int K, const void *alpha_ptr, const void *__restrict__ lhs_ptr, const void *__restrict__ rhs_ptr,
			const void *beta_ptr, void *__restrict__ dst_ptr, int dst_stride)
	{
		assert(M == 8);
		assert(N == 4);
		assert(K >= 0);
		assert(alpha_ptr != nullptr);
		assert(lhs_ptr != nullptr && cpu::is_aligned(lhs_ptr, 16));
		assert(rhs_ptr != nullptr && cpu::is_aligned(rhs_ptr, 16));
		assert(beta_ptr != nullptr);
		assert(dst_ptr != nullptr);
		assert(dst_stride > 0);

		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		uint64_t stride = dst_stride;

		asm volatile(
				"movq %[lhs_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[rhs_ptr], %%rbx \n\t"// rhs pointer is in rbx

				// Set accumulators to zero.
				"pxor %%xmm8, %%xmm8 \n\t"
				"pxor %%xmm9, %%xmm9 \n\t"
				"pxor %%xmm10, %%xmm10 \n\t"
				"pxor %%xmm11, %%xmm11 \n\t"
				"pxor %%xmm12, %%xmm12 \n\t"
				"pxor %%xmm13, %%xmm13 \n\t"
				"pxor %%xmm14, %%xmm14 \n\t"
				"pxor %%xmm15, %%xmm15 \n\t"

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
				"movq %[stride], %%r12 \n\t"// stride is r12
				"shlq $2, %%r12 \n\t"// multiply stride by sizeof(float)
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"pxor %%xmm0, %%xmm0 \n\t"
				"ucomiss %%xmm1, %%xmm0 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// beta != 0 case
				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm7 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"mulps %%xmm1, %%xmm4 \n\t"
				"mulps %%xmm1, %%xmm5 \n\t"
				"mulps %%xmm1, %%xmm6 \n\t"
				"mulps %%xmm1, %%xmm7 \n\t"

				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm7 \n\t"

				"mulps %%xmm1, %%xmm4 \n\t"
				"mulps %%xmm1, %%xmm5 \n\t"
				"mulps %%xmm1, %%xmm6 \n\t"
				"mulps %%xmm1, %%xmm7 \n\t"

				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"BETAZERO%=: \n\t"
				// beta == 0 case
				"movups %%xmm8, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm9, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm10, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm11, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm12, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm13, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm14, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm15, 0x00(%%rcx) \n\t"

				:// outputs
				:// inputs
				[lhs_ptr] "m"(lhs_ptr),
				[rhs_ptr] "m"(rhs_ptr),
				[dst_ptr] "m"(dst_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[stride] "m"(stride),
				[alpha_ptr] "m"(alpha_ptr),
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
				"%r12", "%r13", "%r14");
	}

	void pack_sse2_8xK_fp32(Fragment &dst, const Fragment &src, int src_row, int src_col, char src_op) noexcept
	{
		assert(src_op == 'n' || src_op == 't');

		const int M = dst.columns();
		const int K = dst.rows();
		const int max_rows = std::min(K, src.rows() - src_row);
		const int max_cols = std::min(M, src.columns() - src_col);

		const float *src_ptr = src.data<float>() + src.offset_at(src_row, src_col);
		float *dst_ptr = dst.data<float>();

		SIMD_NAMESPACE::Loader<XMM> load;
		SIMD_NAMESPACE::Storer<XMM> store;

		if (src_op == 'n')
		{ // pack K x M fragment
			if (max_cols == 8)
			{ // fragment can be loaded fully
				for (int k = 0; k < max_rows; k++)
				{
					const __m128 a = load(src_ptr, 4);
					const __m128 b = load(src_ptr + 4, 4);
					store(dst_ptr, a, 4);
					store(dst_ptr + 4, b, 4);
					src_ptr += src.stride();
					dst_ptr += dst.stride();
				}
			}
			else
			{
				if (max_cols >= 4)
				{ // at least the first half of a fragment can be loaded fully
					const int partial_size = max_cols - 4;
					for (int k = 0; k < max_rows; k++)
					{
						const __m128 a = load(src_ptr, 4);
						const __m128 b = load(src_ptr + 4, partial_size);
						store(dst_ptr, a, 4);
						store(dst_ptr + 4, b, partial_size);
						src_ptr += src.stride();
						dst_ptr += dst.stride();
					}
				}
				else
				{
					const int partial_size = max_cols;
					for (int k = 0; k < max_rows; k++)
					{
						const __m128 a = load(src_ptr, partial_size);
						store(dst_ptr, a, partial_size);
						src_ptr += src.stride();
						dst_ptr += dst.stride();
					}
				}
			}
		}
		else
		{ // pack M x K fragment
			__m128 tile[8];
			if (max_rows == 8)
			{ // fragment can be loaded fully
				if (max_cols >= 4)
				{

				}
				else
				{

				}
				for (int k = 0; k < max_rows; k++)
				{

//					for(int )
					const __m128 a = load(src_ptr, 4);
					const __m128 b = load(src_ptr + 4, 4);
					store(dst_ptr, a, 4);
					store(dst_ptr + 4, b, 4);
					src_ptr += src.stride();
					dst_ptr += dst.stride();
				}
			}
			else
			{
				if (max_rows >= 4)
				{ // at least the first half of a fragment can be loaded fully
					const int partial_size = max_cols - 4;
					for (int k = 0; k < max_rows; k++)
					{
						const __m128 a = load(src_ptr, 4);
						const __m128 b = load(src_ptr + 4, partial_size);
						store(dst_ptr, a, 4);
						store(dst_ptr + 4, b, partial_size);
						src_ptr += src.stride();
						dst_ptr += dst.stride();
					}
				}
				else
				{
					const int partial_size = max_cols;
					for (int k = 0; k < max_rows; k++)
					{
						const __m128 a = load(src_ptr, partial_size);
						store(dst_ptr, a, partial_size);
						src_ptr += src.stride();
						dst_ptr += dst.stride();
					}
				}
			}
		}
	}
	void pack_sse2_4xK_fp32(Fragment &dst, const Fragment &src, int src_row, int src_col, char src_op) noexcept
	{
		assert(src_op == 'n' || src_op == 't');

		const int M = dst.columns();
		const int K = dst.rows();
		const int max_rows = std::min(K, src.rows() - src_row);
		const int max_cols = std::min(M, src.columns() - src_col);

		const float *src_ptr = src.data<float>() + src.offset_at(src_row, src_col);
		float *dst_ptr = dst.data<float>();

		SIMD_NAMESPACE::Loader<XMM> load;
		SIMD_NAMESPACE::Storer<XMM> store;

		if (src_op == 'n')
		{ // pack K x M fragment
			if (max_cols == 4)
			{ // fragment can be loaded fully
				for (int k = 0; k < max_rows; k++)
				{
					const __m128 a = load(src_ptr, 4);
					store(dst_ptr, a, 4);
					src_ptr += src.stride();
					dst_ptr += dst.stride();
				}
			}
			else
			{
				const int partial_size = max_cols;
				for (int k = 0; k < max_rows; k++)
				{
					const __m128 a = load(src_ptr, partial_size);
					store(dst_ptr, a, partial_size);
					src_ptr += src.stride();
					dst_ptr += dst.stride();
				}
			}
		}
		else
		{ // pack M x K fragment
			const int full_cols = max_cols - max_cols % 4;
			for (int k = 0; k < full_cols; k += 4)
			{
				__m128 tile[4];
			}
			__m128 tile[8];
			if (max_rows == 8)
			{ // fragment can be loaded fully
				if (max_cols >= 4)
				{

				}
				else
				{

				}
				for (int k = 0; k < max_rows; k++)
				{

					//					for(int )
					const __m128 a = load(src_ptr, 4);
					const __m128 b = load(src_ptr + 4, 4);
					store(dst_ptr, a, 4);
					store(dst_ptr + 4, b, 4);
					src_ptr += src.stride();
					dst_ptr += dst.stride();
				}
			}
			else
			{
				if (max_rows >= 4)
				{ // at least the first half of a fragment can be loaded fully
					const int partial_size = max_cols - 4;
					for (int k = 0; k < max_rows; k++)
					{
						const __m128 a = load(src_ptr, 4);
						const __m128 b = load(src_ptr + 4, partial_size);
						store(dst_ptr, a, 4);
						store(dst_ptr + 4, b, partial_size);
						src_ptr += src.stride();
						dst_ptr += dst.stride();
					}
				}
				else
				{
					const int partial_size = max_cols;
					for (int k = 0; k < max_rows; k++)
					{
						const __m128 a = load(src_ptr, partial_size);
						store(dst_ptr, a, partial_size);
						src_ptr += src.stride();
						dst_ptr += dst.stride();
					}
				}
			}
		}
	}

} /* namespace ml */

