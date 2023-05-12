/*
 * avx_gemm_kernels.cpp
 *
 *  Created on: May 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include "gemm_kernels.hpp"
#include "Fragment.hpp"
#include "../../utils.hpp"
#include "../../vectors/vectors.hpp"

#include <x86intrin.h>
#include <cinttypes>
#include <cassert>

namespace ml
{

	void gemm_avx_8x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 8);
		assert(B.columns() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(A.data(), register_size<YMM>()));
		assert(cpu::is_aligned(B.data(), register_size<YMM>()));
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
				"movq %[A_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx

				// Set accumulators to zero.
				"vpxor %%ymm8, %%ymm8, %%ymm8 \n\t"
				"vpxor %%ymm9, %%ymm9, %%ymm9 \n\t"
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rax), %%ymm0 \n\t"
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x20(%%rax), %%ymm0 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x40(%%rax), %%ymm0 \n\t"
				"vmovaps 0x40(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0x60(%%rax), %%ymm0 \n\t"
				"vmovaps 0x60(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"

				"add $0x80, %%rax \n\t"
				"add $0x80, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rax), %%ymm0 \n\t"
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"
				"add $0x20, %%rax \n\t"
				"add $0x20, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				// permute back to row-najor storage
				"vperm2f128 $0x12, %%ymm8, %%ymm12, %%ymm0 \n\t"
				"vperm2f128 $0x30, %%ymm8, %%ymm12, %%ymm4 \n\t"
				"vperm2f128 $0x12, %%ymm9, %%ymm13, %%ymm1 \n\t"
				"vperm2f128 $0x30, %%ymm9, %%ymm13, %%ymm5 \n\t"
				"vperm2f128 $0x12, %%ymm10, %%ymm14, %%ymm2 \n\t"
				"vperm2f128 $0x30, %%ymm10, %%ymm14, %%ymm6 \n\t"
				"vperm2f128 $0x12, %%ymm11, %%ymm15, %%ymm3 \n\t"
				"vperm2f128 $0x30, %%ymm11, %%ymm15, %%ymm7 \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm8 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm9 \n\t"

				// scale by alpha
				"vmulps %%ymm8, %%ymm0, %%ymm0 \n\t"
				"vmulps %%ymm8, %%ymm1, %%ymm1 \n\t"
				"vmulps %%ymm8, %%ymm2, %%ymm2 \n\t"
				"vmulps %%ymm8, %%ymm3, %%ymm3 \n\t"
				"vmulps %%ymm8, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm8, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm8, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm8, %%ymm7, %%ymm7 \n\t"

				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"
				"ucomiss %%xmm9, %%xmm15 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// beta != 0 case
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"vmovups 0x00(%%rcx), %%ymm12 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm13 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm14 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm15 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmulps %%ymm9, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm9, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm9, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm9, %%ymm15, %%ymm15 \n\t"

				"vaddps %%ymm0, %%ymm12, %%ymm0 \n\t"
				"vaddps %%ymm1, %%ymm13, %%ymm1 \n\t"
				"vaddps %%ymm2, %%ymm14, %%ymm2 \n\t"
				"vaddps %%ymm3, %%ymm15, %%ymm3 \n\t"

				"vmovups 0x00(%%rcx), %%ymm12 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm13 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm14 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm15 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmulps %%ymm9, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm9, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm9, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm9, %%ymm15, %%ymm15 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm4 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm5 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm6 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm7 \n\t"

				"BETAZERO%=: \n\t"
				// beta == 0 case
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx

				"vmovups %%ymm0, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm1, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm2, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm3, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm4, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm5, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm6, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm7, 0x00(%%rcx) \n\t"

				"vzeroupper \n\t"

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
	void gemm_avx_4x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 4);
		assert(B.columns() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(A.data(), register_size<YMM>()));
		assert(cpu::is_aligned(B.data(), register_size<YMM>()));
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
				"movq %[A_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx

				// Set accumulators to zero.
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm0 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm4 \n\t"

				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"

				"vaddps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm2, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm3, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm4, %%ymm15, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x20(%%rbx), %%ymm0 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x18(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm4 \n\t"

				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"

				"vaddps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm2, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm3, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm4, %%ymm15, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x40(%%rbx), %%ymm0 \n\t"
				"vbroadcastss 0x20(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x28(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm4 \n\t"

				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"

				"vaddps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm2, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm3, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm4, %%ymm15, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0x80(%%rbx), %%ymm0 \n\t"
				"vbroadcastss 0x30(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x38(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm4 \n\t"

				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"

				"vaddps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm2, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm3, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm4, %%ymm15, %%ymm15 \n\t"

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
				"vmovaps 0x00(%%rbx), %%ymm0 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm4 \n\t"

				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"

				"vaddps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm2, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm3, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm4, %%ymm15, %%ymm15 \n\t"

				"add $0x10, %%rax \n\t"
				"add $0x20, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"
				"ucomiss %%xmm9, %%xmm15 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// beta != 0 case
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"vmovups 0x00(%%rcx), %%ymm4 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm6 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm7 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmulps %%ymm1, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm1, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm1, %%ymm7, %%ymm7 \n\t"

				"vaddps %%ymm12, %%ymm4, %%ymm12 \n\t"
				"vaddps %%ymm13, %%ymm5, %%ymm13 \n\t"
				"vaddps %%ymm14, %%ymm6, %%ymm14 \n\t"
				"vaddps %%ymm15, %%ymm7, %%ymm15 \n\t"

				"BETAZERO%=: \n\t"
				// beta == 0 case
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx

				"vmovups %%ymm12, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm13, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm14, 0x00(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm15, 0x00(%%rcx) \n\t"

				"vzeroupper \n\t"

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

	void pack_avx_8xK_fp32(int M, int K, const void *__restrict__ src, int src_stride, void *__restrict__ dst, char op)
	{
		assert(M == 8);
		assert(K >= 0);
		assert(src != nullptr);
		assert(dst != nullptr && cpu::is_aligned(dst, 32));
		assert(op == 'n' || op == 't');
	}

} /* namespace ml */

