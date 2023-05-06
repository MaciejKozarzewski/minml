/*
 * avx2_gemm_kernels.cpp
 *
 *  Created on: May 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include "../../utils.hpp"

#include <x86intrin.h>
#include <cinttypes>
#include <cassert>

namespace ml
{
	void gemm_avx2_fma_6x16_fp32(int M, int N, int K, const void *alpha_ptr, const void *__restrict__ lhs_ptr, const void *__restrict__ rhs_ptr,
			const void *beta_ptr, void *__restrict__ dst_ptr, int dst_stride)
	{
		assert(M == 6);
		assert(N == 16);
		assert(K >= 0);
		assert(alpha_ptr != nullptr);
		assert(lhs_ptr != nullptr && is_aligned(lhs_ptr, 32));
		assert(rhs_ptr != nullptr && is_aligned(rhs_ptr, 32));
		assert(beta_ptr != nullptr);
		assert(dst_ptr != nullptr);
		assert(dst_stride > 0);

		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		uint64_t stride = dst_stride;

		asm volatile(
				"movq %[lhs_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[rhs_ptr], %%rbx \n\t"// rhs pointer is in rbx
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				// Set accumulators to zero.
				"vpxor %%ymm4, %%ymm4, %%ymm4 \n\t"
				"vpxor %%ymm5, %%ymm5, %%ymm5 \n\t"
				"vpxor %%ymm6, %%ymm6, %%ymm6 \n\t"
				"vpxor %%ymm7, %%ymm7, %%ymm7 \n\t"
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

				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x40(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x60(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x18(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x20(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x28(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x80(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xA0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x38(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x40(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0xC0(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xE0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x48(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x50(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x54(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x58(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x5C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				"add $0x60, %%rax \n\t"
				"add $0x100, %%rbx \n\t"
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"
				"add $0x18, %%rax \n\t"
				"add $0x40, %%rbx \n\t"

				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm0, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm0, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm0, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm0, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm0, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"movq %[stride], %%r12 \n\t"// stride is r12
				"shlq $2, %%r12 \n\t"// multiply stride by sizeof(float)
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"ucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm4 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm6 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm7 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm8 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm9 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm10 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm11 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm14 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm15 \n\t"
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"BETAZERO%=: \n\t"
				"vmovups %%ymm4, 0x00(%%rcx) \n\t"
				"vmovups %%ymm5, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm6, 0x00(%%rcx) \n\t"
				"vmovups %%ymm7, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm8, 0x00(%%rcx) \n\t"
				"vmovups %%ymm9, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm10, 0x00(%%rcx) \n\t"
				"vmovups %%ymm11, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm12, 0x00(%%rcx) \n\t"
				"vmovups %%ymm13, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm14, 0x00(%%rcx) \n\t"
				"vmovups %%ymm15, 0x20(%%rcx) \n\t"

				"vzeroupper \n\t"

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

	void pack_avx2_6xK_fp32(int M, int K, const void *__restrict__ src, void *__restrict__ dst, char op)
	{
		assert(M == 6);
		assert(K >= 0);
		assert(src != nullptr);
		assert(dst != nullptr && is_aligned(dst, 32));
		assert(op == 'n' || op == 't');
	}
	void pack_avx2_16xK_fp32(int M, int K, const void *__restrict__ src, void *__restrict__ dst, char op)
	{
		assert(M == 16);
		assert(K >= 0);
		assert(src != nullptr);
		assert(dst != nullptr && is_aligned(dst, 32));
		assert(op == 'n' || op == 't');
	}

} /* namespace ml */

