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
#include "../vectors/vectors.hpp"

#include <x86intrin.h>
#include <cinttypes>
#include <cassert>

namespace ml
{
	void gemm_avx2_fma_6x16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.stride() == 6);
		assert(B.stride() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), register_size<YMM>()));

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
				"movq %[D_stride], %%r14 \n\t" // D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx
				"prefetcht0 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"prefetcht0 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"prefetcht0 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"prefetcht0 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"prefetcht0 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"prefetcht0 0x0(%%rcx) \n\t"

				"movq %[A_ptr], %%rax \n\t"// A pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// B pointer is in rbx

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
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

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

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

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
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"vucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm4 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm6 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm7 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm8 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm9 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm10 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm11 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm13 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm14 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm15 \n\t"

				"BETAZERO%=: \n\t"
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx

				"vmovups %%ymm4, 0x00(%%rcx) \n\t"
				"vmovups %%ymm5, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm6, 0x00(%%rcx) \n\t"
				"vmovups %%ymm7, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm8, 0x00(%%rcx) \n\t"
				"vmovups %%ymm9, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm10, 0x00(%%rcx) \n\t"
				"vmovups %%ymm11, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm12, 0x00(%%rcx) \n\t"
				"vmovups %%ymm13, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm14, 0x00(%%rcx) \n\t"
				"vmovups %%ymm15, 0x20(%%rcx) \n\t"

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
	void gemm_avx2_fma_6x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.stride() == 6);
		assert(B.stride() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
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
				"vmovaps 0x00(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x20(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x18(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x20(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x28(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x40(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x38(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x40(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0x60(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x48(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x50(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x54(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x58(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x5C(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				"add $0x60, %%rax \n\t"
				"add $0x80, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"
				"add $0x18, %%rax \n\t"
				"add $0x20, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"vucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"vmovups 0x0(%%rcx), %%ymm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm3 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm4 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm6 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm7 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm8 \n\t"

				"vfmadd231ps %%ymm2, %%ymm1, %%ymm10 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm11 \n\t"
				"vfmadd231ps %%ymm4, %%ymm1, %%ymm12 \n\t"
				"vfmadd231ps %%ymm5, %%ymm1, %%ymm13 \n\t"
				"vfmadd231ps %%ymm6, %%ymm1, %%ymm14 \n\t"
				"vfmadd231ps %%ymm7, %%ymm1, %%ymm15 \n\t"

				"BETAZERO%=: \n\t"
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx

				"vmovups %%ymm10, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm11, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm12, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm13, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm14, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm15, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride

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
	void gemm_avx2_fma_12x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.stride() == 12);
		assert(B.stride() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), register_size<YMM>()));

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
				"vmovaps (4*0*8)(%%rbx), %%ymm3 \n\t"

				"vbroadcastss (4*0*12+4*0)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*0*12+4*1)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*0*12+4*2)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastss (4*0*12+4*3)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*0*12+4*4)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*0*12+4*5)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastss (4*0*12+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*0*12+4*7)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*0*12+4*8)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastss (4*0*12+4*9)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*0*12+4*10)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*0*12+4*11)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				// iteration 1
				"vmovaps (4*1*8)(%%rbx), %%ymm3 \n\t"

				"vbroadcastss (4*1*12+4*0)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*1*12+4*1)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*1*12+4*2)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastss (4*1*12+4*3)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*1*12+4*4)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*1*12+4*5)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastss (4*1*12+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*1*12+4*7)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*1*12+4*8)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastss (4*1*12+4*9)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*1*12+4*10)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*1*12+4*11)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				// iteration 2
				"vmovaps (4*2*8)(%%rbx), %%ymm3 \n\t"

				"vbroadcastss (4*2*12+4*0)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*2*12+4*1)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*2*12+4*2)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastss (4*2*12+4*3)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*2*12+4*4)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*2*12+4*5)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastss (4*2*12+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*2*12+4*7)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*2*12+4*8)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastss (4*2*12+4*9)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*2*12+4*10)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*2*12+4*11)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				// iteration 3
				"vmovaps (4*3*8)(%%rbx), %%ymm3 \n\t"

				"vbroadcastss (4*3*12+4*0)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*3*12+4*1)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*3*12+4*2)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastss (4*3*12+4*3)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*3*12+4*4)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*3*12+4*5)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastss (4*3*12+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*3*12+4*7)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*3*12+4*8)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastss (4*3*12+4*9)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*3*12+4*10)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*3*12+4*11)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				"add $(4*4*12), %%rax \n\t"
				"add $(4*4*8), %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps (4*0*8)(%%rbx), %%ymm3 \n\t"

				"vbroadcastss (4*0*12+4*0)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*0*12+4*1)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*0*12+4*2)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastss (4*0*12+4*3)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*0*12+4*4)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*0*12+4*5)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastss (4*0*12+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*0*12+4*7)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*0*12+4*8)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastss (4*0*12+4*9)(%%rax), %%ymm0 \n\t"
				"vbroadcastss (4*0*12+4*10)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*0*12+4*11)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				"add $(4*1*12), %%rax \n\t"
				"add $(4*1*8), %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm0 \n\t"

				// scale by alpha
				"vmulps %%ymm1, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm1, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm1, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm1, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm1, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm1, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm1, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm1, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm1, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"vucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				// rows 0-2
				"vmovups 0x0(%%rcx), %%ymm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm3 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vfmadd231ps %%ymm1, %%ymm0, %%ymm4 \n\t"
				"vfmadd231ps %%ymm2, %%ymm0, %%ymm5 \n\t"
				"vfmadd231ps %%ymm3, %%ymm0, %%ymm6 \n\t"
				// rows 3-5
				"vmovups 0x0(%%rcx), %%ymm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm3 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vfmadd231ps %%ymm1, %%ymm0, %%ymm7 \n\t"
				"vfmadd231ps %%ymm2, %%ymm0, %%ymm8 \n\t"
				"vfmadd231ps %%ymm3, %%ymm0, %%ymm9 \n\t"
				// rows 6-8
				"vmovups 0x0(%%rcx), %%ymm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm3 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vfmadd231ps %%ymm1, %%ymm0, %%ymm10 \n\t"
				"vfmadd231ps %%ymm2, %%ymm0, %%ymm11 \n\t"
				"vfmadd231ps %%ymm3, %%ymm0, %%ymm12 \n\t"
				// rows 9-11
				"vmovups 0x0(%%rcx), %%ymm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm1, %%ymm0, %%ymm13 \n\t"
				"vfmadd231ps %%ymm2, %%ymm0, %%ymm14 \n\t"
				"vfmadd231ps %%ymm3, %%ymm0, %%ymm15 \n\t"

				"BETAZERO%=: \n\t"
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx

				"vmovups %%ymm4, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm5, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm6, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm7, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm8, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm9, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm10, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm11, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm12, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm13, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm14, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm15, 0x0(%%rcx) \n\t"

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
	void gemm_avx2_fma_24x4_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.stride() == 24);
		assert(B.stride() == 4);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), register_size<YMM>()));

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
				"movq %[beta_ptr], %%rbx \n\t" // load address of beta
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"vucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je ZEROACC%= \n\t"
				// load destination pointer and stride
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx
				"movq %%rcx, %%rax \n\t"// C pointer is in rax
				"add %%r14, %%rax \n\t"// add stride
				"shlq $1, %%r14 \n\t"// multiply stride by 2

				// rows 0-7
				"vmovups 0x0(%%rcx), %%xmm4 \n\t"
				"vmovups 0x0(%%rax), %%xmm0 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups 0x0(%%rcx), %%xmm5 \n\t"
				"vmovups 0x0(%%rax), %%xmm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups 0x0(%%rcx), %%xmm6 \n\t"
				"vmovups 0x0(%%rax), %%xmm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups 0x0(%%rcx), %%xmm7 \n\t"
				"vmovups 0x0(%%rax), %%xmm3 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vinsertf128 $0x1, %%xmm0, %%ymm4, %%ymm4 \n\t"
				"vinsertf128 $0x1, %%xmm1, %%ymm5, %%ymm5 \n\t"
				"vinsertf128 $0x1, %%xmm2, %%ymm6, %%ymm6 \n\t"
				"vinsertf128 $0x1, %%xmm3, %%ymm7, %%ymm7 \n\t"

				// rows 8-15
				"vmovups 0x0(%%rcx), %%xmm8 \n\t"
				"vmovups 0x0(%%rax), %%xmm0 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups 0x0(%%rcx), %%xmm9 \n\t"
				"vmovups 0x0(%%rax), %%xmm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups 0x0(%%rcx), %%xmm10 \n\t"
				"vmovups 0x0(%%rax), %%xmm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups 0x0(%%rcx), %%xmm11 \n\t"
				"vmovups 0x0(%%rax), %%xmm3 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vinsertf128 $0x1, %%xmm0, %%ymm8, %%ymm8 \n\t"
				"vinsertf128 $0x1, %%xmm1, %%ymm9, %%ymm9 \n\t"
				"vinsertf128 $0x1, %%xmm2, %%ymm10, %%ymm10 \n\t"
				"vinsertf128 $0x1, %%xmm3, %%ymm11, %%ymm11 \n\t"

				// rows 16-23
				"vmovups 0x0(%%rcx), %%xmm12 \n\t"
				"vmovups 0x0(%%rax), %%xmm0 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups 0x0(%%rcx), %%xmm13 \n\t"
				"vmovups 0x0(%%rax), %%xmm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups 0x0(%%rcx), %%xmm14 \n\t"
				"vmovups 0x0(%%rax), %%xmm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups 0x0(%%rcx), %%xmm15 \n\t"
				"vmovups 0x0(%%rax), %%xmm3 \n\t"
				"vinsertf128 $0x1, %%xmm0, %%ymm12, %%ymm12 \n\t"
				"vinsertf128 $0x1, %%xmm1, %%ymm13, %%ymm13 \n\t"
				"vinsertf128 $0x1, %%xmm2, %%ymm14, %%ymm14 \n\t"
				"vinsertf128 $0x1, %%xmm3, %%ymm15, %%ymm15 \n\t"

				"vpermpd $0xD8, %%ymm4, %%ymm4 \n\t"
				"vpermpd $0xD8, %%ymm5, %%ymm5 \n\t"
				"vpermpd $0xD8, %%ymm6, %%ymm6 \n\t"
				"vpermpd $0xD8, %%ymm7, %%ymm7 \n\t"
				"vpermpd $0xD8, %%ymm8, %%ymm8 \n\t"
				"vpermpd $0xD8, %%ymm9, %%ymm9 \n\t"
				"vpermpd $0xD8, %%ymm10, %%ymm10 \n\t"
				"vpermpd $0xD8, %%ymm11, %%ymm11 \n\t"
				"vpermpd $0xD8, %%ymm12, %%ymm12 \n\t"
				"vpermpd $0xD8, %%ymm13, %%ymm13 \n\t"
				"vpermpd $0xD8, %%ymm14, %%ymm14 \n\t"
				"vpermpd $0xD8, %%ymm15, %%ymm15 \n\t"

				"vpermilps $0xD8, %%ymm4, %%ymm4 \n\t"
				"vpermilps $0xD8, %%ymm5, %%ymm5 \n\t"
				"vpermilps $0xD8, %%ymm6, %%ymm6 \n\t"
				"vpermilps $0xD8, %%ymm7, %%ymm7 \n\t"
				"vpermilps $0xD8, %%ymm8, %%ymm8 \n\t"
				"vpermilps $0xD8, %%ymm9, %%ymm9 \n\t"
				"vpermilps $0xD8, %%ymm10, %%ymm10 \n\t"
				"vpermilps $0xD8, %%ymm11, %%ymm11 \n\t"
				"vpermilps $0xD8, %%ymm12, %%ymm12 \n\t"
				"vpermilps $0xD8, %%ymm13, %%ymm13 \n\t"
				"vpermilps $0xD8, %%ymm14, %%ymm14 \n\t"
				"vpermilps $0xD8, %%ymm15, %%ymm15 \n\t"

				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"// load beta again
				"vmulps %%ymm1, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm1, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm1, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm1, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm1, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm1, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm1, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm1, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm1, %%ymm15, %%ymm15 \n\t"
				"jmp LOOPSTART%= \n\t"

				"ZEROACC%=: \n\t"
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

				"LOOPSTART%=: \n\t"
				"movq %[A_ptr], %%rax \n\t"// lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx
				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps (4*0*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"// b0 b0 b1 b1 b2 b2 b3 b3

				"vbroadcastsd (0*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (0*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (0*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (0*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (0*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (0*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				// iteration 1
				"vmovaps (4*1*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"

				"vbroadcastsd (1*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (1*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (1*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (1*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (1*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (1*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (1*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (1*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (1*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (1*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (1*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (1*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				// iteration 2
				"vmovaps (4*2*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"

				"vbroadcastsd (2*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (2*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (2*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (2*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (2*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (2*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (2*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (2*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (2*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (2*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (2*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (2*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				// iteration 3
				"vmovaps (4*3*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"

				"vbroadcastsd (3*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (3*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (3*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (3*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (3*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (3*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (3*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (3*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (3*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (3*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (3*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (3*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				"add $(4*24*4), %%rax \n\t"
				"add $(4*4*4), %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps (4*0*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"

				"vbroadcastsd (0*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (0*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (0*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (0*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (0*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (0*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				"add $(4*24*1), %%rax \n\t"
				"add $(4*1*4), %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"

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

				// shuffle back into correct layout
				"vpermilps $0xD8, %%ymm4, %%ymm4 \n\t"
				"vpermilps $0xD8, %%ymm5, %%ymm5 \n\t"
				"vpermilps $0xD8, %%ymm6, %%ymm6 \n\t"
				"vpermilps $0xD8, %%ymm7, %%ymm7 \n\t"
				"vpermilps $0xD8, %%ymm8, %%ymm8 \n\t"
				"vpermilps $0xD8, %%ymm9, %%ymm9 \n\t"
				"vpermilps $0xD8, %%ymm10, %%ymm10 \n\t"
				"vpermilps $0xD8, %%ymm11, %%ymm11 \n\t"
				"vpermilps $0xD8, %%ymm12, %%ymm12 \n\t"
				"vpermilps $0xD8, %%ymm13, %%ymm13 \n\t"
				"vpermilps $0xD8, %%ymm14, %%ymm14 \n\t"
				"vpermilps $0xD8, %%ymm15, %%ymm15 \n\t"

				"vpermpd $0xD8, %%ymm4, %%ymm4 \n\t"
				"vpermpd $0xD8, %%ymm5, %%ymm5 \n\t"
				"vpermpd $0xD8, %%ymm6, %%ymm6 \n\t"
				"vpermpd $0xD8, %%ymm7, %%ymm7 \n\t"
				"vpermpd $0xD8, %%ymm8, %%ymm8 \n\t"
				"vpermpd $0xD8, %%ymm9, %%ymm9 \n\t"
				"vpermpd $0xD8, %%ymm10, %%ymm10 \n\t"
				"vpermpd $0xD8, %%ymm11, %%ymm11 \n\t"
				"vpermpd $0xD8, %%ymm12, %%ymm12 \n\t"
				"vpermpd $0xD8, %%ymm13, %%ymm13 \n\t"
				"vpermpd $0xD8, %%ymm14, %%ymm14 \n\t"
				"vpermpd $0xD8, %%ymm15, %%ymm15 \n\t"

				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx
				"movq %%rcx, %%rax \n\t"// C pointer is in rax
				"add %%r14, %%rax \n\t"// add stride
				"shlq $1, %%r14 \n\t"// multiply stride by 2

				// rows 0-7
				"vextractf128 $(0x1), %%ymm4, %%xmm0 \n\t"
				"vextractf128 $(0x1), %%ymm5, %%xmm1 \n\t"
				"vextractf128 $(0x1), %%ymm6, %%xmm2 \n\t"
				"vextractf128 $(0x1), %%ymm7, %%xmm3 \n\t"
				"vmovups %%xmm4, 0x0(%%rcx) \n\t"
				"vmovups %%xmm0, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups %%xmm5, 0x0(%%rcx) \n\t"
				"vmovups %%xmm1, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups %%xmm6, 0x0(%%rcx) \n\t"
				"vmovups %%xmm2, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups %%xmm7, 0x0(%%rcx) \n\t"
				"vmovups %%xmm3, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride

				// rows 8-15
				"vextractf128 $(0x1), %%ymm8, %%xmm0 \n\t"
				"vextractf128 $(0x1), %%ymm9, %%xmm1 \n\t"
				"vextractf128 $(0x1), %%ymm10, %%xmm2 \n\t"
				"vextractf128 $(0x1), %%ymm11, %%xmm3 \n\t"
				"vmovups %%xmm8, 0x0(%%rcx) \n\t"
				"vmovups %%xmm0, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups %%xmm9, 0x0(%%rcx) \n\t"
				"vmovups %%xmm1, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups %%xmm10, 0x0(%%rcx) \n\t"
				"vmovups %%xmm2, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups %%xmm11, 0x0(%%rcx) \n\t"
				"vmovups %%xmm3, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride

				// rows 16-23
				"vextractf128 $(0x1), %%ymm12, %%xmm0 \n\t"
				"vextractf128 $(0x1), %%ymm13, %%xmm1 \n\t"
				"vextractf128 $(0x1), %%ymm14, %%xmm2 \n\t"
				"vextractf128 $(0x1), %%ymm15, %%xmm3 \n\t"
				"vmovups %%xmm12, 0x0(%%rcx) \n\t"
				"vmovups %%xmm0, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups %%xmm13, 0x0(%%rcx) \n\t"
				"vmovups %%xmm1, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups %%xmm14, 0x0(%%rcx) \n\t"
				"vmovups %%xmm2, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovups %%xmm15, 0x0(%%rcx) \n\t"
				"vmovups %%xmm3, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride

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
	void gemm_avx2_fma_6x16_fp16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.stride() == 6);
		assert(B.stride() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), register_size<YMM>()));
		assert(beta_ptr != nullptr);

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float16 *C_ptr = C.data<float16>();
		float16 *D_ptr = D.data<float16>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float16);
		const uint64_t D_stride = D.stride() * sizeof(float16);
		float alpha = reinterpret_cast<const float*>(alpha_ptr)[0];
		assert(alpha != 0.0f);
		float beta = reinterpret_cast<const float*>(beta_ptr)[0] / alpha;
		const float *_beta = &beta;

		asm volatile(
				"movq %[beta_ptr], %%rbx \n\t" // load address of beta
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"vucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je ZEROACC%= \n\t"
				// load and convert dst
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"vmovups 0x00(%%rcx), %%xmm4 \n\t"
				"vmovups 0x10(%%rcx), %%xmm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%xmm6 \n\t"
				"vmovups 0x10(%%rcx), %%xmm7 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%xmm8 \n\t"
				"vmovups 0x10(%%rcx), %%xmm9 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%xmm10 \n\t"
				"vmovups 0x10(%%rcx), %%xmm11 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%xmm12 \n\t"
				"vmovups 0x10(%%rcx), %%xmm13 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%xmm14 \n\t"
				"vmovups 0x10(%%rcx), %%xmm15 \n\t"

				"vcvtph2ps %%xmm4, %%ymm4 \n\t"
				"vcvtph2ps %%xmm5, %%ymm5 \n\t"
				"vcvtph2ps %%xmm6, %%ymm6 \n\t"
				"vcvtph2ps %%xmm7, %%ymm7 \n\t"
				"vcvtph2ps %%xmm8, %%ymm8 \n\t"
				"vcvtph2ps %%xmm9, %%ymm9 \n\t"
				"vcvtph2ps %%xmm10, %%ymm10 \n\t"
				"vcvtph2ps %%xmm11, %%ymm11 \n\t"
				"vcvtph2ps %%xmm12, %%ymm12 \n\t"
				"vcvtph2ps %%xmm13, %%ymm13 \n\t"
				"vcvtph2ps %%xmm14, %%ymm14 \n\t"
				"vcvtph2ps %%xmm15, %%ymm15 \n\t"

				"vmulps %%ymm1, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm1, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm1, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm1, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm1, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm1, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm1, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm1, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm1, %%ymm15, %%ymm15 \n\t"
				"jmp LOOPSTART%= \n\t"

				"ZEROACC%=: \n\t"
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

				"LOOPSTART%=: \n\t"

				"movq %[A_ptr], %%rax \n\t"// lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx
				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

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
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

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

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"

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
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx
				"movq %%rcx, %%rax \n\t"// D pointer is in rax

				"vcvtps2ph $0x03, %%ymm4, %%xmm4 \n\t"
				"vcvtps2ph $0x03, %%ymm5, %%xmm5 \n\t"
				"vcvtps2ph $0x03, %%ymm6, %%xmm6 \n\t"
				"vcvtps2ph $0x03, %%ymm7, %%xmm7 \n\t"
				"vcvtps2ph $0x03, %%ymm8, %%xmm8 \n\t"
				"vcvtps2ph $0x03, %%ymm9, %%xmm9 \n\t"
				"vcvtps2ph $0x03, %%ymm10, %%xmm10 \n\t"
				"vcvtps2ph $0x03, %%ymm11, %%xmm11 \n\t"
				"vcvtps2ph $0x03, %%ymm12, %%xmm12 \n\t"
				"vcvtps2ph $0x03, %%ymm13, %%xmm13 \n\t"
				"vcvtps2ph $0x03, %%ymm14, %%xmm14 \n\t"
				"vcvtps2ph $0x03, %%ymm15, %%xmm15 \n\t"

				"vmovups %%xmm4, 0x00(%%rcx) \n\t"
				"vmovups %%xmm5, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%xmm6, 0x00(%%rcx) \n\t"
				"vmovups %%xmm7, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%xmm8, 0x00(%%rcx) \n\t"
				"vmovups %%xmm9, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%xmm10, 0x00(%%rcx) \n\t"
				"vmovups %%xmm11, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%xmm12, 0x00(%%rcx) \n\t"
				"vmovups %%xmm13, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%xmm14, 0x00(%%rcx) \n\t"
				"vmovups %%xmm15, 0x10(%%rcx) \n\t"

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
				[beta_ptr] "m"(_beta)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14");
	}
	void gemm_avx2_fma_6x8_fp16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 6);
		assert(B.columns() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), register_size<YMM>()));
		assert(beta_ptr != nullptr);

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float16 *C_ptr = C.data<float16>();
		float16 *D_ptr = D.data<float16>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float16);
		const uint64_t D_stride = D.stride() * sizeof(float16);
		float alpha = reinterpret_cast<const float*>(alpha_ptr)[0];
		float beta = reinterpret_cast<const float*>(beta_ptr)[0];
		if (beta != 0.0f)
			alpha /= beta;
		const float *_alpha = &alpha;

		asm volatile(
				"movq %[beta_ptr], %%rbx \n\t" // load address of beta
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"vucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je ZEROACC%= \n\t"
				// load and convert dst
				"movq %[C_stride], %%r12 \n\t"// C stride is r12
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"movups 0x00(%%rcx), %%xmm10 \n\t"
				"movups 0x10(%%rcx), %%xmm11 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm12 \n\t"
				"movups 0x10(%%rcx), %%xmm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm14 \n\t"
				"movups 0x10(%%rcx), %%xmm15 \n\t"

				"vcvtph2ps %%xmm10, %%ymm10 \n\t"
				"vcvtph2ps %%xmm11, %%ymm11 \n\t"
				"vcvtph2ps %%xmm12, %%ymm12 \n\t"
				"vcvtph2ps %%xmm13, %%ymm13 \n\t"
				"vcvtph2ps %%xmm14, %%ymm14 \n\t"
				"vcvtph2ps %%xmm15, %%ymm15 \n\t"

				"vmulps %%ymm1, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm1, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm1, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm1, %%ymm15, %%ymm15 \n\t"
				"jmp LOOPSTART%= \n\t"

				"ZEROACC%=: \n\t"
				// Set accumulators to zero.
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"LOOPSTART%=: \n\t"

				"movq %[A_ptr], %%rax \n\t"// lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx
				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x20(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x18(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x20(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x28(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x40(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x38(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x40(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0x60(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x48(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x50(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x54(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x58(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x5C(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				"add $0x60, %%rax \n\t"
				"add $0x80, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"
				"add $0x18, %%rax \n\t"
				"add $0x20, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx

				"vcvtps2ph $0x03, %%ymm10, %%xmm10 \n\t"
				"vcvtps2ph $0x03, %%ymm11, %%xmm11 \n\t"
				"vcvtps2ph $0x03, %%ymm12, %%xmm12 \n\t"
				"vcvtps2ph $0x03, %%ymm13, %%xmm13 \n\t"
				"vcvtps2ph $0x03, %%ymm14, %%xmm14 \n\t"
				"vcvtps2ph $0x03, %%ymm15, %%xmm15 \n\t"

				"movups %%xmm10, 0x00(%%rcx) \n\t"
				"movups %%xmm11, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm12, 0x00(%%rcx) \n\t"
				"movups %%xmm13, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm14, 0x00(%%rcx) \n\t"
				"movups %%xmm15, 0x10(%%rcx) \n\t"

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
				[alpha_ptr] "m"(_alpha),
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14");
	}
	void gemm_avx2_fma_24x4_fp16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.stride() == 24);
		assert(B.stride() == 4);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), register_size<YMM>()));

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float16 *C_ptr = C.data<float16>();
		float16 *D_ptr = D.data<float16>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float16);
		const uint64_t D_stride = D.stride() * sizeof(float16);

		asm volatile(
				"movq %[beta_ptr], %%rbx \n\t" // load address of beta
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"vucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je ZEROACC%= \n\t"
				// load destination pointer and stride
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx
				"movq %%rcx, %%rax \n\t"// C pointer is in rax
				"add %%r14, %%rax \n\t"// add stride
				"shlq $1, %%r14 \n\t"// multiply stride by 2

				// rows 0-7
				"vmovsd 0x0(%%rcx), %%xmm4 \n\t"
				"vmovsd 0x0(%%rax), %%xmm0 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd 0x0(%%rcx), %%xmm5 \n\t"
				"vmovsd 0x0(%%rax), %%xmm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd 0x0(%%rcx), %%xmm6 \n\t"
				"vmovsd 0x0(%%rax), %%xmm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd 0x0(%%rcx), %%xmm7 \n\t"
				"vmovsd 0x0(%%rax), %%xmm3 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovlhps %%xmm0, %%xmm4, %%xmm4 \n\t"
				"vmovlhps %%xmm1, %%xmm5, %%xmm5 \n\t"
				"vmovlhps %%xmm2, %%xmm6, %%xmm6 \n\t"
				"vmovlhps %%xmm3, %%xmm7, %%xmm7 \n\t"

				// rows 8-15
				"vmovsd 0x0(%%rcx), %%xmm8 \n\t"
				"vmovsd 0x0(%%rax), %%xmm0 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd 0x0(%%rcx), %%xmm9 \n\t"
				"vmovsd 0x0(%%rax), %%xmm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd 0x0(%%rcx), %%xmm10 \n\t"
				"vmovsd 0x0(%%rax), %%xmm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd 0x0(%%rcx), %%xmm11 \n\t"
				"vmovsd 0x0(%%rax), %%xmm3 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovlhps %%xmm0, %%xmm8, %%xmm8 \n\t"
				"vmovlhps %%xmm1, %%xmm9, %%xmm9 \n\t"
				"vmovlhps %%xmm2, %%xmm10, %%xmm10 \n\t"
				"vmovlhps %%xmm3, %%xmm11, %%xmm11 \n\t"

				// rows 16-23
				"vmovsd 0x0(%%rcx), %%xmm12 \n\t"
				"vmovsd 0x0(%%rax), %%xmm0 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd 0x0(%%rcx), %%xmm13 \n\t"
				"vmovsd 0x0(%%rax), %%xmm1 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd 0x0(%%rcx), %%xmm14 \n\t"
				"vmovsd 0x0(%%rax), %%xmm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd 0x0(%%rcx), %%xmm15 \n\t"
				"vmovsd 0x0(%%rax), %%xmm3 \n\t"
				"vmovlhps %%xmm0, %%xmm12, %%xmm12 \n\t"
				"vmovlhps %%xmm1, %%xmm13, %%xmm13 \n\t"
				"vmovlhps %%xmm2, %%xmm14, %%xmm14 \n\t"
				"vmovlhps %%xmm3, %%xmm15, %%xmm15 \n\t"

				"vcvtph2ps %%xmm4, %%ymm4 \n\t"
				"vcvtph2ps %%xmm5, %%ymm5 \n\t"
				"vcvtph2ps %%xmm6, %%ymm6 \n\t"
				"vcvtph2ps %%xmm7, %%ymm7 \n\t"
				"vcvtph2ps %%xmm8, %%ymm8 \n\t"
				"vcvtph2ps %%xmm9, %%ymm9 \n\t"
				"vcvtph2ps %%xmm10, %%ymm10 \n\t"
				"vcvtph2ps %%xmm11, %%ymm11 \n\t"
				"vcvtph2ps %%xmm12, %%ymm12 \n\t"
				"vcvtph2ps %%xmm13, %%ymm13 \n\t"
				"vcvtph2ps %%xmm14, %%ymm14 \n\t"
				"vcvtph2ps %%xmm15, %%ymm15 \n\t"

				"vpermpd $0xD8, %%ymm4, %%ymm4 \n\t"
				"vpermpd $0xD8, %%ymm5, %%ymm5 \n\t"
				"vpermpd $0xD8, %%ymm6, %%ymm6 \n\t"
				"vpermpd $0xD8, %%ymm7, %%ymm7 \n\t"
				"vpermpd $0xD8, %%ymm8, %%ymm8 \n\t"
				"vpermpd $0xD8, %%ymm9, %%ymm9 \n\t"
				"vpermpd $0xD8, %%ymm10, %%ymm10 \n\t"
				"vpermpd $0xD8, %%ymm11, %%ymm11 \n\t"
				"vpermpd $0xD8, %%ymm12, %%ymm12 \n\t"
				"vpermpd $0xD8, %%ymm13, %%ymm13 \n\t"
				"vpermpd $0xD8, %%ymm14, %%ymm14 \n\t"
				"vpermpd $0xD8, %%ymm15, %%ymm15 \n\t"

				"vpermilps $0xD8, %%ymm4, %%ymm4 \n\t"
				"vpermilps $0xD8, %%ymm5, %%ymm5 \n\t"
				"vpermilps $0xD8, %%ymm6, %%ymm6 \n\t"
				"vpermilps $0xD8, %%ymm7, %%ymm7 \n\t"
				"vpermilps $0xD8, %%ymm8, %%ymm8 \n\t"
				"vpermilps $0xD8, %%ymm9, %%ymm9 \n\t"
				"vpermilps $0xD8, %%ymm10, %%ymm10 \n\t"
				"vpermilps $0xD8, %%ymm11, %%ymm11 \n\t"
				"vpermilps $0xD8, %%ymm12, %%ymm12 \n\t"
				"vpermilps $0xD8, %%ymm13, %%ymm13 \n\t"
				"vpermilps $0xD8, %%ymm14, %%ymm14 \n\t"
				"vpermilps $0xD8, %%ymm15, %%ymm15 \n\t"

				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"// load beta again
				"vmulps %%ymm1, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm1, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm1, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm1, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm1, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm1, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm1, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm1, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm1, %%ymm15, %%ymm15 \n\t"
				"jmp LOOPSTART%= \n\t"

				"ZEROACC%=: \n\t"
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

				"LOOPSTART%=: \n\t"
				"movq %[A_ptr], %%rax \n\t"// lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx
				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps (4*0*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"// b0 b0 b1 b1 b2 b2 b3 b3

				"vbroadcastsd (0*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (0*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (0*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (0*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (0*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (0*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				// iteration 1
				"vmovaps (4*1*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"

				"vbroadcastsd (1*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (1*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (1*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (1*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (1*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (1*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (1*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (1*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (1*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (1*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (1*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (1*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				// iteration 2
				"vmovaps (4*2*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"

				"vbroadcastsd (2*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (2*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (2*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (2*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (2*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (2*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (2*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (2*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (2*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (2*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (2*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (2*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				// iteration 3
				"vmovaps (4*3*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"

				"vbroadcastsd (3*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (3*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (3*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (3*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (3*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (3*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (3*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (3*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (3*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (3*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (3*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (3*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				"add $(4*24*4), %%rax \n\t"
				"add $(4*4*4), %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps (4*0*4)(%%rbx), %%xmm0 \n\t"// b0 b1 b2 b3
				"vunpcklps %%xmm0, %%xmm0, %%xmm1 \n\t"// b0 b0 b1 b2
				"vunpckhps %%xmm0, %%xmm0, %%xmm2 \n\t"// b2 b2 b3 b3
				"vinsertf128 $0x1, %%xmm2, %%ymm1, %%ymm3 \n\t"

				"vbroadcastsd (0*96+4*0)(%%rax), %%ymm0 \n\t"// a0 a1 a0 a1 a0 a1 a0 a1
				"vbroadcastsd (0*96+4*2)(%%rax), %%ymm1 \n\t"// a2 a3 a2 a3 a2 a3 a2 a3
				"vbroadcastsd (0*96+4*4)(%%rax), %%ymm2 \n\t"// a4 a5 a4 a5 a4 a5 a4 a5
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm4 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm6 \n\t"

				"vbroadcastsd (0*96+4*6)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*8)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*10)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm9 \n\t"

				"vbroadcastsd (0*96+4*12)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*14)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*16)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm12 \n\t"

				"vbroadcastsd (0*96+4*18)(%%rax), %%ymm0 \n\t"
				"vbroadcastsd (0*96+4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastsd (0*96+4*22)(%%rax), %%ymm2 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm3, %%ymm15 \n\t"

				"add $(4*24*1), %%rax \n\t"
				"add $(4*1*4), %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"

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

				// shuffle back into correct layout
				"vpermilps $0xD8, %%ymm4, %%ymm4 \n\t"
				"vpermilps $0xD8, %%ymm5, %%ymm5 \n\t"
				"vpermilps $0xD8, %%ymm6, %%ymm6 \n\t"
				"vpermilps $0xD8, %%ymm7, %%ymm7 \n\t"
				"vpermilps $0xD8, %%ymm8, %%ymm8 \n\t"
				"vpermilps $0xD8, %%ymm9, %%ymm9 \n\t"
				"vpermilps $0xD8, %%ymm10, %%ymm10 \n\t"
				"vpermilps $0xD8, %%ymm11, %%ymm11 \n\t"
				"vpermilps $0xD8, %%ymm12, %%ymm12 \n\t"
				"vpermilps $0xD8, %%ymm13, %%ymm13 \n\t"
				"vpermilps $0xD8, %%ymm14, %%ymm14 \n\t"
				"vpermilps $0xD8, %%ymm15, %%ymm15 \n\t"

				"vpermpd $0xD8, %%ymm4, %%ymm4 \n\t"
				"vpermpd $0xD8, %%ymm5, %%ymm5 \n\t"
				"vpermpd $0xD8, %%ymm6, %%ymm6 \n\t"
				"vpermpd $0xD8, %%ymm7, %%ymm7 \n\t"
				"vpermpd $0xD8, %%ymm8, %%ymm8 \n\t"
				"vpermpd $0xD8, %%ymm9, %%ymm9 \n\t"
				"vpermpd $0xD8, %%ymm10, %%ymm10 \n\t"
				"vpermpd $0xD8, %%ymm11, %%ymm11 \n\t"
				"vpermpd $0xD8, %%ymm12, %%ymm12 \n\t"
				"vpermpd $0xD8, %%ymm13, %%ymm13 \n\t"
				"vpermpd $0xD8, %%ymm14, %%ymm14 \n\t"
				"vpermpd $0xD8, %%ymm15, %%ymm15 \n\t"

				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx
				"movq %%rcx, %%rax \n\t"// C pointer is in rax
				"add %%r14, %%rax \n\t"// add stride
				"shlq $1, %%r14 \n\t"// multiply stride by 2

				// rows 0-7
				"vcvtps2ph $0x03, %%ymm4, %%xmm4 \n\t"
				"vcvtps2ph $0x03, %%ymm5, %%xmm5 \n\t"
				"vcvtps2ph $0x03, %%ymm6, %%xmm6 \n\t"
				"vcvtps2ph $0x03, %%ymm7, %%xmm7 \n\t"
				"vcvtps2ph $0x03, %%ymm8, %%xmm8 \n\t"
				"vcvtps2ph $0x03, %%ymm9, %%xmm9 \n\t"
				"vcvtps2ph $0x03, %%ymm10, %%xmm10 \n\t"
				"vcvtps2ph $0x03, %%ymm11, %%xmm11 \n\t"
				"vcvtps2ph $0x03, %%ymm12, %%xmm12 \n\t"
				"vcvtps2ph $0x03, %%ymm13, %%xmm13 \n\t"
				"vcvtps2ph $0x03, %%ymm14, %%xmm14 \n\t"
				"vcvtps2ph $0x03, %%ymm15, %%xmm15 \n\t"

				"vmovhlps %%xmm4, %%xmm4, %%xmm0 \n\t"
				"vmovhlps %%xmm5, %%xmm5, %%xmm1 \n\t"
				"vmovhlps %%xmm6, %%xmm6, %%xmm2 \n\t"
				"vmovhlps %%xmm7, %%xmm7, %%xmm3 \n\t"
				"vmovsd %%xmm4, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm0, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd %%xmm5, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm1, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd %%xmm6, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm2, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd %%xmm7, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm3, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride

				// rows 8-15
				"vmovhlps %%xmm8, %%xmm8, %%xmm0 \n\t"
				"vmovhlps %%xmm9, %%xmm9, %%xmm1 \n\t"
				"vmovhlps %%xmm10, %%xmm10, %%xmm2 \n\t"
				"vmovhlps %%xmm11, %%xmm11, %%xmm3 \n\t"
				"vmovsd %%xmm8, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm0, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd %%xmm9, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm1, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd %%xmm10, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm2, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd %%xmm11, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm3, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride

				// rows 16-23
				"vmovhlps %%xmm12, %%xmm12, %%xmm0 \n\t"
				"vmovhlps %%xmm13, %%xmm13, %%xmm1 \n\t"
				"vmovhlps %%xmm14, %%xmm14, %%xmm2 \n\t"
				"vmovhlps %%xmm15, %%xmm15, %%xmm3 \n\t"
				"vmovsd %%xmm12, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm0, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd %%xmm13, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm1, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd %%xmm14, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm2, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride
				"vmovsd %%xmm15, 0x0(%%rcx) \n\t"
				"vmovsd %%xmm3, 0x0(%%rax) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"add %%r14, %%rax \n\t"// add stride

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

	void pack_avx2_fma_4xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 4);
		assert(ml::cpu::is_aligned(dst.data(), register_size<XMM>()));

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

					"vmovaps %%xmm0, 0x00(%%rbx) \n\t"
					"vmovaps %%xmm1, 0x10(%%rbx) \n\t"
					"vmovaps %%xmm2, 0x20(%%rbx) \n\t"
					"vmovaps %%xmm3, 0x30(%%rbx) \n\t"
					"vmovaps %%xmm4, 0x40(%%rbx) \n\t"
					"vmovaps %%xmm5, 0x50(%%rbx) \n\t"
					"vmovaps %%xmm6, 0x60(%%rbx) \n\t"
					"vmovaps %%xmm7, 0x70(%%rbx) \n\t"

					"add $(4*8*4), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%xmm0 \n\t"
					"vmovaps %%xmm0, 0x00(%%rbx) \n\t"

					"add %%r12, %%rax \n\t"// add stride to src pointer
					"add $(4*1*4), %%rbx \n\t"// add stride to dst pointer

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
					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

					// load 4x8 fp32
					"vmovups 0x0(%%r13), %%ymm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm3 \n\t"

					"vunpcklps %%ymm1, %%ymm0, %%ymm6 \n\t"// a0 b0 a1 b1 a4 b4 a5 b5
					"vunpckhps %%ymm1, %%ymm0, %%ymm7 \n\t"// a2 b2 a3 b3 a6 b6 a7 b7
					"vunpcklps %%ymm3, %%ymm2, %%ymm8 \n\t"// c0 d0 c1 d1 c4 d4 c5 d5
					"vunpckhps %%ymm3, %%ymm2, %%ymm9 \n\t"// c2 d2 c3 d3 c6 d6 c7 d7
					// second shuffle
					"vunpcklpd %%ymm8, %%ymm6, %%ymm12 \n\t"// a0 b0 c0 d0 a4 b4 c4 d4
					"vunpckhpd %%ymm8, %%ymm6, %%ymm13 \n\t"// a1 b1 c1 d1 a5 b5 c5 d5
					"vunpcklpd %%ymm9, %%ymm7, %%ymm14 \n\t"// a2 b2 c2 d2 a6 b6 c6 d6
					"vunpckhpd %%ymm9, %%ymm7, %%ymm15 \n\t"// a3 b3 c3 d3 a7 b7 c7 d7

					"vextractf128 $0x1, %%ymm12, %%xmm0 \n\t"// a4 b4 c4 d4
					"vextractf128 $0x1, %%ymm13, %%xmm1 \n\t"// a5 b5 c5 d5
					"vextractf128 $0x1, %%ymm14, %%xmm2 \n\t"// a6 b6 c6 d6
					"vextractf128 $0x1, %%ymm15, %%xmm3 \n\t"// a7 b7 c7 d7

					"vmovaps %%xmm12, (4*0*4)(%%rbx) \n\t"// a0 b0 c0 d0
					"vmovaps %%xmm13, (4*1*4)(%%rbx) \n\t"// a1 b1 c1 d1
					"vmovaps %%xmm14, (4*2*4)(%%rbx) \n\t"// a2 b2 c2 d2
					"vmovaps %%xmm15, (4*3*4)(%%rbx) \n\t"// a3 b3 c3 d3
					"vmovaps %%xmm0, (4*4*4)(%%rbx) \n\t"// a4 b4 c4 d4
					"vmovaps %%xmm1, (4*5*4)(%%rbx) \n\t"// a5 b5 c5 d5
					"vmovaps %%xmm2, (4*6*4)(%%rbx) \n\t"// a6 b6 c6 d6
					"vmovaps %%xmm3, (4*7*4)(%%rbx) \n\t"// a7 b7 c7 d7

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

					"vmovss 0x0(%%r13), %%xmm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm3 \n\t"

					"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
					"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
					"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
					"vmovss %%xmm3, (4*3)(%%rbx) \n\t"

					"add $(4*1), %%rax \n\t"// add stride to src pointer
					"add $(4*4*1), %%rbx \n\t"// add stride to dst pointer

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
					"%r12", "%r13", "%r14");
		}
	}
	void pack_avx2_fma_6xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 6);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

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
					"vmovups 0x00(%%rax), %%xmm0 \n\t"
					"vmovsd  0x10(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm2 \n\t"
					"vmovsd  0x10(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm4 \n\t"
					"vmovsd  0x10(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm6 \n\t"
					"vmovsd  0x10(%%rax), %%xmm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm8 \n\t"
					"vmovsd  0x10(%%rax), %%xmm9 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm10 \n\t"
					"vmovsd  0x10(%%rax), %%xmm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm12 \n\t"
					"vmovsd  0x10(%%rax), %%xmm13 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm14 \n\t"
					"vmovsd  0x10(%%rax), %%xmm15 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vmovups %%xmm0, (4*(0*6+0))(%%rbx) \n\t"
					"vmovsd  %%xmm1, (4*(0*6+4))(%%rbx) \n\t"
					"vmovups %%xmm2, (4*(1*6+0))(%%rbx) \n\t"
					"vmovsd  %%xmm3, (4*(1*6+4))(%%rbx) \n\t"
					"vmovups %%xmm4, (4*(2*6+0))(%%rbx) \n\t"
					"vmovsd  %%xmm5, (4*(2*6+4))(%%rbx) \n\t"
					"vmovups %%xmm6, (4*(3*6+0))(%%rbx) \n\t"
					"vmovsd  %%xmm7, (4*(3*6+4))(%%rbx) \n\t"
					"vmovups %%xmm8, (4*(4*6+0))(%%rbx) \n\t"
					"vmovsd  %%xmm9, (4*(4*6+4))(%%rbx) \n\t"
					"vmovups %%xmm10, (4*(5*6+0))(%%rbx) \n\t"
					"vmovsd  %%xmm11, (4*(5*6+4))(%%rbx) \n\t"
					"vmovups %%xmm12, (4*(6*6+0))(%%rbx) \n\t"
					"vmovsd  %%xmm13, (4*(6*6+4))(%%rbx) \n\t"
					"vmovups %%xmm14, (4*(7*6+0))(%%rbx) \n\t"
					"vmovsd  %%xmm15, (4*(7*6+4))(%%rbx) \n\t"

					"add $0xC0, %%rbx \n\t"// add stride to dst pointer (8 * 4 * 6)

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"movups 0x00(%%rax), %%xmm0 \n\t"
					"movsd  0x10(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"movups %%xmm0, (4*(0*6+0))(%%rbx) \n\t"
					"movsd  %%xmm1, (4*(0*6+4))(%%rbx) \n\t"
					"add $0x18, %%rbx \n\t"// add stride to dst pointer (1 * 4 * 6)

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
					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13
					// first 8x8 tile
					"vmovups 0x0(%%r13), %%ymm0 \n\t"// a0 a1 a2 a3 a4 a5 a6 a7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm1 \n\t"// b0 b1 b2 b3 b4 b5 b6 b7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm2 \n\t"// c0 c1 c2 c3 c4 c5 c6 c7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm3 \n\t"// d0 d1 d2 d3 d4 d5 d6 d7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm4 \n\t"// e0 e1 e2 e3 e4 e5 e6 e7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm5 \n\t"// f0 f1 f2 f3 f4 f5 f6 f7

					// transpose 6x8
					// first shuffle
					"vunpcklps %%ymm1, %%ymm0, %%ymm6 \n\t"// a0 b0 a1 b1 a4 b4 a5 b5
					"vunpckhps %%ymm1, %%ymm0, %%ymm7 \n\t"// a2 b2 a3 b3 a6 b6 a7 b7
					"vunpcklps %%ymm3, %%ymm2, %%ymm8 \n\t"// c0 d0 c1 d1 c4 d4 c5 d5
					"vunpckhps %%ymm3, %%ymm2, %%ymm9 \n\t"// c2 d2 c3 d3 c6 d6 c7 d7
					"vunpcklps %%ymm5, %%ymm4, %%ymm10 \n\t"// e0 f0 e1 f1 e4 f4 e5 f5
					"vunpckhps %%ymm5, %%ymm4, %%ymm11 \n\t"// e2 f2 e3 f3 e6 f6 e7 f7
					// second shuffle
					"vunpcklpd %%ymm8, %%ymm6, %%ymm12 \n\t"// a0 b0 c0 d0 a4 b4 c4 d4
					"vunpckhpd %%ymm8, %%ymm6, %%ymm13 \n\t"// a1 b1 c1 d1 a5 b5 c5 d5
					"vunpcklpd %%ymm9, %%ymm7, %%ymm14 \n\t"// a2 b2 c2 d2 a6 b6 c6 d6
					"vunpckhpd %%ymm9, %%ymm7, %%ymm15 \n\t"// a3 b3 c3 d3 a7 b7 c7 d7

					"vextractf128 $0x1, %%ymm12, %%xmm0 \n\t"// a4 b4 c4 d4
					"vextractf128 $0x1, %%ymm13, %%xmm1 \n\t"// a5 b5 c5 d5
					"vextractf128 $0x1, %%ymm14, %%xmm2 \n\t"// a6 b6 c6 d6
					"vextractf128 $0x1, %%ymm15, %%xmm3 \n\t"// a7 b7 c7 d7
					"vextractf128 $0x1, %%ymm10, %%xmm4 \n\t"// e4 f4 e5 f5
					"vextractf128 $0x1, %%ymm11, %%xmm5 \n\t"// e6 f6 e7 f7

					"vmovups %%xmm12, (4*(0*6+0))(%%rbx) \n\t"// a0 b0 c0 d0
					"vmovlpd %%xmm10, (4*(0*6+4))(%%rbx) \n\t"// e0 f0
					"vmovups %%xmm13, (4*(1*6+0))(%%rbx) \n\t"// a1 b1 c1 d1
					"vmovhpd %%xmm10, (4*(1*6+4))(%%rbx) \n\t"// e1 f1
					"vmovups %%xmm14, (4*(2*6+0))(%%rbx) \n\t"// a2 b2 c2 d2
					"vmovlpd %%xmm11, (4*(2*6+4))(%%rbx) \n\t"// e2 f2
					"vmovups %%xmm15, (4*(3*6+0))(%%rbx) \n\t"// a3 b3 c3 d3
					"vmovhpd %%xmm11, (4*(3*6+4))(%%rbx) \n\t"// e3 f3

					"vmovups %%xmm0, (4*(4*6+0))(%%rbx) \n\t"// a4 b4 c4 d4
					"vmovlpd %%xmm4, (4*(4*6+4))(%%rbx) \n\t"// e4 f4
					"vmovups %%xmm1, (4*(5*6+0))(%%rbx) \n\t"// a5 b5 c5 d5
					"vmovhpd %%xmm4, (4*(5*6+4))(%%rbx) \n\t"// e5 f5
					"vmovups %%xmm2, (4*(6*6+0))(%%rbx) \n\t"// a6 b6 c6 d6
					"vmovlpd %%xmm5, (4*(6*6+4))(%%rbx) \n\t"// e6 f6
					"vmovups %%xmm3, (4*(7*6+0))(%%rbx) \n\t"// a7 b7 c7 d7
					"vmovhpd %%xmm5, (4*(7*6+4))(%%rbx) \n\t"// e7 f7

//					"vmovups 0x0(%%r13), %%ymm4 \n\t"// a0 a1 a2 a3 a4 a5 a6 a7
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%ymm6 \n\t"// b0 b1 b2 b3 b4 b5 b6 b7
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%ymm8 \n\t"// c0 c1 c2 c3 c4 c5 c6 c7
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%ymm10 \n\t"// d0 d1 d2 d3 d4 d5 d6 d7
//
//					"vunpcklps %%ymm6, %%ymm4, %%ymm0 \n\t"// a0 b0 a1 b1 a4 b4 a5 b5
//					"vunpckhps %%ymm10, %%ymm8, %%ymm1 \n\t"// a2 b2 a3 b3 a6 b6 a7 b7
//					"vshufps  $0x4E, %%ymm1, %%ymm0, %%ymm2 \n\t"
//					"vblendps $0xCC, %%ymm2, %%ymm0, %%ymm0 \n\t"
//					"vblendps $0x33, %%ymm2, %%ymm1, %%ymm1 \n\t"
//
//					"vextractf128 $0x1, %%ymm0, %%xmm2 \n\t"
//					"vmovups %%xmm0, (0*24)(%%rbx) \n\t"// store ( gamma00..gamma30 )
//					"vmovups %%xmm2, (4*24)(%%rbx) \n\t"// store ( gamma04..gamma34 )
//
//					"vextractf128 $0x1, %%ymm1, %%xmm2 \n\t"
//					"vmovups %%xmm1, (1*24)(%%rbx) \n\t"// store ( gamma01..gamma31 )
//					"vmovups %%xmm2, (5*24)(%%rbx) \n\t"// store ( gamma05..gamma35 )
//
//					"vunpckhps %%ymm6, %%ymm4, %%ymm0 \n\t"
//					"vunpckhps %%ymm10, %%ymm8, %%ymm1 \n\t"
//					"vshufps  $0x4E, %%ymm1, %%ymm0, %%ymm2 \n\t"
//					"vblendps $0xCC, %%ymm2, %%ymm0, %%ymm0 \n\t"
//					"vblendps $0x33, %%ymm2, %%ymm1, %%ymm1 \n\t"
//
//					"vextractf128 $0x1, %%ymm0, %%xmm2 \n\t"
//					"vmovups %%xmm0, (2*24)(%%rbx) \n\t"// store ( gamma02..gamma32 )
//					"vmovups %%xmm2, (6*24)(%%rbx) \n\t"// store ( gamma06..gamma36 )
//
//					"vextractf128 $0x1, %%ymm1, %%xmm2 \n\t"
//					"vmovups %%xmm1, (3*24)(%%rbx) \n\t"// store ( gamma03..gamma33 )
//					"vmovups %%xmm2, (7*24)(%%rbx) \n\t"// store ( gamma07..gamma37 )
//
//					"vmovups 0x0(%%r13), %%ymm12 \n\t"// a0 a1 a2 a3 a4 a5 a6 a7
//					"add %%r12, %%r13 \n\t"// add stride to src pointer
//					"vmovups 0x0(%%r13), %%ymm14 \n\t"// b0 b1 b2 b3 b4 b5 b6 b7
//
//					"vunpcklps %%ymm14, %%ymm12, %%ymm0 \n\t"
//					"vextractf128 $0x1, %%ymm0, %%xmm2 \n\t"
//					"vmovlpd %%xmm0, (0*24+16)(%%rbx) \n\t"
//					"vmovhpd %%xmm0, (1*24+16)(%%rbx) \n\t"
//					"vmovlpd %%xmm2, (4*24+16)(%%rbx) \n\t"
//					"vmovhpd %%xmm2, (5*24+16)(%%rbx) \n\t"
//
//					"vunpckhps %%ymm14, %%ymm12, %%ymm0 \n\t"
//					"vextractf128 $0x1, %%ymm0, %%xmm2 \n\t"
//					"vmovlpd %%xmm0, (2*24+16)(%%rbx) \n\t"
//					"vmovhpd %%xmm0, (3*24+16)(%%rbx) \n\t"
//					"vmovlpd %%xmm2, (6*24+16)(%%rbx) \n\t"
//					"vmovhpd %%xmm2, (7*24+16)(%%rbx) \n\t"

					"add $(4*8), %%rax \n\t"// add stride to src pointer
					"add $(4*6*8), %%rbx \n\t"// add stride to dst pointer

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

					"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
					"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
					"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
					"vmovss %%xmm3, (4*3)(%%rbx) \n\t"
					"vmovss %%xmm4, (4*4)(%%rbx) \n\t"
					"vmovss %%xmm5, (4*5)(%%rbx) \n\t"

					"add $(4*1), %%rax \n\t"
					"add $(4*6*1), %%rbx \n\t"// add stride to dst pointer

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
	void pack_avx2_fma_16xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 16);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

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
					"vmovups 0x20(%%rax), %%ymm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm2 \n\t"
					"vmovups 0x20(%%rax), %%ymm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm4 \n\t"
					"vmovups 0x20(%%rax), %%ymm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm6 \n\t"
					"vmovups 0x20(%%rax), %%ymm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm8 \n\t"
					"vmovups 0x20(%%rax), %%ymm9 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm10 \n\t"
					"vmovups 0x20(%%rax), %%ymm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm12 \n\t"
					"vmovups 0x20(%%rax), %%ymm13 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm14 \n\t"
					"vmovups 0x20(%%rax), %%ymm15 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vmovaps %%ymm0, 0x000(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x020(%%rbx) \n\t"
					"vmovaps %%ymm2, 0x040(%%rbx) \n\t"
					"vmovaps %%ymm3, 0x060(%%rbx) \n\t"
					"vmovaps %%ymm4, 0x080(%%rbx) \n\t"
					"vmovaps %%ymm5, 0x0A0(%%rbx) \n\t"
					"vmovaps %%ymm6, 0x0C0(%%rbx) \n\t"
					"vmovaps %%ymm7, 0x0E0(%%rbx) \n\t"
					"vmovaps %%ymm8, 0x100(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x120(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x140(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x160(%%rbx) \n\t"
					"vmovaps %%ymm12, 0x180(%%rbx) \n\t"
					"vmovaps %%ymm13, 0x1A0(%%rbx) \n\t"
					"vmovaps %%ymm14, 0x1C0(%%rbx) \n\t"
					"vmovaps %%ymm15, 0x1E0(%%rbx) \n\t"

					"add $(4*8*16), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"vmovups 0x20(%%rax), %%ymm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovaps %%ymm0, 0x00(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x20(%%rbx) \n\t"
					"add $(4*1*16), %%rbx \n\t"// add stride to dst pointer

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

					"vmovaps %%ymm8, 0x000(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x040(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x080(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x0C0(%%rbx) \n\t"
					"vmovaps %%ymm12, 0x100(%%rbx) \n\t"
					"vmovaps %%ymm13, 0x140(%%rbx) \n\t"
					"vmovaps %%ymm14, 0x180(%%rbx) \n\t"
					"vmovaps %%ymm15, 0x1C0(%%rbx) \n\t"

					// second 8x8 tile
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

					"vmovaps %%ymm8, 0x020(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x060(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x0A0(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x0E0(%%rbx) \n\t"
					"vmovaps %%ymm12, 0x120(%%rbx) \n\t"
					"vmovaps %%ymm13, 0x160(%%rbx) \n\t"
					"vmovaps %%ymm14, 0x1A0(%%rbx) \n\t"
					"vmovaps %%ymm15, 0x1E0(%%rbx) \n\t"

					"add $(4*8), %%rax \n\t"// add stride to src pointer
					"add $(4*16*8), %%rbx \n\t"// add stride to dst pointer

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
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm12 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm13 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm14 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm15 \n\t"

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
					"vmovss %%xmm12, (4*12)(%%rbx) \n\t"
					"vmovss %%xmm13, (4*13)(%%rbx) \n\t"
					"vmovss %%xmm14, (4*14)(%%rbx) \n\t"
					"vmovss %%xmm15, (4*15)(%%rbx) \n\t"

					"add $(4*1), %%rax \n\t"// add stride to src pointer
					"add $(4*16*1), %%rbx \n\t"// add stride to dst pointer

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
	void pack_avx2_fma_24xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 24);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

		const uint64_t src_stride = src.stride() * sizeof(float);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
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

					"UNROLLED8%=: \n\t"
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"vmovups 0x20(%%rax), %%ymm1 \n\t"
					"vmovups 0x40(%%rax), %%ymm2 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm3 \n\t"
					"vmovups 0x20(%%rax), %%ymm4 \n\t"
					"vmovups 0x40(%%rax), %%ymm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm6 \n\t"
					"vmovups 0x20(%%rax), %%ymm7 \n\t"
					"vmovups 0x40(%%rax), %%ymm8 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm9 \n\t"
					"vmovups 0x20(%%rax), %%ymm10 \n\t"
					"vmovups 0x40(%%rax), %%ymm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vmovaps %%ymm0, 0x000(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x020(%%rbx) \n\t"
					"vmovaps %%ymm2, 0x040(%%rbx) \n\t"
					"vmovaps %%ymm3, 0x060(%%rbx) \n\t"
					"vmovaps %%ymm4, 0x080(%%rbx) \n\t"
					"vmovaps %%ymm5, 0x0A0(%%rbx) \n\t"
					"vmovaps %%ymm6, 0x0C0(%%rbx) \n\t"
					"vmovaps %%ymm7, 0x0E0(%%rbx) \n\t"
					"vmovaps %%ymm8, 0x100(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x120(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x140(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x160(%%rbx) \n\t"

					"add $(4*4*24), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"vmovups 0x20(%%rax), %%ymm1 \n\t"
					"vmovups 0x40(%%rax), %%ymm2 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovaps %%ymm0, 0x00(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x20(%%rbx) \n\t"
					"vmovaps %%ymm2, 0x40(%%rbx) \n\t"
					"add $(4*1*24), %%rbx \n\t"// add stride to dst pointer

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
					// store
					"vmovaps %%ymm8, (0*4*8+4*0*24)(%%rbx) \n\t"
					"vmovaps %%ymm9, (0*4*8+4*1*24)(%%rbx) \n\t"
					"vmovaps %%ymm10, (0*4*8+4*2*24)(%%rbx) \n\t"
					"vmovaps %%ymm11, (0*4*8+4*3*24)(%%rbx) \n\t"
					"vmovaps %%ymm12, (0*4*8+4*4*24)(%%rbx) \n\t"
					"vmovaps %%ymm13, (0*4*8+4*5*24)(%%rbx) \n\t"
					"vmovaps %%ymm14, (0*4*8+4*6*24)(%%rbx) \n\t"
					"vmovaps %%ymm15, (0*4*8+4*7*24)(%%rbx) \n\t"

					// second 8x8 tile
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
					// store
					"vmovaps %%ymm8, (1*4*8+4*0*24)(%%rbx) \n\t"
					"vmovaps %%ymm9, (1*4*8+4*1*24)(%%rbx) \n\t"
					"vmovaps %%ymm10, (1*4*8+4*2*24)(%%rbx) \n\t"
					"vmovaps %%ymm11, (1*4*8+4*3*24)(%%rbx) \n\t"
					"vmovaps %%ymm12, (1*4*8+4*4*24)(%%rbx) \n\t"
					"vmovaps %%ymm13, (1*4*8+4*5*24)(%%rbx) \n\t"
					"vmovaps %%ymm14, (1*4*8+4*6*24)(%%rbx) \n\t"
					"vmovaps %%ymm15, (1*4*8+4*7*24)(%%rbx) \n\t"

					// third 8x8 tile
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
					// store
					"vmovaps %%ymm8, (2*4*8+4*0*24)(%%rbx) \n\t"
					"vmovaps %%ymm9, (2*4*8+4*1*24)(%%rbx) \n\t"
					"vmovaps %%ymm10, (2*4*8+4*2*24)(%%rbx) \n\t"
					"vmovaps %%ymm11, (2*4*8+4*3*24)(%%rbx) \n\t"
					"vmovaps %%ymm12, (2*4*8+4*4*24)(%%rbx) \n\t"
					"vmovaps %%ymm13, (2*4*8+4*5*24)(%%rbx) \n\t"
					"vmovaps %%ymm14, (2*4*8+4*6*24)(%%rbx) \n\t"
					"vmovaps %%ymm15, (2*4*8+4*7*24)(%%rbx) \n\t"

					"add $(4*8), %%rax \n\t"// add stride to src pointer
					"add $(4*24*8), %%rbx \n\t"// add stride to dst pointer

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
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm12 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm13 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm14 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovss 0x0(%%r13), %%xmm15 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer

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
					"vmovss %%xmm12, (4*12)(%%rbx) \n\t"
					"vmovss %%xmm13, (4*13)(%%rbx) \n\t"
					"vmovss %%xmm14, (4*14)(%%rbx) \n\t"
					"vmovss %%xmm15, (4*15)(%%rbx) \n\t"

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

					"vmovss %%xmm0, (4*16)(%%rbx) \n\t"
					"vmovss %%xmm1, (4*17)(%%rbx) \n\t"
					"vmovss %%xmm2, (4*18)(%%rbx) \n\t"
					"vmovss %%xmm3, (4*19)(%%rbx) \n\t"
					"vmovss %%xmm4, (4*20)(%%rbx) \n\t"
					"vmovss %%xmm5, (4*21)(%%rbx) \n\t"
					"vmovss %%xmm6, (4*22)(%%rbx) \n\t"
					"vmovss %%xmm7, (4*23)(%%rbx) \n\t"

					"add $(4*1), %%rax \n\t"// add stride to src pointer
					"add $(4*24*1), %%rbx \n\t"// add stride to dst pointer

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

	void pack_avx2_fma_4xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 4);
		assert(ml::cpu::is_aligned(dst.data(), register_size<XMM>()));

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
					"vmovsd 0x00(%%rax), %%xmm0 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovsd 0x00(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovsd 0x00(%%rax), %%xmm2 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovsd 0x00(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovsd 0x00(%%rax), %%xmm4 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovsd 0x00(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovsd 0x00(%%rax), %%xmm6 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovsd 0x00(%%rax), %%xmm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vcvtph2ps %%xmm0, %%xmm0 \n\t"
					"vcvtph2ps %%xmm1, %%xmm1 \n\t"
					"vcvtph2ps %%xmm2, %%xmm2 \n\t"
					"vcvtph2ps %%xmm3, %%xmm3 \n\t"
					"vcvtph2ps %%xmm4, %%xmm4 \n\t"
					"vcvtph2ps %%xmm5, %%xmm5 \n\t"
					"vcvtph2ps %%xmm6, %%xmm6 \n\t"
					"vcvtph2ps %%xmm7, %%xmm7 \n\t"

					"vmovaps %%xmm0, 0x00(%%rbx) \n\t"
					"vmovaps %%xmm1, 0x10(%%rbx) \n\t"
					"vmovaps %%xmm2, 0x20(%%rbx) \n\t"
					"vmovaps %%xmm3, 0x30(%%rbx) \n\t"
					"vmovaps %%xmm4, 0x40(%%rbx) \n\t"
					"vmovaps %%xmm5, 0x50(%%rbx) \n\t"
					"vmovaps %%xmm6, 0x60(%%rbx) \n\t"
					"vmovaps %%xmm7, 0x70(%%rbx) \n\t"

					"add $(4*8*4), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovsd 0x00(%%rax), %%xmm0 \n\t"
					"vcvtph2ps %%xmm0, %%xmm0 \n\t"
					"vmovaps %%xmm0, 0x00(%%rbx) \n\t"

					"add %%r12, %%rax \n\t"// add stride to src pointer
					"add $(4*1*4), %%rbx \n\t"// add stride to dst pointer

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

					"UNROLLED4%=: \n\t"
					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13

					// load 4x8 fp16
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

					"vunpcklps %%ymm1, %%ymm0, %%ymm6 \n\t"// a0 b0 a1 b1 a4 b4 a5 b5
					"vunpckhps %%ymm1, %%ymm0, %%ymm7 \n\t"// a2 b2 a3 b3 a6 b6 a7 b7
					"vunpcklps %%ymm3, %%ymm2, %%ymm8 \n\t"// c0 d0 c1 d1 c4 d4 c5 d5
					"vunpckhps %%ymm3, %%ymm2, %%ymm9 \n\t"// c2 d2 c3 d3 c6 d6 c7 d7
					// second shuffle
					"vunpcklpd %%ymm8, %%ymm6, %%ymm12 \n\t"// a0 b0 c0 d0 a4 b4 c4 d4
					"vunpckhpd %%ymm8, %%ymm6, %%ymm13 \n\t"// a1 b1 c1 d1 a5 b5 c5 d5
					"vunpcklpd %%ymm9, %%ymm7, %%ymm14 \n\t"// a2 b2 c2 d2 a6 b6 c6 d6
					"vunpckhpd %%ymm9, %%ymm7, %%ymm15 \n\t"// a3 b3 c3 d3 a7 b7 c7 d7

					"vextractf128 $0x1, %%ymm12, %%xmm0 \n\t"// a4 b4 c4 d4
					"vextractf128 $0x1, %%ymm13, %%xmm1 \n\t"// a5 b5 c5 d5
					"vextractf128 $0x1, %%ymm14, %%xmm2 \n\t"// a6 b6 c6 d6
					"vextractf128 $0x1, %%ymm15, %%xmm3 \n\t"// a7 b7 c7 d7

					"vmovaps %%xmm12, (4*0*4)(%%rbx) \n\t"// a0 b0 c0 d0
					"vmovaps %%xmm13, (4*1*4)(%%rbx) \n\t"// a1 b1 c1 d1
					"vmovaps %%xmm14, (4*2*4)(%%rbx) \n\t"// a2 b2 c2 d2
					"vmovaps %%xmm15, (4*3*4)(%%rbx) \n\t"// a3 b3 c3 d3
					"vmovaps %%xmm0, (4*4*4)(%%rbx) \n\t"// a4 b4 c4 d4
					"vmovaps %%xmm1, (4*5*4)(%%rbx) \n\t"// a5 b5 c5 d5
					"vmovaps %%xmm2, (4*6*4)(%%rbx) \n\t"// a6 b6 c6 d6
					"vmovaps %%xmm3, (4*7*4)(%%rbx) \n\t"// a7 b7 c7 d7

					"add $(2*8), %%rax \n\t"// add stride to src pointer
					"add $(4*8*4), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED4%= \n\t"

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

					"vcvtph2ps %%xmm0, %%xmm0 \n\t"
					"vcvtph2ps %%xmm1, %%xmm1 \n\t"
					"vcvtph2ps %%xmm2, %%xmm2 \n\t"
					"vcvtph2ps %%xmm3, %%xmm3 \n\t"

					"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
					"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
					"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
					"vmovss %%xmm3, (4*3)(%%rbx) \n\t"

					"add $(2*1), %%rax \n\t"// add stride to src pointer
					"add $(4*4*1), %%rbx \n\t"// add stride to dst pointer

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
	void pack_avx2_fma_6xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 6);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

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
					"vmovdqu 0x0(%%rax), %%xmm0 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovdqu 0x0(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovdqu 0x0(%%rax), %%xmm2 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovdqu 0x0(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovdqu 0x0(%%rax), %%xmm4 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovdqu 0x0(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovdqu 0x0(%%rax), %%xmm6 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovq 0x0(%%rax), %%xmm7 \n\t"
					"vmovd 0x8(%%rax), %%xmm8 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
					"vcvtph2ps %%xmm8, %%ymm8 \n\t"

					"vmovups %%xmm0, (4*(0*6+0))(%%rbx) \n\t"
					"vmovups %%xmm1, (4*(1*6+0))(%%rbx) \n\t"
					"vmovups %%xmm2, (4*(2*6+0))(%%rbx) \n\t"
					"vmovups %%xmm3, (4*(3*6+0))(%%rbx) \n\t"
					"vmovups %%xmm4, (4*(4*6+0))(%%rbx) \n\t"
					"vmovups %%xmm5, (4*(5*6+0))(%%rbx) \n\t"
					"vmovups %%xmm6, (4*(6*6+0))(%%rbx) \n\t"
					"vmovups %%xmm7, (4*(7*6+0))(%%rbx) \n\t"

					"vextractf128 $0x1, %%ymm0, %%xmm0 \n\t"
					"vextractf128 $0x1, %%ymm1, %%xmm1 \n\t"
					"vextractf128 $0x1, %%ymm2, %%xmm2 \n\t"
					"vextractf128 $0x1, %%ymm3, %%xmm3 \n\t"
					"vextractf128 $0x1, %%ymm4, %%xmm4 \n\t"
					"vextractf128 $0x1, %%ymm5, %%xmm5 \n\t"
					"vextractf128 $0x1, %%ymm6, %%xmm6 \n\t"

					"vmovsd %%xmm0, (4*(0*6+4))(%%rbx) \n\t"
					"vmovsd %%xmm1, (4*(1*6+4))(%%rbx) \n\t"
					"vmovsd %%xmm2, (4*(2*6+4))(%%rbx) \n\t"
					"vmovsd %%xmm3, (4*(3*6+4))(%%rbx) \n\t"
					"vmovsd %%xmm4, (4*(4*6+4))(%%rbx) \n\t"
					"vmovsd %%xmm5, (4*(5*6+4))(%%rbx) \n\t"
					"vmovsd %%xmm6, (4*(6*6+4))(%%rbx) \n\t"
					"vmovsd %%xmm8, (4*(7*6+4))(%%rbx) \n\t"

					"add $(4*6*8), %%rbx \n\t"// add stride to dst pointer (8 * 4 * 6)

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"movups 0x0(%%rax), %%xmm0 \n\t"
					"movsd  0x8(%%rax), %%xmm1 \n\t"
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"movups %%xmm0, (4*(0*6+0))(%%rbx) \n\t"
					"movsd  %%xmm1, (4*(0*6+4))(%%rbx) \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"add $(4*1*6), %%rbx \n\t"// add stride to dst pointer (4 * 1 * 6)

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
					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13
					// first 8x8 tile
					"vmovups 0x0(%%r13), %%xmm0 \n\t"// a0 a1 a2 a3 a4 a5 a6 a7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm1 \n\t"// b0 b1 b2 b3 b4 b5 b6 b7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm2 \n\t"// c0 c1 c2 c3 c4 c5 c6 c7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm3 \n\t"// d0 d1 d2 d3 d4 d5 d6 d7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm4 \n\t"// e0 e1 e2 e3 e4 e5 e6 e7
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm5 \n\t"// f0 f1 f2 f3 f4 f5 f6 f7
					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
					"vcvtph2ps %%xmm8, %%ymm8 \n\t"
					// transpose 6x8
					// first shuffle
					"vunpcklps %%ymm1, %%ymm0, %%ymm6 \n\t"// a0 b0 a1 b1 a4 b4 a5 b5
					"vunpckhps %%ymm1, %%ymm0, %%ymm7 \n\t"// a2 b2 a3 b3 a6 b6 a7 b7
					"vunpcklps %%ymm3, %%ymm2, %%ymm8 \n\t"// c0 d0 c1 d1 c4 d4 c5 d5
					"vunpckhps %%ymm3, %%ymm2, %%ymm9 \n\t"// c2 d2 c3 d3 c6 d6 c7 d7
					"vunpcklps %%ymm5, %%ymm4, %%ymm10 \n\t"// e0 f0 e1 f1 e4 f4 e5 f5
					"vunpckhps %%ymm5, %%ymm4, %%ymm11 \n\t"// e2 f2 e3 f3 e6 f6 e7 f7
					// second shuffle
					"vunpcklpd %%ymm8, %%ymm6, %%ymm12 \n\t"// a0 b0 c0 d0 a4 b4 c4 d4
					"vunpckhpd %%ymm8, %%ymm6, %%ymm13 \n\t"// a1 b1 c1 d1 a5 b5 c5 d5
					"vunpcklpd %%ymm9, %%ymm7, %%ymm14 \n\t"// a2 b2 c2 d2 a6 b6 c6 d6
					"vunpckhpd %%ymm9, %%ymm7, %%ymm15 \n\t"// a3 b3 c3 d3 a7 b7 c7 d7

					"vextractf128 $0x1, %%ymm12, %%xmm0 \n\t"// a4 b4 c4 d4
					"vextractf128 $0x1, %%ymm13, %%xmm1 \n\t"// a5 b5 c5 d5
					"vextractf128 $0x1, %%ymm14, %%xmm2 \n\t"// a6 b6 c6 d6
					"vextractf128 $0x1, %%ymm15, %%xmm3 \n\t"// a7 b7 c7 d7
					"vextractf128 $0x1, %%ymm10, %%xmm4 \n\t"// e4 f4 e5 f5
					"vextractf128 $0x1, %%ymm11, %%xmm5 \n\t"// e6 f6 e7 f7

					"vmovups %%xmm12, (4*(0*6+0))(%%rbx) \n\t"// a0 b0 c0 d0
					"vmovlpd %%xmm10, (4*(0*6+4))(%%rbx) \n\t"// e0 f0
					"vmovups %%xmm13, (4*(1*6+0))(%%rbx) \n\t"// a1 b1 c1 d1
					"vmovhpd %%xmm10, (4*(1*6+4))(%%rbx) \n\t"// e1 f1
					"vmovups %%xmm14, (4*(2*6+0))(%%rbx) \n\t"// a2 b2 c2 d2
					"vmovlpd %%xmm11, (4*(2*6+4))(%%rbx) \n\t"// e2 f2
					"vmovups %%xmm15, (4*(3*6+0))(%%rbx) \n\t"// a3 b3 c3 d3
					"vmovhpd %%xmm11, (4*(3*6+4))(%%rbx) \n\t"// e3 f3

					"vmovups %%xmm0, (4*(4*6+0))(%%rbx) \n\t"// a4 b4 c4 d4
					"vmovlpd %%xmm4, (4*(4*6+4))(%%rbx) \n\t"// e4 f4
					"vmovups %%xmm1, (4*(5*6+0))(%%rbx) \n\t"// a5 b5 c5 d5
					"vmovhpd %%xmm4, (4*(5*6+4))(%%rbx) \n\t"// e5 f5
					"vmovups %%xmm2, (4*(6*6+0))(%%rbx) \n\t"// a6 b6 c6 d6
					"vmovlpd %%xmm5, (4*(6*6+4))(%%rbx) \n\t"// e6 f6
					"vmovups %%xmm3, (4*(7*6+0))(%%rbx) \n\t"// a7 b7 c7 d7
					"vmovhpd %%xmm5, (4*(7*6+4))(%%rbx) \n\t"// e7 f7

					"add $(2*8), %%rax \n\t"// add stride to src pointer
					"add $(4*6*8), %%rbx \n\t"// add stride to dst pointer

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
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm3 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm4 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm5 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"

					"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
					"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
					"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
					"vmovss %%xmm3, (4*3)(%%rbx) \n\t"
					"vmovss %%xmm4, (4*4)(%%rbx) \n\t"
					"vmovss %%xmm5, (4*5)(%%rbx) \n\t"

					"add $(2*1), %%rax \n\t"
					"add $(4*6*1), %%rbx \n\t"// add stride to dst pointer

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
	void pack_avx2_fma_16xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 16);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

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
					"vmovups 0x10(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm2 \n\t"
					"vmovups 0x10(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm4 \n\t"
					"vmovups 0x10(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm6 \n\t"
					"vmovups 0x10(%%rax), %%xmm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm8 \n\t"
					"vmovups 0x10(%%rax), %%xmm9 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm10 \n\t"
					"vmovups 0x10(%%rax), %%xmm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm12 \n\t"
					"vmovups 0x10(%%rax), %%xmm13 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm14 \n\t"
					"vmovups 0x10(%%rax), %%xmm15 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
					"vcvtph2ps %%xmm8, %%ymm8 \n\t"
					"vcvtph2ps %%xmm9, %%ymm9 \n\t"
					"vcvtph2ps %%xmm10, %%ymm10 \n\t"
					"vcvtph2ps %%xmm11, %%ymm11 \n\t"
					"vcvtph2ps %%xmm12, %%ymm12 \n\t"
					"vcvtph2ps %%xmm13, %%ymm13 \n\t"
					"vcvtph2ps %%xmm14, %%ymm14 \n\t"
					"vcvtph2ps %%xmm15, %%ymm15 \n\t"

					"vmovaps %%ymm0, 0x000(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x020(%%rbx) \n\t"
					"vmovaps %%ymm2, 0x040(%%rbx) \n\t"
					"vmovaps %%ymm3, 0x060(%%rbx) \n\t"
					"vmovaps %%ymm4, 0x080(%%rbx) \n\t"
					"vmovaps %%ymm5, 0x0A0(%%rbx) \n\t"
					"vmovaps %%ymm6, 0x0C0(%%rbx) \n\t"
					"vmovaps %%ymm7, 0x0E0(%%rbx) \n\t"
					"vmovaps %%ymm8, 0x100(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x120(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x140(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x160(%%rbx) \n\t"
					"vmovaps %%ymm12, 0x180(%%rbx) \n\t"
					"vmovaps %%ymm13, 0x1A0(%%rbx) \n\t"
					"vmovaps %%ymm14, 0x1C0(%%rbx) \n\t"
					"vmovaps %%ymm15, 0x1E0(%%rbx) \n\t"

					"add $(4*8*16), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%xmm0 \n\t"
					"vmovups 0x10(%%rax), %%xmm1 \n\t"
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vmovaps %%ymm0, 0x00(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x20(%%rbx) \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"add $(4*1*16), %%rbx \n\t"// add stride to dst pointer

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

					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
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
					// store
					"vmovaps %%ymm8, 0x000(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x040(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x080(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x0C0(%%rbx) \n\t"
					"vmovaps %%ymm12, 0x100(%%rbx) \n\t"
					"vmovaps %%ymm13, 0x140(%%rbx) \n\t"
					"vmovaps %%ymm14, 0x180(%%rbx) \n\t"
					"vmovaps %%ymm15, 0x1C0(%%rbx) \n\t"

					// second 8x8 tile
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
					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
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
					// store
					"vmovaps %%ymm8, 0x020(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x060(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x0A0(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x0E0(%%rbx) \n\t"
					"vmovaps %%ymm12, 0x120(%%rbx) \n\t"
					"vmovaps %%ymm13, 0x160(%%rbx) \n\t"
					"vmovaps %%ymm14, 0x1A0(%%rbx) \n\t"
					"vmovaps %%ymm15, 0x1E0(%%rbx) \n\t"

					"add $(2*8), %%rax \n\t"// add stride to src pointer
					"add $(4*16*8), %%rbx \n\t"// add stride to dst pointer

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
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm12 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm13 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm14 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm15 \n\t"

					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
					"vcvtph2ps %%xmm8, %%ymm8 \n\t"
					"vcvtph2ps %%xmm9, %%ymm9 \n\t"
					"vcvtph2ps %%xmm10, %%ymm10 \n\t"
					"vcvtph2ps %%xmm11, %%ymm11 \n\t"
					"vcvtph2ps %%xmm12, %%ymm12 \n\t"
					"vcvtph2ps %%xmm13, %%ymm13 \n\t"
					"vcvtph2ps %%xmm14, %%ymm14 \n\t"
					"vcvtph2ps %%xmm15, %%ymm15 \n\t"

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
					"vmovss %%xmm12, (4*12)(%%rbx) \n\t"
					"vmovss %%xmm13, (4*13)(%%rbx) \n\t"
					"vmovss %%xmm14, (4*14)(%%rbx) \n\t"
					"vmovss %%xmm15, (4*15)(%%rbx) \n\t"

					"add $(2*1), %%rax \n\t"// add stride to src pointer
					"add $(4*16*1), %%rbx \n\t"// add stride to dst pointer

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
	void pack_avx2_fma_24xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 24);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

		const uint64_t src_stride = src.stride() * sizeof(float16);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
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

					"UNROLLED8%=: \n\t"
					"vmovups 0x00(%%rax), %%xmm0 \n\t"
					"vmovups 0x10(%%rax), %%xmm1 \n\t"
					"vmovups 0x20(%%rax), %%xmm2 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm3 \n\t"
					"vmovups 0x10(%%rax), %%xmm4 \n\t"
					"vmovups 0x20(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm6 \n\t"
					"vmovups 0x10(%%rax), %%xmm7 \n\t"
					"vmovups 0x20(%%rax), %%xmm8 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm9 \n\t"
					"vmovups 0x10(%%rax), %%xmm10 \n\t"
					"vmovups 0x20(%%rax), %%xmm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
					"vcvtph2ps %%xmm8, %%ymm8 \n\t"
					"vcvtph2ps %%xmm9, %%ymm9 \n\t"
					"vcvtph2ps %%xmm10, %%ymm10 \n\t"
					"vcvtph2ps %%xmm11, %%ymm11 \n\t"

					"vmovaps %%ymm0, 0x000(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x020(%%rbx) \n\t"
					"vmovaps %%ymm2, 0x040(%%rbx) \n\t"
					"vmovaps %%ymm3, 0x060(%%rbx) \n\t"
					"vmovaps %%ymm4, 0x080(%%rbx) \n\t"
					"vmovaps %%ymm5, 0x0A0(%%rbx) \n\t"
					"vmovaps %%ymm6, 0x0C0(%%rbx) \n\t"
					"vmovaps %%ymm7, 0x0E0(%%rbx) \n\t"
					"vmovaps %%ymm8, 0x100(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x120(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x140(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x160(%%rbx) \n\t"

					"add $(4*4*24), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%xmm0 \n\t"
					"vmovups 0x10(%%rax), %%xmm1 \n\t"
					"vmovups 0x20(%%rax), %%xmm2 \n\t"
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vmovaps %%ymm0, 0x00(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x20(%%rbx) \n\t"
					"vmovaps %%ymm2, 0x40(%%rbx) \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"add $(4*1*24), %%rbx \n\t"// add stride to dst pointer

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

					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"

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
					// store
					"vmovaps %%ymm8, (0*4*8+4*0*24)(%%rbx) \n\t"
					"vmovaps %%ymm9, (0*4*8+4*1*24)(%%rbx) \n\t"
					"vmovaps %%ymm10, (0*4*8+4*2*24)(%%rbx) \n\t"
					"vmovaps %%ymm11, (0*4*8+4*3*24)(%%rbx) \n\t"
					"vmovaps %%ymm12, (0*4*8+4*4*24)(%%rbx) \n\t"
					"vmovaps %%ymm13, (0*4*8+4*5*24)(%%rbx) \n\t"
					"vmovaps %%ymm14, (0*4*8+4*6*24)(%%rbx) \n\t"
					"vmovaps %%ymm15, (0*4*8+4*7*24)(%%rbx) \n\t"

					// second 8x8 tile
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
					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
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
					// store
					"vmovaps %%ymm8, (1*4*8+4*0*24)(%%rbx) \n\t"
					"vmovaps %%ymm9, (1*4*8+4*1*24)(%%rbx) \n\t"
					"vmovaps %%ymm10, (1*4*8+4*2*24)(%%rbx) \n\t"
					"vmovaps %%ymm11, (1*4*8+4*3*24)(%%rbx) \n\t"
					"vmovaps %%ymm12, (1*4*8+4*4*24)(%%rbx) \n\t"
					"vmovaps %%ymm13, (1*4*8+4*5*24)(%%rbx) \n\t"
					"vmovaps %%ymm14, (1*4*8+4*6*24)(%%rbx) \n\t"
					"vmovaps %%ymm15, (1*4*8+4*7*24)(%%rbx) \n\t"

					// third 8x8 tile
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
					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
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
					// store
					"vmovaps %%ymm8, (2*4*8+4*0*24)(%%rbx) \n\t"
					"vmovaps %%ymm9, (2*4*8+4*1*24)(%%rbx) \n\t"
					"vmovaps %%ymm10, (2*4*8+4*2*24)(%%rbx) \n\t"
					"vmovaps %%ymm11, (2*4*8+4*3*24)(%%rbx) \n\t"
					"vmovaps %%ymm12, (2*4*8+4*4*24)(%%rbx) \n\t"
					"vmovaps %%ymm13, (2*4*8+4*5*24)(%%rbx) \n\t"
					"vmovaps %%ymm14, (2*4*8+4*6*24)(%%rbx) \n\t"
					"vmovaps %%ymm15, (2*4*8+4*7*24)(%%rbx) \n\t"

					"add $(2*8), %%rax \n\t"// add stride to src pointer
					"add $(4*24*8), %%rbx \n\t"// add stride to dst pointer

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
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm12 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm13 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm14 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"movzw 0x0(%%r13), %%rcx \n\t"
					"vmovq %%rcx, %%xmm15 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer

					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"
					"vcvtph2ps %%xmm8, %%ymm8 \n\t"
					"vcvtph2ps %%xmm9, %%ymm9 \n\t"
					"vcvtph2ps %%xmm10, %%ymm10 \n\t"
					"vcvtph2ps %%xmm11, %%ymm11 \n\t"
					"vcvtph2ps %%xmm12, %%ymm12 \n\t"
					"vcvtph2ps %%xmm13, %%ymm13 \n\t"
					"vcvtph2ps %%xmm14, %%ymm14 \n\t"
					"vcvtph2ps %%xmm15, %%ymm15 \n\t"

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
					"vmovss %%xmm12, (4*12)(%%rbx) \n\t"
					"vmovss %%xmm13, (4*13)(%%rbx) \n\t"
					"vmovss %%xmm14, (4*14)(%%rbx) \n\t"
					"vmovss %%xmm15, (4*15)(%%rbx) \n\t"

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

					// convert fp16 -> fp32
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"

					"vmovss %%xmm0, (4*16)(%%rbx) \n\t"
					"vmovss %%xmm1, (4*17)(%%rbx) \n\t"
					"vmovss %%xmm2, (4*18)(%%rbx) \n\t"
					"vmovss %%xmm3, (4*19)(%%rbx) \n\t"
					"vmovss %%xmm4, (4*20)(%%rbx) \n\t"
					"vmovss %%xmm5, (4*21)(%%rbx) \n\t"
					"vmovss %%xmm6, (4*22)(%%rbx) \n\t"
					"vmovss %%xmm7, (4*23)(%%rbx) \n\t"

					"add $(2*1), %%rax \n\t"// add stride to src pointer
					"add $(4*24*1), %%rbx \n\t"// add stride to dst pointer

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

