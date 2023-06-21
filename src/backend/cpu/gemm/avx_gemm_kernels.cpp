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
#include "../vectors/vectors.hpp"

#include <x86intrin.h>
#include <cinttypes>
#include <cassert>

#include "../src/backend/cpu/assembly_macros.hpp"

#define SUBBLOCK(n) movups(mem(rax, 4*10+n), xmm(n))

namespace ml
{
	void func_test()
	{
		const void *A_ptr = nullptr;

		begin_asm()
		movq(var(A_ptr), rax)
		SUBBLOCK(0)
		SUBBLOCK(1)
		end_asm(
				: // outputs
				:// inputs
				[A_ptr] "m"(A_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14")
	}

	void gemm_avx_10x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 10);
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
				"vxorps %%ymm6, %%ymm6, %%ymm6 \n\t"
				"vxorps %%ymm7, %%ymm7, %%ymm7 \n\t"
				"vxorps %%ymm8, %%ymm8, %%ymm8 \n\t"
				"vxorps %%ymm9, %%ymm9, %%ymm9 \n\t"
				"vxorps %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vxorps %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vxorps %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vxorps %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vxorps %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vxorps %%ymm15, %%ymm15, %%ymm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm0 \n\t"

				"vbroadcastss (4*0)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*1)(%%rax), %%ymm2 \n\t"
				"vbroadcastss (4*2)(%%rax), %%ymm3 \n\t"
				"vbroadcastss (4*3)(%%rax), %%ymm4 \n\t"
				"vbroadcastss (4*4)(%%rax), %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm0, %%ymm5 \n\t"

				"vaddps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vaddps %%ymm2, %%ymm7, %%ymm7 \n\t"
				"vaddps %%ymm3, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm4, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm5, %%ymm10, %%ymm10 \n\t"

				"vbroadcastss (4*5)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*6)(%%rax), %%ymm2 \n\t"
				"vbroadcastss (4*7)(%%rax), %%ymm3 \n\t"
				"vbroadcastss (4*8)(%%rax), %%ymm4 \n\t"
				"vbroadcastss (4*9)(%%rax), %%ymm5 \n\t"

				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm0, %%ymm5 \n\t"

				"vaddps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vaddps %%ymm2, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm3, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm4, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm5, %%ymm15, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x20(%%rbx), %%ymm0 \n\t"

				"vbroadcastss (4*10)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*11)(%%rax), %%ymm2 \n\t"
				"vbroadcastss (4*12)(%%rax), %%ymm3 \n\t"
				"vbroadcastss (4*13)(%%rax), %%ymm4 \n\t"
				"vbroadcastss (4*14)(%%rax), %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm0, %%ymm5 \n\t"

				"vaddps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vaddps %%ymm2, %%ymm7, %%ymm7 \n\t"
				"vaddps %%ymm3, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm4, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm5, %%ymm10, %%ymm10 \n\t"

				"vbroadcastss (4*15)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*16)(%%rax), %%ymm2 \n\t"
				"vbroadcastss (4*17)(%%rax), %%ymm3 \n\t"
				"vbroadcastss (4*18)(%%rax), %%ymm4 \n\t"
				"vbroadcastss (4*19)(%%rax), %%ymm5 \n\t"

				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm0, %%ymm5 \n\t"

				"vaddps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vaddps %%ymm2, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm3, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm4, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm5, %%ymm15, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x40(%%rbx), %%ymm0 \n\t"

				"vbroadcastss (4*20)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*21)(%%rax), %%ymm2 \n\t"
				"vbroadcastss (4*22)(%%rax), %%ymm3 \n\t"
				"vbroadcastss (4*23)(%%rax), %%ymm4 \n\t"
				"vbroadcastss (4*24)(%%rax), %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm0, %%ymm5 \n\t"

				"vaddps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vaddps %%ymm2, %%ymm7, %%ymm7 \n\t"
				"vaddps %%ymm3, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm4, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm5, %%ymm10, %%ymm10 \n\t"

				"vbroadcastss (4*25)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*26)(%%rax), %%ymm2 \n\t"
				"vbroadcastss (4*27)(%%rax), %%ymm3 \n\t"
				"vbroadcastss (4*28)(%%rax), %%ymm4 \n\t"
				"vbroadcastss (4*29)(%%rax), %%ymm5 \n\t"

				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm0, %%ymm5 \n\t"

				"vaddps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vaddps %%ymm2, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm3, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm4, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm5, %%ymm15, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0x60(%%rbx), %%ymm0 \n\t"

				"vbroadcastss (4*30)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*31)(%%rax), %%ymm2 \n\t"
				"vbroadcastss (4*32)(%%rax), %%ymm3 \n\t"
				"vbroadcastss (4*33)(%%rax), %%ymm4 \n\t"
				"vbroadcastss (4*34)(%%rax), %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm0, %%ymm5 \n\t"

				"vaddps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vaddps %%ymm2, %%ymm7, %%ymm7 \n\t"
				"vaddps %%ymm3, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm4, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm5, %%ymm10, %%ymm10 \n\t"

				"vbroadcastss (4*35)(%%rax), %%ymm1 \n\t"
				"vbroadcastss (4*36)(%%rax), %%ymm2 \n\t"
				"vbroadcastss (4*37)(%%rax), %%ymm3 \n\t"
				"vbroadcastss (4*38)(%%rax), %%ymm4 \n\t"
				"vbroadcastss (4*39)(%%rax), %%ymm5 \n\t"

				"vmulps %%ymm1, %%ymm0, %%ymm1 \n\t"
				"vmulps %%ymm2, %%ymm0, %%ymm2 \n\t"
				"vmulps %%ymm3, %%ymm0, %%ymm3 \n\t"
				"vmulps %%ymm4, %%ymm0, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm0, %%ymm5 \n\t"

				"vaddps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vaddps %%ymm2, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm3, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm4, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm5, %%ymm15, %%ymm15 \n\t"

				"add $0xA0, %%rax \n\t"
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

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm0 \n\t"

				// scale by alpha
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

				"vxorps %%ymm1, %%ymm1, %%ymm1 \n\t"
				"vucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
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

	void pack_avx_8xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 8);
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
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm2 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm4 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm6 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

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
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
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
	}
	void pack_avx_10xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 10);
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
					"vmovsd  0x20(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm2 \n\t"
					"vmovsd  0x20(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm4 \n\t"
					"vmovsd  0x20(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm6 \n\t"
					"vmovsd  0x20(%%rax), %%xmm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm8 \n\t"
					"vmovsd  0x20(%%rax), %%xmm9 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm10 \n\t"
					"vmovsd  0x20(%%rax), %%xmm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm12 \n\t"
					"vmovsd  0x20(%%rax), %%xmm13 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm14 \n\t"
					"vmovsd  0x20(%%rax), %%xmm15 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vmovups %%ymm0, (4*(0*10+0))(%%rbx) \n\t"
					"vmovsd  %%xmm1, (4*(0*10+8))(%%rbx) \n\t"
					"vmovups %%ymm2, (4*(1*10+0))(%%rbx) \n\t"
					"vmovsd  %%xmm3, (4*(1*10+8))(%%rbx) \n\t"
					"vmovups %%ymm4, (4*(2*10+0))(%%rbx) \n\t"
					"vmovsd  %%xmm5, (4*(2*10+8))(%%rbx) \n\t"
					"vmovups %%ymm6, (4*(3*10+0))(%%rbx) \n\t"
					"vmovsd  %%xmm7, (4*(3*10+8))(%%rbx) \n\t"
					"vmovups %%ymm8, (4*(4*10+0))(%%rbx) \n\t"
					"vmovsd  %%xmm9, (4*(4*10+8))(%%rbx) \n\t"
					"vmovups %%ymm10, (4*(5*10+0))(%%rbx) \n\t"
					"vmovsd  %%xmm11, (4*(5*10+8))(%%rbx) \n\t"
					"vmovups %%ymm12, (4*(6*10+0))(%%rbx) \n\t"
					"vmovsd  %%xmm13, (4*(6*10+8))(%%rbx) \n\t"
					"vmovups %%ymm14, (4*(7*10+0))(%%rbx) \n\t"
					"vmovsd  %%xmm15, (4*(7*10+8))(%%rbx) \n\t"

					"add $(4*8*10), %%rbx \n\t"// add stride to dst pointer

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"vmovsd  0x20(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups %%ymm0, 0x00(%%rbx) \n\t"
					"vmovsd  %%xmm1, 0x20(%%rbx) \n\t"
					"add $(4*1*10), %%rbx \n\t"// add stride to dst pointer

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
					// rows 0-7
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

					"vmovups %%ymm8, (4*(0*10+0))(%%rbx) \n\t"
					"vmovups %%ymm9, (4*(1*10+0))(%%rbx) \n\t"
					"vmovups %%ymm10, (4*(2*10+0))(%%rbx) \n\t"
					"vmovups %%ymm11, (4*(3*10+0))(%%rbx) \n\t"
					"vmovups %%ymm12, (4*(4*10+0))(%%rbx) \n\t"
					"vmovups %%ymm13, (4*(5*10+0))(%%rbx) \n\t"
					"vmovups %%ymm14, (4*(6*10+0))(%%rbx) \n\t"
					"vmovups %%ymm15, (4*(7*10+0))(%%rbx) \n\t"

					// rows 8-9
					"vmovups 0x0(%%r13), %%ymm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%ymm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer

					"vunpcklps %%ymm1, %%ymm0, %%ymm4 \n\t"
					"vunpckhps %%ymm1, %%ymm0, %%ymm5 \n\t"

					"vextractf128 $0x1, %%ymm4, %%xmm6 \n\t"// e4 f4 e5 f5
					"vextractf128 $0x1, %%ymm5, %%xmm7 \n\t"// e6 f6 e7 f7

					"vmovlpd %%xmm4, (4*(0*10+8))(%%rbx) \n\t"
					"vmovhpd %%xmm4, (4*(1*10+8))(%%rbx) \n\t"
					"vmovlpd %%xmm5, (4*(2*10+8))(%%rbx) \n\t"
					"vmovhpd %%xmm5, (4*(3*10+8))(%%rbx) \n\t"
					"vmovlpd %%xmm6, (4*(4*10+8))(%%rbx) \n\t"
					"vmovhpd %%xmm6, (4*(5*10+8))(%%rbx) \n\t"
					"vmovlpd %%xmm7, (4*(6*10+8))(%%rbx) \n\t"
					"vmovhpd %%xmm7, (4*(7*10+8))(%%rbx) \n\t"

					"add $(4*8), %%rax \n\t"// add stride to src pointer
					"add $(4*8*10), %%rbx \n\t"// add stride to dst pointer

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

					"add $(4*1), %%rax \n\t"// add stride to src pointer
					"add $(4*10*1), %%rbx \n\t"// add stride to dst pointer

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

