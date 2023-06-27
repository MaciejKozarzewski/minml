/*
 * avx2_winograd_kernels.cpp
 *
 *  Created on: Jun 26, 2023
 *      Author: Maciej Kozarzewski
 */

#include "winograd_kernels.hpp"

#include "../assembly_macros.hpp"

#include <array>
#include <cinttypes>
#include <cassert>

namespace
{
	bool is_not_null(const void **src, int size) noexcept
	{
		if (src == nullptr)
			return false;
		for (int i = 0; i < size; i++)
			if (src[i] == nullptr)
				return false;
		return true;
	}

#define LOAD_INPUT_7x8xFP32()\
	movq(mem(rsi, 0*7*8), r12)\
	movq(mem(rsi, 1*7*8), r13)\
	movq(mem(rsi, 2*7*8), r14)\
	movq(mem(rsi, 3*7*8), r15)\
	vmovups(mem(r12, r9, 1), ymm0)\
	vmovups(mem(r13, r9, 1), ymm1)\
	vmovups(mem(r14, r9, 1), ymm2)\
	vmovups(mem(r15, r9, 1), ymm3)\
	movq(mem(rsi, 4*7*8), r12)\
	movq(mem(rsi, 5*7*8), r13)\
	movq(mem(rsi, 6*7*8), r14)\
	vmovups(mem(r12, r9, 1), ymm4)\
	vmovups(mem(r13, r9, 1), ymm5)\
	vmovups(mem(r14, r9, 1), ymm6)

#define STORE_OUTPUT_7x8xFP32()\
	movq(mem(rdi, 0*8), r12)\
	movq(mem(rdi, 1*8), r13)\
	movq(mem(rdi, 2*8), r14)\
	movq(mem(rdi, 3*8), r15)\
	vmovups(ymm0, mem(r12, r9, 1))\
	vmovups(ymm1, mem(r13, r9, 1))\
	vmovups(ymm2, mem(r14, r9, 1))\
	vmovups(ymm3, mem(r15, r9, 1))\
	movq(mem(rdi, 4*8), r12)\
	movq(mem(rdi, 5*8), r13)\
	movq(mem(rdi, 6*8), r14)\
	vmovups(ymm4, mem(r12, r9, 1))\
	vmovups(ymm5, mem(r13, r9, 1))\
	vmovups(ymm6, mem(r14, r9, 1))

}

namespace ml
{
	void winograd_input_transform_5x5_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		assert(is_not_null(src));
		assert(is_not_null(dst));
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 8;
		const uint64_t k_left = filters % 8;

		const float constants[2] = { 0.25f, 0.5f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r9) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r9, 0), ymm14)// 0.25f
		vbroadcastss(mem(r9, 4), ymm15)// 0.5f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED8)// main loop over channels, in steps of 8 elements
		movq(rax, rsi)
		movq(rbx, rdi)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1)
		LOAD_INPUT_7x8xFP32()

		// here goes the actual transform
		// rows 0 and 1
		vsubps(ymm1, ymm3, ymm7)// ymm7 = ymm3-ymm1
		vsubps(ymm5, ymm3, ymm8)// ymm8 = ymm3-ymm5
		vsubps(ymm2, ymm4, ymm9)// ymm9 = ymm4-ymm2
		vsubps(ymm2, ymm0, ymm10)// ymm10 = ymm0-ymm2
		vsubps(ymm1, ymm2, ymm11)// ymm11 = ymm2-ymm1
		vsubps(ymm4, ymm3, ymm12)// ymm12 = ymm3-ymm4
		vaddps(ymm3, ymm3, ymm13)// ymm13 = 2*ymm3
		vaddps(ymm7, ymm7, ymm7)// ymm7 = 2*(ymm3-ymm1) -> part of row 0

		vfmadd231ps(ymm15, ymm8, ymm10)// ymm10 = 0.5*(ymm3-ymm5) + (ymm0-ymm2) -> part of row 0
		vfmadd231ps(ymm14, ymm12, ymm11)// ymm11 = 0.25*(ymm3-ymm4) + (ymm2-ymm1) -> part of row 1
		vfnmadd231ps(ymm15, ymm5, ymm13)// ymm13 = -0.5*ymm5 + 2*ymm3 -> part of row 1
		vfmadd231ps(ymm14, ymm9, ymm7)// ymm9 = 0.25*(ymm4-ymm2) + 2*(ymm3-ymm1) -> part of row 0

		vaddps(ymm7, ymm10, ymm7)
		vaddps(ymm11, ymm13, ymm11)
		// store rows 0 and 1
		vmovaps(ymm7, mem(rdi, 0*7*8*4))// store row 0
		vmovaps(ymm11, mem(rdi, 1*7*8*4))// store row 1

		// rows 2 and 3
		vsubps(ymm1, ymm2, ymm7)// ymm7 = ymm2-ymm1
		vaddps(ymm3, ymm4, ymm8)// ymm8 = ymm3+ymm4
		vmovaps(ymm2, ymm9)// ymm9 = ymm2
		vsubps(ymm3, ymm2, ymm10)// ymm10 = ymm2-ymm3
		vaddps(ymm1, ymm5, ymm11)// ymm11 = ymm1+ymm5
		vsubps(ymm4, ymm2, ymm12)// ymm12 = ymm2-ymm4
		vmovaps(ymm3, ymm13)// ymm13 = ymm3

		vfmadd231ps(ymm14, ymm8, ymm7)// ymm7 = 0.25*(ymm3+ymm4) + (ymm2-ymm1)
		vfmadd231ps(ymm15, ymm5, ymm9)// ymm9 = 0.5*ymm5 + ymm2
		vsubps(ymm8, ymm10, ymm8)// ymm8 = (ymm2-ymm3) - (ymm3+ymm4)
		vfmsub231ps(ymm15, ymm11, ymm13)// ymm11 = 0.5*(ymm1+ymm5) - ymm3
		vfmsub231ps(ymm14, ymm12, ymm12)// ymm12 = 0.25*(ymm2-ymm4) - (ymm2-ymm4)

		vaddps(ymm7, ymm8, ymm7)
		vaddps(ymm7, ymm9, ymm7)
		vaddps(ymm13, ymm12, ymm11)
		// store rows 2 and 3
		vmovaps(ymm7, mem(rdi, 2*7*8*4))
		vmovaps(ymm11, mem(rdi, 3*7*8*4))

		// rows 4, 5 and 6
		vsubps(ymm5, ymm1, ymm7)// ymm7 = ymm1-ymm5
		vsubps(ymm4, ymm2, ymm8)// ymm8 = ymm2-ymm4
		vsubps(ymm1, ymm3, ymm9)// ymm9 = ymm3-ymm1
		vsubps(ymm5, ymm3, ymm10)// ymm10 = ymm3-ymm5
		vsubps(ymm4, ymm6, ymm6)// ymm6 = ymm6-ymm4
		vmovaps(ymm8, ymm11)// ymm11 = ymm2-ymm4
		vmovaps(ymm9, ymm12)// ymm12 = ymm3-ymm1
		vaddps(ymm8, ymm8, ymm13)// ymm13 = 2*(ymm2-ymm4)

		vfmsub231ps(ymm15, ymm7, ymm11)// ymm11 = 0.5*(ymm1-ymm5) - (ymm2-ymm4)
		vfnmsub231ps(ymm14, ymm10, ymm9)// ymm9 = -0.25*(ymm3-ymm5) - (ymm3-ymm1) -> complete row 5
		vfmadd231ps(ymm14, ymm10, ymm12)// ymm12 = 0.25*(ymm3-ymm5) + (ymm3-ymm1)
		vfmadd231ps(ymm15, ymm6, ymm13)// ymm13 = 0.5*(ymm6-ymm4) + 2*(ymm2-ymm4)

		vfnmadd231ps(ymm14, ymm8, ymm11)// complete row 4
		vaddps(ymm12, ymm13, ymm12)// complete row 6

		vmovaps(ymm11, mem(rdi, 4*7*8*4))
		vmovaps(ymm9, mem(rdi, 5*7*8*4))
		vmovaps(ymm12, mem(rdi, 6*7*8*4))

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*8*4), rdi)// add 8*4 (8 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2)
		// second transform
		vmovaps(mem(rsi, 0*8*4), ymm0)// loading row 0
		vmovaps(mem(rsi, 1*8*4), ymm1)
		vmovaps(mem(rsi, 2*8*4), ymm2)
		vmovaps(mem(rsi, 3*8*4), ymm3)
		vmovaps(mem(rsi, 4*8*4), ymm4)
		vmovaps(mem(rsi, 5*8*4), ymm5)
		vmovaps(mem(rsi, 6*8*4), ymm6)

		movq(mem(rdi, 0*8), r12)
		movq(mem(rdi, 1*8), r13)
		movq(mem(rdi, 2*8), r14)
		movq(mem(rdi, 3*8), r15)

		// here goes the actual transform
		// rows 0 and 1
		vsubps(ymm1, ymm3, ymm7)// ymm7 = ymm3-ymm1
		vsubps(ymm5, ymm3, ymm8)// ymm8 = ymm3-ymm5
		vsubps(ymm2, ymm4, ymm9)// ymm9 = ymm4-ymm2
		vsubps(ymm2, ymm0, ymm10)// ymm10 = ymm0-ymm2
		vsubps(ymm1, ymm2, ymm11)// ymm11 = ymm2-ymm1
		vsubps(ymm4, ymm3, ymm12)// ymm12 = ymm3-ymm4
		vaddps(ymm3, ymm3, ymm13)// ymm13 = 2*ymm3
		vaddps(ymm7, ymm7, ymm7)// ymm7 = 2*(ymm3-ymm1) -> part of row 0

		vfmadd231ps(ymm15, ymm8, ymm10)// ymm10 = 0.5*(ymm3-ymm5) + (ymm0-ymm2) -> part of row 0
		vfmadd231ps(ymm14, ymm12, ymm11)// ymm11 = 0.25*(ymm3-ymm4) + (ymm2-ymm1) -> part of row 1
		vfnmadd231ps(ymm15, ymm5, ymm13)// ymm13 = -0.5*ymm5 + 2*ymm3 -> part of row 1
		vfmadd231ps(ymm14, ymm9, ymm7)// ymm9 = 0.25*(ymm4-ymm2) + 2*(ymm3-ymm1) -> part of row 0

		vaddps(ymm11, ymm13, ymm11)
		vaddps(ymm7, ymm10, ymm7)
		// store rows 0 and 1
		vmovups(ymm7, mem(r12, r9, 1))
		vmovups(ymm11, mem(r13, r9, 1))

		// rows 2 and 3
		vsubps(ymm1, ymm2, ymm7)// ymm7 = ymm2-ymm1
		vaddps(ymm3, ymm4, ymm8)// ymm8 = ymm3+ymm4
		vmovaps(ymm2, ymm9)// ymm9 = ymm2
		vsubps(ymm3, ymm2, ymm10)// ymm10 = ymm2-ymm3
		vaddps(ymm1, ymm5, ymm11)// ymm11 = ymm1+ymm5
		vsubps(ymm4, ymm2, ymm12)// ymm12 = ymm2-ymm4
		vmovaps(ymm3, ymm13)// ymm13 = ymm3

		vfmadd231ps(ymm14, ymm8, ymm7)// ymm7 = 0.25*(ymm3+ymm4) + (ymm2-ymm1)
		vfmadd231ps(ymm15, ymm5, ymm9)// ymm9 = 0.5*ymm5 + ymm2
		vsubps(ymm8, ymm10, ymm8)// ymm8 = (ymm2-ymm3) - (ymm3+ymm4)
		vfmsub231ps(ymm15, ymm11, ymm13)// ymm11 = 0.5*(ymm1+ymm5) - ymm3
		vfmsub231ps(ymm14, ymm12, ymm12)// ymm12 = 0.25*(ymm2-ymm4) - (ymm2-ymm4)

		vaddps(ymm7, ymm8, ymm7)
		vaddps(ymm7, ymm9, ymm7)
		vaddps(ymm13, ymm12, ymm11)

		// store rows 2 and 3
		vmovups(ymm7, mem(r14, r9, 1))
		vmovups(ymm11, mem(r15, r9, 1))

		// rows 4, 5 and 6
		movq(mem(rdi, 4*8), r12)
		movq(mem(rdi, 5*8), r13)
		movq(mem(rdi, 6*8), r14)

		vsubps(ymm5, ymm1, ymm7)// ymm7 = ymm1-ymm5
		vsubps(ymm4, ymm2, ymm8)// ymm8 = ymm2-ymm4
		vsubps(ymm1, ymm3, ymm9)// ymm9 = ymm3-ymm1
		vsubps(ymm5, ymm3, ymm10)// ymm10 = ymm3-ymm5
		vsubps(ymm4, ymm6, ymm6)// ymm6 = ymm6-ymm4
		vmovaps(ymm8, ymm11)// ymm11 = ymm2-ymm4
		vmovaps(ymm9, ymm12)// ymm12 = ymm3-ymm1
		vaddps(ymm8, ymm8, ymm13)// ymm13 = 2*(ymm2-ymm4)

		vfmsub231ps(ymm15, ymm7, ymm11)// ymm11 = 0.5*(ymm1-ymm5) - (ymm2-ymm4)
		vfnmsub231ps(ymm14, ymm10, ymm9)// ymm9 = -0.25*(ymm3-ymm5) - (ymm3-ymm1) -> complete row 5
		vfmadd231ps(ymm14, ymm10, ymm12)// ymm12 = 0.25*(ymm3-ymm5) + (ymm3-ymm1)
		vfmadd231ps(ymm15, ymm6, ymm13)// ymm13 = 0.5*(ymm6-ymm4) + 2*(ymm2-ymm4)

		vfnmadd231ps(ymm14, ymm8, ymm11)// complete row 4
		vaddps(ymm12, ymm13, ymm12)// complete row 6
		// store rows 4, 5 and 6
		vmovups(ymm11, mem(r12, r9, 1))
		vmovups(ymm9, mem(r13, r9, 1))
		vmovups(ymm12, mem(r14, r9, 1))

		add(imm(7*8*4), rsi)// add 7*8 (7 pointers) to rsi (workspace), moving to next row
		add(imm(7*8), rdi)// add 7*8*4 (7*8 floats) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2)

		add(imm(8*4), r9)// add 8*4 (8 floats) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED8)

		label(FINALLOOP)
//		movq(var(k_left), r10)// load the number of 1-unrolled iterations
//		test(r10, r10)
//		je(EPILOGUE)
//
//		label(UNROLLED1)
//
//		add(imm(1*4), r9)// add 1*4 (1 float) to r9, the offset in channels
//		dec(r10)
//		jne(UNROLLED1)
//
//		label(EPILOGUE)

		end_asm(
				:// outputs
				:// inputs
				[src_ptr] "m"(src_ptr),
				[dst_ptr] "m"(dst_ptr),
				[workspace_ptr] "m"(workspace_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[c_ptr] "m"(c_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
	}

} /* namespace ml */

