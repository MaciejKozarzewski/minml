/*
 * sse2_winograd_kernels.cpp
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
#define LOAD_1x4_FP32(reg, offset, src, dst)\
	movq(mem(reg, src), r15)\
	movups(mem(r15, offset, 1), dst)
#define LOAD_2x4_FP32(reg, offset, src1, src2, dst1, dst2)\
	movq(mem(reg, src1), r14)\
	movq(mem(reg, src2), r15)\
	movups(mem(r14, offset, 1), dst1)\
	movups(mem(r15, offset, 1), dst2)
#define LOAD_1x1_FP32(reg, offset, src, dst)\
	movq(mem(reg, src), r15)\
	movss(mem(r15, offset, 1), dst)
#define LOAD_2x1_FP32(reg, offset, src1, src2, dst1, dst2)\
	movq(mem(reg, src1), r14)\
	movq(mem(reg, src2), r15)\
	movss(mem(r14, offset, 1), dst1)\
	movss(mem(r15, offset, 1), dst2)

#define LOAD_INPUT_4x4xFP32(offset)\
	LOAD_2x4_FP32(rax, offset, 0*4*8, 1*4*8, xmm0, xmm1)\
	LOAD_2x4_FP32(rax, offset, 2*4*8, 3*4*8, xmm2, xmm3)
#define LOAD_INPUT_5x4xFP32(offset)\
	LOAD_2x4_FP32(rax, offset, 0*5*8, 1*5*8, xmm0, xmm1)\
	LOAD_2x4_FP32(rax, offset, 2*5*8, 3*5*8, xmm2, xmm3)\
	LOAD_1x4_FP32(rax, offset, 4*5*8, xmm4)
#define LOAD_INPUT_6x4xFP32(offset)\
	LOAD_2x4_FP32(rax, offset, 0*6*8, 1*6*8, xmm0, xmm1)\
	LOAD_2x4_FP32(rax, offset, 2*6*8, 3*6*8, xmm2, xmm3)\
	LOAD_2x4_FP32(rax, offset, 4*6*8, 5*6*8, xmm4, xmm5)
#define LOAD_INPUT_7x4xFP32(offset)\
	LOAD_2x4_FP32(rax, offset, 0*7*8, 1*7*8, xmm0, xmm1)\
	LOAD_2x4_FP32(rax, offset, 2*7*8, 3*7*8, xmm2, xmm3)\
	LOAD_2x4_FP32(rax, offset, 4*7*8, 5*7*8, xmm4, xmm5)\
	LOAD_1x4_FP32(rax, offset, 6*7*8, xmm6)

#define LOAD_INPUT_4x1xFP32(offset)\
	LOAD_2x1_FP32(rax, offset, 0*4*8, 1*4*8, xmm0, xmm1)\
	LOAD_2x1_FP32(rax, offset, 2*4*8, 3*4*8, xmm2, xmm3)
#define LOAD_INPUT_5x1xFP32(offset)\
	LOAD_2x1_FP32(rax, offset, 0*5*8, 1*5*8, xmm0, xmm1)\
	LOAD_2x1_FP32(rax, offset, 2*5*8, 3*5*8, xmm2, xmm3)\
	LOAD_1x1_FP32(rax, offset, 4*5*8, xmm4)
#define LOAD_INPUT_6x1xFP32(offset)\
	LOAD_2x1_FP32(rax, offset, 0*6*8, 1*6*8, xmm0, xmm1)\
	LOAD_2x1_FP32(rax, offset, 2*6*8, 3*6*8, xmm2, xmm3)\
	LOAD_2x1_FP32(rax, offset, 4*6*8, 5*6*8, xmm4, xmm5)
#define LOAD_INPUT_7x1xFP32(offset)\
	LOAD_2x1_FP32(rax, offset, 0*7*8, 1*7*8, xmm0, xmm1)\
	LOAD_2x1_FP32(rax, offset, 2*7*8, 3*7*8, xmm2, xmm3)\
	LOAD_2x1_FP32(rax, offset, 4*7*8, 5*7*8, xmm4, xmm5)\
	LOAD_1x1_FP32(rax, offset, 6*7*8, xmm6)

#define STORE_WORKSPACE_1x1xFP32(reg, row, columns) movss(reg, mem(rbx, row*columns*1*4))
#define STORE_WORKSPACE_1x4xFP32(reg, row, columns) movaps(reg, mem(rbx, row*columns*4*4))

#define STORE_OUTPUT_1x1xFP32(offset, reg, dst) movss(reg, mem(dst, offset, 1))
#define STORE_OUTPUT_1x4xFP32(offset, reg, dst) movups(reg, mem(dst, offset, 1))

#define LOAD_WORKSPACE_4x4xFP32()\
	movaps(mem(rbx, 0*4*4), xmm0)\
	movaps(mem(rbx, 1*4*4), xmm1)\
	movaps(mem(rbx, 2*4*4), xmm2)\
	movaps(mem(rbx, 3*4*4), xmm3)
#define LOAD_WORKSPACE_5x4xFP32()\
	LOAD_WORKSPACE_4x4xFP32()\
	movaps(mem(rbx, 4*4*4), xmm4)
#define LOAD_WORKSPACE_6x4xFP32()\
	LOAD_WORKSPACE_5x4xFP32()\
	movaps(mem(rbx, 5*4*4), xmm5)
#define LOAD_WORKSPACE_7x4xFP32()\
	LOAD_WORKSPACE_6x4xFP32()\
	movaps(mem(rbx, 6*4*4), xmm6)

#define LOAD_WORKSPACE_4x1xFP32()\
	movss(mem(rbx, 0*1*4), xmm0)\
	movss(mem(rbx, 1*1*4), xmm1)\
	movss(mem(rbx, 2*1*4), xmm2)\
	movss(mem(rbx, 3*1*4), xmm3)
#define LOAD_WORKSPACE_5x1xFP32()\
	LOAD_WORKSPACE_4x1xFP32()\
	movss(mem(rbx, 4*1*4), xmm4)
#define LOAD_WORKSPACE_6x1xFP32()\
	LOAD_WORKSPACE_5x1xFP32()\
	movss(mem(rbx, 5*1*4), xmm5)
#define LOAD_WORKSPACE_7x1xFP32()\
	LOAD_WORKSPACE_6x1xFP32()\
	movss(mem(rbx, 6*1*4), xmm6)

#define ADD_BIAS_4x4xFP32(reg)\
	addps(reg, xmm0)\
	addps(reg, xmm1)\
	addps(reg, xmm2)\
	addps(reg, xmm3)
#define ADD_BIAS_5x4xFP32(reg)\
	addps(reg, xmm0)\
	addps(reg, xmm1)\
	addps(reg, xmm2)\
	addps(reg, xmm3)\
	addps(reg, xmm4)

#define LOAD_EXT_4x1xFP32(offset)\
	LOAD_2x1_FP32(rdx, offset, 0*8, 1*8, xmm4, xmm5)\
	LOAD_2x1_FP32(rdx, offset, 2*8, 3*8, xmm6, xmm7)
#define LOAD_EXT_5x1xFP32(offset)\
	LOAD_2x1_FP32(rdx, offset, 0*8, 1*8, xmm5, xmm6)\
	LOAD_2x1_FP32(rdx, offset, 2*8, 3*8, xmm7, xmm8)\
	LOAD_1x1_FP32(rdx, offset, 4*8, xmm9)
#define LOAD_EXT_4x4xFP32(offset)\
	LOAD_2x4_FP32(rdx, offset, 0*8, 1*8, xmm4, xmm5)\
	LOAD_2x4_FP32(rdx, offset, 2*8, 3*8, xmm6, xmm7)
#define LOAD_EXT_5x4xFP32(offset)\
	LOAD_2x4_FP32(rdx, offset, 0*8, 1*8, xmm5, xmm6)\
	LOAD_2x4_FP32(rdx, offset, 2*8, 3*8, xmm7, xmm8)\
	LOAD_1x4_FP32(rdx, offset, 4*8, xmm9)

#define ADD_EXT_4x4xFP32()\
	addps(xmm4, xmm0)\
	addps(xmm5, xmm1)\
	addps(xmm6, xmm2)\
	addps(xmm7, xmm3)
#define ADD_EXT_5x4xFP32()\
	addps(xmm5, xmm0)\
	addps(xmm6, xmm1)\
	addps(xmm7, xmm2)\
	addps(xmm8, xmm3)\
	addps(xmm9, xmm4)

#define APPLY_RELU_4x4xFP32(tmp)\
	xorps(tmp, tmp)\
	maxps(tmp, xmm0)\
	maxps(tmp, xmm1)\
	maxps(tmp, xmm2)\
	maxps(tmp, xmm3)
#define APPLY_RELU_5x4xFP32(tmp)\
	xorps(tmp, tmp)\
	maxps(tmp, xmm0)\
	maxps(tmp, xmm1)\
	maxps(tmp, xmm2)\
	maxps(tmp, xmm3)\
	maxps(tmp, xmm4)

#define INPUT_TRANSFORM_4x4_3x3_ROW_0()\
	movaps(xmm0, xmm7)\
	movaps(xmm4, xmm8)\
	subps(xmm2, xmm7)\
	subps(xmm2, xmm8)\
	mulps(xmm14, xmm8)\
	addps(xmm8, xmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_1()\
	movaps(xmm2, xmm7)\
	movaps(xmm4, xmm8)\
	addps(xmm1, xmm7)\
	addps(xmm3, xmm8)\
	mulps(xmm14, xmm8)\
	subps(xmm8, xmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_2()\
	movaps(xmm2, xmm7)\
	movaps(xmm3, xmm8)\
	subps(xmm1, xmm7)\
	subps(xmm4, xmm8)\
	mulps(xmm14, xmm8)\
	addps(xmm8, xmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_3()\
	movaps(xmm3, xmm7)\
	movaps(xmm4, xmm8)\
	subps(xmm1, xmm7)\
	subps(xmm2, xmm8)\
	mulps(xmm15, xmm8)\
	addps(xmm8, xmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_4()\
	movaps(xmm1, xmm7)\
	movaps(xmm4, xmm8)\
	subps(xmm3, xmm7)\
	subps(xmm2, xmm8)\
	mulps(xmm15, xmm8)\
	addps(xmm8, xmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_5()\
	movaps(xmm1, xmm7)\
	movaps(xmm5, xmm8)\
	subps(xmm3, xmm7)\
	subps(xmm3, xmm8)\
	mulps(xmm14, xmm8)\
	addps(xmm8, xmm7)

#define OUTPUT_TRANSFORM_4x4_3x3()\
	movaps(xmm2, xmm7)\
	movaps(xmm1, xmm8)\
	movaps(xmm4, xmm9)\
	movaps(xmm3, xmm10)\
	addps(xmm1, xmm7)\
	subps(xmm2, xmm8)\
	addps(xmm3, xmm9)\
	subps(xmm4, xmm10)\
	movaps(xmm9, xmm1)\
	movaps(xmm10, xmm2)\
	movaps(xmm10, xmm3)\
	mulps(xmm13, xmm1)\
	mulps(xmm14, xmm2)\
	mulps(xmm15, xmm3)\
	mulps(xmm15, xmm5)\
	addps(xmm7, xmm0)\
	addps(xmm1, xmm0)\
	movaps(xmm2, xmm1)\
	movaps(xmm9, xmm2)\
	addps(xmm8, xmm1)\
	addps(xmm7, xmm2)\
	addps(xmm8, xmm3)\
	addps(xmm5, xmm3)

#define INPUT_TRANSFORM_5x5_3x3_ROW_0()\
	movaps(xmm3, xmm7)\
	movaps(xmm3, xmm8)\
	movaps(xmm4, xmm9)\
	movaps(xmm0, xmm10)\
	subps(xmm1, xmm7)\
	subps(xmm5, xmm8)\
	subps(xmm2, xmm9)\
	subps(xmm2, xmm10)\
	addps(xmm7, xmm7)\
	mulps(xmm15, xmm8)\
	mulps(xmm14, xmm9)\
	addps(xmm8, xmm10)\
	addps(xmm9, xmm7)\
	addps(xmm10, xmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_1()\
	movaps(xmm2, xmm7)\
	movaps(xmm3, xmm8)\
	movaps(xmm3, xmm9)\
	subps(xmm1, xmm7)\
	subps(xmm4, xmm8)\
	addps(xmm3, xmm9)\
	mulps(xmm14, xmm8)\
	movaps(xmm5, xmm10)\
	mulps(xmm15, xmm10)\
	addps(xmm8, xmm7)\
	subps(xmm10, xmm9)\
	addps(xmm9, xmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_2()\
	movaps(xmm2, xmm7)\
	movaps(xmm4, xmm8)\
	movaps(xmm2, xmm9)\
	subps(xmm1, xmm7)\
	addps(xmm3, xmm8)\
	subps(xmm3, xmm9)\
	movaps(xmm8, xmm10)\
	movaps(xmm5, xmm11)\
	mulps(xmm14, xmm10)\
	mulps(xmm15, xmm11)\
	subps(xmm8, xmm10)\
	addps(xmm11, xmm9)\
	addps(xmm10, xmm7)\
	addps(xmm2, xmm9)\
	addps(xmm9, xmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_3()\
	movaps(xmm5, xmm7)\
	movaps(xmm2, xmm8)\
	addps(xmm1, xmm7)\
	subps(xmm4, xmm8)\
	mulps(xmm15, xmm7)\
	movaps(xmm8, xmm9)\
	mulps(xmm14, xmm9)\
	subps(xmm3, xmm7)\
	subps(xmm8, xmm9)\
	addps(xmm9, xmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_4()\
	movaps(xmm1, xmm7)\
	movaps(xmm4, xmm8)\
	subps(xmm5, xmm7)\
	subps(xmm2, xmm8)\
	mulps(xmm15, xmm7)\
	movaps(xmm8, xmm9)\
	mulps(xmm14, xmm9)\
	addps(xmm9, xmm8)\
	addps(xmm8, xmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_5()\
	movaps(xmm1, xmm7)\
	movaps(xmm5, xmm8)\
	subps(xmm3, xmm7)\
	subps(xmm3, xmm8)\
	mulps(xmm14, xmm8)\
	addps(xmm8, xmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_6()\
	movaps(xmm3, xmm7)\
	movaps(xmm2, xmm8)\
	movaps(xmm3, xmm9)\
	movaps(xmm6, xmm10)\
	subps(xmm1, xmm7)\
	subps(xmm4, xmm8)\
	subps(xmm5, xmm9)\
	subps(xmm4, xmm10)\
	addps(xmm8, xmm8)\
	mulps(xmm14, xmm9)\
	mulps(xmm15, xmm10)\
	addps(xmm9, xmm7)\
	addps(xmm10, xmm8)\
	addps(xmm8, xmm7)

#define OUTPUT_TRANSFORM_5x5_3x3()\
	movaps(xmm2, xmm7)\
	movaps(xmm1, xmm8)\
	movaps(xmm4, xmm9)\
	movaps(xmm3, xmm10)\
	addps(xmm1, xmm7)\
	subps(xmm2, xmm8)\
	addps(xmm3, xmm9)\
	subps(xmm4, xmm10)\
	addps(xmm7, xmm0)\
	movaps(xmm9, xmm1)\
	movaps(xmm10, xmm2)\
	movaps(xmm10, xmm3)\
	movaps(xmm9, xmm4)\
	mulps(xmm12, xmm1)\
	mulps(xmm13, xmm2)\
	mulps(xmm14, xmm3)\
	mulps(xmm15, xmm4)\
	addps(xmm1, xmm0)\
	movaps(xmm2, xmm1)\
	addps(xmm8, xmm1)\
	movaps(xmm9, xmm2)\
	addps(xmm7, xmm2)\
	addps(xmm8, xmm3)\
	addps(xmm7, xmm4)\
	movaps(xmm5, xmm7)\
	movaps(xmm5, xmm8)\
	movaps(xmm5, xmm9)\
	movaps(xmm5, xmm10)\
	mulps(xmm15, xmm7)\
	mulps(xmm14, xmm8)\
	mulps(xmm13, xmm9)\
	mulps(xmm12, xmm10)\
	addps(xmm7, xmm0)\
	addps(xmm8, xmm1)\
	addps(xmm5, xmm2)\
	addps(xmm9, xmm3)\
	addps(xmm10, xmm4)\
	addps(xmm6, xmm4)
}

namespace ml
{
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP32
	 */
	void winograd_input_transform_4x4_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 4;
		const uint64_t k_left = filters % 4;

		const float constants[2] = { 0.25f, 0.5f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		movss(mem(r8, 0), xmm14)// 0.25f
		movss(mem(r8, 4), xmm15)// 0.5f
		pshufd(imm(0), xmm14, xmm14)
		pshufd(imm(0), xmm15, xmm15)

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 4-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED4)// main loop over channels, in steps of 4 elements

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x4)
		LOAD_INPUT_6x4xFP32(r9)

		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_WORKSPACE_1x4xFP32(xmm7, 0, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_WORKSPACE_1x4xFP32(xmm7, 1, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_WORKSPACE_1x4xFP32(xmm7, 2, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_WORKSPACE_1x4xFP32(xmm7, 3, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_WORKSPACE_1x4xFP32(xmm7, 4, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_WORKSPACE_1x4xFP32(xmm7, 5, 6)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rax (src), moving to next column
		add(imm(1*4*4), rbx)// add 4*4 (4 floats) to rbx (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x4)
		sub(imm(6*1*8), rax)// subtract 6*1*8 (6*1 pointer) to rax (src), moving to start
		sub(imm(6*1*4*4), rbx)// subtract 6*4*4 (6*4 floats) to rbx (workspace), moving to start

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM2x4)
		// second transform
		LOAD_WORKSPACE_6x4xFP32()

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r15)

		movq(mem(rcx, 3*8), r13)
		movq(mem(rcx, 4*8), r14)
		movq(mem(rcx, 5*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r15)

		add(imm(6*4*4), rbx)// add 7*4 (7*4 floats) to rbx (workspace), moving to next row
		add(imm(6*8), rcx)// add 7*8*4 (7 pointers) to rcx (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x4)
		sub(imm(6*6*4*4), rbx)// subtract 6*6*4 (6*6*4 floats) to rbx (workspace), moving to start
		sub(imm(6*6*8), rcx)// subtract 6*6*8*4 (6*6 pointers) to rcx (dst), moving to start

		add(imm(4*4), r9)// add 4*4 (4 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED4)

		/* END OF MAIN 4-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x1)

		LOAD_INPUT_6x1xFP32(r9)

		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_WORKSPACE_1x1xFP32(xmm7, 0, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_WORKSPACE_1x1xFP32(xmm7, 1, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_WORKSPACE_1x1xFP32(xmm7, 2, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_WORKSPACE_1x1xFP32(xmm7, 3, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_WORKSPACE_1x1xFP32(xmm7, 4, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_WORKSPACE_1x1xFP32(xmm7, 5, 6)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*1*4), rbx)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)
		sub(imm(6*1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		sub(imm(6*1*1*4), rbx)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_6x1xFP32()

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r15)

		movq(mem(rcx, 3*8), r13)
		movq(mem(rcx, 4*8), r14)
		movq(mem(rcx, 5*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r15)

		add(imm(6*1*4), rbx)// add 7*8 (7*1 floats) to rsi (workspace), moving to next row
		add(imm(6*8), rcx)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)
		sub(imm(6*6*1*4), rbx)// add 7*8 (7*1 floats) to rsi (workspace), moving to next row
		sub(imm(6*6*8), rcx)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		add(imm(1*4), r9)// add 1*4 (1 float32) to r9, the offset in channels
		dec(r10)
		jne(UNROLLED1)

		label(EPILOGUE)

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
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12",
				"%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%r8", "%r9", "%r10", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_4x4_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		const void **ext_ptr = ext;
		const void *bias_ptr = bias;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 4;
		const uint64_t k_left = filters % 4;
		const uint64_t flag_relu = static_cast<uint64_t>(use_relu);

		const float constants[3] = { 0.25f, 0.5f, 2.0f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		movss(mem(r8, 0), xmm13)// 0.25f
		movss(mem(r8, 4), xmm14)// 0.5f
		movss(mem(r8, 8), xmm15)// 2.0f
		pshufd(imm(0), xmm13, xmm13)
		pshufd(imm(0), xmm14, xmm14)
		pshufd(imm(0), xmm15, xmm15)

		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)
		movq(var(bias_ptr), rdi)// bias pointer
		movq(var(ext_ptr), rdx)// external data pointer
		movq(rdx, r11)// holding ext flag
		movq(var(flag_relu), r12)// holding flag whether to apply ReLU

		movq(imm(0), r9)// channel offset
		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED4)// main loop over channels, in steps of 8 elements

		xorps(xmm11, xmm11)
		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x4)
		movups(mem(rdi, r9, 1), xmm11)// load bias
		label(SKIP_BIAS_LOAD_x4)

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x4)
		LOAD_INPUT_6x4xFP32(r9)// load column
		OUTPUT_TRANSFORM_4x4_3x3()
		STORE_WORKSPACE_1x4xFP32(xmm0, 0, 6)
		STORE_WORKSPACE_1x4xFP32(xmm1, 1, 6)
		STORE_WORKSPACE_1x4xFP32(xmm2, 2, 6)
		STORE_WORKSPACE_1x4xFP32(xmm3, 3, 6)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*4*4), rbx)// add 4*4 (4 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x4)
		sub(imm(6*1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		sub(imm(6*1*4*4), rbx)// add 4*4 (4 floats) to rdi (workspace), moving to next column

		movq(imm(4), r8)// transform row counter
		label(TRANSFORM2x4)
		// second transform
		LOAD_WORKSPACE_6x4xFP32()// load row
		OUTPUT_TRANSFORM_4x4_3x3()
		ADD_BIAS_4x4xFP32(xmm11)

		test(r11, r11)
		je(SKIP_LOAD_EXT_x4)
		LOAD_EXT_4x4xFP32(r9)
		ADD_EXT_4x4xFP32()
		label(SKIP_LOAD_EXT_x4)

		test(r12, r12)
		je(SKIP_RELU_x4)
		APPLY_RELU_4x4xFP32(xmm10)
		label(SKIP_RELU_x4)

		movq(mem(rcx, 0*8), r14)
		movq(mem(rcx, 1*8), r15)
		STORE_OUTPUT_1x4xFP32(r9, xmm0, r14)
		STORE_OUTPUT_1x4xFP32(r9, xmm1, r15)
		movq(mem(rcx, 2*8), r14)
		movq(mem(rcx, 3*8), r15)
		STORE_OUTPUT_1x4xFP32(r9, xmm2, r14)
		STORE_OUTPUT_1x4xFP32(r9, xmm3, r15)

		add(imm(6*4*4), rbx)// add 6*4 (6*4 floats) to rbx (workspace), moving to next row
		add(imm(4*8), rcx)// add 4*8 (4 pointers) to rcx (dst), moving to next row
		add(imm(4*8), rdx)// add 4*8 (4 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x4)
		sub(imm(4*6*4*4), rbx)// add 6*4 (6*4 floats) to rbx (workspace), moving to next row
		sub(imm(4*4*8), rcx)// add 4*8 (4 pointers) to rcx (dst), moving to next row
		sub(imm(4*4*8), rdx)// add 4*8 (4 pointers) to rdx (ext), moving to next row

		add(imm(4*4), r9)// add 8*4 (8 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED4)

		/* END OF MAIN 4-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x1)
		movss(mem(rdi, r9, 1), xmm11)// load bias
		label(SKIP_BIAS_LOAD_x1)

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x1)

		LOAD_INPUT_6x1xFP32(r9)
		OUTPUT_TRANSFORM_4x4_3x3()
		STORE_WORKSPACE_1x1xFP32(xmm0, 0, 6)
		STORE_WORKSPACE_1x1xFP32(xmm1, 1, 6)
		STORE_WORKSPACE_1x1xFP32(xmm2, 2, 6)
		STORE_WORKSPACE_1x1xFP32(xmm3, 3, 6)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*1*4), rbx)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)
		sub(imm(6*1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		sub(imm(6*1*1*4), rbx)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		movq(imm(4), r8)// transform row counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_6x1xFP32()// load row
		OUTPUT_TRANSFORM_4x4_3x3()
		ADD_BIAS_4x4xFP32(xmm11)

		test(r11, r11)// check if external tensor pointer is not null
		je(SKIP_LOAD_EXT_x1)
		LOAD_EXT_4x1xFP32(r9)
		ADD_EXT_4x4xFP32()
		label(SKIP_LOAD_EXT_x1)

		test(r12, r12)// check if ReLU should be applied
		je(SKIP_RELU_x1)
		APPLY_RELU_4x4xFP32(xmm10)
		label(SKIP_RELU_x1)

		movq(mem(rcx, 0*8), r14)
		movq(mem(rcx, 1*8), r15)
		STORE_OUTPUT_1x1xFP32(r9, xmm0, r14)
		STORE_OUTPUT_1x1xFP32(r9, xmm1, r15)
		movq(mem(rcx, 2*8), r14)
		movq(mem(rcx, 3*8), r15)
		STORE_OUTPUT_1x1xFP32(r9, xmm2, r14)
		STORE_OUTPUT_1x1xFP32(r9, xmm3, r15)

		add(imm(6*1*4), rbx)// add 7*8 (7*1 floats) to rbx (workspace), moving to next row
		add(imm(4*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		add(imm(4*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)
		sub(imm(4*6*1*4), rbx)// add 7*8 (7*1 floats) to rbx (workspace), moving to next row
		sub(imm(4*4*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		sub(imm(4*4*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		add(imm(1*4), r9)// add 1*4 (1 float32) to r9, the offset in channels
		dec(r10)
		jne(UNROLLED1)

		label(EPILOGUE)

		end_asm(
				:// outputs
				:// inputs
				[src_ptr] "m"(src_ptr),
				[dst_ptr] "m"(dst_ptr),
				[workspace_ptr] "m"(workspace_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[c_ptr] "m"(c_ptr),
				[ext_ptr] "m"(ext_ptr),
				[bias_ptr] "m"(bias_ptr),
				[flag_relu] "m"(flag_relu)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12",
				"%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
	}

	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_input_transform_5x5_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 4;
		const uint64_t k_left = filters % 4;

		const float constants[2] = { 0.25f, 0.5f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		movss(mem(r8, 0), xmm14)// 0.25f
		movss(mem(r8, 4), xmm15)// 0.5f
		pshufd(imm(0), xmm14, xmm14)
		pshufd(imm(0), xmm15, xmm15)

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 4-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED4)// main loop over channels, in steps of 4 elements

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x4)
		LOAD_INPUT_7x4xFP32(r9)

		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_WORKSPACE_1x4xFP32(xmm7, 0, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_WORKSPACE_1x4xFP32(xmm7, 1, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_WORKSPACE_1x4xFP32(xmm7, 2, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_WORKSPACE_1x4xFP32(xmm7, 3, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_WORKSPACE_1x4xFP32(xmm7, 4, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_WORKSPACE_1x4xFP32(xmm7, 5, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_WORKSPACE_1x4xFP32(xmm7, 6, 7)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rax (src), moving to next column
		add(imm(1*4*4), rbx)// add 4*4 (4 floats) to rbx (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x4)
		sub(imm(7*8), rax)// subtract 7*8 (7 pointers) from rax, moving to the start
		sub(imm(7*4*4), rbx)// subtract 7*4*4 (7*4 float32) from rbx, moving to the start

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x4)
		// second transform
		LOAD_WORKSPACE_7x4xFP32()

		movq(mem(rcx, 0*8), r12)
		movq(mem(rcx, 1*8), r13)
		movq(mem(rcx, 2*8), r14)
		movq(mem(rcx, 3*8), r15)
		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r15)

		movq(mem(rcx, 4*8), r12)
		movq(mem(rcx, 5*8), r13)
		movq(mem(rcx, 6*8), r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_OUTPUT_1x4xFP32(r9, xmm7, r14)

		add(imm(7*4*4), rbx)// add 7*4 (7*4 floats) to rbx (workspace), moving to next row
		add(imm(7*8), rcx)// add 7*8*4 (7 pointers) to rcx (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x4)
		sub(imm(7*7*4*4), rbx)// subtract 7*7*4*4 (7*7*4 float32) from rbx, moving to the start
		sub(imm(7*7*8), rcx)// subtract 7*8*4 (7*7*8 pointers) from rcx, moving to the start

		add(imm(4*4), r9)// add 8*4 (8 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED4)

		/* END OF MAIN 4-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x1)

		LOAD_INPUT_7x1xFP32(r9)

		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_WORKSPACE_1x1xFP32(xmm7, 0, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_WORKSPACE_1x1xFP32(xmm7, 1, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_WORKSPACE_1x1xFP32(xmm7, 2, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_WORKSPACE_1x1xFP32(xmm7, 3, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_WORKSPACE_1x1xFP32(xmm7, 4, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_WORKSPACE_1x1xFP32(xmm7, 5, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_WORKSPACE_1x1xFP32(xmm7, 6, 7)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*1*4), rbx)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)
		sub(imm(7*8), rax)// subtract 7*8 (7 pointers) from rax, moving to the start
		sub(imm(7*1*4), rbx)// subtract 7*1*4 (7*1 float32) from rbx, moving to the start

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_7x1xFP32()

		movq(mem(rcx, 0*8), r12)
		movq(mem(rcx, 1*8), r13)
		movq(mem(rcx, 2*8), r14)
		movq(mem(rcx, 3*8), r15)
		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r15)

		movq(mem(rcx, 4*8), r12)
		movq(mem(rcx, 5*8), r13)
		movq(mem(rcx, 6*8), r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r14)

		add(imm(7*1*4), rbx)// add 7*8 (7*1 floats) to rsi (workspace), moving to next row
		add(imm(7*8), rcx)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)
		sub(imm(7*7*1*4), rbx)// subtract 7*7*1*4 (7*7*1 float32) from rbx, moving to the start
		sub(imm(7*7*8), rcx)// subtract 7*8*4 (7*7*8 pointers) from rcx, moving to the start

		add(imm(1*4), r9)// add 1*4 (1 float32) to r9, the offset in channels
		dec(r10)
		jne(UNROLLED1)

		label(EPILOGUE)

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
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12",
				"%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%r8", "%r9", "%r10", "%r12", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_5x5_3x3_sse2_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		const void **ext_ptr = ext;
		const void *bias_ptr = bias;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 4;
		const uint64_t k_left = filters % 4;
		const uint64_t flag_relu = static_cast<uint64_t>(use_relu);

		const float constants[4] = { 0.25f, 0.5f, 2.0f, 4.0f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		movss(mem(r8, 0), xmm12)// 0.25f
		movss(mem(r8, 4), xmm13)// 0.5f
		movss(mem(r8, 8), xmm14)// 2.0f
		movss(mem(r8, 12), xmm15)// 4.0f
		pshufd(imm(0), xmm12, xmm12)
		pshufd(imm(0), xmm13, xmm13)
		pshufd(imm(0), xmm14, xmm14)
		pshufd(imm(0), xmm15, xmm15)

		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)
		movq(var(bias_ptr), rdi)// bias pointer
		movq(var(ext_ptr), rdx)// external data pointer
		movq(rdx, r11)// holding ext flag
		movq(var(flag_relu), r12)// holding flag whether to apply ReLU

		movq(imm(0), r9)// channel offset
		movq(var(k_iter), r10)// load the number of 4-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED4)// main loop over channels, in steps of 4 elements

		xorps(xmm11, xmm11)
		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x4)
		movups(mem(rdi, r9, 1), xmm11)// load bias
		label(SKIP_BIAS_LOAD_x4)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x4)
		LOAD_INPUT_7x4xFP32(r9)// load column
		OUTPUT_TRANSFORM_5x5_3x3()
		STORE_WORKSPACE_1x4xFP32(xmm0, 0, 7)
		STORE_WORKSPACE_1x4xFP32(xmm1, 1, 7)
		STORE_WORKSPACE_1x4xFP32(xmm2, 2, 7)
		STORE_WORKSPACE_1x4xFP32(xmm3, 3, 7)
		STORE_WORKSPACE_1x4xFP32(xmm4, 4, 7)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*4*4), rbx)// add 4*4 (4 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x4)
		sub(imm(7*1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		sub(imm(7*1*4*4), rbx)// add 4*4 (4 floats) to rdi (workspace), moving to next column

		movq(imm(5), r8)// transform row counter
		label(TRANSFORM2x4)
		// second transform
		LOAD_WORKSPACE_7x4xFP32()// load row
		OUTPUT_TRANSFORM_5x5_3x3()
		ADD_BIAS_5x4xFP32(xmm11)

		test(r11, r11)
		je(SKIP_LOAD_EXT_x4)
		LOAD_EXT_5x4xFP32(r9)
		ADD_EXT_5x4xFP32()
		label(SKIP_LOAD_EXT_x4)

		test(r12, r12)
		je(SKIP_RELU_x4)
		APPLY_RELU_5x4xFP32(xmm10)
		label(SKIP_RELU_x4)

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		STORE_OUTPUT_1x4xFP32(r9, xmm0, r13)
		STORE_OUTPUT_1x4xFP32(r9, xmm1, r14)
		STORE_OUTPUT_1x4xFP32(r9, xmm2, r15)
		movq(mem(rcx, 3*8), r14)
		movq(mem(rcx, 4*8), r15)
		STORE_OUTPUT_1x4xFP32(r9, xmm3, r14)
		STORE_OUTPUT_1x4xFP32(r9, xmm4, r15)

		add(imm(7*4*4), rbx)// add 7*4 (7*4 floats) to rbx (workspace), moving to next row
		add(imm(5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		add(imm(5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x4)
		sub(imm(5*7*4*4), rbx)// add 7*4 (7*4 floats) to rbx (workspace), moving to next row
		sub(imm(5*5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		sub(imm(5*5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		add(imm(4*4), r9)// add 4*4 (4 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED4)

		/* END OF MAIN 4-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x1)
		movss(mem(rdi, r9, 1), xmm11)// load bias
		label(SKIP_BIAS_LOAD_x1)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x1)

		LOAD_INPUT_7x1xFP32(r9)
		OUTPUT_TRANSFORM_5x5_3x3()
		STORE_WORKSPACE_1x1xFP32(xmm0, 0, 7)
		STORE_WORKSPACE_1x1xFP32(xmm1, 1, 7)
		STORE_WORKSPACE_1x1xFP32(xmm2, 2, 7)
		STORE_WORKSPACE_1x1xFP32(xmm3, 3, 7)
		STORE_WORKSPACE_1x1xFP32(xmm4, 4, 7)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*1*4), rbx)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)
		sub(imm(7*1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		sub(imm(7*1*1*4), rbx)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		movq(imm(5), r8)// transform row counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_7x1xFP32()// load row
		OUTPUT_TRANSFORM_5x5_3x3()
		ADD_BIAS_5x4xFP32(xmm11)

		test(r11, r11)// check if external tensor pointer is not null
		je(SKIP_LOAD_EXT_x1)
		LOAD_EXT_5x1xFP32(r9)
		ADD_EXT_5x4xFP32()
		label(SKIP_LOAD_EXT_x1)

		test(r12, r12)// check if ReLU should be applied
		je(SKIP_RELU_x1)
		APPLY_RELU_5x4xFP32(xmm10)
		label(SKIP_RELU_x1)

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		STORE_OUTPUT_1x1xFP32(r9, xmm0, r13)
		STORE_OUTPUT_1x1xFP32(r9, xmm1, r14)
		STORE_OUTPUT_1x1xFP32(r9, xmm2, r15)
		movq(mem(rcx, 3*8), r13)
		movq(mem(rcx, 4*8), r14)
		STORE_OUTPUT_1x1xFP32(r9, xmm3, r13)
		STORE_OUTPUT_1x1xFP32(r9, xmm4, r14)

		add(imm(7*1*4), rbx)// add 7*8 (7*1 floats) to rbx (workspace), moving to next row
		add(imm(5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		add(imm(5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)
		sub(imm(5*7*1*4), rbx)// add 7*8 (7*1 floats) to rbx (workspace), moving to next row
		sub(imm(5*5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		sub(imm(5*5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		add(imm(1*4), r9)// add 1*4 (1 float32) to r9, the offset in channels
		dec(r10)
		jne(UNROLLED1)

		label(EPILOGUE)

		end_asm(
				:// outputs
				:// inputs
				[src_ptr] "m"(src_ptr),
				[dst_ptr] "m"(dst_ptr),
				[workspace_ptr] "m"(workspace_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[c_ptr] "m"(c_ptr),
				[ext_ptr] "m"(ext_ptr),
				[bias_ptr] "m"(bias_ptr),
				[flag_relu] "m"(flag_relu)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12",
				"%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx", "%rcx", "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
	}

} /* namespace ml */
