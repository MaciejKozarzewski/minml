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
#define LOAD_INPUT_1x8_FP32(offset, src, dst)\
	movq(mem(rsi, src), r15)\
	vmovups(mem(r15, offset, 1), dst)
#define LOAD_INPUT_2x8_FP32(offset, src1, src2, dst1, dst2)\
	movq(mem(rsi, src1), r14)\
	movq(mem(rsi, src2), r15)\
	vmovups(mem(r14, offset, 1), dst1)\
	vmovups(mem(r15, offset, 1), dst2)
#define LOAD_INPUT_1x1_FP32(offset, src, dst)\
	movq(mem(rsi, src), r15)\
	vmovss(mem(r15, offset, 1), dst)
#define LOAD_INPUT_2x1_FP32(offset, src1, src2, dst1, dst2)\
	movq(mem(rsi, src1), r14)\
	movq(mem(rsi, src2), r15)\
	vmovss(mem(r14, offset, 1), dst1)\
	vmovss(mem(r15, offset, 1), dst2)

#define LOAD_INPUT_1x8_FP16(offset, src, dst)\
	movq(mem(rsi, src), r15)\
	vmovups(mem(r15, offset, 1), xmm(dst))\
	vcvtph2ps(xmm(dst), ymm(dst))
#define LOAD_INPUT_2x8_FP16(offset, src1, src2, dst1, dst2)\
	movq(mem(rsi, src1), r14)\
	movq(mem(rsi, src2), r15)\
	vmovups(mem(r14, offset, 1), xmm(dst1))\
	vmovups(mem(r15, offset, 1), xmm(dst2))\
	vcvtph2ps(xmm(dst1), ymm(dst1))\
	vcvtph2ps(xmm(dst2), ymm(dst2))
#define LOAD_INPUT_1x1_FP16(offset, src, dst)\
	movq(mem(rsi, src), r15)\
	movzw(mem(r15, offset, 1), r15)\
	vmovq(r15, xmm(dst))\
	vcvtph2ps(xmm(dst), xmm(dst))
#define LOAD_INPUT_2x1_FP16(offset, src1, src2, dst1, dst2)\
	movq(mem(rsi, src1), r14)\
	movq(mem(rsi, src2), r15)\
	movzw(mem(r14, offset, 1), r14)\
	movzw(mem(r15, offset, 1), r15)\
	vmovq(r14, xmm(dst1))\
	vmovq(r15, xmm(dst2))\
	vcvtph2ps(xmm(dst1), xmm(dst1))\
	vcvtph2ps(xmm(dst2), xmm(dst2))

#define LOAD_INPUT_4x8xFP32(offset)\
	LOAD_INPUT_2x8_FP32(offset, 0*4*8, 1*4*8, ymm0, ymm1)\
	LOAD_INPUT_2x8_FP32(offset, 2*4*8, 3*4*8, ymm2, ymm3)
#define LOAD_INPUT_5x8xFP32(offset)\
	LOAD_INPUT_2x8_FP32(offset, 0*5*8, 1*5*8, ymm0, ymm1)\
	LOAD_INPUT_2x8_FP32(offset, 2*5*8, 3*5*8, ymm2, ymm3)\
	LOAD_INPUT_1x8_FP32(offset, 4*5*8, ymm4)
#define LOAD_INPUT_6x8xFP32(offset)\
	LOAD_INPUT_2x8_FP32(offset, 0*6*8, 1*6*8, ymm0, ymm1)\
	LOAD_INPUT_2x8_FP32(offset, 2*6*8, 3*6*8, ymm2, ymm3)\
	LOAD_INPUT_2x8_FP32(offset, 4*6*8, 5*6*8, ymm4, ymm5)
#define LOAD_INPUT_7x8xFP32(offset)\
	LOAD_INPUT_2x8_FP32(offset, 0*7*8, 1*7*8, ymm0, ymm1)\
	LOAD_INPUT_2x8_FP32(offset, 2*7*8, 3*7*8, ymm2, ymm3)\
	LOAD_INPUT_2x8_FP32(offset, 4*7*8, 5*7*8, ymm4, ymm5)\
	LOAD_INPUT_1x8_FP32(offset, 6*7*8, ymm6)

#define LOAD_INPUT_4x1xFP32(offset)\
	LOAD_INPUT_2x1_FP32(offset, 0*4*8, 1*4*8, xmm0, xmm1)\
	LOAD_INPUT_2x1_FP32(offset, 2*4*8, 3*4*8, xmm2, xmm3)
#define LOAD_INPUT_5x1xFP32(offset)\
	LOAD_INPUT_2x1_FP32(offset, 0*5*8, 1*5*8, xmm0, xmm1)\
	LOAD_INPUT_2x1_FP32(offset, 2*5*8, 3*5*8, xmm2, xmm3)\
	LOAD_INPUT_1x1_FP32(offset, 4*5*8, xmm4)
#define LOAD_INPUT_6x1xFP32(offset)\
	LOAD_INPUT_2x1_FP32(offset, 0*6*8, 1*6*8, xmm0, xmm1)\
	LOAD_INPUT_2x1_FP32(offset, 2*6*8, 3*6*8, xmm2, xmm3)\
	LOAD_INPUT_2x1_FP32(offset, 4*6*8, 5*6*8, xmm4, xmm5)
#define LOAD_INPUT_7x1xFP32(offset)\
	LOAD_INPUT_2x1_FP32(offset, 0*7*8, 1*7*8, xmm0, xmm1)\
	LOAD_INPUT_2x1_FP32(offset, 2*7*8, 3*7*8, xmm2, xmm3)\
	LOAD_INPUT_2x1_FP32(offset, 4*7*8, 5*7*8, xmm4, xmm5)\
	LOAD_INPUT_1x1_FP32(offset, 6*7*8, xmm6)

#define LOAD_INPUT_4x8xFP16(offset)\
	LOAD_INPUT_2x8_FP16(offset, 0*4*8, 1*4*8, 0, 1)\
	LOAD_INPUT_2x8_FP16(offset, 2*4*8, 3*4*8, 2, 3)
#define LOAD_INPUT_5x8xFP16(offset)\
	LOAD_INPUT_2x8_FP16(offset, 0*5*8, 1*5*8, 0, 1)\
	LOAD_INPUT_2x8_FP16(offset, 2*5*8, 3*5*8, 2, 3)\
	LOAD_INPUT_1x8_FP16(offset, 4*5*8, 4)
#define LOAD_INPUT_6x8xFP16(offset)\
	LOAD_INPUT_2x8_FP16(offset, 0*6*8, 1*6*8, 0, 1)\
	LOAD_INPUT_2x8_FP16(offset, 2*6*8, 3*6*8, 2, 3)\
	LOAD_INPUT_2x8_FP16(offset, 4*6*8, 5*6*8, 4, 5)
#define LOAD_INPUT_7x8xFP16(offset)\
	LOAD_INPUT_2x8_FP16(offset, 0*7*8, 1*7*8, 0, 1)\
	LOAD_INPUT_2x8_FP16(offset, 2*7*8, 3*7*8, 2, 3)\
	LOAD_INPUT_2x8_FP16(offset, 4*7*8, 5*7*8, 4, 5)\
	LOAD_INPUT_1x8_FP16(offset, 6*7*8, 6)

#define LOAD_INPUT_4x1xFP16(offset)\
	LOAD_INPUT_2x1_FP16(offset, 0*4*8, 1*4*8, 0, 1)\
	LOAD_INPUT_2x1_FP16(offset, 2*4*8, 3*4*8, 2, 3)
#define LOAD_INPUT_5x1xFP16(offset)\
	LOAD_INPUT_2x1_FP16(offset, 0*5*8, 1*5*8, 0, 1)\
	LOAD_INPUT_2x1_FP16(offset, 2*5*8, 3*5*8, 2, 3)\
	LOAD_INPUT_1x1_FP16(offset, 4*5*8, 4)
#define LOAD_INPUT_6x1xFP16(offset)\
	LOAD_INPUT_2x1_FP16(offset, 0*6*8, 1*6*8, 0, 1)\
	LOAD_INPUT_2x1_FP16(offset, 2*6*8, 3*6*8, 2, 3)\
	LOAD_INPUT_2x1_FP16(offset, 4*6*8, 5*6*8, 4, 5)
#define LOAD_INPUT_7x1xFP16(offset)\
	LOAD_INPUT_2x1_FP16(offset, 0*7*8, 1*7*8, 0, 1)\
	LOAD_INPUT_2x1_FP16(offset, 2*7*8, 3*7*8, 2, 3)\
	LOAD_INPUT_2x1_FP16(offset, 4*7*8, 5*7*8, 4, 5)\
	LOAD_INPUT_1x1_FP16(offset, 6*7*8, 6)

#define STORE_WORKSPACE_1x1xFP32(reg, row, columns) vmovss(reg, mem(rdi, row*columns*1*4))
#define STORE_WORKSPACE_1x8xFP32(reg, row, columns) vmovaps(reg, mem(rdi, row*columns*8*4))

#define STORE_OUTPUT_1x1xFP32(offset, reg, dst) vmovss(reg, mem(dst, offset, 1))
#define STORE_OUTPUT_1x8xFP32(offset, reg, dst) vmovups(reg, mem(dst, offset, 1))

#define STORE_OUTPUT_1x1xFP16(offset, reg, dst)\
	vcvtps2ph(imm(0x03), xmm(reg), xmm(reg))\
	vmovss(xmm(reg), mem(dst, offset, 1))
#define STORE_OUTPUT_1x8xFP16(offset, reg, dst)\
	vcvtps2ph(imm(0x03), ymm(reg), xmm(reg))\
	vmovups(xmm(reg), mem(dst, offset, 1))

#define LOAD_WORKSPACE_4x8xFP32()\
	vmovaps(mem(rsi, 0*8*4), ymm0)\
	vmovaps(mem(rsi, 1*8*4), ymm1)\
	vmovaps(mem(rsi, 2*8*4), ymm2)\
	vmovaps(mem(rsi, 3*8*4), ymm3)
#define LOAD_WORKSPACE_5x8xFP32()\
	LOAD_WORKSPACE_4x8xFP32()\
	vmovaps(mem(rsi, 4*8*4), ymm4)
#define LOAD_WORKSPACE_6x8xFP32()\
	LOAD_WORKSPACE_5x8xFP32()\
	vmovaps(mem(rsi, 5*8*4), ymm5)
#define LOAD_WORKSPACE_7x8xFP32()\
	LOAD_WORKSPACE_6x8xFP32()\
	vmovaps(mem(rsi, 6*8*4), ymm6)

#define LOAD_WORKSPACE_4x1xFP32()\
	vmovss(mem(rsi, 0*1*4), xmm0)\
	vmovss(mem(rsi, 1*1*4), xmm1)\
	vmovss(mem(rsi, 2*1*4), xmm2)\
	vmovss(mem(rsi, 3*1*4), xmm3)
#define LOAD_WORKSPACE_5x1xFP32()\
	LOAD_WORKSPACE_4x1xFP32()\
	vmovss(mem(rsi, 4*1*4), xmm4)
#define LOAD_WORKSPACE_6x1xFP32()\
	LOAD_WORKSPACE_5x1xFP32()\
	vmovss(mem(rsi, 5*1*4), xmm5)
#define LOAD_WORKSPACE_7x1xFP32()\
	LOAD_WORKSPACE_6x1xFP32()\
	vmovss(mem(rsi, 6*1*4), xmm6)

#define INPUT_TRANSFORM_4x4_3x3_ROW_0()\
	vsubps(ymm2, ymm0, ymm7)\
	vsubps(ymm2, ymm4, ymm8)\
	vfmadd231ps(ymm14, ymm8, ymm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_1()\
	vaddps(ymm1, ymm2, ymm7)\
	vaddps(ymm3, ymm4, ymm8)\
	vfnmadd231ps(ymm14, ymm8, ymm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_2()\
	vsubps(ymm1, ymm2, ymm7)\
	vsubps(ymm4, ymm3, ymm8)\
	vfmadd231ps(ymm14, ymm8, ymm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_3()\
	vsubps(ymm1, ymm3, ymm7)\
	vsubps(ymm2, ymm4, ymm8)\
	vfmadd231ps(ymm15, ymm8, ymm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_4()\
	vsubps(ymm3, ymm1, ymm7)\
	vsubps(ymm2, ymm4, ymm8)\
	vfmadd231ps(ymm15, ymm8, ymm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_5()\
	vsubps(ymm3, ymm1, ymm7)\
	vsubps(ymm3, ymm5, ymm8)\
	vfmadd231ps(ymm14, ymm8, ymm7)

#define OUTPUT_TRANSFORM_4x4_3x3()\
	vmulps(ymm6, ymm15, ymm6)\
	vaddps(ymm1, ymm2, ymm7)\
	vsubps(ymm2, ymm1, ymm8)\
	vaddps(ymm3, ymm4, ymm9)\
	vsubps(ymm4, ymm3, ymm10)\
	vaddps(ymm0, ymm7, ymm0)\
	vmovaps(ymm8, ymm1)\
	vmovaps(ymm7, ymm2)\
	vaddps(ymm8, ymm6, ymm3)\
	vfmadd231ps(ymm13, ymm9, ymm0)\
	vfmadd231ps(ymm14, ymm10, ymm1)\
	vaddps(ymm9, ymm2, ymm2)\
	vfmadd231ps(ymm15, ymm10, ymm3)

#define INPUT_TRANSFORM_5x5_3x3_ROW_0()\
	vsubps(ymm1, ymm3, ymm7)\
	vsubps(ymm5, ymm3, ymm8)\
	vsubps(ymm2, ymm4, ymm9)\
	vsubps(ymm2, ymm0, ymm10)\
	vaddps(ymm7, ymm7, ymm7)\
	vfmadd231ps(ymm15, ymm8, ymm10)\
	vfmadd231ps(ymm14, ymm9, ymm7)\
	vaddps(ymm7, ymm10, ymm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_1()\
	vsubps(ymm1, ymm2, ymm7)\
	vsubps(ymm4, ymm3, ymm8)\
	vaddps(ymm3, ymm3, ymm9)\
	vfmadd231ps(ymm14, ymm8, ymm7)\
	vfnmadd231ps(ymm15, ymm5, ymm9)\
	vaddps(ymm7, ymm9, ymm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_2()\
	vsubps(ymm1, ymm2, ymm7)\
	vaddps(ymm3, ymm4, ymm8)\
	vsubps(ymm3, ymm2, ymm9)\
	vfmsub231ps(ymm14, ymm8, ymm8)\
	vfmadd231ps(ymm15, ymm5, ymm9)\
	vaddps(ymm7, ymm8, ymm7)\
	vaddps(ymm9, ymm2, ymm9)\
	vaddps(ymm7, ymm9, ymm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_3()\
	vaddps(ymm1, ymm5, ymm7)\
	vsubps(ymm4, ymm2, ymm8)\
	vmovaps(ymm3, ymm9)\
	vfmsub231ps(ymm15, ymm7, ymm9)\
	vfmsub231ps(ymm14, ymm8, ymm8)\
	vaddps(ymm8, ymm9, ymm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_4()\
	vsubps(ymm5, ymm1, ymm7)\
	vsubps(ymm2, ymm4, ymm8)\
	vmulps(ymm15, ymm7, ymm7)\
	vfmadd231ps(ymm14, ymm8, ymm8)\
	vaddps(ymm7, ymm8, ymm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_5()\
	vsubps(ymm3, ymm1, ymm7)\
	vsubps(ymm3, ymm5, ymm8)\
	vfmadd231ps(ymm14, ymm8, ymm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_6()\
	vsubps(ymm1, ymm3, ymm7)\
	vsubps(ymm4, ymm2, ymm8)\
	vsubps(ymm5, ymm3, ymm9)\
	vsubps(ymm4, ymm6, ymm10)\
	vaddps(ymm8, ymm8, ymm8)\
	vfmadd231ps(ymm14, ymm9, ymm7)\
	vfmadd231ps(ymm15, ymm10, ymm8)\
	vaddps(ymm7, ymm8, ymm7)

#define OUTPUT_TRANSFORM_5x5_3x3()\
	vaddps(ymm1, ymm2, ymm7)\
	vsubps(ymm2, ymm1, ymm8)\
	vaddps(ymm3, ymm4, ymm9)\
	vsubps(ymm4, ymm3, ymm10)\
	vaddps(ymm0, ymm7, ymm0)\
	vmovaps(ymm8, ymm1)\
	vmovaps(ymm7, ymm2)\
	vmovaps(ymm8, ymm3)\
	vaddps(ymm6, ymm7, ymm4)\
	vfmadd231ps(ymm12, ymm9, ymm0)\
	vfmadd231ps(ymm13, ymm10, ymm1)\
	vaddps(ymm9, ymm2, ymm2)\
	vfmadd231ps(ymm14, ymm10, ymm3)\
	vfmadd231ps(ymm15, ymm9, ymm4)\
	vfmadd231ps(ymm15, ymm5, ymm0)\
	vfmadd231ps(ymm14, ymm5, ymm1)\
	vaddps(ymm5, ymm2, ymm2)\
	vfmadd231ps(ymm13, ymm5, ymm3)\
	vfmadd231ps(ymm12, ymm5, ymm4)
}

namespace ml
{
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP32
	 */
	void winograd_input_transform_4x4_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
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

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r8, 0), ymm14)// 0.25f
		vbroadcastss(mem(r8, 4), ymm15)// 0.5f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED8)// main loop over channels, in steps of 8 elements
		movq(rax, rsi)
		movq(rbx, rdi)

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x8)
		LOAD_INPUT_6x8xFP32(r9)

		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_WORKSPACE_1x8xFP32(ymm7, 0, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_WORKSPACE_1x8xFP32(ymm7, 1, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_WORKSPACE_1x8xFP32(ymm7, 2, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_WORKSPACE_1x8xFP32(ymm7, 3, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_WORKSPACE_1x8xFP32(ymm7, 4, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_WORKSPACE_1x8xFP32(ymm7, 5, 6)

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*8*4), rdi)// add 8*4 (8 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x8)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(6), r8)// transform col counter
		label(TRANSFORM2x8)
		// second transform
		LOAD_WORKSPACE_6x8xFP32()

		movq(mem(rdi, 0*8), r13)
		movq(mem(rdi, 1*8), r14)
		movq(mem(rdi, 2*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r15)

		movq(mem(rdi, 3*8), r13)
		movq(mem(rdi, 4*8), r14)
		movq(mem(rdi, 5*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r15)

		add(imm(6*8*4), rsi)// add 7*8 (7*8 floats) to rsi (workspace), moving to next row
		add(imm(6*8), rdi)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x8)

		add(imm(8*4), r9)// add 8*4 (8 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED8)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)
		movq(rax, rsi)
		movq(rbx, rdi)

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

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*1*4), rdi)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_6x1xFP32()

		movq(mem(rdi, 0*8), r13)
		movq(mem(rdi, 1*8), r14)
		movq(mem(rdi, 2*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r15)

		movq(mem(rdi, 3*8), r13)
		movq(mem(rdi, 4*8), r14)
		movq(mem(rdi, 5*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r15)

		add(imm(6*1*4), rsi)// add 7*8 (7*1 floats) to rsi (workspace), moving to next row
		add(imm(6*8), rdi)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)

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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_4x4_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
	}
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP16
	 */
	void winograd_input_transform_4x4_3x3_avx2_fma_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
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

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r8, 0), ymm14)// 0.25f
		vbroadcastss(mem(r8, 4), ymm15)// 0.5f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED8)// main loop over channels, in steps of 8 elements
		movq(rax, rsi)
		movq(rbx, rdi)

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x8)
		LOAD_INPUT_6x8xFP16(r9)

		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_WORKSPACE_1x8xFP32(ymm7, 0, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_WORKSPACE_1x8xFP32(ymm7, 1, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_WORKSPACE_1x8xFP32(ymm7, 2, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_WORKSPACE_1x8xFP32(ymm7, 3, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_WORKSPACE_1x8xFP32(ymm7, 4, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_WORKSPACE_1x8xFP32(ymm7, 5, 6)

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*8*4), rdi)// add 8*4 (8 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x8)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(6), r8)// transform col counter
		label(TRANSFORM2x8)
		// second transform
		LOAD_WORKSPACE_6x8xFP32()

		movq(mem(rdi, 0*8), r13)
		movq(mem(rdi, 1*8), r14)
		movq(mem(rdi, 2*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_OUTPUT_1x8xFP16(r9, 7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_OUTPUT_1x8xFP16(r9, 7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_OUTPUT_1x8xFP16(r9, 7, r15)

		movq(mem(rdi, 3*8), r13)
		movq(mem(rdi, 4*8), r14)
		movq(mem(rdi, 5*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_OUTPUT_1x8xFP16(r9, 7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_OUTPUT_1x8xFP16(r9, 7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_OUTPUT_1x8xFP16(r9, 7, r15)

		add(imm(6*8*4), rsi)// add 7*8 (7*8 floats) to rsi (workspace), moving to next row
		add(imm(6*8), rdi)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x8)

		add(imm(8*2), r9)// add 8*2 (8 float16) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED8)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)
		movq(rax, rsi)
		movq(rbx, rdi)

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x1)

		LOAD_INPUT_6x1xFP16(r9)

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

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*1*4), rdi)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_6x1xFP32()

		movq(mem(rdi, 0*8), r13)
		movq(mem(rdi, 1*8), r14)
		movq(mem(rdi, 2*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_OUTPUT_1x1xFP16(r9, 7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_OUTPUT_1x1xFP16(r9, 7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_OUTPUT_1x1xFP16(r9, 7, r15)

		movq(mem(rdi, 3*8), r13)
		movq(mem(rdi, 4*8), r14)
		movq(mem(rdi, 5*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_OUTPUT_1x1xFP16(r9, 7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_OUTPUT_1x1xFP16(r9, 7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_OUTPUT_1x1xFP16(r9, 7, r15)

		add(imm(6*1*4), rsi)// add 7*8 (7*1 floats) to rsi (workspace), moving to next row
		add(imm(6*8), rdi)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)

		add(imm(1*2), r9)// add 1*2 (1 float16) to r9, the offset in channels
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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_4x4_3x3_avx2_fma_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
	}
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_input_transform_5x5_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
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

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r8, 0), ymm14)// 0.25f
		vbroadcastss(mem(r8, 4), ymm15)// 0.5f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED8)// main loop over channels, in steps of 8 elements
		movq(rax, rsi)
		movq(rbx, rdi)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x8)
		LOAD_INPUT_7x8xFP32(r9)

		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_WORKSPACE_1x8xFP32(ymm7, 0, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_WORKSPACE_1x8xFP32(ymm7, 1, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_WORKSPACE_1x8xFP32(ymm7, 2, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_WORKSPACE_1x8xFP32(ymm7, 3, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_WORKSPACE_1x8xFP32(ymm7, 4, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_WORKSPACE_1x8xFP32(ymm7, 5, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_WORKSPACE_1x8xFP32(ymm7, 6, 7)

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*8*4), rdi)// add 8*4 (8 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x8)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x8)
		// second transform
		LOAD_WORKSPACE_7x8xFP32()

		movq(mem(rdi, 0*8), r12)
		movq(mem(rdi, 1*8), r13)
		movq(mem(rdi, 2*8), r14)
		movq(mem(rdi, 3*8), r15)
		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r15)

		movq(mem(rdi, 4*8), r12)
		movq(mem(rdi, 5*8), r13)
		movq(mem(rdi, 6*8), r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_OUTPUT_1x8xFP32(r9, ymm7, r14)

		add(imm(7*8*4), rsi)// add 7*8 (7*8 floats) to rsi (workspace), moving to next row
		add(imm(7*8), rdi)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x8)

		add(imm(8*4), r9)// add 8*4 (8 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED8)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)
		movq(rax, rsi)
		movq(rbx, rdi)

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

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*1*4), rdi)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_7x1xFP32()

		movq(mem(rdi, 0*8), r12)
		movq(mem(rdi, 1*8), r13)
		movq(mem(rdi, 2*8), r14)
		movq(mem(rdi, 3*8), r15)
		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r15)

		movq(mem(rdi, 4*8), r12)
		movq(mem(rdi, 5*8), r13)
		movq(mem(rdi, 6*8), r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_OUTPUT_1x1xFP32(r9, xmm7, r14)

		add(imm(7*1*4), rsi)// add 7*8 (7*1 floats) to rsi (workspace), moving to next row
		add(imm(7*8), rdi)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)

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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r12", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_5x5_3x3_avx2_fma_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		const void **ext_ptr = ext;
		const void *bias_ptr = bias;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 8;
		const uint64_t k_left = filters % 8;
		const uint64_t flag_relu = use_relu;

		const float constants[4] = { 0.25f, 0.5f, 2.0f, 4.0f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r8, 0), ymm12)// 0.25f
		vbroadcastss(mem(r8, 4), ymm13)// 0.5f
		vbroadcastss(mem(r8, 8), ymm14)// 2.0f
		vbroadcastss(mem(r8, 12), ymm15)// 4.0f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED8)// main loop over channels, in steps of 8 elements
		movq(rax, rsi)
		movq(rbx, rdi)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x8)
		LOAD_INPUT_7x8xFP32(r9) // load column
		OUTPUT_TRANSFORM_5x5_3x3()
		STORE_WORKSPACE_1x8xFP32(ymm0, 0, 7)
		STORE_WORKSPACE_1x8xFP32(ymm1, 1, 7)
		STORE_WORKSPACE_1x8xFP32(ymm2, 2, 7)
		STORE_WORKSPACE_1x8xFP32(ymm3, 3, 7)
		STORE_WORKSPACE_1x8xFP32(ymm4, 4, 7)

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*8*4), rdi)// add 8*4 (8 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x8)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(5), r8)// transform row counter
		label(TRANSFORM2x8)
		// second transform
		LOAD_WORKSPACE_7x8xFP32() // load row
		OUTPUT_TRANSFORM_5x5_3x3()

		movq(mem(rdi, 0*8), r13)
		movq(mem(rdi, 1*8), r14)
		movq(mem(rdi, 2*8), r15)
		STORE_OUTPUT_1x8xFP32(r9, ymm0, r13)
		STORE_OUTPUT_1x8xFP32(r9, ymm1, r14)
		STORE_OUTPUT_1x8xFP32(r9, ymm2, r15)
		movq(mem(rdi, 3*8), r14)
		movq(mem(rdi, 4*8), r15)
		STORE_OUTPUT_1x8xFP32(r9, ymm3, r14)
		STORE_OUTPUT_1x8xFP32(r9, ymm4, r15)

		add(imm(7*8*4), rsi)// add 7*8 (7*8 floats) to rsi (workspace), moving to next row
		add(imm(5*8), rdi)// add 5*8 (5 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x8)

		add(imm(8*4), r9)// add 8*4 (8 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED8)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)
		movq(rax, rsi)
		movq(rbx, rdi)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x1)

		LOAD_INPUT_7x1xFP32(r9)
		OUTPUT_TRANSFORM_5x5_3x3()
		STORE_WORKSPACE_1x1xFP32(xmm0, 0, 7)
		STORE_WORKSPACE_1x1xFP32(xmm1, 1, 7)
		STORE_WORKSPACE_1x1xFP32(xmm2, 2, 7)
		STORE_WORKSPACE_1x1xFP32(xmm3, 3, 7)
		STORE_WORKSPACE_1x1xFP32(xmm4, 4, 7)

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*1*4), rdi)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(5), r8)// transform row counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_7x1xFP32() // load row
		OUTPUT_TRANSFORM_5x5_3x3()

		movq(mem(rdi, 0*8), r13)
		movq(mem(rdi, 1*8), r14)
		movq(mem(rdi, 2*8), r15)
		STORE_OUTPUT_1x1xFP32(r9, xmm0, r13)
		STORE_OUTPUT_1x1xFP32(r9, xmm1, r14)
		STORE_OUTPUT_1x1xFP32(r9, xmm2, r15)
		movq(mem(rdi, 3*8), r13)
		movq(mem(rdi, 4*8), r14)
		STORE_OUTPUT_1x1xFP32(r9, xmm3, r13)
		STORE_OUTPUT_1x1xFP32(r9, xmm4, r14)

		add(imm(7*1*4), rsi)// add 7*8 (7*1 floats) to rsi (workspace), moving to next row
		add(imm(5*8), rdi)// add 5*8 (5 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)

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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r13", "%r14", "%r15")
	}
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP16
	 */
	void winograd_input_transform_5x5_3x3_avx2_fma_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
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

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r8, 0), ymm14)// 0.25f
		vbroadcastss(mem(r8, 4), ymm15)// 0.5f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED8)// main loop over channels, in steps of 8 elements
		movq(rax, rsi)
		movq(rbx, rdi)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x8)
		LOAD_INPUT_7x8xFP16(r9)

		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_WORKSPACE_1x8xFP32(ymm7, 0, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_WORKSPACE_1x8xFP32(ymm7, 1, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_WORKSPACE_1x8xFP32(ymm7, 2, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_WORKSPACE_1x8xFP32(ymm7, 3, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_WORKSPACE_1x8xFP32(ymm7, 4, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_WORKSPACE_1x8xFP32(ymm7, 5, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_WORKSPACE_1x8xFP32(ymm7, 6, 7)

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*8*4), rdi)// add 8*4 (8 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x8)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x8)
		// second transform
		LOAD_WORKSPACE_7x8xFP32()

		movq(mem(rdi, 0*8), r12)
		movq(mem(rdi, 1*8), r13)
		movq(mem(rdi, 2*8), r14)
		movq(mem(rdi, 3*8), r15)
		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_OUTPUT_1x8xFP16(r9, 7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_OUTPUT_1x8xFP16(r9, 7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_OUTPUT_1x8xFP16(r9, 7, r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_OUTPUT_1x8xFP16(r9, 7, r15)

		movq(mem(rdi, 4*8), r12)
		movq(mem(rdi, 5*8), r13)
		movq(mem(rdi, 6*8), r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_OUTPUT_1x8xFP16(r9, 7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_OUTPUT_1x8xFP16(r9, 7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_OUTPUT_1x8xFP16(r9, 7, r14)

		add(imm(7*8*4), rsi)// add 7*8 (7*8 floats) to rsi (workspace), moving to next row
		add(imm(7*8), rdi)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x8)

		add(imm(8*2), r9)// add 8*2 (8 float16) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED8)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)
		movq(rax, rsi)
		movq(rbx, rdi)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x1)

		LOAD_INPUT_7x1xFP16(r9)

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

		add(imm(1*8), rsi)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*1*4), rdi)// add 1*4 (1 float32) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)

		movq(rbx, rsi)
		movq(rcx, rdi)
		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_7x1xFP32()

		movq(mem(rdi, 0*8), r12)
		movq(mem(rdi, 1*8), r13)
		movq(mem(rdi, 2*8), r14)
		movq(mem(rdi, 3*8), r15)
		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_OUTPUT_1x1xFP16(r9, 7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_OUTPUT_1x1xFP16(r9, 7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_OUTPUT_1x1xFP16(r9, 7, r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_OUTPUT_1x1xFP16(r9, 7, r15)

		movq(mem(rdi, 4*8), r12)
		movq(mem(rdi, 5*8), r13)
		movq(mem(rdi, 6*8), r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_OUTPUT_1x1xFP16(r9, 7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_OUTPUT_1x1xFP16(r9, 7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_OUTPUT_1x1xFP16(r9, 7, r14)

		add(imm(7*1*4), rsi)// add 7*8 (7*1 floats) to rsi (workspace), moving to next row
		add(imm(7*8), rdi)// add 7*8*4 (7 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)

		add(imm(1*2), r9)// add 1*2 (1 float16) to r9, the offset in channels
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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r12", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_5x5_3x3_avx2_fma_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
	}

} /* namespace ml */

