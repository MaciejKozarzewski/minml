/*
 * avx512_winograd_kernels.cpp
 *
 *  Created on: Sep 22, 2023
 *      Author: Maciej Kozarzewski
 */

#include "winograd_kernels.hpp"

#include "../assembly_macros.hpp"

#include <array>
#include <cinttypes>
#include <cassert>

namespace
{
#define LOAD_1x16_FP32(reg, offset, src, dst)\
	movq(mem(reg, src), r15)\
	vmovups(mem(r15, offset, 1), dst)
#define LOAD_2x16_FP32(reg, offset, src1, src2, dst1, dst2)\
	movq(mem(reg, src1), r14)\
	movq(mem(reg, src2), r15)\
	vmovups(mem(r14, offset, 1), dst1)\
	vmovups(mem(r15, offset, 1), dst2)
#define LOAD_1x1_FP32(reg, offset, src, dst)\
	movq(mem(reg, src), r15)\
	vmovss(mem(r15, offset, 1), dst)
#define LOAD_2x1_FP32(reg, offset, src1, src2, dst1, dst2)\
	movq(mem(reg, src1), r14)\
	movq(mem(reg, src2), r15)\
	vmovss(mem(r14, offset, 1), dst1)\
	vmovss(mem(r15, offset, 1), dst2)

#define LOAD_1x16_FP16(reg, offset, src, dst)\
	movq(mem(reg, src), r15)\
	vmovups(mem(r15, offset, 1), ymm(dst))\
	vcvtph2ps(ymm(dst), zmm(dst))
#define LOAD_2x16_FP16(reg, offset, src1, src2, dst1, dst2)\
	movq(mem(reg, src1), r14)\
	movq(mem(reg, src2), r15)\
	vmovups(mem(r14, offset, 1), ymm(dst1))\
	vmovups(mem(r15, offset, 1), ymm(dst2))\
	vcvtph2ps(ymm(dst1), zmm(dst1))\
	vcvtph2ps(ymm(dst2), zmm(dst2))
#define LOAD_1x1_FP16(reg, offset, src, dst)\
	movq(mem(reg, src), r15)\
	movzw(mem(r15, offset, 1), r15)\
	vmovq(r15, xmm(dst))\
	vcvtph2ps(xmm(dst), xmm(dst))
#define LOAD_2x1_FP16(reg, offset, src1, src2, dst1, dst2)\
	movq(mem(reg, src1), r14)\
	movq(mem(reg, src2), r15)\
	movzw(mem(r14, offset, 1), r14)\
	movzw(mem(r15, offset, 1), r15)\
	vmovq(r14, xmm(dst1))\
	vmovq(r15, xmm(dst2))\
	vcvtph2ps(xmm(dst1), xmm(dst1))\
	vcvtph2ps(xmm(dst2), xmm(dst2))

#define LOAD_INPUT_4x16xFP32(offset)\
	LOAD_2x16_FP32(rax, offset, 0*4*8, 1*4*8, zmm0, zmm1)\
	LOAD_2x16_FP32(rax, offset, 2*4*8, 3*4*8, zmm2, zmm3)
#define LOAD_INPUT_5x16xFP32(offset)\
	LOAD_2x16_FP32(rax, offset, 0*5*8, 1*5*8, zmm0, zmm1)\
	LOAD_2x16_FP32(rax, offset, 2*5*8, 3*5*8, zmm2, zmm3)\
	LOAD_1x16_FP32(rax, offset, 4*5*8, zmm4)
#define LOAD_INPUT_6x16xFP32(offset)\
	LOAD_2x16_FP32(rax, offset, 0*6*8, 1*6*8, zmm0, zmm1)\
	LOAD_2x16_FP32(rax, offset, 2*6*8, 3*6*8, zmm2, zmm3)\
	LOAD_2x16_FP32(rax, offset, 4*6*8, 5*6*8, zmm4, zmm5)
#define LOAD_INPUT_7x16xFP32(offset)\
	LOAD_2x16_FP32(rax, offset, 0*7*8, 1*7*8, zmm0, zmm1)\
	LOAD_2x16_FP32(rax, offset, 2*7*8, 3*7*8, zmm2, zmm3)\
	LOAD_2x16_FP32(rax, offset, 4*7*8, 5*7*8, zmm4, zmm5)\
	LOAD_1x16_FP32(rax, offset, 6*7*8, zmm6)

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

#define LOAD_INPUT_4x16xFP16(offset)\
	LOAD_2x16_FP16(rax, offset, 0*4*8, 1*4*8, 0, 1)\
	LOAD_2x16_FP16(rax, offset, 2*4*8, 3*4*8, 2, 3)
#define LOAD_INPUT_5x16xFP16(offset)\
	LOAD_2x16_FP16(rax, offset, 0*5*8, 1*5*8, 0, 1)\
	LOAD_2x16_FP16(rax, offset, 2*5*8, 3*5*8, 2, 3)\
	LOAD_1x16_FP16(rax, offset, 4*5*8, 4)
#define LOAD_INPUT_6x16xFP16(offset)\
	LOAD_2x16_FP16(rax, offset, 0*6*8, 1*6*8, 0, 1)\
	LOAD_2x16_FP16(rax, offset, 2*6*8, 3*6*8, 2, 3)\
	LOAD_2x16_FP16(rax, offset, 4*6*8, 5*6*8, 4, 5)
#define LOAD_INPUT_7x16xFP16(offset)\
	LOAD_2x16_FP16(rax, offset, 0*7*8, 1*7*8, 0, 1)\
	LOAD_2x16_FP16(rax, offset, 2*7*8, 3*7*8, 2, 3)\
	LOAD_2x16_FP16(rax, offset, 4*7*8, 5*7*8, 4, 5)\
	LOAD_1x16_FP16(rax, offset, 6*7*8, 6)

#define LOAD_INPUT_4x1xFP16(offset)\
	LOAD_2x1_FP16(rax, offset, 0*4*8, 1*4*8, 0, 1)\
	LOAD_2x1_FP16(rax, offset, 2*4*8, 3*4*8, 2, 3)
#define LOAD_INPUT_5x1xFP16(offset)\
	LOAD_2x1_FP16(rax, offset, 0*5*8, 1*5*8, 0, 1)\
	LOAD_2x1_FP16(rax, offset, 2*5*8, 3*5*8, 2, 3)\
	LOAD_1x1_FP16(rax, offset, 4*5*8, 4)
#define LOAD_INPUT_6x1xFP16(offset)\
	LOAD_2x1_FP16(rax, offset, 0*6*8, 1*6*8, 0, 1)\
	LOAD_2x1_FP16(rax, offset, 2*6*8, 3*6*8, 2, 3)\
	LOAD_2x1_FP16(rax, offset, 4*6*8, 5*6*8, 4, 5)
#define LOAD_INPUT_7x1xFP16(offset)\
	LOAD_2x1_FP16(rax, offset, 0*7*8, 1*7*8, 0, 1)\
	LOAD_2x1_FP16(rax, offset, 2*7*8, 3*7*8, 2, 3)\
	LOAD_2x1_FP16(rax, offset, 4*7*8, 5*7*8, 4, 5)\
	LOAD_1x1_FP16(rax, offset, 6*7*8, 6)

#define STORE_WORKSPACE_1x1xFP32(reg, row, columns) vmovss(reg, mem(rbx, row*columns*1*4))
#define STORE_WORKSPACE_1x16xFP32(reg, row, columns) vmovaps(reg, mem(rbx, row*columns*16*4))

#define STORE_OUTPUT_1x1xFP32(offset, reg, dst) vmovss(reg, mem(dst, offset, 1))
#define STORE_OUTPUT_1x16xFP32(offset, reg, dst) vmovups(reg, mem(dst, offset, 1))

#define STORE_OUTPUT_1x1xFP16(offset, reg, dst)\
	vcvtps2ph(imm(0x03), xmm(reg), xmm(reg))\
	vmovq(xmm(reg), rsi)\
	mov(si, mem(dst, offset, 1))
#define STORE_OUTPUT_1x16xFP16(offset, reg, dst)\
	vcvtps2ph(imm(0x03), zmm(reg), ymm(reg))\
	vmovups(ymm(reg), mem(dst, offset, 1))

#define LOAD_WORKSPACE_4x16xFP32()\
	vmovaps(mem(rbx, 0*16*4), zmm0)\
	vmovaps(mem(rbx, 1*16*4), zmm1)\
	vmovaps(mem(rbx, 2*16*4), zmm2)\
	vmovaps(mem(rbx, 3*16*4), zmm3)
#define LOAD_WORKSPACE_5x16xFP32()\
	LOAD_WORKSPACE_4x16xFP32()\
	vmovaps(mem(rbx, 4*16*4), zmm4)
#define LOAD_WORKSPACE_6x16xFP32()\
	LOAD_WORKSPACE_5x16xFP32()\
	vmovaps(mem(rbx, 5*16*4), zmm5)
#define LOAD_WORKSPACE_7x16xFP32()\
	LOAD_WORKSPACE_6x16xFP32()\
	vmovaps(mem(rbx, 6*16*4), zmm6)

#define LOAD_WORKSPACE_4x1xFP32()\
	vmovss(mem(rbx, 0*1*4), xmm0)\
	vmovss(mem(rbx, 1*1*4), xmm1)\
	vmovss(mem(rbx, 2*1*4), xmm2)\
	vmovss(mem(rbx, 3*1*4), xmm3)
#define LOAD_WORKSPACE_5x1xFP32()\
	LOAD_WORKSPACE_4x1xFP32()\
	vmovss(mem(rbx, 4*1*4), xmm4)
#define LOAD_WORKSPACE_6x1xFP32()\
	LOAD_WORKSPACE_5x1xFP32()\
	vmovss(mem(rbx, 5*1*4), xmm5)
#define LOAD_WORKSPACE_7x1xFP32()\
	LOAD_WORKSPACE_6x1xFP32()\
	vmovss(mem(rbx, 6*1*4), xmm6)

#define ADD_BIAS_4x16xFP32(reg)\
	vaddps(zmm0, reg, zmm0)\
	vaddps(zmm1, reg, zmm1)\
	vaddps(zmm2, reg, zmm2)\
	vaddps(zmm3, reg, zmm3)
#define ADD_BIAS_5x16xFP32(reg)\
	vaddps(zmm0, reg, zmm0)\
	vaddps(zmm1, reg, zmm1)\
	vaddps(zmm2, reg, zmm2)\
	vaddps(zmm3, reg, zmm3)\
	vaddps(zmm4, reg, zmm4)

#define LOAD_EXT_4x1xFP32(offset)\
	LOAD_2x1_FP32(rdx, offset, 0*8, 1*8, xmm4, xmm5)\
	LOAD_2x1_FP32(rdx, offset, 2*8, 3*8, xmm6, xmm7)
#define LOAD_EXT_5x1xFP32(offset)\
	LOAD_2x1_FP32(rdx, offset, 0*8, 1*8, xmm5, xmm6)\
	LOAD_2x1_FP32(rdx, offset, 2*8, 3*8, xmm7, xmm8)\
	LOAD_1x1_FP32(rdx, offset, 4*8, xmm9)
#define LOAD_EXT_4x16xFP32(offset)\
	LOAD_2x16_FP32(rdx, offset, 0*8, 1*8, zmm4, zmm5)\
	LOAD_2x16_FP32(rdx, offset, 2*8, 3*8, zmm6, zmm7)
#define LOAD_EXT_5x16xFP32(offset)\
	LOAD_2x16_FP32(rdx, offset, 0*8, 1*8, zmm5, zmm6)\
	LOAD_2x16_FP32(rdx, offset, 2*8, 3*8, zmm7, zmm8)\
	LOAD_1x16_FP32(rdx, offset, 4*8, zmm9)

#define LOAD_EXT_4x1xFP16(offset)\
	LOAD_2x1_FP16(rdx, offset, 0*8, 1*8, 4, 5)\
	LOAD_2x1_FP16(rdx, offset, 2*8, 3*8, 6, 7)
#define LOAD_EXT_5x1xFP16(offset)\
	LOAD_2x1_FP16(rdx, offset, 0*8, 1*8, 5, 6)\
	LOAD_2x1_FP16(rdx, offset, 2*8, 3*8, 7, 8)\
	LOAD_1x1_FP16(rdx, offset, 4*8, 9)
#define LOAD_EXT_4x16xFP16(offset)\
	LOAD_2x16_FP16(rdx, offset, 0*8, 1*8, 4, 5)\
	LOAD_2x16_FP16(rdx, offset, 2*8, 3*8, 6, 7)
#define LOAD_EXT_5x16xFP16(offset)\
	LOAD_2x16_FP16(rdx, offset, 0*8, 1*8, 5, 6)\
	LOAD_2x16_FP16(rdx, offset, 2*8, 3*8, 7, 8)\
	LOAD_1x16_FP16(rdx, offset, 4*8, 9)

#define ADD_EXT_4x16xFP32()\
	vaddps(zmm0, zmm4, zmm0)\
	vaddps(zmm1, zmm5, zmm1)\
	vaddps(zmm2, zmm6, zmm2)\
	vaddps(zmm3, zmm7, zmm3)
#define ADD_EXT_5x16xFP32()\
	vaddps(zmm0, zmm5, zmm0)\
	vaddps(zmm1, zmm6, zmm1)\
	vaddps(zmm2, zmm7, zmm2)\
	vaddps(zmm3, zmm8, zmm3)\
	vaddps(zmm4, zmm9, zmm4)

#define APPLY_RELU_4x16xFP32(tmp)\
	vxorps(tmp, tmp, tmp)\
	vmaxps(zmm0, tmp, zmm0)\
	vmaxps(zmm1, tmp, zmm1)\
	vmaxps(zmm2, tmp, zmm2)\
	vmaxps(zmm3, tmp, zmm3)
#define APPLY_RELU_5x16xFP32(tmp)\
	vxorps(tmp, tmp, tmp)\
	vmaxps(zmm0, tmp, zmm0)\
	vmaxps(zmm1, tmp, zmm1)\
	vmaxps(zmm2, tmp, zmm2)\
	vmaxps(zmm3, tmp, zmm3)\
	vmaxps(zmm4, tmp, zmm4)

#define INPUT_TRANSFORM_4x4_3x3_ROW_0()\
	vsubps(zmm2, zmm0, zmm7)\
	vsubps(zmm2, zmm4, zmm8)\
	vfmadd231ps(zmm14, zmm8, zmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_1()\
	vaddps(zmm1, zmm2, zmm7)\
	vaddps(zmm3, zmm4, zmm8)\
	vfnmadd231ps(zmm14, zmm8, zmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_2()\
	vsubps(zmm1, zmm2, zmm7)\
	vsubps(zmm4, zmm3, zmm8)\
	vfmadd231ps(zmm14, zmm8, zmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_3()\
	vsubps(zmm1, zmm3, zmm7)\
	vsubps(zmm2, zmm4, zmm8)\
	vfmadd231ps(zmm15, zmm8, zmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_4()\
	vsubps(zmm3, zmm1, zmm7)\
	vsubps(zmm2, zmm4, zmm8)\
	vfmadd231ps(zmm15, zmm8, zmm7)
#define INPUT_TRANSFORM_4x4_3x3_ROW_5()\
	vsubps(zmm3, zmm1, zmm7)\
	vsubps(zmm3, zmm5, zmm8)\
	vfmadd231ps(zmm14, zmm8, zmm7)

#define OUTPUT_TRANSFORM_4x4_3x3()\
	vaddps(zmm1, zmm2, zmm7)\
	vsubps(zmm2, zmm1, zmm8)\
	vaddps(zmm3, zmm4, zmm9)\
	vsubps(zmm4, zmm3, zmm10)\
	vaddps(zmm0, zmm7, zmm0)\
	vmovaps(zmm8, zmm1)\
	vmovaps(zmm7, zmm2)\
	vmovaps(zmm8, zmm3)\
	vfmadd231ps(zmm15, zmm5, zmm3)\
	vfmadd231ps(zmm13, zmm9, zmm0)\
	vfmadd231ps(zmm14, zmm10, zmm1)\
	vaddps(zmm9, zmm2, zmm2)\
	vfmadd231ps(zmm15, zmm10, zmm3)

#define INPUT_TRANSFORM_5x5_3x3_ROW_0()\
	vsubps(zmm1, zmm3, zmm7)\
	vsubps(zmm5, zmm3, zmm8)\
	vsubps(zmm2, zmm4, zmm9)\
	vsubps(zmm2, zmm0, zmm10)\
	vaddps(zmm7, zmm7, zmm7)\
	vfmadd231ps(zmm15, zmm8, zmm10)\
	vfmadd231ps(zmm14, zmm9, zmm7)\
	vaddps(zmm7, zmm10, zmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_1()\
	vsubps(zmm1, zmm2, zmm7)\
	vsubps(zmm4, zmm3, zmm8)\
	vaddps(zmm3, zmm3, zmm9)\
	vfmadd231ps(zmm14, zmm8, zmm7)\
	vfnmadd231ps(zmm15, zmm5, zmm9)\
	vaddps(zmm7, zmm9, zmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_2()\
	vsubps(zmm1, zmm2, zmm7)\
	vaddps(zmm3, zmm4, zmm8)\
	vsubps(zmm3, zmm2, zmm9)\
	vfmsub231ps(zmm14, zmm8, zmm8)\
	vfmadd231ps(zmm15, zmm5, zmm9)\
	vaddps(zmm7, zmm8, zmm7)\
	vaddps(zmm9, zmm2, zmm9)\
	vaddps(zmm7, zmm9, zmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_3()\
	vaddps(zmm1, zmm5, zmm7)\
	vsubps(zmm4, zmm2, zmm8)\
	vmovaps(zmm3, zmm9)\
	vfmsub231ps(zmm15, zmm7, zmm9)\
	vfmsub231ps(zmm14, zmm8, zmm8)\
	vaddps(zmm8, zmm9, zmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_4()\
	vsubps(zmm5, zmm1, zmm7)\
	vsubps(zmm2, zmm4, zmm8)\
	vmulps(zmm15, zmm7, zmm7)\
	vfmadd231ps(zmm14, zmm8, zmm8)\
	vaddps(zmm7, zmm8, zmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_5()\
	vsubps(zmm3, zmm1, zmm7)\
	vsubps(zmm3, zmm5, zmm8)\
	vfmadd231ps(zmm14, zmm8, zmm7)
#define INPUT_TRANSFORM_5x5_3x3_ROW_6()\
	vsubps(zmm1, zmm3, zmm7)\
	vsubps(zmm4, zmm2, zmm8)\
	vsubps(zmm5, zmm3, zmm9)\
	vsubps(zmm4, zmm6, zmm10)\
	vaddps(zmm8, zmm8, zmm8)\
	vfmadd231ps(zmm14, zmm9, zmm7)\
	vfmadd231ps(zmm15, zmm10, zmm8)\
	vaddps(zmm7, zmm8, zmm7)

#define OUTPUT_TRANSFORM_5x5_3x3()\
	vaddps(zmm1, zmm2, zmm7)\
	vsubps(zmm2, zmm1, zmm8)\
	vaddps(zmm3, zmm4, zmm9)\
	vsubps(zmm4, zmm3, zmm10)\
	vaddps(zmm0, zmm7, zmm0)\
	vmovaps(zmm8, zmm1)\
	vmovaps(zmm7, zmm2)\
	vmovaps(zmm8, zmm3)\
	vaddps(zmm6, zmm7, zmm4)\
	vfmadd231ps(zmm12, zmm9, zmm0)\
	vfmadd231ps(zmm13, zmm10, zmm1)\
	vaddps(zmm9, zmm2, zmm2)\
	vfmadd231ps(zmm14, zmm10, zmm3)\
	vfmadd231ps(zmm15, zmm9, zmm4)\
	vfmadd231ps(zmm15, zmm5, zmm0)\
	vfmadd231ps(zmm14, zmm5, zmm1)\
	vaddps(zmm5, zmm2, zmm2)\
	vfmadd231ps(zmm13, zmm5, zmm3)\
	vfmadd231ps(zmm12, zmm5, zmm4)
}

namespace ml
{
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP32
	 */
	void winograd_input_transform_4x4_3x3_avx512_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 16;
		const uint64_t k_left = filters % 16;

		const float constants[2] = { 0.25f, 0.5f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r8, 0), zmm14)// 0.25f
		vbroadcastss(mem(r8, 4), zmm15)// 0.5f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED16)// main loop over channels, in steps of 16 elements

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x16)
		LOAD_INPUT_6x16xFP32(r9)

		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_WORKSPACE_1x16xFP32(zmm7, 0, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_WORKSPACE_1x16xFP32(zmm7, 1, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_WORKSPACE_1x16xFP32(zmm7, 2, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_WORKSPACE_1x16xFP32(zmm7, 3, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_WORKSPACE_1x16xFP32(zmm7, 4, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_WORKSPACE_1x16xFP32(zmm7, 5, 6)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rax (src), moving to next column
		add(imm(1*16*4), rbx)// add 16*4 (16 floats) to rbx (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x16)
		sub(imm(6*1*8), rax)// subtract 6*1*8 (6*1 pointer) to rax (src), moving to start
		sub(imm(6*1*16*4), rbx)// subtract 6*16*4 (6*16 floats) to rbx (workspace), moving to start

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM2x16)
		// second transform
		LOAD_WORKSPACE_6x16xFP32()

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r15)

		movq(mem(rcx, 3*8), r13)
		movq(mem(rcx, 4*8), r14)
		movq(mem(rcx, 5*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r15)

		add(imm(6*16*4), rbx)// add 6*16 (6*16 floats) to rbx (workspace), moving to next row
		add(imm(6*8), rcx)// add 6*8 (6 pointers) to rcx (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x16)
		sub(imm(6*6*16*4), rbx)// subtract 6*6*16 (6*6*16 floats) to rbx (workspace), moving to start
		sub(imm(6*6*8), rcx)// subtract 6*6*8*4 (6*6 pointers) to rcx (dst), moving to start

		add(imm(16*4), r9)// add 8*4 (8 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED16)

		/* END OF MAIN 8-UNROLLED LOOP */
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
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%r8", "%r9", "%r10", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_4x4_3x3_avx512_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		const void **ext_ptr = ext;
		const void *bias_ptr = bias;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 16;
		const uint64_t k_left = filters % 16;
		const uint64_t flag_relu = static_cast<uint64_t>(use_relu);

		const float constants[3] = { 0.25f, 0.5f, 2.0f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		vbroadcastss(mem(r8, 0), zmm13)// 0.25f
		vbroadcastss(mem(r8, 4), zmm14)// 0.5f
		vbroadcastss(mem(r8, 8), zmm15)// 2.0f

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

		label(UNROLLED16)// main loop over channels, in steps of 8 elements

		vxorps(zmm11, zmm11, zmm11)
		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x16)
		vmovups(mem(rdi, r9, 1), zmm11)// load bias
		label(SKIP_BIAS_LOAD_x16)

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x16)
		LOAD_INPUT_6x16xFP32(r9)// load column
		OUTPUT_TRANSFORM_4x4_3x3()
		STORE_WORKSPACE_1x16xFP32(zmm0, 0, 6)
		STORE_WORKSPACE_1x16xFP32(zmm1, 1, 6)
		STORE_WORKSPACE_1x16xFP32(zmm2, 2, 6)
		STORE_WORKSPACE_1x16xFP32(zmm3, 3, 6)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*16*4), rbx)// add 16*4 (16 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x16)
		sub(imm(6*1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		sub(imm(6*1*16*4), rbx)// add 16*4 (16 floats) to rdi (workspace), moving to next column

		movq(imm(4), r8)// transform row counter
		label(TRANSFORM2x16)
		// second transform
		LOAD_WORKSPACE_6x16xFP32()// load row
		OUTPUT_TRANSFORM_4x4_3x3()
		ADD_BIAS_4x16xFP32(zmm11)

		test(r11, r11)
		je(SKIP_LOAD_EXT_x16)
		LOAD_EXT_4x16xFP32(r9)
		ADD_EXT_4x16xFP32()
		label(SKIP_LOAD_EXT_x16)

		test(r12, r12)
		je(SKIP_RELU_x16)
		APPLY_RELU_4x16xFP32(zmm10)
		label(SKIP_RELU_x16)

		movq(mem(rcx, 0*8), r14)
		movq(mem(rcx, 1*8), r15)
		STORE_OUTPUT_1x16xFP32(r9, zmm0, r14)
		STORE_OUTPUT_1x16xFP32(r9, zmm1, r15)
		movq(mem(rcx, 2*8), r14)
		movq(mem(rcx, 3*8), r15)
		STORE_OUTPUT_1x16xFP32(r9, zmm2, r14)
		STORE_OUTPUT_1x16xFP32(r9, zmm3, r15)

		add(imm(6*16*4), rbx)// add 6*16 (6*16 floats) to rbx (workspace), moving to next row
		add(imm(4*8), rcx)// add 4*8 (4 pointers) to rcx (dst), moving to next row
		add(imm(4*8), rdx)// add 4*8 (4 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x16)
		sub(imm(4*6*16*4), rbx)// add 6*16 (6*16 floats) to rbx (workspace), moving to next row
		sub(imm(4*4*8), rcx)// add 4*8 (4 pointers) to rcx (dst), moving to next row
		sub(imm(4*4*8), rdx)// add 4*8 (4 pointers) to rdx (ext), moving to next row

		add(imm(16*4), r9)// add 16*4 (16 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED16)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x1)
		vmovss(mem(rdi, r9, 1), xmm11)// load bias
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
		ADD_BIAS_4x16xFP32(zmm11)

		test(r11, r11)// check if external tensor pointer is not null
		je(SKIP_LOAD_EXT_x1)
		LOAD_EXT_4x1xFP32(r9)
		ADD_EXT_4x16xFP32()
		label(SKIP_LOAD_EXT_x1)

		test(r12, r12)// check if ReLU should be applied
		je(SKIP_RELU_x1)
		APPLY_RELU_4x16xFP32(zmm10)
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
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
	}
	/*
	 * Transforms for 3x3 kernel and 4x4 tile size in FP16
	 */
	void winograd_input_transform_4x4_3x3_avx512_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 16;
		const uint64_t k_left = filters % 16;

		const float constants[2] = { 0.25f, 0.5f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r8, 0), zmm14)// 0.25f
		vbroadcastss(mem(r8, 4), zmm15)// 0.5f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED16)// main loop over channels, in steps of 8 elements

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x16)
		LOAD_INPUT_6x16xFP16(r9)

		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_WORKSPACE_1x16xFP32(zmm7, 0, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_WORKSPACE_1x16xFP32(zmm7, 1, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_WORKSPACE_1x16xFP32(zmm7, 2, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_WORKSPACE_1x16xFP32(zmm7, 3, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_WORKSPACE_1x16xFP32(zmm7, 4, 6)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_WORKSPACE_1x16xFP32(zmm7, 5, 6)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rax (src), moving to next column
		add(imm(1*16*4), rbx)// add 16*4 (16 floats) to rbx (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x16)
		sub(imm(6*1*8), rax)// add 6*1*8 (1 pointer) to rax (src), moving to start
		sub(imm(6*1*16*4), rbx)// add 6*16*4 (16 floats) to rbx (workspace), moving to start

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM2x16)
		// second transform
		LOAD_WORKSPACE_6x16xFP32()

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_OUTPUT_1x16xFP16(r9, 7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_OUTPUT_1x16xFP16(r9, 7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_OUTPUT_1x16xFP16(r9, 7, r15)

		movq(mem(rcx, 3*8), r13)
		movq(mem(rcx, 4*8), r14)
		movq(mem(rcx, 5*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_OUTPUT_1x16xFP16(r9, 7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_OUTPUT_1x16xFP16(r9, 7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_OUTPUT_1x16xFP16(r9, 7, r15)

		add(imm(6*16*4), rbx)// add 6*16 (6*16 floats) to rbx (workspace), moving to next row
		add(imm(6*8), rcx)// add 6*8*4 (6 pointers) to rcx (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x16)
		sub(imm(6*6*16*4), rbx)// add 6*6*16 (6*6*16 floats) to rbx (workspace), moving to start
		sub(imm(6*6*8), rcx)// add 6*6*8*4 (6*6 pointers) to rcx (dst), moving to start

		add(imm(16*2), r9)// add 8*2 (8 float16) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED16)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

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

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rax (src), moving to next column
		add(imm(1*1*4), rbx)// add 1*4 (1 float32) to rbx (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x1)
		sub(imm(6*1*8), rax)// subtract 1*8 (1 pointer) to rax (src), moving to start
		sub(imm(6*1*1*4), rbx)// subtract 1*4 (1 float32) to rbx (workspace), moving to start

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM2x1)
		// second transform
		LOAD_WORKSPACE_6x1xFP32()

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_0()
		STORE_OUTPUT_1x1xFP16(r9, 7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_1()
		STORE_OUTPUT_1x1xFP16(r9, 7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_2()
		STORE_OUTPUT_1x1xFP16(r9, 7, r15)

		movq(mem(rcx, 3*8), r13)
		movq(mem(rcx, 4*8), r14)
		movq(mem(rcx, 5*8), r15)
		INPUT_TRANSFORM_4x4_3x3_ROW_3()
		STORE_OUTPUT_1x1xFP16(r9, 7, r13)
		INPUT_TRANSFORM_4x4_3x3_ROW_4()
		STORE_OUTPUT_1x1xFP16(r9, 7, r14)
		INPUT_TRANSFORM_4x4_3x3_ROW_5()
		STORE_OUTPUT_1x1xFP16(r9, 7, r15)

		add(imm(6*1*4), rbx)// add 6*8 (6*1 floats) to rsi (workspace), moving to next row
		add(imm(6*8), rcx)// add 6*8*4 (6 pointers) to rdi (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)
		sub(imm(6*6*1*4), rbx)// subtract 6*8 (6*1 floats) to rbx (workspace), moving to start
		sub(imm(6*6*8), rcx)// subtract 6*8*4 (6 pointers) to rcx (dst), moving to start

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
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%rsi", "%r8", "%r9", "%r10", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_4x4_3x3_avx512_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		const void **ext_ptr = ext;
		const void *bias_ptr = bias;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 16;
		const uint64_t k_left = filters % 16;
		const uint64_t flag_relu = static_cast<uint64_t>(use_relu);

		const float constants[3] = { 0.25f, 0.5f, 2.0f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		vbroadcastss(mem(r8, 0), zmm13)// 0.25f
		vbroadcastss(mem(r8, 4), zmm14)// 0.5f
		vbroadcastss(mem(r8, 8), zmm15)// 2.0f

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

		label(UNROLLED16)// main loop over channels, in steps of 8 elements

		vxorps(zmm11, zmm11, zmm11)
		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x16)
		vmovups(mem(rdi, r9, 1), ymm11)// load bias
		vcvtph2ps(ymm11, zmm11)// convert to fp32
		label(SKIP_BIAS_LOAD_x16)

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x16)
		LOAD_INPUT_6x16xFP16(r9)// load column
		OUTPUT_TRANSFORM_4x4_3x3()
		STORE_WORKSPACE_1x16xFP32(zmm0, 0, 6)
		STORE_WORKSPACE_1x16xFP32(zmm1, 1, 6)
		STORE_WORKSPACE_1x16xFP32(zmm2, 2, 6)
		STORE_WORKSPACE_1x16xFP32(zmm3, 3, 6)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rax (src), moving to next column
		add(imm(1*16*4), rbx)// add 16*4 (16 floats) to rbx (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x16)
		sub(imm(6*1*8), rax)// add 1*8 (1 pointer) to rax (src), moving to next column
		sub(imm(6*1*16*4), rbx)// add 16*4 (16 floats) to rbx (workspace), moving to next column

		movq(imm(4), r8)// transform row counter
		label(TRANSFORM2x16)
		// second transform
		LOAD_WORKSPACE_6x16xFP32()// load row
		OUTPUT_TRANSFORM_4x4_3x3()
		ADD_BIAS_4x16xFP32(zmm11)

		test(r11, r11)
		je(SKIP_LOAD_EXT_x16)
		LOAD_EXT_4x16xFP16(r9)
		ADD_EXT_4x16xFP32()
		label(SKIP_LOAD_EXT_x16)

		test(r12, r12)
		je(SKIP_RELU_x16)
		APPLY_RELU_4x16xFP32(zmm10)
		label(SKIP_RELU_x16)

		movq(mem(rcx, 0*8), r14)
		movq(mem(rcx, 1*8), r15)
		STORE_OUTPUT_1x16xFP16(r9, 0, r14)
		STORE_OUTPUT_1x16xFP16(r9, 1, r15)
		movq(mem(rcx, 2*8), r14)
		movq(mem(rcx, 3*8), r15)
		STORE_OUTPUT_1x16xFP16(r9, 2, r14)
		STORE_OUTPUT_1x16xFP16(r9, 3, r15)

		add(imm(6*16*4), rbx)// add 6*16 (6*16 floats) to rbx (workspace), moving to next row
		add(imm(4*8), rcx)// add 4*8 (4 pointers) to rcx (dst), moving to next row
		add(imm(4*8), rdx)// add 4*8 (4 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x16)
		sub(imm(4*6*16*4), rbx)// add 6*16 (6*16 floats) to rbx (workspace), moving to next row
		sub(imm(4*4*8), rcx)// add 4*8 (4 pointers) to rcx (dst), moving to next row
		sub(imm(4*4*8), rdx)// add 4*8 (4 pointers) to rdx (ext), moving to next row

		add(imm(16*2), r9)// add 16*2 (16 float16) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED16)

		/* END OF MAIN 16-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x1)
		movzw(mem(rdi, r9, 1), r15)\
		vmovq(r15, xmm11)\
		vcvtph2ps(xmm11, xmm11)
		label(SKIP_BIAS_LOAD_x1)

		movq(imm(6), r8)// transform col counter
		label(TRANSFORM1x1)

		LOAD_INPUT_6x1xFP16(r9)
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
		ADD_BIAS_4x16xFP32(zmm11)

		test(r11, r11)// check if external tensor pointer is not null
		je(SKIP_LOAD_EXT_x1)
		LOAD_EXT_4x1xFP16(r9)
		ADD_EXT_4x16xFP32()
		label(SKIP_LOAD_EXT_x1)

		test(r12, r12)// check if ReLU should be applied
		je(SKIP_RELU_x1)
		APPLY_RELU_4x16xFP32(zmm10)
		label(SKIP_RELU_x1)

		movq(mem(rcx, 0*8), r14)
		movq(mem(rcx, 1*8), r15)
		STORE_OUTPUT_1x1xFP16(r9, 0, r14)
		STORE_OUTPUT_1x1xFP16(r9, 1, r15)
		movq(mem(rcx, 2*8), r14)
		movq(mem(rcx, 3*8), r15)
		STORE_OUTPUT_1x1xFP16(r9, 2, r14)
		STORE_OUTPUT_1x1xFP16(r9, 3, r15)

		add(imm(6*1*4), rbx)// add 7*8 (7*1 floats) to rbx (workspace), moving to next row
		add(imm(4*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		add(imm(4*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)
		sub(imm(4*6*1*4), rbx)// add 7*8 (7*1 floats) to rbx (workspace), moving to next row
		sub(imm(4*4*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		sub(imm(4*4*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

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
				[c_ptr] "m"(c_ptr),
				[ext_ptr] "m"(ext_ptr),
				[bias_ptr] "m"(bias_ptr),
				[flag_relu] "m"(flag_relu)
				:// clobbers
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
	}
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP32
	 */
	void winograd_input_transform_5x5_3x3_avx512_fp32(const void *src[], void *dst[], void *workspace, int filters)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 16;
		const uint64_t k_left = filters % 16;

		const float constants[2] = { 0.25f, 0.5f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r8, 0), zmm14)// 0.25f
		vbroadcastss(mem(r8, 4), zmm15)// 0.5f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED16)// main loop over channels, in steps of 8 elements

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x16)
		LOAD_INPUT_7x16xFP32(r9)

		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_WORKSPACE_1x16xFP32(zmm7, 0, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_WORKSPACE_1x16xFP32(zmm7, 1, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_WORKSPACE_1x16xFP32(zmm7, 2, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_WORKSPACE_1x16xFP32(zmm7, 3, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_WORKSPACE_1x16xFP32(zmm7, 4, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_WORKSPACE_1x16xFP32(zmm7, 5, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_WORKSPACE_1x16xFP32(zmm7, 6, 7)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rax (src), moving to next column
		add(imm(1*16*4), rbx)// add 16*4 (16 floats) to rbx (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x16)
		sub(imm(7*8), rax)// subtract 7*8 (7 pointers) from rax, moving to the start
		sub(imm(7*16*4), rbx)// subtract 7*16*4 (7*16 float32) from rbx, moving to the start

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x16)
		// second transform
		LOAD_WORKSPACE_7x16xFP32()

		movq(mem(rcx, 0*8), r12)
		movq(mem(rcx, 1*8), r13)
		movq(mem(rcx, 2*8), r14)
		movq(mem(rcx, 3*8), r15)
		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r15)

		movq(mem(rcx, 4*8), r12)
		movq(mem(rcx, 5*8), r13)
		movq(mem(rcx, 6*8), r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_OUTPUT_1x16xFP32(r9, zmm7, r14)

		add(imm(7*16*4), rbx)// add 7*16 (7*16 floats) to rbx (workspace), moving to next row
		add(imm(7*8), rcx)// add 7*8*4 (7 pointers) to rcx (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x16)
		sub(imm(7*7*16*4), rbx)// subtract 7*7*16*4 (7*7*16 float32) from rbx, moving to the start
		sub(imm(7*7*8), rcx)// subtract 7*8*4 (7*7*8 pointers) from rcx, moving to the start

		add(imm(16*4), r9)// add 16*4 (16 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED16)

		/* END OF MAIN 8-UNROLLED LOOP */
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
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%r8", "%r9", "%r10", "%r12", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_5x5_3x3_avx512_fp32(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		const void **ext_ptr = ext;
		const void *bias_ptr = bias;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 16;
		const uint64_t k_left = filters % 16;
		const uint64_t flag_relu = static_cast<uint64_t>(use_relu);

		const float constants[4] = { 0.25f, 0.5f, 2.0f, 4.0f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		vbroadcastss(mem(r8, 0), zmm12)// 0.25f
		vbroadcastss(mem(r8, 4), zmm13)// 0.5f
		vbroadcastss(mem(r8, 8), zmm14)// 2.0f
		vbroadcastss(mem(r8, 12), zmm15)// 4.0f

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

		label(UNROLLED16)// main loop over channels, in steps of 8 elements

		vxorps(zmm11, zmm11, zmm11)
		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x16)
		vmovups(mem(rdi, r9, 1), zmm11)// load bias
		label(SKIP_BIAS_LOAD_x16)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x16)
		LOAD_INPUT_7x16xFP32(r9)// load column
		OUTPUT_TRANSFORM_5x5_3x3()
		STORE_WORKSPACE_1x16xFP32(zmm0, 0, 7)
		STORE_WORKSPACE_1x16xFP32(zmm1, 1, 7)
		STORE_WORKSPACE_1x16xFP32(zmm2, 2, 7)
		STORE_WORKSPACE_1x16xFP32(zmm3, 3, 7)
		STORE_WORKSPACE_1x16xFP32(zmm4, 4, 7)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*16*4), rbx)// add 16*4 (16 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x16)
		sub(imm(7*1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		sub(imm(7*1*16*4), rbx)// add 16*4 (16 floats) to rdi (workspace), moving to next column

		movq(imm(5), r8)// transform row counter
		label(TRANSFORM2x16)
		// second transform
		LOAD_WORKSPACE_7x16xFP32()// load row
		OUTPUT_TRANSFORM_5x5_3x3()
		ADD_BIAS_5x16xFP32(zmm11)

		test(r11, r11)
		je(SKIP_LOAD_EXT_x16)
		LOAD_EXT_5x16xFP32(r9)
		ADD_EXT_5x16xFP32()
		label(SKIP_LOAD_EXT_x16)

		test(r12, r12)
		je(SKIP_RELU_x16)
		APPLY_RELU_5x16xFP32(zmm10)
		label(SKIP_RELU_x16)

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		STORE_OUTPUT_1x16xFP32(r9, zmm0, r13)
		STORE_OUTPUT_1x16xFP32(r9, zmm1, r14)
		STORE_OUTPUT_1x16xFP32(r9, zmm2, r15)
		movq(mem(rcx, 3*8), r14)
		movq(mem(rcx, 4*8), r15)
		STORE_OUTPUT_1x16xFP32(r9, zmm3, r14)
		STORE_OUTPUT_1x16xFP32(r9, zmm4, r15)

		add(imm(7*16*4), rbx)// add 7*16 (7*16 floats) to rbx (workspace), moving to next row
		add(imm(5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		add(imm(5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x16)
		sub(imm(5*7*16*4), rbx)// add 7*16 (7*16 floats) to rbx (workspace), moving to next row
		sub(imm(5*5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		sub(imm(5*5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		add(imm(16*4), r9)// add 8*4 (8 float32) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED16)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x1)
		vmovss(mem(rdi, r9, 1), xmm11)// load bias
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
		ADD_BIAS_5x16xFP32(zmm11)

		test(r11, r11)// check if external tensor pointer is not null
		je(SKIP_LOAD_EXT_x1)
		LOAD_EXT_5x1xFP32(r9)
		ADD_EXT_5x16xFP32()
		label(SKIP_LOAD_EXT_x1)

		test(r12, r12)// check if ReLU should be applied
		je(SKIP_RELU_x1)
		APPLY_RELU_5x16xFP32(zmm10)
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
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
	}
	/*
	 * Transforms for 3x3 kernel and 5x5 tile size in FP16
	 */
	void winograd_input_transform_5x5_3x3_avx512_fp16(const void *src[], void *dst[], void *workspace, int filters)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 16;
		const uint64_t k_left = filters % 16;

		const float constants[2] = { 0.25f, 0.5f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		movq(var(src_ptr), rax)
		movq(var(workspace_ptr), rbx)
		movq(var(dst_ptr), rcx)

		vbroadcastss(mem(r8, 0), zmm14)// 0.25f
		vbroadcastss(mem(r8, 4), zmm15)// 0.5f

		movq(imm(0), r9)// channel offset

		movq(var(k_iter), r10)// load the number of 8-unrolled iterations
		test(r10, r10)
		je(FINALLOOP)

		label(UNROLLED16)// main loop over channels, in steps of 8 elements

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x16)
		LOAD_INPUT_7x16xFP16(r9)

		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_WORKSPACE_1x16xFP32(zmm7, 0, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_WORKSPACE_1x16xFP32(zmm7, 1, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_WORKSPACE_1x16xFP32(zmm7, 2, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_WORKSPACE_1x16xFP32(zmm7, 3, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_WORKSPACE_1x16xFP32(zmm7, 4, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_WORKSPACE_1x16xFP32(zmm7, 5, 7)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_WORKSPACE_1x16xFP32(zmm7, 6, 7)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rax (src), moving to next column
		add(imm(1*16*4), rbx)// add 16*4 (16 floats) to rbx (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x16)
		sub(imm(7*8), rax)// subtract 7*8 (7 pointers) from rax, moving to the start
		sub(imm(7*16*4), rbx)// subtract 7*16*4 (7*16 float32) from rbx, moving to the start

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM2x16)
		// second transform
		LOAD_WORKSPACE_7x16xFP32()

		movq(mem(rcx, 0*8), r12)
		movq(mem(rcx, 1*8), r13)
		movq(mem(rcx, 2*8), r14)
		movq(mem(rcx, 3*8), r15)
		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_OUTPUT_1x16xFP16(r9, 7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_OUTPUT_1x16xFP16(r9, 7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_OUTPUT_1x16xFP16(r9, 7, r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_OUTPUT_1x16xFP16(r9, 7, r15)

		movq(mem(rcx, 4*8), r12)
		movq(mem(rcx, 5*8), r13)
		movq(mem(rcx, 6*8), r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_OUTPUT_1x16xFP16(r9, 7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_OUTPUT_1x16xFP16(r9, 7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_OUTPUT_1x16xFP16(r9, 7, r14)

		add(imm(7*16*4), rbx)// add 7*16 (7*16 floats) to rbx (workspace), moving to next row
		add(imm(7*8), rcx)// add 7*8*4 (7 pointers) to rcx (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x16)
		sub(imm(7*7*16*4), rbx)// subtract 7*7*16*4 (7*7*16 float32) from rbx, moving to the start
		sub(imm(7*7*8), rcx)// subtract 7*8*4 (7*7*8 pointers) from rcx, moving to the start

		add(imm(16*2), r9)// add 16*2 (16 float16) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED16)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

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
		INPUT_TRANSFORM_5x5_3x3_ROW_0()
		STORE_OUTPUT_1x1xFP16(r9, 7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_1()
		STORE_OUTPUT_1x1xFP16(r9, 7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_2()
		STORE_OUTPUT_1x1xFP16(r9, 7, r14)

		movq(mem(rcx, 3*8), r12)
		movq(mem(rcx, 4*8), r13)
		movq(mem(rcx, 5*8), r14)
		INPUT_TRANSFORM_5x5_3x3_ROW_3()
		STORE_OUTPUT_1x1xFP16(r9, 7, r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_4()
		STORE_OUTPUT_1x1xFP16(r9, 7, r13)
		INPUT_TRANSFORM_5x5_3x3_ROW_5()
		STORE_OUTPUT_1x1xFP16(r9, 7, r14)

		movq(mem(rcx, 6*8), r12)
		INPUT_TRANSFORM_5x5_3x3_ROW_6()
		STORE_OUTPUT_1x1xFP16(r9, 7, r12)

		add(imm(7*1*4), rbx)// add 7*8 (7*1 floats) to rbx (workspace), moving to next row
		add(imm(7*8), rcx)// add 7*8*4 (7 pointers) to rcx (dst), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)
		sub(imm(7*7*1*4), rbx)// subtract 7*7*1*4 (7*7*1 float32) from rbx, moving to the start
		sub(imm(7*7*8), rcx)// subtract 7*8*4 (7*7*8 pointers) from rcx, moving to the start

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
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%rsi", "%r8", "%r9", "%r10", "%r12", "%r13", "%r14", "%r15")
	}
	void winograd_output_transform_5x5_3x3_avx512_fp16(const void *src[], void *dst[], void *workspace, int filters, const void *ext[],
			const void *bias, bool use_relu)
	{
		assert(workspace != nullptr);
		assert(filters >= 0);

		const void **src_ptr = src;
		void **dst_ptr = dst;
		const void **ext_ptr = ext;
		const void *bias_ptr = bias;
		void *workspace_ptr = workspace;

		const uint64_t k_iter = filters / 16;
		const uint64_t k_left = filters % 16;
		const uint64_t flag_relu = static_cast<uint64_t>(use_relu);

		const float constants[4] = { 0.25f, 0.5f, 2.0f, 4.0f };
		const void *c_ptr = constants;

		begin_asm()

		movq(var(c_ptr), r8) // table of constants
		vbroadcastss(mem(r8, 0), zmm12)// 0.25f
		vbroadcastss(mem(r8, 4), zmm13)// 0.5f
		vbroadcastss(mem(r8, 8), zmm14)// 2.0f
		vbroadcastss(mem(r8, 12), zmm15)// 4.0f

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

		label(UNROLLED16)// main loop over channels, in steps of 8 elements

		vxorps(zmm11, zmm11, zmm11)
		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x16)
		vmovups(mem(rdi, r9, 1), ymm11)// load bias
		vcvtph2ps(ymm11, zmm11)// convert to fp32
		label(SKIP_BIAS_LOAD_x16)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x16)
		LOAD_INPUT_7x16xFP16(r9)// load column
		OUTPUT_TRANSFORM_5x5_3x3()
		STORE_WORKSPACE_1x16xFP32(zmm0, 0, 7)
		STORE_WORKSPACE_1x16xFP32(zmm1, 1, 7)
		STORE_WORKSPACE_1x16xFP32(zmm2, 2, 7)
		STORE_WORKSPACE_1x16xFP32(zmm3, 3, 7)
		STORE_WORKSPACE_1x16xFP32(zmm4, 4, 7)

		add(imm(1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		add(imm(1*16*4), rbx)// add 16*4 (16 floats) to rdi (workspace), moving to next column

		dec(r8)
		jne(TRANSFORM1x16)
		sub(imm(7*1*8), rax)// add 1*8 (1 pointer) to rsi (src), moving to next column
		sub(imm(7*1*16*4), rbx)// add 16*4 (16 floats) to rdi (workspace), moving to next column

		movq(imm(5), r8)// transform row counter
		label(TRANSFORM2x16)
		// second transform
		LOAD_WORKSPACE_7x16xFP32()// load row
		OUTPUT_TRANSFORM_5x5_3x3()
		ADD_BIAS_5x16xFP32(zmm11)

		test(r11, r11)
		je(SKIP_LOAD_EXT_x16)
		LOAD_EXT_5x16xFP16(r9)
		ADD_EXT_5x16xFP32()
		label(SKIP_LOAD_EXT_x16)

		test(r12, r12)
		je(SKIP_RELU_x16)
		APPLY_RELU_5x16xFP32(zmm10)
		label(SKIP_RELU_x16)

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		STORE_OUTPUT_1x16xFP16(r9, 0, r13)
		STORE_OUTPUT_1x16xFP16(r9, 1, r14)
		STORE_OUTPUT_1x16xFP16(r9, 2, r15)
		movq(mem(rcx, 3*8), r14)
		movq(mem(rcx, 4*8), r15)
		STORE_OUTPUT_1x16xFP16(r9, 3, r14)
		STORE_OUTPUT_1x16xFP16(r9, 4, r15)

		add(imm(7*16*4), rbx)// add 7*16 (7*16 floats) to rbx (workspace), moving to next row
		add(imm(5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		add(imm(5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x16)
		sub(imm(5*7*16*4), rbx)// subtract 5*7*16 (5*7*16 floats) to rbx (workspace), moving to next row
		sub(imm(5*5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		sub(imm(5*5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		add(imm(16*2), r9)// add 8*2 (8 float16) to r9, the offset in channels

		dec(r10)
		jne(UNROLLED16)

		/* END OF MAIN 8-UNROLLED LOOP */
		label(FINALLOOP)
		movq(var(k_left), r10) // load the number of 1-unrolled iterations
		test(r10, r10)
		je(EPILOGUE)

		label(UNROLLED1)

		test(rdi, rdi)
		je(SKIP_BIAS_LOAD_x1)
		movzw(mem(rdi, r9, 1), r15)\
		vmovq(r15, xmm11)\
		vcvtph2ps(xmm11, xmm11)
		label(SKIP_BIAS_LOAD_x1)

		movq(imm(7), r8)// transform col counter
		label(TRANSFORM1x1)

		LOAD_INPUT_7x1xFP16(r9)
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
		ADD_BIAS_5x16xFP32(zmm11)

		test(r11, r11)// check if external tensor pointer is not null
		je(SKIP_LOAD_EXT_x1)
		LOAD_EXT_5x1xFP16(r9)
		ADD_EXT_5x16xFP32()
		label(SKIP_LOAD_EXT_x1)

		test(r12, r12)// check if ReLU should be applied
		je(SKIP_RELU_x1)
		APPLY_RELU_5x16xFP32(zmm10)
		label(SKIP_RELU_x1)

		movq(mem(rcx, 0*8), r13)
		movq(mem(rcx, 1*8), r14)
		movq(mem(rcx, 2*8), r15)
		STORE_OUTPUT_1x1xFP16(r9, 0, r13)
		STORE_OUTPUT_1x1xFP16(r9, 1, r14)
		STORE_OUTPUT_1x1xFP16(r9, 2, r15)
		movq(mem(rcx, 3*8), r13)
		movq(mem(rcx, 4*8), r14)
		STORE_OUTPUT_1x1xFP16(r9, 3, r13)
		STORE_OUTPUT_1x1xFP16(r9, 4, r14)

		add(imm(7*1*4), rbx)// add 7*8 (7*1 floats) to rbx (workspace), moving to next row
		add(imm(5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		add(imm(5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

		dec(r8)
		jne(TRANSFORM2x1)
		sub(imm(5*7*1*4), rbx)// add 7*8 (7*1 floats) to rbx (workspace), moving to next row
		sub(imm(5*5*8), rcx)// add 5*8 (5 pointers) to rcx (dst), moving to next row
		sub(imm(5*5*8), rdx)// add 5*8 (5 pointers) to rdx (ext), moving to next row

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
				[c_ptr] "m"(c_ptr),
				[ext_ptr] "m"(ext_ptr),
				[bias_ptr] "m"(bias_ptr),
				[flag_relu] "m"(flag_relu)
				:// clobbers
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
				"%zmm13", "%zmm14", "%zmm15", "%rax", "%rbx", "%rcx", "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
	}

} /* namespace ml */

