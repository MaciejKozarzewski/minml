/*
 * avx512f_gemm_kernels.cpp
 *
 *  Created on: Sep 24, 2023
 *      Author: Maciej Kozarzewski
 */

#include "Fragment.hpp"
#include "Matrix.hpp"
#include "gemm_kernels.hpp"
#include "../utils.hpp"
#include "../fp16.hpp"

#include <cinttypes>
#include <cassert>

#include "../assembly_macros.hpp"

#define ZERO_ACCUMULATORS()\
	vxorps(zmm8, zmm8, zmm8)\
	vxorps(zmm9, zmm9, zmm9)\
	vxorps(zmm10, zmm10, zmm10)\
	vxorps(zmm11, zmm11, zmm11)\
	vxorps(zmm12, zmm12, zmm12)\
	vxorps(zmm13, zmm13, zmm13)\
	vxorps(zmm14, zmm14, zmm14)\
	vxorps(zmm15, zmm15, zmm15)\
	vxorps(zmm16, zmm16, zmm16)\
	vxorps(zmm17, zmm17, zmm17)\
	vxorps(zmm18, zmm18, zmm18)\
	vxorps(zmm19, zmm19, zmm19)\
	vxorps(zmm20, zmm20, zmm20)\
	vxorps(zmm21, zmm21, zmm21)\
	vxorps(zmm22, zmm22, zmm22)\
	vxorps(zmm23, zmm23, zmm23)\
	vxorps(zmm24, zmm24, zmm24)\
	vxorps(zmm25, zmm25, zmm25)\
	vxorps(zmm26, zmm26, zmm26)\
	vxorps(zmm27, zmm27, zmm27)\
	vxorps(zmm28, zmm28, zmm28)\
	vxorps(zmm29, zmm29, zmm29)\
	vxorps(zmm30, zmm30, zmm30)\
	vxorps(zmm31, zmm31, zmm31)

#define SUB_KERNEL_24xFP32_16xFP32(n) \
	vmovaps(mem(rbx, n*16*4), zmm6)\
	vpermilps(imm(0xB1), zmm6, zmm7)\
	vbroadcastsd(mem(rax, (24*n+0)*4), zmm0)\
	vbroadcastsd(mem(rax, (24*n+2)*4), zmm1)\
	vbroadcastsd(mem(rax, (24*n+4)*4), zmm2)\
	vbroadcastsd(mem(rax, (24*n+6)*4), zmm3)\
	vbroadcastsd(mem(rax, (24*n+8)*4), zmm4)\
	vbroadcastsd(mem(rax, (24*n+10)*4), zmm5)\
	vfmadd231ps(zmm0, zmm6, zmm8)\
	vfmadd231ps(zmm0, zmm7, zmm9)\
	vfmadd231ps(zmm1, zmm6, zmm10)\
	vfmadd231ps(zmm1, zmm7, zmm11)\
	vfmadd231ps(zmm2, zmm6, zmm12)\
	vfmadd231ps(zmm2, zmm7, zmm13)\
	vfmadd231ps(zmm3, zmm6, zmm14)\
	vfmadd231ps(zmm3, zmm7, zmm15)\
	vfmadd231ps(zmm4, zmm6, zmm16)\
	vfmadd231ps(zmm4, zmm7, zmm17)\
	vfmadd231ps(zmm5, zmm6, zmm18)\
	vfmadd231ps(zmm5, zmm7, zmm19)\
	vbroadcastsd(mem(rax, (24*n+12)*4), zmm0)\
	vbroadcastsd(mem(rax, (24*n+14)*4), zmm1)\
	vbroadcastsd(mem(rax, (24*n+16)*4), zmm2)\
	vbroadcastsd(mem(rax, (24*n+18)*4), zmm3)\
	vbroadcastsd(mem(rax, (24*n+20)*4), zmm4)\
	vbroadcastsd(mem(rax, (24*n+22)*4), zmm5)\
	vfmadd231ps(zmm0, zmm6, zmm20)\
	vfmadd231ps(zmm0, zmm7, zmm21)\
	vfmadd231ps(zmm1, zmm6, zmm22)\
	vfmadd231ps(zmm1, zmm7, zmm23)\
	vfmadd231ps(zmm2, zmm6, zmm24)\
	vfmadd231ps(zmm2, zmm7, zmm25)\
	vfmadd231ps(zmm3, zmm6, zmm26)\
	vfmadd231ps(zmm3, zmm7, zmm27)\
	vfmadd231ps(zmm4, zmm6, zmm28)\
	vfmadd231ps(zmm4, zmm7, zmm29)\
	vfmadd231ps(zmm5, zmm6, zmm30)\
	vfmadd231ps(zmm5, zmm7, zmm31)

#define LOAD_ADD_3x16xFP32(beta, reg0, reg1, reg2)\
	vmovups(mem(rcx), zmm1)\
	vmovups(mem(rcx, r14, 1), zmm2)\
	vmovups(mem(rcx, r14, 2), zmm3)\
	vfmadd231ps(zmm1, beta, reg0)\
	vfmadd231ps(zmm2, beta, reg1)\
	vfmadd231ps(zmm3, beta, reg2)\
	add(r15, rcx)

#define LOAD_ADD_16xFP16(beta, reg)\
	vmovups(mem(rcx), ymm2)\
	vcvtph2ps(ymm2, zmm2)\
	vfmadd231ps(zmm2, beta, reg)\
	add(r14, rcx)

#define STORE_3x16xFP32(reg0, reg1, reg2)\
	vmovups(reg0, mem(rcx))\
	vmovups(reg1, mem(rcx, r14, 1))\
	vmovups(reg2, mem(rcx, r14, 2))\
	add(r15, rcx)

#define PERMUTE_AND_SCALE_6x16xFP32(reg0, reg1, reg2, reg3, reg4, reg5)\
	movq(imm(0x5555), rdx)\
	kmovw(edx, k(1))\
	movq(imm(0xAAAA), rdx)\
	kmovw(edx, k(2))\
	vpermilps(imm(0xB1), reg1, reg1)\
	vpermilps(imm(0xB1), reg3, reg3)\
	vpermilps(imm(0xB1), reg5, reg5)\
	vblendmps(reg0, reg1, zmm2 mask_k(1))\
	vblendmps(reg0, reg1, reg1 mask_k(2))\
	vblendmps(reg2, reg3, zmm3 mask_k(1))\
	vblendmps(reg2, reg3, reg3 mask_k(2))\
	vblendmps(reg4, reg5, zmm4 mask_k(1))\
	vblendmps(reg4, reg5, reg5 mask_k(2))\
	vmulps(zmm1, zmm2, reg0)\
	vmulps(zmm1, reg1, reg1)\
	vmulps(zmm1, zmm3, reg2)\
	vmulps(zmm1, reg3, reg3)\
	vmulps(zmm1, zmm4, reg4)\
	vmulps(zmm1, reg5, reg5)

#define PERMUTE_8x16xFP32(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7)\
	movq(imm(0x5555), rdx) \
	kmovw(edx, k(1)) \
	movq(imm(0xAAAA), rdx) \
	kmovw(edx, k(2)) \
	vpermilps(imm(0xB1), reg1, zmm0) \
	vpermilps(imm(0xB1), reg3, zmm1) \
	vpermilps(imm(0xB1), reg5, zmm2) \
	vpermilps(imm(0xB1), reg7, zmm3) \
	vblendmps(reg0, zmm0, reg1 mask_k(2)) \
	vblendmps(reg0, zmm0, reg0 mask_k(1)) \
	vblendmps(reg2, zmm1, reg3 mask_k(2)) \
	vblendmps(reg2, zmm1, reg2 mask_k(1)) \
	vblendmps(reg4, zmm2, reg5 mask_k(2)) \
	vblendmps(reg4, zmm2, reg4 mask_k(1)) \
	vblendmps(reg6, zmm3, reg7 mask_k(2)) \
	vblendmps(reg6, zmm3, reg6 mask_k(1))
#define SCALE_ACCUMULATORS_1x1() \
	vbroadcastss(mem(rax), zmm0) \
	vmulps(zmm0, zmm8, zmm8) \
	vmulps(zmm0, zmm9, zmm9) \
	vmulps(zmm0, zmm10, zmm10) \
	vmulps(zmm0, zmm11, zmm11) \
	vmulps(zmm0, zmm12, zmm12) \
	vmulps(zmm0, zmm13, zmm13) \
	vmulps(zmm0, zmm14, zmm14) \
	vmulps(zmm0, zmm15, zmm15) \
	vmulps(zmm0, zmm16, zmm16) \
	vmulps(zmm0, zmm17, zmm17) \
	vmulps(zmm0, zmm18, zmm18) \
	vmulps(zmm0, zmm19, zmm19) \
	vmulps(zmm0, zmm20, zmm20) \
	vmulps(zmm0, zmm21, zmm21) \
	vmulps(zmm0, zmm22, zmm22) \
	vmulps(zmm0, zmm23, zmm23) \
	vmulps(zmm0, zmm24, zmm24) \
	vmulps(zmm0, zmm25, zmm25) \
	vmulps(zmm0, zmm26, zmm26) \
	vmulps(zmm0, zmm27, zmm27) \
	vmulps(zmm0, zmm28, zmm28) \
	vmulps(zmm0, zmm29, zmm29) \
	vmulps(zmm0, zmm30, zmm30) \
	vmulps(zmm0, zmm31, zmm31)
#define SCALE_ACCUMULATORS_24x1() \
	vbroadcastss(mem(rax, 0*4), zmm0) \
	vbroadcastss(mem(rax, 1*4), zmm1) \
	vbroadcastss(mem(rax, 2*4), zmm2) \
	vbroadcastss(mem(rax, 3*4), zmm3) \
	vbroadcastss(mem(rax, 4*4), zmm4) \
	vbroadcastss(mem(rax, 5*4), zmm5) \
	vbroadcastss(mem(rax, 6*4), zmm6) \
	vbroadcastss(mem(rax, 7*4), zmm7) \
	vmulps(zmm0, zmm8, zmm8) \
	vmulps(zmm1, zmm9, zmm9) \
	vmulps(zmm2, zmm10, zmm10) \
	vmulps(zmm3, zmm11, zmm11) \
	vmulps(zmm4, zmm12, zmm12) \
	vmulps(zmm5, zmm13, zmm13) \
	vmulps(zmm6, zmm14, zmm14) \
	vmulps(zmm7, zmm15, zmm15) \
	vbroadcastss(mem(rax, 8*4), zmm0) \
	vbroadcastss(mem(rax, 9*4), zmm1) \
	vbroadcastss(mem(rax, 10*4), zmm2) \
	vbroadcastss(mem(rax, 11*4), zmm3) \
	vbroadcastss(mem(rax, 12*4), zmm4) \
	vbroadcastss(mem(rax, 13*4), zmm5) \
	vbroadcastss(mem(rax, 14*4), zmm6) \
	vbroadcastss(mem(rax, 15*4), zmm7) \
	vmulps(zmm0, zmm16, zmm16) \
	vmulps(zmm1, zmm17, zmm17) \
	vmulps(zmm2, zmm18, zmm18) \
	vmulps(zmm3, zmm19, zmm19) \
	vmulps(zmm4, zmm20, zmm20) \
	vmulps(zmm5, zmm21, zmm21) \
	vmulps(zmm6, zmm22, zmm22) \
	vmulps(zmm7, zmm23, zmm23) \
	vbroadcastss(mem(rax, 16*4), zmm0) \
	vbroadcastss(mem(rax, 17*4), zmm1) \
	vbroadcastss(mem(rax, 18*4), zmm2) \
	vbroadcastss(mem(rax, 19*4), zmm3) \
	vbroadcastss(mem(rax, 20*4), zmm4) \
	vbroadcastss(mem(rax, 21*4), zmm5) \
	vbroadcastss(mem(rax, 22*4), zmm6) \
	vbroadcastss(mem(rax, 23*4), zmm7) \
	vmulps(zmm0, zmm24, zmm24) \
	vmulps(zmm1, zmm25, zmm25) \
	vmulps(zmm2, zmm26, zmm26) \
	vmulps(zmm3, zmm27, zmm27) \
	vmulps(zmm4, zmm28, zmm28) \
	vmulps(zmm5, zmm29, zmm29) \
	vmulps(zmm6, zmm30, zmm30) \
	vmulps(zmm7, zmm31, zmm31)
#define CONVERT_ACCUMULATORS_TO_FP16()\
	vcvtps2ph(imm(0x03), zmm8, ymm8)\
	vcvtps2ph(imm(0x03), zmm9, ymm9)\
	vcvtps2ph(imm(0x03), zmm10, ymm10)\
	vcvtps2ph(imm(0x03), zmm11, ymm11)\
	vcvtps2ph(imm(0x03), zmm12, ymm12)\
	vcvtps2ph(imm(0x03), zmm13, ymm13)\
	vcvtps2ph(imm(0x03), zmm14, ymm14)\
	vcvtps2ph(imm(0x03), zmm15, ymm15)\
	vcvtps2ph(imm(0x03), zmm16, ymm16)\
	vcvtps2ph(imm(0x03), zmm17, ymm17)\
	vcvtps2ph(imm(0x03), zmm18, ymm18)\
	vcvtps2ph(imm(0x03), zmm19, ymm19)\
	vcvtps2ph(imm(0x03), zmm20, ymm20)\
	vcvtps2ph(imm(0x03), zmm21, ymm21)\
	vcvtps2ph(imm(0x03), zmm22, ymm22)\
	vcvtps2ph(imm(0x03), zmm23, ymm23)\
	vcvtps2ph(imm(0x03), zmm24, ymm24)\
	vcvtps2ph(imm(0x03), zmm25, ymm25)\
	vcvtps2ph(imm(0x03), zmm26, ymm26)\
	vcvtps2ph(imm(0x03), zmm27, ymm27)\
	vcvtps2ph(imm(0x03), zmm28, ymm28)\
	vcvtps2ph(imm(0x03), zmm29, ymm29)\
	vcvtps2ph(imm(0x03), zmm30, ymm30)\
	vcvtps2ph(imm(0x03), zmm31, ymm31)

#define LOAD_ADD_3x16xFP16(beta, reg0, reg1, reg2)\
	vmovups(mem(rcx), ymm1)\
	vmovups(mem(rcx, r14, 1), ymm2)\
	vmovups(mem(rcx, r14, 2), ymm3)\
	vcvtph2ps(ymm1, zmm1)\
	vcvtph2ps(ymm2, zmm2)\
	vcvtph2ps(ymm3, zmm3)\
	vfmadd231ps(zmm1, beta, reg0)\
	vfmadd231ps(zmm2, beta, reg1)\
	vfmadd231ps(zmm3, beta, reg2)\
	add(r15, rcx)

#define STORE_3x16xFP16(reg0, reg1, reg2)\
	vmovups(reg0, mem(rcx))\
	vmovups(reg1, mem(rcx, r14, 1))\
	vmovups(reg2, mem(rcx, r14, 2))\
	add(r15, rcx)

#define ADD_BIAS_24x16xFP32(reg)\
	vaddps(reg, zmm8, zmm8)\
	vaddps(reg, zmm9, zmm9)\
	vaddps(reg, zmm10, zmm10)\
	vaddps(reg, zmm11, zmm11)\
	vaddps(reg, zmm12, zmm12)\
	vaddps(reg, zmm13, zmm13)\
	vaddps(reg, zmm14, zmm14)\
	vaddps(reg, zmm15, zmm15)\
	vaddps(reg, zmm16, zmm16)\
	vaddps(reg, zmm17, zmm17)\
	vaddps(reg, zmm18, zmm18)\
	vaddps(reg, zmm19, zmm19)\
	vaddps(reg, zmm20, zmm20)\
	vaddps(reg, zmm21, zmm21)\
	vaddps(reg, zmm22, zmm22)\
	vaddps(reg, zmm23, zmm23)\
	vaddps(reg, zmm24, zmm24)\
	vaddps(reg, zmm25, zmm25)\
	vaddps(reg, zmm26, zmm26)\
	vaddps(reg, zmm27, zmm27)\
	vaddps(reg, zmm28, zmm28)\
	vaddps(reg, zmm29, zmm29)\
	vaddps(reg, zmm30, zmm30)\
	vaddps(reg, zmm31, zmm31)

#define RELU_24x16xFP32()\
	vxorps(zmm0, zmm0, zmm0)\
	vmaxps(zmm0, zmm8, zmm8)\
	vmaxps(zmm0, zmm9, zmm9)\
	vmaxps(zmm0, zmm10, zmm10)\
	vmaxps(zmm0, zmm11, zmm11)\
	vmaxps(zmm0, zmm12, zmm12)\
	vmaxps(zmm0, zmm13, zmm13)\
	vmaxps(zmm0, zmm14, zmm14)\
	vmaxps(zmm0, zmm15, zmm15)\
	vmaxps(zmm0, zmm16, zmm16)\
	vmaxps(zmm0, zmm17, zmm17)\
	vmaxps(zmm0, zmm18, zmm18)\
	vmaxps(zmm0, zmm19, zmm19)\
	vmaxps(zmm0, zmm20, zmm20)\
	vmaxps(zmm0, zmm21, zmm21)\
	vmaxps(zmm0, zmm22, zmm22)\
	vmaxps(zmm0, zmm23, zmm23)\
	vmaxps(zmm0, zmm24, zmm24)\
	vmaxps(zmm0, zmm25, zmm25)\
	vmaxps(zmm0, zmm26, zmm26)\
	vmaxps(zmm0, zmm27, zmm27)\
	vmaxps(zmm0, zmm28, zmm28)\
	vmaxps(zmm0, zmm29, zmm29)\
	vmaxps(zmm0, zmm30, zmm30)\
	vmaxps(zmm0, zmm31, zmm31)

/*
 * FP16 -> FP32 conversion
 */
#define CONVERT_1x16xFP16_TO_FP32(reg) \
	vcvtph2ps(ymm(reg), zmm(reg))
#define CONVERT_1x24xFP16_TO_FP32(reg0, reg1) \
	vcvtph2ps(ymm(reg0), zmm(reg0)) \
	vcvtph2ps(xmm(reg1), ymm(reg1))

#define CONVERT_4x16xFP16_TO_FP32(reg0, reg1, reg2, reg3) \
	CONVERT_1x16xFP16_TO_FP32(reg0) \
	CONVERT_1x16xFP16_TO_FP32(reg1) \
	CONVERT_1x16xFP16_TO_FP32(reg2) \
	CONVERT_1x16xFP16_TO_FP32(reg3)
#define CONVERT_4x24xFP16_TO_FP32(reg00, reg01, reg10, reg11, reg20, reg21, reg30, reg31) \
	CONVERT_1x24xFP16_TO_FP32(reg00, reg01) \
	CONVERT_1x24xFP16_TO_FP32(reg10, reg11) \
	CONVERT_1x24xFP16_TO_FP32(reg20, reg21) \
	CONVERT_1x24xFP16_TO_FP32(reg30, reg31)

/*
 * FP16 loads
 */
#define LOAD_4x1xFP16(ptr, reg0, reg1, reg2, reg3)\
	movzw(mem(ptr), r8) \
	movzw(mem(ptr, r12, 1), r9) \
	movzw(mem(ptr, r12, 2), r10) \
	movzw(mem(ptr, r13, 1), r11) \
	add(r15, ptr) \
	vmovq(r8, xmm(reg0)) \
	vmovq(r9, xmm(reg1)) \
	vmovq(r10, xmm(reg2)) \
	vmovq(r11, xmm(reg3)) \
	vcvtph2ps(xmm(reg0), xmm(reg0)) \
	vcvtph2ps(xmm(reg1), xmm(reg1)) \
	vcvtph2ps(xmm(reg2), xmm(reg2)) \
	vcvtph2ps(xmm(reg3), xmm(reg3))
#define LOAD_1x16xFP16(ptr, reg)\
	vmovups(mem(ptr), reg) \
	add(r12, ptr)
#define LOAD_1x24xFP16(ptr, reg0, reg1) \
	vmovups(mem(ptr, 0*2), reg0) \
	vmovups(mem(ptr, 16*2), reg1) \
	add(r12, ptr)

#define LOAD_4x16xFP16(ptr, reg0, reg1, reg2, reg3) \
	vmovups(mem(ptr), reg0) \
	vmovups(mem(ptr, r12, 1), reg1) \
	vmovups(mem(ptr, r12, 2), reg2) \
	vmovups(mem(ptr, r13, 1), reg3) \
	add(r15, ptr)
#define LOAD_4x24xFP16(ptr, reg00, reg01, reg10, reg11, reg20, reg21, reg30, reg31) \
	vmovups(mem(ptr), reg00) \
	vmovups(mem(ptr, 16*2), reg01) \
	vmovups(mem(ptr, r12, 1), reg10) \
	vmovups(mem(ptr, r12, 1, 16*2), reg11) \
	vmovups(mem(ptr, r12, 2), reg20) \
	vmovups(mem(ptr, r12, 2, 16*2), reg21) \
	vmovups(mem(ptr, r13, 1), reg30) \
	vmovups(mem(ptr, r13, 1, 16*2), reg31) \
	add(r15, ptr)

#define LOAD_8x16xFP16(ptr) \
	LOAD_4x16xFP16(ptr, ymm24, ymm25, ymm26, ymm27) \
	LOAD_4x16xFP16(ptr, ymm28, ymm29, ymm30, ymm31) \
	CONVERT_4x16xFP16_TO_FP32(24, 25, 26, 27) \
	CONVERT_4x16xFP16_TO_FP32(28, 29, 30, 31)
#define LOAD_16x16xFP16(ptr) \
	LOAD_4x16xFP16(ptr, ymm8, ymm9, ymm10, ymm11) \
	LOAD_4x16xFP16(ptr, ymm12, ymm13, ymm14, ymm15) \
	LOAD_4x16xFP16(ptr, ymm16, ymm17, ymm18, ymm19) \
	LOAD_4x16xFP16(ptr, ymm20, ymm21, ymm22, ymm23) \
	CONVERT_4x16xFP16_TO_FP32(8, 9, 10, 11) \
	CONVERT_4x16xFP16_TO_FP32(12, 13, 14, 15) \
	CONVERT_4x16xFP16_TO_FP32(16, 17, 18, 19) \
	CONVERT_4x16xFP16_TO_FP32(20, 21, 22, 23)
#define LOAD_16x24xFP16(ptr) \
	LOAD_4x24xFP16(ptr, ymm0, xmm1, ymm2, xmm3, ymm4, xmm5, ymm6, xmm7) \
	LOAD_4x24xFP16(ptr, ymm8, xmm9, ymm10, xmm11, ymm12, xmm13, ymm14, xmm15) \
	LOAD_4x24xFP16(ptr, ymm16, xmm17, ymm18, xmm19, ymm20, xmm21, ymm22, xmm23) \
	LOAD_4x24xFP16(ptr, ymm24, xmm25, ymm26, xmm27, ymm28, xmm29, ymm30, xmm31) \
	CONVERT_4x24xFP16_TO_FP32(0, 1, 2, 3, 4, 5, 6, 7) \
	CONVERT_4x24xFP16_TO_FP32(8, 9, 10, 11, 12, 13, 14, 15) \
	CONVERT_4x24xFP16_TO_FP32(16, 17, 18, 19, 20, 21, 22, 23) \
	CONVERT_4x24xFP16_TO_FP32(24, 25, 26, 27, 28, 29, 30, 31)

/*
 * FP32 loads
 */
#define LOAD_4x1xFP32(ptr, reg0, reg1, reg2, reg3) \
	vmovss(mem(ptr), reg0) \
	vmovss(mem(ptr, r12, 1), reg1) \
	vmovss(mem(ptr, r12, 2), reg2) \
	vmovss(mem(ptr, r13, 1), reg3) \
	add(r15, ptr)
#define LOAD_1x16xFP32(ptr, reg) \
	vmovups(mem(ptr), reg) \
	add(r12, ptr)
#define LOAD_1x24xFP32(ptr, reg0, reg1) \
	vmovups(mem(ptr, 0*4), reg0) \
	vmovups(mem(ptr, 16*4), reg1) \
	add(r12, ptr)

#define LOAD_4x16xFP32(ptr, reg0, reg1, reg2, reg3) \
	vmovups(mem(ptr), reg0) \
	vmovups(mem(ptr, r12, 1), reg1) \
	vmovups(mem(ptr, r12, 2), reg2) \
	vmovups(mem(ptr, r13, 1), reg3) \
	add(r15, ptr)
#define LOAD_4x24xFP32(ptr, reg00, reg01, reg10, reg11, reg20, reg21, reg30, reg31) \
	vmovups(mem(ptr), reg00) \
	vmovups(mem(ptr, 16*4), reg01) \
	vmovups(mem(ptr, r12, 1), reg10) \
	vmovups(mem(ptr, r12, 1, 16*4), reg11) \
	vmovups(mem(ptr, r12, 2), reg20) \
	vmovups(mem(ptr, r12, 2, 16*4), reg21) \
	vmovups(mem(ptr, r13, 1), reg30) \
	vmovups(mem(ptr, r13, 1, 16*4), reg31) \
	add(r15, ptr)

#define LOAD_8x16xFP32(ptr) \
	LOAD_4x16xFP32(ptr, zmm24, zmm25, zmm26, zmm27) \
	LOAD_4x16xFP32(ptr, zmm28, zmm29, zmm30, zmm31)
#define LOAD_16x16xFP32(ptr) \
	LOAD_4x16xFP32(ptr, zmm8, zmm9, zmm10, zmm11) \
	LOAD_4x16xFP32(ptr, zmm12, zmm13, zmm14, zmm15) \
	LOAD_4x16xFP32(ptr, zmm16, zmm17, zmm18, zmm19) \
	LOAD_4x16xFP32(ptr, zmm20, zmm21, zmm22, zmm23)
#define LOAD_16x24xFP32(ptr) \
	LOAD_4x24xFP32(ptr, zmm0, ymm1, zmm2, ymm3, zmm4, ymm5, zmm6, ymm7) \
	LOAD_4x24xFP32(ptr, zmm8, ymm9, zmm10, ymm11, zmm12, ymm13, zmm14, ymm15) \
	LOAD_4x24xFP32(ptr, zmm16, ymm17, zmm18, ymm19, zmm20, ymm21, zmm22, ymm23) \
	LOAD_4x24xFP32(ptr, zmm24, ymm25, zmm26, ymm27, zmm28, ymm29, zmm30, ymm31)

/*
 * FP32 store
 */
#define STORE_4x1xFP32(offset, reg0, reg1, reg2, reg3) \
	vmovss(reg0, mem(rbx, 4*(offset+0))) \
	vmovss(reg1, mem(rbx, 4*(offset+1))) \
	vmovss(reg2, mem(rbx, 4*(offset+2))) \
	vmovss(reg3, mem(rbx, 4*(offset+3))) \

#define STORE_1x16xFP32(reg, row) \
	vmovups(reg, mem(rbx, row*16*4))
#define STORE_1x24xFP32(reg0, reg1, row)\
	vmovups(reg0, mem(rbx, (row*24+0)*4))\
	vmovaps(reg1, mem(rbx, (row*24+16)*4))

#define STORE_16x24xFP32() \
	STORE_1x24xFP32(zmm0, ymm1, 0) \
	STORE_1x24xFP32(zmm2, ymm3, 1) \
	STORE_1x24xFP32(zmm4, ymm5, 2) \
	STORE_1x24xFP32(zmm6, ymm7, 3) \
	STORE_1x24xFP32(zmm8, ymm9, 4) \
	STORE_1x24xFP32(zmm10, ymm11, 5) \
	STORE_1x24xFP32(zmm12, ymm13, 6) \
	STORE_1x24xFP32(zmm14, ymm15, 7) \
	STORE_1x24xFP32(zmm16, ymm17, 8) \
	STORE_1x24xFP32(zmm18, ymm19, 9) \
	STORE_1x24xFP32(zmm20, ymm21, 10) \
	STORE_1x24xFP32(zmm22, ymm23, 11) \
	STORE_1x24xFP32(zmm24, ymm25, 12) \
	STORE_1x24xFP32(zmm26, ymm27, 13) \
	STORE_1x24xFP32(zmm28, ymm29, 14) \
	STORE_1x24xFP32(zmm30, ymm31, 15)
#define STORE_16x16xFP32(ptr, stride, offset) \
	vmovups(zmm0, mem(ptr, (0*stride+offset)*4)) \
	vmovups(zmm1, mem(ptr, (1*stride+offset)*4)) \
	vmovups(zmm2, mem(ptr, (2*stride+offset)*4)) \
	vmovups(zmm3, mem(ptr, (3*stride+offset)*4)) \
	vmovups(zmm4, mem(ptr, (4*stride+offset)*4)) \
	vmovups(zmm5, mem(ptr, (5*stride+offset)*4)) \
	vmovups(zmm6, mem(ptr, (6*stride+offset)*4)) \
	vmovups(zmm7, mem(ptr, (7*stride+offset)*4)) \
	vmovups(zmm24, mem(ptr, (8*stride+offset)*4)) \
	vmovups(zmm25, mem(ptr, (9*stride+offset)*4)) \
	vmovups(zmm26, mem(ptr, (10*stride+offset)*4)) \
	vmovups(zmm27, mem(ptr, (11*stride+offset)*4)) \
	vmovups(zmm28, mem(ptr, (12*stride+offset)*4)) \
	vmovups(zmm29, mem(ptr, (13*stride+offset)*4)) \
	vmovups(zmm30, mem(ptr, (14*stride+offset)*4)) \
	vmovups(zmm31, mem(ptr, (15*stride+offset)*4))
#define STORE_16x8xFP32(ptr, stride, offset) \
	vmovups(ymm0, mem(ptr, (0*stride+offset)*4)) \
	vmovups(ymm1, mem(ptr, (1*stride+offset)*4)) \
	vmovups(ymm2, mem(ptr, (2*stride+offset)*4)) \
	vmovups(ymm3, mem(ptr, (3*stride+offset)*4)) \
	vmovups(ymm4, mem(ptr, (4*stride+offset)*4)) \
	vmovups(ymm5, mem(ptr, (5*stride+offset)*4)) \
	vmovups(ymm6, mem(ptr, (6*stride+offset)*4)) \
	vmovups(ymm7, mem(ptr, (7*stride+offset)*4)) \
	vmovups(ymm24, mem(ptr, (8*stride+offset)*4)) \
	vmovups(ymm25, mem(ptr, (9*stride+offset)*4)) \
	vmovups(ymm26, mem(ptr, (10*stride+offset)*4)) \
	vmovups(ymm27, mem(ptr, (11*stride+offset)*4)) \
	vmovups(ymm28, mem(ptr, (12*stride+offset)*4)) \
	vmovups(ymm29, mem(ptr, (13*stride+offset)*4)) \
	vmovups(ymm30, mem(ptr, (14*stride+offset)*4)) \
	vmovups(ymm31, mem(ptr, (15*stride+offset)*4))

// MHA macros
#define ADD_BIAS_8x16xFP32(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7) \
	vmovaps(mem(rbx), zmm0) \
	vmovaps(mem(rbx, r14, 1), zmm1) \
	vmovaps(mem(rbx, r14, 2), zmm2) \
	vmovaps(mem(rbx, r13, 1), zmm3) \
	add(r15, rbx) \
	vmovaps(mem(rbx), zmm4) \
	vmovaps(mem(rbx, r14, 1), zmm5) \
	vmovaps(mem(rbx, r14, 2), zmm6) \
	vmovaps(mem(rbx, r13, 1), zmm7) \
	add(r15, rbx) \
	vaddps(zmm0, reg0, reg0) \
	vaddps(zmm1, reg1, reg1) \
	vaddps(zmm2, reg2, reg2) \
	vaddps(zmm3, reg3, reg3) \
	vaddps(zmm4, reg4, reg4) \
	vaddps(zmm5, reg5, reg5) \
	vaddps(zmm6, reg6, reg6) \
	vaddps(zmm7, reg7, reg7)
#define SETUP_EXP_CONSTANTS() \
	movq(imm(0x4ab8aa3b4ab8aa3b), r14) \
	movq(imm(0x3f7de0683f7de068), r15) \
	vmovq(r14, xmm0) \
	vmovq(r15, xmm1) \
	vbroadcastss(xmm0, zmm0) \
	vbroadcastss(xmm1, zmm1)
#define EXP_6x16xFP32(reg0, reg1, reg2, reg3, reg4, reg5) \
	vmulps(zmm0, reg0, reg0) \
	vmulps(zmm0, reg1, reg1) \
	vmulps(zmm0, reg2, reg2) \
	vmulps(zmm0, reg3, reg3) \
	vmulps(zmm0, reg4, reg4) \
	vmulps(zmm0, reg5, reg5) \
	vcvtps2dq(reg0, reg0) \
	vcvtps2dq(reg1, reg1) \
	vcvtps2dq(reg2, reg2) \
	vcvtps2dq(reg3, reg3) \
	vcvtps2dq(reg4, reg4) \
	vcvtps2dq(reg5, reg5) \
	vpsubd(reg0, zmm1, zmm2) \
	vpsubd(reg1, zmm1, zmm3) \
	vpsubd(reg2, zmm1, zmm4) \
	vpsubd(reg3, zmm1, zmm5) \
	vpsubd(reg4, zmm1, zmm6) \
	vpsubd(reg5, zmm1, zmm7) \
	vrcp14ps(zmm2, zmm2) \
	vrcp14ps(zmm3, zmm3) \
	vrcp14ps(zmm4, zmm4) \
	vrcp14ps(zmm5, zmm5) \
	vrcp14ps(zmm6, zmm6) \
	vrcp14ps(zmm7, zmm7) \
	vpaddd(reg0, zmm1, reg0) \
	vpaddd(reg1, zmm1, reg1) \
	vpaddd(reg2, zmm1, reg2) \
	vpaddd(reg3, zmm1, reg3) \
	vpaddd(reg4, zmm1, reg4) \
	vpaddd(reg5, zmm1, reg5) \
	vmulps(zmm2, reg0, reg0) \
	vmulps(zmm3, reg1, reg1) \
	vmulps(zmm4, reg2, reg2) \
	vmulps(zmm5, reg3, reg3) \
	vmulps(zmm6, reg4, reg4) \
	vmulps(zmm7, reg5, reg5)

// transposes registers zmm8-zmm23 into ymm0-ymm7 and ymm24-ymm31
#define TRANSPOSE_8x16xFP32() \
	vunpcklps(zmm25, zmm24, zmm0) \
	vunpckhps(zmm25, zmm24, zmm1) \
	vunpcklps(zmm27, zmm26, zmm2) \
	vunpckhps(zmm27, zmm26, zmm3) \
	vunpcklps(zmm29, zmm28, zmm4) \
	vunpckhps(zmm29, zmm28, zmm5) \
	vunpcklps(zmm31, zmm30, zmm6) \
	vunpckhps(zmm31, zmm30, zmm7) \
	vunpcklpd(zmm2, zmm0, zmm24) \
	vunpckhpd(zmm2, zmm0, zmm25) \
	vunpcklpd(zmm3, zmm1, zmm26) \
	vunpckhpd(zmm3, zmm1, zmm27) \
	vunpcklpd(zmm6, zmm4, zmm28) \
	vunpckhpd(zmm6, zmm4, zmm29) \
	vunpcklpd(zmm7, zmm5, zmm30) \
	vunpckhpd(zmm7, zmm5, zmm31) \
	vshuff32x4(imm(0x88), zmm28, zmm24, zmm0) \
	vshuff32x4(imm(0x88), zmm29, zmm25, zmm1) \
	vshuff32x4(imm(0x88), zmm30, zmm26, zmm2) \
	vshuff32x4(imm(0x88), zmm31, zmm27, zmm3) \
	vshuff32x4(imm(0xDD), zmm28, zmm24, zmm4) \
	vshuff32x4(imm(0xDD), zmm29, zmm25, zmm5) \
	vshuff32x4(imm(0xDD), zmm30, zmm26, zmm6) \
	vshuff32x4(imm(0xDD), zmm31, zmm27, zmm7) \
	vshuff32x4(imm(0xD8), zmm0, zmm0, zmm0) \
	vshuff32x4(imm(0xD8), zmm1, zmm1, zmm1) \
	vshuff32x4(imm(0xD8), zmm2, zmm2, zmm2) \
	vshuff32x4(imm(0xD8), zmm3, zmm3, zmm3) \
	vshuff32x4(imm(0xD8), zmm4, zmm4, zmm4) \
	vshuff32x4(imm(0xD8), zmm5, zmm5, zmm5) \
	vshuff32x4(imm(0xD8), zmm6, zmm6, zmm6) \
	vshuff32x4(imm(0xD8), zmm7, zmm7, zmm7) \
	vextractf32x8(imm(0x1), zmm0, ymm24) \
	vextractf32x8(imm(0x1), zmm1, ymm25) \
	vextractf32x8(imm(0x1), zmm2, ymm26) \
	vextractf32x8(imm(0x1), zmm3, ymm27) \
	vextractf32x8(imm(0x1), zmm4, ymm28) \
	vextractf32x8(imm(0x1), zmm5, ymm29) \
	vextractf32x8(imm(0x1), zmm6, ymm30) \
	vextractf32x8(imm(0x1), zmm7, ymm31)
// transposes registers zmm8-zmm23 into zmm0-zmm7 and zmm24-zmm31
#define TRANSPOSE_16x16xFP32() \
	vunpcklps(zmm9, zmm8, zmm0) \
	vunpckhps(zmm9, zmm8, zmm1) \
	vunpcklps(zmm11, zmm10, zmm2) \
	vunpckhps(zmm11, zmm10, zmm3) \
	vunpcklps(zmm13, zmm12, zmm4) \
	vunpckhps(zmm13, zmm12, zmm5) \
	vunpcklps(zmm15, zmm14, zmm6) \
	vunpckhps(zmm15, zmm14, zmm7) \
	vunpcklps(zmm17, zmm16, zmm24) \
	vunpckhps(zmm17, zmm16, zmm25) \
	vunpcklps(zmm19, zmm18, zmm26) \
	vunpckhps(zmm19, zmm18, zmm27) \
	vunpcklps(zmm21, zmm20, zmm28) \
	vunpckhps(zmm21, zmm20, zmm29) \
	vunpcklps(zmm23, zmm22, zmm30) \
	vunpckhps(zmm23, zmm22, zmm31) \
	vunpcklpd(zmm2, zmm0, zmm8) \
	vunpckhpd(zmm2, zmm0, zmm9) \
	vunpcklpd(zmm3, zmm1, zmm10) \
	vunpckhpd(zmm3, zmm1, zmm11) \
	vunpcklpd(zmm6, zmm4, zmm12) \
	vunpckhpd(zmm6, zmm4, zmm13) \
	vunpcklpd(zmm7, zmm5, zmm14) \
	vunpckhpd(zmm7, zmm5, zmm15) \
	vunpcklpd(zmm26, zmm24, zmm16) \
	vunpckhpd(zmm26, zmm24, zmm17) \
	vunpcklpd(zmm27, zmm25, zmm18) \
	vunpckhpd(zmm27, zmm25, zmm19) \
	vunpcklpd(zmm30, zmm28, zmm20) \
	vunpckhpd(zmm30, zmm28, zmm21) \
	vunpcklpd(zmm31, zmm29, zmm22) \
	vunpckhpd(zmm31, zmm29, zmm23) \
	vshuff32x4(imm(0x88), zmm12, zmm8, zmm0) \
	vshuff32x4(imm(0x88), zmm13, zmm9, zmm1) \
	vshuff32x4(imm(0x88), zmm14, zmm10, zmm2) \
	vshuff32x4(imm(0x88), zmm15, zmm11, zmm3) \
	vshuff32x4(imm(0xDD), zmm12, zmm8, zmm4) \
	vshuff32x4(imm(0xDD), zmm13, zmm9, zmm5) \
	vshuff32x4(imm(0xDD), zmm14, zmm10, zmm6) \
	vshuff32x4(imm(0xDD), zmm15, zmm11, zmm7) \
	vshuff32x4(imm(0x88), zmm20, zmm16, zmm24) \
	vshuff32x4(imm(0x88), zmm21, zmm17, zmm25) \
	vshuff32x4(imm(0x88), zmm22, zmm18, zmm26) \
	vshuff32x4(imm(0x88), zmm23, zmm19, zmm27) \
	vshuff32x4(imm(0xDD), zmm20, zmm16, zmm28) \
	vshuff32x4(imm(0xDD), zmm21, zmm17, zmm29) \
	vshuff32x4(imm(0xDD), zmm22, zmm18, zmm30) \
	vshuff32x4(imm(0xDD), zmm23, zmm19, zmm31) \
	vshuff32x4(imm(0x88), zmm24, zmm0, zmm8) \
	vshuff32x4(imm(0x88), zmm25, zmm1, zmm9) \
	vshuff32x4(imm(0x88), zmm26, zmm2, zmm10) \
	vshuff32x4(imm(0x88), zmm27, zmm3, zmm11) \
	vshuff32x4(imm(0x88), zmm28, zmm4, zmm12) \
	vshuff32x4(imm(0x88), zmm29, zmm5, zmm13) \
	vshuff32x4(imm(0x88), zmm30, zmm6, zmm14) \
	vshuff32x4(imm(0x88), zmm31, zmm7, zmm15) \
	vshuff32x4(imm(0xDD), zmm24, zmm0, zmm24) \
	vshuff32x4(imm(0xDD), zmm25, zmm1, zmm25) \
	vshuff32x4(imm(0xDD), zmm26, zmm2, zmm26) \
	vshuff32x4(imm(0xDD), zmm27, zmm3, zmm27) \
	vshuff32x4(imm(0xDD), zmm28, zmm4, zmm28) \
	vshuff32x4(imm(0xDD), zmm29, zmm5, zmm29) \
	vshuff32x4(imm(0xDD), zmm30, zmm6, zmm30) \
	vshuff32x4(imm(0xDD), zmm31, zmm7, zmm31) \
	vmovaps(zmm8, zmm0) \
	vmovaps(zmm9, zmm1) \
	vmovaps(zmm10, zmm2) \
	vmovaps(zmm11, zmm3) \
	vmovaps(zmm12, zmm4) \
	vmovaps(zmm13, zmm5) \
	vmovaps(zmm14, zmm6) \
	vmovaps(zmm15, zmm7)
#define REDUCE_SUM() \
	vaddps(zmm24, zmm0, zmm0) \
	vaddps(zmm25, zmm1, zmm1) \
	vaddps(zmm26, zmm2, zmm2) \
	vaddps(zmm27, zmm3, zmm3) \
	vaddps(zmm28, zmm4, zmm4) \
	vaddps(zmm29, zmm5, zmm5) \
	vaddps(zmm30, zmm6, zmm6) \
	vaddps(zmm31, zmm7, zmm7) \
	vaddps(zmm1, zmm0, zmm0) \
	vaddps(zmm3, zmm2, zmm2) \
	vaddps(zmm5, zmm4, zmm4) \
	vaddps(zmm7, zmm6, zmm6) \
	vaddps(zmm2, zmm0, zmm0) \
	vaddps(zmm6, zmm4, zmm4) \
	vaddps(zmm4, zmm0, zmm0)

namespace ml
{
//#define __AVX512F__
	using namespace ml::cpu;
#ifdef __AVX512F__
	void gemm_avx512f_24x16(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(C.is_fp32() || C.is_fp16());
		assert(D.is_fp32() || D.is_fp16());
		assert(A.rows() == B.rows());
		assert(A.stride() == 24);
		assert(B.stride() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha.is_packed());
		assert(alpha.is_fp32());
		assert(cpu::is_aligned(A.data(), 64));
		assert(cpu::is_aligned(B.data(), 64));
		assert(beta_ptr != nullptr);
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 64));
		}

		const void *A_ptr = A.data();
		const void *B_ptr = B.data();
		const void *C_ptr = C.data();
		void *D_ptr = D.data();
		const void *bias_ptr = bias.is_packed() ? bias.data() : nullptr;
		const void *alpha_ptr = alpha.data();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride_in_bytes();
		const uint64_t D_stride = D.stride_in_bytes();
		const uint64_t flag_relu = use_relu;
		const uint64_t cd_in_fp16 = C.is_fp16() | (D.is_fp16() << 1);
		const uint64_t scalar_alpha = alpha.rows() == 1;

		begin_asm()

		movq(var(A_ptr), rax)
		movq(var(B_ptr), rbx)
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14) // load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_24xFP32_16xFP32(0)
		SUB_KERNEL_24xFP32_16xFP32(1)
		SUB_KERNEL_24xFP32_16xFP32(2)
		SUB_KERNEL_24xFP32_16xFP32(3)

		add(imm(4*24*4), rax)// 4 iterations x 24 elements x 4 bytes
		add(imm(4*16*4), rbx)// 4 iterations x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_24xFP32_16xFP32(0)
		add(imm(1*24*4), rax)// 1 iteration x 24 elements x 4 bytes
		add(imm(1*16*4), rbx)// 1 iteration x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)
		// permute back to original layout
		PERMUTE_8x16xFP32(zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15)
		PERMUTE_8x16xFP32(zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23)
		PERMUTE_8x16xFP32(zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(scalar_alpha), r14)
		test(r14, r14)
		je(COLUMN_ALPHA)
		SCALE_ACCUMULATORS_1x1()
		jmp(AFTER_ALPHA_SCALING)

		label(COLUMN_ALPHA)
		SCALE_ACCUMULATORS_24x1()
		label(AFTER_ALPHA_SCALING)

		movq(var(bias_ptr), rax)// load address of bias pointer
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax), zmm2)// load bias
		ADD_BIAS_24x16xFP32(zmm2)
		label(AFTER_BIAS)

		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rbx), zmm0)
		vxorps(zmm1, zmm1, zmm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(AFTER_LOAD_C)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)

		movq(var(cd_in_fp16), r11)// load fp16 flags
		and_(imm(0x1), r11)// if set
		test(r11, r11)
		je(C_IN_FP32)
		LOAD_ADD_3x16xFP16(zmm0, zmm8, zmm9, zmm10)
		LOAD_ADD_3x16xFP16(zmm0, zmm11, zmm12, zmm13)
		LOAD_ADD_3x16xFP16(zmm0, zmm14, zmm15, zmm16)
		LOAD_ADD_3x16xFP16(zmm0, zmm17, zmm18, zmm19)
		LOAD_ADD_3x16xFP16(zmm0, zmm20, zmm21, zmm22)
		LOAD_ADD_3x16xFP16(zmm0, zmm23, zmm24, zmm25)
		LOAD_ADD_3x16xFP16(zmm0, zmm26, zmm27, zmm28)
		LOAD_ADD_3x16xFP16(zmm0, zmm29, zmm30, zmm31)
		jmp(AFTER_LOAD_C)

		label(C_IN_FP32)
		LOAD_ADD_3x16xFP32(zmm0, zmm8, zmm9, zmm10)
		LOAD_ADD_3x16xFP32(zmm0, zmm11, zmm12, zmm13)
		LOAD_ADD_3x16xFP32(zmm0, zmm14, zmm15, zmm16)
		LOAD_ADD_3x16xFP32(zmm0, zmm17, zmm18, zmm19)
		LOAD_ADD_3x16xFP32(zmm0, zmm20, zmm21, zmm22)
		LOAD_ADD_3x16xFP32(zmm0, zmm23, zmm24, zmm25)
		LOAD_ADD_3x16xFP32(zmm0, zmm26, zmm27, zmm28)
		LOAD_ADD_3x16xFP32(zmm0, zmm29, zmm30, zmm31)
		label(AFTER_LOAD_C)

		// load flag if to use relu
		movq(var(flag_relu), r14)
		test(r14, r14)
		je(AFTER_RELU)
		RELU_24x16xFP32()
		label(AFTER_RELU)

		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*D_stride)

		movq(var(cd_in_fp16), r11)// load fp16 flags
		and_(imm(0x2), r11)// if set
		test(r11, r11)
		je(D_IN_FP32)
		CONVERT_ACCUMULATORS_TO_FP16()
		STORE_3x16xFP16(ymm8, ymm9, ymm10)
		STORE_3x16xFP16(ymm11, ymm12, ymm13)
		STORE_3x16xFP16(ymm14, ymm15, ymm16)
		STORE_3x16xFP16(ymm17, ymm18, ymm19)
		STORE_3x16xFP16(ymm20, ymm21, ymm22)
		STORE_3x16xFP16(ymm23, ymm24, ymm25)
		STORE_3x16xFP16(ymm26, ymm27, ymm28)
		STORE_3x16xFP16(ymm29, ymm30, ymm31)
		jmp(END)

		label(D_IN_FP32)
		STORE_3x16xFP32(zmm8, zmm9, zmm10)
		STORE_3x16xFP32(zmm11, zmm12, zmm13)
		STORE_3x16xFP32(zmm14, zmm15, zmm16)
		STORE_3x16xFP32(zmm17, zmm18, zmm19)
		STORE_3x16xFP32(zmm20, zmm21, zmm22)
		STORE_3x16xFP32(zmm23, zmm24, zmm25)
		STORE_3x16xFP32(zmm26, zmm27, zmm28)
		STORE_3x16xFP32(zmm29, zmm30, zmm31)

		label(END)
		vzeroupper()

		end_asm(
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
				[beta_ptr] "m"(beta_ptr),
				[flag_relu] "m"(flag_relu),
				[bias_ptr] "m"(bias_ptr),
				[cd_in_fp16] "m"(cd_in_fp16),
				[scalar_alpha] "m"(scalar_alpha)
				:// clobbers
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
				"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
				"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
				"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%rcx", "%rdx", "%r11", "%r14", "%r15")
	}

	void pack_avx512f_24xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		if (dst.is_partial())
		{
			pack_def_MxK(dst, src, src_pos, src_op);
			return;
		}

		assert(src.is_fp32() || src.is_fp16());
		assert(dst.is_fp32());
		assert(dst.stride() == 24);
		assert(ml::cpu::is_aligned(dst.data(), 64));

		const uint64_t k_iter = dst.rows() / 16;
		const uint64_t k_left = dst.rows() % 16;
		const uint64_t src_stride = src.stride_in_bytes();
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			if (src.is_fp32())
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 16-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED16)
				LOAD_16x24xFP32(rax)
				STORE_16x24xFP32()
				add(imm(16*24*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED16)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				LOAD_1x24xFP32(rax, zmm0, ymm1)
				STORE_1x24xFP32(zmm0, ymm1, 0)
				add(imm(1*24*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
						"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
						"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
						"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 16-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED16)
				LOAD_16x24xFP16(rax)
				STORE_16x24xFP32()
				add(imm(16*24*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED16)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				LOAD_1x24xFP16(rax, zmm0, ymm1)
				CONVERT_1x24xFP16_TO_FP32(0, 1)
				STORE_1x24xFP32(zmm0, ymm1, 0)
				add(imm(1*24*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
						"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
						"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
						"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%r12", "%r13", "%r14", "%r15")
			}
		}
		else
		{
			if (src.is_fp32())
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 16-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED16)
				// first 16x16 tile
				movq(rax, rcx)

				// first 16x16 tile
				LOAD_16x16xFP32(rcx)
				TRANSPOSE_16x16xFP32()
				STORE_16x16xFP32(rbx, 24, 0)

				// second 8x16 tile
				LOAD_8x16xFP32(rcx)
				TRANSPOSE_8x16xFP32()
				STORE_16x8xFP32(rbx, 24, 16)

				add(imm(4*16), rax)// add stride to src pointer
				add(imm(4*24*16), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED16)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)

				LOAD_4x1xFP32(rcx, xmm0, xmm1, xmm2, xmm3)
				LOAD_4x1xFP32(rcx, xmm4, xmm5, xmm6, xmm7)
				LOAD_4x1xFP32(rcx, xmm8, xmm9, xmm10, xmm11)
				LOAD_4x1xFP32(rcx, xmm12, xmm13, xmm14, xmm15)
				LOAD_4x1xFP32(rcx, xmm16, xmm17, xmm18, xmm19)
				LOAD_4x1xFP32(rcx, xmm20, xmm21, xmm22, xmm23)

				STORE_4x1xFP32(0, xmm0, xmm1, xmm2, xmm3)
				STORE_4x1xFP32(4, xmm4, xmm5, xmm6, xmm7)
				STORE_4x1xFP32(8, xmm8, xmm9, xmm10, xmm11)
				STORE_4x1xFP32(12, xmm12, xmm13, xmm14, xmm15)
				STORE_4x1xFP32(16, xmm16, xmm17, xmm18, xmm19)
				STORE_4x1xFP32(20, xmm20, xmm21, xmm22, xmm23)

				add(imm(4*1), rax)// add stride to src pointer
				add(imm(24*1*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()
				end_asm(:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
						"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
						"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
						"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%rcx",
						"%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 16-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED16)
				// first 16x16 tile
				movq(rax, rcx)

				// first 16x16 tile
				LOAD_16x16xFP16(rcx)
				TRANSPOSE_16x16xFP32()
				STORE_16x16xFP32(rbx, 24, 0)

				// second 8x16 tile
				LOAD_8x16xFP16(rcx)
				TRANSPOSE_8x16xFP32()
				STORE_16x8xFP32(rbx, 24, 16)

				add(imm(2*16), rax)// add stride to src pointer
				add(imm(4*24*16), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED16)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)

				LOAD_4x1xFP16(rcx, 0, 1, 2, 3)
				LOAD_4x1xFP16(rcx, 4, 5, 6, 7)
				LOAD_4x1xFP16(rcx, 8, 9, 10, 11)
				LOAD_4x1xFP16(rcx, 12, 13, 14, 15)
				LOAD_4x1xFP16(rcx, 16, 17, 18, 19)
				LOAD_4x1xFP16(rcx, 20, 21, 22, 23)

				STORE_4x1xFP32(0, xmm0, xmm1, xmm2, xmm3)
				STORE_4x1xFP32(4, xmm4, xmm5, xmm6, xmm7)
				STORE_4x1xFP32(8, xmm8, xmm9, xmm10, xmm11)
				STORE_4x1xFP32(12, xmm12, xmm13, xmm14, xmm15)
				STORE_4x1xFP32(16, xmm16, xmm17, xmm18, xmm19)
				STORE_4x1xFP32(20, xmm20, xmm21, xmm22, xmm23)

				add(imm(2*1), rax)// add stride to src pointer
				add(imm(24*1*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()
				end_asm(:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
						"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
						"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
						"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%rcx",
						"%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
			}
		}
	}
	void pack_avx512f_16xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		if (dst.is_partial())
		{
			pack_def_MxK(dst, src, src_pos, src_op);
			return;
		}

		assert(src.is_fp32() || src.is_fp16());
		assert(dst.is_fp32());
		assert(dst.stride() == 16);
		assert(ml::cpu::is_aligned(dst.data(), 64));

		const uint64_t k_iter = dst.rows() / 16;
		const uint64_t k_left = dst.rows() % 16;
		const uint64_t src_stride = src.stride_in_bytes();
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			if (src.is_fp32())
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 16-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED16)
				LOAD_4x16xFP32(rax, zmm0, zmm1, zmm2, zmm3)
				LOAD_4x16xFP32(rax, zmm4, zmm5, zmm6, zmm7)
				LOAD_4x16xFP32(rax, zmm24, zmm25, zmm26, zmm27)
				LOAD_4x16xFP32(rax, zmm28, zmm29, zmm30, zmm31)
				STORE_16x16xFP32(rbx, 16, 0)
				add(imm(16*16*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED16)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				LOAD_1x16xFP32(rax, zmm0)
				STORE_1x16xFP32(zmm0, 0)
				add(imm(1*16*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
						"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
						"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
						"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx",
						"%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 16-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED16)
				LOAD_4x16xFP16(rax, zmm0, zmm1, zmm2, zmm3)
				LOAD_4x16xFP16(rax, zmm4, zmm5, zmm6, zmm7)
				LOAD_4x16xFP16(rax, zmm24, zmm25, zmm26, zmm27)
				LOAD_4x16xFP16(rax, zmm28, zmm29, zmm30, zmm31)
				CONVERT_4x16xFP16_TO_FP32(0, 1, 2, 3)
				CONVERT_4x16xFP16_TO_FP32(4, 5, 6, 7)
				CONVERT_4x16xFP16_TO_FP32(24, 25, 26, 27)
				CONVERT_4x16xFP16_TO_FP32(28, 29, 30, 31)
				STORE_16x16xFP32(rbx, 16, 0)
				add(imm(16*16*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED16)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				LOAD_1x16xFP16(rax, ymm0)
				CONVERT_1x16xFP16_TO_FP32(0)
				STORE_1x16xFP32(zmm0, 0)
				add(imm(1*16*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
						"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
						"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
						"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx",
						"%r12", "%r13", "%r14", "%r15")
			}
		}
		else
		{
			if (src.is_fp32())
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 16-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED16)
				// first 16x16 tile
				movq(rax, rcx)// tmp src pointer is in r13

				LOAD_16x16xFP32(rcx)
				TRANSPOSE_16x16xFP32()
				STORE_16x16xFP32(rbx, 16, 0)

				add(imm(4*16), rax)// add stride to src pointer
				add(imm(4*16*16), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED16)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)

				LOAD_4x1xFP32(rcx, xmm0, xmm1, xmm2, xmm3)
				LOAD_4x1xFP32(rcx, xmm4, xmm5, xmm6, xmm7)
				LOAD_4x1xFP32(rcx, xmm8, xmm9, xmm10, xmm11)
				LOAD_4x1xFP32(rcx, xmm12, xmm13, xmm14, xmm15)

				STORE_4x1xFP32(0, xmm0, xmm1, xmm2, xmm3)
				STORE_4x1xFP32(4, xmm4, xmm5, xmm6, xmm7)
				STORE_4x1xFP32(8, xmm8, xmm9, xmm10, xmm11)
				STORE_4x1xFP32(12, xmm12, xmm13, xmm14, xmm15)

				add(imm(4*1), rax)// add stride to src pointer
				add(imm(16*1*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()
				end_asm(:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
						"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
						"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
						"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%rcx",
						"%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				begin_asm()
				movq(var(src_ptr), rax) // src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 16-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED16)
				// first 16x16 tile
				movq(rax, rcx)// tmp src pointer is in rcx

				LOAD_16x16xFP16(rcx)
				TRANSPOSE_16x16xFP32()
				STORE_16x16xFP32(rbx, 16, 0)

				add(imm(2*16), rax)// add stride to src pointer
				add(imm(4*16*16), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED16)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)

				LOAD_4x1xFP16(rcx, 0, 1, 2, 3)
				LOAD_4x1xFP16(rcx, 4, 5, 6, 7)
				LOAD_4x1xFP16(rcx, 8, 9, 10, 11)
				LOAD_4x1xFP16(rcx, 12, 13, 14, 15)

				STORE_4x1xFP32(0, xmm0, xmm1, xmm2, xmm3)
				STORE_4x1xFP32(4, xmm4, xmm5, xmm6, xmm7)
				STORE_4x1xFP32(8, xmm8, xmm9, xmm10, xmm11)
				STORE_4x1xFP32(12, xmm12, xmm13, xmm14, xmm15)

				add(imm(2*1), rax)// add stride to src pointer
				add(imm(16*1*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()
				end_asm(:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[k_iter] "m"(k_iter),
						[k_left] "m"(k_left),
						[src_stride] "m"(src_stride)
						:// clobbers
						"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
						"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
						"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
						"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%rcx",
						"%r12", "%r13", "%r14", "%r15")
			}
		}
	}

	void mha_qk_avx512f_24x16(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept
	{
		assert(temp.is_fp32());
		assert(Q.is_fp32());
		assert(K.is_fp32());
		assert(softmax_sum.is_fp32());
		assert(Q.rows() == K.rows());
		assert(Q.stride() == 24);
		assert(K.stride() == 16);
		assert(temp.columns() == Q.columns());
		assert(temp.rows() == K.columns());
		assert(temp.stride() == 24);

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(Q.data(), 64));
		assert(cpu::is_aligned(K.data(), 64));

		const float *Q_ptr = Q.data<float>();
		const float *K_ptr = K.data<float>();
		float *temp_ptr = temp.data<float>();
		const float *bias_ptr = bias.is_packed() ? bias.data<float>() : nullptr;
		if (bias.is_packed())
		{
			assert(bias.is_fp32());
			assert(cpu::is_aligned(bias.data(), 16));
		}
		float *softmax_ptr = softmax_sum.is_packed() ? softmax_sum.data<float>() : nullptr;
		if(softmax_sum.is_packed())
		{
			assert(softmax_sum.is_fp32());
			assert(softmax_sum.rows() >= temp.columns());
			assert(cpu::is_aligned(softmax_sum.data(), 32));
		}

		uint64_t k_iter = Q.rows() / 4;
		uint64_t k_left = Q.rows() % 4;
		const uint64_t bias_stride = bias.stride_in_bytes();
		assert(bias_stride % 64 == 0);

		begin_asm()

		movq(var(Q_ptr), rax)
		movq(var(K_ptr), rbx)
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14) // load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_24xFP32_16xFP32(0)
		SUB_KERNEL_24xFP32_16xFP32(1)
		SUB_KERNEL_24xFP32_16xFP32(2)
		SUB_KERNEL_24xFP32_16xFP32(3)

		add(imm(4*24*4), rax)// 4 iterations x 24 elements x 4 bytes
		add(imm(4*16*4), rbx)// 4 iterations x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_24xFP32_16xFP32(0)
		add(imm(1*24*4), rax)// 1 iteration x 24 elements x 4 bytes
		add(imm(1*16*4), rbx)// 1 iteration x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		vbroadcastss(mem(rax), zmm1)

		// permute back to original layout
		PERMUTE_AND_SCALE_6x16xFP32(zmm8, zmm9, zmm10, zmm11, zmm12, zmm13)
		PERMUTE_AND_SCALE_6x16xFP32(zmm14, zmm15, zmm16, zmm17, zmm18, zmm19)
		PERMUTE_AND_SCALE_6x16xFP32(zmm20, zmm21, zmm22, zmm23, zmm24, zmm25)
		PERMUTE_AND_SCALE_6x16xFP32(zmm26, zmm27, zmm28, zmm29, zmm30, zmm31)

		movq(var(bias_ptr), rbx)// load address of bias pointer
		test(rbx, rbx)
		je(AFTER_BIAS)
		movq(var(bias_stride), r14)// load address of bias stride into r14
		movq(r14, r13)
		sal(imm(1), r13)// r13 = stride * 2
		add(r14, r13)// r13 == stride * 3
		movq(r14, r15)
		sal(imm(2), r15)// r15 = stride * 4
		ADD_BIAS_8x16xFP32(zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15)
		ADD_BIAS_8x16xFP32(zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23)
		ADD_BIAS_8x16xFP32(zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31)
		label(AFTER_BIAS)

		SETUP_EXP_CONSTANTS()
		EXP_6x16xFP32(zmm8, zmm9, zmm10, zmm11, zmm12, zmm13)
		EXP_6x16xFP32(zmm14, zmm15, zmm16, zmm17, zmm18, zmm19)
		EXP_6x16xFP32(zmm20, zmm21, zmm22, zmm23, zmm24, zmm25)
		EXP_6x16xFP32(zmm26, zmm27, zmm28, zmm29, zmm30, zmm31)

		movq(var(temp_ptr), rbx)// temp pointer is in rbx
		movq(var(softmax_ptr), rcx)// softmax sum pointer is in rcx
		// store 8x16 tile
		TRANSPOSE_8x16xFP32()
		STORE_16x8xFP32(rbx, 24, 16)
		test(rcx, rcx)
		je(SKIP_REDUCTION_1)
		REDUCE_SUM()
		vmovaps(mem(rcx, 16*4), ymm1)
		vaddps(ymm1, ymm0, ymm0)
		vmovaps(ymm0, mem(rcx, 16*4))
		label(SKIP_REDUCTION_1)

		//store 16x16 tile
		TRANSPOSE_16x16xFP32()
		STORE_16x16xFP32(rbx, 24, 0)
		test(rcx, rcx)
		je(SKIP_REDUCTION_2)
		REDUCE_SUM()
		vmovaps(mem(rcx), zmm1)
		vaddps(zmm1, zmm0, zmm0)
		vmovaps(zmm0, mem(rcx))
		label(SKIP_REDUCTION_2)

		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[Q_ptr] "m"(Q_ptr),
				[K_ptr] "m"(K_ptr),
				[temp_ptr] "m"(temp_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[bias_stride] "m"(bias_stride),
				[alpha_ptr] "m"(alpha_ptr),
				[bias_ptr] "m"(bias_ptr),
				[softmax_ptr] "m"(softmax_ptr)
				:// clobbers
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
				"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15",
				"%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",
				"%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31",
				"%rax", "%rbx", "%rcx", "%rdx", "%r12", "%r13", "%r14", "%r15")
	}
#else
	void gemm_avx512f_24x16(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
	}
	void pack_avx512f_24xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
	}
	void pack_avx512f_16xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
	}
	void mha_qk_avx512f_24x16(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept
	{
	}
#endif
} /* namespace ml */

