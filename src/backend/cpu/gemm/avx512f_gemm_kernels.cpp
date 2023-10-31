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

#define LOAD_4x1xFP16(reg0, reg1, reg2, reg3)\
	movzw(mem(rax), rcx)\
	vmovq(rcx, xmm(reg0))\
	vcvtph2ps(xmm(reg0), xmm(reg0))\
	add(r12, rax)\
	movzw(mem(rax), rcx)\
	vmovq(rcx, xmm(reg1))\
	vcvtph2ps(xmm(reg1), xmm(reg1))\
	add(r12, rax)\
	movzw(mem(rax), rcx)\
	vmovq(rcx, xmm(reg2))\
	vcvtph2ps(xmm(reg2), xmm(reg2))\
	add(r12, rax)\
	movzw(mem(rax), rcx)\
	vmovq(rcx, xmm(reg3))\
	vcvtph2ps(xmm(reg3), xmm(reg3))\
	add(r12, rax)
#define LOAD_4x1xFP32(reg0, reg1, reg2, reg3)\
	vmovss(mem(rax), xmm(reg0))\
	add(r12, rax)\
	vmovss(mem(rax), xmm(reg1))\
	add(r12, rax)\
	vmovss(mem(rax), xmm(reg2))\
	add(r12, rax)\
	vmovss(mem(rax), xmm(reg3))\
	add(r12, rax)
#define STORE_4x1xFP32(reg0, reg1, reg2, reg3)\
	vmovss(xmm(reg0), mem(rbx, 4*reg0))\
	vmovss(xmm(reg1), mem(rbx, 4*reg1))\
	vmovss(xmm(reg2), mem(rbx, 4*reg2))\
	vmovss(xmm(reg3), mem(rbx, 4*reg3))

#define LOAD_1x16xFP32(reg)\
	vmovups(mem(rax), reg)\
	add(r12, rax)
#define STORE_1x16xFP32(reg, row) vmovups(reg, mem(rbx, row*16*4))
#define LOAD_1x24xFP32(reg0, reg1)\
	vmovups(mem(rax, 0*4), reg0)\
	vmovups(mem(rax, 16*4), reg1)\
	add(r12, rax)
#define STORE_1x24xFP32(reg0, reg1, row)\
	vmovups(reg0, mem(rbx, (row*24+0)*4))\
	vmovaps(reg1, mem(rbx, (row*24+16)*4))

#define LOAD_1x16xFP16(reg)\
	vmovups(mem(rax), reg)\
	add(r12, rax)
#define LOAD_1x24xFP16(reg0, reg1)\
	vmovups(mem(rax, 0*2), reg0)\
	vmovups(mem(rax, 16*2), reg1)\
	add(r12, rax)

#define CONVERT_1x24xFP16_TO_FP32(reg0, reg1)\
	vcvtph2ps(ymm(reg0), zmm(reg0))\
	vcvtph2ps(xmm(reg1), ymm(reg1))
#define CONVERT_1x16xFP16_TO_FP32(reg)\
	vcvtph2ps(ymm(reg), zmm(reg))
#define CONVERT_2x24xFP16_TO_FP32(reg0, reg1, reg2, reg3)\
	vcvtph2ps(ymm(reg0), zmm(reg0))\
	vcvtph2ps(xmm(reg1), ymm(reg1))\
	vcvtph2ps(ymm(reg2), zmm(reg2))\
	vcvtph2ps(xmm(reg3), ymm(reg3))
#define CONVERT_4x16xFP16_TO_FP32(reg0, reg1, reg2, reg3)\
	vcvtph2ps(ymm(reg0), zmm(reg0))\
	vcvtph2ps(ymm(reg1), zmm(reg1))\
	vcvtph2ps(ymm(reg2), zmm(reg2))\
	vcvtph2ps(ymm(reg3), zmm(reg3))

#define LOAD_16x24xFP16()\
	LOAD_1x24xFP16(ymm0, xmm1)\
	LOAD_1x24xFP16(ymm2, xmm3)\
	LOAD_1x24xFP16(ymm4, xmm5)\
	LOAD_1x24xFP16(ymm6, xmm7)\
	LOAD_1x24xFP16(ymm8, xmm9)\
	LOAD_1x24xFP16(ymm10, xmm11)\
	LOAD_1x24xFP16(ymm12, xmm13)\
	LOAD_1x24xFP16(ymm14, xmm15)\
	LOAD_1x24xFP16(ymm16, xmm17)\
	LOAD_1x24xFP16(ymm18, xmm19)\
	LOAD_1x24xFP16(ymm20, xmm21)\
	LOAD_1x24xFP16(ymm22, xmm23)\
	LOAD_1x24xFP16(ymm24, xmm25)\
	LOAD_1x24xFP16(ymm26, xmm27)\
	LOAD_1x24xFP16(ymm28, xmm29)\
	LOAD_1x24xFP16(ymm30, xmm31)\
	CONVERT_2x24xFP16_TO_FP32(0, 1, 2, 3)\
	CONVERT_2x24xFP16_TO_FP32(4, 5, 6, 7)\
	CONVERT_2x24xFP16_TO_FP32(8, 9, 10, 11)\
	CONVERT_2x24xFP16_TO_FP32(12, 13, 14, 15)\
	CONVERT_2x24xFP16_TO_FP32(16, 17, 18, 19)\
	CONVERT_2x24xFP16_TO_FP32(20, 21, 22, 23)\
	CONVERT_2x24xFP16_TO_FP32(24, 25, 26, 27)\
	CONVERT_2x24xFP16_TO_FP32(28, 29, 30, 31)
#define LOAD_16x16xFP16()\
	LOAD_1x16xFP16(ymm0)\
	LOAD_1x16xFP16(ymm1)\
	LOAD_1x16xFP16(ymm2)\
	LOAD_1x16xFP16(ymm3)\
	LOAD_1x16xFP16(ymm4)\
	LOAD_1x16xFP16(ymm5)\
	LOAD_1x16xFP16(ymm6)\
	LOAD_1x16xFP16(ymm7)\
	LOAD_1x16xFP16(ymm8)\
	LOAD_1x16xFP16(ymm9)\
	LOAD_1x16xFP16(ymm10)\
	LOAD_1x16xFP16(ymm11)\
	LOAD_1x16xFP16(ymm12)\
	LOAD_1x16xFP16(ymm13)\
	LOAD_1x16xFP16(ymm14)\
	LOAD_1x16xFP16(ymm15)\
	CONVERT_4x16xFP16_TO_FP32(0, 1, 2, 3)\
	CONVERT_4x16xFP16_TO_FP32(4, 5, 6, 7)\
	CONVERT_4x16xFP16_TO_FP32(8, 9, 10, 11)\
	CONVERT_4x16xFP16_TO_FP32(12, 13, 14, 15)
#define LOAD_8x16xFP16()\
	LOAD_1x16xFP16(ymm0)\
	LOAD_1x16xFP16(ymm1)\
	LOAD_1x16xFP16(ymm2)\
	LOAD_1x16xFP16(ymm3)\
	LOAD_1x16xFP16(ymm4)\
	LOAD_1x16xFP16(ymm5)\
	LOAD_1x16xFP16(ymm6)\
	LOAD_1x16xFP16(ymm7)\
	CONVERT_4x16xFP16_TO_FP32(0, 1, 2, 3)\
	CONVERT_4x16xFP16_TO_FP32(4, 5, 6, 7)

#define LOAD_16x24xFP32()\
	LOAD_1x24xFP32(zmm0, ymm1)\
	LOAD_1x24xFP32(zmm2, ymm3)\
	LOAD_1x24xFP32(zmm4, ymm5)\
	LOAD_1x24xFP32(zmm6, ymm7)\
	LOAD_1x24xFP32(zmm8, ymm9)\
	LOAD_1x24xFP32(zmm10, ymm11)\
	LOAD_1x24xFP32(zmm12, ymm13)\
	LOAD_1x24xFP32(zmm14, ymm15)\
	LOAD_1x24xFP32(zmm16, ymm17)\
	LOAD_1x24xFP32(zmm18, ymm19)\
	LOAD_1x24xFP32(zmm20, ymm21)\
	LOAD_1x24xFP32(zmm22, ymm23)\
	LOAD_1x24xFP32(zmm24, ymm25)\
	LOAD_1x24xFP32(zmm26, ymm27)\
	LOAD_1x24xFP32(zmm28, ymm29)\
	LOAD_1x24xFP32(zmm30, ymm31)
#define STORE_16x24xFP32()\
	STORE_1x24xFP32(zmm0, ymm1, 0)\
	STORE_1x24xFP32(zmm2, ymm3, 1)\
	STORE_1x24xFP32(zmm4, ymm5, 2)\
	STORE_1x24xFP32(zmm6, ymm7, 3)\
	STORE_1x24xFP32(zmm8, ymm9, 4)\
	STORE_1x24xFP32(zmm10, ymm11, 5)\
	STORE_1x24xFP32(zmm12, ymm13, 6)\
	STORE_1x24xFP32(zmm14, ymm15, 7)\
	STORE_1x24xFP32(zmm16, ymm17, 8)\
	STORE_1x24xFP32(zmm18, ymm19, 9)\
	STORE_1x24xFP32(zmm20, ymm21, 10)\
	STORE_1x24xFP32(zmm22, ymm23, 11)\
	STORE_1x24xFP32(zmm24, ymm25, 12)\
	STORE_1x24xFP32(zmm26, ymm27, 13)\
	STORE_1x24xFP32(zmm28, ymm29, 14)\
	STORE_1x24xFP32(zmm30, ymm31, 15)

#define LOAD_16x16xFP32()\
	LOAD_1x16xFP32(zmm0)\
	LOAD_1x16xFP32(zmm1)\
	LOAD_1x16xFP32(zmm2)\
	LOAD_1x16xFP32(zmm3)\
	LOAD_1x16xFP32(zmm4)\
	LOAD_1x16xFP32(zmm5)\
	LOAD_1x16xFP32(zmm6)\
	LOAD_1x16xFP32(zmm7)\
	LOAD_1x16xFP32(zmm8)\
	LOAD_1x16xFP32(zmm9)\
	LOAD_1x16xFP32(zmm10)\
	LOAD_1x16xFP32(zmm11)\
	LOAD_1x16xFP32(zmm12)\
	LOAD_1x16xFP32(zmm13)\
	LOAD_1x16xFP32(zmm14)\
	LOAD_1x16xFP32(zmm15)
#define STORE_16x16xFP32(stride)\
	vmovups(zmm0, mem(rbx, 0*stride*4))\
	vmovups(zmm1, mem(rbx, 1*stride*4))\
	vmovups(zmm2, mem(rbx, 2*stride*4))\
	vmovups(zmm3, mem(rbx, 3*stride*4))\
	vmovups(zmm4, mem(rbx, 4*stride*4))\
	vmovups(zmm5, mem(rbx, 5*stride*4))\
	vmovups(zmm6, mem(rbx, 6*stride*4))\
	vmovups(zmm7, mem(rbx, 7*stride*4))\
	vmovups(zmm8, mem(rbx, 8*stride*4))\
	vmovups(zmm9, mem(rbx, 9*stride*4))\
	vmovups(zmm10, mem(rbx, 10*stride*4))\
	vmovups(zmm11, mem(rbx, 11*stride*4))\
	vmovups(zmm12, mem(rbx, 12*stride*4))\
	vmovups(zmm13, mem(rbx, 13*stride*4))\
	vmovups(zmm14, mem(rbx, 14*stride*4))\
	vmovups(zmm15, mem(rbx, 15*stride*4))

#define LOAD_8x16xFP32()\
	LOAD_1x16xFP32(zmm0)\
	LOAD_1x16xFP32(zmm1)\
	LOAD_1x16xFP32(zmm2)\
	LOAD_1x16xFP32(zmm3)\
	LOAD_1x16xFP32(zmm4)\
	LOAD_1x16xFP32(zmm5)\
	LOAD_1x16xFP32(zmm6)\
	LOAD_1x16xFP32(zmm7)
#define STORE_16x8xFP32()\
	vextractf32x8(imm(0x1), zmm0, ymm8)\
	vextractf32x8(imm(0x1), zmm1, ymm9)\
	vextractf32x8(imm(0x1), zmm2, ymm10)\
	vextractf32x8(imm(0x1), zmm3, ymm11)\
	vextractf32x8(imm(0x1), zmm4, ymm12)\
	vextractf32x8(imm(0x1), zmm5, ymm13)\
	vextractf32x8(imm(0x1), zmm6, ymm14)\
	vextractf32x8(imm(0x1), zmm7, ymm15)\
	vmovaps(ymm0, mem(rbx, (0*24+16)*4))\
	vmovaps(ymm1, mem(rbx, (1*24+16)*4))\
	vmovaps(ymm2, mem(rbx, (2*24+16)*4))\
	vmovaps(ymm3, mem(rbx, (3*24+16)*4))\
	vmovaps(ymm4, mem(rbx, (4*24+16)*4))\
	vmovaps(ymm5, mem(rbx, (5*24+16)*4))\
	vmovaps(ymm6, mem(rbx, (6*24+16)*4))\
	vmovaps(ymm7, mem(rbx, (7*24+16)*4))\
	vmovaps(ymm8, mem(rbx, (8*24+16)*4))\
	vmovaps(ymm9, mem(rbx, (9*24+16)*4))\
	vmovaps(ymm10, mem(rbx, (10*24+16)*4))\
	vmovaps(ymm11, mem(rbx, (11*24+16)*4))\
	vmovaps(ymm12, mem(rbx, (12*24+16)*4))\
	vmovaps(ymm13, mem(rbx, (13*24+16)*4))\
	vmovaps(ymm14, mem(rbx, (14*24+16)*4))\
	vmovaps(ymm15, mem(rbx, (15*24+16)*4))

#define TRANSPOSE_SHUFFLE_1()\
	vunpcklps(zmm1, zmm0, zmm16)\
	vunpckhps(zmm1, zmm0, zmm17)\
	vunpcklps(zmm3, zmm2, zmm18)\
	vunpckhps(zmm3, zmm2, zmm19)\
	vunpcklps(zmm5, zmm4, zmm20)\
	vunpckhps(zmm5, zmm4, zmm21)\
	vunpcklps(zmm7, zmm6, zmm22)\
	vunpckhps(zmm7, zmm6, zmm23)\
	vunpcklps(zmm9, zmm8, zmm24)\
	vunpckhps(zmm9, zmm8, zmm25)\
	vunpcklps(zmm11, zmm10, zmm26)\
	vunpckhps(zmm11, zmm10, zmm27)\
	vunpcklps(zmm13, zmm12, zmm28)\
	vunpckhps(zmm13, zmm12, zmm29)\
	vunpcklps(zmm15, zmm14, zmm30)\
	vunpckhps(zmm15, zmm14, zmm31)
#define TRANSPOSE_SHUFFLE_2()\
	vunpcklpd(zmm18, zmm16, zmm0)\
	vunpckhpd(zmm18, zmm16, zmm1)\
	vunpcklpd(zmm19, zmm17, zmm2)\
	vunpckhpd(zmm19, zmm17, zmm3)\
	vunpcklpd(zmm22, zmm20, zmm4)\
	vunpckhpd(zmm22, zmm20, zmm5)\
	vunpcklpd(zmm23, zmm21, zmm6)\
	vunpckhpd(zmm23, zmm21, zmm7)\
	vunpcklpd(zmm26, zmm24, zmm8)\
	vunpckhpd(zmm26, zmm24, zmm9)\
	vunpcklpd(zmm27, zmm25, zmm10)\
	vunpckhpd(zmm27, zmm25, zmm11)\
	vunpcklpd(zmm30, zmm28, zmm12)\
	vunpckhpd(zmm30, zmm28, zmm13)\
	vunpcklpd(zmm31, zmm29, zmm14)\
	vunpckhpd(zmm31, zmm29, zmm15)
#define TRANSPOSE_SHUFFLE_3()\
	vshuff32x4(imm(0x88), zmm4, zmm0, zmm16)\
	vshuff32x4(imm(0x88), zmm5, zmm1, zmm17)\
	vshuff32x4(imm(0x88), zmm6, zmm2, zmm18)\
	vshuff32x4(imm(0x88), zmm7, zmm3, zmm19)\
	vshuff32x4(imm(0xDD), zmm4, zmm0, zmm20)\
	vshuff32x4(imm(0xDD), zmm5, zmm1, zmm21)\
	vshuff32x4(imm(0xDD), zmm6, zmm2, zmm22)\
	vshuff32x4(imm(0xDD), zmm7, zmm3, zmm23)\
	vshuff32x4(imm(0x88), zmm12, zmm8, zmm24)\
	vshuff32x4(imm(0x88), zmm13, zmm9, zmm25)\
	vshuff32x4(imm(0x88), zmm14, zmm10, zmm26)\
	vshuff32x4(imm(0x88), zmm15, zmm11, zmm27)\
	vshuff32x4(imm(0xDD), zmm12, zmm8, zmm28)\
	vshuff32x4(imm(0xDD), zmm13, zmm9, zmm29)\
	vshuff32x4(imm(0xDD), zmm14, zmm10, zmm30)\
	vshuff32x4(imm(0xDD), zmm15, zmm11, zmm31)\

#define TRANSPOSE_SHUFFLE_4a()\
	vshuff32x4(imm(0x88), zmm24, zmm16, zmm0)\
	vshuff32x4(imm(0x88), zmm25, zmm17, zmm1)\
	vshuff32x4(imm(0x88), zmm26, zmm18, zmm2)\
	vshuff32x4(imm(0x88), zmm27, zmm19, zmm3)\
	vshuff32x4(imm(0x88), zmm28, zmm20, zmm4)\
	vshuff32x4(imm(0x88), zmm29, zmm21, zmm5)\
	vshuff32x4(imm(0x88), zmm30, zmm22, zmm6)\
	vshuff32x4(imm(0x88), zmm31, zmm23, zmm7)\
	vshuff32x4(imm(0xDD), zmm24, zmm16, zmm8)\
	vshuff32x4(imm(0xDD), zmm25, zmm17, zmm9)\
	vshuff32x4(imm(0xDD), zmm26, zmm18, zmm10)\
	vshuff32x4(imm(0xDD), zmm27, zmm19, zmm11)\
	vshuff32x4(imm(0xDD), zmm28, zmm20, zmm12)\
	vshuff32x4(imm(0xDD), zmm29, zmm21, zmm13)\
	vshuff32x4(imm(0xDD), zmm30, zmm22, zmm14)\
	vshuff32x4(imm(0xDD), zmm31, zmm23, zmm15)
#define TRANSPOSE_SHUFFLE_4b()\
	vshuff32x4(imm(0xD8), zmm16, zmm16, zmm0)\
	vshuff32x4(imm(0xD8), zmm17, zmm17, zmm1)\
	vshuff32x4(imm(0xD8), zmm18, zmm18, zmm2)\
	vshuff32x4(imm(0xD8), zmm19, zmm19, zmm3)\
	vshuff32x4(imm(0xD8), zmm20, zmm20, zmm4)\
	vshuff32x4(imm(0xD8), zmm21, zmm21, zmm5)\
	vshuff32x4(imm(0xD8), zmm22, zmm22, zmm6)\
	vshuff32x4(imm(0xD8), zmm23, zmm23, zmm7)

#define TRANSPOSE_16x16xFP32()\
	TRANSPOSE_SHUFFLE_1()\
	TRANSPOSE_SHUFFLE_2()\
	TRANSPOSE_SHUFFLE_3()\
	TRANSPOSE_SHUFFLE_4a()

#define TRANSPOSE_8x16xFP32()\
	TRANSPOSE_SHUFFLE_1()\
	TRANSPOSE_SHUFFLE_2()\
	TRANSPOSE_SHUFFLE_3()\
	TRANSPOSE_SHUFFLE_4b()

namespace ml
{
	using namespace ml::cpu;
#ifdef __AVX512F__
	void gemm_avx512f_24x16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			bool use_relu) noexcept
	{
		assert(A.dtype() == DTYPE_FLOAT32);
		assert(B.dtype() == DTYPE_FLOAT32);
		assert(C.dtype() == DTYPE_FLOAT32);
		assert(D.dtype() == DTYPE_FLOAT32);
		assert(A.rows() == B.rows());
		assert(A.stride() == 24);
		assert(B.stride() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), 64));

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float *C_ptr = C.data<float>();
		float *D_ptr = D.data<float>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float);
		const uint64_t D_stride = D.stride() * sizeof(float);
		const uint64_t flag_relu = use_relu;

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

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rax), zmm1)
		vbroadcastss(mem(rbx), zmm0)

		// permute back to original layout
		PERMUTE_AND_SCALE_6x16xFP32(zmm8, zmm9, zmm10, zmm11, zmm12, zmm13)
		PERMUTE_AND_SCALE_6x16xFP32(zmm14, zmm15, zmm16, zmm17, zmm18, zmm19)
		PERMUTE_AND_SCALE_6x16xFP32(zmm20, zmm21, zmm22, zmm23, zmm24, zmm25)
		PERMUTE_AND_SCALE_6x16xFP32(zmm26, zmm27, zmm28, zmm29, zmm30, zmm31)

		vxorps(zmm1, zmm1, zmm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(APPLY_RELU)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)

		LOAD_ADD_3x16xFP32(zmm0, zmm8, zmm9, zmm10)
		LOAD_ADD_3x16xFP32(zmm0, zmm11, zmm12, zmm13)
		LOAD_ADD_3x16xFP32(zmm0, zmm14, zmm15, zmm16)
		LOAD_ADD_3x16xFP32(zmm0, zmm17, zmm18, zmm19)
		LOAD_ADD_3x16xFP32(zmm0, zmm20, zmm21, zmm22)
		LOAD_ADD_3x16xFP32(zmm0, zmm23, zmm24, zmm25)
		LOAD_ADD_3x16xFP32(zmm0, zmm26, zmm27, zmm28)
		LOAD_ADD_3x16xFP32(zmm0, zmm29, zmm30, zmm31)

		label(APPLY_RELU)
		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(STORE_D)
		// apply ReLU case
		RELU_24x16xFP32()

		label(STORE_D)
		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*D_stride)

		STORE_3x16xFP32(zmm8, zmm9, zmm10)
		STORE_3x16xFP32(zmm11, zmm12, zmm13)
		STORE_3x16xFP32(zmm14, zmm15, zmm16)
		STORE_3x16xFP32(zmm17, zmm18, zmm19)
		STORE_3x16xFP32(zmm20, zmm21, zmm22)
		STORE_3x16xFP32(zmm23, zmm24, zmm25)
		STORE_3x16xFP32(zmm26, zmm27, zmm28)
		STORE_3x16xFP32(zmm29, zmm30, zmm31)

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
				[flag_relu] "m"(flag_relu)
				:// clobbers
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
				"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
				"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
				"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%rcx", "%rdx", "%r14", "%r15")
	}
	void gemm_avx512f_24x16_fp32_fp16(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C, bool use_relu) noexcept
	{
		assert(A.dtype() == DTYPE_FLOAT32);
		assert(B.dtype() == DTYPE_FLOAT32);
		assert(C.dtype() == DTYPE_FLOAT16);
		assert(D.dtype() == DTYPE_FLOAT16);
		assert(A.rows() == B.rows());
		assert(A.stride() == 24);
		assert(B.stride() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(cpu::is_aligned(B.data(), 64));

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float16 *C_ptr = C.data<float16>();
		float16 *D_ptr = D.data<float16>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float16);
		const uint64_t D_stride = D.stride() * sizeof(float16);
		const uint64_t flag_relu = use_relu;

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

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rax), zmm1)
		vbroadcastss(mem(rbx), zmm0)

		// permute back to original layout
		PERMUTE_AND_SCALE_6x16xFP32(zmm8, zmm9, zmm10, zmm11, zmm12, zmm13)
		PERMUTE_AND_SCALE_6x16xFP32(zmm14, zmm15, zmm16, zmm17, zmm18, zmm19)
		PERMUTE_AND_SCALE_6x16xFP32(zmm20, zmm21, zmm22, zmm23, zmm24, zmm25)
		PERMUTE_AND_SCALE_6x16xFP32(zmm26, zmm27, zmm28, zmm29, zmm30, zmm31)

		vxorps(zmm1, zmm1, zmm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(APPLY_RELU)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)

		LOAD_ADD_3x16xFP16(zmm0, zmm8, zmm9, zmm10)
		LOAD_ADD_3x16xFP16(zmm0, zmm11, zmm12, zmm13)
		LOAD_ADD_3x16xFP16(zmm0, zmm14, zmm15, zmm16)
		LOAD_ADD_3x16xFP16(zmm0, zmm17, zmm18, zmm19)
		LOAD_ADD_3x16xFP16(zmm0, zmm20, zmm21, zmm22)
		LOAD_ADD_3x16xFP16(zmm0, zmm23, zmm24, zmm25)
		LOAD_ADD_3x16xFP16(zmm0, zmm26, zmm27, zmm28)
		LOAD_ADD_3x16xFP16(zmm0, zmm29, zmm30, zmm31)

		label(APPLY_RELU)
		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(STORE_D)
		// apply ReLU case
		RELU_24x16xFP32()

		label(STORE_D)
		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r15)// r15 = r14
		sal(imm(1), r15)// r15 = 2 * r14
		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*D_stride)

		CONVERT_ACCUMULATORS_TO_FP16()

		STORE_3x16xFP16(ymm8, ymm9, ymm10)
		STORE_3x16xFP16(ymm11, ymm12, ymm13)
		STORE_3x16xFP16(ymm14, ymm15, ymm16)
		STORE_3x16xFP16(ymm17, ymm18, ymm19)
		STORE_3x16xFP16(ymm20, ymm21, ymm22)
		STORE_3x16xFP16(ymm23, ymm24, ymm25)
		STORE_3x16xFP16(ymm26, ymm27, ymm28)
		STORE_3x16xFP16(ymm29, ymm30, ymm31)

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
				[flag_relu] "m"(flag_relu)
				:// clobbers
				"cc", "memory", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
				"%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
				"%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
				"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%rcx", "%rdx", "%r14", "%r15")
	}

	void pack_avx512f_24xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 24);
		assert(ml::cpu::is_aligned(dst.data(), 64));

		const uint64_t k_iter = dst.rows() / 16;
		const uint64_t k_left = dst.rows() % 16;
		const uint64_t src_stride = src.stride() * sizeof(float);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 16-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			LOAD_16x24xFP32()
			STORE_16x24xFP32()
			add(imm(4*16*24), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			LOAD_1x24xFP32(zmm0, ymm1)
			STORE_1x24xFP32(zmm0, ymm1, 0)
			add(imm(4*1*24), rbx)// add stride to dst pointer

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
					"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%r12", "%r14")
		}
		else
		{
			begin_asm()
			movq(var(src_ptr), r13) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 16-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			// first 16x16 tile
			movq(r13, rax)

			// first 16x16 tile
			LOAD_16x16xFP32()
			TRANSPOSE_16x16xFP32()
			STORE_16x16xFP32(24)

			// second 8x16 tile
			LOAD_8x16xFP32()
			TRANSPOSE_8x16xFP32()
			STORE_16x8xFP32()

			add(imm(4*16), r13)// add stride to src pointer
			add(imm(4*24*16), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(r13, rax)

			LOAD_4x1xFP32(0, 1, 2, 3)
			LOAD_4x1xFP32(4, 5, 6, 7)
			LOAD_4x1xFP32(8, 9, 10, 11)
			LOAD_4x1xFP32(12, 13, 14, 15)
			LOAD_4x1xFP32(16, 17, 18, 19)
			LOAD_4x1xFP32(20, 21, 22, 23)

			STORE_4x1xFP32(0, 1, 2, 3)
			STORE_4x1xFP32(4, 5, 6, 7)
			STORE_4x1xFP32(8, 9, 10, 11)
			STORE_4x1xFP32(12, 13, 14, 15)
			STORE_4x1xFP32(16, 17, 18, 19)
			STORE_4x1xFP32(20, 21, 22, 23)

			add(imm(4*1), r13)// add stride to src pointer
			add(imm(4*24*1), rbx)// add stride to dst pointer

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
					"%r12", "%r13", "%r14")
		}
	}
	void pack_avx512f_16xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 16);
		assert(ml::cpu::is_aligned(dst.data(), 64));

		const uint64_t k_iter = dst.rows() / 16;
		const uint64_t k_left = dst.rows() % 16;
		const uint64_t src_stride = src.stride() * sizeof(float);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 16-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			LOAD_16x16xFP32()
			STORE_16x16xFP32(16)
			add(imm(4*16*16), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			LOAD_1x16xFP32(zmm0)
			STORE_1x16xFP32(zmm0, 0)
			add(imm(4*1*16), rbx)// add stride to dst pointer

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
					"%r12", "%r14")
		}
		else
		{
			begin_asm()
			movq(var(src_ptr), r13) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 16-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			// first 16x16 tile
			movq(r13, rax)// tmp src pointer is in r13

			LOAD_16x16xFP32()
			TRANSPOSE_16x16xFP32()
			STORE_16x16xFP32(16)

			add(imm(4*16), r13)// add stride to src pointer
			add(imm(4*16*16), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(r13, rax)// tmp src pointer is in r13

			LOAD_4x1xFP32(0, 1, 2, 3)
			LOAD_4x1xFP32(4, 5, 6, 7)
			LOAD_4x1xFP32(8, 9, 10, 11)
			LOAD_4x1xFP32(12, 13, 14, 15)

			STORE_4x1xFP32(0, 1, 2, 3)
			STORE_4x1xFP32(4, 5, 6, 7)
			STORE_4x1xFP32(8, 9, 10, 11)
			STORE_4x1xFP32(12, 13, 14, 15)

			add(imm(4*1), r13)// add stride to src pointer
			add(imm(4*16*1), rbx)// add stride to dst pointer

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
					"%r12", "%r13", "%r14")
		}
	}

	void pack_avx512f_24xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 24);
		assert(ml::cpu::is_aligned(dst.data(), 64));

		const uint64_t k_iter = dst.rows() / 16;
		const uint64_t k_left = dst.rows() % 16;
		const uint64_t src_stride = src.stride() * sizeof(float16);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 16-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			LOAD_16x24xFP16()
			STORE_16x24xFP32()
			add(imm(4*16*24), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			LOAD_1x24xFP16(ymm0, xmm1)
			CONVERT_1x24xFP16_TO_FP32(0, 1)
			STORE_1x24xFP32(zmm0, ymm1, 0)
			add(imm(4*1*24), rbx)// add stride to dst pointer

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
					"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%r12", "%r14")
		}
		else
		{
			begin_asm()
			movq(var(src_ptr), r13) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 16-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			// first 16x16 tile
			movq(r13, rax)

			// first 16x16 tile
			LOAD_16x16xFP16()
			TRANSPOSE_16x16xFP32()
			STORE_16x16xFP32(24)

			// second 8x16 tile
			LOAD_8x16xFP16()
			TRANSPOSE_8x16xFP32()
			STORE_16x8xFP32()

			add(imm(2*16), r13)// add stride to src pointer
			add(imm(4*24*16), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(r13, rax)

			LOAD_4x1xFP16(0, 1, 2, 3)
			LOAD_4x1xFP16(4, 5, 6, 7)
			LOAD_4x1xFP16(8, 9, 10, 11)
			LOAD_4x1xFP16(12, 13, 14, 15)
			LOAD_4x1xFP16(16, 17, 18, 19)
			LOAD_4x1xFP16(20, 21, 22, 23)

			STORE_4x1xFP32(0, 1, 2, 3)
			STORE_4x1xFP32(4, 5, 6, 7)
			STORE_4x1xFP32(8, 9, 10, 11)
			STORE_4x1xFP32(12, 13, 14, 15)
			STORE_4x1xFP32(16, 17, 18, 19)
			STORE_4x1xFP32(20, 21, 22, 23)

			add(imm(2*1), r13)// add stride to src pointer
			add(imm(4*24*1), rbx)// add stride to dst pointer

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
					"%r12", "%r13", "%r14")
		}
	}
	void pack_avx512f_16xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.stride() == 16);
		assert(ml::cpu::is_aligned(dst.data(), 64));

		const uint64_t k_iter = dst.rows() / 16;
		const uint64_t k_left = dst.rows() % 16;
		const uint64_t src_stride = src.stride() * sizeof(float16);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 16-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			LOAD_16x16xFP16()
			STORE_16x16xFP32(16)
			add(imm(4*16*16), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			LOAD_1x16xFP16(ymm0)
			CONVERT_1x16xFP16_TO_FP32(0)
			STORE_1x16xFP32(zmm0, 0)
			add(imm(4*1*16), rbx)// add stride to dst pointer

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
					"%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "%rax", "%rbx", "%r12", "%r14")
		}
		else
		{
			begin_asm()
			movq(var(src_ptr), r13) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 16-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED16)
			// first 16x16 tile
			movq(r13, rax)// tmp src pointer is in r13

			LOAD_16x16xFP16()
			TRANSPOSE_16x16xFP32()
			STORE_16x16xFP32(16)

			add(imm(2*16), r13)// add stride to src pointer
			add(imm(4*16*16), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED16)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(r13, rax)// tmp src pointer is in r13

			LOAD_4x1xFP16(0, 1, 2, 3)
			LOAD_4x1xFP16(4, 5, 6, 7)
			LOAD_4x1xFP16(8, 9, 10, 11)
			LOAD_4x1xFP16(12, 13, 14, 15)

			STORE_4x1xFP32(0, 1, 2, 3)
			STORE_4x1xFP32(4, 5, 6, 7)
			STORE_4x1xFP32(8, 9, 10, 11)
			STORE_4x1xFP32(12, 13, 14, 15)

			add(imm(2*1), r13)// add stride to src pointer
			add(imm(4*16*1), rbx)// add stride to dst pointer

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
					"%r12", "%r13", "%r14")
		}
	}
#else
	void gemm_avx512f_24x16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			bool use_relu) noexcept
	{
	}
	void gemm_avx512f_24x16_fp32_fp16(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C, bool use_relu) noexcept
	{
	}
	void pack_avx512f_24xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
	}
	void pack_avx512f_24xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
	}
	void pack_avx512f_16xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
	}
	void pack_avx512f_16xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
	}
#endif
} /* namespace ml */

