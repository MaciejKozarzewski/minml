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
#include "../fp16.hpp"

#include <cinttypes>
#include <cassert>
#include <cmath>
#include <x86intrin.h>

#include "../assembly_macros.hpp"
#include "common_operations.hpp"

#define ZERO_ACCUMULATORS()\
	vxorps(ymm4, ymm4, ymm4)\
	vxorps(ymm5, ymm5, ymm5)\
	vxorps(ymm6, ymm6, ymm6)\
	vxorps(ymm7, ymm7, ymm7)\
	vxorps(ymm8, ymm8, ymm8)\
	vxorps(ymm9, ymm9, ymm9)\
	vxorps(ymm10, ymm10, ymm10)\
	vxorps(ymm11, ymm11, ymm11)\
	vxorps(ymm12, ymm12, ymm12)\
	vxorps(ymm13, ymm13, ymm13)\
	vxorps(ymm14, ymm14, ymm14)\
	vxorps(ymm15, ymm15, ymm15)\

#define SUB_KERNEL_12xFP32_8xFP32(n) \
	vmovaps(mem(rbx, n*8*4), ymm2)\
	vpermilps(imm(0xB1), ymm2, ymm3)\
	vbroadcastsd(mem(rax, (12*n+0)*4), ymm0)\
	vbroadcastsd(mem(rax, (12*n+2)*4), ymm1)\
	vfmadd231ps(ymm0, ymm2, ymm4)\
	vfmadd231ps(ymm0, ymm3, ymm5)\
	vfmadd231ps(ymm1, ymm2, ymm6)\
	vfmadd231ps(ymm1, ymm3, ymm7)\
	vbroadcastsd(mem(rax, (12*n+4)*4), ymm0)\
	vbroadcastsd(mem(rax, (12*n+6)*4), ymm1)\
	vfmadd231ps(ymm0, ymm2, ymm8)\
	vfmadd231ps(ymm0, ymm3, ymm9)\
	vfmadd231ps(ymm1, ymm2, ymm10)\
	vfmadd231ps(ymm1, ymm3, ymm11)\
	vbroadcastsd(mem(rax, (12*n+8)*4), ymm0)\
	vbroadcastsd(mem(rax, (12*n+10)*4), ymm1)\
	vfmadd231ps(ymm0, ymm2, ymm12)\
	vfmadd231ps(ymm0, ymm3, ymm13)\
	vfmadd231ps(ymm1, ymm2, ymm14)\
	vfmadd231ps(ymm1, ymm3, ymm15)
#define SUB_KERNEL_6xFP32_16xFP32(n) \
	vmovaps(mem(rbx, (n*16+0)*4), ymm2)\
	vmovaps(mem(rbx, (n*16+8)*4), ymm3)\
	vbroadcastss(mem(rax, (6*n+0)*4), ymm0)\
	vbroadcastss(mem(rax, (6*n+1)*4), ymm1)\
	vfmadd231ps(ymm0, ymm2, ymm4)\
	vfmadd231ps(ymm0, ymm3, ymm5)\
	vfmadd231ps(ymm1, ymm2, ymm6)\
	vfmadd231ps(ymm1, ymm3, ymm7)\
	vbroadcastss(mem(rax, (6*n+2)*4), ymm0)\
	vbroadcastss(mem(rax, (6*n+3)*4), ymm1)\
	vfmadd231ps(ymm0, ymm2, ymm8)\
	vfmadd231ps(ymm0, ymm3, ymm9)\
	vfmadd231ps(ymm1, ymm2, ymm10)\
	vfmadd231ps(ymm1, ymm3, ymm11)\
	vbroadcastss(mem(rax, (6*n+4)*4), ymm0)\
	vbroadcastss(mem(rax, (6*n+5)*4), ymm1)\
	vfmadd231ps(ymm0, ymm2, ymm12)\
	vfmadd231ps(ymm0, ymm3, ymm13)\
	vfmadd231ps(ymm1, ymm2, ymm14)\
	vfmadd231ps(ymm1, ymm3, ymm15)

#define SUB_KERNEL_12xINT16_8xINT16(n) \
	vmovaps(mem(rbx, n*16*2), ymm0)\
	vbroadcastss(mem(rax, (12*n+0)*4), ymm1)\
	vbroadcastss(mem(rax, (12*n+1)*4), ymm2)\
	vpmaddwd(ymm0, ymm1, ymm1)\
	vpmaddwd(ymm0, ymm2, ymm2)\
	vpaddd(ymm1, ymm4, ymm4)\
	vpaddd(ymm2, ymm5, ymm5)\
	vbroadcastss(mem(rax, (12*n+2)*4), ymm1)\
	vbroadcastss(mem(rax, (12*n+3)*4), ymm2)\
	vpmaddwd(ymm0, ymm1, ymm1)\
	vpmaddwd(ymm0, ymm2, ymm2)\
	vpaddd(ymm1, ymm6, ymm6)\
	vpaddd(ymm2, ymm7, ymm7)\
	vbroadcastss(mem(rax, (12*n+4)*4), ymm1)\
	vbroadcastss(mem(rax, (12*n+5)*4), ymm2)\
	vpmaddwd(ymm0, ymm1, ymm1)\
	vpmaddwd(ymm0, ymm2, ymm2)\
	vpaddd(ymm1, ymm8, ymm8)\
	vpaddd(ymm2, ymm9, ymm9)\
	vbroadcastss(mem(rax, (12*n+6)*4), ymm1)\
	vbroadcastss(mem(rax, (12*n+7)*4), ymm2)\
	vpmaddwd(ymm0, ymm1, ymm1)\
	vpmaddwd(ymm0, ymm2, ymm2)\
	vpaddd(ymm1, ymm10, ymm10)\
	vpaddd(ymm2, ymm11, ymm11)\
	vbroadcastss(mem(rax, (12*n+8)*4), ymm1)\
	vbroadcastss(mem(rax, (12*n+9)*4), ymm2)\
	vpmaddwd(ymm0, ymm1, ymm1)\
	vpmaddwd(ymm0, ymm2, ymm2)\
	vpaddd(ymm1, ymm12, ymm12)\
	vpaddd(ymm2, ymm13, ymm13)\
	vbroadcastss(mem(rax, (12*n+10)*4), ymm1)\
	vbroadcastss(mem(rax, (12*n+11)*4), ymm2)\
	vpmaddwd(ymm0, ymm1, ymm1)\
	vpmaddwd(ymm0, ymm2, ymm2)\
	vpaddd(ymm1, ymm14, ymm14)\
	vpaddd(ymm2, ymm15, ymm15)
//#define SUB_KERNEL_12xINT16_8xINT16(n) \
//	vmovaps(mem(rbx, n*16*2), ymm0)\
//	vbroadcastss(mem(rax, (12*n+0)*4), ymm2)\
//	vbroadcastss(mem(rax, (12*n+1)*4), ymm3)\
//	vpmaddubsw(ymm0, ymm2, ymm2)\
//	vpmaddubsw(ymm0, ymm3, ymm3)\
//	vpmaddwd(ymm1, ymm2, ymm2)\
//	vpmaddwd(ymm1, ymm3, ymm3)\
//	vpaddd(ymm2, ymm4, ymm4)\
//	vpaddd(ymm3, ymm5, ymm5)\
//	vbroadcastss(mem(rax, (12*n+2)*4), ymm2)\
//	vbroadcastss(mem(rax, (12*n+3)*4), ymm3)\
//	vpmaddubsw(ymm0, ymm2, ymm2)\
//	vpmaddubsw(ymm0, ymm3, ymm3)\
//	vpmaddwd(ymm1, ymm2, ymm2)\
//	vpmaddwd(ymm1, ymm3, ymm3)\
//	vpaddd(ymm2, ymm6, ymm6)\
//	vpaddd(ymm3, ymm7, ymm7)\
//	vbroadcastss(mem(rax, (12*n+4)*4), ymm2)\
//	vbroadcastss(mem(rax, (12*n+5)*4), ymm3)\
//	vpmaddubsw(ymm0, ymm2, ymm2)\
//	vpmaddubsw(ymm0, ymm3, ymm3)\
//	vpmaddwd(ymm1, ymm2, ymm2)\
//	vpmaddwd(ymm1, ymm3, ymm3)\
//	vpaddd(ymm2, ymm8, ymm8)\
//	vpaddd(ymm3, ymm9, ymm9)\
//	vbroadcastss(mem(rax, (12*n+6)*4), ymm2)\
//	vbroadcastss(mem(rax, (12*n+7)*4), ymm3)\
//	vpmaddubsw(ymm0, ymm2, ymm2)\
//	vpmaddubsw(ymm0, ymm3, ymm3)\
//	vpmaddwd(ymm1, ymm2, ymm2)\
//	vpmaddwd(ymm1, ymm3, ymm3)\
//	vpaddd(ymm2, ymm10, ymm10)\
//	vpaddd(ymm3, ymm11, ymm11)\
//	vbroadcastss(mem(rax, (12*n+8)*4), ymm2)\
//	vbroadcastss(mem(rax, (12*n+9)*4), ymm3)\
//	vpmaddubsw(ymm0, ymm2, ymm2)\
//	vpmaddubsw(ymm0, ymm3, ymm3)\
//	vpmaddwd(ymm1, ymm2, ymm2)\
//	vpmaddwd(ymm1, ymm3, ymm3)\
//	vpaddd(ymm2, ymm12, ymm12)\
//	vpaddd(ymm3, ymm13, ymm13)\
//	vbroadcastss(mem(rax, (12*n+10)*4), ymm2)\
//	vbroadcastss(mem(rax, (12*n+11)*4), ymm3)\
//	vpmaddubsw(ymm0, ymm2, ymm2)\
//	vpmaddubsw(ymm0, ymm3, ymm3)\
//	vpmaddwd(ymm1, ymm2, ymm2)\
//	vpmaddwd(ymm1, ymm3, ymm3)\
//	vpaddd(ymm2, ymm14, ymm14)\
//	vpaddd(ymm3, ymm15, ymm15)
//#define SUB_KERNEL_12xINT16_8xINT16(n) \
//	vpmaddwd(ymm0, ymm2, ymm4)\
//	vpmaddwd(ymm0, ymm3, ymm5)\
//	vpmaddwd(ymm1, ymm2, ymm6)\
//	vpmaddwd(ymm1, ymm3, ymm7)\
//	vpmaddwd(ymm0, ymm2, ymm8)\
//	vpmaddwd(ymm0, ymm3, ymm9)\
//	vpmaddwd(ymm1, ymm2, ymm10)\
//	vpmaddwd(ymm1, ymm3, ymm11)\
//	vpmaddwd(ymm0, ymm2, ymm12)\
//	vpmaddwd(ymm0, ymm3, ymm13)\
//	vpmaddwd(ymm1, ymm2, ymm14)\
//	vpmaddwd(ymm1, ymm3, ymm15)\
//	vpaddd(ymm4, ymm4, ymm4)\
//	vpaddd(ymm5, ymm5, ymm5)\
//	vpaddd(ymm6, ymm6, ymm6)\
//	vpaddd(ymm7, ymm7, ymm7)\
//	vpaddd(ymm8, ymm8, ymm8)\
//	vpaddd(ymm9, ymm9, ymm9)\
//	vpaddd(ymm10, ymm10, ymm10)\
//	vpaddd(ymm11, ymm11, ymm11)\
//	vpaddd(ymm12, ymm12, ymm12)\
//	vpaddd(ymm13, ymm13, ymm13)\
//	vpaddd(ymm14, ymm14, ymm14)\
//	vpaddd(ymm15, ymm15, ymm15)

#define STORE_4x8xINT8(reg0, reg1, reg2, reg3)\
	movsd(reg0, mem(rcx)) \
	movsd(reg1, mem(rcx, r14, 1)) \
	movsd(reg2, mem(rcx, r14, 2)) \
	movsd(reg3, mem(rcx, r13, 1)) \
	add(r15, rcx)

#define ADD_BIAS_12x8xFP32(reg)\
	vaddps(reg, ymm4, ymm4) \
	vaddps(reg, ymm5, ymm5) \
	vaddps(reg, ymm6, ymm6) \
	vaddps(reg, ymm7, ymm7) \
	vaddps(reg, ymm8, ymm8) \
	vaddps(reg, ymm9, ymm9) \
	vaddps(reg, ymm10, ymm10) \
	vaddps(reg, ymm11, ymm11) \
	vaddps(reg, ymm12, ymm12) \
	vaddps(reg, ymm13, ymm13) \
	vaddps(reg, ymm14, ymm14) \
	vaddps(reg, ymm15, ymm15)
#define ADD_BIAS_6x16xFP32(reg0, reg1)\
	vaddps(reg0, ymm4, ymm4) \
	vaddps(reg1, ymm5, ymm5) \
	vaddps(reg0, ymm6, ymm6) \
	vaddps(reg1, ymm7, ymm7) \
	vaddps(reg0, ymm8, ymm8) \
	vaddps(reg1, ymm9, ymm9) \
	vaddps(reg0, ymm10, ymm10) \
	vaddps(reg1, ymm11, ymm11) \
	vaddps(reg0, ymm12, ymm12) \
	vaddps(reg1, ymm13, ymm13) \
	vaddps(reg0, ymm14, ymm14) \
	vaddps(reg1, ymm15, ymm15)

#define LOAD_ADD_3x8xFP32(beta, reg0, reg1, reg2)\
	vmovups(mem(rcx), ymm1)\
	vmovups(mem(rcx, r14, 1), ymm2)\
	vmovups(mem(rcx, r14, 2), ymm3)\
	vfmadd231ps(ymm1, beta, reg0)\
	vfmadd231ps(ymm2, beta, reg1)\
	vfmadd231ps(ymm3, beta, reg2)\
	add(r15, rcx)
#define LOAD_ADD_1x16xFP32(beta, reg0, reg1)\
	vmovups(mem(rcx), ymm1)\
	vmovups(mem(rcx, 8*4), ymm2)\
	vfmadd231ps(ymm1, beta, reg0)\
	vfmadd231ps(ymm2, beta, reg1)\
	add(r14, rcx)

#define LOAD_ADD_8xFP16(beta, reg)\
	vmovups(mem(rcx), xmm2)\
	vcvtph2ps(xmm2, ymm2)\
	vfmadd231ps(ymm2, beta, reg)\
	add(r14, rcx)

#define STORE_3x8xFP32(reg0, reg1, reg2)\
	vmovups(reg0, mem(rcx))\
	vmovups(reg1, mem(rcx, r14, 1))\
	vmovups(reg2, mem(rcx, r14, 2))\
	add(r15, rcx)
#define STORE_4x8xFP32(reg0, reg1, reg2, reg3)\
	vmovups(reg0, mem(rcx)) \
	vmovups(reg1, mem(rcx, r14, 1)) \
	vmovups(reg2, mem(rcx, r14, 2)) \
	vmovups(reg3, mem(rcx, r13, 1)) \
	add(r15, rcx)
#define STORE_2x16xFP32(reg0, reg1, reg2, reg3) \
	vmovups(reg0, mem(rcx)) \
	vmovups(reg1, mem(rcx, 8*4)) \
	vmovups(reg2, mem(rcx, r14, 1)) \
	vmovups(reg3, mem(rcx, r14, 1, 8*4)) \
	add(r15, rcx)

#define PERMUTE_AND_SCALE_4x8xFP32(reg0, reg1, reg2, reg3)\
	vpermilps(imm(0xB1), reg1, reg1)\
	vpermilps(imm(0xB1), reg3, reg3)\
	vblendps(imm(0x55), reg0, reg1, ymm2)\
	vblendps(imm(0xAA), reg0, reg1, reg1)\
	vblendps(imm(0x55), reg2, reg3, ymm3)\
	vblendps(imm(0xAA), reg2, reg3, reg3)\
	vmulps(ymm1, ymm2, reg0)\
	vmulps(ymm1, reg1, reg1)\
	vmulps(ymm1, ymm3, reg2)\
	vmulps(ymm1, reg3, reg3)
#define PERMUTE_6x8xFP32(reg0, reg1, reg2, reg3, reg4, reg5) \
	vpermilps(imm(0xB1), reg1, ymm1) \
	vpermilps(imm(0xB1), reg3, ymm2) \
	vpermilps(imm(0xB1), reg5, ymm3) \
	vblendps(imm(0xAA), reg0, ymm1, reg1) \
	vblendps(imm(0x55), reg0, ymm1, reg0) \
	vblendps(imm(0xAA), reg2, ymm2, reg3) \
	vblendps(imm(0x55), reg2, ymm2, reg2) \
	vblendps(imm(0xAA), reg4, ymm3, reg5) \
	vblendps(imm(0x55), reg4, ymm3, reg4)

#define SCALE_ACCUMULATORS_1xN(scale) \
	vmulps(scale, ymm4, ymm4) \
	vmulps(scale, ymm5, ymm5) \
	vmulps(scale, ymm6, ymm6) \
	vmulps(scale, ymm7, ymm7) \
	vmulps(scale, ymm8, ymm8) \
	vmulps(scale, ymm9, ymm9) \
	vmulps(scale, ymm10, ymm10) \
	vmulps(scale, ymm11, ymm11) \
	vmulps(scale, ymm12, ymm12) \
	vmulps(scale, ymm13, ymm13) \
	vmulps(scale, ymm14, ymm14) \
	vmulps(scale, ymm15, ymm15)
#define SCALE_ACCUMULATORS_12x1() \
	vbroadcastss(mem(rax, 0*4), ymm0) \
	vbroadcastss(mem(rax, 1*4), ymm1) \
	vbroadcastss(mem(rax, 2*4), ymm2) \
	vbroadcastss(mem(rax, 3*4), ymm3) \
	vmulps(ymm0, ymm4, ymm4) \
	vmulps(ymm1, ymm5, ymm5) \
	vmulps(ymm2, ymm6, ymm6) \
	vmulps(ymm3, ymm7, ymm7) \
	vbroadcastss(mem(rax, 4*4), ymm0) \
	vbroadcastss(mem(rax, 5*4), ymm1) \
	vbroadcastss(mem(rax, 6*4), ymm2) \
	vbroadcastss(mem(rax, 7*4), ymm3) \
	vmulps(ymm0, ymm8, ymm8) \
	vmulps(ymm1, ymm9, ymm9) \
	vmulps(ymm2, ymm10, ymm10) \
	vmulps(ymm3, ymm11, ymm11) \
	vbroadcastss(mem(rax, 8*4), ymm0) \
	vbroadcastss(mem(rax, 9*4), ymm1) \
	vbroadcastss(mem(rax, 10*4), ymm2) \
	vbroadcastss(mem(rax, 11*4), ymm3) \
	vmulps(ymm0, ymm12, ymm12) \
	vmulps(ymm1, ymm13, ymm13) \
	vmulps(ymm2, ymm14, ymm14) \
	vmulps(ymm3, ymm15, ymm15)
#define SCALE_ACCUMULATORS_6x1() \
	vbroadcastss(mem(rax, 0*4), ymm0) \
	vbroadcastss(mem(rax, 1*4), ymm1) \
	vbroadcastss(mem(rax, 2*4), ymm2) \
	vbroadcastss(mem(rax, 3*4), ymm3) \
	vmulps(ymm0, ymm4, ymm4) \
	vmulps(ymm0, ymm5, ymm5) \
	vmulps(ymm1, ymm6, ymm6) \
	vmulps(ymm1, ymm7, ymm7) \
	vmulps(ymm2, ymm8, ymm8) \
	vmulps(ymm2, ymm9, ymm9) \
	vmulps(ymm3, ymm10, ymm10) \
	vmulps(ymm3, ymm11, ymm11) \
	vbroadcastss(mem(rax, 4*4), ymm0) \
	vbroadcastss(mem(rax, 5*4), ymm1) \
	vmulps(ymm0, ymm12, ymm12) \
	vmulps(ymm0, ymm13, ymm13) \
	vmulps(ymm1, ymm14, ymm14) \
	vmulps(ymm1, ymm15, ymm15)

#define CONVERT_ACCUMULATORS_TO_FP16()\
	vcvtps2ph(imm(0x03), ymm4, xmm4)\
	vcvtps2ph(imm(0x03), ymm5, xmm5)\
	vcvtps2ph(imm(0x03), ymm6, xmm6)\
	vcvtps2ph(imm(0x03), ymm7, xmm7)\
	vcvtps2ph(imm(0x03), ymm8, xmm8)\
	vcvtps2ph(imm(0x03), ymm9, xmm9)\
	vcvtps2ph(imm(0x03), ymm10, xmm10)\
	vcvtps2ph(imm(0x03), ymm11, xmm11)\
	vcvtps2ph(imm(0x03), ymm12, xmm12)\
	vcvtps2ph(imm(0x03), ymm13, xmm13)\
	vcvtps2ph(imm(0x03), ymm14, xmm14)\
	vcvtps2ph(imm(0x03), ymm15, xmm15)

#define LOAD_ADD_3x8xFP16(beta, reg0, reg1, reg2) \
	vmovups(mem(rcx), xmm1) \
	vmovups(mem(rcx, r14, 1), xmm2) \
	vmovups(mem(rcx, r14, 2), xmm3) \
	vcvtph2ps(xmm1, ymm1) \
	vcvtph2ps(xmm2, ymm2) \
	vcvtph2ps(xmm3, ymm3) \
	vfmadd231ps(ymm1, beta, reg0) \
	vfmadd231ps(ymm2, beta, reg1) \
	vfmadd231ps(ymm3, beta, reg2) \
	add(r15, rcx)
#define LOAD_ADD_1x16xFP16(beta, reg0, reg1) \
	vmovups(mem(rcx), xmm1) \
	vmovups(mem(rcx, 8*2), xmm2) \
	vcvtph2ps(xmm1, ymm1) \
	vcvtph2ps(xmm2, ymm2) \
	vfmadd231ps(ymm1, beta, reg0) \
	vfmadd231ps(ymm2, beta, reg1) \
	add(r14, rcx)

#define STORE_3x8xFP16(reg0, reg1, reg2) \
	vmovups(reg0, mem(rcx)) \
	vmovups(reg1, mem(rcx, r14, 1)) \
	vmovups(reg2, mem(rcx, r14, 2)) \
	add(r15, rcx)
#define STORE_4x8xFP16(reg0, reg1, reg2, reg3) \
	vmovups(reg0, mem(rcx)) \
	vmovups(reg1, mem(rcx, r14, 1)) \
	vmovups(reg2, mem(rcx, r14, 2)) \
	vmovups(reg3, mem(rcx, r13, 1)) \
	add(r15, rcx)

#define STORE_2x16xFP16(reg0, reg1, reg2, reg3) \
	vmovups(reg0, mem(rcx)) \
	vmovups(reg1, mem(rcx, 8*2)) \
	vmovups(reg2, mem(rcx, r14, 1)) \
	vmovups(reg3, mem(rcx, r14, 1, 8*2)) \
	add(r15, rcx)

#define RELU_12x8xFP32()\
	vxorps(ymm0, ymm0, ymm0)\
	vmaxps(ymm0, ymm4, ymm4)\
	vmaxps(ymm0, ymm5, ymm5)\
	vmaxps(ymm0, ymm6, ymm6)\
	vmaxps(ymm0, ymm7, ymm7)\
	vmaxps(ymm0, ymm8, ymm8)\
	vmaxps(ymm0, ymm9, ymm9)\
	vmaxps(ymm0, ymm10, ymm10)\
	vmaxps(ymm0, ymm11, ymm11)\
	vmaxps(ymm0, ymm12, ymm12)\
	vmaxps(ymm0, ymm13, ymm13)\
	vmaxps(ymm0, ymm14, ymm14)\
	vmaxps(ymm0, ymm15, ymm15)

/*
 * packing helper macros
 */
#define LOAD_4x12xFP32() \
	vmovups(mem(rax), ymm0) \
	vmovups(mem(rax, 8*4), xmm1) \
	vmovups(mem(rax, r12, 1), ymm2) \
	vmovups(mem(rax, r12, 1, 8*4), xmm3) \
	vmovups(mem(rax, r12, 2), ymm4) \
	vmovups(mem(rax, r12, 2, 8*4), xmm5) \
	vmovups(mem(rax, r13, 1), ymm6) \
	vmovups(mem(rax, r13, 1, 8*4), xmm7) \
	add(r15, rax)
#define LOAD_8x12xFP32() \
	vmovups(mem(rax), ymm0) \
	vmovups(mem(rax, 8*4), xmm1) \
	vmovups(mem(rax, r12, 1), ymm2) \
	vmovups(mem(rax, r12, 1, 8*4), xmm3) \
	vmovups(mem(rax, r12, 2), ymm4) \
	vmovups(mem(rax, r12, 2, 8*4), xmm5) \
	vmovups(mem(rax, r13, 1), ymm6) \
	vmovups(mem(rax, r13, 1, 8*4), xmm7) \
	add(r15, rax) \
	vmovups(mem(rax), ymm8) \
	vmovups(mem(rax, 8*4), xmm9) \
	vmovups(mem(rax, r12, 1), ymm10) \
	vmovups(mem(rax, r12, 1, 8*4), xmm11) \
	vmovups(mem(rax, r12, 2), ymm12) \
	vmovups(mem(rax, r12, 2, 8*4), xmm13) \
	vmovups(mem(rax, r13, 1), ymm14) \
	vmovups(mem(rax, r13, 1, 8*4), xmm15) \
	add(r15, rax)
#define LOAD_4x12xFP16() \
	vmovups(mem(rax), xmm0) \
	vmovsd (mem(rax, 8*2), xmm1) \
	vmovups(mem(rax, r12, 1), xmm2) \
	vmovsd (mem(rax, r12, 1, 8*2), xmm3) \
	vmovups(mem(rax, r12, 2), xmm4) \
	vmovsd (mem(rax, r12, 2, 8*2), xmm5) \
	vmovups(mem(rax, r13, 1), xmm6) \
	vmovsd (mem(rax, r13, 1, 8*2), xmm7) \
	add(r15, rax)
#define LOAD_8x12xFP16() \
	vmovups(mem(rax), xmm0) \
	vmovsd (mem(rax, 8*2), xmm1) \
	vmovups(mem(rax, r12, 1), xmm2) \
	vmovsd (mem(rax, r12, 1, 8*2), xmm3) \
	vmovups(mem(rax, r12, 2), xmm4) \
	vmovsd (mem(rax, r12, 2, 8*2), xmm5) \
	vmovups(mem(rax, r13, 1), xmm6) \
	vmovsd (mem(rax, r13, 1, 8*2), xmm7) \
	add(r15, rax) \
	vmovups(mem(rax), xmm8) \
	vmovsd (mem(rax, 8*2), xmm9) \
	vmovups(mem(rax, r12, 1), xmm10) \
	vmovsd (mem(rax, r12, 1, 8*2), xmm11) \
	vmovups(mem(rax, r12, 2), xmm12) \
	vmovsd (mem(rax, r12, 2, 8*2), xmm13) \
	vmovups(mem(rax, r13, 1), xmm14) \
	vmovsd (mem(rax, r13, 1, 8*2), xmm15) \
	add(r15, rax)
#define LOAD_8x8xFP32() \
	vmovups(mem(rcx), ymm0) \
	vmovups(mem(rcx, r12, 1), ymm1) \
	vmovups(mem(rcx, r12, 2), ymm2) \
	vmovups(mem(rcx, r13, 1), ymm3) \
	add(r15, rcx) \
	vmovups(mem(rcx), ymm4) \
	vmovups(mem(rcx, r12, 1), ymm5) \
	vmovups(mem(rcx, r12, 2), ymm6) \
	vmovups(mem(rcx, r13, 1), ymm7) \
	add(r15, rcx)
#define LOAD_4x8xFP32() \
	vmovups(mem(rcx), ymm4) \
	vmovups(mem(rcx, r12, 1), ymm5) \
	vmovups(mem(rcx, r12, 2), ymm6) \
	vmovups(mem(rcx, r13, 1), ymm7)
#define LOAD_12x1xFP32() \
	vmovss(mem(rcx), xmm0) \
	vmovss(mem(rcx, r12, 1), xmm1) \
	vmovss(mem(rcx, r12, 2), xmm2) \
	vmovss(mem(rcx, r13, 1), xmm3) \
	add(r15, rcx) \
	vmovss(mem(rcx), xmm4) \
	vmovss(mem(rcx, r12, 1), xmm5) \
	vmovss(mem(rcx, r12, 2), xmm6) \
	vmovss(mem(rcx, r13, 1), xmm7) \
	add(r15, rcx) \
	vmovss(mem(rcx), xmm8) \
	vmovss(mem(rcx, r12, 1), xmm9) \
	vmovss(mem(rcx, r12, 2), xmm10) \
	vmovss(mem(rcx, r13, 1), xmm11)
#define LOAD_16x1xFP32() \
	vmovss(mem(rcx), xmm0) \
	vmovss(mem(rcx, r12, 1), xmm1) \
	vmovss(mem(rcx, r12, 2), xmm2) \
	vmovss(mem(rcx, r13, 1), xmm3) \
	add(r15, rcx) \
	vmovss(mem(rcx), xmm4) \
	vmovss(mem(rcx, r12, 1), xmm5) \
	vmovss(mem(rcx, r12, 2), xmm6) \
	vmovss(mem(rcx, r13, 1), xmm7) \
	add(r15, rcx) \
	vmovss(mem(rcx), xmm8) \
	vmovss(mem(rcx, r12, 1), xmm9) \
	vmovss(mem(rcx, r12, 2), xmm10) \
	vmovss(mem(rcx, r13, 1), xmm11) \
	add(r15, rcx) \
	vmovss(mem(rcx), xmm12) \
	vmovss(mem(rcx, r12, 1), xmm13) \
	vmovss(mem(rcx, r12, 2), xmm14) \
	vmovss(mem(rcx, r13, 1), xmm15)
#define LOAD_8x8xFP16() \
	vmovups(mem(rcx), xmm0) \
	vmovups(mem(rcx, r12, 1), xmm1) \
	vmovups(mem(rcx, r12, 2), xmm2) \
	vmovups(mem(rcx, r13, 1), xmm3) \
	add(r15, rcx) \
	vmovups(mem(rcx), xmm4) \
	vmovups(mem(rcx, r12, 1), xmm5) \
	vmovups(mem(rcx, r12, 2), xmm6) \
	vmovups(mem(rcx, r13, 1), xmm7) \
	add(r15, rcx)
#define LOAD_4x8xFP16() \
	vmovups(mem(rcx), xmm4) \
	vmovups(mem(rcx, r12, 1), xmm5) \
	vmovups(mem(rcx, r12, 2), xmm6) \
	vmovups(mem(rcx, r13, 1), xmm7)
#define LOAD_4x16xFP32() \
	vmovups(mem(rax), ymm0) \
	vmovups(mem(rax, 8*4), ymm1) \
	vmovups(mem(rax, r12, 1), ymm2) \
	vmovups(mem(rax, r12, 1, 8*4), ymm3) \
	vmovups(mem(rax, r12, 2), ymm4) \
	vmovups(mem(rax, r12, 2, 8*4), ymm5) \
	vmovups(mem(rax, r13, 1), ymm6) \
	vmovups(mem(rax, r13, 1, 8*4), ymm7) \
	add(r15, rax)
#define LOAD_4x16xFP16() \
	vmovups(mem(rax), xmm0) \
	vmovups(mem(rax, 8*2), xmm1) \
	vmovups(mem(rax, r12, 1), xmm2) \
	vmovups(mem(rax, r12, 1, 8*2), xmm3) \
	vmovups(mem(rax, r12, 2), xmm4) \
	vmovups(mem(rax, r12, 2, 8*2), xmm5) \
	vmovups(mem(rax, r13, 1), xmm6) \
	vmovups(mem(rax, r13, 1, 8*2), xmm7) \
	add(r15, rax)

#define STORE_4x12xFP32() \
	vmovaps(ymm0, mem(rbx, (0*12+0)*4)) \
	vmovaps(xmm1, mem(rbx, (0*12+8)*4)) \
	vmovups(ymm2, mem(rbx, (1*12+0)*4)) \
	vmovaps(xmm3, mem(rbx, (1*12+8)*4)) \
	vmovaps(ymm4, mem(rbx, (2*12+0)*4)) \
	vmovaps(xmm5, mem(rbx, (2*12+8)*4)) \
	vmovups(ymm6, mem(rbx, (3*12+0)*4)) \
	vmovaps(xmm7, mem(rbx, (3*12+8)*4))
#define STORE_8x12xFP32() \
	vmovaps(ymm0, mem(rbx, (0*12+0)*4)) \
	vmovaps(xmm1, mem(rbx, (0*12+8)*4)) \
	vmovups(ymm2, mem(rbx, (1*12+0)*4)) \
	vmovaps(xmm3, mem(rbx, (1*12+8)*4)) \
	vmovaps(ymm4, mem(rbx, (2*12+0)*4)) \
	vmovaps(xmm5, mem(rbx, (2*12+8)*4)) \
	vmovups(ymm6, mem(rbx, (3*12+0)*4)) \
	vmovaps(xmm7, mem(rbx, (3*12+8)*4)) \
	vmovaps(ymm8, mem(rbx, (4*12+0)*4)) \
	vmovaps(xmm9, mem(rbx, (4*12+8)*4)) \
	vmovups(ymm10, mem(rbx, (5*12+0)*4)) \
	vmovaps(xmm11, mem(rbx, (5*12+8)*4)) \
	vmovaps(ymm12, mem(rbx, (6*12+0)*4)) \
	vmovaps(xmm13, mem(rbx, (6*12+8)*4)) \
	vmovups(ymm14, mem(rbx, (7*12+0)*4)) \
	vmovaps(xmm15, mem(rbx, (7*12+8)*4))
#define STORE_8x8xFP32() \
	vmovaps(ymm8, mem(rbx, (0*12+0)*4)) \
	vmovups(ymm9, mem(rbx, (1*12+0)*4)) \
	vmovaps(ymm10, mem(rbx, (2*12+0)*4)) \
	vmovups(ymm11, mem(rbx, (3*12+0)*4)) \
	vmovaps(ymm12, mem(rbx, (4*12+0)*4)) \
	vmovups(ymm13, mem(rbx, (5*12+0)*4)) \
	vmovaps(ymm14, mem(rbx, (6*12+0)*4)) \
	vmovups(ymm15, mem(rbx, (7*12+0)*4))
#define STORE_8x4xFP32() \
	vmovaps(xmm4, mem(rbx, (0*12+8)*4)) \
	vmovaps(xmm5, mem(rbx, (1*12+8)*4)) \
	vmovaps(xmm6, mem(rbx, (2*12+8)*4)) \
	vmovaps(xmm7, mem(rbx, (3*12+8)*4)) \
	vmovaps(xmm0, mem(rbx, (4*12+8)*4)) \
	vmovaps(xmm1, mem(rbx, (5*12+8)*4)) \
	vmovaps(xmm2, mem(rbx, (6*12+8)*4)) \
	vmovaps(xmm3, mem(rbx, (7*12+8)*4))
#define STORE_12x1xFP32() \
	vmovss(xmm0, mem(rbx, 4*0)) \
	vmovss(xmm1, mem(rbx, 4*1)) \
	vmovss(xmm2, mem(rbx, 4*2)) \
	vmovss(xmm3, mem(rbx, 4*3)) \
	vmovss(xmm4, mem(rbx, 4*4)) \
	vmovss(xmm5, mem(rbx, 4*5)) \
	vmovss(xmm6, mem(rbx, 4*6)) \
	vmovss(xmm7, mem(rbx, 4*7)) \
	vmovss(xmm8, mem(rbx, 4*8)) \
	vmovss(xmm9, mem(rbx, 4*9)) \
	vmovss(xmm10, mem(rbx, 4*10)) \
	vmovss(xmm11, mem(rbx, 4*11))
#define STORE_16x1xFP32() \
	vmovss(xmm0, mem(rbx, 4*0)) \
	vmovss(xmm1, mem(rbx, 4*1)) \
	vmovss(xmm2, mem(rbx, 4*2)) \
	vmovss(xmm3, mem(rbx, 4*3)) \
	vmovss(xmm4, mem(rbx, 4*4)) \
	vmovss(xmm5, mem(rbx, 4*5)) \
	vmovss(xmm6, mem(rbx, 4*6)) \
	vmovss(xmm7, mem(rbx, 4*7)) \
	vmovss(xmm8, mem(rbx, 4*8)) \
	vmovss(xmm9, mem(rbx, 4*9)) \
	vmovss(xmm10, mem(rbx, 4*10)) \
	vmovss(xmm11, mem(rbx, 4*11)) \
	vmovss(xmm12, mem(rbx, 4*12)) \
	vmovss(xmm13, mem(rbx, 4*13)) \
	vmovss(xmm14, mem(rbx, 4*14)) \
	vmovss(xmm15, mem(rbx, 4*15))
#define STORE_4x16xFP32() \
	vmovaps(ymm0, mem(rbx, (0*16+0)*4)) \
	vmovaps(ymm1, mem(rbx, (0*16+8)*4)) \
	vmovups(ymm2, mem(rbx, (1*16+0)*4)) \
	vmovaps(ymm3, mem(rbx, (1*16+8)*4)) \
	vmovaps(ymm4, mem(rbx, (2*16+0)*4)) \
	vmovaps(ymm5, mem(rbx, (2*16+8)*4)) \
	vmovups(ymm6, mem(rbx, (3*16+0)*4)) \
	vmovaps(ymm7, mem(rbx, (3*16+8)*4))
#define STORE_8x6xFP32() \
	vmovups(xmm0, mem(rbx, (0*6+0)*4)) \
	vmovsd (xmm1, mem(rbx, (0*6+4)*4)) \
	vmovups(xmm2, mem(rbx, (1*6+0)*4)) \
	vmovsd (xmm3, mem(rbx, (1*6+4)*4)) \
	vmovups(xmm4, mem(rbx, (2*6+0)*4)) \
	vmovsd (xmm5, mem(rbx, (2*6+4)*4)) \
	vmovups(xmm6, mem(rbx, (3*6+0)*4)) \
	vmovsd (xmm7, mem(rbx, (3*6+4)*4)) \
	vmovups(xmm8, mem(rbx, (4*6+0)*4)) \
	vmovsd (xmm9, mem(rbx, (4*6+4)*4)) \
	vmovups(xmm10, mem(rbx, (5*6+0)*4)) \
	vmovsd (xmm11, mem(rbx, (5*6+4)*4)) \
	vmovups(xmm12, mem(rbx, (6*6+0)*4)) \
	vmovsd (xmm13, mem(rbx, (6*6+4)*4)) \
	vmovups(xmm14, mem(rbx, (7*6+0)*4)) \
	vmovsd (xmm15, mem(rbx, (7*6+4)*4))

/*
 * MHA helper macros
 */
#define ADD_BIAS_4x8xFP32(reg0, reg1, reg2, reg3) \
	vmovaps(mem(rbx), ymm0) \
	vmovaps(mem(rbx, r14, 1), ymm1) \
	vmovaps(mem(rbx, r14, 2), ymm2) \
	vmovaps(mem(rbx, r13, 1), ymm3) \
	add(r15, rbx) \
	vaddps(ymm0, reg0, reg0) \
	vaddps(ymm1, reg1, reg1) \
	vaddps(ymm2, reg2, reg2) \
	vaddps(ymm3, reg3, reg3)
#define EXP_FIRST_STAGE_12x8xFP32() \
	movq(imm(0x4ab8aa3b4ab8aa3b), r14) \
	vmovq(r14, xmm0) \
	vpermpd(imm(0), ymm0, ymm0) \
	vmulps(ymm0, ymm4, ymm4) \
	vmulps(ymm0, ymm5, ymm5) \
	vmulps(ymm0, ymm6, ymm6) \
	vmulps(ymm0, ymm7, ymm7) \
	vmulps(ymm0, ymm8, ymm8) \
	vmulps(ymm0, ymm9, ymm9) \
	vmulps(ymm0, ymm10, ymm10) \
	vmulps(ymm0, ymm11, ymm11) \
	vmulps(ymm0, ymm12, ymm12) \
	vmulps(ymm0, ymm13, ymm13) \
	vmulps(ymm0, ymm14, ymm14) \
	vmulps(ymm0, ymm15, ymm15) \
	vcvtps2dq(ymm4, ymm4) \
	vcvtps2dq(ymm5, ymm5) \
	vcvtps2dq(ymm6, ymm6) \
	vcvtps2dq(ymm7, ymm7) \
	vcvtps2dq(ymm8, ymm8) \
	vcvtps2dq(ymm9, ymm9) \
	vcvtps2dq(ymm10, ymm10) \
	vcvtps2dq(ymm11, ymm11) \
	vcvtps2dq(ymm12, ymm12) \
	vcvtps2dq(ymm13, ymm13) \
	vcvtps2dq(ymm14, ymm14) \
	vcvtps2dq(ymm15, ymm15) \
	movq(imm(0x3f7de0683f7de068), r14) \
	vmovq(r14, xmm0) \
	vpermpd(imm(0), ymm0, ymm0)
#define EXP_SECOND_STAGE_3x8xFP32(reg0, reg1, reg2) \
	vpsubd(reg0, ymm0, ymm1) \
	vpsubd(reg1, ymm0, ymm2) \
	vpsubd(reg2, ymm0, ymm3) \
	vrcpps(ymm1, ymm1) \
	vrcpps(ymm2, ymm2) \
	vrcpps(ymm3, ymm3) \
	vpaddd(reg0, ymm0, reg0) \
	vpaddd(reg1, ymm0, reg1) \
	vpaddd(reg2, ymm0, reg2) \
	vmulps(ymm1, reg0, reg0) \
	vmulps(ymm2, reg1, reg1) \
	vmulps(ymm3, reg2, reg2)
#define REDUCE_SUM() \
	vaddps(ymm4, ymm0, ymm0) \
	vaddps(ymm5, ymm1, ymm1) \
	vaddps(ymm6, ymm2, ymm2) \
	vaddps(ymm7, ymm3, ymm3) \
	vaddps(ymm1, ymm0, ymm0) \
	vaddps(ymm3, ymm2, ymm2) \
	vaddps(ymm2, ymm0, ymm0)

/*
 * Depthwise conv helper
 */
#define SUB_KERNEL_GEMV(n) \
	vmovaps(mem(rbx, n*8*4), ymm0)\
	vmovaps(mem(rax, (12*n+0)*8*4), ymm1)\
	vmovaps(mem(rax, (12*n+1)*8*4), ymm2)\
	vfmadd231ps(ymm0, ymm1, ymm4)\
	vfmadd231ps(ymm0, ymm2, ymm5)\
	vmovaps(mem(rax, (12*n+2)*8*4), ymm3)\
	vmovaps(mem(rax, (12*n+3)*8*4), ymm1)\
	vfmadd231ps(ymm0, ymm3, ymm6)\
	vfmadd231ps(ymm0, ymm1, ymm7)\
	vmovaps(mem(rax, (12*n+4)*8*4), ymm2)\
	vmovaps(mem(rax, (12*n+5)*8*4), ymm3)\
	vfmadd231ps(ymm0, ymm2, ymm8)\
	vfmadd231ps(ymm0, ymm3, ymm9)\
	vmovaps(mem(rax, (12*n+6)*8*4), ymm1)\
	vmovaps(mem(rax, (12*n+7)*8*4), ymm2)\
	vfmadd231ps(ymm0, ymm1, ymm10)\
	vfmadd231ps(ymm0, ymm2, ymm11)\
	vmovaps(mem(rax, (12*n+8)*8*4), ymm3)\
	vmovaps(mem(rax, (12*n+9)*8*4), ymm1)\
	vfmadd231ps(ymm0, ymm3, ymm12)\
	vfmadd231ps(ymm0, ymm1, ymm13)\
	vmovaps(mem(rax, (12*n+10)*8*4), ymm2)\
	vmovaps(mem(rax, (12*n+11)*8*4), ymm3)\
	vfmadd231ps(ymm0, ymm2, ymm14)\
	vfmadd231ps(ymm0, ymm3, ymm15)

namespace ml
{
	using namespace ml::cpu;

	void gemm_avx2_12x8(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(C.is_fp32() || C.is_fp16());
		assert(D.is_fp32() || D.is_fp16());
		assert(A.rows() == B.rows());
		assert(A.stride() == 12);
		assert(B.stride() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha.is_packed());
		assert(alpha.is_fp32());
		assert(cpu::is_aligned(A.data(), 32));
		assert(cpu::is_aligned(B.data(), 32));
		assert(beta_ptr != nullptr);
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 32));
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
		SUB_KERNEL_12xFP32_8xFP32(0)
		SUB_KERNEL_12xFP32_8xFP32(1)
		SUB_KERNEL_12xFP32_8xFP32(2)
		SUB_KERNEL_12xFP32_8xFP32(3)

		add(imm(4*12*4), rax)// 4 iterations x 12 elements x 4 bytes
		add(imm(4*8*4), rbx)// 4 iterations x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_12xFP32_8xFP32(0)
		add(imm(1*12*4), rax)// 1 iteration x 12 elements x 4 bytes
		add(imm(1*8*4), rbx)// 1 iteration x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)
		// permute back to original layout
		PERMUTE_6x8xFP32(ymm4, ymm5, ymm6, ymm7, ymm8, ymm9)
		PERMUTE_6x8xFP32(ymm10, ymm11, ymm12, ymm13, ymm14, ymm15)

		// now perform scaling by alpha
		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(scalar_alpha), r14)
		test(r14, r14)
		je(COLUMN_ALPHA)
		vbroadcastss(mem(rax), ymm0)
		SCALE_ACCUMULATORS_1xN(ymm0)
		jmp(AFTER_ALPHA_SCALING)

		label(COLUMN_ALPHA)
		SCALE_ACCUMULATORS_12x1()
		label(AFTER_ALPHA_SCALING)

		// now perform bias addition
		// load address of bias pointer
		movq(var(bias_ptr), rax)
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax), ymm2)// load bias
		ADD_BIAS_12x8xFP32(ymm2)
		label(AFTER_BIAS)

		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rbx), ymm0)
		vxorps(ymm1, ymm1, ymm1)
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
		LOAD_ADD_3x8xFP16(ymm0, ymm4, ymm5, ymm6)
		LOAD_ADD_3x8xFP16(ymm0, ymm7, ymm8, ymm9)
		LOAD_ADD_3x8xFP16(ymm0, ymm10, ymm11, ymm12)
		LOAD_ADD_3x8xFP16(ymm0, ymm13, ymm14, ymm15)
		jmp(AFTER_LOAD_C)

		label(C_IN_FP32)
		LOAD_ADD_3x8xFP32(ymm0, ymm4, ymm5, ymm6)
		LOAD_ADD_3x8xFP32(ymm0, ymm7, ymm8, ymm9)
		LOAD_ADD_3x8xFP32(ymm0, ymm10, ymm11, ymm12)
		LOAD_ADD_3x8xFP32(ymm0, ymm13, ymm14, ymm15)
		label(AFTER_LOAD_C)

		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(AFTER_RELU)
		RELU_12x8xFP32()
		label(AFTER_RELU)

		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r13)// r13 = r14
		sal(imm(1), r13)// r13 = 2 * r14 (2 * stride)
		add(r14, r13)// r13 = 2 * r14 + r14 = 3 * r14 (3*D_stride)
		movq(r14, r15)// r15 = r14
		sal(imm(2), r15)// r15 = 4 * r14 (4 * stride)

		movq(var(cd_in_fp16), r11)// load fp16 flags
		and_(imm(0x2), r11)// if set
		test(r11, r11)
		je(D_IN_FP32)
		CONVERT_ACCUMULATORS_TO_FP16()
		STORE_4x8xFP16(xmm4, xmm5, xmm6, xmm7)
		STORE_4x8xFP16(xmm8, xmm9, xmm10, xmm11)
		STORE_4x8xFP16(xmm12, xmm13, xmm14, xmm15)
		jmp(END)

		label(D_IN_FP32)
		STORE_4x8xFP32(ymm4, ymm5, ymm6, ymm7)
		STORE_4x8xFP32(ymm8, ymm9, ymm10, ymm11)
		STORE_4x8xFP32(ymm12, ymm13, ymm14, ymm15)

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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r11", "%r13", "%r14", "%r15")
	}
	void gemm_avx2_6x16(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(C.is_fp32() || C.is_fp16());
		assert(D.is_fp32() || D.is_fp16());
		assert(A.rows() == B.rows());
		assert(A.stride() == 6);
		assert(B.stride() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha.is_packed());
		assert(alpha.is_fp32());
		assert(cpu::is_aligned(A.data(), 32));
		assert(cpu::is_aligned(B.data(), 32));
		assert(beta_ptr != nullptr);
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 32));
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
		SUB_KERNEL_6xFP32_16xFP32(0)
		SUB_KERNEL_6xFP32_16xFP32(1)
		SUB_KERNEL_6xFP32_16xFP32(2)
		SUB_KERNEL_6xFP32_16xFP32(3)

		add(imm(4*6*4), rax)// 4 iterations x 6 elements x 4 bytes
		add(imm(4*16*4), rbx)// 4 iterations x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_6xFP32_16xFP32(0)
		add(imm(1*6*4), rax)// 1 iteration x 6 elements x 4 bytes
		add(imm(1*16*4), rbx)// 1 iteration x 16 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(scalar_alpha), r14)
		test(r14, r14)
		je(COLUMN_ALPHA)
		vbroadcastss(mem(rax), ymm0)
		SCALE_ACCUMULATORS_1xN(ymm0)
		jmp(AFTER_ALPHA_SCALING)

		label(COLUMN_ALPHA)
		SCALE_ACCUMULATORS_6x1()
		label(AFTER_ALPHA_SCALING)

		// load address of bias pointer
		movq(var(bias_ptr), rax)
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax, 0*8*4), ymm2)// load bias
		vmovaps(mem(rax, 1*8*4), ymm3)// load bias
		ADD_BIAS_6x16xFP32(ymm2, ymm3)
		label(AFTER_BIAS)

		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rbx), ymm0)
		vxorps(ymm1, ymm1, ymm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(AFTER_LOAD_C)
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx

		movq(var(cd_in_fp16), r11)// load fp16 flags
		and_(imm(0x1), r11)// if set
		test(r11, r11)
		je(C_IN_FP32)
		add(r15, rcx)
		LOAD_ADD_1x16xFP16(ymm0, ymm4, ymm5)
		LOAD_ADD_1x16xFP16(ymm0, ymm6, ymm7)
		LOAD_ADD_1x16xFP16(ymm0, ymm8, ymm9)
		LOAD_ADD_1x16xFP16(ymm0, ymm10, ymm11)
		LOAD_ADD_1x16xFP16(ymm0, ymm12, ymm13)
		LOAD_ADD_1x16xFP16(ymm0, ymm14, ymm15)
		jmp(AFTER_LOAD_C)

		label(C_IN_FP32)
		LOAD_ADD_1x16xFP32(ymm0, ymm4, ymm5)
		LOAD_ADD_1x16xFP32(ymm0, ymm6, ymm7)
		LOAD_ADD_1x16xFP32(ymm0, ymm8, ymm9)
		LOAD_ADD_1x16xFP32(ymm0, ymm10, ymm11)
		LOAD_ADD_1x16xFP32(ymm0, ymm12, ymm13)
		LOAD_ADD_1x16xFP32(ymm0, ymm14, ymm15)
		label(AFTER_LOAD_C)

		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(AFTER_RELU)
		RELU_12x8xFP32()
		label(AFTER_RELU)

		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r15)// r13 = r14
		sal(imm(1), r15)// r13 = 2 * r14 (2 * stride)

		movq(var(cd_in_fp16), r11)// load fp16 flags
		and_(imm(0x2), r11)// if set
		test(r11, r11)
		je(D_IN_FP32)
		CONVERT_ACCUMULATORS_TO_FP16()
		STORE_2x16xFP16(xmm4, xmm5, xmm6, xmm7)
		STORE_2x16xFP16(xmm8, xmm9, xmm10, xmm11)
		STORE_2x16xFP16(xmm12, xmm13, xmm14, xmm15)
		jmp(END)

		label(D_IN_FP32)
		STORE_2x16xFP32(ymm4, ymm5, ymm6, ymm7)
		STORE_2x16xFP32(ymm8, ymm9, ymm10, ymm11)
		STORE_2x16xFP32(ymm12, ymm13, ymm14, ymm15)

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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r11", "%r13", "%r14", "%r15")
	}

	void pack_avx2_12xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		if (dst.is_partial())
		{
			pack_def_MxK(dst, src, src_pos, src_op);
			return;
		}

		assert(src.is_fp32() || src.is_fp16());
		assert(dst.is_fp32());
		assert(dst.stride() == 12);
		assert(ml::cpu::is_aligned(dst.data(), 32));

		const uint64_t src_stride = src.stride() * size_of(src.dtype());
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();
		const uint64_t convert_fp16 = (src.dtype() == DTYPE_FLOAT16);

		if (src_op == MatrixOp::NORMAL)
		{
			uint64_t k_iter = dst.rows() / 4;
			uint64_t k_left = dst.rows() % 4;
			if (src.dtype() == DTYPE_FLOAT32)
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

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(EDGELOOP)

				label(UNROLLED8)// starting fp32 loop
				LOAD_4x12xFP32()
				STORE_4x12xFP32()
				add(imm(4*4*12), rbx)// add stride to dst pointer
				dec(r14)
				jne(UNROLLED8)

				label(EDGELOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)// starting fp32 loop
				vmovups(mem(rax), ymm0)
				vmovups(mem(rax, 8*4), xmm1)
				add(r12, rax)// add stride to src pointer
				vmovups(ymm0, mem(rbx))
				vmovaps(xmm1, mem(rbx, 8*4))
				add(imm(4*1*12), rbx)// add stride to dst pointer
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
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				begin_asm()
				movq(var(convert_fp16), r11) // store conversion flag in r11
				movq(var(src_ptr), rax)// src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(EDGELOOP)

				label(UNROLLED8)// starting fp16 loop
				LOAD_4x12xFP16()
				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, xmm1)
				vcvtph2ps(xmm2, ymm2)
				vcvtph2ps(xmm3, xmm3)
				vcvtph2ps(xmm4, ymm4)
				vcvtph2ps(xmm5, xmm5)
				vcvtph2ps(xmm6, ymm6)
				vcvtph2ps(xmm7, xmm7)
				STORE_4x12xFP32()
				add(imm(4*4*12), rbx)// add stride to dst pointer
				dec(r14)
				jne(UNROLLED8)

				label(EDGELOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)// starting fp16 loop
				vmovups(mem(rax), xmm0)
				vmovsd (mem(rax, 8*2), xmm1)
				add(r12, rax)// add stride to src pointer
				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, xmm1)
				vmovups(ymm0, mem(rbx))
				vmovaps(xmm1, mem(rbx, 8*4))
				add(imm(4*1*12), rbx)// add stride to dst pointer
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
						[src_stride] "m"(src_stride),
						[convert_fp16] "m"(convert_fp16)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%r12", "%r13", "%r14", "%r15")
			}
		}
		else
		{
			uint64_t k_iter = dst.rows() / 8;
			uint64_t k_left = dst.rows() % 8;

			if (src.dtype() == DTYPE_FLOAT32)
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

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				movq(rax, rcx)// tmp src pointer is in rcx
				LOAD_8x8xFP32()
				AVX_8x8_TRANSPOSE()
				STORE_8x8xFP32()
				LOAD_4x8xFP32()
				AVX_4x8_TRANSPOSE()
				STORE_8x4xFP32()
				add(imm(4*8), rax)// add stride to src pointer
				add(imm(4*12*8), rbx)// add stride to dst pointer
				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)// tmp src pointer is in rcx
				LOAD_12x1xFP32()
				STORE_12x1xFP32()
				add(imm(4*1), rax)// add stride to src pointer
				add(imm(4*12*1), rbx)// add stride to dst pointer
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
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
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

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				// first 8x8 tile
				movq(rax, rcx)// tmp src pointer is in rcx
				LOAD_8x8xFP16()
				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, ymm1)
				vcvtph2ps(xmm2, ymm2)
				vcvtph2ps(xmm3, ymm3)
				vcvtph2ps(xmm4, ymm4)
				vcvtph2ps(xmm5, ymm5)
				vcvtph2ps(xmm6, ymm6)
				vcvtph2ps(xmm7, ymm7)
				AVX_8x8_TRANSPOSE()
				STORE_8x8xFP32()
				LOAD_4x8xFP16()
				vcvtph2ps(xmm4, ymm4)
				vcvtph2ps(xmm5, ymm5)
				vcvtph2ps(xmm6, ymm6)
				vcvtph2ps(xmm7, ymm7)
				AVX_4x8_TRANSPOSE()
				STORE_8x4xFP32()
				add(imm(8*2), rax)// add stride to src pointer
				add(imm(12*8*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)// tmp src pointer is in rcx

				movzw(mem(rcx), r8)
				movzw(mem(rcx, r12, 1), r9)
				movzw(mem(rcx, r12, 2), r10)
				movzw(mem(rcx, r13, 1), r11)
				vmovq(r8, xmm0)
				vmovq(r9, xmm1)
				vmovq(r10, xmm2)
				vmovq(r11, xmm3)
				add(r15, rcx)
				movzw(mem(rcx), r8)
				movzw(mem(rcx, r12, 1), r9)
				movzw(mem(rcx, r12, 2), r10)
				movzw(mem(rcx, r13, 1), r11)
				vmovq(r8, xmm4)
				vmovq(r9, xmm5)
				vmovq(r10, xmm6)
				vmovq(r11, xmm7)
				add(r15, rcx)
				movzw(mem(rcx), r8)
				movzw(mem(rcx, r12, 1), r9)
				movzw(mem(rcx, r12, 2), r10)
				movzw(mem(rcx, r13, 1), r11)
				vmovq(r8, xmm8)
				vmovq(r9, xmm9)
				vmovq(r10, xmm10)
				vmovq(r11, xmm11)

				vcvtph2ps(xmm0, xmm0)
				vcvtph2ps(xmm1, xmm1)
				vcvtph2ps(xmm2, xmm2)
				vcvtph2ps(xmm3, xmm3)
				vcvtph2ps(xmm4, xmm4)
				vcvtph2ps(xmm5, xmm5)
				vcvtph2ps(xmm6, xmm6)
				vcvtph2ps(xmm7, xmm7)
				vcvtph2ps(xmm8, xmm8)
				vcvtph2ps(xmm9, xmm9)
				vcvtph2ps(xmm10, xmm10)
				vcvtph2ps(xmm11, xmm11)

				STORE_12x1xFP32()
				add(imm(1*2), rax)// add stride to src pointer
				add(imm(12*1*4), rbx)// add stride to dst pointer

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
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
						"%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
			}
		}
	}

	void pack_avx2_6xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		if (dst.is_partial())
		{
			pack_def_MxK(dst, src, src_pos, src_op);
			return;
		}
		assert(src.is_fp32() || src.is_fp16());
		assert(dst.is_fp32());
		assert(dst.stride() == 6);
		assert(ml::cpu::is_aligned(dst.data(), 32));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = dst.rows() % 8;
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

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				vmovups(mem(rax), xmm0)
				vmovsd (mem(rax, 4*4), xmm1)
				vmovups(mem(rax, r12, 1), xmm2)
				vmovsd (mem(rax, r12, 1, 4*4), xmm3)
				vmovups(mem(rax, r12, 2), xmm4)
				vmovsd (mem(rax, r12, 2, 4*4), xmm5)
				vmovups(mem(rax, r13, 1), xmm6)
				vmovsd (mem(rax, r13, 1, 4*4), xmm7)
				add(r15, rax)
				vmovups(mem(rax), xmm8)
				vmovsd (mem(rax, 4*4), xmm9)
				vmovups(mem(rax, r12, 1), xmm10)
				vmovsd (mem(rax, r12, 1, 4*4), xmm11)
				vmovups(mem(rax, r12, 2), xmm12)
				vmovsd (mem(rax, r12, 2, 4*4), xmm13)
				vmovups(mem(rax, r13, 1), xmm14)
				vmovsd (mem(rax, r13, 1, 4*4), xmm15)
				add(r15, rax)

				STORE_8x6xFP32()

				add(imm(8*6*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				vmovups(mem(rax), xmm0)
				vmovsd (mem(rax, 4*4), xmm1)
				add(r12, rax)
				vmovups(xmm0, mem(rbx))
				vmovsd (xmm1, mem(rbx, 4*4))
				add(imm(1*6*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(
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

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				vmovsd(mem(rax), xmm0)
				vmovss(mem(rax, 4*2), xmm1)
				vmovsd(mem(rax, r12, 1), xmm2)
				vmovss(mem(rax, r12, 1, 4*2), xmm3)
				vmovsd(mem(rax, r12, 2), xmm4)
				vmovss(mem(rax, r12, 2, 4*2), xmm5)
				vmovsd(mem(rax, r13, 1), xmm6)
				vmovss(mem(rax, r13, 1, 4*2), xmm7)
				add(r15, rax)
				vmovsd(mem(rax), xmm8)
				vmovss(mem(rax, 4*2), xmm9)
				vmovsd(mem(rax, r12, 1), xmm10)
				vmovss(mem(rax, r12, 1, 4*2), xmm11)
				vmovsd(mem(rax, r12, 2), xmm12)
				vmovss(mem(rax, r12, 2, 4*2), xmm13)
				vmovsd(mem(rax, r13, 1), xmm14)
				vmovss(mem(rax, r13, 1, 4*2), xmm15)
				add(r15, rax)

				vcvtph2ps(xmm0, xmm0)
				vcvtph2ps(xmm1, xmm1)
				vcvtph2ps(xmm2, xmm2)
				vcvtph2ps(xmm3, xmm3)
				vcvtph2ps(xmm4, xmm4)
				vcvtph2ps(xmm5, xmm5)
				vcvtph2ps(xmm6, xmm6)
				vcvtph2ps(xmm7, xmm7)
				vcvtph2ps(xmm8, xmm8)
				vcvtph2ps(xmm9, xmm9)
				vcvtph2ps(xmm10, xmm10)
				vcvtph2ps(xmm11, xmm11)
				vcvtph2ps(xmm12, xmm12)
				vcvtph2ps(xmm13, xmm13)
				vcvtph2ps(xmm14, xmm14)
				vcvtph2ps(xmm15, xmm15)
				STORE_8x6xFP32()

				add(imm(4*8*6), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				vmovsd(mem(rax), xmm0)
				vmovss(mem(rax, 4*2), xmm1)
				add(r12, rax)
				vcvtph2ps(xmm0, xmm0)
				vcvtph2ps(xmm1, xmm1)
				vmovups(xmm0, mem(rbx))
				vmovsd (xmm1, mem(rbx, 4*4))
				add(imm(4*1*6), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(
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
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				// load 4x8 tile
				movq(rax, rcx)// tmp src pointer is in rcx
				vmovups(mem(rcx), ymm4)
				vmovups(mem(rcx, r12, 1), ymm5)
				vmovups(mem(rcx, r12, 2), ymm6)
				vmovups(mem(rcx, r13, 1), ymm7)
				add(r15, rcx)
				AVX_4x8_TRANSPOSE()
				vmovups(xmm4, mem(rbx, (0*6+0)*4))
				vmovups(xmm5, mem(rbx, (1*6+0)*4))
				vmovups(xmm6, mem(rbx, (2*6+0)*4))
				vmovups(xmm7, mem(rbx, (3*6+0)*4))
				vmovups(xmm0, mem(rbx, (4*6+0)*4))
				vmovups(xmm1, mem(rbx, (5*6+0)*4))
				vmovups(xmm2, mem(rbx, (6*6+0)*4))
				vmovups(xmm3, mem(rbx, (7*6+0)*4))

				// rows 4-5 (2x8 tile)
				vmovups(mem(rcx), ymm0)
				vmovups(mem(rcx, r12, 1), ymm1)

				vunpcklps(ymm1, ymm0, ymm4)
				vunpckhps(ymm1, ymm0, ymm5)

				vextractf128(imm(0x1), ymm4, xmm6)// e4 f4 e5 f5
				vextractf128(imm(0x1), ymm5, xmm7)// e6 f6 e7 f7

				vmovlpd(xmm4, mem(rbx, 4*(0*6+4)))
				vmovhpd(xmm4, mem(rbx, 4*(1*6+4)))
				vmovlpd(xmm5, mem(rbx, 4*(2*6+4)))
				vmovhpd(xmm5, mem(rbx, 4*(3*6+4)))
				vmovlpd(xmm6, mem(rbx, 4*(4*6+4)))
				vmovhpd(xmm6, mem(rbx, 4*(5*6+4)))
				vmovlpd(xmm7, mem(rbx, 4*(6*6+4)))
				vmovhpd(xmm7, mem(rbx, 4*(7*6+4)))

				add(imm(4*8), rax)// add stride to src pointer
				add(imm(4*8*6), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)// tmp src pointer is in rcx
				vmovss(mem(rcx), xmm0)
				vmovss(mem(rcx, r12, 1), xmm1)
				vmovss(mem(rcx, r12, 2), xmm2)
				vmovss(mem(rcx, r13, 1), xmm3)
				add(r15, rcx)
				vmovss(mem(rcx), xmm4)
				vmovss(mem(rcx, r12, 1), xmm5)

				vmovss(xmm0, mem(rbx, 0*4))
				vmovss(xmm1, mem(rbx, 1*4))
				vmovss(xmm2, mem(rbx, 2*4))
				vmovss(xmm3, mem(rbx, 3*4))
				vmovss(xmm4, mem(rbx, 4*4))
				vmovss(xmm5, mem(rbx, 5*4))

				add(imm(4*1), rax)// add stride to src pointer
				add(imm(4*6*1), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(
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
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				// load 4x8 tile
				movq(rax, rcx)// tmp src pointer is in rcx
				vmovups(mem(rcx), xmm4)
				vmovups(mem(rcx, r12, 1), xmm5)
				vmovups(mem(rcx, r12, 2), xmm6)
				vmovups(mem(rcx, r13, 1), xmm7)
				add(r15, rcx)
				vcvtph2ps(xmm4, ymm4)
				vcvtph2ps(xmm5, ymm5)
				vcvtph2ps(xmm6, ymm6)
				vcvtph2ps(xmm7, ymm7)
				AVX_4x8_TRANSPOSE()
				vmovups(xmm4, mem(rbx, (0*6+0)*4))
				vmovups(xmm5, mem(rbx, (1*6+0)*4))
				vmovups(xmm6, mem(rbx, (2*6+0)*4))
				vmovups(xmm7, mem(rbx, (3*6+0)*4))
				vmovups(xmm0, mem(rbx, (4*6+0)*4))
				vmovups(xmm1, mem(rbx, (5*6+0)*4))
				vmovups(xmm2, mem(rbx, (6*6+0)*4))
				vmovups(xmm3, mem(rbx, (7*6+0)*4))

				// rows 4-5 (2x8 tile)
				vmovups(mem(rcx), xmm0)
				vmovups(mem(rcx, r12, 1), xmm1)
				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, ymm1)

				vunpcklps(ymm1, ymm0, ymm4)
				vunpckhps(ymm1, ymm0, ymm5)

				vextractf128(imm(0x1), ymm4, xmm6)// e4 f4 e5 f5
				vextractf128(imm(0x1), ymm5, xmm7)// e6 f6 e7 f7

				vmovlpd(xmm4, mem(rbx, 4*(0*6+4)))
				vmovhpd(xmm4, mem(rbx, 4*(1*6+4)))
				vmovlpd(xmm5, mem(rbx, 4*(2*6+4)))
				vmovhpd(xmm5, mem(rbx, 4*(3*6+4)))
				vmovlpd(xmm6, mem(rbx, 4*(4*6+4)))
				vmovhpd(xmm6, mem(rbx, 4*(5*6+4)))
				vmovlpd(xmm7, mem(rbx, 4*(6*6+4)))
				vmovhpd(xmm7, mem(rbx, 4*(7*6+4)))

				add(imm(2*8), rax)// add stride to src pointer
				add(imm(4*8*6), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)// tmp src pointer is in rcx
				movzw(mem(rcx), r8)
				movzw(mem(rcx, r12, 1), r9)
				movzw(mem(rcx, r12, 2), r10)
				movzw(mem(rcx, r13, 1), r11)
				vmovq(r8, xmm0)
				vmovq(r9, xmm1)
				vmovq(r10, xmm2)
				vmovq(r11, xmm3)
				add(r15, rcx)
				movzw(mem(rcx), r8)
				movzw(mem(rcx, r12, 1), r9)
				vmovq(r8, xmm4)
				vmovq(r9, xmm5)

				vcvtph2ps(xmm0, xmm0)
				vcvtph2ps(xmm1, xmm1)
				vcvtph2ps(xmm2, xmm2)
				vcvtph2ps(xmm3, xmm3)
				vcvtph2ps(xmm4, xmm4)
				vcvtph2ps(xmm5, xmm5)

				vmovss(xmm0, mem(rbx, 0*4))
				vmovss(xmm1, mem(rbx, 1*4))
				vmovss(xmm2, mem(rbx, 2*4))
				vmovss(xmm3, mem(rbx, 3*4))
				vmovss(xmm4, mem(rbx, 4*4))
				vmovss(xmm5, mem(rbx, 5*4))

				add(imm(2*1), rax)// add stride to src pointer
				add(imm(4*6*1), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED1)

				label(EPILOGUE)
				vzeroupper()

				end_asm(
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
						"%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
			}
		}
	}
	void pack_avx2_16xK(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		if (dst.is_partial())
		{
			pack_def_MxK(dst, src, src_pos, src_op);
			return;
		}

		assert(src.is_fp32() || src.is_fp16());
		assert(dst.is_fp32());
		assert(dst.stride() == 16);
		assert(ml::cpu::is_aligned(dst.data(), 32));

		const uint64_t src_stride = src.stride() * size_of(src.dtype());
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();
		const uint64_t convert_fp16 = (src.dtype() == DTYPE_FLOAT16);

		if (src_op == MatrixOp::NORMAL)
		{
			uint64_t k_iter = dst.rows() / 4;
			uint64_t k_left = dst.rows() % 4;
			if (src.dtype() == DTYPE_FLOAT32)
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

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(EDGELOOP)

				label(UNROLLED8)// starting fp32 loop
				LOAD_4x16xFP32()
				STORE_4x16xFP32()
				add(imm(4*4*16), rbx)// add stride to dst pointer
				dec(r14)
				jne(UNROLLED8)

				label(EDGELOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)// starting fp32 loop
				vmovups(mem(rax), ymm0)
				vmovups(mem(rax, 8*4), ymm1)
				add(r12, rax)// add stride to src pointer
				vmovups(ymm0, mem(rbx))
				vmovaps(ymm1, mem(rbx, 8*4))
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
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
						"%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				begin_asm()
				movq(var(convert_fp16), r11) // store conversion flag in r11
				movq(var(src_ptr), rax)// src pointer is in rax
				movq(var(dst_ptr), rbx)// dst pointer is in rbx
				movq(var(src_stride), r12)// src stride is in r12
				movq(r12, r13)// r13 = r12
				sal(imm(1), r13)// r13 = 2 * r12 (2 * stride)
				add(r12, r13)// r13 = 2 * r12 + r12 = 3 * r12 (3*D_stride)
				movq(r12, r15)// r15 = r12
				sal(imm(2), r15)// r15 = 4 * r12 (4 * stride)

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(EDGELOOP)

				label(UNROLLED8)// starting fp16 loop
				LOAD_4x16xFP16()
				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, ymm1)
				vcvtph2ps(xmm2, ymm2)
				vcvtph2ps(xmm3, ymm3)
				vcvtph2ps(xmm4, ymm4)
				vcvtph2ps(xmm5, ymm5)
				vcvtph2ps(xmm6, ymm6)
				vcvtph2ps(xmm7, ymm7)
				STORE_4x16xFP32()
				add(imm(4*4*16), rbx)// add stride to dst pointer
				dec(r14)
				jne(UNROLLED8)

				label(EDGELOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)// starting fp16 loop
				vmovups(mem(rax), xmm0)
				vmovups(mem(rax, 8*2), xmm1)
				add(r12, rax)// add stride to src pointer
				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, ymm1)
				vmovups(ymm0, mem(rbx))
				vmovaps(ymm1, mem(rbx, 8*4))
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
						[src_stride] "m"(src_stride),
						[convert_fp16] "m"(convert_fp16)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
						"%r12", "%r13", "%r14", "%r15")
			}
		}
		else
		{
			uint64_t k_iter = dst.rows() / 8;
			uint64_t k_left = dst.rows() % 8;

			if (src.dtype() == DTYPE_FLOAT32)
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

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				movq(rax, rcx)// tmp src pointer is in rcx
				LOAD_8x8xFP32()
				AVX_8x8_TRANSPOSE()
				vmovaps(ymm8, mem(rbx, (0*16+0)*4))
				vmovaps(ymm9, mem(rbx, (1*16+0)*4))
				vmovaps(ymm10, mem(rbx, (2*16+0)*4))
				vmovaps(ymm11, mem(rbx, (3*16+0)*4))
				vmovaps(ymm12, mem(rbx, (4*16+0)*4))
				vmovaps(ymm13, mem(rbx, (5*16+0)*4))
				vmovaps(ymm14, mem(rbx, (6*16+0)*4))
				vmovaps(ymm15, mem(rbx, (7*16+0)*4))
				LOAD_8x8xFP32()
				AVX_8x8_TRANSPOSE()
				vmovaps(ymm8, mem(rbx, (0*16+8)*4))
				vmovaps(ymm9, mem(rbx, (1*16+8)*4))
				vmovaps(ymm10, mem(rbx, (2*16+8)*4))
				vmovaps(ymm11, mem(rbx, (3*16+8)*4))
				vmovaps(ymm12, mem(rbx, (4*16+8)*4))
				vmovaps(ymm13, mem(rbx, (5*16+8)*4))
				vmovaps(ymm14, mem(rbx, (6*16+8)*4))
				vmovaps(ymm15, mem(rbx, (7*16+8)*4))
				add(imm(4*8), rax)// add stride to src pointer
				add(imm(4*16*8), rbx)// add stride to dst pointer
				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)// tmp src pointer is in rcx
				LOAD_16x1xFP32()
				STORE_16x1xFP32()
				add(imm(4*1), rax)// add stride to src pointer
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
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
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

				movq(var(k_iter), r14)// load the number of 8-unrolled iterations
				test(r14, r14)
				je(FINALLOOP)

				label(UNROLLED8)
				// first 8x8 tile
				movq(rax, rcx)// tmp src pointer is in rcx
				LOAD_8x8xFP16()
				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, ymm1)
				vcvtph2ps(xmm2, ymm2)
				vcvtph2ps(xmm3, ymm3)
				vcvtph2ps(xmm4, ymm4)
				vcvtph2ps(xmm5, ymm5)
				vcvtph2ps(xmm6, ymm6)
				vcvtph2ps(xmm7, ymm7)
				AVX_8x8_TRANSPOSE()
				vmovaps(ymm8, mem(rbx, (0*16+0)*4))
				vmovaps(ymm9, mem(rbx, (1*16+0)*4))
				vmovaps(ymm10, mem(rbx, (2*16+0)*4))
				vmovaps(ymm11, mem(rbx, (3*16+0)*4))
				vmovaps(ymm12, mem(rbx, (4*16+0)*4))
				vmovaps(ymm13, mem(rbx, (5*16+0)*4))
				vmovaps(ymm14, mem(rbx, (6*16+0)*4))
				vmovaps(ymm15, mem(rbx, (7*16+0)*4))
				LOAD_8x8xFP16()
				vcvtph2ps(xmm0, ymm0)
				vcvtph2ps(xmm1, ymm1)
				vcvtph2ps(xmm2, ymm2)
				vcvtph2ps(xmm3, ymm3)
				vcvtph2ps(xmm4, ymm4)
				vcvtph2ps(xmm5, ymm5)
				vcvtph2ps(xmm6, ymm6)
				vcvtph2ps(xmm7, ymm7)
				AVX_8x8_TRANSPOSE()
				vmovaps(ymm8, mem(rbx, (0*16+8)*4))
				vmovaps(ymm9, mem(rbx, (1*16+8)*4))
				vmovaps(ymm10, mem(rbx, (2*16+8)*4))
				vmovaps(ymm11, mem(rbx, (3*16+8)*4))
				vmovaps(ymm12, mem(rbx, (4*16+8)*4))
				vmovaps(ymm13, mem(rbx, (5*16+8)*4))
				vmovaps(ymm14, mem(rbx, (6*16+8)*4))
				vmovaps(ymm15, mem(rbx, (7*16+8)*4))
				add(imm(8*2), rax)// add stride to src pointer
				add(imm(16*8*4), rbx)// add stride to dst pointer

				dec(r14)
				jne(UNROLLED8)

				label(FINALLOOP)
				movq(var(k_left), r14)// load the number of 1-unrolled iterations
				test(r14, r14)
				je(EPILOGUE)

				label(UNROLLED1)
				movq(rax, rcx)// tmp src pointer is in rcx

				movzw(mem(rcx), r8)
				movzw(mem(rcx, r12, 1), r9)
				movzw(mem(rcx, r12, 2), r10)
				movzw(mem(rcx, r13, 1), r11)
				vmovq(r8, xmm0)
				vmovq(r9, xmm1)
				vmovq(r10, xmm2)
				vmovq(r11, xmm3)
				add(r15, rcx)
				movzw(mem(rcx), r8)
				movzw(mem(rcx, r12, 1), r9)
				movzw(mem(rcx, r12, 2), r10)
				movzw(mem(rcx, r13, 1), r11)
				vmovq(r8, xmm4)
				vmovq(r9, xmm5)
				vmovq(r10, xmm6)
				vmovq(r11, xmm7)
				add(r15, rcx)
				movzw(mem(rcx), r8)
				movzw(mem(rcx, r12, 1), r9)
				movzw(mem(rcx, r12, 2), r10)
				movzw(mem(rcx, r13, 1), r11)
				vmovq(r8, xmm8)
				vmovq(r9, xmm9)
				vmovq(r10, xmm10)
				vmovq(r11, xmm11)
				add(r15, rcx)
				movzw(mem(rcx), r8)
				movzw(mem(rcx, r12, 1), r9)
				movzw(mem(rcx, r12, 2), r10)
				movzw(mem(rcx, r13, 1), r11)
				vmovq(r8, xmm12)
				vmovq(r9, xmm13)
				vmovq(r10, xmm14)
				vmovq(r11, xmm15)

				vcvtph2ps(xmm0, xmm0)
				vcvtph2ps(xmm1, xmm1)
				vcvtph2ps(xmm2, xmm2)
				vcvtph2ps(xmm3, xmm3)
				vcvtph2ps(xmm4, xmm4)
				vcvtph2ps(xmm5, xmm5)
				vcvtph2ps(xmm6, xmm6)
				vcvtph2ps(xmm7, xmm7)
				vcvtph2ps(xmm8, xmm8)
				vcvtph2ps(xmm9, xmm9)
				vcvtph2ps(xmm10, xmm10)
				vcvtph2ps(xmm11, xmm11)
				vcvtph2ps(xmm12, xmm12)
				vcvtph2ps(xmm13, xmm13)
				vcvtph2ps(xmm14, xmm14)
				vcvtph2ps(xmm15, xmm15)

				STORE_16x1xFP32()
				add(imm(1*2), rax)// add stride to src pointer
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
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
						"%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
			}
		}
	}
	// multi-head attention (MHA) kernel
	void mha_qk_avx2_12x8(Fragment &temp, const void *alpha_ptr, const Fragment &Q, const Fragment &K, const Fragment &bias,
			Fragment &softmax_sum) noexcept
	{
		assert(temp.is_fp32());
		assert(Q.is_fp32());
		assert(K.is_fp32());
		assert(Q.rows() == K.rows());
		assert(Q.stride() == 12);
		assert(K.stride() == 8);
		assert(temp.columns() == Q.columns());
		assert(temp.rows() == K.columns());
		assert(temp.stride() == 12);

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(Q.data(), 32));
		assert(cpu::is_aligned(K.data(), 32));

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
		if (softmax_sum.is_packed())
		{
			assert(softmax_sum.is_fp32());
			assert(softmax_sum.rows() >= temp.columns());
			assert(cpu::is_aligned(softmax_sum.data(), 32));
		}

		uint64_t k_iter = Q.rows() / 4;
		uint64_t k_left = Q.rows() % 4;
		const uint64_t bias_stride = bias.stride_in_bytes();
		assert(bias_stride % 32 == 0);

		begin_asm()
		movq(var(Q_ptr), rax) // lhs pointer is in rax
		movq(var(K_ptr), rbx)// rhs pointer is in rbx
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_12xFP32_8xFP32(0)
		SUB_KERNEL_12xFP32_8xFP32(1)
		SUB_KERNEL_12xFP32_8xFP32(2)
		SUB_KERNEL_12xFP32_8xFP32(3)

		add(imm(4*12*4), rax)
		add(imm(4*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_12xFP32_8xFP32(0)
		add(imm(1*12*4), rax)
		add(imm(1*8*4), rbx)
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		vbroadcastss(mem(rax), ymm1)
		PERMUTE_AND_SCALE_4x8xFP32(ymm4, ymm5, ymm6, ymm7)
		PERMUTE_AND_SCALE_4x8xFP32(ymm8, ymm9, ymm10, ymm11)
		PERMUTE_AND_SCALE_4x8xFP32(ymm12, ymm13, ymm14, ymm15)

		movq(var(bias_ptr), rbx)// load address of bias pointer
		test(rbx, rbx)
		je(AFTER_BIAS)
		movq(var(bias_stride), r14)// load address of bias stride into r14
		movq(r14, r13)
		sal(imm(1), r13)// r13 = stride * 2
		add(r14, r13)// r13 == stride * 3
		movq(r14, r15)
		sal(imm(2), r15)// r15 = stride * 4
		ADD_BIAS_4x8xFP32(ymm4, ymm5, ymm6, ymm7)
		ADD_BIAS_4x8xFP32(ymm8, ymm9, ymm10, ymm11)
		ADD_BIAS_4x8xFP32(ymm12, ymm13, ymm14, ymm15)
		label(AFTER_BIAS)

		EXP_FIRST_STAGE_12x8xFP32()
		EXP_SECOND_STAGE_3x8xFP32(ymm4, ymm5, ymm6)
		EXP_SECOND_STAGE_3x8xFP32(ymm7, ymm8, ymm9)
		EXP_SECOND_STAGE_3x8xFP32(ymm10, ymm11, ymm12)
		EXP_SECOND_STAGE_3x8xFP32(ymm13, ymm14, ymm15)

		movq(var(temp_ptr), rbx)// temp pointer is in rbx
		movq(var(softmax_ptr), rcx)// softmax sum pointer is in rcx

		AVX_4x8_TRANSPOSE()
		vmovups(xmm4, mem(rbx, (0*12+0)*4))
		vmovups(xmm5, mem(rbx, (1*12+0)*4))
		vmovups(xmm6, mem(rbx, (2*12+0)*4))
		vmovups(xmm7, mem(rbx, (3*12+0)*4))
		vmovups(xmm0, mem(rbx, (4*12+0)*4))
		vmovups(xmm1, mem(rbx, (5*12+0)*4))
		vmovups(xmm2, mem(rbx, (6*12+0)*4))
		vmovups(xmm3, mem(rbx, (7*12+0)*4))

		test(rcx, rcx)
		je(SKIP_REDUCTION_1)
		REDUCE_SUM()// sum registers ymm0-ymm7 and place result in ymm0
		vmovaps(mem(rcx), xmm1)// load previous sum
		vaddps(xmm1, xmm0, xmm0)// add current sum
		vmovaps(xmm0, mem(rcx))
		label(SKIP_REDUCTION_1)

		AVX_8x8_TRANSPOSE_INV()
		vmovups(ymm0, mem(rbx, (0*12+4)*4))
		vmovups(ymm1, mem(rbx, (1*12+4)*4))
		vmovups(ymm2, mem(rbx, (2*12+4)*4))
		vmovups(ymm3, mem(rbx, (3*12+4)*4))
		vmovups(ymm4, mem(rbx, (4*12+4)*4))
		vmovups(ymm5, mem(rbx, (5*12+4)*4))
		vmovups(ymm6, mem(rbx, (6*12+4)*4))
		vmovups(ymm7, mem(rbx, (7*12+4)*4))

		test(rcx, rcx)
		je(SKIP_REDUCTION_2)
		REDUCE_SUM()// sum registers ymm0-ymm7 and place result in ymm0
		vmovups(mem(rcx, 4*4), ymm1)// load previous sum
		vaddps(ymm1, ymm0, ymm0)// add current sum
		vmovups(ymm0, mem(rcx, 4*4))
		label(SKIP_REDUCTION_2)

		vzeroupper()

		end_asm(:// outputs
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
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15",
				"%rax", "%rbx", "%rcx", "%r12", "%r13", "%r14", "%r15")
	}
	void mha_softmax_avx2_12x8(Fragment &temp, Fragment &softmax_sum) noexcept
	{
		assert(temp.is_fp32());
		assert(temp.columns() == 12);
		assert(temp.rows() == 8);
		assert(temp.stride() == 12);
		assert(cpu::is_aligned(temp.data(), 32));

		float *temp_ptr = temp.data<float>();
		float *softmax_ptr = softmax_sum.is_packed() ? softmax_sum.data<float>() : nullptr;
		if (softmax_sum.is_packed())
		{
			assert(softmax_sum.is_fp32());
			assert(softmax_sum.rows() >= temp.columns());
			assert(cpu::is_aligned(softmax_sum.data(), 32));
		}

		begin_asm()
		movq(var(temp_ptr), rcx) // temp pointer is in rbx
		vmovaps(mem(rcx, 8*4*0), ymm4)
		vmovaps(mem(rcx, 8*4*1), ymm5)
		vmovaps(mem(rcx, 8*4*2), ymm6)
		vmovaps(mem(rcx, 8*4*3), ymm7)
		vmovaps(mem(rcx, 8*4*4), ymm8)
		vmovaps(mem(rcx, 8*4*5), ymm9)
		vmovaps(mem(rcx, 8*4*6), ymm10)
		vmovaps(mem(rcx, 8*4*7), ymm11)
		vmovaps(mem(rcx, 8*4*8), ymm12)
		vmovaps(mem(rcx, 8*4*9), ymm13)
		vmovaps(mem(rcx, 8*4*10), ymm14)
		vmovaps(mem(rcx, 8*4*11), ymm15)

		EXP_FIRST_STAGE_12x8xFP32()
		EXP_SECOND_STAGE_3x8xFP32(ymm4, ymm5, ymm6)
		EXP_SECOND_STAGE_3x8xFP32(ymm7, ymm8, ymm9)
		EXP_SECOND_STAGE_3x8xFP32(ymm10, ymm11, ymm12)
		EXP_SECOND_STAGE_3x8xFP32(ymm13, ymm14, ymm15)

		vmovaps(ymm4, mem(rcx, 8*4*0))
		vmovaps(ymm5, mem(rcx, 8*4*1))
		vmovaps(ymm6, mem(rcx, 8*4*2))
		vmovaps(ymm7, mem(rcx, 8*4*3))
		vmovaps(ymm8, mem(rcx, 8*4*4))
		vmovaps(ymm9, mem(rcx, 8*4*5))
		vmovaps(ymm10, mem(rcx, 8*4*6))
		vmovaps(ymm11, mem(rcx, 8*4*7))
		vmovaps(ymm12, mem(rcx, 8*4*8))
		vmovaps(ymm13, mem(rcx, 8*4*9))
		vmovaps(ymm14, mem(rcx, 8*4*10))
		vmovaps(ymm15, mem(rcx, 8*4*11))

		movq(var(softmax_ptr), rcx)// softmax sum pointer is in rcx
		test(rcx, rcx)
		je(SKIP_REDUCTION)
		vxorps(ymm0, ymm0, ymm0)
		vxorps(ymm1, ymm1, ymm1)
		vxorps(ymm2, ymm2, ymm2)

		vaddps(ymm4, ymm0, ymm0)
		vaddps(ymm5, ymm1, ymm1)
		vaddps(ymm6, ymm2, ymm2)
		vaddps(ymm7, ymm0, ymm0)
		vaddps(ymm8, ymm1, ymm1)
		vaddps(ymm9, ymm2, ymm2)
		vaddps(ymm10, ymm0, ymm0)
		vaddps(ymm11, ymm1, ymm1)
		vaddps(ymm12, ymm2, ymm2)
		vaddps(ymm13, ymm0, ymm0)
		vaddps(ymm14, ymm1, ymm1)
		vaddps(ymm15, ymm2, ymm2)

		vmovaps(mem(rcx, 4*4*0), xmm13)// load previous sum
		vmovaps(mem(rcx, 4*4*1), xmm14)// load previous sum
		vmovaps(mem(rcx, 4*4*2), xmm15)// load previous sum

		vextractf128(imm(0x1), ymm0, xmm3)
		vextractf128(imm(0x1), ymm1, xmm4)
		vextractf128(imm(0x1), ymm2, xmm5)

		vaddps(xmm13, xmm0, xmm13)
		vaddps(xmm14, xmm3, xmm14)
		vaddps(xmm15, xmm1, xmm15)

		vaddps(xmm13, xmm4, xmm13)
		vaddps(xmm14, xmm2, xmm14)
		vaddps(xmm15, xmm5, xmm15)

		vmovaps(xmm13, mem(rcx, 4*4*0))
		vmovaps(xmm14, mem(rcx, 4*4*1))
		vmovaps(xmm15, mem(rcx, 4*4*2))
		label(SKIP_REDUCTION)

		vzeroupper()

		end_asm(:// outputs
				:// inputs
				[temp_ptr] "m"(temp_ptr),
				[softmax_ptr] "m"(softmax_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15",
				"%rcx", "r14")
	}

	// batched depthwise convolution kernel

#define DWCONV_MAIN_LOOP_FP32() \
	vmovups(mem(rax, 0*8*4), ymm8) \
	vmovups(mem(rax, 1*8*4), ymm9) \
	vmovups(mem(rax, 2*8*4), ymm10) \
	vmovups(mem(rax, 3*8*4), ymm11) \
	vmovups(mem(rbx, 0*8*4), ymm12) \
	vmovups(mem(rbx, 1*8*4), ymm13) \
	vmovups(mem(rbx, 2*8*4), ymm14) \
	vmovups(mem(rbx, 3*8*4), ymm15) \
	vfmadd231ps(ymm8, ymm12, ymm0) \
	vfmadd231ps(ymm9, ymm13, ymm1) \
	vfmadd231ps(ymm10, ymm14, ymm2) \
	vfmadd231ps(ymm11, ymm15, ymm3) \
	vmovups(mem(rax, 4*8*4), ymm8) \
	vmovups(mem(rax, 5*8*4), ymm9) \
	vmovups(mem(rax, 6*8*4), ymm10) \
	vmovups(mem(rax, 7*8*4), ymm11) \
	vmovups(mem(rbx, 4*8*4), ymm12) \
	vmovups(mem(rbx, 5*8*4), ymm13) \
	vmovups(mem(rbx, 6*8*4), ymm14) \
	vmovups(mem(rbx, 7*8*4), ymm15) \
	vfmadd231ps(ymm8, ymm12, ymm4) \
	vfmadd231ps(ymm9, ymm13, ymm5) \
	vfmadd231ps(ymm10, ymm14, ymm6) \
	vfmadd231ps(ymm11, ymm15, ymm7)
#define DWCONV_MAIN_LOOP_FP16() \
	vmovups(mem(rax, 0*8*2), xmm8) \
	vmovups(mem(rax, 1*8*2), xmm9) \
	vmovups(mem(rax, 2*8*2), xmm10) \
	vmovups(mem(rax, 3*8*2), xmm11) \
	vmovups(mem(rbx, 0*8*2), xmm12) \
	vmovups(mem(rbx, 1*8*2), xmm13) \
	vmovups(mem(rbx, 2*8*2), xmm14) \
	vmovups(mem(rbx, 3*8*2), xmm15) \
	vcvtph2ps(xmm8, ymm8) \
	vcvtph2ps(xmm9, ymm9) \
	vcvtph2ps(xmm10, ymm10) \
	vcvtph2ps(xmm11, ymm11) \
	vcvtph2ps(xmm12, ymm12) \
	vcvtph2ps(xmm13, ymm13) \
	vcvtph2ps(xmm14, ymm14) \
	vcvtph2ps(xmm15, ymm15) \
	vfmadd231ps(ymm8, ymm12, ymm0) \
	vfmadd231ps(ymm9, ymm13, ymm1) \
	vfmadd231ps(ymm10, ymm14, ymm2) \
	vfmadd231ps(ymm11, ymm15, ymm3) \
	vmovups(mem(rax, 4*8*2), xmm8) \
	vmovups(mem(rax, 5*8*2), xmm9) \
	vmovups(mem(rax, 6*8*2), xmm10) \
	vmovups(mem(rax, 7*8*2), xmm11) \
	vmovups(mem(rbx, 4*8*2), xmm12) \
	vmovups(mem(rbx, 5*8*2), xmm13) \
	vmovups(mem(rbx, 6*8*2), xmm14) \
	vmovups(mem(rbx, 7*8*2), xmm15) \
	vcvtph2ps(xmm8, ymm8) \
	vcvtph2ps(xmm9, ymm9) \
	vcvtph2ps(xmm10, ymm10) \
	vcvtph2ps(xmm11, ymm11) \
	vcvtph2ps(xmm12, ymm12) \
	vcvtph2ps(xmm13, ymm13) \
	vcvtph2ps(xmm14, ymm14) \
	vcvtph2ps(xmm15, ymm15) \
	vfmadd231ps(ymm8, ymm12, ymm4) \
	vfmadd231ps(ymm9, ymm13, ymm5) \
	vfmadd231ps(ymm10, ymm14, ymm6) \
	vfmadd231ps(ymm11, ymm15, ymm7)

#define DWCONV_LOAD_BIAS_FP32() \
	vmovups(mem(rdx, 0*8*4), ymm0) \
	vmovups(mem(rdx, 1*8*4), ymm1) \
	vmovups(mem(rdx, 2*8*4), ymm2) \
	vmovups(mem(rdx, 3*8*4), ymm3) \
	vmovups(mem(rdx, 4*8*4), ymm4) \
	vmovups(mem(rdx, 5*8*4), ymm5) \
	vmovups(mem(rdx, 6*8*4), ymm6) \
	vmovups(mem(rdx, 7*8*4), ymm7)
#define DWCONV_LOAD_BIAS_FP16() \
	vmovups(mem(rdx, 0*8*2), xmm0) \
	vmovups(mem(rdx, 1*8*2), xmm1) \
	vmovups(mem(rdx, 2*8*2), xmm2) \
	vmovups(mem(rdx, 3*8*2), xmm3) \
	vmovups(mem(rdx, 4*8*2), xmm4) \
	vmovups(mem(rdx, 5*8*2), xmm5) \
	vmovups(mem(rdx, 6*8*2), xmm6) \
	vmovups(mem(rdx, 7*8*2), xmm7) \
	vcvtph2ps(xmm0, ymm0) \
	vcvtph2ps(xmm1, ymm1) \
	vcvtph2ps(xmm2, ymm2) \
	vcvtph2ps(xmm3, ymm3) \
	vcvtph2ps(xmm4, ymm4) \
	vcvtph2ps(xmm5, ymm5) \
	vcvtph2ps(xmm6, ymm6) \
	vcvtph2ps(xmm7, ymm7)

#define DWCONV_STORE_OUTPUT_FP32() \
	vmovups(ymm0, mem(rcx, 0*8*4)) \
	vmovups(ymm1, mem(rcx, 1*8*4)) \
	vmovups(ymm2, mem(rcx, 2*8*4)) \
	vmovups(ymm3, mem(rcx, 3*8*4)) \
	vmovups(ymm4, mem(rcx, 4*8*4)) \
	vmovups(ymm5, mem(rcx, 5*8*4)) \
	vmovups(ymm6, mem(rcx, 6*8*4)) \
	vmovups(ymm7, mem(rcx, 7*8*4))
#define DWCONV_STORE_OUTPUT_FP16() \
	vcvtps2ph(imm(0x03), ymm0, xmm0) \
	vcvtps2ph(imm(0x03), ymm1, xmm1) \
	vcvtps2ph(imm(0x03), ymm2, xmm2) \
	vcvtps2ph(imm(0x03), ymm3, xmm3) \
	vcvtps2ph(imm(0x03), ymm4, xmm4) \
	vcvtps2ph(imm(0x03), ymm5, xmm5) \
	vcvtps2ph(imm(0x03), ymm6, xmm6) \
	vcvtps2ph(imm(0x03), ymm7, xmm7) \
	vmovups(xmm0, mem(rcx, 0*8*2)) \
	vmovups(xmm1, mem(rcx, 1*8*2)) \
	vmovups(xmm2, mem(rcx, 2*8*2)) \
	vmovups(xmm3, mem(rcx, 3*8*2)) \
	vmovups(xmm4, mem(rcx, 4*8*2)) \
	vmovups(xmm5, mem(rcx, 5*8*2)) \
	vmovups(xmm6, mem(rcx, 6*8*2)) \
	vmovups(xmm7, mem(rcx, 7*8*2)) \

	void depthwise_conv_avx2_12x8(Matrix &output, const Matrix &input, const Matrix &weights, const Matrix &bias, const int *args,
			void *workspace) noexcept
	{
		assert(args != nullptr);
		const int batch_size = args[0];
		const int height = args[1];
		const int width = args[2];
		const uint64_t channels = args[3];
		const int kernel_height = args[4];
		const int kernel_width = args[5];

		const int padding_h = (kernel_height - 1) / 2;
		const int padding_w = (kernel_width - 1) / 2;

		assert(output.is_fp32() || output.is_fp16());
		assert(input.is_fp32() || input.is_fp16());
		assert(weights.is_fp32() || weights.is_fp16());
		assert(bias.is_fp32() || bias.is_fp16());
		assert(channels % 64 == 0);

		const uint8_t *b_ptr = reinterpret_cast<const uint8_t*>(bias.data());

		const uint64_t stride_w = input.stride_in_bytes();

		for (int i = 0; i < output.rows(); i++)
		{
			uint8_t *output_ptr = reinterpret_cast<uint8_t*>(output.data()) + i * output.stride_in_bytes();

			const int origin_b = reinterpret_cast<int*>(workspace)[3 * i + 0];
			const int origin_h = reinterpret_cast<int*>(workspace)[3 * i + 1];
			const int origin_w = reinterpret_cast<int*>(workspace)[3 * i + 2];

			const int kh0 = padding_h - std::min(origin_h, padding_h);
			const int kh1 = padding_h + std::min(height - 1 - origin_h, padding_h);

			const int kw0 = padding_w - std::min(origin_w, padding_w);
			const int kw1 = padding_w + std::min(width - 1 - origin_w, padding_w);

			const uint64_t elements_h = 1 + kh1 - kh0;
			const uint64_t elements_w = 1 + kw1 - kw0;

			const int h0 = origin_h + kh0 - padding_h;
			const int w0 = origin_w + kw0 - padding_w;

			const uint64_t in_stride_h = (width - elements_w) * stride_w;
			const uint64_t w_stride_h = (kernel_width - elements_w) * stride_w;

			const uint8_t *in_ptr = reinterpret_cast<const uint8_t*>(input.data()) + ((origin_b * height + h0) * width + w0) * stride_w;
			const uint8_t *w_ptr = reinterpret_cast<const uint8_t*>(weights.data()) + (kh0 * kernel_width + kw0) * stride_w;

			const uint64_t in_stride_back = elements_h * width * stride_w;
			const uint64_t w_stride_back = elements_h * kernel_width * stride_w;

			const uint64_t c_iter = channels / 64;

			if (input.is_fp32())
			{
				begin_asm()
				movq(var(input_ptr), rax)
				movq(var(weights_ptr), rbx)
				movq(var(output_ptr), rcx)
				movq(var(bias_ptr), rdx)

				movq(var(stride_w), r11)

				movq(var(in_stride_h), r12)
				movq(var(w_stride_h), r13)

				movq(var(in_stride_back), r14)
				movq(var(w_stride_back), r15)

				movq(var(c_iter), r8)
				label(CHANNELLOOP)

				DWCONV_LOAD_BIAS_FP32()

				movq(var(elements_h), r9)
				label(VERTICALLOOP)

				movq(var(elements_w), r10)
				label(HORIZONTALLOOP)

				DWCONV_MAIN_LOOP_FP32()

				add(r11, rax) // add channels
				add(r11, rbx)// add channels
				dec(r10)
				jne(HORIZONTALLOOP)

				add(r12, rax)// add channels
				add(r13, rbx)// add channels
				dec(r9)
				jne(VERTICALLOOP)

				sub(r14, rax)
				sub(r15, rbx)
				add(imm(8*8*4), rax)
				add(imm(8*8*4), rbx)

				DWCONV_STORE_OUTPUT_FP32()

				add(imm(8*8*4), rcx)
				add(imm(8*8*4), rdx)
				dec(r8)
				jne(CHANNELLOOP)

				vzeroupper()

				end_asm(
						:// outputs
						:// inputs
						[input_ptr] "m"(in_ptr),
						[weights_ptr] "m"(w_ptr),
						[output_ptr] "m"(output_ptr),
						[bias_ptr] "m"(b_ptr),
						[stride_w] "m"(stride_w),
						[in_stride_h] "m"(in_stride_h),
						[w_stride_h] "m"(w_stride_h),
						[in_stride_back] "m"(in_stride_back),
						[w_stride_back] "m"(w_stride_back),
						[c_iter] "m"(c_iter),
						[elements_h] "m"(elements_h),
						[elements_w] "m"(elements_w)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
						"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
			}
			else
			{
				if (output.is_fp16())
				{
					begin_asm()
					movq(var(input_ptr), rax)
					movq(var(weights_ptr), rbx)
					movq(var(output_ptr), rcx)
					movq(var(bias_ptr), rdx)

					movq(var(stride_w), r11)

					movq(var(in_stride_h), r12)
					movq(var(w_stride_h), r13)

					movq(var(in_stride_back), r14)
					movq(var(w_stride_back), r15)

					movq(var(c_iter), r8)
					label(CHANNELLOOP)

					DWCONV_LOAD_BIAS_FP16()

					movq(var(elements_h), r9)
					label(VERTICALLOOP)

					movq(var(elements_w), r10)
					label(HORIZONTALLOOP)

					DWCONV_MAIN_LOOP_FP16()

					add(r11, rax) // add channels
					add(r11, rbx)// add channels
					dec(r10)
					jne(HORIZONTALLOOP)

					add(r12, rax)// add channels
					add(r13, rbx)// add channels
					dec(r9)
					jne(VERTICALLOOP)

					sub(r14, rax)
					sub(r15, rbx)
					add(imm(8*8*2), rax)
					add(imm(8*8*2), rbx)

					DWCONV_STORE_OUTPUT_FP16()

					add(imm(8*8*2), rcx)
					add(imm(8*8*2), rdx)
					dec(r8)
					jne(CHANNELLOOP)

					vzeroupper()

					end_asm(
							:// outputs
							:// inputs
							[input_ptr] "m"(in_ptr),
							[weights_ptr] "m"(w_ptr),
							[output_ptr] "m"(output_ptr),
							[bias_ptr] "m"(b_ptr),
							[stride_w] "m"(stride_w),
							[in_stride_h] "m"(in_stride_h),
							[w_stride_h] "m"(w_stride_h),
							[in_stride_back] "m"(in_stride_back),
							[w_stride_back] "m"(w_stride_back),
							[c_iter] "m"(c_iter),
							[elements_h] "m"(elements_h),
							[elements_w] "m"(elements_w)
							:// clobbers
							"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
							"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
				}
				else
				{
					begin_asm()
					movq(var(input_ptr), rax)
					movq(var(weights_ptr), rbx)
					movq(var(output_ptr), rcx)
					movq(var(bias_ptr), rdx)

					movq(var(stride_w), r11)

					movq(var(in_stride_h), r12)
					movq(var(w_stride_h), r13)

					movq(var(in_stride_back), r14)
					movq(var(w_stride_back), r15)

					movq(var(c_iter), r8)
					label(CHANNELLOOP)

					DWCONV_LOAD_BIAS_FP16()

					movq(var(elements_h), r9)
					label(VERTICALLOOP)

					movq(var(elements_w), r10)
					label(HORIZONTALLOOP)

					DWCONV_MAIN_LOOP_FP16()

					add(r11, rax) // add channels
					add(r11, rbx)// add channels
					dec(r10)
					jne(HORIZONTALLOOP)

					add(r12, rax)// add channels
					add(r13, rbx)// add channels
					dec(r9)
					jne(VERTICALLOOP)

					sub(r14, rax)
					sub(r15, rbx)
					add(imm(8*8*2), rax)
					add(imm(8*8*2), rbx)

					DWCONV_STORE_OUTPUT_FP32()

					add(imm(8*8*4), rcx)
					add(imm(8*8*2), rdx)
					dec(r8)
					jne(CHANNELLOOP)

					vzeroupper()

					end_asm(
							:// outputs
							:// inputs
							[input_ptr] "m"(in_ptr),
							[weights_ptr] "m"(w_ptr),
							[output_ptr] "m"(output_ptr),
							[bias_ptr] "m"(b_ptr),
							[stride_w] "m"(stride_w),
							[in_stride_h] "m"(in_stride_h),
							[w_stride_h] "m"(w_stride_h),
							[in_stride_back] "m"(in_stride_back),
							[w_stride_back] "m"(w_stride_back),
							[c_iter] "m"(c_iter),
							[elements_h] "m"(elements_h),
							[elements_w] "m"(elements_w)
							:// clobbers
							"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
							"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15")
				}
			}
		}
	}
	void fused_conv_block_stage_1_avx2_12x8(Fragment &temp, const Fragment &A, const Fragment &B, const Fragment &bias) noexcept
	{
		assert(temp.is_fp32());
		assert(A.is_fp32());
		assert(B.is_fp32());
		assert(bias.is_fp32());

		assert(temp.is_packed());
		assert(A.is_packed());
		assert(B.is_packed());
		assert(bias.is_packed());

		assert(cpu::is_aligned(A.data(), 32));
		assert(cpu::is_aligned(B.data(), 32));
		assert(cpu::is_aligned(bias.data(), 32));

		assert(A.rows() == B.rows());
		assert(A.stride() == 12);
		assert(B.stride() == 8);
		assert(temp.columns() == A.columns());
		assert(temp.rows() == B.columns());

		const void *A_ptr = A.data();
		const void *B_ptr = B.data();
		void *temp_ptr = temp.data();
		const void *bias_ptr = bias.data();

		uint64_t k_iter = A.rows() / 4;
		uint64_t k_left = A.rows() % 4;

		begin_asm()
		movq(var(A_ptr), rax)
		movq(var(B_ptr), rbx)
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14) // load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_12xFP32_8xFP32(0)
		SUB_KERNEL_12xFP32_8xFP32(1)
		SUB_KERNEL_12xFP32_8xFP32(2)
		SUB_KERNEL_12xFP32_8xFP32(3)

		add(imm(4*12*4), rax)// 4 iterations x 12 elements x 4 bytes
		add(imm(4*8*4), rbx)// 4 iterations x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_12xFP32_8xFP32(0)
		add(imm(1*12*4), rax)// 1 iteration x 12 elements x 4 bytes
		add(imm(1*8*4), rbx)// 1 iteration x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)
		// permute back to original layout
		PERMUTE_6x8xFP32(ymm4, ymm5, ymm6, ymm7, ymm8, ymm9)
		PERMUTE_6x8xFP32(ymm10, ymm11, ymm12, ymm13, ymm14, ymm15)

		movq(var(bias_ptr), rax)// load address of bias pointer
		vmovaps(mem(rax), ymm2)// load bias value
		ADD_BIAS_12x8xFP32(ymm2)
		RELU_12x8xFP32()

		// transpose and store into packed fragment of D
		movq(var(temp_ptr), rcx)// temp pointer is in rcx
		AVX_4x8_TRANSPOSE()
		vmovups(xmm4, mem(rcx, (0*12+0)*4))
		vmovups(xmm5, mem(rcx, (1*12+0)*4))
		vmovups(xmm6, mem(rcx, (2*12+0)*4))
		vmovups(xmm7, mem(rcx, (3*12+0)*4))
		vmovups(xmm0, mem(rcx, (4*12+0)*4))
		vmovups(xmm1, mem(rcx, (5*12+0)*4))
		vmovups(xmm2, mem(rcx, (6*12+0)*4))
		vmovups(xmm3, mem(rcx, (7*12+0)*4))

		AVX_8x8_TRANSPOSE_INV()
		vmovups(ymm0, mem(rcx, (0*12+4)*4))
		vmovups(ymm1, mem(rcx, (1*12+4)*4))
		vmovups(ymm2, mem(rcx, (2*12+4)*4))
		vmovups(ymm3, mem(rcx, (3*12+4)*4))
		vmovups(ymm4, mem(rcx, (4*12+4)*4))
		vmovups(ymm5, mem(rcx, (5*12+4)*4))
		vmovups(ymm6, mem(rcx, (6*12+4)*4))
		vmovups(ymm7, mem(rcx, (7*12+4)*4))
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[A_ptr] "m"(A_ptr),
				[B_ptr] "m"(B_ptr),
				[temp_ptr] "m"(temp_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[bias_ptr] "m"(bias_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r11", "%r13", "%r14", "%r15")
	}
	void quantize_avx2_8xK(Fragment &dst, const Fragment &src) noexcept
	{
		assert(src.is_fp32());
		assert(dst.stride() == 8);
		assert(ml::cpu::is_aligned(dst.data(), 32));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = (dst.rows() % 8 + 1) / 2;
		const void *src_ptr = src.data();
		void *dst_ptr = dst.data();
		const uint32_t tmp[2] = { 0x7FFFFFFF, 4095 };
		const void *tmp_ptr = &tmp;

		begin_asm()
		movq(var(tmp_ptr), rcx)
		vbroadcastss(mem(rcx), ymm14)
		vxorps(ymm15, ymm15, ymm15) // zero scale

		movq(var(src_ptr), rax)// src pointer is in rax

		movq(var(k_iter), r14)// load the number of 8-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED8)
		vmovaps(mem(rax, 0*8*4), ymm0)
		vmovaps(mem(rax, 1*8*4), ymm1)
		vmovaps(mem(rax, 2*8*4), ymm2)
		vmovaps(mem(rax, 3*8*4), ymm3)
		vmovaps(mem(rax, 4*8*4), ymm4)
		vmovaps(mem(rax, 5*8*4), ymm5)
		vmovaps(mem(rax, 6*8*4), ymm6)
		vmovaps(mem(rax, 7*8*4), ymm7)

		vandps(ymm0, ymm14, ymm0)// calculate abs(x)
		vandps(ymm1, ymm14, ymm1)
		vandps(ymm2, ymm14, ymm2)
		vandps(ymm3, ymm14, ymm3)
		vandps(ymm4, ymm14, ymm4)
		vandps(ymm5, ymm14, ymm5)
		vandps(ymm6, ymm14, ymm6)
		vandps(ymm7, ymm14, ymm7)

		vmaxps(ymm0, ymm1, ymm0)// reduce max()
		vmaxps(ymm2, ymm3, ymm2)
		vmaxps(ymm4, ymm5, ymm4)
		vmaxps(ymm6, ymm6, ymm6)
		vmaxps(ymm0, ymm2, ymm0)
		vmaxps(ymm4, ymm6, ymm4)
		vmaxps(ymm0, ymm4, ymm0)
		vmaxps(ymm0, ymm15, ymm15)

		add(imm(8*8*4), rax)// add stride to src pointer
		dec(r14)
		jne(UNROLLED8)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 2-unrolled iterations
		test(r14, r14)
		je(AFTER_SCALE)

		label(UNROLLED2)
		vmovaps(mem(rax, 0*8*4), ymm0)
		vmovaps(mem(rax, 1*8*4), ymm1)

		vandps(ymm0, ymm14, ymm0)// calculate abs(x)
		vandps(ymm1, ymm14, ymm1)// calculate abs(x)
		vmaxps(ymm0, ymm1, ymm0)
		vmaxps(ymm0, ymm15, ymm15)

		add(imm(2*8*4), rax)// add stride to src pointer
		dec(r14)
		jne(UNROLLED2)

		label(AFTER_SCALE)

		vbroadcastss(mem(rcx, 4), ymm14)
		vdivps(ymm15, ymm14, ymm15)

		movq(var(src_ptr), rax)// src pointer is in rax
		movq(var(dst_ptr), rbx)// dst pointer is in rbx

		movq(var(k_iter), r14)// load the number of 8-unrolled iterations
		test(r14, r14)
		je(FINALLOOP_2)

		label(UNROLLED8_2)
		vmovaps(mem(rax, 0*8*4), ymm0)
		vmovaps(mem(rax, 1*8*4), ymm1)
		vmovaps(mem(rax, 2*8*4), ymm2)
		vmovaps(mem(rax, 3*8*4), ymm3)
		vmovaps(mem(rax, 4*8*4), ymm4)
		vmovaps(mem(rax, 5*8*4), ymm5)
		vmovaps(mem(rax, 6*8*4), ymm6)
		vmovaps(mem(rax, 7*8*4), ymm7)

		vmulps(ymm0, ymm15, ymm0)
		vmulps(ymm1, ymm15, ymm1)
		vmulps(ymm2, ymm15, ymm2)
		vmulps(ymm3, ymm15, ymm3)
		vmulps(ymm4, ymm15, ymm4)
		vmulps(ymm5, ymm15, ymm5)
		vmulps(ymm6, ymm15, ymm6)
		vmulps(ymm7, ymm15, ymm7)

		vcvtps2dq(ymm0, ymm0)
		vcvtps2dq(ymm1, ymm1)
		vcvtps2dq(ymm2, ymm2)
		vcvtps2dq(ymm3, ymm3)
		vcvtps2dq(ymm4, ymm4)
		vcvtps2dq(ymm5, ymm5)
		vcvtps2dq(ymm6, ymm6)
		vcvtps2dq(ymm7, ymm7)

		vpslld(imm(16), ymm0, ymm0)
		vpslld(imm(16), ymm2, ymm2)
		vpslld(imm(16), ymm4, ymm4)
		vpslld(imm(16), ymm6, ymm6)
		vorps(ymm0, ymm1, ymm0)
		vorps(ymm2, ymm3, ymm2)
		vorps(ymm4, ymm5, ymm4)
		vorps(ymm6, ymm7, ymm6)

		vmovaps(ymm0, mem(rbx, 0*16*2))
		vmovaps(ymm2, mem(rbx, 1*16*2))
		vmovaps(ymm4, mem(rbx, 2*16*2))
		vmovaps(ymm6, mem(rbx, 3*16*2))

		add(imm(8*8*4), rax)// add stride to src pointer
		add(imm(4*8*4), rbx)// add stride to dst pointer
		dec(r14)
		jne(UNROLLED8_2)

		label(FINALLOOP_2)
		movq(var(k_left), r14)// load the number of 2-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED2_2)
		vmovaps(mem(rax, 0*8*4), ymm0)
		vmovaps(mem(rax, 1*8*4), ymm1)

		vmulps(ymm0, ymm15, ymm0)
		vmulps(ymm1, ymm15, ymm1)

		vcvtps2dq(ymm0, ymm0)
		vcvtps2dq(ymm1, ymm1)

		vpslld(imm(16), ymm0, ymm0)
		vorps(ymm0, ymm1, ymm0)

		vmovaps(ymm0, mem(rbx))

		add(imm(2*8*4), rax)// add stride to src pointer
		add(imm(1*8*4), rbx)// add stride to dst pointer
		dec(r14)
		jne(UNROLLED2_2)

		label(EPILOGUE)
		vzeroupper()

		end_asm(:// outputs
				:// inputs
				[src_ptr] "m"(src_ptr),
				[dst_ptr] "m"(dst_ptr),
				[tmp_ptr] "m"(tmp_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void quantize_avx2_12xK(Fragment &dst, const Fragment &src) noexcept
	{
		assert(src.is_fp32());
		assert(dst.stride() == 12);
		assert(ml::cpu::is_aligned(dst.data(), 32));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = (dst.rows() % 8 + 1) / 2;
		const void *src_ptr = src.data();
		void *dst_ptr = dst.data();
		const uint32_t tmp[2] = { 0x7FFFFFFF, 4095 };
		const void *tmp_ptr = &tmp;

		begin_asm()
		movq(var(tmp_ptr), rcx)
		vbroadcastss(mem(rcx), ymm12)
		vxorps(ymm13, ymm13, ymm13) // zero scale
		vxorps(ymm14, ymm14, ymm14)// zero scale
		vxorps(ymm15, ymm15, ymm15)// zero scale

		movq(var(src_ptr), rax)// src pointer is in rax

		movq(var(k_iter), r14)// load the number of 8-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED8)
		vmovaps(mem(rax, 0*8*4), ymm0)
		vmovaps(mem(rax, 1*8*4), ymm1)
		vmovaps(mem(rax, 2*8*4), ymm2)
		vmovaps(mem(rax, 3*8*4), ymm3)
		vmovaps(mem(rax, 4*8*4), ymm4)
		vmovaps(mem(rax, 5*8*4), ymm5)
		vmovaps(mem(rax, 6*8*4), ymm6)
		vmovaps(mem(rax, 7*8*4), ymm7)
		vmovaps(mem(rax, 8*8*4), ymm8)
		vmovaps(mem(rax, 9*8*4), ymm9)
		vmovaps(mem(rax, 10*8*4), ymm10)
		vmovaps(mem(rax, 11*8*4), ymm11)

		vandps(ymm0, ymm12, ymm0)// calculate abs(x)
		vandps(ymm1, ymm12, ymm1)
		vandps(ymm2, ymm12, ymm2)
		vandps(ymm3, ymm12, ymm3)
		vandps(ymm4, ymm12, ymm4)
		vandps(ymm5, ymm12, ymm5)
		vandps(ymm6, ymm12, ymm6)
		vandps(ymm7, ymm12, ymm7)

		vmaxps(ymm0, ymm3, ymm0)// reduce max()
		vmaxps(ymm1, ymm4, ymm1)
		vmaxps(ymm2, ymm5, ymm2)
		vmaxps(ymm6, ymm9, ymm6)
		vmaxps(ymm7, ymm10, ymm7)
		vmaxps(ymm8, ymm11, ymm8)
		vmaxps(ymm0, ymm6, ymm0)
		vmaxps(ymm1, ymm7, ymm1)
		vmaxps(ymm2, ymm8, ymm2)
		vmaxps(ymm0, ymm13, ymm13)
		vmaxps(ymm1, ymm14, ymm14)
		vmaxps(ymm2, ymm15, ymm15)

		add(imm(8*12*4), rax)// add stride to src pointer
		dec(r14)
		jne(UNROLLED8)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 2-unrolled iterations
		test(r14, r14)
		je(AFTER_SCALE)

		label(UNROLLED2)
		vmovaps(mem(rax, 0*8*4), ymm0)
		vmovaps(mem(rax, 1*8*4), ymm1)
		vmovaps(mem(rax, 2*8*4), ymm2)

		vandps(ymm0, ymm12, ymm0)// calculate abs(x)
		vandps(ymm1, ymm12, ymm1)
		vandps(ymm2, ymm12, ymm2)
		vmaxps(ymm0, ymm13, ymm13)
		vmaxps(ymm1, ymm14, ymm14)
		vmaxps(ymm2, ymm15, ymm15)

		add(imm(2*12*4), rax)// add stride to src pointer
		dec(r14)
		jne(UNROLLED2)

		label(AFTER_SCALE)

		vperm2f128(imm(0x21), ymm14, ymm15, ymm12)
		vmaxps(ymm12, ymm13, ymm13)

		vbroadcastss(mem(rcx, 4), ymm12)
		vdivps(ymm13, ymm12, ymm13)
		vdivps(ymm14, ymm12, ymm14)
		vdivps(xmm15, xmm12, xmm15)

		movq(var(src_ptr), rax)// src pointer is in rax
		movq(var(dst_ptr), rbx)// dst pointer is in rbx

		movq(var(k_iter), r14)// load the number of 8-unrolled iterations
		test(r14, r14)
		je(FINALLOOP_2)

		label(UNROLLED8_2)
		vmovaps(mem(rax, 0*8*4), ymm0)
		vmovaps(mem(rax, 1*8*4), ymm1)
		vmovaps(mem(rax, 2*8*4), ymm2)
		vmovaps(mem(rax, 3*8*4), ymm3)
		vmovaps(mem(rax, 4*8*4), ymm4)
		vmovaps(mem(rax, 5*8*4), ymm5)
		vmovaps(mem(rax, 6*8*4), ymm6)
		vmovaps(mem(rax, 7*8*4), ymm7)
		vmovaps(mem(rax, 8*8*4), ymm8)
		vmovaps(mem(rax, 9*8*4), ymm9)
		vmovaps(mem(rax, 10*8*4), ymm10)
		vmovaps(mem(rax, 11*8*4), ymm11)

		vmulps(ymm0, ymm13, ymm0)
		vmulps(ymm1, ymm14, ymm1)
		vmulps(ymm2, ymm15, ymm2)
		vmulps(ymm3, ymm13, ymm3)
		vmulps(ymm4, ymm14, ymm4)
		vmulps(ymm5, ymm15, ymm5)
		vmulps(ymm6, ymm13, ymm6)
		vmulps(ymm7, ymm14, ymm7)
		vmulps(ymm8, ymm15, ymm8)
		vmulps(ymm9, ymm13, ymm9)
		vmulps(ymm10, ymm14, ymm10)
		vmulps(ymm11, ymm15, ymm11)

		vcvtps2dq(ymm0, ymm0)
		vcvtps2dq(ymm1, ymm1)
		vcvtps2dq(ymm2, ymm2)
		vcvtps2dq(ymm3, ymm3)
		vcvtps2dq(ymm4, ymm4)
		vcvtps2dq(ymm5, ymm5)
		vcvtps2dq(ymm6, ymm6)
		vcvtps2dq(ymm7, ymm7)
		vcvtps2dq(ymm8, ymm8)
		vcvtps2dq(ymm9, ymm9)
		vcvtps2dq(ymm10, ymm10)
		vcvtps2dq(ymm11, ymm11)

		vpslld(imm(16), ymm0, ymm0)
		vpslld(imm(16), ymm2, ymm2)
		vpslld(imm(16), ymm4, ymm4)
		vpslld(imm(16), ymm6, ymm6)
		vorps(ymm0, ymm1, ymm0)
		vorps(ymm2, ymm3, ymm2)
		vorps(ymm4, ymm5, ymm4)
		vorps(ymm6, ymm7, ymm6)

		vmovaps(ymm0, mem(rbx, 0*16*2))
		vmovaps(ymm2, mem(rbx, 1*16*2))
		vmovaps(ymm4, mem(rbx, 2*16*2))
		vmovaps(ymm6, mem(rbx, 3*16*2))
		vmovaps(ymm8, mem(rbx, 4*16*2))
		vmovaps(ymm10, mem(rbx, 5*16*2))

		add(imm(8*12*4), rax)// add stride to src pointer
		add(imm(4*12*4), rbx)// add stride to dst pointer
		dec(r14)
		jne(UNROLLED8_2)

		label(FINALLOOP_2)
		movq(var(k_left), r14)// load the number of 2-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED2_2)
		vmovaps(mem(rax, 0*8*4), ymm0)
		vmovaps(mem(rax, 1*8*4), ymm1)
		vmovaps(mem(rax, 2*8*4), ymm2)

		vmulps(ymm0, ymm13, ymm0)
		vmulps(ymm1, ymm14, ymm1)
		vmulps(ymm2, ymm15, ymm2)

		vcvtps2dq(ymm0, ymm0)
		vcvtps2dq(ymm1, ymm1)
		vcvtps2dq(ymm2, ymm2)

		vpslld(imm(16), ymm0, ymm0)
		vorps(ymm0, ymm1, ymm0)

		add(imm(2*12*4), rax)// add stride to src pointer
		add(imm(1*12*4), rbx)// add stride to dst pointer
		dec(r14)
		jne(UNROLLED2_2)

		label(EPILOGUE)
		vzeroupper()

		end_asm(:// outputs
				:// inputs
				[src_ptr] "m"(src_ptr),
				[dst_ptr] "m"(dst_ptr),
				[tmp_ptr] "m"(tmp_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
				"%r12", "%r13", "%r14", "%r15")
	}

	void intgemm_avx2_12x8(Fragment &D, const Fragment &alpha, const Fragment &A, const Fragment &B, const void *beta_ptr, const Fragment &C,
			const Fragment &bias, bool use_relu) noexcept
	{
//		assert(A.is_fp32());
//		assert(B.is_fp32());
//		assert(C.is_fp32() || C.is_fp16());
//		assert(D.is_fp32() || D.is_fp16());
//		assert(A.rows() == B.rows());
//		assert(A.stride() == 12);
//		assert(B.stride() == 8);
//		assert(D.rows() == A.columns());
//		assert(D.columns() == B.columns());
//
//		assert(alpha.is_packed());
//		assert(alpha.is_fp32());
//		assert(cpu::is_aligned(A.data(), 32));
//		assert(cpu::is_aligned(B.data(), 32));
//		assert(beta_ptr != nullptr);
		if (bias.is_packed())
		{
			assert(cpu::is_aligned(bias.data(), 32));
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
		const uint64_t cd_in_fp32 = C.is_fp32() | (D.is_fp32() << 1);
		const uint64_t scalar_alpha = alpha.rows() == 1;

		begin_asm()
		movq(var(A_ptr), rax)
		movq(var(B_ptr), rbx)
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14) // load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED_x4)
		SUB_KERNEL_12xINT16_8xINT16(0)
		SUB_KERNEL_12xINT16_8xINT16(1)
		SUB_KERNEL_12xINT16_8xINT16(2)
		SUB_KERNEL_12xINT16_8xINT16(3)

		add(imm(4*12*4), rax)// 4 iterations x 12 elements x 4 bytes
		add(imm(4*8*4), rbx)// 4 iterations x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED_x1)
		SUB_KERNEL_12xINT16_8xINT16(0)
		add(imm(1*12*4), rax)// 1 iteration x 12 elements x 4 bytes
		add(imm(1*8*4), rbx)// 1 iteration x 8 elements x 4 bytes
		dec(r14)
		jne(UNROLLED_x1)

		label(EPILOGUE)
		vcvtdq2ps(ymm4, ymm4)
		vcvtdq2ps(ymm5, ymm5)
		vcvtdq2ps(ymm6, ymm6)
		vcvtdq2ps(ymm7, ymm7)
		vcvtdq2ps(ymm8, ymm8)
		vcvtdq2ps(ymm9, ymm9)
		vcvtdq2ps(ymm10, ymm10)
		vcvtdq2ps(ymm11, ymm11)
		vcvtdq2ps(ymm12, ymm12)
		vcvtdq2ps(ymm13, ymm13)
		vcvtdq2ps(ymm14, ymm14)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(scalar_alpha), r14)
		test(r14, r14)
		je(COLUMN_ALPHA)
		vbroadcastss(mem(rax), ymm0)
		SCALE_ACCUMULATORS_1xN(ymm0)
		jmp(AFTER_ALPHA_SCALING)

		label(COLUMN_ALPHA)
		SCALE_ACCUMULATORS_12x1()
		label(AFTER_ALPHA_SCALING)

		// load address of bias pointer
		movq(var(bias_ptr), rax)
		test(rax, rax)
		je(AFTER_BIAS)
		vmovaps(mem(rax), ymm2)// load bias
		ADD_BIAS_12x8xFP32(ymm2)
		label(AFTER_BIAS)

//		movq(var(beta_ptr), rbx)// load address of beta
//		vbroadcastss(mem(rbx), ymm0)
//		vxorps(ymm1, ymm1, ymm1)
//		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
//		je(AFTER_LOAD_C)
//		movq(var(C_stride), r14)// C stride is r14
//		movq(var(C_ptr), rcx)// C pointer is in rcx
//		movq(r14, r15)// r15 = r14
//		sal(imm(1), r15)// r15 = 2 * r14
//		add(r14, r15)// r15 = 2 * r14 + r14 = 3 * r14 (3*C_stride)
//
//		movq(var(cd_in_fp32), r11)// load fp16 flags
//		and_(imm(0x1), r11)// if set
//		test(r11, r11)
//		je(C_IN_FP32)
//		LOAD_ADD_3x8xFP16(ymm0, ymm4, ymm5, ymm6)
//		LOAD_ADD_3x8xFP16(ymm0, ymm7, ymm8, ymm9)
//		LOAD_ADD_3x8xFP16(ymm0, ymm10, ymm11, ymm12)
//		LOAD_ADD_3x8xFP16(ymm0, ymm13, ymm14, ymm15)
//		jmp(AFTER_LOAD_C)
//
//		label(C_IN_FP32)
//		LOAD_ADD_3x8xFP32(ymm0, ymm4, ymm5, ymm6)
//		LOAD_ADD_3x8xFP32(ymm0, ymm7, ymm8, ymm9)
//		LOAD_ADD_3x8xFP32(ymm0, ymm10, ymm11, ymm12)
//		LOAD_ADD_3x8xFP32(ymm0, ymm13, ymm14, ymm15)
//		label(AFTER_LOAD_C)

		movq(var(flag_relu), r14)// load flag if to use relu
		test(r14, r14)
		je(AFTER_RELU)
		RELU_12x8xFP32()
		label(AFTER_RELU)

		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx
		movq(r14, r13)// r13 = r14
		sal(imm(1), r13)// r13 = 2 * r14 (2 * stride)
		add(r14, r13)// r13 = 2 * r14 + r14 = 3 * r14 (3*D_stride)
		movq(r14, r15)// r15 = r14
		sal(imm(2), r15)// r15 = 4 * r14 (4 * stride)

		movq(var(cd_in_fp32), r11)// load fp16 flags
		and_(imm(0x2), r11)// if set
		test(r11, r11)
		je(D_IN_INT8)
		STORE_4x8xFP32(ymm4, ymm5, ymm6, ymm7)
		STORE_4x8xFP32(ymm8, ymm9, ymm10, ymm11)
		STORE_4x8xFP32(ymm12, ymm13, ymm14, ymm15)
		jmp(END)

		label(D_IN_INT8)
		STORE_4x8xINT8(xmm4, xmm5, xmm6, xmm7)
		STORE_4x8xINT8(xmm8, xmm9, xmm10, xmm11)
		STORE_4x8xINT8(xmm12, xmm13, xmm14, xmm15)

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
				[cd_in_fp32] "m"(cd_in_fp32),
				[scalar_alpha] "m"(scalar_alpha)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
				"%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r11", "%r13", "%r14", "%r15")
	}
} /* namespace ml */

