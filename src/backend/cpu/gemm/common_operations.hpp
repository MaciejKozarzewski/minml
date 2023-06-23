/*
 * common_operations.hpp
 *
 *  Created on: Jun 23, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_GEMM_COMMON_OPERATIONS_HPP_
#define BACKEND_CPU_GEMM_COMMON_OPERATIONS_HPP_

#include "../assembly_macros.hpp"

/*
 * Takes 8 registers from ymm0-ymm7, transposes them and puts into ymm8-ymm15
 */
#define AVX_8x8_TRANSPOSE()\
	vunpcklps(ymm1, ymm0, ymm8)\
	vunpckhps(ymm1, ymm0, ymm9)\
	vunpcklps(ymm3, ymm2, ymm10)\
	vunpckhps(ymm3, ymm2, ymm11)\
	vunpcklps(ymm5, ymm4, ymm12)\
	vunpckhps(ymm5, ymm4, ymm13)\
	vunpcklps(ymm7, ymm6, ymm14)\
	vunpckhps(ymm7, ymm6, ymm15)\
	vunpcklpd(ymm10, ymm8, ymm0)\
	vunpckhpd(ymm10, ymm8, ymm1)\
	vunpcklpd(ymm11, ymm9, ymm2)\
	vunpckhpd(ymm11, ymm9, ymm3)\
	vunpcklpd(ymm14, ymm12, ymm4)\
	vunpckhpd(ymm14, ymm12, ymm5)\
	vunpcklpd(ymm15, ymm13, ymm6)\
	vunpckhpd(ymm15, ymm13, ymm7)\
	vperm2f128(imm(0x20), ymm4, ymm0, ymm8)\
	vperm2f128(imm(0x20), ymm5, ymm1, ymm9)\
	vperm2f128(imm(0x20), ymm6, ymm2, ymm10)\
	vperm2f128(imm(0x20), ymm7, ymm3, ymm11)\
	vperm2f128(imm(0x31), ymm4, ymm0, ymm12)\
	vperm2f128(imm(0x31), ymm5, ymm1, ymm13)\
	vperm2f128(imm(0x31), ymm6, ymm2, ymm14)\
	vperm2f128(imm(0x31), ymm7, ymm3, ymm15)

#endif /* BACKEND_CPU_GEMM_COMMON_OPERATIONS_HPP_ */
