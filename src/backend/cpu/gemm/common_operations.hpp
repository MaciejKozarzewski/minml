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
#define AVX_8x8_TRANSPOSE() \
	vunpcklps(ymm1, ymm0, ymm8) \
	vunpckhps(ymm1, ymm0, ymm9) \
	vunpcklps(ymm3, ymm2, ymm10) \
	vunpckhps(ymm3, ymm2, ymm11) \
	vunpcklps(ymm5, ymm4, ymm12) \
	vunpckhps(ymm5, ymm4, ymm13) \
	vunpcklps(ymm7, ymm6, ymm14) \
	vunpckhps(ymm7, ymm6, ymm15) \
	vunpcklpd(ymm10, ymm8, ymm0) \
	vunpckhpd(ymm10, ymm8, ymm1) \
	vunpcklpd(ymm11, ymm9, ymm2) \
	vunpckhpd(ymm11, ymm9, ymm3) \
	vunpcklpd(ymm14, ymm12, ymm4) \
	vunpckhpd(ymm14, ymm12, ymm5) \
	vunpcklpd(ymm15, ymm13, ymm6) \
	vunpckhpd(ymm15, ymm13, ymm7) \
	vperm2f128(imm(0x20), ymm4, ymm0, ymm8) \
	vperm2f128(imm(0x20), ymm5, ymm1, ymm9) \
	vperm2f128(imm(0x20), ymm6, ymm2, ymm10) \
	vperm2f128(imm(0x20), ymm7, ymm3, ymm11) \
	vperm2f128(imm(0x31), ymm4, ymm0, ymm12) \
	vperm2f128(imm(0x31), ymm5, ymm1, ymm13) \
	vperm2f128(imm(0x31), ymm6, ymm2, ymm14) \
	vperm2f128(imm(0x31), ymm7, ymm3, ymm15)

/*
 * Takes 8 registers from ymm8-ymm15, transposes them and puts into ymm0-ymm7
 */
#define AVX_8x8_TRANSPOSE_INV() \
	vunpcklps(ymm9, ymm8, ymm0) \
	vunpckhps(ymm9, ymm8, ymm1) \
	vunpcklps(ymm11, ymm10, ymm2) \
	vunpckhps(ymm11, ymm10, ymm3) \
	vunpcklps(ymm13, ymm12, ymm4) \
	vunpckhps(ymm13, ymm12, ymm5) \
	vunpcklps(ymm15, ymm14, ymm6) \
	vunpckhps(ymm15, ymm14, ymm7) \
	vunpcklpd(ymm2, ymm0, ymm8) \
	vunpckhpd(ymm2, ymm0, ymm9) \
	vunpcklpd(ymm3, ymm1, ymm10) \
	vunpckhpd(ymm3, ymm1, ymm11) \
	vunpcklpd(ymm6, ymm4, ymm12) \
	vunpckhpd(ymm6, ymm4, ymm13) \
	vunpcklpd(ymm7, ymm5, ymm14) \
	vunpckhpd(ymm7, ymm5, ymm15) \
	vperm2f128(imm(0x20), ymm12, ymm8, ymm0) \
	vperm2f128(imm(0x20), ymm13, ymm9, ymm1) \
	vperm2f128(imm(0x20), ymm14, ymm10, ymm2) \
	vperm2f128(imm(0x20), ymm15, ymm11, ymm3) \
	vperm2f128(imm(0x31), ymm12, ymm8, ymm4) \
	vperm2f128(imm(0x31), ymm13, ymm9, ymm5) \
	vperm2f128(imm(0x31), ymm14, ymm10, ymm6) \
	vperm2f128(imm(0x31), ymm15, ymm11, ymm7)

#endif /* BACKEND_CPU_GEMM_COMMON_OPERATIONS_HPP_ */
