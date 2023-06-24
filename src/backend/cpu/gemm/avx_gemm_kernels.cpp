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
#include "common_operations.hpp"

#define ZERO_ACCUMULATORS()\
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


#define SUB_KERNEL_10xFP32_8xFP32(n) \
	vmovaps(mem(rbx, n*8*4), ymm0)\
	vbroadcastss(mem(rax, (10*n+0)*4), ymm1)\
	vbroadcastss(mem(rax, (10*n+1)*4), ymm2)\
	vbroadcastss(mem(rax, (10*n+2)*4), ymm3)\
	vbroadcastss(mem(rax, (10*n+3)*4), ymm4)\
	vbroadcastss(mem(rax, (10*n+4)*4), ymm5)\
	vmulps(ymm1, ymm0, ymm1)\
	vmulps(ymm2, ymm0, ymm2)\
	vmulps(ymm3, ymm0, ymm3)\
	vmulps(ymm4, ymm0, ymm4)\
	vmulps(ymm5, ymm0, ymm5)\
	vaddps(ymm1, ymm6, ymm6)\
	vaddps(ymm2, ymm7, ymm7)\
	vaddps(ymm3, ymm8, ymm8)\
	vaddps(ymm4, ymm9, ymm9)\
	vaddps(ymm5, ymm10, ymm10)\
	vbroadcastss(mem(rax, (10*n+5)*4), ymm1)\
	vbroadcastss(mem(rax, (10*n+6)*4), ymm2)\
	vbroadcastss(mem(rax, (10*n+7)*4), ymm3)\
	vbroadcastss(mem(rax, (10*n+8)*4), ymm4)\
	vbroadcastss(mem(rax, (10*n+9)*4), ymm5)\
	vmulps(ymm1, ymm0, ymm1)\
	vmulps(ymm2, ymm0, ymm2)\
	vmulps(ymm3, ymm0, ymm3)\
	vmulps(ymm4, ymm0, ymm4)\
	vmulps(ymm5, ymm0, ymm5)\
	vaddps(ymm1, ymm11, ymm11)\
	vaddps(ymm2, ymm12, ymm12)\
	vaddps(ymm3, ymm13, ymm13)\
	vaddps(ymm4, ymm14, ymm14)\
	vaddps(ymm5, ymm15, ymm15)

#define SCALE_ACCUMULATORS_BY(reg)\
	vmulps(reg, ymm6, ymm6) \
	vmulps(reg, ymm7, ymm7) \
	vmulps(reg, ymm8, ymm8) \
	vmulps(reg, ymm9, ymm9) \
	vmulps(reg, ymm10, ymm10) \
	vmulps(reg, ymm11, ymm11) \
	vmulps(reg, ymm12, ymm12) \
	vmulps(reg, ymm13, ymm13) \
	vmulps(reg, ymm14, ymm14) \
	vmulps(reg, ymm15, ymm15)

#define LOAD_5x8xFP32()\
	vmovups(mem(rcx), ymm1)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm2)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm3)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm4)\
	add(r14, rcx)\
	vmovups(mem(rcx), ymm5)\
	add(r14, rcx)

#define LOAD_5x8xFP16()\
	vmovups(mem(rcx), xmm1)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm2)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm3)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm4)\
	add(r14, rcx)\
	vmovups(mem(rcx), xmm5)\
	add(r14, rcx)

#define SCALE_5x8xFP32_BY_BETA()\
	vmulps(ymm1, ymm0, ymm1)\
	vmulps(ymm2, ymm0, ymm2)\
	vmulps(ymm3, ymm0, ymm3)\
	vmulps(ymm4, ymm0, ymm4)\
	vmulps(ymm5, ymm0, ymm5)

#define ADD_5x8xFP32_TO_ACCUMULATORS(acc0, acc1, acc2, acc3, acc4)\
	vaddps(ymm1, acc0, acc0)\
	vaddps(ymm2, acc1, acc1)\
	vaddps(ymm3, acc2, acc2)\
	vaddps(ymm4, acc3, acc3)\
	vaddps(ymm5, acc4, acc4)

#define LOAD_ADD_16xFP16(beta, reg1, reg2, stride)\
	vmovups(mem(rcx, 0*stride), xmm2)\
	vmovups(mem(rcx, 1*stride), xmm3)\
	vcvtph2ps(xmm2, ymm2)\
	vcvtph2ps(xmm3, ymm3)\
	vmulps(reg1, beta, reg1)\
	vmulps(reg2, beta, reg2)\
	vaddps(reg1, ymm2, reg1)\
	vaddps(reg1, ymm3, reg2)\
	add(r14, rcx)

#define STORE_5x8xFP32(reg1, reg2, reg3, reg4, reg5)\
	vmovups(reg1, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg2, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg3, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg4, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg5, mem(rcx))\
	add(r14, rcx)

#define STORE_5x8xFP16(reg0, reg1, reg2, reg3, reg4)\
	vmovups(reg0, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg1, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg2, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg3, mem(rcx))\
	add(r14, rcx)\
	vmovups(reg4, mem(rcx))\
	add(r14, rcx)

#define CONVERT_5x8xFP16_TO_5x8xFP32()\
	vcvtph2ps(xmm1, ymm1)\
	vcvtph2ps(xmm2, ymm2)\
	vcvtph2ps(xmm3, ymm3)\
	vcvtph2ps(xmm4, ymm4)\
	vcvtph2ps(xmm5, ymm5)

#define CONVERT_ACCUMULATORS_TO_FP16()\
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

#define LOAD_1x10xFP32(reg0, reg1)\
	vmovups(mem(rax), ymm(reg0))\
	vmovsd (mem(rax, 8*4), xmm(reg1))\
	add(r12, rax)

#define LOAD_2x10xFP32(reg0, reg1, reg2, reg3)\
	vmovups(mem(rax), ymm(reg0))\
	vmovsd (mem(rax, 8*4), xmm(reg1))\
	vmovups(mem(rax, r15, 1), ymm(reg2))\
	vmovsd (mem(rax, r15, 1, 8*4), xmm(reg3))\
	add(r12, rax)

#define LOAD_1x10xFP16(reg0, reg1)\
	vmovups(mem(rax), xmm(reg0))\
	vmovss (mem(rax, 8*2), xmm(reg1))\
	add(r12, rax)\
	vcvtph2ps(xmm(reg0), ymm(reg0))\
	vcvtph2ps(xmm(reg1), xmm(reg1))

#define LOAD_2x10xFP16(reg0, reg1, reg2, reg3)\
	vmovups(mem(rax), xmm(reg0))\
	vmovss (mem(rax, 8*2), xmm(reg1))\
	vmovups(mem(rax, r15, 1), xmm(reg2))\
	vmovss (mem(rax, r15, 1, 8*2), xmm(reg3))\
	add(r12, rax)\
	vcvtph2ps(xmm(reg0), ymm(reg0))\
	vcvtph2ps(xmm(reg1), xmm(reg1))\
	vcvtph2ps(xmm(reg2), ymm(reg2))\
	vcvtph2ps(xmm(reg3), xmm(reg3))

#define STORE_1x10xFP32(n, reg0, reg1)\
	vmovups(ymm(reg0), (4*(n*10+0))(rbx))\
	vmovsd (xmm(reg1), (4*(n*10+8))(rbx))

namespace ml
{

	void gemm_avx_10x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.dtype() == DTYPE_FLOAT32);
		assert(B.dtype() == DTYPE_FLOAT32);
		assert(C.dtype() == DTYPE_FLOAT32);
		assert(D.dtype() == DTYPE_FLOAT32);
		assert(A.rows() == B.rows());
		assert(A.stride() == 10);
		assert(B.stride() == 8);
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

		begin_asm()
		movq(var(A_ptr), rax) // lhs pointer is in rax
		movq(var(B_ptr), rbx)// rhs pointer is in rbx
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED4)
		SUB_KERNEL_10xFP32_8xFP32(0)
		SUB_KERNEL_10xFP32_8xFP32(1)
		SUB_KERNEL_10xFP32_8xFP32(2)
		SUB_KERNEL_10xFP32_8xFP32(3)

		add(imm(4*10*4), rax)
		add(imm(4*8*4), rbx)
		dec(r14)
		jne(UNROLLED4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED1)
		SUB_KERNEL_10xFP32_8xFP32(0)
		add(imm(1*10*4), rax)
		add(imm(1*8*4), rbx)
		dec(r14)
		jne(UNROLLED1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rax), ymm1)
		vbroadcastss(mem(rbx), ymm0)

		SCALE_ACCUMULATORS_BY(ymm1)

		vxorps(ymm1, ymm1, ymm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(BETAZERO)
		// beta != 0 case
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx

		LOAD_5x8xFP32()
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm6, ymm7, ymm8, ymm9, ymm10)

		LOAD_5x8xFP32()
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm11, ymm12, ymm13, ymm14, ymm15)

		label(BETAZERO)
		// beta == 0 case
		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx

		STORE_5x8xFP32(ymm6, ymm7, ymm8, ymm9, ymm10)
		STORE_5x8xFP32(ymm11, ymm12, ymm13, ymm14, ymm15)

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
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14")
	}
	void gemm_avx_10x8_fp32_fp16(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.dtype() == DTYPE_FLOAT32);
		assert(B.dtype() == DTYPE_FLOAT32);
		assert(C.dtype() == DTYPE_FLOAT16);
		assert(D.dtype() == DTYPE_FLOAT16);
		assert(A.rows() == B.rows());
		assert(A.stride() == 10);
		assert(B.stride() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
		assert(cpu::is_aligned(A.data(), register_size<YMM>()));
		assert(cpu::is_aligned(B.data(), register_size<YMM>()));
		assert(beta_ptr != nullptr);

		const float *A_ptr = A.data<float>();
		const float *B_ptr = B.data<float>();
		const float16 *C_ptr = C.data<float16>();
		float16 *D_ptr = D.data<float16>();

		const int K = A.rows();
		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		const uint64_t C_stride = C.stride() * sizeof(float16);
		const uint64_t D_stride = D.stride() * sizeof(float16);

		begin_asm()
		movq(var(A_ptr), rax) // lhs pointer is in rax
		movq(var(B_ptr), rbx)// rhs pointer is in rbx
		ZERO_ACCUMULATORS()

		movq(var(k_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(FINALLOOP)

		label(UNROLLED4)
		SUB_KERNEL_10xFP32_8xFP32(0)
		SUB_KERNEL_10xFP32_8xFP32(1)
		SUB_KERNEL_10xFP32_8xFP32(2)
		SUB_KERNEL_10xFP32_8xFP32(3)

		add(imm(4*10*4), rax)
		add(imm(4*8*4), rbx)
		dec(r14)
		jne(UNROLLED4)

		label(FINALLOOP)
		movq(var(k_left), r14)// load the number of 1-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED1)
		SUB_KERNEL_10xFP32_8xFP32(0)
		add(imm(1*10*4), rax)
		add(imm(1*8*4), rbx)
		dec(r14)
		jne(UNROLLED1)

		label(EPILOGUE)

		movq(var(alpha_ptr), rax)// load address of alpha
		movq(var(beta_ptr), rbx)// load address of beta
		vbroadcastss(mem(rax), ymm1)
		vbroadcastss(mem(rbx), ymm0)

		SCALE_ACCUMULATORS_BY(ymm1)

		vxorps(ymm1, ymm1, ymm1)
		vucomiss(xmm0, xmm1)// set ZF if beta == 0.
		je(BETAZERO)
		// beta != 0 case
		movq(var(C_stride), r14)// C stride is r14
		movq(var(C_ptr), rcx)// C pointer is in rcx

		LOAD_5x8xFP16()
		CONVERT_5x8xFP16_TO_5x8xFP32()
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm6, ymm7, ymm8, ymm9, ymm10)

		LOAD_5x8xFP16()
		CONVERT_5x8xFP16_TO_5x8xFP32()
		SCALE_5x8xFP32_BY_BETA()
		ADD_5x8xFP32_TO_ACCUMULATORS(ymm11, ymm12, ymm13, ymm14, ymm15)

		label(BETAZERO)
		// beta == 0 case
		movq(var(D_stride), r14)// D stride is r14
		movq(var(D_ptr), rcx)// D pointer is in rcx

		CONVERT_ACCUMULATORS_TO_FP16()
		STORE_5x8xFP16(xmm6, xmm7, xmm8, xmm9, xmm10)
		STORE_5x8xFP16(xmm11, xmm12, xmm13, xmm14, xmm15)

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
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14")
	}

	void pack_avx_10xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.dtype() == DTYPE_FLOAT32);
		assert(src.dtype() == DTYPE_FLOAT32);
		assert(dst.stride() == 10);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = dst.rows() % 8;
		const uint64_t src_stride = src.stride() * sizeof(float);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12
			movq(r12, r15)
			sal(imm(1), r12)

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED8)
			LOAD_2x10xFP32(0, 1, 2, 3)
			LOAD_2x10xFP32(4, 5, 6, 7)
			LOAD_2x10xFP32(8, 9, 10, 11)
			LOAD_2x10xFP32(12, 13, 14, 15)

			STORE_1x10xFP32(0, 0, 1)
			STORE_1x10xFP32(1, 2, 3)
			STORE_1x10xFP32(2, 4, 5)
			STORE_1x10xFP32(3, 6, 7)
			STORE_1x10xFP32(4, 8, 9)
			STORE_1x10xFP32(5, 10, 11)
			STORE_1x10xFP32(6, 12, 13)
			STORE_1x10xFP32(7, 14, 15)

			add(imm(4*8*10), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED8)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			sar(imm(1), r12)// divide stride by 2, effectively reverting to reading single row at the time
			label(UNROLLED1)
			LOAD_1x10xFP32(0, 1)
			STORE_1x10xFP32(0, 0, 1)
			add(imm(4*1*10), rbx)// add stride to dst pointer

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
					"%r12", "%r14", "%r15")
		}
		else
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED8)
			// first 8x8 tile
			movq(rax, r13)// tmp src pointer is in r13
			// rows 0-7
			vmovups(mem(r13), ymm0)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), ymm1)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), ymm2)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), ymm3)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), ymm4)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), ymm5)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), ymm6)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), ymm7)
			add(r12, r13)// add stride to src pointer

			AVX_8x8_TRANSPOSE()

			vmovups(ymm8, mem(rbx,4*(0*10+0)))
			vmovups(ymm9, mem(rbx,4*(1*10+0)))
			vmovups(ymm10, mem(rbx,4*(2*10+0)))
			vmovups(ymm11, mem(rbx,4*(3*10+0)))
			vmovups(ymm12, mem(rbx,4*(4*10+0)))
			vmovups(ymm13, mem(rbx,4*(5*10+0)))
			vmovups(ymm14, mem(rbx,4*(6*10+0)))
			vmovups(ymm15, mem(rbx,4*(7*10+0)))

			// rows 8-9
			vmovups(mem(r13), ymm0)
			add(r12, r13)
			vmovups(mem(r13), ymm1)
			add(r12, r13)

			vunpcklps(ymm1, ymm0, ymm4)
			vunpckhps(ymm1, ymm0, ymm5)

			vextractf128(imm(0x1), ymm4, xmm6)// e4 f4 e5 f5
			vextractf128(imm(0x1), ymm5, xmm7)// e6 f6 e7 f7

			vmovlpd(xmm4, mem(rbx, 4*(0*10+8)))
			vmovhpd(xmm4, mem(rbx, 4*(1*10+8)))
			vmovlpd(xmm5, mem(rbx, 4*(2*10+8)))
			vmovhpd(xmm5, mem(rbx, 4*(3*10+8)))
			vmovlpd(xmm6, mem(rbx, 4*(4*10+8)))
			vmovhpd(xmm6, mem(rbx, 4*(5*10+8)))
			vmovlpd(xmm7, mem(rbx, 4*(6*10+8)))
			vmovhpd(xmm7, mem(rbx, 4*(7*10+8)))

			add(imm(4*8), rax)// add stride to src pointer
			add(imm(4*8*10), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED8)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(rax, r13)// tmp src pointer is in r13

			vmovss(mem(r13), xmm0)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm1)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm2)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm3)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm4)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm5)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm6)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm7)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm8)
			add(r12, r13)// add stride to src pointer
			vmovss(mem(r13), xmm9)
			add(r12, r13)// add stride to src pointer

			vmovss(xmm0, mem(rbx, 0*4))
			vmovss(xmm1, mem(rbx, 1*4))
			vmovss(xmm2, mem(rbx, 2*4))
			vmovss(xmm3, mem(rbx, 3*4))
			vmovss(xmm4, mem(rbx, 4*4))
			vmovss(xmm5, mem(rbx, 5*4))
			vmovss(xmm6, mem(rbx, 6*4))
			vmovss(xmm7, mem(rbx, 7*4))
			vmovss(xmm8, mem(rbx, 8*4))
			vmovss(xmm9, mem(rbx, 9*4))

			add(imm(4*1), rax)// add stride to src pointer
			add(imm(4*10*1), rbx)// add stride to dst pointer

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

		//		assert(dst.stride() == 10);
		//		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));
		//
		//		uint64_t k_iter = dst.rows() / 8;
		//		uint64_t k_left = dst.rows() % 8;
		//		const uint64_t src_stride = src.stride() * sizeof(float);
		//		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		//		void *dst_ptr = dst.data();
		//
		//		if (src_op == MatrixOp::NORMAL)
		//		{
		//			asm volatile(
		//					"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
		//					"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
		//					"movq %[src_stride], %%r12 \n\t"// src stride is in r12
		//
		//					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
		//					"test %%r14, %%r14 \n\t"
		//					"je FINALLOOP%= \n\t"
		//
		//					"UNROLLED8%=: \n\t"
		//					"vmovups 0x00(%%rax), %%ymm0 \n\t"
		//					"vmovsd  0x20(%%rax), %%xmm1 \n\t"
		//					"add %%r12, %%rax \n\t"// add stride to src pointer
		//					"vmovups 0x00(%%rax), %%ymm2 \n\t"
		//					"vmovsd  0x20(%%rax), %%xmm3 \n\t"
		//					"add %%r12, %%rax \n\t"// add stride to src pointer
		//					"vmovups 0x00(%%rax), %%ymm4 \n\t"
		//					"vmovsd  0x20(%%rax), %%xmm5 \n\t"
		//					"add %%r12, %%rax \n\t"// add stride to src pointer
		//					"vmovups 0x00(%%rax), %%ymm6 \n\t"
		//					"vmovsd  0x20(%%rax), %%xmm7 \n\t"
		//					"add %%r12, %%rax \n\t"// add stride to src pointer
		//					"vmovups 0x00(%%rax), %%ymm8 \n\t"
		//					"vmovsd  0x20(%%rax), %%xmm9 \n\t"
		//					"add %%r12, %%rax \n\t"// add stride to src pointer
		//					"vmovups 0x00(%%rax), %%ymm10 \n\t"
		//					"vmovsd  0x20(%%rax), %%xmm11 \n\t"
		//					"add %%r12, %%rax \n\t"// add stride to src pointer
		//					"vmovups 0x00(%%rax), %%ymm12 \n\t"
		//					"vmovsd  0x20(%%rax), %%xmm13 \n\t"
		//					"add %%r12, %%rax \n\t"// add stride to src pointer
		//					"vmovups 0x00(%%rax), %%ymm14 \n\t"
		//					"vmovsd  0x20(%%rax), %%xmm15 \n\t"
		//					"add %%r12, %%rax \n\t"// add stride to src pointer
		//
		//					"vmovups %%ymm0, (4*(0*10+0))(%%rbx) \n\t"
		//					"vmovsd  %%xmm1, (4*(0*10+8))(%%rbx) \n\t"
		//					"vmovups %%ymm2, (4*(1*10+0))(%%rbx) \n\t"
		//					"vmovsd  %%xmm3, (4*(1*10+8))(%%rbx) \n\t"
		//					"vmovups %%ymm4, (4*(2*10+0))(%%rbx) \n\t"
		//					"vmovsd  %%xmm5, (4*(2*10+8))(%%rbx) \n\t"
		//					"vmovups %%ymm6, (4*(3*10+0))(%%rbx) \n\t"
		//					"vmovsd  %%xmm7, (4*(3*10+8))(%%rbx) \n\t"
		//					"vmovups %%ymm8, (4*(4*10+0))(%%rbx) \n\t"
		//					"vmovsd  %%xmm9, (4*(4*10+8))(%%rbx) \n\t"
		//					"vmovups %%ymm10, (4*(5*10+0))(%%rbx) \n\t"
		//					"vmovsd  %%xmm11, (4*(5*10+8))(%%rbx) \n\t"
		//					"vmovups %%ymm12, (4*(6*10+0))(%%rbx) \n\t"
		//					"vmovsd  %%xmm13, (4*(6*10+8))(%%rbx) \n\t"
		//					"vmovups %%ymm14, (4*(7*10+0))(%%rbx) \n\t"
		//					"vmovsd  %%xmm15, (4*(7*10+8))(%%rbx) \n\t"
		//
		//					"add $(4*8*10), %%rbx \n\t"// add stride to dst pointer
		//
		//					"dec %%r14 \n\t"
		//					"jne UNROLLED8%= \n\t"
		//
		//					"FINALLOOP%=: \n\t"
		//					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
		//					"test %%r14, %%r14 \n\t"
		//					"je EPILOGUE%= \n\t"
		//
		//					"UNROLLED1%=: \n\t"
		//					"vmovups 0x00(%%rax), %%ymm0 \n\t"
		//					"vmovsd  0x20(%%rax), %%xmm1 \n\t"
		//					"add %%r12, %%rax \n\t"// add stride to src pointer
		//					"vmovups %%ymm0, 0x00(%%rbx) \n\t"
		//					"vmovsd  %%xmm1, 0x20(%%rbx) \n\t"
		//					"add $(4*1*10), %%rbx \n\t"// add stride to dst pointer
		//
		//					"dec %%r14 \n\t"
		//					"jne UNROLLED1%= \n\t"
		//
		//					"EPILOGUE%=: \n\t"
		//					"vzeroupper \n\t"
		//
		//					:// outputs
		//					:// inputs
		//					[src_ptr] "m"(src_ptr),
		//					[dst_ptr] "m"(dst_ptr),
		//					[k_iter] "m"(k_iter),
		//					[k_left] "m"(k_left),
		//					[src_stride] "m"(src_stride)
		//					:// clobbers
		//					"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
		//					"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
		//					"%r12", "%r14");
		//		}
		//		else
		//		{
		//			asm volatile(
		//					"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
		//					"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
		//					"movq %[src_stride], %%r12 \n\t"// src stride is in r12
		//
		//					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
		//					"test %%r14, %%r14 \n\t"
		//					"je FINALLOOP%= \n\t"
		//
		//					"UNROLLED8%=: \n\t"
		//					// first 8x8 tile
		//					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13
		//					// rows 0-7
		//					"vmovups 0x0(%%r13), %%ymm0 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovups 0x0(%%r13), %%ymm1 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovups 0x0(%%r13), %%ymm2 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovups 0x0(%%r13), %%ymm3 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovups 0x0(%%r13), %%ymm4 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovups 0x0(%%r13), %%ymm5 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovups 0x0(%%r13), %%ymm6 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovups 0x0(%%r13), %%ymm7 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//
		//					// transpose 8x8
		//					// first shuffle
		//					"vunpcklps %%ymm1, %%ymm0, %%ymm8 \n\t"
		//					"vunpckhps %%ymm1, %%ymm0, %%ymm9 \n\t"
		//					"vunpcklps %%ymm3, %%ymm2, %%ymm10 \n\t"
		//					"vunpckhps %%ymm3, %%ymm2, %%ymm11 \n\t"
		//					"vunpcklps %%ymm5, %%ymm4, %%ymm12 \n\t"
		//					"vunpckhps %%ymm5, %%ymm4, %%ymm13 \n\t"
		//					"vunpcklps %%ymm7, %%ymm6, %%ymm14 \n\t"
		//					"vunpckhps %%ymm7, %%ymm6, %%ymm15 \n\t"
		//
		//					// second shuffle
		//					"vunpcklpd %%ymm10, %%ymm8, %%ymm0 \n\t"
		//					"vunpckhpd %%ymm10, %%ymm8, %%ymm1 \n\t"
		//					"vunpcklpd %%ymm11, %%ymm9, %%ymm2 \n\t"
		//					"vunpckhpd %%ymm11, %%ymm9, %%ymm3 \n\t"
		//					"vunpcklpd %%ymm14, %%ymm12, %%ymm4 \n\t"
		//					"vunpckhpd %%ymm14, %%ymm12, %%ymm5 \n\t"
		//					"vunpcklpd %%ymm15, %%ymm13, %%ymm6 \n\t"
		//					"vunpckhpd %%ymm15, %%ymm13, %%ymm7 \n\t"
		//
		//					// third shuffle
		//					"vperm2f128 $0x20, %%ymm4, %%ymm0, %%ymm8 \n\t"
		//					"vperm2f128 $0x20, %%ymm5, %%ymm1, %%ymm9 \n\t"
		//					"vperm2f128 $0x20, %%ymm6, %%ymm2, %%ymm10 \n\t"
		//					"vperm2f128 $0x20, %%ymm7, %%ymm3, %%ymm11 \n\t"
		//					"vperm2f128 $0x31, %%ymm4, %%ymm0, %%ymm12 \n\t"
		//					"vperm2f128 $0x31, %%ymm5, %%ymm1, %%ymm13 \n\t"
		//					"vperm2f128 $0x31, %%ymm6, %%ymm2, %%ymm14 \n\t"
		//					"vperm2f128 $0x31, %%ymm7, %%ymm3, %%ymm15 \n\t"
		//
		//					"vmovups %%ymm8, (4*(0*10+0))(%%rbx) \n\t"
		//					"vmovups %%ymm9, (4*(1*10+0))(%%rbx) \n\t"
		//					"vmovups %%ymm10, (4*(2*10+0))(%%rbx) \n\t"
		//					"vmovups %%ymm11, (4*(3*10+0))(%%rbx) \n\t"
		//					"vmovups %%ymm12, (4*(4*10+0))(%%rbx) \n\t"
		//					"vmovups %%ymm13, (4*(5*10+0))(%%rbx) \n\t"
		//					"vmovups %%ymm14, (4*(6*10+0))(%%rbx) \n\t"
		//					"vmovups %%ymm15, (4*(7*10+0))(%%rbx) \n\t"
		//
		//					// rows 8-9
		//					"vmovups 0x0(%%r13), %%ymm0 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovups 0x0(%%r13), %%ymm1 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//
		//					"vunpcklps %%ymm1, %%ymm0, %%ymm4 \n\t"
		//					"vunpckhps %%ymm1, %%ymm0, %%ymm5 \n\t"
		//
		//					"vextractf128 $0x1, %%ymm4, %%xmm6 \n\t"// e4 f4 e5 f5
		//					"vextractf128 $0x1, %%ymm5, %%xmm7 \n\t"// e6 f6 e7 f7
		//
		//					"vmovlpd %%xmm4, (4*(0*10+8))(%%rbx) \n\t"
		//					"vmovhpd %%xmm4, (4*(1*10+8))(%%rbx) \n\t"
		//					"vmovlpd %%xmm5, (4*(2*10+8))(%%rbx) \n\t"
		//					"vmovhpd %%xmm5, (4*(3*10+8))(%%rbx) \n\t"
		//					"vmovlpd %%xmm6, (4*(4*10+8))(%%rbx) \n\t"
		//					"vmovhpd %%xmm6, (4*(5*10+8))(%%rbx) \n\t"
		//					"vmovlpd %%xmm7, (4*(6*10+8))(%%rbx) \n\t"
		//					"vmovhpd %%xmm7, (4*(7*10+8))(%%rbx) \n\t"
		//
		//					"add $(4*8), %%rax \n\t"// add stride to src pointer
		//					"add $(4*8*10), %%rbx \n\t"// add stride to dst pointer
		//
		//					"dec %%r14 \n\t"
		//					"jne UNROLLED8%= \n\t"
		//
		//					"FINALLOOP%=: \n\t"
		//					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
		//					"test %%r14, %%r14 \n\t"
		//					"je EPILOGUE%= \n\t"
		//
		//					"UNROLLED1%=: \n\t"
		//					"movq %%rax, %%r13 \n\t"// tmp src pointer is in r13
		//
		//					"vmovss 0x0(%%r13), %%xmm0 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovss 0x0(%%r13), %%xmm1 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovss 0x0(%%r13), %%xmm2 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovss 0x0(%%r13), %%xmm3 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovss 0x0(%%r13), %%xmm4 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovss 0x0(%%r13), %%xmm5 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovss 0x0(%%r13), %%xmm6 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovss 0x0(%%r13), %%xmm7 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovss 0x0(%%r13), %%xmm8 \n\t"
		//					"add %%r12, %%r13 \n\t"// add stride to src pointer
		//					"vmovss 0x0(%%r13), %%xmm9 \n\t"
		//
		//					"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
		//					"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
		//					"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
		//					"vmovss %%xmm3, (4*3)(%%rbx) \n\t"
		//					"vmovss %%xmm4, (4*4)(%%rbx) \n\t"
		//					"vmovss %%xmm5, (4*5)(%%rbx) \n\t"
		//					"vmovss %%xmm6, (4*6)(%%rbx) \n\t"
		//					"vmovss %%xmm7, (4*7)(%%rbx) \n\t"
		//					"vmovss %%xmm8, (4*8)(%%rbx) \n\t"
		//					"vmovss %%xmm9, (4*9)(%%rbx) \n\t"
		//
		//					"add $(4*1), %%rax \n\t"// add stride to src pointer
		//					"add $(4*10*1), %%rbx \n\t"// add stride to dst pointer
		//
		//					"dec %%r14 \n\t"
		//					"jne UNROLLED1%= \n\t"
		//
		//					"EPILOGUE%=: \n\t"
		//					"vzeroupper \n\t"
		//
		//					:// outputs
		//					:// inputs
		//					[src_ptr] "m"(src_ptr),
		//					[dst_ptr] "m"(dst_ptr),
		//					[k_iter] "m"(k_iter),
		//					[k_left] "m"(k_left),
		//					[src_stride] "m"(src_stride)
		//					:// clobbers
		//					"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
		//					"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
		//					"%r12", "%r13", "%r14");
		//		}
	}
	void pack_avx_10xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.dtype() == DTYPE_FLOAT32);
		assert(src.dtype() == DTYPE_FLOAT16);
		assert(dst.stride() == 10);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = dst.rows() % 8;
		const uint64_t src_stride = src.stride() * sizeof(float16);
		const void *src_ptr = src.pointer_at(src_pos.row, src_pos.column);
		void *dst_ptr = dst.data();

		if (src_op == MatrixOp::NORMAL)
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12
			movq(r12, r15)
			sal(imm(1), r12)

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED8)
			LOAD_2x10xFP16(0, 1, 2, 3)
			LOAD_2x10xFP16(4, 5, 6, 7)
			LOAD_2x10xFP16(8, 9, 10, 11)
			LOAD_2x10xFP16(12, 13, 14, 15)

			STORE_1x10xFP32(0, 0, 1)
			STORE_1x10xFP32(1, 2, 3)
			STORE_1x10xFP32(2, 4, 5)
			STORE_1x10xFP32(3, 6, 7)
			STORE_1x10xFP32(4, 8, 9)
			STORE_1x10xFP32(5, 10, 11)
			STORE_1x10xFP32(6, 12, 13)
			STORE_1x10xFP32(7, 14, 15)

			add(imm(4*8*10), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED8)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			sar(imm(1), r12)// divide stride by 2, effectively reverting to reading single row at the time
			label(UNROLLED1)
			LOAD_1x10xFP16(0, 1)
			STORE_1x10xFP32(0, 0, 1)
			add(imm(4*1*10), rbx)// add stride to dst pointer

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
					"%r12", "%r14", "%r15")
		}
		else
		{
			begin_asm()
			movq(var(src_ptr), rax) // src pointer is in rax
			movq(var(dst_ptr), rbx)// dst pointer is in rbx
			movq(var(src_stride), r12)// src stride is in r12

			movq(var(k_iter), r14)// load the number of 8-unrolled iterations
			test(r14, r14)
			je(FINALLOOP)

			label(UNROLLED8)
			// first 8x8 tile
			movq(rax, r13)// tmp src pointer is in r13
			// rows 0-7
			vmovups(mem(r13), xmm0)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), xmm1)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), xmm2)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), xmm3)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), xmm4)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), xmm5)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), xmm6)
			add(r12, r13)// add stride to src pointer
			vmovups(mem(r13), xmm7)
			add(r12, r13)// add stride to src pointer

			vcvtph2ps(xmm0, ymm0)
			vcvtph2ps(xmm1, ymm1)
			vcvtph2ps(xmm2, ymm2)
			vcvtph2ps(xmm3, ymm3)
			vcvtph2ps(xmm4, ymm4)
			vcvtph2ps(xmm5, ymm5)
			vcvtph2ps(xmm6, ymm6)
			vcvtph2ps(xmm7, ymm7)

			AVX_8x8_TRANSPOSE()

			vmovups(ymm8, mem(rbx,4*(0*10+0)))
			vmovups(ymm9, mem(rbx,4*(1*10+0)))
			vmovups(ymm10, mem(rbx,4*(2*10+0)))
			vmovups(ymm11, mem(rbx,4*(3*10+0)))
			vmovups(ymm12, mem(rbx,4*(4*10+0)))
			vmovups(ymm13, mem(rbx,4*(5*10+0)))
			vmovups(ymm14, mem(rbx,4*(6*10+0)))
			vmovups(ymm15, mem(rbx,4*(7*10+0)))

			// rows 8-9
			vmovups(mem(r13), xmm0)
			add(r12, r13)
			vmovups(mem(r13), xmm1)
			add(r12, r13)

			vcvtph2ps(xmm0, ymm0)
			vcvtph2ps(xmm1, ymm1)

			vunpcklps(ymm1, ymm0, ymm4)
			vunpckhps(ymm1, ymm0, ymm5)

			vextractf128(imm(0x1), ymm4, xmm6)// e4 f4 e5 f5
			vextractf128(imm(0x1), ymm5, xmm7)// e6 f6 e7 f7

			vmovlpd(xmm4, mem(rbx, 4*(0*10+8)))
			vmovhpd(xmm4, mem(rbx, 4*(1*10+8)))
			vmovlpd(xmm5, mem(rbx, 4*(2*10+8)))
			vmovhpd(xmm5, mem(rbx, 4*(3*10+8)))
			vmovlpd(xmm6, mem(rbx, 4*(4*10+8)))
			vmovhpd(xmm6, mem(rbx, 4*(5*10+8)))
			vmovlpd(xmm7, mem(rbx, 4*(6*10+8)))
			vmovhpd(xmm7, mem(rbx, 4*(7*10+8)))

			add(imm(2*8), rax)// add stride to src pointer
			add(imm(4*8*10), rbx)// add stride to dst pointer

			dec(r14)
			jne(UNROLLED8)

			label(FINALLOOP)
			movq(var(k_left), r14)// load the number of 1-unrolled iterations
			test(r14, r14)
			je(EPILOGUE)

			label(UNROLLED1)
			movq(rax, r13)// tmp src pointer is in r13

			movzw(mem(r13), rcx)
			vmovq(rcx, xmm0)
			add(r12, r13)// add stride to src pointer
			movzw(mem(r13), rcx)
			vmovq(rcx, xmm1)
			add(r12, r13)// add stride to src pointer
			movzw(mem(r13), rcx)
			vmovq(rcx, xmm2)
			add(r12, r13)// add stride to src pointer
			movzw(mem(r13), rcx)
			vmovq(rcx, xmm3)
			add(r12, r13)// add stride to src pointer
			movzw(mem(r13), rcx)
			vmovq(rcx, xmm4)
			add(r12, r13)// add stride to src pointer
			movzw(mem(r13), rcx)
			vmovq(rcx, xmm5)
			add(r12, r13)// add stride to src pointer
			movzw(mem(r13), rcx)
			vmovq(rcx, xmm6)
			add(r12, r13)// add stride to src pointer
			movzw(mem(r13), rcx)
			vmovq(rcx, xmm7)
			add(r12, r13)// add stride to src pointer
			movzw(mem(r13), rcx)
			vmovq(rcx, xmm8)
			add(r12, r13)// add stride to src pointer
			movzw(mem(r13), rcx)
			vmovq(rcx, xmm9)
			add(r12, r13)// add stride to src pointer

			vcvtph2ps(xmm0, ymm0)
			vcvtph2ps(xmm1, ymm1)
			vcvtph2ps(xmm2, ymm2)
			vcvtph2ps(xmm3, ymm3)
			vcvtph2ps(xmm4, ymm4)
			vcvtph2ps(xmm5, ymm5)
			vcvtph2ps(xmm6, ymm6)
			vcvtph2ps(xmm7, ymm7)
			vcvtph2ps(xmm8, ymm8)
			vcvtph2ps(xmm9, ymm9)

			vmovss(xmm0, mem(rbx, 0*4))
			vmovss(xmm1, mem(rbx, 1*4))
			vmovss(xmm2, mem(rbx, 2*4))
			vmovss(xmm3, mem(rbx, 3*4))
			vmovss(xmm4, mem(rbx, 4*4))
			vmovss(xmm5, mem(rbx, 5*4))
			vmovss(xmm6, mem(rbx, 6*4))
			vmovss(xmm7, mem(rbx, 7*4))
			vmovss(xmm8, mem(rbx, 8*4))
			vmovss(xmm9, mem(rbx, 9*4))

			add(imm(2*1), rax)// add stride to src pointer
			add(imm(4*10*1), rbx)// add stride to dst pointer

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

	}

	void pack_avx_8xK_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.dtype() == DTYPE_FLOAT32);
		assert(src.dtype() == DTYPE_FLOAT32);
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
	void pack_avx_8xK_fp16_fp32(Fragment &dst, const Matrix &src, const Position2D &src_pos, MatrixOp src_op) noexcept
	{
		assert(dst.dtype() == DTYPE_FLOAT32);
		assert(src.dtype() == DTYPE_FLOAT16);
		assert(dst.stride() == 8);
		assert(ml::cpu::is_aligned(dst.data(), register_size<YMM>()));

		uint64_t k_iter = dst.rows() / 8;
		uint64_t k_left = dst.rows() % 8;
		const uint64_t src_stride = src.stride() * sizeof(float16);
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
					"vmovups 0x00(%%rax), %%xmm0 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm2 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm4 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm6 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%xmm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"

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
					"vmovups 0x00(%%rax), %%xmm0 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
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

					"vmovups 0x0(%%r13), %%xmm0 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm1 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm2 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm3 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm4 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm5 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm6 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer
					"vmovups 0x0(%%r13), %%xmm7 \n\t"
					"add %%r12, %%r13 \n\t"// add stride to src pointer

					"vcvtph2ps %%xmm0, %%ymm0 \n\t"
					"vcvtph2ps %%xmm1, %%ymm1 \n\t"
					"vcvtph2ps %%xmm2, %%ymm2 \n\t"
					"vcvtph2ps %%xmm3, %%ymm3 \n\t"
					"vcvtph2ps %%xmm4, %%ymm4 \n\t"
					"vcvtph2ps %%xmm5, %%ymm5 \n\t"
					"vcvtph2ps %%xmm6, %%ymm6 \n\t"
					"vcvtph2ps %%xmm7, %%ymm7 \n\t"

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

					"add $(2*8), %%rax \n\t"// add stride to src pointer
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

					"vcvtph2ps %%xmm0, %%xmm0 \n\t"
					"vcvtph2ps %%xmm1, %%xmm1 \n\t"
					"vcvtph2ps %%xmm2, %%xmm2 \n\t"
					"vcvtph2ps %%xmm3, %%xmm3 \n\t"
					"vcvtph2ps %%xmm4, %%xmm4 \n\t"
					"vcvtph2ps %%xmm5, %%xmm5 \n\t"
					"vcvtph2ps %%xmm6, %%xmm6 \n\t"
					"vcvtph2ps %%xmm7, %%xmm7 \n\t"

					"vmovss %%xmm0, (4*0)(%%rbx) \n\t"
					"vmovss %%xmm1, (4*1)(%%rbx) \n\t"
					"vmovss %%xmm2, (4*2)(%%rbx) \n\t"
					"vmovss %%xmm3, (4*3)(%%rbx) \n\t"
					"vmovss %%xmm4, (4*4)(%%rbx) \n\t"
					"vmovss %%xmm5, (4*5)(%%rbx) \n\t"
					"vmovss %%xmm6, (4*6)(%%rbx) \n\t"
					"vmovss %%xmm7, (4*7)(%%rbx) \n\t"

					"add $(2*1), %%rax \n\t"// add stride to src pointer
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

} /* namespace ml */

