/*
 * avx2_gemm_kernels.cpp
 *
 *  Created on: May 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include "Fragment.hpp"
#include "gemm_kernels.hpp"
#include "../../utils.hpp"
#include "../../vectors/vectors.hpp"
#include "../../tiles/tile_transpose.hpp"

#include <x86intrin.h>
#include <cinttypes>
#include <cassert>

namespace
{
	template<int Rows, int Columns, typename SrcT, typename DstT, char OP>
	struct TileCopyAndTransform;

	template<>
	struct TileCopyAndTransform<8, 16, float, float, 'n'>
	{
			void operator()(float *dst_ptr, int dst_stride, const float *src_ptr, int src_stride) const noexcept
			{
				assert(dst_stride == 16);
				assert(ml::cpu::is_aligned(dst_ptr, 32));
				const uint64_t SrcStride = src_stride * sizeof(float);
				asm volatile(
						"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
						"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
						"movq %[SrcStride], %%r12 \n\t"// src stride is in r12

						"vmovups 0x00(%%rax), %%ymm0 \n\t"
						"vmovups 0x20(%%rax), %%ymm1 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%ymm2 \n\t"
						"vmovups 0x20(%%rax), %%ymm3 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%ymm4 \n\t"
						"vmovups 0x20(%%rax), %%ymm5 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%ymm6 \n\t"
						"vmovups 0x20(%%rax), %%ymm7 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%ymm8 \n\t"
						"vmovups 0x20(%%rax), %%ymm9 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%ymm10 \n\t"
						"vmovups 0x20(%%rax), %%ymm11 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%ymm12 \n\t"
						"vmovups 0x20(%%rax), %%ymm13 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x00(%%rax), %%ymm14 \n\t"
						"vmovups 0x20(%%rax), %%ymm15 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer

						"vmovaps %%ymm0, 0x000(%%rbx) \n\t"
						"vmovaps %%ymm1, 0x020(%%rbx) \n\t"
						"vmovaps %%ymm2, 0x040(%%rbx) \n\t"
						"vmovaps %%ymm3, 0x060(%%rbx) \n\t"
						"vmovaps %%ymm4, 0x080(%%rbx) \n\t"
						"vmovaps %%ymm5, 0x0A0(%%rbx) \n\t"
						"vmovaps %%ymm6, 0x0C0(%%rbx) \n\t"
						"vmovaps %%ymm7, 0x0E0(%%rbx) \n\t"
						"vmovaps %%ymm8, 0x100(%%rbx) \n\t"
						"vmovaps %%ymm9, 0x120(%%rbx) \n\t"
						"vmovaps %%ymm10, 0x140(%%rbx) \n\t"
						"vmovaps %%ymm11, 0x160(%%rbx) \n\t"
						"vmovaps %%ymm12, 0x180(%%rbx) \n\t"
						"vmovaps %%ymm13, 0x1A0(%%rbx) \n\t"
						"vmovaps %%ymm14, 0x1C0(%%rbx) \n\t"
						"vmovaps %%ymm15, 0x1E0(%%rbx) \n\t"

						"vzeroupper \n\t"

						:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[SrcStride] "m"(SrcStride)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
						"%r12");
			}
	};
	template<>
	struct TileCopyAndTransform<8, 8, float, float, 'n'>
	{
			void operator()(float *dst_ptr, int dst_stride, const float *src_ptr, int src_stride) const noexcept
			{
				assert(dst_stride == 8);
				assert(ml::cpu::is_aligned(dst_ptr, 32));
				const uint64_t SrcStride = src_stride * sizeof(float);
				asm volatile(
						"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
						"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
						"movq %[SrcStride], %%r12 \n\t"// src stride is in r12

						"vmovups 0x0(%%rax), %%ymm0 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x0(%%rax), %%ymm1 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x0(%%rax), %%ymm2 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x0(%%rax), %%ymm3 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x0(%%rax), %%ymm4 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x0(%%rax), %%ymm5 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x0(%%rax), %%ymm6 \n\t"
						"add %%r12, %%rax \n\t"// add stride to src pointer
						"vmovups 0x0(%%rax), %%ymm7 \n\t"

						"vmovaps %%ymm0, 0x00(%%rbx) \n\t"
						"vmovaps %%ymm1, 0x20(%%rbx) \n\t"
						"vmovaps %%ymm2, 0x40(%%rbx) \n\t"
						"vmovaps %%ymm3, 0x60(%%rbx) \n\t"
						"vmovaps %%ymm4, 0x80(%%rbx) \n\t"
						"vmovaps %%ymm5, 0xA0(%%rbx) \n\t"
						"vmovaps %%ymm6, 0xC0(%%rbx) \n\t"
						"vmovaps %%ymm7, 0xE0(%%rbx) \n\t"

						"vzeroupper \n\t"

						:// outputs
						:// inputs
						[src_ptr] "m"(src_ptr),
						[dst_ptr] "m"(dst_ptr),
						[SrcStride] "m"(SrcStride)
						:// clobbers
						"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
						"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
						"%r12");
			}
	};

	void kernel_pack_16xK_fp32(int K, void *dst_ptr, const void *src_ptr, int src_stride, char src_op)
	{
		assert(ml::cpu::is_aligned(dst_ptr, register_size<YMM>()));

		uint64_t k_iter = K / 8;
		uint64_t k_left = K % 8;
		const uint64_t SrcStride = src_stride * sizeof(float);

		if (src_op == 'n')
		{
			asm volatile(
					"movq %[src_ptr], %%rax \n\t" // src pointer is in rax
					"movq %[dst_ptr], %%rbx \n\t"// dst pointer is in rbx
					"movq %[SrcStride], %%r12 \n\t"// src stride is in r12

					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je FINALLOOP%= \n\t"

					"UNROLLED8%=: \n\t"
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"vmovups 0x20(%%rax), %%ymm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm2 \n\t"
					"vmovups 0x20(%%rax), %%ymm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm4 \n\t"
					"vmovups 0x20(%%rax), %%ymm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm6 \n\t"
					"vmovups 0x20(%%rax), %%ymm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm8 \n\t"
					"vmovups 0x20(%%rax), %%ymm9 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm10 \n\t"
					"vmovups 0x20(%%rax), %%ymm11 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm12 \n\t"
					"vmovups 0x20(%%rax), %%ymm13 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x00(%%rax), %%ymm14 \n\t"
					"vmovups 0x20(%%rax), %%ymm15 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

					"vmovaps %%ymm0, 0x000(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x020(%%rbx) \n\t"
					"vmovaps %%ymm2, 0x040(%%rbx) \n\t"
					"vmovaps %%ymm3, 0x060(%%rbx) \n\t"
					"vmovaps %%ymm4, 0x080(%%rbx) \n\t"
					"vmovaps %%ymm5, 0x0A0(%%rbx) \n\t"
					"vmovaps %%ymm6, 0x0C0(%%rbx) \n\t"
					"vmovaps %%ymm7, 0x0E0(%%rbx) \n\t"
					"vmovaps %%ymm8, 0x100(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x120(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x140(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x160(%%rbx) \n\t"
					"vmovaps %%ymm12, 0x180(%%rbx) \n\t"
					"vmovaps %%ymm13, 0x1A0(%%rbx) \n\t"
					"vmovaps %%ymm14, 0x1C0(%%rbx) \n\t"
					"vmovaps %%ymm15, 0x1E0(%%rbx) \n\t"

					"add $0x200, %%rbx \n\t"// add stride to dst pointer (8 * 4 * 16)

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"vmovups 0x20(%%rax), %%ymm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovaps %%ymm0, 0x00(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x20(%%rbx) \n\t"
					"add $0x40, %%rbx \n\t"// add stride to dst pointer (1 * 4 * 16)

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
					[SrcStride] "m"(SrcStride)
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
					"movq %[SrcStride], %%r12 \n\t"// src stride is in r12

					"movq %[k_iter], %%r14 \n\t"// load the number of 8-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je FINALLOOP%= \n\t"

					"UNROLLED8%=: \n\t"
					// first 8x8 tile
					"vmovups 0x0(%%rax), %%ymm0 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x0(%%rax), %%ymm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x0(%%rax), %%ymm2 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x0(%%rax), %%ymm3 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x0(%%rax), %%ymm4 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x0(%%rax), %%ymm5 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x0(%%rax), %%ymm6 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovups 0x0(%%rax), %%ymm7 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer

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

					"vmovaps %%ymm8, 0x000(%%rbx) \n\t"
					"vmovaps %%ymm9, 0x040(%%rbx) \n\t"
					"vmovaps %%ymm10, 0x080(%%rbx) \n\t"
					"vmovaps %%ymm11, 0x0C0(%%rbx) \n\t"
					"vmovaps %%ymm12, 0x100(%%rbx) \n\t"
					"vmovaps %%ymm13, 0x140(%%rbx) \n\t"
					"vmovaps %%ymm14, 0x180(%%rbx) \n\t"
					"vmovaps %%ymm15, 0x1C0(%%rbx) \n\t"

					"add $0x200, %%rbx \n\t"// add stride to dst pointer (8 * 4 * 16)

					"dec %%r14 \n\t"
					"jne UNROLLED8%= \n\t"

					"FINALLOOP%=: \n\t"
					"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
					"test %%r14, %%r14 \n\t"
					"je EPILOGUE%= \n\t"

					"UNROLLED1%=: \n\t"
					"vmovups 0x00(%%rax), %%ymm0 \n\t"
					"vmovups 0x20(%%rax), %%ymm1 \n\t"
					"add %%r12, %%rax \n\t"// add stride to src pointer
					"vmovaps %%ymm0, 0x00(%%rbx) \n\t"
					"vmovaps %%ymm1, 0x20(%%rbx) \n\t"
					"add $0x40, %%rbx \n\t"// add stride to dst pointer (1 * 4 * 16)

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
					[SrcStride] "m"(SrcStride)
					:// clobbers
					"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
					"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
					"%r12", "%r14");
		}
	}
}

namespace ml
{
	void gemm_avx2_fma_6x16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.stride() == 6);
		assert(B.stride() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
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

		asm volatile(
				"movq %[A_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx

				// Set accumulators to zero.
				"vpxor %%ymm4, %%ymm4, %%ymm4 \n\t"
				"vpxor %%ymm5, %%ymm5, %%ymm5 \n\t"
				"vpxor %%ymm6, %%ymm6, %%ymm6 \n\t"
				"vpxor %%ymm7, %%ymm7, %%ymm7 \n\t"
				"vpxor %%ymm8, %%ymm8, %%ymm8 \n\t"
				"vpxor %%ymm9, %%ymm9, %%ymm9 \n\t"
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x40(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x60(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x18(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x20(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x28(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x80(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xA0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x38(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x40(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0xC0(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xE0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x48(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x50(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x54(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x58(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x5C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				"add $0x60, %%rax \n\t"
				"add $0x100, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"
				"add $0x18, %%rax \n\t"
				"add $0x40, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm0, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm0, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm0, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm0, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm0, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"ucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm4 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm6 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm7 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm8 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm9 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm10 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm11 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm13 \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm14 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm15 \n\t"

				"BETAZERO%=: \n\t"
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx

				"vmovups %%ymm4, 0x00(%%rcx) \n\t"
				"vmovups %%ymm5, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups %%ymm6, 0x00(%%rcx) \n\t"
				"vmovups %%ymm7, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups %%ymm8, 0x00(%%rcx) \n\t"
				"vmovups %%ymm9, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups %%ymm10, 0x00(%%rcx) \n\t"
				"vmovups %%ymm11, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups %%ymm12, 0x00(%%rcx) \n\t"
				"vmovups %%ymm13, 0x20(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vmovups %%ymm14, 0x00(%%rcx) \n\t"
				"vmovups %%ymm15, 0x20(%%rcx) \n\t"

				"vzeroupper \n\t"

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
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14");
	}
	void gemm_avx2_fma_6x8_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 6);
		assert(B.columns() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
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

		asm volatile(
				"movq %[A_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx

				// Set accumulators to zero.
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x20(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x18(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x20(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x28(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x40(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x38(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x40(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0x60(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x48(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x50(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x54(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x58(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x5C(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				"add $0x60, %%rax \n\t"
				"add $0x80, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"
				"add $0x18, %%rax \n\t"
				"add $0x20, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"ucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				"movq %[C_stride], %%r14 \n\t"// C stride is r14
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"vmovups 0x0(%%rcx), %%ymm2 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm3 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm4 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm5 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm6 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm7 \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups 0x0(%%rcx), %%ymm8 \n\t"

				"vfmadd231ps %%ymm2, %%ymm1, %%ymm10 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm11 \n\t"
				"vfmadd231ps %%ymm4, %%ymm1, %%ymm12 \n\t"
				"vfmadd231ps %%ymm5, %%ymm1, %%ymm13 \n\t"
				"vfmadd231ps %%ymm6, %%ymm1, %%ymm14 \n\t"
				"vfmadd231ps %%ymm7, %%ymm1, %%ymm15 \n\t"

				"BETAZERO%=: \n\t"
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx

				"vmovups %%ymm10, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm11, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm12, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm13, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm14, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"vmovups %%ymm15, 0x0(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride

				"vzeroupper \n\t"

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
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14");
	}

	void gemm_avx2_fma_6x16_fp16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 6);
		assert(B.columns() == 16);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
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
		float alpha = reinterpret_cast<const float*>(alpha_ptr)[0];
		float beta = reinterpret_cast<const float*>(beta_ptr)[0];
		if (beta != 0.0f)
			alpha /= beta;
		const float *_alpha = &alpha;

		asm volatile(
				"movq %[beta_ptr], %%rbx \n\t" // load address of beta
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"ucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je ZEROACC%= \n\t"
				// load and convert dst
				"movq %[C_stride], %%r12 \n\t"// C stride is r12
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"movups 0x10(%%rcx), %%xmm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"movups 0x10(%%rcx), %%xmm7 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm8 \n\t"
				"movups 0x10(%%rcx), %%xmm9 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm10 \n\t"
				"movups 0x10(%%rcx), %%xmm11 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm12 \n\t"
				"movups 0x10(%%rcx), %%xmm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm14 \n\t"
				"movups 0x10(%%rcx), %%xmm15 \n\t"

				"vcvtph2ps %%xmm4, %%ymm4 \n\t"
				"vcvtph2ps %%xmm5, %%ymm5 \n\t"
				"vcvtph2ps %%xmm6, %%ymm6 \n\t"
				"vcvtph2ps %%xmm7, %%ymm7 \n\t"
				"vcvtph2ps %%xmm8, %%ymm8 \n\t"
				"vcvtph2ps %%xmm9, %%ymm9 \n\t"
				"vcvtph2ps %%xmm10, %%ymm10 \n\t"
				"vcvtph2ps %%xmm11, %%ymm11 \n\t"
				"vcvtph2ps %%xmm12, %%ymm12 \n\t"
				"vcvtph2ps %%xmm13, %%ymm13 \n\t"
				"vcvtph2ps %%xmm14, %%ymm14 \n\t"
				"vcvtph2ps %%xmm15, %%ymm15 \n\t"

				"vmulps %%ymm1, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm1, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm1, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm1, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm1, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm1, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm1, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm1, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm1, %%ymm15, %%ymm15 \n\t"
				"jmp LOOPSTART%= \n\t"

				"ZEROACC%=: \n\t"
				// Set accumulators to zero.
				"vpxor %%ymm4, %%ymm4, %%ymm4 \n\t"
				"vpxor %%ymm5, %%ymm5, %%ymm5 \n\t"
				"vpxor %%ymm6, %%ymm6, %%ymm6 \n\t"
				"vpxor %%ymm7, %%ymm7, %%ymm7 \n\t"
				"vpxor %%ymm8, %%ymm8, %%ymm8 \n\t"
				"vpxor %%ymm9, %%ymm9, %%ymm9 \n\t"
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"LOOPSTART%=: \n\t"

				"movq %[A_ptr], %%rax \n\t"// lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx
				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0

				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x40(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x60(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x18(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x20(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x28(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x80(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xA0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x38(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x40(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0xC0(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xE0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x48(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x50(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x54(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x58(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x5C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				"add $0x60, %%rax \n\t"
				"add $0x100, %%rbx \n\t"
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"
				"add $0x18, %%rax \n\t"
				"add $0x40, %%rbx \n\t"

				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm0, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm0, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm0, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm0, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm0, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"movq %[D_stride], %%r12 \n\t"// D stride is r12
				"movq %[D_ptr], %%rdx \n\t"// D pointer is in rdx

				"vcvtps2ph $0x03, %%ymm4, %%xmm4 \n\t"
				"vcvtps2ph $0x03, %%ymm5, %%xmm5 \n\t"
				"vcvtps2ph $0x03, %%ymm6, %%xmm6 \n\t"
				"vcvtps2ph $0x03, %%ymm7, %%xmm7 \n\t"
				"vcvtps2ph $0x03, %%ymm8, %%xmm8 \n\t"
				"vcvtps2ph $0x03, %%ymm9, %%xmm9 \n\t"
				"vcvtps2ph $0x03, %%ymm10, %%xmm10 \n\t"
				"vcvtps2ph $0x03, %%ymm11, %%xmm11 \n\t"
				"vcvtps2ph $0x03, %%ymm12, %%xmm12 \n\t"
				"vcvtps2ph $0x03, %%ymm13, %%xmm13 \n\t"
				"vcvtps2ph $0x03, %%ymm14, %%xmm14 \n\t"
				"vcvtps2ph $0x03, %%ymm15, %%xmm15 \n\t"

				"movups %%xmm4, 0x00(%%rdx) \n\t"
				"movups %%xmm5, 0x10(%%rdx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm6, 0x00(%%rdx) \n\t"
				"movups %%xmm7, 0x10(%%rdx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm8, 0x00(%%rdx) \n\t"
				"movups %%xmm9, 0x10(%%rdx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm10, 0x00(%%rdx) \n\t"
				"movups %%xmm11, 0x10(%%rdx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm12, 0x00(%%rdx) \n\t"
				"movups %%xmm13, 0x10(%%rdx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm14, 0x00(%%rdx) \n\t"
				"movups %%xmm15, 0x10(%%rdx) \n\t"

				"vzeroupper \n\t"

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
				[alpha_ptr] "m"(_alpha),
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx","%rdx",
				"%r12", "%r13", "%r14");
	}
	void gemm_avx2_fma_6x8_fp16_fp32(Fragment &D, const void *alpha_ptr, const Fragment &A, const Fragment &B, const void *beta_ptr,
			const Fragment &C) noexcept
	{
		assert(A.rows() == B.rows());
		assert(A.columns() == 6);
		assert(B.columns() == 8);
		assert(D.rows() == A.columns());
		assert(D.columns() == B.columns());

		assert(alpha_ptr != nullptr);
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
		float alpha = reinterpret_cast<const float*>(alpha_ptr)[0];
		float beta = reinterpret_cast<const float*>(beta_ptr)[0];
		if (beta != 0.0f)
			alpha /= beta;
		const float *_alpha = &alpha;

		asm volatile(
				"movq %[beta_ptr], %%rbx \n\t" // load address of beta
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"ucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je ZEROACC%= \n\t"
				// load and convert dst
				"movq %[C_stride], %%r12 \n\t"// C stride is r12
				"movq %[C_ptr], %%rcx \n\t"// C pointer is in rcx

				"movups 0x00(%%rcx), %%xmm10 \n\t"
				"movups 0x10(%%rcx), %%xmm11 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm12 \n\t"
				"movups 0x10(%%rcx), %%xmm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm14 \n\t"
				"movups 0x10(%%rcx), %%xmm15 \n\t"

				"vcvtph2ps %%xmm10, %%ymm10 \n\t"
				"vcvtph2ps %%xmm11, %%ymm11 \n\t"
				"vcvtph2ps %%xmm12, %%ymm12 \n\t"
				"vcvtph2ps %%xmm13, %%ymm13 \n\t"
				"vcvtph2ps %%xmm14, %%ymm14 \n\t"
				"vcvtph2ps %%xmm15, %%ymm15 \n\t"

				"vmulps %%ymm1, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm1, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm1, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm1, %%ymm15, %%ymm15 \n\t"
				"jmp LOOPSTART%= \n\t"

				"ZEROACC%=: \n\t"
				// Set accumulators to zero.
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"LOOPSTART%=: \n\t"

				"movq %[A_ptr], %%rax \n\t"// lhs pointer is in rax
				"movq %[B_ptr], %%rbx \n\t"// rhs pointer is in rbx
				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x20(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x18(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x20(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x28(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x40(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x38(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x40(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0x60(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x48(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x50(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x54(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x58(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x5C(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"

				"add $0x60, %%rax \n\t"
				"add $0x80, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm6 \n\t"
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm4 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm5 \n\t"
				"vfmadd231ps %%ymm0, %%ymm6, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm6, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm6, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm6, %%ymm13 \n\t"
				"vfmadd231ps %%ymm4, %%ymm6, %%ymm14 \n\t"
				"vfmadd231ps %%ymm5, %%ymm6, %%ymm15 \n\t"
				"add $0x18, %%rax \n\t"
				"add $0x20, %%rbx \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"movq %[D_stride], %%r14 \n\t"// D stride is r14
				"movq %[D_ptr], %%rcx \n\t"// D pointer is in rcx

				"vcvtps2ph $0x03, %%ymm10, %%xmm10 \n\t"
				"vcvtps2ph $0x03, %%ymm11, %%xmm11 \n\t"
				"vcvtps2ph $0x03, %%ymm12, %%xmm12 \n\t"
				"vcvtps2ph $0x03, %%ymm13, %%xmm13 \n\t"
				"vcvtps2ph $0x03, %%ymm14, %%xmm14 \n\t"
				"vcvtps2ph $0x03, %%ymm15, %%xmm15 \n\t"

				"movups %%xmm10, 0x00(%%rcx) \n\t"
				"movups %%xmm11, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm12, 0x00(%%rcx) \n\t"
				"movups %%xmm13, 0x10(%%rcx) \n\t"
				"add %%r14, %%rcx \n\t"// add stride
				"movups %%xmm14, 0x00(%%rcx) \n\t"
				"movups %%xmm15, 0x10(%%rcx) \n\t"

				"vzeroupper \n\t"

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
				[alpha_ptr] "m"(_alpha),
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx", "%r14");
	}

	void pack_avx2_6xK_fp32(int M, int K, const void *__restrict__ src, int src_stride, void *__restrict__ dst, char op)
	{
		assert(M == 6);
		assert(K >= 0);
		assert(src != nullptr);
		assert(dst != nullptr && cpu::is_aligned(dst, 32));
		assert(op == 'n' || op == 't');

		if (op == 'n')
		{
			for (int k = 0; k < K; k++)
				for (int m = 0; m < 6; m++)
				{

				}
		}
		else
		{

		}
	}

	void pack_avx2_16xK_fp32(Fragment &dst, const Fragment &src, int src_row, int src_col, char src_op) noexcept
	{
		assert(src_op == 'n' || src_op == 't');

		const int M = dst.columns();
		const int K = dst.rows();

		const float *src_ptr = src.data<float>() + src.offset_at(src_row, src_col);
		float *dst_ptr = dst.data<float>();

		if (src_op == 'n')
		{ // pack K x M fragment
			const int max_rows = std::min(K, src.rows() - src_row);
			const int max_cols = std::min(M, src.columns() - src_col);

			if (max_cols == 16)
			{ // fragment can be loaded fully
				for (int k = 0; k < max_rows; k++)
				{
					_mm256_storeu_ps(dst_ptr, _mm256_loadu_ps(src_ptr));
					_mm256_storeu_ps(dst_ptr + 8, _mm256_loadu_ps(src_ptr + 8));
					src_ptr += src.stride();
					dst_ptr += dst.stride();
				}
			}
			else
			{
				if (max_cols >= 8)
				{ // at least the first half of a fragment can be loaded fully
					const int partial_size = max_cols - 8;
					const __m256i mask = _mm256_castps_si256(SIMD_NAMESPACE::get_cutoff_mask_ps(partial_size));
					for (int k = 0; k < max_rows; k++)
					{
						_mm256_storeu_ps(dst_ptr, _mm256_loadu_ps(src_ptr));
						_mm256_maskstore_ps(dst_ptr + 8, mask, _mm256_maskload_ps(src_ptr + 8, mask));
						src_ptr += src.stride();
						dst_ptr += dst.stride();
					}
				}
				else
				{
					const int partial_size = max_cols;
					const __m256i mask = _mm256_castps_si256(SIMD_NAMESPACE::get_cutoff_mask_ps(partial_size));
					for (int k = 0; k < max_rows; k++)
					{
						_mm256_maskstore_ps(dst_ptr, mask, _mm256_maskload_ps(src_ptr, mask));
						src_ptr += src.stride();
						dst_ptr += dst.stride();
					}
				}
			}
		}
		else
		{ // pack M x K fragment
			const int max_rows = std::min(M, src.rows() - src_row);
			const int max_cols = std::min(K, src.columns() - src_col);

			const int full_K = max_cols - max_cols % 16;
			if (max_rows == 16)
			{

			}
		}
	}

	void pack_avx2_16xK_fp32(int M, int K, const void *__restrict__ src, int src_stride, void *__restrict__ dst, char op)
	{
		assert(M == 16);
		assert(K >= 0);
		assert(src != nullptr);
		assert(dst != nullptr && cpu::is_aligned(dst, 32));
		assert(op == 'n' || op == 't');
	}

} /* namespace ml */

