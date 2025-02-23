/*
 * avx_gemm_kernels.cpp
 *
 *  Created on: May 5, 2023
 *      Author: Maciej Kozarzewski
 */

#include "TensorFragment.hpp"
#include "../utils.hpp"
#include "../fp16.hpp"

#include <cinttypes>
#include <cassert>

#include "../src/backend/cpu/assembly_macros.hpp"

#define ZERO_ACCUMULATORS() \
	vxorps(ymm0, ymm0, ymm0) \
	vxorps(ymm1, ymm1, ymm1) \
	vxorps(ymm2, ymm2, ymm2) \
	vxorps(ymm3, ymm3, ymm3) \
	vxorps(ymm4, ymm4, ymm4) \
	vxorps(ymm5, ymm5, ymm5) \
	vxorps(ymm6, ymm6, ymm6) \
	vxorps(ymm7, ymm7, ymm7)
#define ACCUMULATE_8x8xFP32() \
	vaddps(ymm8, ymm0, ymm0) \
	vaddps(ymm9, ymm1, ymm1) \
	vaddps(ymm10, ymm2, ymm2) \
	vaddps(ymm11, ymm3, ymm3) \
	vaddps(ymm12, ymm4, ymm4) \
	vaddps(ymm13, ymm5, ymm5) \
	vaddps(ymm14, ymm6, ymm6) \
	vaddps(ymm15, ymm7, ymm7)
#define STORE_ACCUMULATORS_8x8xFP32(ptr) \
	vmovups(ymm0, mem(ptr, 0*8*4)) \
	vmovups(ymm1, mem(ptr, 1*8*4)) \
	vmovups(ymm2, mem(ptr, 2*8*4)) \
	vmovups(ymm3, mem(ptr, 3*8*4)) \
	vmovups(ymm4, mem(ptr, 4*8*4)) \
	vmovups(ymm5, mem(ptr, 5*8*4)) \
	vmovups(ymm6, mem(ptr, 6*8*4)) \
	vmovups(ymm7, mem(ptr, 7*8*4))
#define STORE_ACCUMULATORS_8x8xFP16(ptr) \
	vcvtps2ph(imm(0x03), ymm0, xmm0) \
	vcvtps2ph(imm(0x03), ymm1, xmm1) \
	vcvtps2ph(imm(0x03), ymm2, xmm2) \
	vcvtps2ph(imm(0x03), ymm3, xmm3) \
	vcvtps2ph(imm(0x03), ymm4, xmm4) \
	vcvtps2ph(imm(0x03), ymm5, xmm5) \
	vcvtps2ph(imm(0x03), ymm6, xmm6) \
	vcvtps2ph(imm(0x03), ymm7, xmm7) \
	vmovups(xmm0, mem(ptr, 0*8*2)) \
	vmovups(xmm1, mem(ptr, 1*8*2)) \
	vmovups(xmm2, mem(ptr, 2*8*2)) \
	vmovups(xmm3, mem(ptr, 3*8*2)) \
	vmovups(xmm4, mem(ptr, 4*8*2)) \
	vmovups(xmm5, mem(ptr, 5*8*2)) \
	vmovups(xmm6, mem(ptr, 6*8*2)) \
	vmovups(xmm7, mem(ptr, 7*8*2))

#define DIVIDE_BY_M_FP32(reg) \
	vmulps(reg, ymm0, ymm0) \
	vmulps(reg, ymm1, ymm1) \
	vmulps(reg, ymm2, ymm2) \
	vmulps(reg, ymm3, ymm3) \
	vmulps(reg, ymm4, ymm4) \
	vmulps(reg, ymm5, ymm5) \
	vmulps(reg, ymm6, ymm6) \
	vmulps(reg, ymm7, ymm7)

namespace ml
{
	void convert_fp32_to_fp16_avx(void *dst, const void *src, size_t elements) noexcept
	{
		const uint64_t kx128 = elements / 128;
		const uint64_t kx8 = (elements % 128) / 8;
		const uint64_t kx1 = elements % 8;

		begin_asm()
		movq(var(src), rax)
		movq(var(dst), rbx)

		movq(var(kx128), r15)
		test(r15, r15)
		je(MIDDLE_LOOP)

		label(UNROLLED_x128)
		vmovups(mem(rax, 0*4*8), ymm0)
		vmovups(mem(rax, 1*4*8), ymm1)
		vmovups(mem(rax, 2*4*8), ymm2)
		vmovups(mem(rax, 3*4*8), ymm3)
		vmovups(mem(rax, 4*4*8), ymm4)
		vmovups(mem(rax, 5*4*8), ymm5)
		vmovups(mem(rax, 6*4*8), ymm6)
		vmovups(mem(rax, 7*4*8), ymm7)
		vmovups(mem(rax, 8*4*8), ymm8)
		vmovups(mem(rax, 9*4*8), ymm9)
		vmovups(mem(rax, 10*4*8), ymm10)
		vmovups(mem(rax, 11*4*8), ymm11)
		vmovups(mem(rax, 12*4*8), ymm12)
		vmovups(mem(rax, 13*4*8), ymm13)
		vmovups(mem(rax, 14*4*8), ymm14)
		vmovups(mem(rax, 15*4*8), ymm15)
		vcvtps2ph(imm(0x03), ymm0, xmm0)
		vcvtps2ph(imm(0x03), ymm1, xmm1)
		vcvtps2ph(imm(0x03), ymm2, xmm2)
		vcvtps2ph(imm(0x03), ymm3, xmm3)
		vcvtps2ph(imm(0x03), ymm4, xmm4)
		vcvtps2ph(imm(0x03), ymm5, xmm5)
		vcvtps2ph(imm(0x03), ymm6, xmm6)
		vcvtps2ph(imm(0x03), ymm7, xmm7)
		vcvtps2ph(imm(0x03), ymm8, xmm8)
		vcvtps2ph(imm(0x03), ymm9, xmm9)
		vcvtps2ph(imm(0x03), ymm10, xmm10)
		vcvtps2ph(imm(0x03), ymm11, xmm11)
		vcvtps2ph(imm(0x03), ymm12, xmm12)
		vcvtps2ph(imm(0x03), ymm13, xmm13)
		vcvtps2ph(imm(0x03), ymm14, xmm14)
		vcvtps2ph(imm(0x03), ymm15, xmm15)
		vmovups(xmm0, mem(rbx, 0*8*2))
		vmovups(xmm1, mem(rbx, 1*8*2))
		vmovups(xmm2, mem(rbx, 2*8*2))
		vmovups(xmm3, mem(rbx, 3*8*2))
		vmovups(xmm4, mem(rbx, 4*8*2))
		vmovups(xmm5, mem(rbx, 5*8*2))
		vmovups(xmm6, mem(rbx, 6*8*2))
		vmovups(xmm7, mem(rbx, 7*8*2))
		vmovups(xmm8, mem(rbx, 8*8*2))
		vmovups(xmm9, mem(rbx, 9*8*2))
		vmovups(xmm10, mem(rbx, 10*8*2))
		vmovups(xmm11, mem(rbx, 11*8*2))
		vmovups(xmm12, mem(rbx, 12*8*2))
		vmovups(xmm13, mem(rbx, 13*8*2))
		vmovups(xmm14, mem(rbx, 14*8*2))
		vmovups(xmm15, mem(rbx, 15*8*2))

		add(imm(4*128), rax)
		add(imm(2*128), rbx)
		dec(r15)
		jne(UNROLLED_x128)

		label(MIDDLE_LOOP)
		movq(var(kx8), r15)
		test(r15, r15)
		je(FINAL_LOOP)

		label(UNROLLED_x8)
		vmovups(mem(rax), ymm0)
		vcvtps2ph(imm(0x03), ymm0, xmm0)
		vmovups(xmm0, mem(rbx))
		add(imm(4*8), rax)
		add(imm(2*8), rbx)
		dec(r15)
		jne(UNROLLED_x128)

		label(FINAL_LOOP)
		movq(var(kx1), r15)
		test(r15, r15)
		je(EPILOGUE)

		label(UNROLLED_x1)
		vmovss(mem(rax), xmm0)
		vcvtps2ph(imm(0x03), xmm0, xmm0)
		vmovq(xmm0, rcx)
		mov(cx, mem(rbx))

		add(imm(4), rax)
		add(imm(2), rbx)
		dec(r15)
		jne(UNROLLED_x1)

		label(EPILOGUE)
		vzeroupper()
		end_asm(
				: // outputs
				:// inputs
				[src] "m"(src),
				[dst] "m"(dst),
				[kx128] "m"(kx128),
				[kx8] "m"(kx8),
				[kx1] "m"(kx1)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15",
				"%rax", "%rbx", "%r13", "%r14", "%r15")
	}
	void convert_fp16_to_fp32_avx(void *dst, const void *src, size_t elements) noexcept
	{
		const uint64_t kx128 = elements / 128;
		const uint64_t kx8 = (elements % 128) / 8;
		const uint64_t kx1 = elements % 8;

		begin_asm()
		movq(var(src), rax)
		movq(var(dst), rbx)

		movq(var(kx128), r15)
		test(r15, r15)
		je(MIDDLE_LOOP)

		label(UNROLLED_x128)
		vmovups(mem(rax, 0*2*8), xmm0)
		vmovups(mem(rax, 1*2*8), xmm1)
		vmovups(mem(rax, 2*2*8), xmm2)
		vmovups(mem(rax, 3*2*8), xmm3)
		vmovups(mem(rax, 4*2*8), xmm4)
		vmovups(mem(rax, 5*2*8), xmm5)
		vmovups(mem(rax, 6*2*8), xmm6)
		vmovups(mem(rax, 7*2*8), xmm7)
		vmovups(mem(rax, 8*2*8), xmm8)
		vmovups(mem(rax, 9*2*8), xmm9)
		vmovups(mem(rax, 10*2*8), xmm10)
		vmovups(mem(rax, 11*2*8), xmm11)
		vmovups(mem(rax, 12*2*8), xmm12)
		vmovups(mem(rax, 13*2*8), xmm13)
		vmovups(mem(rax, 14*2*8), xmm14)
		vmovups(mem(rax, 15*2*8), xmm15)
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
		vcvtph2ps(xmm10, ymm10)
		vcvtph2ps(xmm11, ymm11)
		vcvtph2ps(xmm12, ymm12)
		vcvtph2ps(xmm13, ymm13)
		vcvtph2ps(xmm14, ymm14)
		vcvtph2ps(xmm15, ymm15)
		vmovups(ymm0, mem(rbx, 0*8*4))
		vmovups(ymm1, mem(rbx, 1*8*4))
		vmovups(ymm2, mem(rbx, 2*8*4))
		vmovups(ymm3, mem(rbx, 3*8*4))
		vmovups(ymm4, mem(rbx, 4*8*4))
		vmovups(ymm5, mem(rbx, 5*8*4))
		vmovups(ymm6, mem(rbx, 6*8*4))
		vmovups(ymm7, mem(rbx, 7*8*4))
		vmovups(ymm8, mem(rbx, 8*8*4))
		vmovups(ymm9, mem(rbx, 9*8*4))
		vmovups(ymm10, mem(rbx, 10*8*4))
		vmovups(ymm11, mem(rbx, 11*8*4))
		vmovups(ymm12, mem(rbx, 12*8*4))
		vmovups(ymm13, mem(rbx, 13*8*4))
		vmovups(ymm14, mem(rbx, 14*8*4))
		vmovups(ymm15, mem(rbx, 15*8*4))

		add(imm(2*128), rax)
		add(imm(4*128), rbx)
		dec(r15)
		jne(UNROLLED_x128)

		label(MIDDLE_LOOP)
		movq(var(kx8), r15)
		test(r15, r15)
		je(FINAL_LOOP)

		label(UNROLLED_x8)
		vmovups(mem(rax), xmm0)
		vcvtph2ps(xmm0, ymm0)
		vmovups(ymm0, mem(rbx))
		add(imm(2*8), rax)
		add(imm(4*8), rbx)
		dec(r15)
		jne(UNROLLED_x128)

		label(FINAL_LOOP)
		movq(var(kx1), r15)
		test(r15, r15)
		je(EPILOGUE)

		label(UNROLLED_x1)
		vmovq(rcx, xmm0)
		vcvtph2ps(xmm0, xmm0)
		vmovss(xmm0, mem(rbx))
		mov(cx, mem(rbx))

		add(imm(2), rax)
		add(imm(4), rbx)
		dec(r15)
		jne(UNROLLED_x1)

		label(EPILOGUE)
		vzeroupper()
		end_asm(
				: // outputs
				:// inputs
				[src] "m"(src),
				[dst] "m"(dst),
				[kx128] "m"(kx128),
				[kx8] "m"(kx8),
				[kx1] "m"(kx1)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15",
				"%rax", "%rbx", "%r13", "%r14", "%r15")
	}

	void average_pooling_avx_1x64xfp16(const TensorFragment &input, TensorFragment &output) noexcept
	{
		assert(input.is_fp16());
		assert(output.is_fp16() || output.is_fp32());
		assert(input.columns() >= 64);
		assert(output.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();

		const uint64_t output_in_fp32 = output.is_fp32();
		const float inv_m = 1.0f / input.rows();
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input_ptr), rax) // input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax, 0*8*2), xmm8)
		vmovups(mem(rax, 1*8*2), xmm9)
		vmovups(mem(rax, 2*8*2), xmm10)
		vmovups(mem(rax, 3*8*2), xmm11)
		vmovups(mem(rax, 4*8*2), xmm12)
		vmovups(mem(rax, 5*8*2), xmm13)
		vmovups(mem(rax, 6*8*2), xmm14)
		vmovups(mem(rax, 7*8*2), xmm15)

		vcvtph2ps(xmm8, ymm8)
		vcvtph2ps(xmm9, ymm9)
		vcvtph2ps(xmm10, ymm10)
		vcvtph2ps(xmm11, ymm11)
		vcvtph2ps(xmm12, ymm12)
		vcvtph2ps(xmm13, ymm13)
		vcvtph2ps(xmm14, ymm14)
		vcvtph2ps(xmm15, ymm15)

		ACCUMULATE_8x8xFP32()
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		vbroadcastss(mem(r13), ymm8)
		DIVIDE_BY_M_FP32(ymm8)

		movq(var(output_in_fp32), r12)
		test(r12, r12)
		je(OUTPUT_IN_FP16)
		STORE_ACCUMULATORS_8x8xFP32(rbx)
		jmp(END)

		label(OUTPUT_IN_FP16)
		STORE_ACCUMULATORS_8x8xFP16(rbx)

		label(END)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride),
				[inv_ptr] "m"(inv_ptr),
				[output_in_fp32] "m"(output_in_fp32)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void average_pooling_avx_1x64xfp32(const TensorFragment &input, TensorFragment &output) noexcept
	{
		assert(input.is_fp32());
		assert(output.is_fp16() || output.is_fp32());
		assert(input.columns() >= 64);
		assert(output.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();

		const uint64_t output_in_fp32 = output.is_fp32();
		const float inv_m = 1.0f / input.rows();
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input_ptr), rax) // input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax, 0*8*4), ymm8)
		vmovups(mem(rax, 1*8*4), ymm9)
		vmovups(mem(rax, 2*8*4), ymm10)
		vmovups(mem(rax, 3*8*4), ymm11)
		vmovups(mem(rax, 4*8*4), ymm12)
		vmovups(mem(rax, 5*8*4), ymm13)
		vmovups(mem(rax, 6*8*4), ymm14)
		vmovups(mem(rax, 7*8*4), ymm15)

		ACCUMULATE_8x8xFP32()
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		vbroadcastss(mem(r13), ymm8)
		DIVIDE_BY_M_FP32(ymm8)

		movq(var(output_in_fp32), r12)
		test(r12, r12)
		je(OUTPUT_IN_FP16)
		STORE_ACCUMULATORS_8x8xFP32(rbx)
		jmp(END)

		label(OUTPUT_IN_FP16)
		STORE_ACCUMULATORS_8x8xFP16(rbx)

		label(END)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride),
				[inv_ptr] "m"(inv_ptr),
				[output_in_fp32] "m"(output_in_fp32)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void average_pooling_avx_1x32xfp64(const TensorFragment &input, TensorFragment &output) noexcept
	{
		assert(input.is_fp64());
		assert(output.is_fp64());
		assert(input.columns() >= 32);
		assert(output.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();

		const double inv_m = 1.0 / input.rows();
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input_ptr), rax) // input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovupd(mem(rax, 0*4*8), ymm8)
		vmovupd(mem(rax, 1*4*8), ymm9)
		vmovupd(mem(rax, 2*4*8), ymm10)
		vmovupd(mem(rax, 3*4*8), ymm11)
		vmovupd(mem(rax, 4*4*8), ymm12)
		vmovupd(mem(rax, 5*4*8), ymm13)
		vmovupd(mem(rax, 6*4*8), ymm14)
		vmovupd(mem(rax, 7*4*8), ymm15)

		vaddpd(ymm8, ymm0, ymm0)
		vaddpd(ymm9, ymm1, ymm1)
		vaddpd(ymm10, ymm2, ymm2)
		vaddpd(ymm11, ymm3, ymm3)
		vaddpd(ymm12, ymm4, ymm4)
		vaddpd(ymm13, ymm5, ymm5)
		vaddpd(ymm14, ymm6, ymm6)
		vaddpd(ymm15, ymm7, ymm7)
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		vbroadcastsd(mem(r13), ymm8)
		vmulpd(ymm8, ymm0, ymm0)
		vmulpd(ymm8, ymm1, ymm1)
		vmulpd(ymm8, ymm2, ymm2)
		vmulpd(ymm8, ymm3, ymm3)
		vmulpd(ymm8, ymm4, ymm4)
		vmulpd(ymm8, ymm5, ymm5)
		vmulpd(ymm8, ymm6, ymm6)
		vmulpd(ymm8, ymm7, ymm7)

		vmovupd(ymm0, mem(rbx, 0*4*8))
		vmovupd(ymm1, mem(rbx, 1*4*8))
		vmovupd(ymm2, mem(rbx, 2*4*8))
		vmovupd(ymm3, mem(rbx, 3*4*8))
		vmovupd(ymm4, mem(rbx, 4*4*8))
		vmovupd(ymm5, mem(rbx, 5*4*8))
		vmovupd(ymm6, mem(rbx, 6*4*8))
		vmovupd(ymm7, mem(rbx, 7*4*8))

		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride),
				[inv_ptr] "m"(inv_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}

	void average_pooling_avx_1x8xfp16(const TensorFragment &input, TensorFragment &output) noexcept
	{
		assert(input.is_fp16());
		assert(output.is_fp16() || output.is_fp32());
		assert(input.columns() >= 64);
		assert(output.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();

		const uint64_t output_in_fp32 = output.is_fp32();
		const float inv_m = 1.0f / input.rows();
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input_ptr), rax) // input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax), xmm8)
		vcvtph2ps(xmm8, ymm8)
		vaddps(ymm8, ymm0, ymm0)
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		vbroadcastss(mem(r13), ymm8)
		vmulps(ymm8, ymm0, ymm0)

		movq(var(output_in_fp32), r12)
		test(r12, r12)
		je(OUTPUT_IN_FP16)
		vmovups(ymm0, mem(rbx))
		jmp(END)

		label(OUTPUT_IN_FP16)
		vcvtps2ph(imm(0x03), ymm0, xmm0)
		vmovups(xmm0, mem(rbx))

		label(END)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride),
				[inv_ptr] "m"(inv_ptr),
				[output_in_fp32] "m"(output_in_fp32)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void average_pooling_avx_1x8xfp32(const TensorFragment &input, TensorFragment &output) noexcept
	{
		assert(input.is_fp32());
		assert(output.is_fp16() || output.is_fp32());
		assert(input.columns() >= 64);
		assert(output.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();

		const uint64_t output_in_fp32 = output.is_fp32();
		const float inv_m = 1.0f / input.rows();
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input_ptr), rax) // input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax), ymm8)
		vaddps(ymm8, ymm0, ymm0)
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		vbroadcastss(mem(r13), ymm8)
		vmulps(ymm8, ymm0, ymm0)

		movq(var(output_in_fp32), r12)
		test(r12, r12)
		je(OUTPUT_IN_FP16)
		vmovups(ymm0, mem(rbx))
		jmp(END)

		label(OUTPUT_IN_FP16)
		vcvtps2ph(imm(0x03), ymm0, xmm0)
		vmovups(xmm0, mem(rbx))

		label(END)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride),
				[inv_ptr] "m"(inv_ptr),
				[output_in_fp32] "m"(output_in_fp32)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void average_pooling_avx_1x4xfp64(const TensorFragment &input, TensorFragment &output) noexcept
	{
		assert(input.is_fp64());
		assert(output.is_fp64());
		assert(input.columns() >= 4);
		assert(output.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();

		const double inv_m = 1.0 / input.rows();
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input_ptr), rax) // input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovupd(mem(rax), ymm8)
		vaddpd(ymm8, ymm0, ymm0)
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		vbroadcastsd(mem(r13), ymm8)
		vmulpd(ymm8, ymm0, ymm0)

		vmovupd(ymm0, mem(rbx))
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride),
				[inv_ptr] "m"(inv_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}

	void channel_scaling_avx_1x64xfp16(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept
	{
		assert(input.is_fp16());
		assert(output.is_fp16());
		assert(input.columns() >= 64);
		assert(output.columns() == input.columns());
		assert(scales.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();
		const void *scales_ptr = scales.data();

		begin_asm()
		movq(var(scales_ptr), rax) // input pointer is in rax
		vmovups(mem(rax, 0*8*2), xmm8)
		vmovups(mem(rax, 1*8*2), xmm9)
		vmovups(mem(rax, 2*8*2), xmm10)
		vmovups(mem(rax, 3*8*2), xmm11)
		vmovups(mem(rax, 4*8*2), xmm12)
		vmovups(mem(rax, 5*8*2), xmm13)
		vmovups(mem(rax, 6*8*2), xmm14)
		vmovups(mem(rax, 7*8*2), xmm15)

		vcvtph2ps(xmm8, ymm8)
		vcvtph2ps(xmm9, ymm9)
		vcvtph2ps(xmm10, ymm10)
		vcvtph2ps(xmm11, ymm11)
		vcvtph2ps(xmm12, ymm12)
		vcvtph2ps(xmm13, ymm13)
		vcvtph2ps(xmm14, ymm14)
		vcvtph2ps(xmm15, ymm15)

		movq(var(input_ptr), rax)// input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax, 0*8*2), xmm0)
		vmovups(mem(rax, 1*8*2), xmm1)
		vmovups(mem(rax, 2*8*2), xmm2)
		vmovups(mem(rax, 3*8*2), xmm3)
		vmovups(mem(rax, 4*8*2), xmm4)
		vmovups(mem(rax, 5*8*2), xmm5)
		vmovups(mem(rax, 6*8*2), xmm6)
		vmovups(mem(rax, 7*8*2), xmm7)

		vcvtph2ps(xmm0, ymm0)
		vcvtph2ps(xmm1, ymm1)
		vcvtph2ps(xmm2, ymm2)
		vcvtph2ps(xmm3, ymm3)
		vcvtph2ps(xmm4, ymm4)
		vcvtph2ps(xmm5, ymm5)
		vcvtph2ps(xmm6, ymm6)
		vcvtph2ps(xmm7, ymm7)

		vmulps(ymm0, ymm8, ymm0)
		vmulps(ymm1, ymm9, ymm1)
		vmulps(ymm2, ymm10, ymm2)
		vmulps(ymm3, ymm11, ymm3)
		vmulps(ymm4, ymm12, ymm4)
		vmulps(ymm5, ymm13, ymm5)
		vmulps(ymm6, ymm14, ymm6)
		vmulps(ymm7, ymm15, ymm7)

		STORE_ACCUMULATORS_8x8xFP16(rbx)

		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[scales_ptr] "m"(scales_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void channel_scaling_avx_1x64xfp32(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept
	{
		assert(input.is_fp32());
		assert(output.is_fp32());
		assert(input.columns() >= 64);
		assert(output.columns() == input.columns());
		assert(scales.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();
		const void *scales_ptr = scales.data();

		begin_asm()
		movq(var(scales_ptr), rax) // input pointer is in rax
		vmovups(mem(rax, 0*8*4), ymm8)
		vmovups(mem(rax, 1*8*4), ymm9)
		vmovups(mem(rax, 2*8*4), ymm10)
		vmovups(mem(rax, 3*8*4), ymm11)
		vmovups(mem(rax, 4*8*4), ymm12)
		vmovups(mem(rax, 5*8*4), ymm13)
		vmovups(mem(rax, 6*8*4), ymm14)
		vmovups(mem(rax, 7*8*4), ymm15)

		movq(var(input_ptr), rax)// input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax, 0*8*4), ymm0)
		vmovups(mem(rax, 1*8*4), ymm1)
		vmovups(mem(rax, 2*8*4), ymm2)
		vmovups(mem(rax, 3*8*4), ymm3)
		vmovups(mem(rax, 4*8*4), ymm4)
		vmovups(mem(rax, 5*8*4), ymm5)
		vmovups(mem(rax, 6*8*4), ymm6)
		vmovups(mem(rax, 7*8*4), ymm7)

		vmulps(ymm0, ymm8, ymm0)
		vmulps(ymm1, ymm9, ymm1)
		vmulps(ymm2, ymm10, ymm2)
		vmulps(ymm3, ymm11, ymm3)
		vmulps(ymm4, ymm12, ymm4)
		vmulps(ymm5, ymm13, ymm5)
		vmulps(ymm6, ymm14, ymm6)
		vmulps(ymm7, ymm15, ymm7)

		STORE_ACCUMULATORS_8x8xFP32(rbx)

		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[scales_ptr] "m"(scales_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void channel_scaling_avx_1x32xfp64(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept
	{
		assert(input.is_fp64());
		assert(output.is_fp64());
		assert(input.columns() >= 32);
		assert(output.columns() == input.columns());
		assert(scales.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();
		const void *scales_ptr = scales.data();

		begin_asm()
		movq(var(scales_ptr), rax) // input pointer is in rax
		vmovupd(mem(rax, 0*4*8), ymm8)
		vmovupd(mem(rax, 1*4*8), ymm9)
		vmovupd(mem(rax, 2*4*8), ymm10)
		vmovupd(mem(rax, 3*4*8), ymm11)
		vmovupd(mem(rax, 4*4*8), ymm12)
		vmovupd(mem(rax, 5*4*8), ymm13)
		vmovupd(mem(rax, 6*4*8), ymm14)
		vmovupd(mem(rax, 7*4*8), ymm15)

		movq(var(input_ptr), rax)// input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovupd(mem(rax, 0*4*8), ymm0)
		vmovupd(mem(rax, 1*4*8), ymm1)
		vmovupd(mem(rax, 2*4*8), ymm2)
		vmovupd(mem(rax, 3*4*8), ymm3)
		vmovupd(mem(rax, 4*4*8), ymm4)
		vmovupd(mem(rax, 5*4*8), ymm5)
		vmovupd(mem(rax, 6*4*8), ymm6)
		vmovupd(mem(rax, 7*4*8), ymm7)

		vmulpd(ymm0, ymm8, ymm0)
		vmulpd(ymm1, ymm9, ymm1)
		vmulpd(ymm2, ymm10, ymm2)
		vmulpd(ymm3, ymm11, ymm3)
		vmulpd(ymm4, ymm12, ymm4)
		vmulpd(ymm5, ymm13, ymm5)
		vmulpd(ymm6, ymm14, ymm6)
		vmulpd(ymm7, ymm15, ymm7)

		vmovupd(ymm0, mem(rbx, 0*4*8))
		vmovupd(ymm1, mem(rbx, 1*4*8))
		vmovupd(ymm2, mem(rbx, 2*4*8))
		vmovupd(ymm3, mem(rbx, 3*4*8))
		vmovupd(ymm4, mem(rbx, 4*4*8))
		vmovupd(ymm5, mem(rbx, 5*4*8))
		vmovupd(ymm6, mem(rbx, 6*4*8))
		vmovupd(ymm7, mem(rbx, 7*4*8))

		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[scales_ptr] "m"(scales_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}

	void channel_scaling_avx_1x8xfp16(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept
	{
		assert(input.is_fp16());
		assert(output.is_fp16());
		assert(input.columns() >= 64);
		assert(output.columns() == input.columns());
		assert(scales.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();
		const void *scales_ptr = scales.data();

		begin_asm()
		movq(var(scales_ptr), rax) // input pointer is in rax
		vmovups(mem(rax), xmm8)
		vcvtph2ps(xmm8, ymm8)

		movq(var(input_ptr), rax)// input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax), xmm0)
		vcvtph2ps(xmm0, ymm0)
		vmulps(ymm0, ymm8, ymm0)
		vcvtps2ph(imm(0x03), ymm0, xmm0)
		vmovups(xmm0, mem(rbx))
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[scales_ptr] "m"(scales_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void channel_scaling_avx_1x8xfp32(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept
	{
		assert(input.is_fp32());
		assert(output.is_fp32());
		assert(input.columns() >= 8);
		assert(output.columns() == input.columns());
		assert(scales.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();
		const void *scales_ptr = scales.data();

		begin_asm()
		movq(var(scales_ptr), rax) // input pointer is in rax
		vmovups(mem(rax), ymm8)

		movq(var(input_ptr), rax)// input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax), ymm0)
		vmulps(ymm0, ymm8, ymm0)
		vmovups(ymm0, mem(rbx))

		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[scales_ptr] "m"(scales_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void channel_scaling_avx_1x4xfp64(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept
	{
		assert(input.is_fp64());
		assert(output.is_fp64());
		assert(input.columns() >= 4);
		assert(output.columns() == input.columns());
		assert(scales.columns() == input.columns());
		assert(input.rows() == output.rows());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();
		const void *scales_ptr = scales.data();

		begin_asm()
		movq(var(scales_ptr), rax) // input pointer is in rax
		vmovupd(mem(rax), ymm8)

		movq(var(input_ptr), rax)// input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovupd(mem(rax), ymm0)
		vmulpd(ymm0, ymm8, ymm0)
		vmovupd(ymm0, mem(rbx))
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[scales_ptr] "m"(scales_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}

} /* namespace ml */
