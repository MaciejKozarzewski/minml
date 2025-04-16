/*
 * sse2_kernels.cpp
 *
 *  Created on: Feb 10, 2025
 *      Author: Maciej Kozarzewski
 */

#include "TensorFragment.hpp"
#include "../utils.hpp"
#include "../fp16.hpp"

#include <cinttypes>
#include <cassert>

#include "../src/backend/cpu/assembly_macros.hpp"

#define STORE_ACCUMULATORS_FP32(ptr) \
	movups(xmm0, mem(ptr, 0*4*4)) \
	movups(xmm1, mem(ptr, 1*4*4)) \
	movups(xmm2, mem(ptr, 2*4*4)) \
	movups(xmm3, mem(ptr, 3*4*4)) \
	movups(xmm4, mem(ptr, 4*4*4)) \
	movups(xmm5, mem(ptr, 5*4*4)) \
	movups(xmm6, mem(ptr, 6*4*4)) \
	movups(xmm7, mem(ptr, 7*4*4))

namespace ml
{
	void average_pooling_sse2_1x32xfp32(const TensorFragment &input, TensorFragment &output) noexcept
	{
		assert(input.is_fp32());
		assert(output.is_fp32());
		assert(input.columns() >= 32);
		assert(output.columns() == input.columns());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();

		const float inv_m = 1.0f / input.rows();
		const void *inv_ptr = &inv_m;

		begin_asm()
		xorps(xmm0, xmm0)
		xorps(xmm1, xmm1)
		xorps(xmm2, xmm2)
		xorps(xmm3, xmm3)
		xorps(xmm4, xmm4)
		xorps(xmm5, xmm5)
		xorps(xmm6, xmm6)
		xorps(xmm7, xmm7)

		movq(var(input_ptr), rax) // input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED)
		movups(mem(rax, 0*4*4), xmm8)
		movups(mem(rax, 1*4*4), xmm9)
		movups(mem(rax, 2*4*4), xmm10)
		movups(mem(rax, 3*4*4), xmm11)
		movups(mem(rax, 4*4*4), xmm12)
		movups(mem(rax, 5*4*4), xmm13)
		movups(mem(rax, 6*4*4), xmm14)
		movups(mem(rax, 7*4*4), xmm15)

		addps(xmm8, xmm0)
		addps(xmm9, xmm1)
		addps(xmm10, xmm2)
		addps(xmm11, xmm3)
		addps(xmm12, xmm4)
		addps(xmm13, xmm5)
		addps(xmm14, xmm6)
		addps(xmm15, xmm7)

		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		movss(mem(r13), xmm8)
		pshufd(imm(0x00), xmm8, xmm8)
		mulps(xmm8, xmm0)
		mulps(xmm8, xmm1)
		mulps(xmm8, xmm2)
		mulps(xmm8, xmm3)
		mulps(xmm8, xmm4)
		mulps(xmm8, xmm5)
		mulps(xmm8, xmm6)
		mulps(xmm8, xmm7)

		STORE_ACCUMULATORS_FP32(rbx)

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride),
				[inv_ptr] "m"(inv_ptr)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void average_pooling_sse2_1x4xfp32(const TensorFragment &input, TensorFragment &output) noexcept
	{
		assert(input.is_fp32());
		assert(output.is_fp32());
		assert(input.columns() >= 4);
		assert(output.columns() == input.columns());
		assert(input.stride() == output.stride());
		const uint64_t m_iter = input.rows();
		const uint64_t stride = input.stride_in_bytes();

		const void *input_ptr = input.data();
		void *output_ptr = output.data();

		const float inv_m = 1.0f / input.rows();
		const void *inv_ptr = &inv_m;

		begin_asm()
		xorps(xmm0, xmm0)

		movq(var(input_ptr), rax) // input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED)
		movups(mem(rax), xmm8)
		addps(xmm8, xmm0)
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		movss(mem(r13), xmm8)
		pshufd(imm(0x00), xmm8, xmm8)
		mulps(xmm8, xmm0)

		movups(xmm0, mem(rbx))

		end_asm(
				:// outputs
				:// inputs
				[input_ptr] "m"(input_ptr),
				[output_ptr] "m"(output_ptr),
				[m_iter] "m"(m_iter),
				[stride] "m"(stride),
				[inv_ptr] "m"(inv_ptr)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}

	void channel_scaling_sse2_1x32xfp32(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept
	{
		assert(input.is_fp32());
		assert(output.is_fp32());
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
		movups(mem(rax, 0*4*4), xmm8)
		movups(mem(rax, 1*4*4), xmm9)
		movups(mem(rax, 2*4*4), xmm10)
		movups(mem(rax, 3*4*4), xmm11)
		movups(mem(rax, 4*4*4), xmm12)
		movups(mem(rax, 5*4*4), xmm13)
		movups(mem(rax, 6*4*4), xmm14)
		movups(mem(rax, 7*4*4), xmm15)

		movq(var(input_ptr), rax)// input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED)
		movups(mem(rax, 0*8*4), xmm0)
		movups(mem(rax, 1*8*4), xmm1)
		movups(mem(rax, 2*8*4), xmm2)
		movups(mem(rax, 3*8*4), xmm3)
		movups(mem(rax, 4*8*4), xmm4)
		movups(mem(rax, 5*8*4), xmm5)
		movups(mem(rax, 6*8*4), xmm6)
		movups(mem(rax, 7*8*4), xmm7)

		mulps(xmm8, xmm0)
		mulps(xmm9, xmm1)
		mulps(xmm10, xmm2)
		mulps(xmm11, xmm3)
		mulps(xmm12, xmm4)
		mulps(xmm13, xmm5)
		mulps(xmm14, xmm6)
		mulps(xmm15, xmm7)

		STORE_ACCUMULATORS_FP32(rbx)

		add(r12, rax)// add stride to input pointer
		add(r12, rbx)// add stride to output pointer

		dec(r14)
		jne(UNROLLED)

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
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void channel_scaling_sse2_1x4xfp32(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) noexcept
	{
		assert(input.is_fp32());
		assert(output.is_fp32());
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
		movups(mem(rax), xmm8)

		movq(var(input_ptr), rax)// input pointer is in rax
		movq(var(output_ptr), rbx)// output pointer is in rbx
		movq(var(stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED)
		movups(mem(rax), xmm0)
		mulps(xmm8, xmm0)
		movups(xmm0, mem(rbx))

		add(r12, rax)// add stride to input pointer
		add(r12, rbx)// add stride to output pointer
		dec(r14)
		jne(UNROLLED)

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
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}

} /* namespace ml */

