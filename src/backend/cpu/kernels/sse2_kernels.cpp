/*
 * sse2_kernels.cpp
 *
 *  Created on: Feb 10, 2025
 *      Author: Maciej Kozarzewski
 */

#include "../utils.hpp"
#include "../fp16.hpp"

#include <cinttypes>
#include <cassert>

#include "../src/backend/cpu/assembly_macros.hpp"

#define ZERO_ACCUMULATORS() \
	xorps(xmm0, xmm0) \
	xorps(xmm1, xmm1) \
	xorps(xmm2, xmm2) \
	xorps(xmm3, xmm3) \
	xorps(xmm4, xmm4) \
	xorps(xmm5, xmm5) \
	xorps(xmm6, xmm6) \
	xorps(xmm7, xmm7)
#define ACCUMULATE_FP32() \
	addps(xmm8, xmm0) \
	addps(xmm9, xmm1) \
	addps(xmm10, xmm2) \
	addps(xmm11, xmm3) \
	addps(xmm12, xmm4) \
	addps(xmm13, xmm5) \
	addps(xmm14, xmm6) \
	addps(xmm15, xmm7)
#define STORE_ACCUMULATORS_FP32(ptr) \
	movups(xmm0, mem(ptr, 0*4*4)) \
	movups(xmm1, mem(ptr, 1*4*4)) \
	movups(xmm2, mem(ptr, 2*4*4)) \
	movups(xmm3, mem(ptr, 3*4*4)) \
	movups(xmm4, mem(ptr, 4*4*4)) \
	movups(xmm5, mem(ptr, 5*4*4)) \
	movups(xmm6, mem(ptr, 6*4*4)) \
	movups(xmm7, mem(ptr, 7*4*4))

#define DIVIDE_BY_M_FP32(reg) \
	mulps(reg, xmm0) \
	mulps(reg, xmm1) \
	mulps(reg, xmm2) \
	mulps(reg, xmm3) \
	mulps(reg, xmm4) \
	mulps(reg, xmm5) \
	mulps(reg, xmm6) \
	mulps(reg, xmm7)

namespace ml
{
	void average_pooling_sse2_1x32xfp32(const void *input, mlDataType_t input_dtype, void *output, mlDataType_t output_dtype, int stride, int rows,
			int columns) noexcept
	{
		assert(input_dtype == DTYPE_FLOAT32);
		assert(output_dtype == DTYPE_FLOAT32);
		assert(columns >= 32);
		const uint64_t m_iter = rows;
		const uint64_t input_stride = stride;
		const float inv_m = 1.0f / rows;
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input), rax) // input pointer is in rax
		movq(var(output), rbx)// output pointer is in rbx
		movq(var(input_stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax, 0*4*4), xmm8)
		vmovups(mem(rax, 1*4*4), xmm9)
		vmovups(mem(rax, 2*4*4), xmm10)
		vmovups(mem(rax, 3*4*4), xmm11)
		vmovups(mem(rax, 4*4*4), xmm12)
		vmovups(mem(rax, 5*4*4), xmm13)
		vmovups(mem(rax, 6*4*4), xmm14)
		vmovups(mem(rax, 7*4*4), xmm15)

		ACCUMULATE_FP32()
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		movss(mem(r13), xmm8)
		pshufd(imm(0x00), xmm8, xmm8)
		DIVIDE_BY_M_FP32(xmm8)

		STORE_ACCUMULATORS_FP32(rbx)

		end_asm(
				:// outputs
				:// inputs
				[input] "m"(input),
				[output] "m"(output),
				[m_iter] "m"(m_iter),
				[input_stride] "m"(input_stride),
				[inv_ptr] "m"(inv_ptr)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void average_pooling_sse2_1x16xfp64(const void *input, mlDataType_t input_dtype, void *output, mlDataType_t output_dtype, int stride, int rows,
			int columns) noexcept
	{
		assert(input_dtype == DTYPE_FLOAT64);
		assert(output_dtype == DTYPE_FLOAT64);
		assert(columns >= 16);
		const uint64_t m_iter = rows;
		const uint64_t input_stride = stride;
		const float inv_m = 1.0f / rows;
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input), rax) // input pointer is in rax
		movq(var(output), rbx)// output pointer is in rbx
		movq(var(input_stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		movupd(mem(rax, 0*2*8), xmm8)
		movupd(mem(rax, 1*2*8), xmm9)
		movupd(mem(rax, 2*2*8), xmm10)
		movupd(mem(rax, 3*2*8), xmm11)
		movupd(mem(rax, 4*2*8), xmm12)
		movupd(mem(rax, 5*2*8), xmm13)
		movupd(mem(rax, 6*2*8), xmm14)
		movupd(mem(rax, 7*2*8), xmm15)

		addpd(xmm8, xmm0)
		addpd(xmm9, xmm1)
		addpd(xmm10, xmm2)
		addpd(xmm11, xmm3)
		addpd(xmm12, xmm4)
		addpd(xmm13, xmm5)
		addpd(xmm14, xmm6)
		addpd(xmm15, xmm7)
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		movsd(mem(r13), xmm8)
		shufpd(imm(0x00), xmm8, xmm8)
		mulpd(xmm8, xmm0)
		mulpd(xmm8, xmm1)
		mulpd(xmm8, xmm2)
		mulpd(xmm8, xmm3)
		mulpd(xmm8, xmm4)
		mulpd(xmm8, xmm5)
		mulpd(xmm8, xmm6)
		mulpd(xmm8, xmm7)

		movupd(xmm0, mem(rbx, 0*2*8))
		movupd(xmm1, mem(rbx, 1*2*8))
		movupd(xmm2, mem(rbx, 2*2*8))
		movupd(xmm3, mem(rbx, 3*2*8))
		movupd(xmm4, mem(rbx, 4*2*8))
		movupd(xmm5, mem(rbx, 5*2*8))
		movupd(xmm6, mem(rbx, 6*2*8))
		movupd(xmm7, mem(rbx, 7*2*8))

		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input] "m"(input),
				[output] "m"(output),
				[m_iter] "m"(m_iter),
				[input_stride] "m"(input_stride),
				[inv_ptr] "m"(inv_ptr)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}

	void average_pooling_sse2_1x4xfp32(const void *input, mlDataType_t input_dtype, void *output, mlDataType_t output_dtype, int stride, int rows,
			int columns) noexcept
	{
		assert(input_dtype == DTYPE_FLOAT32);
		assert(output_dtype == DTYPE_FLOAT32);
		assert(columns >= 4);
		const uint64_t m_iter = rows;
		const uint64_t input_stride = stride;
		const float inv_m = 1.0f / rows;
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input), rax) // input pointer is in rax
		movq(var(output), rbx)// output pointer is in rbx
		movq(var(input_stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		vmovups(mem(rax), xmm8)
		addps(xmm8, xmm0)
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		movss(mem(r13), xmm8)
		pshufd(imm(0x00), xmm8, xmm8)
		mulps(xmm8, xmm0)

		movups(xmm0, mem(rbx))

		end_asm(
				:// outputs
				:// inputs
				[input] "m"(input),
				[output] "m"(output),
				[m_iter] "m"(m_iter),
				[input_stride] "m"(input_stride),
				[inv_ptr] "m"(inv_ptr)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}
	void average_pooling_sse2_1x2xfp64(const void *input, mlDataType_t input_dtype, void *output, mlDataType_t output_dtype, int stride, int rows,
			int columns) noexcept
	{
		assert(input_dtype == DTYPE_FLOAT64);
		assert(output_dtype == DTYPE_FLOAT64);
		assert(columns >= 2);
		const uint64_t m_iter = rows;
		const uint64_t input_stride = stride;
		const float inv_m = 1.0f / rows;
		const void *inv_ptr = &inv_m;

		begin_asm()
		ZERO_ACCUMULATORS()

		movq(var(input), rax) // input pointer is in rax
		movq(var(output), rbx)// output pointer is in rbx
		movq(var(input_stride), r12)// input stride is in r12

		movq(var(m_iter), r14)// load the number of 4-unrolled iterations
		test(r14, r14)
		je(EPILOGUE)

		label(UNROLLED4)
		movupd(mem(rax), xmm8)
		addpd(xmm8, xmm0)
		add(r12, rax)// add stride to input pointer

		dec(r14)
		jne(UNROLLED4)

		label(EPILOGUE)
		movq(var(inv_ptr), r13)
		movsd(mem(r13), xmm8)
		shufpd(imm(0x00), xmm8, xmm8)
		mulpd(xmm8, xmm0)

		movupd(xmm0, mem(rbx))

		vzeroupper()

		end_asm(
				:// outputs
				:// inputs
				[input] "m"(input),
				[output] "m"(output),
				[m_iter] "m"(m_iter),
				[input_stride] "m"(input_stride),
				[inv_ptr] "m"(inv_ptr)
				:// clobbers
				"cc", "memory", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
				"%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%rax", "%rbx",
				"%r12", "%r13", "%r14", "%r15")
	}

} /* namespace ml */

