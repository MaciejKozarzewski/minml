/*
 * avx2_kernels.cpp
 *
 *  Created on: Mar 24, 2026
 *      Author: Maciej Kozarzewski
 */

#include "TensorFragment.hpp"
#include "../utils.hpp"

#include <cinttypes>
#include <chrono>
#include <cassert>

#include "../src/backend/cpu/assembly_macros.hpp"

namespace
{
	double get_time()
	{
		auto current_time = std::chrono::system_clock::now();
		auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());

		return duration_in_seconds.count();
	}
}

namespace ml
{
	double measure_timing_fp32_fma(int repeats)
	{
		const uint64_t m_repeats = (repeats + 63) / 64;
		const double start = get_time();

		begin_asm()
		movq(var(m_repeats), r15) // number of repeats is in r15
		test(r15, r15)
		je(EPILOGUE)

		label(UNROLLED)
		vfmadd231ps(ymm0, ymm1, ymm2)// first 16 iterations
		vfmadd231ps(ymm0, ymm1, ymm3)
		vfmadd231ps(ymm0, ymm1, ymm4)
		vfmadd231ps(ymm0, ymm1, ymm5)
		vfmadd231ps(ymm0, ymm1, ymm6)
		vfmadd231ps(ymm0, ymm1, ymm7)
		vfmadd231ps(ymm0, ymm1, ymm8)
		vfmadd231ps(ymm0, ymm1, ymm9)
		vfmadd231ps(ymm0, ymm1, ymm10)
		vfmadd231ps(ymm0, ymm1, ymm11)
		vfmadd231ps(ymm0, ymm1, ymm12)
		vfmadd231ps(ymm0, ymm1, ymm13)
		vfmadd231ps(ymm0, ymm1, ymm14)
		vfmadd231ps(ymm0, ymm1, ymm15)
		vfmadd231ps(ymm0, ymm1, ymm2)
		vfmadd231ps(ymm0, ymm1, ymm3)
		vfmadd231ps(ymm0, ymm1, ymm4)
		vfmadd231ps(ymm0, ymm1, ymm5)

		vfmadd231ps(ymm0, ymm1, ymm2)// second 16 iterations
		vfmadd231ps(ymm0, ymm1, ymm3)
		vfmadd231ps(ymm0, ymm1, ymm4)
		vfmadd231ps(ymm0, ymm1, ymm5)
		vfmadd231ps(ymm0, ymm1, ymm6)
		vfmadd231ps(ymm0, ymm1, ymm7)
		vfmadd231ps(ymm0, ymm1, ymm8)
		vfmadd231ps(ymm0, ymm1, ymm9)
		vfmadd231ps(ymm0, ymm1, ymm10)
		vfmadd231ps(ymm0, ymm1, ymm11)
		vfmadd231ps(ymm0, ymm1, ymm12)
		vfmadd231ps(ymm0, ymm1, ymm13)
		vfmadd231ps(ymm0, ymm1, ymm14)
		vfmadd231ps(ymm0, ymm1, ymm15)
		vfmadd231ps(ymm0, ymm1, ymm2)
		vfmadd231ps(ymm0, ymm1, ymm3)
		vfmadd231ps(ymm0, ymm1, ymm4)
		vfmadd231ps(ymm0, ymm1, ymm5)

		vfmadd231ps(ymm0, ymm1, ymm2)// third 16 iterations
		vfmadd231ps(ymm0, ymm1, ymm3)
		vfmadd231ps(ymm0, ymm1, ymm4)
		vfmadd231ps(ymm0, ymm1, ymm5)
		vfmadd231ps(ymm0, ymm1, ymm6)
		vfmadd231ps(ymm0, ymm1, ymm7)
		vfmadd231ps(ymm0, ymm1, ymm8)
		vfmadd231ps(ymm0, ymm1, ymm9)
		vfmadd231ps(ymm0, ymm1, ymm10)
		vfmadd231ps(ymm0, ymm1, ymm11)
		vfmadd231ps(ymm0, ymm1, ymm12)
		vfmadd231ps(ymm0, ymm1, ymm13)
		vfmadd231ps(ymm0, ymm1, ymm14)
		vfmadd231ps(ymm0, ymm1, ymm15)
		vfmadd231ps(ymm0, ymm1, ymm2)
		vfmadd231ps(ymm0, ymm1, ymm3)
		vfmadd231ps(ymm0, ymm1, ymm4)
		vfmadd231ps(ymm0, ymm1, ymm5)

		vfmadd231ps(ymm0, ymm1, ymm2)// fourth 16 iterations
		vfmadd231ps(ymm0, ymm1, ymm3)
		vfmadd231ps(ymm0, ymm1, ymm4)
		vfmadd231ps(ymm0, ymm1, ymm5)
		vfmadd231ps(ymm0, ymm1, ymm6)
		vfmadd231ps(ymm0, ymm1, ymm7)
		vfmadd231ps(ymm0, ymm1, ymm8)
		vfmadd231ps(ymm0, ymm1, ymm9)
		vfmadd231ps(ymm0, ymm1, ymm10)
		vfmadd231ps(ymm0, ymm1, ymm11)
		vfmadd231ps(ymm0, ymm1, ymm12)
		vfmadd231ps(ymm0, ymm1, ymm13)
		vfmadd231ps(ymm0, ymm1, ymm14)
		vfmadd231ps(ymm0, ymm1, ymm15)
		vfmadd231ps(ymm0, ymm1, ymm2)
		vfmadd231ps(ymm0, ymm1, ymm3)
		vfmadd231ps(ymm0, ymm1, ymm4)
		vfmadd231ps(ymm0, ymm1, ymm5)

		dec(r15)
		jne(UNROLLED)

		label(EPILOGUE)
		vzeroupper()
		end_asm(
				:// outputs
				:// inputs
				[m_repeats] "m"(m_repeats)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%r15")

		const double stop = get_time();
		return stop - start;
	}
	double measure_timing_int8_fma(int repeats)
	{
		const uint64_t m_repeats = (repeats + 63) / 64;
		const double start = get_time();

		begin_asm()
		movq(var(m_repeats), r15) // number of repeats is in r15
		test(r15, r15)
		je(EPILOGUE)

		label(UNROLLED)
		vpmaddwd(ymm0, ymm1, ymm2)// first 16 iterations
		vpmaddwd(ymm0, ymm1, ymm3)
		vpmaddwd(ymm0, ymm1, ymm4)
		vpmaddwd(ymm0, ymm1, ymm5)
		vpmaddwd(ymm0, ymm1, ymm6)
		vpmaddwd(ymm0, ymm1, ymm7)
		vpmaddwd(ymm0, ymm1, ymm8)
		vpmaddwd(ymm0, ymm1, ymm9)
		vpmaddwd(ymm0, ymm1, ymm10)
		vpmaddwd(ymm0, ymm1, ymm11)
		vpmaddwd(ymm0, ymm1, ymm12)
		vpmaddwd(ymm0, ymm1, ymm13)
		vpmaddwd(ymm0, ymm1, ymm14)
		vpmaddwd(ymm0, ymm1, ymm15)
		vpmaddwd(ymm0, ymm1, ymm2)
		vpmaddwd(ymm0, ymm1, ymm3)
		vpmaddwd(ymm0, ymm1, ymm4)
		vpmaddwd(ymm0, ymm1, ymm5)

		vpmaddwd(ymm0, ymm1, ymm2)// second 16 iterations
		vpmaddwd(ymm0, ymm1, ymm3)
		vpmaddwd(ymm0, ymm1, ymm4)
		vpmaddwd(ymm0, ymm1, ymm5)
		vpmaddwd(ymm0, ymm1, ymm6)
		vpmaddwd(ymm0, ymm1, ymm7)
		vpmaddwd(ymm0, ymm1, ymm8)
		vpmaddwd(ymm0, ymm1, ymm9)
		vpmaddwd(ymm0, ymm1, ymm10)
		vpmaddwd(ymm0, ymm1, ymm11)
		vpmaddwd(ymm0, ymm1, ymm12)
		vpmaddwd(ymm0, ymm1, ymm13)
		vpmaddwd(ymm0, ymm1, ymm14)
		vpmaddwd(ymm0, ymm1, ymm15)
		vpmaddwd(ymm0, ymm1, ymm2)
		vpmaddwd(ymm0, ymm1, ymm3)
		vpmaddwd(ymm0, ymm1, ymm4)
		vpmaddwd(ymm0, ymm1, ymm5)

		vpmaddwd(ymm0, ymm1, ymm2)// third 16 iterations
		vpmaddwd(ymm0, ymm1, ymm3)
		vpmaddwd(ymm0, ymm1, ymm4)
		vpmaddwd(ymm0, ymm1, ymm5)
		vpmaddwd(ymm0, ymm1, ymm6)
		vpmaddwd(ymm0, ymm1, ymm7)
		vpmaddwd(ymm0, ymm1, ymm8)
		vpmaddwd(ymm0, ymm1, ymm9)
		vpmaddwd(ymm0, ymm1, ymm10)
		vpmaddwd(ymm0, ymm1, ymm11)
		vpmaddwd(ymm0, ymm1, ymm12)
		vpmaddwd(ymm0, ymm1, ymm13)
		vpmaddwd(ymm0, ymm1, ymm14)
		vpmaddwd(ymm0, ymm1, ymm15)
		vpmaddwd(ymm0, ymm1, ymm2)
		vpmaddwd(ymm0, ymm1, ymm3)
		vpmaddwd(ymm0, ymm1, ymm4)
		vpmaddwd(ymm0, ymm1, ymm5)

		vpmaddwd(ymm0, ymm1, ymm2)// fourth 16 iterations
		vpmaddwd(ymm0, ymm1, ymm3)
		vpmaddwd(ymm0, ymm1, ymm4)
		vpmaddwd(ymm0, ymm1, ymm5)
		vpmaddwd(ymm0, ymm1, ymm6)
		vpmaddwd(ymm0, ymm1, ymm7)
		vpmaddwd(ymm0, ymm1, ymm8)
		vpmaddwd(ymm0, ymm1, ymm9)
		vpmaddwd(ymm0, ymm1, ymm10)
		vpmaddwd(ymm0, ymm1, ymm11)
		vpmaddwd(ymm0, ymm1, ymm12)
		vpmaddwd(ymm0, ymm1, ymm13)
		vpmaddwd(ymm0, ymm1, ymm14)
		vpmaddwd(ymm0, ymm1, ymm15)
		vpmaddwd(ymm0, ymm1, ymm2)
		vpmaddwd(ymm0, ymm1, ymm3)
		vpmaddwd(ymm0, ymm1, ymm4)
		vpmaddwd(ymm0, ymm1, ymm5)

		dec(r15)
		jne(UNROLLED)

		label(EPILOGUE)
		vzeroupper()
		end_asm(
				:// outputs
				:// inputs
				[m_repeats] "m"(m_repeats)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%r15")

		const double stop = get_time();
		return stop - start;
	}

} /* namespace ml */
