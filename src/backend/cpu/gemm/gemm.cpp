/*
 * gemm.cpp
 *
 *  Created on: May 10, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include "../utils.hpp"
#include "Fragment.hpp"
#include "Matrix.hpp"
#include "utilities.hpp"
#include "gemm_kernels.hpp"
#include "gemm_runtime.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace
{
	using namespace ml;

	std::vector<GemmRuntime> get_sse2_gemm_runtime()
	{
		std::vector<GemmRuntime> result(1);
		// 4x8
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 4, 8, 256 };
		result[0].gemm_kernel = gemm_sse2_4x8_fp32;
		result[0].a_packing = pack_sse2_4xK_fp32;
		result[0].b_packing = pack_sse2_8xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(15.8, 14.2);

		return result;
	}
	std::vector<GemmRuntime> get_avx_gemm_runtime()
	{
		std::vector<GemmRuntime> result(2);
		// 10x8 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 10, 8, 256 };
		result[0].gemm_kernel = gemm_avx_10x8_fp32;
		result[0].a_packing = pack_avx_10xK_fp32;
		result[0].b_packing = pack_avx_8xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(31.6, 16.7);

		// 10x8 fp16/fp32
		result[1].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[1].inner_tile = { 10, 8, 512 };
		result[1].gemm_kernel = gemm_avx_10x8_fp32_fp16;
		result[1].a_packing = pack_avx_10xK_fp16_fp32;
		result[1].b_packing = pack_avx_8xK_fp16_fp32;
		result[1].c_packing = pack_def_MxK_fp16;
		result[1].d_packing = pack_def_MxK_fp16;
		result[1].d_unpacking = unpack_def_MxK_fp16;
		result[1].edge_a_packing = pack_def_MxK_fp16_fp32;
		result[1].edge_b_packing = pack_def_MxK_fp16_fp32;
		result[1].perf_estimator = PerfEstimator(31.4, 14.6);

		return result;
	}
	std::vector<GemmRuntime> get_avx2_fma_gemm_runtime()
	{
		std::vector<GemmRuntime> result(2);

		// 12x8 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 12, 8, 1024 };
		result[0].gemm_kernel = gemm_avx2_fma_12x8_fp32;
		result[0].a_packing = pack_avx2_fma_12xK_fp32;
		result[0].b_packing = pack_avx_8xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(62.8, 23.4);

		// 12x8 fp16/fp32
		result[1].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[1].inner_tile = { 12, 8, 1024 };
		result[1].gemm_kernel = gemm_avx2_fma_12x8_fp32_fp16;
		result[1].a_packing = pack_avx2_fma_12xK_fp16_fp32;
		result[1].b_packing = pack_avx_8xK_fp16_fp32;
		result[1].c_packing = pack_def_MxK_fp16;
		result[1].d_packing = pack_def_MxK_fp16;
		result[1].d_unpacking = unpack_def_MxK_fp16;
		result[1].edge_a_packing = pack_def_MxK_fp16_fp32;
		result[1].edge_b_packing = pack_def_MxK_fp16_fp32;
		result[1].perf_estimator = PerfEstimator(61.2, 25.5);

		return result;
	}
	std::vector<GemmRuntime> get_avx512f_gemm_runtime()
	{
		std::vector<GemmRuntime> result(2);

		// 24x16 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 24, 16, 1024 };
		result[0].gemm_kernel = gemm_avx512f_24x16_fp32;
		result[0].a_packing = pack_avx512f_24xK_fp32;
		result[0].b_packing = pack_avx512f_16xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(114.3, 42.8);

		// 24x16 fp16/fp32
		result[1].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[1].inner_tile = { 24, 16, 1024 };
		result[1].gemm_kernel = gemm_avx512f_24x16_fp32_fp16;
		result[1].a_packing = pack_avx512f_24xK_fp16_fp32;
		result[1].b_packing = pack_avx512f_16xK_fp16_fp32;
		result[1].c_packing = pack_def_MxK_fp16;
		result[1].d_packing = pack_def_MxK_fp16;
		result[1].d_unpacking = unpack_def_MxK_fp16;
		result[1].edge_a_packing = pack_def_MxK_fp16_fp32;
		result[1].edge_b_packing = pack_def_MxK_fp16_fp32;
		result[1].perf_estimator = PerfEstimator(114.3, 42.8);

		return result;
	}

	template<typename T>
	void join_vectors(std::vector<T> &dst, const std::vector<T> &src)
	{
		dst.insert(dst.end(), src.begin(), src.end());
	}

	const std::vector<GemmRuntime>& get_gemm_runtime_table(mlContext_t context)
	{
		static const std::vector<GemmRuntime> runtime_table = [context]()
		{
			std::vector<GemmRuntime> result;
			const cpu::SimdLevel simd = cpu::Context::getSimdLevel(context);
			if (simd >= cpu::SimdLevel::AVX512F)
				join_vectors(result, get_avx512f_gemm_runtime());
			if (simd >= cpu::SimdLevel::AVX2)
				join_vectors(result, get_avx2_fma_gemm_runtime());
			if (simd >= cpu::SimdLevel::AVX)
				join_vectors(result, get_avx_gemm_runtime());
			if (simd >= cpu::SimdLevel::SSE2)
				join_vectors(result, get_sse2_gemm_runtime());
			return result;
		}();
		assert(runtime_table.size() > 0);
		return runtime_table;
	}

	GemmRuntime get_runtime(mlContext_t context, mlDataType_t dtype, char opA, mlShape_t shape_A, char opB, mlShape_t shape_B)
	{
		assert(shape_A.rank == 2 || shape_A.rank == 3);
		assert(shape_B.rank == 2 || shape_B.rank == 3);
		const TypeConfiguration tc { dtype, dtype, dtype, dtype, DTYPE_FLOAT32 };

		const int M = is_transpose(opA) ? shape_A.dim[shape_A.rank - 1] : shape_A.dim[shape_A.rank - 2];
		const int N = is_transpose(opB) ? shape_B.dim[shape_B.rank - 2] : shape_B.dim[shape_B.rank - 1];
		const int K = is_transpose(opA) ? shape_A.dim[shape_A.rank - 2] : shape_A.dim[shape_A.rank - 1];

		const std::vector<GemmRuntime> &table = get_gemm_runtime_table(context);
		GemmRuntime result;

		float max_gflops = std::numeric_limits<float>::lowest();
		for (auto iter = table.begin(); iter < table.end(); iter++)
		{
			if (iter->can_work_with_types(tc))
			{
				const float gflops = iter->get_expected_gflops(M, N, K);
				if (gflops > max_gflops)
				{
					result = *iter;
					max_gflops = gflops;
				}
			}
		}
		return result;
	}

}

namespace ml
{
#ifndef USE_OPENBLAS
	void cpu_gemm(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		cpu_gemm_ex(context, dtype, shape_C, C, alpha, opA, shape_A, A, opB, shape_B, B, beta, shape_C, C, ACTIVATION_LINEAR);
	}
	void cpu_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		GemmRuntime rt = get_runtime(context, dtype, opA, shape_A, opB, shape_B);

		const int stride_A = size_of(dtype) * shape_A.dim[1] * shape_A.dim[2];
		const int stride_B = size_of(dtype) * shape_B.dim[1] * shape_B.dim[2];
		const int stride_C = size_of(dtype) * shape_C.dim[1] * shape_C.dim[2];
		const int stride_D = size_of(dtype) * shape_C.dim[1] * shape_C.dim[2];

		for (int i = 0; i < shape_A.dim[0]; i++)
		{
			rt.setMatrixA(getPointer<uint8_t>(A) + i * stride_A, shape_A, dtype, opA);
			rt.setMatrixB(getPointer<uint8_t>(B) + i * stride_B, shape_B, dtype, opB);
			rt.setMatrixC(getPointer<uint8_t>(C) + i * stride_C, shape_C, dtype);
			rt.setMatrixD(getPointer<uint8_t>(C) + i * stride_D, shape_C, dtype);
			if (i == 0)
			{
				rt.setScalingFactors(alpha, beta);
				rt.setup(context);
			}
			rt.run();
		}
	}
#endif

	void cpu_gemm_ex(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A, const void *A,
			char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C, mlActivationType_t act)
	{
		GemmRuntime rt = get_runtime(context, dtype, opA, shape_A, opB, shape_B);

		rt.setMatrixA(A, shape_A, dtype, opA);
		rt.setMatrixB(B, shape_B, dtype, opB);
		rt.setMatrixC(C, shape_C, dtype);
		rt.setMatrixD(D, shape_D, dtype);
		rt.setScalingFactors(alpha, beta);
		rt.useRelu(act == ACTIVATION_RELU);
		rt.setup(context);
		rt.run();
	}

} /* namespace ml */

