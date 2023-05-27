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
		std::vector<GemmRuntime> result(2);
		// 8x4
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 8, 4, 256 };
		result[0].gemm_kernel = gemm_sse2_8x4_fp32;
		result[0].a_packing = pack_sse2_8xK_fp32;
		result[0].b_packing = pack_sse2_4xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(-239.87, 15.65);

		// 4x4
		result[1].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[1].inner_tile = { 4, 4, 256 };
		result[1].gemm_kernel = gemm_sse2_4x4_fp32;
		result[1].a_packing = pack_sse2_4xK_fp32;
		result[1].b_packing = pack_sse2_4xK_fp32;
		result[1].c_packing = pack_def_MxK_fp32;
		result[1].d_packing = pack_def_MxK_fp32;
		result[1].d_unpacking = unpack_def_MxK_fp32;
		result[1].edge_a_packing = pack_def_MxK_fp32;
		result[1].edge_b_packing = pack_def_MxK_fp32;
		result[1].perf_estimator = PerfEstimator(-147.65, 15.61);

		return result;
	}
	std::vector<GemmRuntime> get_avx_gemm_runtime()
	{
		std::vector<GemmRuntime> result(2);
		// 10x8
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
		result[0].perf_estimator = PerfEstimator(-334.44, 31.64);

		// 8x8
		result[1].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[1].inner_tile = { 8, 8, 256 };
		result[1].gemm_kernel = gemm_avx_8x8_fp32;
		result[1].a_packing = pack_avx_8xK_fp32;
		result[1].b_packing = pack_avx_8xK_fp32;
		result[1].c_packing = pack_def_MxK_fp32;
		result[1].d_packing = pack_def_MxK_fp32;
		result[1].d_unpacking = unpack_def_MxK_fp32;
		result[1].edge_a_packing = pack_def_MxK_fp32;
		result[1].edge_b_packing = pack_def_MxK_fp32;
		result[1].perf_estimator = PerfEstimator(-256.41, 26.07);

		return result;
	}
	std::vector<GemmRuntime> get_avx2_fma_f16c_gemm_runtime()
	{
		std::vector<GemmRuntime> result(4);
		// 6x16 fp32
		result[0].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[0].inner_tile = { 6, 16, 1024 };
		result[0].gemm_kernel = gemm_avx2_fma_6x16_fp32;
		result[0].a_packing = pack_avx2_fma_6xK_fp32;
		result[0].b_packing = pack_avx2_fma_16xK_fp32;
		result[0].c_packing = pack_def_MxK_fp32;
		result[0].d_packing = pack_def_MxK_fp32;
		result[0].d_unpacking = unpack_def_MxK_fp32;
		result[0].edge_a_packing = pack_def_MxK_fp32;
		result[0].edge_b_packing = pack_def_MxK_fp32;
		result[0].perf_estimator = PerfEstimator(-945.15, 59.64);

		// 24x4 fp32
		result[1].type_configuration = { DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32, DTYPE_FLOAT32 };
		result[1].inner_tile = { 24, 4, 256 };
		result[1].gemm_kernel = gemm_avx2_fma_24x4_fp32;
		result[1].a_packing = pack_avx2_fma_24xK_fp32;
		result[1].b_packing = pack_avx2_fma_4xK_fp32;
		result[1].c_packing = pack_def_MxK_fp32;
		result[1].d_packing = pack_def_MxK_fp32;
		result[1].d_unpacking = unpack_def_MxK_fp32;
		result[1].edge_a_packing = pack_def_MxK_fp32;
		result[1].edge_b_packing = pack_def_MxK_fp32;
		result[1].perf_estimator = PerfEstimator(-692.08, 46.66);

		// 6x16 fp16/fp32
		result[2].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[2].inner_tile = { 6, 16, 1024 };
		result[2].gemm_kernel = gemm_avx2_fma_6x16_fp16_fp32;
		result[2].a_packing = pack_avx2_fma_6xK_fp16_fp32;
		result[2].b_packing = pack_avx2_fma_16xK_fp16_fp32;
		result[2].c_packing = pack_def_MxK_fp16;
		result[2].d_packing = pack_def_MxK_fp16;
		result[2].d_unpacking = unpack_def_MxK_fp16;
		result[2].edge_a_packing = pack_def_MxK_fp16_fp32;
		result[2].edge_b_packing = pack_def_MxK_fp16_fp32;
		result[2].perf_estimator = PerfEstimator(-819.13, 60.28);

		// 24x4 fp16/fp32
		result[3].type_configuration = { DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT16, DTYPE_FLOAT32 };
		result[3].inner_tile = { 24, 4, 512 };
		result[3].gemm_kernel = gemm_avx2_fma_24x4_fp16_fp32;
		result[3].a_packing = pack_avx2_fma_24xK_fp16_fp32;
		result[3].b_packing = pack_avx2_fma_4xK_fp16_fp32;
		result[3].c_packing = pack_def_MxK_fp16;
		result[3].d_packing = pack_def_MxK_fp16;
		result[3].d_unpacking = unpack_def_MxK_fp16;
		result[3].edge_a_packing = pack_def_MxK_fp16_fp32;
		result[3].edge_b_packing = pack_def_MxK_fp16_fp32;
		result[3].perf_estimator = PerfEstimator(-655.10, 46.88);

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
//			if (simd >= cpu::SimdLevel::AVX512F)
//				;
			if (simd >= cpu::SimdLevel::AVX2)
				return get_avx2_fma_f16c_gemm_runtime();
			if (simd >= cpu::SimdLevel::AVX)
				return get_avx_gemm_runtime();
			if (simd >= cpu::SimdLevel::SSE2)
				return get_sse2_gemm_runtime();
			return result;
		}();
		return runtime_table;
	}

	GemmRuntime get_runtime(mlContext_t context, mlDataType_t dtype, char opA, mlShape_t shape_A, char opB, mlShape_t shape_B)
	{
		assert(shape_A.rank == 2 || shape_A.rank == 3);
		assert(shape_B.rank == 2 || shape_B.rank == 3);
		const TypeConfiguration tc { dtype, dtype, dtype, dtype, DTYPE_FLOAT32 };

		const int M = (opA == 'n') ? shape_A.dim[shape_A.rank - 2] : shape_A.dim[shape_A.rank - 1];
		const int N = (opB == 'n') ? shape_B.dim[shape_B.rank - 1] : shape_B.dim[shape_B.rank - 2];
		const int K = (opA == 'n') ? shape_A.dim[shape_A.rank - 1] : shape_A.dim[shape_A.rank - 2];

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
		cpu_gemm_v2(context, dtype, shape_C, C, alpha, opA, shape_A, A, opB, shape_B, B, beta, shape_C, C);
	}
	void cpu_gemm_batched(mlContext_t context, mlDataType_t dtype, mlShape_t shape_C, void *C, mlShape_t shape_A, const void *A, mlShape_t shape_B,
			const void *B, char opA, char opB, float alpha, float beta)
	{
		cpu_gemm_batched_v2(context, dtype, shape_C, C, alpha, opA, shape_A, A, opB, shape_B, B, beta, shape_C, C);
	}
#endif

	void cpu_gemm_v2(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A, const void *A,
			char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C)
	{
		GemmRuntime rt = get_runtime(context, dtype, opA, shape_A, opB, shape_B);

		rt.setMatrixA(A, shape_A, dtype, opA);
		rt.setMatrixB(B, shape_B, dtype, opB);
		rt.setMatrixC(C, shape_C, dtype);
		rt.setMatrixD(D, shape_D, dtype);
		rt.setScalingFactors(alpha, beta);
		rt.setup(context);
		rt.run();
	}
	void cpu_gemm_batched_v2(mlContext_t context, mlDataType_t dtype, mlShape_t shape_D, void *D, float alpha, char opA, mlShape_t shape_A,
			const void *A, char opB, mlShape_t shape_B, const void *B, float beta, mlShape_t shape_C, const void *C)
	{
		GemmRuntime rt = get_runtime(context, dtype, opA, shape_A, opB, shape_B);

		const int stride_A = size_of(dtype) * shape_A.dim[1] * shape_A.dim[2];
		const int stride_B = size_of(dtype) * shape_B.dim[1] * shape_B.dim[2];
		const int stride_C = size_of(dtype) * shape_C.dim[1] * shape_C.dim[2];
		const int stride_D = size_of(dtype) * shape_D.dim[1] * shape_D.dim[2];

		for (int i = 0; i < shape_A.dim[0]; i++)
		{
			rt.setMatrixA(getPointer<uint8_t>(A) + i * stride_A, shape_A, dtype, opA);
			rt.setMatrixB(getPointer<uint8_t>(B) + i * stride_B, shape_B, dtype, opB);
			rt.setMatrixC(getPointer<uint8_t>(C) + i * stride_C, shape_C, dtype);
			rt.setMatrixD(getPointer<uint8_t>(D) + i * stride_D, shape_D, dtype);
			if (i == 0)
			{
				rt.setScalingFactors(alpha, beta);
				rt.setup(context);
			}
			rt.run();
		}
	}

} /* namespace ml */

