/*
 * pooling_runtime.cpp
 *
 *  Created on: Feb 10, 2025
 *      Author: Maciej Kozarzewski
 */

#include "pooling_runtime.hpp"

#include "../utils.hpp"
#include "TensorFragment.hpp"
#include "kernels.hpp"

#include <limits>

namespace
{
	using namespace ml;
	using processing_function = std::function<void(const TensorFragment&, TensorFragment&)>;

	class PoolingProcessor
	{
			processing_function m_func;
			int m_step = 0;
			mlDataType_t m_dtype = DTYPE_UNKNOWN;
		public:
			PoolingProcessor(processing_function func, int step, mlDataType_t dtype) :
					m_func(func),
					m_step(step),
					m_dtype(dtype)
			{
			}
			void operator()(const TensorFragment &input, TensorFragment &output) const noexcept
			{
				m_func(input, output);
			}
			int step() const noexcept
			{
				return m_step;
			}
			mlDataType_t dtype() const noexcept
			{
				return m_dtype;
			}
			size_t step_in_bytes() const noexcept
			{
				return step() * size_of(dtype());
			}
	};

	const std::vector<PoolingProcessor>& get_pooling_processors_table(mlContext_t context)
	{
		static const std::vector<PoolingProcessor> processor_table = [context]()
		{
			std::vector<PoolingProcessor> result;
			const cpu::SimdLevel simd = cpu::Context::getSimdLevel(context);
//			if (simd >= cpu::SimdLevel::AVX512F)
//			{
//			}
//			if (simd >= cpu::SimdLevel::AVX)
//			{
//				result.emplace_back(average_pooling_avx_1x64xfp16, 64, DTYPE_FLOAT16);
//				result.emplace_back(average_pooling_avx_1x64xfp32, 64, DTYPE_FLOAT32);
//				result.emplace_back(average_pooling_avx_1x32xfp64, 32, DTYPE_FLOAT64);
//
//				result.emplace_back(average_pooling_avx_1x8xfp16, 8, DTYPE_FLOAT16);
//				result.emplace_back(average_pooling_avx_1x8xfp32, 8, DTYPE_FLOAT32);
//				result.emplace_back(average_pooling_avx_1x4xfp64, 4, DTYPE_FLOAT64);
//			}
//			if (simd >= cpu::SimdLevel::SSE2)
//			{
//				result.emplace_back(average_pooling_sse2_1x32xfp32, 32, DTYPE_FLOAT32);
//				result.emplace_back(average_pooling_sse2_1x16xfp64, 16, DTYPE_FLOAT64);
//
//				result.emplace_back(average_pooling_sse2_1x4xfp32, 4, DTYPE_FLOAT32);
//				result.emplace_back(average_pooling_sse2_1x2xfp64, 2, DTYPE_FLOAT64);
//			}
//			result.emplace_back(average_pooling_def_1xN, 1, DTYPE_FLOAT16);
//			result.emplace_back(average_pooling_def_1xN, 1, DTYPE_FLOAT32);
//			result.emplace_back(average_pooling_def_1xN, 1, DTYPE_FLOAT64);

			return result;
		}();
		assert(processor_table.size() > 0);
		return processor_table;
	}
	struct Index3D
	{
			int b = 0;
			int hw = 0;
			int c = 0;
	};

	TensorFragment get_input_fragment(const void *ptr, mlDataType_t dtype, mlShape_t shape, Index3D index) noexcept
	{
		const int offset = index.b * shape.dim[1] * shape.dim[2] + index.hw * shape.dim[2] + index.c;

//		return TensorFragment;
//			if (frag.is_packed())
//			{
//				void *shifted_ptr = frag.data<uint8_t>() + size_of(frag.dtype()) * frag.offset_at(pos.row, pos.column);
//				Fragment result(shifted_ptr, frag.dtype(), frag.stride());
//				result.mark_as_packed_with_size(size);
//				return result;
//			}
//			else
//				return Fragment();
	}
}

namespace ml
{

	void PoolingRuntime::run()
	{
		TensorFragment input_fragment;
		TensorFragment output_fragment;

//		PoolingProcessor bulk_processor;
//		PoolingProcessor edge_processor;
//
//		for (int b = 0; b < batch_size; b++)
//		{
//			int channel_count = 0;
//			for (; channel_count < channels; channel_count += bulk_processor.step())
//			{
//
//			}
//			for (; channel_count < channels; channel_count += edge_processor.step())
//			{
//			}
//		}
	}

} /* namespace ml */

