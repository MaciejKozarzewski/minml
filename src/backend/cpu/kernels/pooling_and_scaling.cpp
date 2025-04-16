/*
 * pooling_and_scaling.cpp
 *
 *  Created on: Feb 10, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>

#include "../utils.hpp"
#include "TensorFragment.hpp"
#include "kernels.hpp"

#include <functional>
#include <limits>

namespace
{
	using namespace ml;
	using pooling_function = std::function<void(const TensorFragment&, TensorFragment&)>;
	using channel_scaling_function = std::function<void(const TensorFragment&, const TensorFragment&, TensorFragment&)>;

	template<typename T>
	class Processor
	{
			T m_func;
			int m_step = 0;
			mlDataType_t m_dtype = DTYPE_UNKNOWN;
		public:
			Processor() noexcept = default;
			Processor(T func, int step, mlDataType_t dtype) noexcept :
					m_func(func),
					m_step(step),
					m_dtype(dtype)
			{
			}
			void operator()(const TensorFragment &input, TensorFragment &output) const noexcept
			{
				if constexpr (std::is_same<T, pooling_function>::value)
					m_func(input, output);
			}
			void operator()(const TensorFragment &input, const TensorFragment &scales, TensorFragment &output) const noexcept
			{
				if constexpr (std::is_same<T, channel_scaling_function>::value)
					m_func(input, scales, output);
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

	Processor<pooling_function> get_pooling_processor(mlContext_t context, mlDataType_t dtype, int channels)
	{
		static const std::vector<Processor<pooling_function>> table = [context]()
		{
			std::vector<Processor<pooling_function>> result;
			const cpu::SimdLevel simd = cpu::Context::getSimdLevel(context);
			if (simd >= cpu::SimdLevel::AVX512F)
			{
			}
			if (simd >= cpu::SimdLevel::AVX)
			{
				result.emplace_back(average_pooling_avx_1x64xfp16, 64, DTYPE_FLOAT16);
				result.emplace_back(average_pooling_avx_1x8xfp16, 8, DTYPE_FLOAT16);

				result.emplace_back(average_pooling_avx_1x64xfp32, 64, DTYPE_FLOAT32);
				result.emplace_back(average_pooling_avx_1x8xfp32, 8, DTYPE_FLOAT32);
			}
			if (simd >= cpu::SimdLevel::SSE2)
			{
				result.emplace_back(average_pooling_sse2_1x32xfp32, 32, DTYPE_FLOAT32);
				result.emplace_back(average_pooling_sse2_1x4xfp32, 4, DTYPE_FLOAT32);
			}
			result.emplace_back(average_pooling_def_1xN, 1, DTYPE_FLOAT16);
			result.emplace_back(average_pooling_def_1xN, 1, DTYPE_FLOAT32);

			return result;
		}();

		for (size_t i = 0; i < table.size(); i++)
			if (table[i].dtype() == dtype and table[i].step() <= channels)
				return table[i];
		return Processor<pooling_function>();
	}

	Processor<channel_scaling_function> get_channel_scaling_processor(mlContext_t context, mlDataType_t dtype, int channels)
	{
		static const std::vector<Processor<channel_scaling_function>> table = [context]()
		{
			std::vector<Processor<channel_scaling_function>> result;
			const cpu::SimdLevel simd = cpu::Context::getSimdLevel(context);
			if (simd >= cpu::SimdLevel::AVX512F)
			{
			}
			if (simd >= cpu::SimdLevel::AVX)
			{
				result.emplace_back(channel_scaling_avx_1x64xfp16, 64, DTYPE_FLOAT16);
				result.emplace_back(channel_scaling_avx_1x8xfp16, 8, DTYPE_FLOAT16);

				result.emplace_back(channel_scaling_avx_1x64xfp32, 64, DTYPE_FLOAT32);
				result.emplace_back(channel_scaling_avx_1x8xfp32, 8, DTYPE_FLOAT32);
			}
			if (simd >= cpu::SimdLevel::SSE2)
			{
				result.emplace_back(channel_scaling_sse2_1x32xfp32, 32, DTYPE_FLOAT32);
				result.emplace_back(channel_scaling_sse2_1x4xfp32, 4, DTYPE_FLOAT32);
			}
			result.emplace_back(channel_scaling_def_1xN, 1, DTYPE_FLOAT16);
			result.emplace_back(channel_scaling_def_1xN, 1, DTYPE_FLOAT32);

			return result;
		}();

		for (size_t i = 0; i < table.size(); i++)
			if (table[i].dtype() == dtype and table[i].step() <= channels)
				return table[i];
		return Processor<channel_scaling_function>();
	}

	void* increment_pointer(void *ptr, size_t bytes) noexcept
	{
		return reinterpret_cast<uint8_t*>(ptr) + bytes;
	}
	const void* increment_pointer(const void *ptr, size_t bytes) noexcept
	{
		return reinterpret_cast<const uint8_t*>(ptr) + bytes;
	}
}

namespace ml
{

	void cpu_global_average_pooling_forward(mlContext_t context, float alpha, const mlTensor_t x, float beta, mlTensor_t y)
	{
		assert(is_fp32(x) || is_fp16(x));
		assert(is_fp32(y) || is_fp16(y));
		assert(x.dtype == y.dtype);
		assert(x.rank == 4);
		assert(y.rank == 2);
		assert(get_first_dim(x) == get_first_dim(y));
		assert(get_last_dim(x) == get_last_dim(y));
		assert(beta == 0.0f);
		assert(alpha == 1.0f);

		const int batch_size = x.dim[0];
		const int height = x.dim[1];
		const int width = x.dim[2];
		const int channels = x.dim[3];

		const Processor<pooling_function> bulk_processor = get_pooling_processor(context, x.dtype, channels);
		const Processor<pooling_function> edge_processor = get_pooling_processor(context, x.dtype, 1);

		for (int b = 0; b < batch_size; b++)
		{
			const void *x_ptr = data<uint8_t>(x) + offset_at(x, { b, 0, 0, 0 }) * size_of(x.dtype);
			void *y_ptr = data<uint8_t>(y) + offset_at(y, { b, 0 }) * size_of(y.dtype);

			for (int c = 0; c < channels;)
			{
				const int channels_left = channels - c;

				TensorFragment input_fragment(x_ptr, x.dtype, height * width, channels_left, channels);
				TensorFragment output_fragment(y_ptr, y.dtype, 1, channels_left, channels);

				const Processor<pooling_function> &processor = (channels_left >= bulk_processor.step()) ? bulk_processor : edge_processor;

				processor(input_fragment, output_fragment);

				x_ptr = increment_pointer(x_ptr, processor.step_in_bytes());
				y_ptr = increment_pointer(y_ptr, processor.step_in_bytes());
				c += processor.step();
			}
		}
	}
	void cpu_global_average_pooling_backward(mlContext_t context, float alpha, const mlTensor_t dy, float beta, mlTensor_t dx)
	{
		assert(is_fp32(dx));
		assert(is_fp32(dy));
		assert(dx.rank == 4);
		assert(dy.rank == 2);
		assert(get_first_dim(dx) == get_first_dim(dy));
		assert(get_last_dim(dx) == get_last_dim(dy));

		const int batch_size = dx.dim[0];
		const int height = dx.dim[1];
		const int width = dx.dim[2];
		const int channels = dx.dim[3];

		const float scale = alpha / (height * width);
		for (int b = 0; b < batch_size; b++)
			for (int h = 0; h < height; h++)
				for (int w = 0; w < width; w++)
				{
					const float *dy_ptr = data<float>(dy) + offset_at(dy, { b, 0 });
					float *dx_ptr = data<float>(dx) + offset_at(dx, { b, h, w, 0 });
					if (beta == 0.0f)
					{
						for (int c = 0; c < channels; c++)
							dx_ptr[c] = dy_ptr[c] * scale;
					}
					else
					{
						for (int c = 0; c < channels; c++)
							dx_ptr[c] = dx_ptr[c] * beta + dy_ptr[c] * scale;
					}
				}
	}

	void cpu_channel_scaling_forward(mlContext_t context, float alpha, const mlTensor_t x, const mlTensor_t scales, float beta, mlTensor_t y)
	{
		assert(is_fp32(x) || is_fp16(x));
		assert(is_fp32(y) || is_fp16(y));
		assert(is_fp32(scales) || is_fp16(scales));
		assert(x.dtype == y.dtype && x.dtype == scales.dtype);
		assert(x.rank == 4);
		assert(y.rank == 4);
		assert(scales.rank == 2);
		assert(get_first_dim(x) == get_first_dim(y));
		assert(get_first_dim(x) == get_first_dim(scales));
		assert(get_last_dim(x) == get_last_dim(y));
		assert(get_last_dim(x) == get_last_dim(scales));
		assert(beta == 0.0f);
		assert(alpha == 1.0f);

		const int batch_size = x.dim[0];
		const int height = x.dim[1];
		const int width = x.dim[2];
		const int channels = x.dim[3];

		const Processor<channel_scaling_function> bulk_processor = get_channel_scaling_processor(context, x.dtype, channels);
		const Processor<channel_scaling_function> edge_processor = get_channel_scaling_processor(context, x.dtype, 1);

		for (int b = 0; b < batch_size; b++)
		{
			const void *x_ptr = data<uint8_t>(x) + offset_at(x, { b, 0, 0, 0 }) * size_of(x.dtype);
			const void *scales_ptr = data<uint8_t>(scales) + offset_at(scales, { b, 0 }) * size_of(scales.dtype);
			void *y_ptr = data<uint8_t>(y) + offset_at(y, { b, 0, 0, 0 }) * size_of(y.dtype);

			for (int c = 0; c < channels;)
			{
				const int channels_left = channels - c;

				TensorFragment input_fragment(x_ptr, x.dtype, height * width, channels_left, channels);
				TensorFragment scales_fragment(scales_ptr, scales.dtype, 1, channels_left, channels);
				TensorFragment output_fragment(y_ptr, y.dtype, height * width, channels_left, channels);

				const Processor<channel_scaling_function> &processor = (channels_left >= bulk_processor.step()) ? bulk_processor : edge_processor;

				processor(input_fragment, scales_fragment, output_fragment);

				x_ptr = increment_pointer(x_ptr, processor.step_in_bytes());
				scales_ptr = increment_pointer(scales_ptr, processor.step_in_bytes());
				y_ptr = increment_pointer(y_ptr, processor.step_in_bytes());
				c += processor.step();
			}
		}
	}
	void cpu_channel_scaling_backward(mlContext_t context, float alpha, const mlTensor_t dy, const mlTensor_t x, const mlTensor_t scales,
			float beta_dx, mlTensor_t dx, float beta_scales, mlTensor_t dscales)
	{
		assert(is_fp32(dx));
		assert(is_fp32(dy));
		assert(is_fp32(dscales));
		assert(dx.rank == 4);
		assert(dy.rank == 2);
		assert(dscales.rank == 2);
		assert(get_first_dim(dx) == get_first_dim(dy));
		assert(get_first_dim(dx) == get_first_dim(dscales));
		assert(get_last_dim(dx) == get_last_dim(dy));
		assert(get_last_dim(dx) == get_last_dim(dscales));

		const int batch_size = dx.dim[0];
		const int height = dx.dim[1];
		const int width = dx.dim[2];
		const int channels = dx.dim[3];

//		const float scale = alpha / (height * width);
//		for (int b = 0; b < batch_size; b++)
//			for (int h = 0; h < height; h++)
//				for (int w = 0; w < width; w++)
//				{
//					const float *dy_ptr = data<float>(dy) + offset_at(dy, { b, 0 });
//					float *dx_ptr = data<float>(dx) + offset_at(dx, { b, h, w, 0 });
//					if (beta == 0.0f)
//					{
//						for (int c = 0; c < channels; c++)
//							dx_ptr[c] = dy_ptr[c] * scale;
//					}
//					else
//					{
//						for (int c = 0; c < channels; c++)
//							dx_ptr[c] = dx_ptr[c] * beta + dy_ptr[c] * scale;
//					}
//				}
	}

} /* namespace ml */

