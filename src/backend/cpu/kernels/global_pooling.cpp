/*
 * global_pooling.cpp
 *
 *  Created on: Jan 15, 2023
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>

#include "../vectors/vectors.hpp"
#include "../helpers/indexers.hpp"
#include "../helpers/tensor_wrappers.hpp"

namespace
{
	template<typename T>
	void set_line(T *ptr, int elements, T value = T { }) noexcept
	{
		for (int i = 0; i < elements; i++)
			ptr[i] = static_cast<T>(0);
	}
	template<typename T>
	void cpu_kernel_global_avg_and_max_pooling_forward(const T *input, T *output, int dim0, int dim1, int dim2, T *workspace)
	{
//		const ConstTensorWrapper<3, T> input_wrapper(input, dim0, dim1, dim2);
//		TensorWrapper<2, T> output_wrapper(output, dim0, 2 * dim2);
//
//		TensorWrapper<1, T> avg_wrapper(workspace, dim2);
//		TensorWrapper<1, T> max_wrapper(workspace + dim2, dim2);
//
//		for (int d0 = 0; d0 < dim0; d0++)
//		{
//			set_line(avg_wrapper.data(), dim2);
//			for (int d2 = 0; d2 < dim2; d2++)
//				max_wrapper.data()[d2] = input[]
//			for (int d1 = 0; d1 < dim1; d1++)
//				for (int d2 = 0; d2 < dim2; d2++)
//				{
//				}
//		}
	}
//	void cpu_kernel_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
//			const void *input)
//	{
//	}
}

namespace SIMD_NAMESPACE
{
	using namespace ml;
	/*
	 * Welford's online algorithm for calculating mean and variance
	 */
	template<typename T>
	class AvgVarStats
	{
			T samples = static_cast<T>(0);
			T M = static_cast<T>(0);
			T M2 = static_cast<T>(0);
		public:
			void add(T x) noexcept
			{
				samples += static_cast<T>(1);
				const T delta = x - M;
				M += delta / samples;
				M2 += delta * (x - M);
			}
			T get_average() const noexcept
			{
				return M;
			}
			T get_variance() const noexcept
			{
				assert(samples >= static_cast<T>(2));
				return M2 / (samples - static_cast<T>(1));
			}

			static AvgVarStats merge(const AvgVarStats<T> &lhs, const AvgVarStats<T> &rhs) noexcept
			{
				assert(lhs.samples >= static_cast<T>(0) && rhs.samples >= static_cast<T>(0));
				AvgVarStats result;
				result.samples = lhs.samples + rhs.samples;
				result.M = (lhs.samples * lhs.M + rhs.samples * rhs.M) / result.samples;
				result.M2 = lhs.M2 + rhs.M2 + square(lhs.M - rhs.M) * (lhs.samples * rhs.samples) / result.samples;
				return result;
			}
	};

	void cpu_kernel_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output)
	{
		const int batch_size = get_first_dim(shape);
		const int hw = shape.dim[1] * shape.dim[2];
		const int channels = get_last_dim(shape);

		assert(cpu::Context::getWorkspaceSize(context) >= 2 * channels * size_of(dtype));
//		*stats = cpu::Context::getWorkspace<AvgVarStats<float>>(context);

		for (int b = 0; b < batch_size; b++)
		{

		}
	}
	void cpu_kernel_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
			const void *input)
	{
	}
}

