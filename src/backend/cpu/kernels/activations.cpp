/*
 * activations.cpp
 *
 *  Created on: Jan 10, 2023
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>

#include <cmath>

namespace
{
	float round_small_to_zero(float x) noexcept
	{
		return (fabsf(x) < 1.0e-12f) ? 0.0f : x;
	}

	template<typename T>
	void kernel_softmax_in_place(T *ptr, int length)
	{
		T max_value = ptr[0];
		for (int i = 1; i < length; i++)
			max_value = std::max(max_value, ptr[i]);

		T sum = 0;
		for (int i = 0; i < length; i++)
		{
			ptr[i] = exp(ptr[i] - max_value);
			sum += ptr[i];
		}

		const T inv_sum = 1 / sum;
		for (int i = 0; i < length; i++)
			ptr[i] = round_small_to_zero(ptr[i] * inv_sum);
	}

	template<typename T>
	void kernel_relu_in_place(T *ptr, int length)
	{
		for (int i = 0; i < length; i++)
			ptr[i] = std::max(static_cast<T>(0), ptr[i]);
	}
	template<typename T>
	void kernel_relu_backward_in_place(T *gradient, const T *output, int length)
	{
		for (int i = 0; i < length; i++)
			if (output[i] == static_cast<T>(0))
				gradient[i] = static_cast<T>(0);
	}
}

namespace SIMD_NAMESPACE
{
	using namespace ml;

	void cpu_kernel_activation_forward_in_place(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, mlActivationType_t act)
	{
		switch (act)
		{
			case ACTIVATION_LINEAR:
				break;
			case ACTIVATION_RELU:
				kernel_relu_in_place(getPointer<float>(input), volume(shape));
				break;
			case ACTIVATION_SOFTMAX:
			{
				const int first_dim = get_first_dim(shape);
				const int last_dim = get_last_dim(shape);
				for (int i = 0; i < first_dim; i++)
					kernel_softmax_in_place(getPointer<float>(input) + i * last_dim, last_dim);
				break;
			}
		}
	}
	void cpu_kernel_activation_backward_in_place(mlContext_t context, mlShape_t shape, void *gradient, const void *output, mlActivationType_t act)
	{
		switch (act)
		{
			case ACTIVATION_LINEAR:
				break;
			case ACTIVATION_RELU:
				kernel_relu_backward_in_place(getPointer<float>(gradient), getPointer<float>(output), volume(shape));
				break;
			case ACTIVATION_SOFTMAX:
				break;
		}
	}
}

