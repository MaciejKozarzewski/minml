/*
 * add_bias_act.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>

#include <cmath>

namespace
{
	template<typename T>
	void kernel_softmax_in_place(T *ptr, int length)
	{
		T max_value = ptr[0];
		for (int i = 0; i < length; i++)
			max_value = std::max(max_value, ptr[i]);

		T sum = 0;
		for (int i = 0; i < length; i++)
		{
			ptr[i] = exp(ptr[i] - max_value);
			sum += ptr[i];
		}

		const T inv_sum = 1 / sum;
		for (int i = 0; i < length; i++)
			ptr[i] *= inv_sum;
	}
}

namespace SIMD_NAMESPACE
{
	using namespace ml;
	void cpu_kernel_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias, mlActivationType_t act)
	{
		const int first_dim = get_first_dim(shape);
		const int last_dim = get_last_dim(shape);

		for (int i = 0; i < first_dim; i++)
		{
			float *ptr = getPointer<float>(input) + i * last_dim;
			for (int j = 0; j < last_dim; j++)
			{
				float tmp = ptr[j] + getPointer<float>(bias)[j];
				if (act == ACTIVATION_RELU)
					tmp = std::max(0.0f, tmp);
				ptr[j] = tmp;
			}
			if (act == ACTIVATION_SOFTMAX)
				kernel_softmax_in_place(ptr, last_dim);
		}
	}
}

