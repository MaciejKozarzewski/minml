/*
 * misc.cpp
 *
 *  Created on: Jan 10, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>

namespace ml
{
	void cuda_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, const void *input, void *output,
			void *max_indices)
	{
	}
	void cuda_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
			const void *max_indices)
	{
	}
}

