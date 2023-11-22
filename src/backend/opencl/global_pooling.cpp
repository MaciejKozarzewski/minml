/*
 * global_pooling.cpp
 *
 *  Created on: Nov 22, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>

namespace ml
{

	void opencl_global_avg_and_max_pooling_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input)
	{
	}
	void opencl_global_avg_and_max_pooling_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next,
			const void *input, const void *output)
	{
	}
	void opencl_global_broadcasting_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input,
			const void *bias, mlActivationType_t act)
	{
	}
	void opencl_global_broadcasting_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, void *gradient_next, const void *output,
			mlActivationType_t act)
	{
	}
} /* namespace ml */

