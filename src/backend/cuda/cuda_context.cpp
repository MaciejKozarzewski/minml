/*
 * cuda_context.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cuda_backend.h>
#include "utils.hpp"

#include <cuda_runtime_api.h>

#include <cassert>

namespace ml
{
	mlContext_t cuda_create_context(int device_index)
	{
		return new cuda::Context(device_index);
	}
	void cuda_synchronize_with_context(mlContext_t context)
	{
		cudaError_t status = cudaStreamSynchronize(cuda::Context::getStream(context));
		assert(status == cudaSuccess);
	}
	bool cuda_is_context_ready(mlContext_t context)
	{
		cudaError_t status = cudaStreamQuery(cuda::Context::getStream(context));
		if (status == cudaSuccess)
			return true;
		else
		{
			if (status == cudaErrorNotReady)
				return false;
			else
			{
				assert(status == cudaSuccess);
				return false;
			}
		}
	}
	void cuda_destroy_context(mlContext_t context)
	{
		delete reinterpret_cast<cuda::Context*>(context);
	}

} /* namespace ml */

