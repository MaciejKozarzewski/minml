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
	mlEvent_t cuda_create_event(mlContext_t context)
	{
		cudaEvent_t *result = new cudaEvent_t();
		cudaError_t status = cudaEventCreate(result);
		assert(status == cudaSuccess);
		status = cudaEventRecord(*result, cuda::Context::getStream(context));
		assert(status == cudaSuccess);
		return reinterpret_cast<mlEvent_t*>(result);
	}
	void cuda_wait_for_event(mlEvent_t event)
	{
		cudaError_t status = cudaEventSynchronize(*reinterpret_cast<cudaEvent_t*>(event));
		assert(status == cudaSuccess);
	}
	bool cuda_is_event_ready(mlEvent_t event)
	{
		cudaError_t status = cudaEventQuery(*reinterpret_cast<cudaEvent_t*>(event));
		return status == cudaSuccess;
	}
	void cuda_destroy_event(mlEvent_t event)
	{
		if (event != nullptr)
		{
			cudaError_t status = cudaEventDestroy(*reinterpret_cast<cudaEvent_t*>(event));
			assert(status == cudaSuccess);
		}
	}
} /* namespace ml */

