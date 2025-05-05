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
		return new ml::cuda_backend::Context(device_index);
	}
	void cuda_synchronize_with_context(mlContext_t context)
	{
		cudaError_t status = cudaStreamSynchronize(ml::cuda_backend::Context::getStream(context));
		assert(status == cudaSuccess);
	}
	bool cuda_is_context_ready(mlContext_t context)
	{
		cudaError_t status = cudaStreamQuery(ml::cuda_backend::Context::getStream(context));
		return status == cudaSuccess;
	}
	void cuda_destroy_context(mlContext_t context)
	{
		delete reinterpret_cast<ml::cuda_backend::Context*>(context);
	}

} /* namespace ml */

