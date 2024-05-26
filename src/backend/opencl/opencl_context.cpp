/*
 * opencl_context.cpp
 *
 *  Created on: Nov 2, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include "utils.hpp"

namespace ml
{
	mlContext_t opencl_create_context(int device_index)
	{
		return new opencl::Context(device_index);
	}
	void opencl_synchronize_with_context(mlContext_t context)
	{
		opencl::Context::synchronizeWith(context);
	}
	bool opencl_is_context_ready(mlContext_t context)
	{
		return opencl::Context::isReady(context);
	}
	void opencl_destroy_context(mlContext_t context)
	{
		delete reinterpret_cast<opencl::Context*>(context);
	}

} /* namespace ml */

