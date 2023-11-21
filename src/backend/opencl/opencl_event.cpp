/*
 * opencl_event.cpp
 *
 *  Created on: Nov 20, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/opencl_backend.h>
#include "utils.hpp"

#include <CL/opencl.hpp>

#include <cassert>

namespace ml
{
	mlEvent_t opencl_create_event(mlContext_t context)
	{
		return reinterpret_cast<mlEvent_t*>(new cl::Event(*opencl::Context::getLastEvent(context)));
	}
	void opencl_wait_for_event(mlEvent_t event)
	{
		if (event != nullptr)
		{
			const cl_int status = reinterpret_cast<cl::Event*>(event)->wait();
			CHECK_OPENCL_STATUS(status);
		}
	}
	bool opencl_is_event_ready(mlEvent_t event)
	{
		if (event != nullptr)
		{
			cl_int result = -1;
			const cl_int status = reinterpret_cast<cl::Event*>(event)->getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &result);
			CHECK_OPENCL_STATUS(status);
			return result == CL_COMPLETE;
		}
		else
			return true;
	}
	void opencl_destroy_event(mlEvent_t event)
	{
		if (event != nullptr)
			delete reinterpret_cast<cl::Event*>(event);
	}
} /* namespace ml */

