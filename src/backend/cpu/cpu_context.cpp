/*
 * cpu_context.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>

#include "utils.hpp"

namespace ml
{
	mlContext_t cpu_create_context()
	{
		return reinterpret_cast<mlContext_t*>(new cpu::Context());
	}
	void cpu_synchronize_with_context(mlContext_t context)
	{
		// intentionally empty
	}
	bool cpu_is_context_ready(mlContext_t context)
	{
		return true;
	}
	void cpu_destroy_context(mlContext_t context)
	{
		if (context != nullptr)
			delete reinterpret_cast<cpu::Context*>(context);
	}

} /* namespace ml */

