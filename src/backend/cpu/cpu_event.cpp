/*
 * cpu_event.cpp
 *
 *  Created on: Nov 30, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>

#include <chrono>
#include <cassert>

namespace
{
	struct event_data
	{
			double time = 0.0;
			event_data() :
					time(std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count())
			{
			}
	};
}

namespace ml
{
	mlEvent_t cpu_create_event(mlContext_t context)
	{
		return reinterpret_cast<mlEvent_t*>(new event_data());
	}
	double cpu_get_time_between_events(mlEvent_t start, mlEvent_t end)
	{
		assert(start != nullptr);
		assert(end != nullptr);
		const double t0 = reinterpret_cast<event_data*>(start)->time;
		const double t1 = reinterpret_cast<event_data*>(end)->time;
		return t1 - t0;
	}
	void cpu_wait_for_event(mlEvent_t event)
	{
		// intentionally empty
	}
	bool cpu_is_event_ready(mlEvent_t event)
	{
		return true; // computations on CPU are synchronous
	}
	void cpu_destroy_event(mlEvent_t event)
	{
		if (event != nullptr)
			delete reinterpret_cast<event_data*>(event);
	}

} /* namespace ml */

