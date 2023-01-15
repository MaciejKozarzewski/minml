/*
 * random.cpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/random.hpp>
#include <math.h>
#include <cstdlib>
#include <assert.h>

namespace ml
{
	float randFloat()
	{
		return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}
	double randDouble()
	{
		return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	}
	float randGaussian()
	{
		double x1, x2, w;
		do
		{
			x1 = 2.0 * randDouble() - 1.0;
			x2 = 2.0 * randDouble() - 1.0;
			w = x1 * x1 + x2 * x2;
		} while (w >= 1.0);

		w = sqrt((-2.0 * log(w)) / w);
		thread_local double stored_value;
		thread_local bool is_stored = false;

		if (is_stored)
		{
			is_stored = false;
			return stored_value;
		}
		else
		{
			stored_value = x2 * w;
			is_stored = true;
			return static_cast<float>(x1 * w);
		}
	}
	int randInt()
	{
		return rand();
	}
	int randInt(int r)
	{
		return rand() % r;
	}
	int randInt(int r0, int r1)
	{
		assert(r1 > r0);
		return r0 + rand() % (r1 - r0);
	}
	int64_t randLong()
	{
		return ((static_cast<int64_t>(rand())) << 33) | ((static_cast<int64_t>(rand())) << 2) | (rand() & 2);
	}
	bool randBool()
	{
		return rand() & 1;
	}

} /* namespace ml */

