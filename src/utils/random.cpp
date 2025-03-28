/*
 * random.cpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/random.hpp>

#include <cmath>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cassert>

namespace
{
#ifdef NDEBUG
	thread_local std::mt19937 int32_generator(std::chrono::system_clock::now().time_since_epoch().count());
	thread_local std::mt19937_64 int64_generator(std::chrono::system_clock::now().time_since_epoch().count());
#else
	thread_local std::mt19937 int32_generator(0);
	thread_local std::mt19937_64 int64_generator(0);
#endif
}

namespace ml
{
	float randFloat()
	{
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		return dist(int32_generator);
	}
	double randDouble()
	{
		std::uniform_real_distribution<double> dist(0.0f, 1.0f);
		return dist(int64_generator);
	}
	float randGaussian()
	{
		return randGaussian(0.0f, 1.0f);
	}
	float randGaussian(float mean, float variance)
	{
		std::normal_distribution<float> dist(mean, variance);
		return dist(int32_generator);
	}
	int32_t randInt()
	{
		return int32_generator();
	}
	int32_t randInt(int r)
	{
		assert(r != 0);
		std::uniform_int_distribution<int32_t> dist(0, r - 1);
		return dist(int32_generator);
	}
	int32_t randInt(int r0, int r1)
	{
		assert(r0 != r1);
		std::uniform_int_distribution<int32_t> dist(r0, r1 - 1);
		return dist(int32_generator);
	}
	uint64_t randLong()
	{
		return int64_generator();
	}
	bool randBool()
	{
		return static_cast<bool>(int32_generator() & 1);
	}

} /* namespace ml */

