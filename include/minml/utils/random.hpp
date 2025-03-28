/*
 * random.hpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_UTILS_RANDOM_HPP_
#define MINML_UTILS_RANDOM_HPP_

#include <cstdint>

namespace ml
{
	float randFloat();
	double randDouble();
	float randGaussian();
	float randGaussian(float mean, float variance);
	int32_t randInt();
	int32_t randInt(int r);
	int32_t randInt(int r0, int r1);
	uint64_t randLong();
	bool randBool();
}

#endif /* MINML_UTILS_RANDOM_HPP_ */
