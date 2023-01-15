/*
 * testing_util.hpp
 *
 *  Created on: Sep 13, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_UTILS_TESTING_UTIL_HPP_
#define MINML_UTILS_TESTING_UTIL_HPP_

namespace ml /* forward declarations */
{
	class Tensor;
}

namespace ml
{
	namespace testing
	{
		void initForTest(Tensor &t, double shift);
		double diffForTest(const Tensor &lhs, const Tensor &rhs);
		double normForTest(const Tensor &tensor);
		void abs(Tensor &tensor);
	}
}

#endif /* MINML_UTILS_TESTING_UTIL_HPP_ */
