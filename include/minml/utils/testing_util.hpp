/*
 * testing_util.hpp
 *
 *  Created on: Sep 13, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_UTILS_TESTING_UTIL_HPP_
#define MINML_UTILS_TESTING_UTIL_HPP_

#include <minml/core/Tensor.hpp>

#include <memory>
#include <vector>

namespace ml /* forward declarations */
{
	class Device;
	class Layer;
	class Shape;
	enum class DataType;
}

namespace ml
{
	namespace testing
	{
		void initForTest(Tensor &t, double shift, double scale = 1.0);
		void initRandom(Tensor &t);
		double diffForTest(const Tensor &lhs, const Tensor &rhs);
		double maxAbsDiff(const Tensor &lhs, const Tensor &rhs);
		double normForTest(const Tensor &tensor);
		double sumForTest(const Tensor &tensor);
		void abs(Tensor &tensor);

		bool has_device_supporting(DataType dtype);
		Device get_device_for_test();

		class GradientCheck
		{
				std::unique_ptr<Layer> m_layer;
				std::vector<Tensor> input, gradient_prev;
				Tensor output, target, gradient_next;
			public:
				GradientCheck(const Layer &layer);
				void setInputShape(const Shape &shape);
				void setInputShape(const std::vector<Shape> &shapes);
				double check(int n, double epsilon, const std::string &mode, bool verbose = false);
			private:
				double compute_gradient(Tensor &t, int idx, double epsilon);
		};
	}
}

#endif /* MINML_UTILS_TESTING_UTIL_HPP_ */
