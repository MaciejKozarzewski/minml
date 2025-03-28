/*
 * WindowMerging.cpp
 *
 *  Created on: Nov 10, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/WindowMerging.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>
#include <minml/utils/testing_util.hpp>

#include <initializer_list>
#include <memory>
#include <string>

namespace ml
{

	WindowMerging::WindowMerging(const Shape &shape, int shift) :
			Layer(),
			m_dst_shape(shape),
			m_window_shift(shift)
	{
	}
	void WindowMerging::setInputShape(const std::vector<Shape> &shapes)
	{
		m_input_shapes = shapes;
	}
	Shape WindowMerging::getOutputShape() const
	{
		return m_dst_shape;
	}
	std::string WindowMerging::name() const
	{
		return "WindowMerging";
	}
	Json WindowMerging::getConfig() const
	{
		Json result = Layer::getConfig();
		result["dst_shape"] = m_dst_shape.serialize();
		result["window_shift"] = m_window_shift;
		return result;
	}
	std::unique_ptr<Layer> WindowMerging::clone(const Json &config) const
	{
		std::unique_ptr<WindowMerging> result = std::make_unique<WindowMerging>(Shape(config["dst_shape"]), config["window_shift"].getInt());
		result->loadConfig(config);
		return result;
	}
	void WindowMerging::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		windowMerging(context(), input[0], output, { m_window_shift, m_window_shift });
	}
	void WindowMerging::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		windowPartitioning(context(), gradient_next, gradient_prev[0], { m_window_shift, m_window_shift });
	}
}

