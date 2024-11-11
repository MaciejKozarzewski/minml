/*
 * WindowPartitioning.cpp
 *
 *  Created on: Nov 10, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/WindowPartitioning.hpp>
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
	WindowPartitioning::WindowPartitioning(int window_size, int window_shift) :
			Layer(),
			m_window_size(window_size),
			m_window_shift(window_shift)
	{
	}
	void WindowPartitioning::setInputShape(const std::vector<Shape> &shapes)
	{
		m_input_shapes = shapes;
	}
	Shape WindowPartitioning::getOutputShape() const
	{
		const int batch_size = getInputShape().dim(0);
		const int num_windows_h = (getInputShape().dim(1) + m_window_size - 1) / m_window_size;
		const int num_windows_w = (getInputShape().dim(2) + m_window_size - 1) / m_window_size;
		const int channels = getInputShape().dim(3);
		return Shape( { batch_size * num_windows_h * num_windows_w, m_window_size, m_window_size, channels });
	}
	std::string WindowPartitioning::name() const
	{
		return "WindowPartitioning";
	}
	Json WindowPartitioning::getConfig() const
	{
		Json result = Layer::getConfig();
		result["window_size"] = m_window_size;
		result["window_shift"] = m_window_shift;
		return result;
	}
	std::unique_ptr<Layer> WindowPartitioning::clone(const Json &config) const
	{
		std::unique_ptr<WindowPartitioning> result = std::make_unique<WindowPartitioning>(config["window_size"].getInt(),
				config["window_shift"].getInt());
		result->m_dtype = typeFromString(config["dtype"].getString());
		return result;
	}
	void WindowPartitioning::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		windowPartitioning(context(), input[0], output, Shape( { m_window_shift, m_window_shift }));
	}
	void WindowPartitioning::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev,
			Tensor &gradient_next)
	{
		windowMerging(context(), gradient_next, gradient_prev[0],  Shape( { m_window_shift, m_window_shift }));
	}
}

