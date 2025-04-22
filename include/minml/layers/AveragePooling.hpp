/*
 * AveragePooling.hpp
 *
 *  Created on: Apr 21, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_AVERAGEPOOLING_HPP_
#define MINML_LAYERS_AVERAGEPOOLING_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class AveragePooling: public Layer
	{
			int m_size = 0;
		public:
			AveragePooling(int size);

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_AVERAGEPOOLING_HPP_ */
