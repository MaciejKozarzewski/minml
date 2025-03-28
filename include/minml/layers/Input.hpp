/*
 * Input.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_INPUT_HPP_
#define MINML_LAYERS_INPUT_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class Input: public Layer
	{
		public:
			Input(Shape input_shape = Shape());

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			void setupQuantization(const std::vector<AffineTransform> &input_transforms, const AffineTransform &output_transform, int bits);

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_INPUT_HPP_ */
