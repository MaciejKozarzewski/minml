/*
 * LearnableScaling.hpp
 *
 *  Created on: Feb 24, 2026
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_LAYERS_LEARNABLESCALING_HPP_
#define MINML_LAYERS_LEARNABLESCALING_HPP_

#include <minml/layers/Layer.hpp>

namespace ml
{

	class LearnableScaling: public Layer
	{
			float m_initial_scale = 0.0f;
			bool m_trainable_scale = true;
		public:
			LearnableScaling(const std::string &act, float initialScale, bool trainable);

			Shape getOutputShape() const;
			Shape getWeightShape() const;

			std::string name() const;
			Json getConfig() const;

			std::unique_ptr<Layer> clone(const Json &config) const;

			void init();
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta);
	};

} /* namespace ml */

#endif /* MINML_LAYERS_LEARNABLESCALING_HPP_ */
