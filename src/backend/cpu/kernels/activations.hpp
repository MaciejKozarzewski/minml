/*
 * activations.hpp
 *
 *  Created on: Apr 17, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CPU_KERNELS_ACTIVATIONS_HPP_
#define BACKEND_CPU_KERNELS_ACTIVATIONS_HPP_

#include <minml/backend/backend_types.h>

#include "../vectors/vectors.hpp"

namespace SIMD_NAMESPACE
{
	template<typename T>
	Vector<T> activation_forward(ml::mlActivationType_t act, Vector<T> input) noexcept
	{
		switch (act)
		{
			default:
			case ml::ACTIVATION_LINEAR:
				return input;
			case ml::ACTIVATION_SIGMOID:
				return Vector<T>::one() / (Vector<T>::one() + exp(-input));
			case ml::ACTIVATION_TANH:
				return tanh(input);
			case ml::ACTIVATION_RELU:
				return max(Vector<T>::zero(), input);
		}
	}

} /* SIMD_NAMESPACE */

namespace ml
{
	namespace cpu
	{
		static inline float activation_forward(ml::mlActivationType_t act, float input) noexcept
		{
			switch (act)
			{
				default:
				case ml::ACTIVATION_LINEAR:
					return input;
				case ml::ACTIVATION_SIGMOID:
					return 1.0f / (1.0f + std::exp(-input));
				case ml::ACTIVATION_TANH:
					return std::tanh(input);
				case ml::ACTIVATION_RELU:
					return std::max(0.0f, input);
			}
		}
		static inline float activation_backward(ml::mlActivationType_t act, float gradient, float output) noexcept
		{
			switch (act)
			{
				default:
				case ml::ACTIVATION_LINEAR:
					return gradient;
				case ml::ACTIVATION_SIGMOID:
					return gradient * output * (1.0f - output);
				case ml::ACTIVATION_TANH:
					return gradient * (1.0f + output) * (1.0f - output);
				case ml::ACTIVATION_RELU:
					return output <= 0.0f ? 0.0f : gradient;
			}
		}
	}
}

#endif /* BACKEND_CPU_KERNELS_ACTIVATIONS_HPP_ */
