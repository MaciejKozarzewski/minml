/*
 * add_bias_act.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>

#include "../vectors/vectors.hpp"

#include <cmath>
#include <iostream>

namespace
{
	using namespace ml;
	using namespace SIMD_NAMESPACE;

	template<typename T>
	void kernel_softmax_in_place(T *ptr, int length)
	{
		assert(ptr != nullptr);
		const Vector<T> first(ptr[0]);

		Vector<T> max_value = first;
		for (int i = 0; i < length; i += Vector<T>::length)
		{
			Vector<T> tmp(ptr + i, length - i);
			tmp.cutoff(length - i, first);
			max_value = max(max_value, tmp);
		}
		max_value = horizontal_max(max_value);

		Vector<T> sum = Vector<T>::zero();
		for (int i = 0; i < length; i += Vector<T>::length)
		{
			Vector<T> tmp(ptr + i, length - i);
			tmp = exp(tmp - max_value);
			tmp.cutoff(length - i, Vector<T>::zero());
			sum += tmp;
			tmp.store(ptr + i, length - i);
		}
		sum = Vector<T>::one() / Vector<T>(horizontal_add(sum));
		for (int i = 0; i < length; i += Vector<T>::length)
		{
			Vector<T> tmp(ptr + i, length - i);
			tmp *= sum;
			tmp.store(ptr + i, length - i);
		}
	}

	template<typename T>
	void kernel_relu_in_place(T *ptr, int length)
	{
		for (int i = 0; i < length; i += Vector<T>::length)
		{
			Vector<T> tmp(ptr + i, length - i);
			tmp = max(Vector<T>::zero(), tmp);
			tmp.store(ptr + i, length - i);
		}
	}
	template<typename T>
	void kernel_relu_backward_in_place(T *gradient, const T *output, int length)
	{
		for (int i = 0; i < length; i++)
			if (output[i] == static_cast<T>(0))
				gradient[i] = static_cast<T>(0);
	}

	template<typename T>
	void launch_add_bias_act(mlShape_t shape, T *input, const T *bias, mlActivationType_t act)
	{
		assert(input != nullptr);
		assert(bias != nullptr);
		const int first_dim = get_first_dim(shape);
		const int last_dim = get_last_dim(shape);

		for (int i = 0; i < first_dim; i++)
		{
			T *ptr = input + i * last_dim;
			for (int j = 0; j < last_dim; j += Vector<T>::length)
			{
				Vector<T> tmp(ptr + j, last_dim - j);
				Vector<T> b(bias + j, last_dim - j);
				tmp += b;
				if (act == ACTIVATION_RELU)
					tmp = max(Vector<T>::zero(), tmp);
				tmp.store(ptr + j, last_dim - j);
			}
			if (act == ACTIVATION_SOFTMAX)
				kernel_softmax_in_place(ptr, last_dim);
		}
	}
}

namespace SIMD_NAMESPACE
{
	using namespace ml;

	void cpu_kernel_activation_forward_in_place(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, mlActivationType_t act)
	{
		switch (act)
		{
			case ACTIVATION_LINEAR:
				break;
			case ACTIVATION_RELU:
			{
				switch (dtype)
				{
					case DTYPE_BFLOAT16:
						kernel_relu_in_place(getPointer<bfloat16>(input), volume(shape));
						break;
					case DTYPE_FLOAT16:
						kernel_relu_in_place(getPointer<float16>(input), volume(shape));
						break;
					case DTYPE_FLOAT32:
						kernel_relu_in_place(getPointer<float>(input), volume(shape));
						break;
					default:
						break;
				}
				break;
			}
			case ACTIVATION_SOFTMAX:
			{
				const int first_dim = get_first_dim(shape);
				const int last_dim = get_last_dim(shape);
				switch (dtype)
				{
					case DTYPE_BFLOAT16:
						for (int i = 0; i < first_dim; i++)
							kernel_softmax_in_place(getPointer<bfloat16>(input) + i * last_dim, last_dim);
						break;
					case DTYPE_FLOAT16:
						for (int i = 0; i < first_dim; i++)
							kernel_softmax_in_place(getPointer<float16>(input) + i * last_dim, last_dim);
						break;
					case DTYPE_FLOAT32:
						for (int i = 0; i < first_dim; i++)
							kernel_softmax_in_place(getPointer<float>(input) + i * last_dim, last_dim);
						break;
					default:
						break;
				}
				break;
			}
		}
	}
	void cpu_kernel_activation_backward_in_place(mlContext_t context, mlShape_t shape, void *gradient, const void *output, mlActivationType_t act)
	{
		switch (act)
		{
			case ACTIVATION_LINEAR:
				break;
			case ACTIVATION_RELU:
				kernel_relu_backward_in_place(getPointer<float>(gradient), getPointer<float>(output), volume(shape));
				break;
			case ACTIVATION_SOFTMAX:
				break;
		}
	}

	void cpu_kernel_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias, mlActivationType_t act)
	{
		switch (dtype)
		{
			case DTYPE_BFLOAT16:
				launch_add_bias_act(shape, getPointer<bfloat16>(input), getPointer<bfloat16>(bias), act);
				break;
			case DTYPE_FLOAT16:
				launch_add_bias_act(shape, getPointer<float16>(input), getPointer<float16>(bias), act);
				break;
			case DTYPE_FLOAT32:
				launch_add_bias_act(shape, getPointer<float>(input), getPointer<float>(bias), act);
				break;
			default:
				break;
		}

//		const int first_dim = get_first_dim(shape);
//		const int last_dim = get_last_dim(shape);
//
//		for (int i = 0; i < first_dim; i++)
//		{
//			float *ptr = getPointer<float>(input) + i * last_dim;
//			for (int j = 0; j < last_dim; j++)
//			{
//				float tmp = ptr[j] + getPointer<float>(bias)[j];
//				if (act == ACTIVATION_RELU)
//					tmp = std::max(0.0f, tmp);
//				ptr[j] = tmp;
//			}
//			if (act == ACTIVATION_SOFTMAX)
//				kernel_softmax_in_place(ptr, last_dim);
//		}
	}
}

