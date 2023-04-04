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

	template<typename CT, typename DT>
	void kernel_softmax_3_channels(DT *dst, const DT *src, int first_dim)
	{
//		assert(dst != nullptr);
//		assert(src != nullptr);
//		const int volume = first_dim * 3;
//		constexpr int stride = Vector<CT>::length * 3;
//
//		float workspace[stride];
//		for (int i = 0; i < volume; i += stride)
//		{
//			int remaining_elements = volume - i;
//			for (int j = 0; j < 3; j++, remaining_elements -= Vector<CT>::length)
//				if (remaining_elements > 0)
//				{
//					const Vector<CT> tmp(src + j * Vector<CT>::length, remaining_elements);
//					tmp.store(workspace + j * Vector<CT>::length);
//				}
//				else
//					Vector<CT>::zero().store(workspace + j * Vector<CT>::length);
//
//			for (int j = 0; j < Vector<CT>::length; j++)
//			{
//				float x0 = workspace[j * 3 + 0];
//				float x1 = workspace[j * 3 + 1];
//				float x2 = workspace[j * 3 + 2];
//
//				const float max_value = std::max(x0, std::max(x1, x2));
//				x0 = std::exp(x0 - max_value);
//				x1 = std::exp(x1 - max_value);
//				x2 = std::exp(x2 - max_value);
//
//				const float inv_sum = 1.0f / (x0 + x1 + x2);
//				workspace[j * 3 + 0] = x0 * inv_sum;
//				workspace[j * 3 + 1] = x1 * inv_sum;
//				workspace[j * 3 + 2] = x2 * inv_sum;
//			}
//
//			remaining_elements = volume - i;
//			for (int j = 0; j < 3; j++, remaining_elements -= Vector<CT>::length)
//				if (remaining_elements > 0)
//				{
//					const Vector<CT> tmp(workspace + j * Vector<CT>::length);
//					tmp.store(dst + j * Vector<CT>::length, remaining_elements);
//				}
//			src += stride;
//			dst += stride;
//		}
	}

	template<typename CT, typename DT>
	void kernel_softmax(DT *dst, const DT *src, int first_dim, int last_dim, float *workspace)
	{
//		assert(dst != nullptr);
//		assert(src != nullptr);
//		for (int i = 0; i < first_dim; i++)
//		{
//			const Vector<CT> first(src[0]);
//
//			Vector<CT> max_value = first;
//			for (int j = 0; j < last_dim; j += Vector<CT>::length)
//			{
//				Vector<CT> tmp(src + j, last_dim - j);
//				tmp.cutoff(last_dim - j, first);
//				max_value = max(max_value, tmp);
//				tmp.store(workspace + j);
//			}
//			max_value = horizontal_max(max_value);
//
//			Vector<CT> sum = Vector<CT>::zero();
//			for (int j = 0; j < last_dim; j += Vector<CT>::length)
//			{
//				Vector<CT> tmp(workspace + j, last_dim - j);
//				tmp = exp(tmp - max_value);
//				tmp.cutoff(last_dim - j, Vector<CT>::zero());
//				sum += tmp;
//				tmp.store(workspace + j, last_dim - j);
//			}
//
//			sum = Vector<CT>::one() / Vector<CT>(horizontal_add(sum));
//			for (int j = 0; j < last_dim; j += Vector<CT>::length)
//			{
//				Vector<CT> tmp(workspace + j, last_dim - j);
//				tmp *= sum;
//				tmp.store(dst + j, last_dim - j);
//			}
//			src += last_dim;
//			dst += last_dim;
//		}
	}

	template<typename CT, typename DT>
	void kernel_sigmoid(DT *dst, const DT *src, int length)
	{
//		for (int i = 0; i < length; i += Vector<CT>::length)
//		{
//			Vector<CT> tmp(src + i, length - i);
//			tmp = Vector<CT>::one() / (Vector<CT>::one() + exp(-tmp));
//			tmp.store(dst + i, length - i);
//		}
	}
	void kernel_sigmoid_backward(float *gradient_prev, const float *gradient_next, const float *output, int length)
	{
//		for (int i = 0; i < length; i++)
//			gradient_prev[i] = gradient_next[i] * output[i] * (1.0f - output[i]);
	}

	template<typename CT, typename DT>
	void kernel_tanh(DT *dst, const DT *src, int length)
	{
//		for (int i = 0; i < length; i += Vector<CT>::length)
//		{
//			Vector<CT> tmp(src + i, length - i);
//			tmp = tanh(tmp);
//			tmp.store(dst + i, length - i);
//		}
	}
	void kernel_tanh_backward(float *gradient_prev, const float *gradient_next, const float *output, int length)
	{
//		for (int i = 0; i < length; i++)
//			gradient_prev[i] = gradient_next[i] * (1.0f - output[i]) * (1.0f + output[i]);
	}

	template<typename CT, typename DT>
	void kernel_relu(DT *dst, const DT *src, int length)
	{
//		for (int i = 0; i < length; i += Vector<CT>::length)
//		{
//			Vector<CT> tmp(src + i, length - i);
//			tmp = max(Vector<CT>::zero(), tmp);
//			tmp.store(dst + i, length - i);
//		}
	}
	void kernel_relu_backward(float *gradient_prev, const float *gradient_next, const float *output, int length)
	{
//		for (int i = 0; i < length; i++)
//			gradient_prev[i] = (output[i] == 0.0f) ? 0.0f : gradient_next[i];
	}

	template<typename CT, typename DT>
	void kernel_add_bias_act(mlShape_t shape, void *input, const void *bias, mlActivationType_t act)
	{
		assert(input != nullptr);
		assert(bias != nullptr);
		DT *input_ptr = getPointer<DT>(input);
		const DT *bias_ptr = getPointer<DT>(bias);

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		for (int i = 0; i < first_dim; i++)
		{
			for (int j = 0; j < last_dim; j += Vector<CT>::size())
			{
				const int elements = std::min(Vector<CT>::size(), last_dim - j);
				Vector<CT> tmp(input_ptr + j, elements);
				Vector<CT> b(bias_ptr + j, elements);
				tmp += b;
				if (act == ACTIVATION_RELU)
					tmp = max(Vector<CT>::zero(), tmp);
				if (act == ACTIVATION_TANH)
					tmp = tanh(tmp);
				if (act == ACTIVATION_SIGMOID)
					tmp = Vector<CT>::one() / (Vector<CT>::one() + exp(-tmp));
				tmp.store(input_ptr + j, elements);
			}
			input_ptr += last_dim;
		}
	}
}

namespace SIMD_NAMESPACE
{
	using namespace ml;

	void cpu_kernel_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input,
			mlActivationType_t act)
	{
//		switch (act)
//		{
//			case ACTIVATION_LINEAR:
//				break;
//			case ACTIVATION_SIGMOID:
//			{
//				switch (dtype)
//				{
//					case DTYPE_BFLOAT16:
//						kernel_sigmoid(getPointer<bfloat16>(output), getPointer<bfloat16>(input), volume(shape));
//						break;
//					case DTYPE_FLOAT16:
//						kernel_sigmoid(getPointer<float16>(output), getPointer<float16>(input), volume(shape));
//						break;
//					case DTYPE_FLOAT32:
//						kernel_sigmoid<float, float>(getPointer<float>(output), getPointer<float>(input), volume(shape));
//						break;
//					default:
//						break;
//				}
//				break;
//			}
//			case ACTIVATION_TANH:
//			{
//				switch (dtype)
//				{
//					case DTYPE_BFLOAT16:
//						kernel_tanh(getPointer<bfloat16>(output), getPointer<bfloat16>(input), volume(shape));
//						break;
//					case DTYPE_FLOAT16:
//						kernel_tanh(getPointer<float16>(output), getPointer<float16>(input), volume(shape));
//						break;
//					case DTYPE_FLOAT32:
//						kernel_tanh<float, float>(getPointer<float>(output), getPointer<float>(input), volume(shape));
//						break;
//					default:
//						break;
//				}
//				break;
//			}
//			case ACTIVATION_RELU:
//			{
//				switch (dtype)
//				{
//					case DTYPE_BFLOAT16:
//						kernel_relu(getPointer<bfloat16>(output), getPointer<bfloat16>(input), volume(shape));
//						break;
//					case DTYPE_FLOAT16:
//						kernel_relu(getPointer<float16>(output), getPointer<float16>(input), volume(shape));
//						break;
//					case DTYPE_FLOAT32:
//						kernel_relu<float, float>(getPointer<float>(output), getPointer<float>(input), volume(shape));
//						break;
//					default:
//						break;
//				}
//				break;
//			}
//			case ACTIVATION_SOFTMAX:
//			{
//				assert(shape.rank == 2);
//				const int first_dim = get_first_dim(shape);
//				const int last_dim = get_last_dim(shape);
//
//				assert(cpu::Context::getWorkspaceSize(context) >= sizeof(float) * last_dim);
//				float *workspace = cpu::Context::getWorkspace<float>(context);
//				switch (dtype)
//				{
//					case DTYPE_BFLOAT16:
//						if (last_dim == 3)
//							kernel_softmax_3_channels<float, bfloat16>(getPointer<bfloat16>(output), getPointer<bfloat16>(input), first_dim);
//						else
//							kernel_softmax(getPointer<bfloat16>(output), getPointer<bfloat16>(input), first_dim, last_dim, workspace);
//						break;
//					case DTYPE_FLOAT16:
//						if (last_dim == 3)
//							kernel_softmax_3_channels<float, float16>(getPointer<float16>(output), getPointer<float16>(input), first_dim);
//						else
//							kernel_softmax(getPointer<float16>(output), getPointer<float16>(input), first_dim, last_dim, workspace);
//						break;
//					case DTYPE_FLOAT32:
//						if (last_dim == 3)
//							kernel_softmax_3_channels<float, float>(getPointer<float>(output), getPointer<float>(input), first_dim);
//						else
//							kernel_softmax<float, float>(getPointer<float>(output), getPointer<float>(input), first_dim, last_dim, workspace);
//						break;
//					default:
//						break;
//				}
//				break;
//			}
//		}
	}
	void cpu_kernel_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
			mlActivationType_t act)
	{
//		switch (act)
//		{
//			case ACTIVATION_LINEAR:
//				break;
//			case ACTIVATION_SIGMOID:
//				kernel_sigmoid_backward(getPointer<float>(gradient_prev), getPointer<float>(gradient_next), getPointer<float>(output), volume(shape));
//				break;
//			case ACTIVATION_TANH:
//				kernel_tanh_backward(getPointer<float>(gradient_prev), getPointer<float>(gradient_next), getPointer<float>(output), volume(shape));
//				break;
//			case ACTIVATION_RELU:
//				kernel_relu_backward(getPointer<float>(gradient_prev), getPointer<float>(gradient_next), getPointer<float>(output), volume(shape));
//				break;
//			case ACTIVATION_SOFTMAX:
//				if (gradient_prev != gradient_next)
//					std::memcpy(gradient_prev, gradient_next, volume(shape) * sizeof(float));
//				break;
//		}
	}

	void cpu_kernel_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias, mlActivationType_t act)
	{
		switch (dtype)
		{
			case DTYPE_BFLOAT16:
				kernel_add_bias_act<float, bfloat16>(shape, input, bias, act);
				break;
			case DTYPE_FLOAT16:
				kernel_add_bias_act<float, float16>(shape, input, bias, act);
				break;
			case DTYPE_FLOAT32:
				kernel_add_bias_act<float, float>(shape, input, bias, act);
				break;
			default:
				break;
		}
	}
}

