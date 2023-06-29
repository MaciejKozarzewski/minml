/*
 * add_bias_act.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include "../kernel_definitions.hpp"
#include <minml/backend/backend_utils.hpp>
#include "../ComputeConfig.hpp"
#include "activations.hpp"

#include "../vectors/vectors.hpp"

#include <cmath>
#include <iostream>

namespace
{
	using namespace ml;
	using namespace SIMD_NAMESPACE;

	template<typename DT, typename CT>
	void kernel_softmax_3_channels(void *dst, const void *src, int first_dim)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const DT *src_ptr = getPointer<DT>(src);
		DT *dst_ptr = getPointer<DT>(dst);

		const int volume = first_dim * 3;
		constexpr int stride = Vector<CT>::size() * 3;

		float workspace[stride];
		for (int i = 0; i < volume; i += stride)
		{
			int remaining_elements = volume - i;
			for (int j = 0; j < 3; j++, remaining_elements -= Vector<CT>::size())
				if (remaining_elements > 0)
				{
					const Vector<CT> tmp(src_ptr + j * Vector<CT>::size(), std::min(Vector<CT>::size(), remaining_elements));
					tmp.store(workspace + j * Vector<CT>::size());
				}
				else
					Vector<CT>::zero().store(workspace + j * Vector<CT>::size());

			for (int j = 0; j < Vector<CT>::size(); j++)
			{
				float x0 = workspace[j * 3 + 0];
				float x1 = workspace[j * 3 + 1];
				float x2 = workspace[j * 3 + 2];

				const float max_value = std::max(x0, std::max(x1, x2));
				x0 = std::exp(x0 - max_value);
				x1 = std::exp(x1 - max_value);
				x2 = std::exp(x2 - max_value);

				const float inv_sum = 1.0f / (x0 + x1 + x2);
				workspace[j * 3 + 0] = x0 * inv_sum;
				workspace[j * 3 + 1] = x1 * inv_sum;
				workspace[j * 3 + 2] = x2 * inv_sum;
			}

			remaining_elements = volume - i;
			for (int j = 0; j < 3; j++, remaining_elements -= Vector<CT>::size())
				if (remaining_elements > 0)
				{
					const Vector<CT> tmp(workspace + j * Vector<CT>::size());
					tmp.store(dst_ptr + j * Vector<CT>::size(), std::min(Vector<CT>::size(), remaining_elements));
				}
			src_ptr += stride;
			dst_ptr += stride;
		}
	}

	template<typename DT, typename CT>
	void kernel_softmax(void *dst, const void *src, int first_dim, int last_dim, void *workspace)
	{
		static_assert(std::is_same<CT, float>::value);

		assert(dst != nullptr);
		assert(src != nullptr);
		const DT *src_ptr = getPointer<DT>(src);
		DT *dst_ptr = getPointer<DT>(dst);
		CT *workspace_ptr = getPointer<CT>(workspace);

		for (int i = 0; i < first_dim; i++)
		{
			for (int j = 0; j < last_dim; j += Vector<CT>::size())
			{
				const int elements = std::min(Vector<CT>::size(), last_dim - j);
				Vector<CT> tmp(src_ptr + j, elements);
				tmp.store(workspace_ptr + j);
			}

			CT max_value = workspace_ptr[0];
			for (int j = 0; j < last_dim; j++)
				max_value = std::max(max_value, workspace_ptr[j]);

			CT sum = static_cast<CT>(0);
			for (int j = 0; j < last_dim; j++)
			{
				const CT tmp = exp(workspace_ptr[j] - max_value);
				sum += tmp;
				workspace_ptr[j] = tmp;
			}

			const Vector<CT> scale(1.0f / sum);
			for (int j = 0; j < last_dim; j += Vector<CT>::size())
			{
				const int elements = std::min(Vector<CT>::size(), last_dim - j);
				Vector<CT> tmp(workspace_ptr + j, elements);
				tmp *= scale;
				tmp.store(dst_ptr + j, elements);
			}
			src_ptr += last_dim;
			dst_ptr += last_dim;
		}
	}

	template<typename DT, typename CT>
	void kernel_sigmoid(void *dst, const void *src, int length)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const DT *src_ptr = getPointer<DT>(src);
		DT *dst_ptr = getPointer<DT>(dst);

		for (int i = 0; i < length; i += Vector<CT>::size())
		{
			const int elements = std::min(Vector<CT>::size(), length - i);
			Vector<CT> tmp(src_ptr + i, elements);
			tmp = Vector<CT>::one() / (Vector<CT>::one() + exp(-tmp));
			tmp.store(dst_ptr + i, elements);
		}
	}
	void kernel_sigmoid_backward(float *gradient_prev, const float *gradient_next, const float *output, int length)
	{
		for (int i = 0; i < length; i++)
			gradient_prev[i] = gradient_next[i] * output[i] * (1.0f - output[i]);
	}

	template<typename DT, typename CT>
	void kernel_tanh(void *dst, const void *src, int length)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const DT *src_ptr = getPointer<DT>(src);
		DT *dst_ptr = getPointer<DT>(dst);

		for (int i = 0; i < length; i += Vector<CT>::size())
		{
			const int elements = std::min(Vector<CT>::size(), length - i);
			Vector<CT> tmp(src_ptr + i, elements);
			tmp = tanh(tmp);
			tmp.store(dst_ptr + i, elements);
		}
	}
	void kernel_tanh_backward(float *gradient_prev, const float *gradient_next, const float *output, int length)
	{
		for (int i = 0; i < length; i++)
			gradient_prev[i] = gradient_next[i] * (1.0f - output[i]) * (1.0f + output[i]);
	}

	template<typename DT, typename CT>
	void kernel_relu(void *dst, const void *src, int length)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const DT *src_ptr = getPointer<DT>(src);
		DT *dst_ptr = getPointer<DT>(dst);

		for (int i = 0; i < length; i += Vector<CT>::size())
		{
			const int elements = std::min(Vector<CT>::size(), length - i);
			Vector<CT> tmp(src_ptr + i, elements);
			tmp = max(Vector<CT>::zero(), tmp);
			tmp.store(dst_ptr + i, elements);
		}
	}
	void kernel_relu_backward(float *gradient_prev, const float *gradient_next, const float *output, int length)
	{
		for (int i = 0; i < length; i++)
			gradient_prev[i] = (output[i] == 0.0f) ? 0.0f : gradient_next[i];
	}

	template<typename DT, typename CT, mlActivationType_t ACT>
	void kernel_add_bias_act(mlShape_t shape, void *input, const void *bias)
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
				tmp = activation_forward(ACT, tmp + b);
				tmp.store(input_ptr + j, elements);
			}
			input_ptr += last_dim;
		}
	}

	template<typename DT, typename CT>
	void launch_add_bias_act(mlShape_t shape, void *input, const void *bias, mlActivationType_t act)
	{
		switch (act)
		{
			default:
			case ACTIVATION_LINEAR:
				kernel_add_bias_act<DT, CT, ACTIVATION_LINEAR>(shape, input, bias);
				break;
			case ACTIVATION_SIGMOID:
				kernel_add_bias_act<DT, CT, ACTIVATION_SIGMOID>(shape, input, bias);
				break;
			case ACTIVATION_TANH:
				kernel_add_bias_act<DT, CT, ACTIVATION_TANH>(shape, input, bias);
				break;
			case ACTIVATION_RELU:
				kernel_add_bias_act<DT, CT, ACTIVATION_RELU>(shape, input, bias);
				break;
		}
	}
	template<typename DT, typename CT>
	void launch_softmax(mlContext_t context, mlShape_t shape, void *output, const void *input)
	{
		const int first_dim = get_first_dim(shape);
		const int last_dim = get_last_dim(shape);

		assert(cpu::Context::getWorkspaceSize(context) >= sizeof(float) * last_dim);
		float *workspace = cpu::Context::getWorkspace<float>(context);
		if (last_dim == 3)
			kernel_softmax_3_channels<DT, CT>(output, input, first_dim);
		else
			kernel_softmax<DT, CT>(output, input, first_dim, last_dim, workspace);
	}
}

namespace SIMD_NAMESPACE
{
	using namespace ml;

	void cpu_kernel_activation_forward(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *output, const void *input,
			mlActivationType_t act)
	{
		switch (act)
		{
			case ACTIVATION_LINEAR:
			{
				if (output != input)
					cpu_memcpy(context, output, 0, input, 0, size_of(dtype) * volume(shape));
				break;
			}
			case ACTIVATION_SIGMOID:
			{
				const ml::cpu::ComputeConfig cfg = ml::cpu::ComputeConfig::getBest(dtype);
				CREATE_KERNEL_TABLE(kernel_sigmoid);
				CALL_KERNEL(kernel_sigmoid, cfg)(output, input, volume(shape));
				break;
			}
			case ACTIVATION_TANH:
			{
				const ml::cpu::ComputeConfig cfg = ml::cpu::ComputeConfig::getBest(dtype);
				CREATE_KERNEL_TABLE(kernel_tanh);
				CALL_KERNEL(kernel_tanh, cfg)(output, input, volume(shape));
				break;
			}
			case ACTIVATION_RELU:
			{
				const ml::cpu::ComputeConfig cfg = ml::cpu::ComputeConfig::getBest(dtype);
				CREATE_KERNEL_TABLE(kernel_relu);
				CALL_KERNEL(kernel_relu, cfg)(output, input, volume(shape));
				break;
			}
			case ACTIVATION_SOFTMAX:
			{
				ml::cpu::ComputeConfig cfg = ml::cpu::ComputeConfig::getBest(dtype);
				cfg.compute_type = ml::cpu::Type::FP32;

				CREATE_KERNEL_TABLE(launch_softmax);
				CALL_KERNEL(launch_softmax, cfg)(context, shape, output, input);
				break;
			}
		}
	}
	void cpu_kernel_activation_backward(mlContext_t context, mlShape_t shape, void *gradient_prev, const void *gradient_next, const void *output,
			mlActivationType_t act)
	{
		switch (act)
		{
			case ACTIVATION_LINEAR:
			{
				if (gradient_prev != gradient_next)
					cpu_memcpy(context, gradient_prev, 0, gradient_next, 0, sizeof(float) * volume(shape));
				break;
			}
			case ACTIVATION_SIGMOID:
				kernel_sigmoid_backward(getPointer<float>(gradient_prev), getPointer<float>(gradient_next), getPointer<float>(output), volume(shape));
				break;
			case ACTIVATION_TANH:
				kernel_tanh_backward(getPointer<float>(gradient_prev), getPointer<float>(gradient_next), getPointer<float>(output), volume(shape));
				break;
			case ACTIVATION_RELU:
				kernel_relu_backward(getPointer<float>(gradient_prev), getPointer<float>(gradient_next), getPointer<float>(output), volume(shape));
				break;
			case ACTIVATION_SOFTMAX:
				if (gradient_prev != gradient_next)
					std::memcpy(gradient_prev, gradient_next, volume(shape) * sizeof(float));
				break;
		}
	}

	void cpu_kernel_add_bias_act(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *input, const void *bias, mlActivationType_t act)
	{
		const ml::cpu::ComputeConfig cfg = ml::cpu::ComputeConfig::getBest(dtype);

		CREATE_KERNEL_TABLE(launch_add_bias_act);
		CALL_KERNEL(launch_add_bias_act, cfg)(shape, input, bias, act);
	}
}

