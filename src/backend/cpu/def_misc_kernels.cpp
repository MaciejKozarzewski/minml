/*
 * def_misc_kernels.cpp
 *
 *  Created on: Oct 19, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include "misc_kernels.hpp"
#include "fp16.hpp"

#include <cstddef>
#include <cstring>
#include <cmath>
#include <cassert>

namespace
{
	using namespace ml::cpu;

	float sigmoid(float x) noexcept
	{
		return 1.0f / (1.0f + std::exp(-x));
	}
	float relu(float x) noexcept
	{
		return std::max(0.0f, x);
	}

	template<typename SrcT, typename DstT>
	DstT convert(SrcT x) noexcept
	{
		return static_cast<DstT>(x);
	}
	template<>
	float16 convert(float x) noexcept
	{
		return convert_fp32_to_fp16(x);
	}
	template<>
	float convert(float16 x) noexcept
	{
		return convert_fp16_to_fp32(x);
	}

	template<typename T>
	void kernel_softmax_3_channels(void *dst, const void *src, int first_dim)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const T *src_ptr = ml::getPointer<T>(src);
		T *dst_ptr = ml::getPointer<T>(dst);

		for (int i = 0; i < first_dim; i++)
		{
			float x0 = convert<T, float>(src_ptr[0]);
			float x1 = convert<T, float>(src_ptr[1]);
			float x2 = convert<T, float>(src_ptr[2]);

			const float max_value = std::max(x0, std::max(x1, x2));
			x0 = std::exp(x0 - max_value);
			x1 = std::exp(x1 - max_value);
			x2 = std::exp(x2 - max_value);

			const float inv_sum = 1.0f / (x0 + x1 + x2);
			dst_ptr[0] = convert<float, T>(x0 * inv_sum);
			dst_ptr[1] = convert<float, T>(x1 * inv_sum);
			dst_ptr[2] = convert<float, T>(x2 * inv_sum);

			src_ptr += 3;
			dst_ptr += 3;
		}
	}
	template<typename T>
	void kernel_softmax(void *dst, const void *src, int first_dim, int last_dim, void *workspace)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const T *src_ptr = ml::getPointer<T>(src);
		T *dst_ptr = ml::getPointer<T>(dst);
		float *workspace_ptr = ml::getPointer<float>(workspace);

		for (int i = 0; i < first_dim; i++)
		{
			for (int j = 0; j < last_dim; j++)
				workspace_ptr[j] = convert<T, float>(src_ptr[j]);

			float max_value = workspace_ptr[0];
			for (int j = 0; j < last_dim; j++)
				max_value = std::max(max_value, workspace_ptr[j]);

			float sum = 0.0f;
			for (int j = 0; j < last_dim; j++)
			{
				const float tmp = std::exp(workspace_ptr[j] - max_value);
				sum += tmp;
				workspace_ptr[j] = tmp;
			}

			const float scale = 1.0f / sum;
			for (int j = 0; j < last_dim; j++)
				dst_ptr[j] = convert<float, T>(workspace_ptr[j] * scale);
			src_ptr += last_dim;
			dst_ptr += last_dim;
		}
	}
	template<typename T>
	void kernel_activation_forward(void *dst, const void *src, size_t elements, ml::mlActivationType_t activation)
	{
		T *dst_ptr = ml::getPointer<T>(dst);
		const T *src_ptr = ml::getPointer<T>(src);
		switch (activation)
		{
			case ml::ACTIVATION_LINEAR:
				if (dst != src)
					std::memcpy(dst, src, sizeof(float) * elements);
				break;
			case ml::ACTIVATION_SIGMOID:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert<float, T>(sigmoid(convert<T, float>(src_ptr[i])));
				break;
			case ml::ACTIVATION_TANH:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert<float, T>(std::tanh(convert<T, float>(src_ptr[i])));
				break;
			case ml::ACTIVATION_RELU:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert<float, T>(relu(convert<T, float>(src_ptr[i])));
				break;
			default:
				break;
		}
	}

	template<typename T, ml::mlActivationType_t ACT>
	void kernel_add_bias_act(void *input, const void *bias, int first_dim, int last_dim)
	{
		T *input_ptr = ml::getPointer<T>(input);
		const T *bias_ptr = ml::getPointer<T>(bias);

		for (int i = 0; i < first_dim; i++)
		{
			for (int j = 0; j < last_dim; j++)
			{
				float tmp = convert<T, float>(input_ptr[j]) + convert<T, float>(bias_ptr[j]);
				switch (ACT)
				{
					default:
					case ml::ACTIVATION_LINEAR:
						break;
					case ml::ACTIVATION_SIGMOID:
						tmp = sigmoid(tmp);
						break;
					case ml::ACTIVATION_TANH:
						tmp = std::tanh(tmp);
						break;
					case ml::ACTIVATION_RELU:
						tmp = relu(tmp);
						break;
				}
				input_ptr[j] = convert<float, T>(tmp);
			}
			input_ptr += last_dim;
		}
	}
}

namespace ml
{
	namespace cpu
	{
		void def_kernel_convert_fp32_to_fp16(void *dst, const void *src, size_t elements)
		{
			for (size_t i = 0; i < elements; i++)
				reinterpret_cast<float16*>(dst)[i] = convert_fp32_to_fp16(reinterpret_cast<const float*>(src)[i]);
		}
		void def_kernel_convert_fp16_to_fp32(void *dst, const void *src, size_t elements)
		{
			for (size_t i = 0; i < elements; i++)
				reinterpret_cast<float*>(dst)[i] = convert_fp16_to_fp32(reinterpret_cast<const float16*>(src)[i]);
		}

		void def_kernel_softmax_3_channels_fp32(void *dst, const void *src, int first_dim)
		{
			kernel_softmax_3_channels<float>(dst, src, first_dim);
		}
		void def_kernel_softmax_3_channels_fp16(void *dst, const void *src, int first_dim)
		{
			kernel_softmax_3_channels<float16>(dst, src, first_dim);
		}
		void def_kernel_softmax_fp32(void *dst, const void *src, int first_dim, int last_dim, void *workspace)
		{
			kernel_softmax<float>(dst, src, first_dim, last_dim, workspace);
		}
		void def_kernel_softmax_fp16(void *dst, const void *src, int first_dim, int last_dim, void *workspace)
		{
			kernel_softmax<float16>(dst, src, first_dim, last_dim, workspace);
		}

		void def_kernel_activation_forward_fp32(void *dst, const void *src, size_t elements, mlActivationType_t activation)
		{
			kernel_activation_forward<float>(dst, src, elements, activation);
		}
		void def_kernel_activation_forward_fp16(void *dst, const void *src, size_t elements, mlActivationType_t activation)
		{
			kernel_activation_forward<float16>(dst, src, elements, activation);
		}

		void def_kernel_activation_backward_fp32(void *gradient_prev, const void *gradient_next, const void *output, size_t elements,
				mlActivationType_t activation)
		{
			float *prev_ptr = reinterpret_cast<float*>(gradient_prev);
			const float *next_ptr = reinterpret_cast<const float*>(gradient_next);
			const float *out_ptr = reinterpret_cast<const float*>(output);
			switch (activation)
			{
				case ACTIVATION_LINEAR:
					if (gradient_prev != gradient_next)
						std::memcpy(gradient_prev, gradient_next, sizeof(float) * elements);
					break;
				case ACTIVATION_SIGMOID:
					for (size_t i = 0; i < elements; i++)
						prev_ptr[i] = next_ptr[i] * out_ptr[i] * (1.0f - out_ptr[i]);
					break;
				case ACTIVATION_TANH:
					for (size_t i = 0; i < elements; i++)
						prev_ptr[i] = next_ptr[i] * (1.0f + out_ptr[i]) * (1.0f - out_ptr[i]);
					break;
				case ACTIVATION_RELU:
					for (size_t i = 0; i < elements; i++)
						prev_ptr[i] = (out_ptr[i] == 0.0f) ? 0.0f : next_ptr[i];
					break;
				case ACTIVATION_SOFTMAX:
					if (gradient_prev != gradient_next)
						std::memcpy(gradient_prev, gradient_next, sizeof(float) * elements);
					break;
				default:
					break;
			}
		}

		void def_kernel_add_bias_act_fp32(void *input, const void *bias, int first_dim, int last_dim, mlActivationType_t act)
		{
			switch (act)
			{
				default:
				case ACTIVATION_LINEAR:
					kernel_add_bias_act<float, ACTIVATION_LINEAR>(input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_SIGMOID:
					kernel_add_bias_act<float, ACTIVATION_SIGMOID>(input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_TANH:
					kernel_add_bias_act<float, ACTIVATION_TANH>(input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_RELU:
					kernel_add_bias_act<float, ACTIVATION_RELU>(input, bias, first_dim, last_dim);
					break;
			}
		}
		void def_kernel_add_bias_act_fp16(void *input, const void *bias, int first_dim, int last_dim, mlActivationType_t act)
		{
			switch (act)
			{
				default:
				case ACTIVATION_LINEAR:
					kernel_add_bias_act<float16, ACTIVATION_LINEAR>(input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_SIGMOID:
					kernel_add_bias_act<float16, ACTIVATION_SIGMOID>(input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_TANH:
					kernel_add_bias_act<float16, ACTIVATION_TANH>(input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_RELU:
					kernel_add_bias_act<float16, ACTIVATION_RELU>(input, bias, first_dim, last_dim);
					break;
			}
		}
	}
}

