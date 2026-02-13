/*
 * def_misc_kernels.cpp
 *
 *  Created on: Oct 19, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>
#include <minml/backend/cpu_backend.h>

#include "misc_kernels.hpp"
#include "common_math.hpp"
#include "fp16.hpp"

#include <cstddef>
#include <cstring>
#include <cmath>
#include <cassert>
#include <iostream>

namespace
{
	using namespace ml::cpu;

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
	template<>
	float16 convert(double x) noexcept
	{
		return convert_fp32_to_fp16(static_cast<float>(x));
	}
	template<>
	double convert(float16 x) noexcept
	{
		return static_cast<double>(convert_fp16_to_fp32(x));
	}

	template<int ACT, typename T>
	T activation_forward(T x) noexcept
	{
		switch (ACT)
		{
			default:
			case ml::ACTIVATION_LINEAR:
				return x;
			case ml::ACTIVATION_SIGMOID:
				return sigmoid(x);
			case ml::ACTIVATION_TANH:
				return std::tanh(x);
			case ml::ACTIVATION_RELU:
				return relu(x);
			case ml::ACTIVATION_LEAKY_RELU:
				return leaky_relu(x);
			case ml::ACTIVATION_EXP:
				return std::exp(x);
		}
	}
	template<int ACT, typename T>
	T activation_backward(T gradient, T output) noexcept
	{
		switch (ACT)
		{
			default:
			case ml::ACTIVATION_LINEAR:
				return gradient;
			case ml::ACTIVATION_SIGMOID:
				return gradient * output * (1.0f - output);
			case ml::ACTIVATION_TANH:
				return gradient * (1.0f - square(output));
			case ml::ACTIVATION_RELU:
				return (output > 0.0f) ? gradient : 0.0f;
			case ml::ACTIVATION_LEAKY_RELU:
				return (output > 0.0f) ? gradient : 0.1f * gradient;
			case ml::ACTIVATION_EXP:
				return gradient * output;
		}
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
					std::memcpy(dst, src, sizeof(T) * elements);
				break;
			case ml::ACTIVATION_SIGMOID:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert<float, T>(activation_forward<ml::ACTIVATION_SIGMOID>(convert<T, float>(src_ptr[i])));
				break;
			case ml::ACTIVATION_TANH:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert<float, T>(activation_forward<ml::ACTIVATION_TANH>(convert<T, float>(src_ptr[i])));
				break;
			case ml::ACTIVATION_RELU:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert<float, T>(activation_forward<ml::ACTIVATION_RELU>(convert<T, float>(src_ptr[i])));
				break;
			case ml::ACTIVATION_LEAKY_RELU:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert<float, T>(activation_forward<ml::ACTIVATION_LEAKY_RELU>(convert<T, float>(src_ptr[i])));
				break;
			case ml::ACTIVATION_EXP:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert<float, T>(activation_forward<ml::ACTIVATION_EXP>(convert<T, float>(src_ptr[i])));
				break;

			default:
				break;
		}
	}
	template<typename T>
	void kernel_activation_backward(void *dx, const void *dy, const void *x, const void *y, size_t elements, ml::mlActivationType_t activation)
	{
		T *prev_ptr = reinterpret_cast<T*>(dx);
		const T *next_ptr = reinterpret_cast<const T*>(dy);
		const T *in_ptr = reinterpret_cast<const T*>(x);
		const T *out_ptr = reinterpret_cast<const T*>(y);
		switch (activation)
		{
			case ml::ACTIVATION_LINEAR:
				if (dx != dy)
					std::memcpy(dx, dy, sizeof(float) * elements);
				break;
			case ml::ACTIVATION_SIGMOID:
				for (size_t i = 0; i < elements; i++)
					prev_ptr[i] = next_ptr[i] * out_ptr[i] * (1.0f - out_ptr[i]);
				break;
			case ml::ACTIVATION_TANH:
				for (size_t i = 0; i < elements; i++)
					prev_ptr[i] = next_ptr[i] * (1.0f + out_ptr[i]) * (1.0f - out_ptr[i]);
				break;
			case ml::ACTIVATION_RELU:
				for (size_t i = 0; i < elements; i++)
					prev_ptr[i] = (out_ptr[i] == 0.0f) ? 0.0f : next_ptr[i];
				break;
			case ml::ACTIVATION_LEAKY_RELU:
				for (size_t i = 0; i < elements; i++)
					prev_ptr[i] = (out_ptr[i] < 0.0f) ? 0.1f * next_ptr[i] : next_ptr[i];
				break;
			case ml::ACTIVATION_EXP:
				for (size_t i = 0; i < elements; i++)
					prev_ptr[i] = out_ptr[i] * next_ptr[i];
				break;
			default:
				break;
		}
	}

	template<typename T, ml::mlActivationType_t ACT>
	void kernel_add_bias_act(void *output, const void *input, const void *bias, int first_dim, int last_dim)
	{
		T *output_ptr = ml::getPointer<T>(output);
		const T *input_ptr = ml::getPointer<T>(input);
		const T *bias_ptr = ml::getPointer<T>(bias);

		for (int i = 0; i < first_dim; i++)
		{
			for (int j = 0; j < last_dim; j++)
			{
				const float tmp = convert<T, float>(input_ptr[j]) + convert<T, float>(bias_ptr[j]);
				output_ptr[j] = convert<float, T>(activation_forward<ACT, T>(tmp));
			}
			output_ptr += last_dim;
			input_ptr += last_dim;
		}
	}

	template<ml::mlActivationType_t ACT>
	void kernel_global_broadcasting(ml::mlShape_t shape, void *output, const void *input, const void *bias)
	{
		const int batch_size = shape.dim[0];
		const int hw = shape.dim[1] * shape.dim[2];
		const int channels = shape.dim[3];

		const float *input_ptr = reinterpret_cast<const float*>(input);
		const float *bias_ptr = reinterpret_cast<const float*>(bias);
		float *output_ptr = reinterpret_cast<float*>(output);

		for (int i = 0; i < batch_size; i++)
		{
			for (int j = 0; j < hw; j++)
			{
				for (int k = 0; k < channels; k++)
				{
					const float tmp = input_ptr[k] + bias_ptr[k];
					output_ptr[k] = activation_forward<ACT, float>(tmp);
				}
				input_ptr += channels;
				output_ptr += channels;
			}
			bias_ptr += channels;
		}
	}

	template<typename SrcT, typename DstT>
	void convert_kernel(void *dst, const void *src, size_t elements)
	{
		for (size_t i = 0; i < elements; i++)
			reinterpret_cast<DstT*>(dst)[i] = convert<SrcT, DstT>(reinterpret_cast<const SrcT*>(src)[i]);
	}
}

namespace ml
{
	namespace cpu
	{
		void def_kernel_convert_fp32_to_fp16(void *dst, const void *src, size_t elements)
		{
			convert_kernel<float, float16>(dst, src, elements);
		}
		void def_kernel_convert_fp16_to_fp32(void *dst, const void *src, size_t elements)
		{
			convert_kernel<float16, float>(dst, src, elements);
		}
		void def_kernel_convert_fp64_to_fp16(void *dst, const void *src, size_t elements)
		{
			convert_kernel<double, float16>(dst, src, elements);
		}
		void def_kernel_convert_fp16_to_fp64(void *dst, const void *src, size_t elements)
		{
			convert_kernel<float16, double>(dst, src, elements);
		}
		void def_kernel_convert_fp64_to_fp32(void *dst, const void *src, size_t elements)
		{
			convert_kernel<double, float>(dst, src, elements);
		}
		void def_kernel_convert_fp32_to_fp64(void *dst, const void *src, size_t elements)
		{
			convert_kernel<float, double>(dst, src, elements);
		}

		void def_kernel_softmax_3_channels_fp32(void *dst, const void *src, int first_dim)
		{
			kernel_softmax_3_channels<float>(dst, src, first_dim);
		}
		void def_kernel_softmax_fp32(void *dst, const void *src, int first_dim, int last_dim, void *workspace)
		{
			kernel_softmax<float>(dst, src, first_dim, last_dim, workspace);
		}

		void def_kernel_activation_forward_fp32(void *dst, const void *src, size_t elements, mlActivationType_t activation)
		{
			kernel_activation_forward<float>(dst, src, elements, activation);
		}
		void def_kernel_activation_forward_fp64(void *dst, const void *src, size_t elements, mlActivationType_t activation)
		{
			kernel_activation_forward<double>(dst, src, elements, activation);
		}

		void def_kernel_activation_backward_fp32(void *gradient_prev, const void *gradient_next, const void *input, const void *output,
				size_t elements, mlActivationType_t activation)
		{
			kernel_activation_backward<float>(gradient_prev, gradient_next, input, output, elements, activation);
		}
		void def_kernel_activation_backward_fp64(void *gradient_prev, const void *gradient_next, const void *input, const void *output,
				size_t elements, mlActivationType_t activation)
		{
			kernel_activation_backward<double>(gradient_prev, gradient_next, input, output, elements, activation);
		}

		void def_kernel_add_bias_act_fp32(void *output, const void *input, const void *bias, int first_dim, int last_dim, mlActivationType_t act)
		{
			switch (act)
			{
				default:
				case ACTIVATION_LINEAR:
					kernel_add_bias_act<float, ACTIVATION_LINEAR>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_SIGMOID:
					kernel_add_bias_act<float, ACTIVATION_SIGMOID>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_TANH:
					kernel_add_bias_act<float, ACTIVATION_TANH>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_RELU:
					kernel_add_bias_act<float, ACTIVATION_RELU>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_LEAKY_RELU:
					kernel_add_bias_act<float, ACTIVATION_LEAKY_RELU>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_EXP:
					kernel_add_bias_act<float, ACTIVATION_EXP>(output, input, bias, first_dim, last_dim);
					break;
			}
		}
		void def_kernel_add_bias_act_fp64(void *output, const void *input, const void *bias, int first_dim, int last_dim, mlActivationType_t act)
		{
			switch (act)
			{
				default:
				case ACTIVATION_LINEAR:
					kernel_add_bias_act<double, ACTIVATION_LINEAR>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_SIGMOID:
					kernel_add_bias_act<double, ACTIVATION_SIGMOID>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_TANH:
					kernel_add_bias_act<double, ACTIVATION_TANH>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_RELU:
					kernel_add_bias_act<double, ACTIVATION_RELU>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_LEAKY_RELU:
					kernel_add_bias_act<double, ACTIVATION_LEAKY_RELU>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_EXP:
					kernel_add_bias_act<double, ACTIVATION_EXP>(output, input, bias, first_dim, last_dim);
					break;
			}
		}

	} /* namespace cpu */
} /* namespace ml */

