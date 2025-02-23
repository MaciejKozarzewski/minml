/*
 * training.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "common_math.hpp"

#include <cmath>
#include <cassert>
#include <iostream>

namespace
{
	using namespace ml::cpu;

	float bounded_pow(float x, float y, float min) noexcept
	{
		assert(0 < x && x < 1);
		assert(min > 0);
		const float max_y = std::log(min) / std::log(x);
		return (y >= max_y) ? 0.0f : std::pow(x, y);
	}
	template<typename T>
	void add_kernel(T *dst, T alpha1, const T *src1, T alpha2, const T *src2, int elements)
	{
		for (int i = 0; i < elements; i++)
			dst[i] = alpha1 * src1[i] + alpha2 * src2[i];
	}
	template<typename T>
	void multiply_kernel(T *dst, const T *src1, const T *src2, int elements)
	{
		for (int i = 0; i < elements; i++)
			dst[i] = src1[i] * src2[i];
	}

	struct float3
	{
			float x, y, z;
	};

	float get_expectation(const float3 &value) noexcept
	{
		return value.x + 0.5 * value.y;
	}
	float3 softmax(const float3 &value) noexcept
	{
		const float max_value = std::max(value.x, std::max(value.y, value.z));
		const float x = std::exp(value.x - max_value);
		const float y = std::exp(value.x - max_value);
		const float z = std::exp(value.x - max_value);
		const float inv_sum = 1.0f / (x + y + z);
		return float3 { x * inv_sum, y * inv_sum, z * inv_sum };
	}
	float cross_entropy(const float3 &output, const float3 &target) noexcept
	{
		const float x = ml::cpu::cross_entropy(output.x, target.x);
		const float y = ml::cpu::cross_entropy(output.y, target.y);
		const float z = ml::cpu::cross_entropy(output.z, target.z);
		return x + y + z;
	}
}

namespace ml
{
	void cpu_emulate_low_precision(mlContext_t context, mlShape_t shape, void *dst, const void *src)
	{
		const int elements = volume(shape);
		for (int i = 0; i < elements; i++)
		{
			uint32_t tmp = reinterpret_cast<const uint32_t*>(src)[i];
			reinterpret_cast<uint32_t*>(dst)[i] = tmp & 0xFFFFF000u;
		}
	}
	void cpu_multiply_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2)
	{
		assert(dst != nullptr);
		assert(src1 != nullptr);
		assert(src2 != nullptr);

		const int elements = volume(shape);

		switch (dtype)
		{
			case DTYPE_FLOAT32:
				multiply_kernel(getPointer<float>(dst), getPointer<float>(src1), getPointer<float>(src2), elements);
				break;
			case DTYPE_FLOAT64:
				multiply_kernel(getPointer<double>(dst), getPointer<double>(src1), getPointer<double>(src2), elements);
				break;
		}

	}
	void cpu_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, float alpha1, const void *src1, float alpha2,
			const void *src2)
	{
		assert(dst != nullptr);
		assert(src1 != nullptr);
		assert(src2 != nullptr);

		const int elements = volume(shape);

		switch (dtype)
		{
			case DTYPE_FLOAT32:
				add_kernel(getPointer<float>(dst), alpha1, getPointer<float>(src1), alpha2, getPointer<float>(src2), elements);
				break;
			case DTYPE_FLOAT64:
				add_kernel(getPointer<double>(dst), (double) alpha1, getPointer<double>(src1), (double) alpha2, getPointer<double>(src2), elements);
				break;
		}
	}
	void cpu_sum_over_first_dim(mlContext_t context, mlShape_t shape, void *dst, const void *src, float beta)
	{
		assert(dst != nullptr);
		assert(src != nullptr);

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		assert(cpu::Context::getWorkspaceSize(context) >= last_dim * sizeof(float));

		float *tmp_ptr = cpu::Context::getWorkspace<float>(context);
		const float *src_ptr = getPointer<float>(src);

		for (int j = 0; j < last_dim; j++)
			tmp_ptr[j] = 0.0f;

		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
				tmp_ptr[j] += src_ptr[i * last_dim + j];

		float *dst_ptr = getPointer<float>(dst);
		if (beta == 0.0f)
			for (int j = 0; j < last_dim; j++)
				dst_ptr[j] = tmp_ptr[j];
		else
			for (int j = 0; j < last_dim; j++)
				dst_ptr[j] = dst_ptr[j] * beta + tmp_ptr[j];
	}
	float cpu_mean_squared_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		assert(output != nullptr);
		assert(target != nullptr);

		const int elements = volume(shape);
		const float inv_batch_size = 1.0f / get_first_dim(shape);

		const float *output_ptr = getPointer<float>(output);
		const float *target_ptr = getPointer<float>(target);

		float result = 0.0f;
		for (int i = 0; i < elements; i++)
			result += square(output_ptr[i] - target_ptr[i]);
		return 0.5f * result * inv_batch_size;
	}
	void cpu_mean_squared_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
	{
		cpu_cross_entropy_gradient(context, shape, gradient, output, target, weight); // in this case both gradients are the same
	}
	float cpu_cross_entropy_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		assert(output != nullptr);
		assert(target != nullptr);

		const int elements = volume(shape);

		const float *output_ptr = getPointer<float>(output);
		const float *target_ptr = getPointer<float>(target);

		const float inv_batch_size = 1.0f / get_first_dim(shape);

		float result = 0.0f;
		for (int i = 0; i < elements; i++)
			result += std::max(0.0f, cross_entropy(output_ptr[i], target_ptr[i]) - cross_entropy(target_ptr[i], target_ptr[i]));
		return result * inv_batch_size;
	}
	void cpu_cross_entropy_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
	{
		assert(output != nullptr);
		assert(target != nullptr);
		assert(gradient != nullptr);

		const int elements = volume(shape);

		float *gradient_ptr = getPointer<float>(gradient);
		const float *output_ptr = getPointer<float>(output);
		const float *target_ptr = getPointer<float>(target);

		const float inv_batch_size = weight / get_first_dim(shape);

		for (int i = 0; i < elements; i++)
			gradient_ptr[i] = inv_batch_size * (output_ptr[i] - target_ptr[i]);
	}
	float cpu_value_head_loss(mlContext_t context, mlShape_t shape, const void *output, const void *target)
	{
		assert(output != nullptr);
		assert(target != nullptr);

		const int first_dim = get_first_dim(shape);
		assert(get_last_dim(shape) == 2);

		const float *output_ptr = getPointer<float>(output);
		const float *target_ptr = getPointer<float>(target);

		float result = 0.0f;
		for (int i = 0; i < first_dim; i++)
		{
			const float mean = output_ptr[i * 2 + 0];
			const float variance = output_ptr[i * 2 + 1];
			const float Q = target_ptr[i];

			result += std::log(variance) + square(mean - Q) / variance;
		}
		return result / first_dim;
	}
	void cpu_value_head_gradient(mlContext_t context, mlShape_t shape, void *gradient, const void *output, const void *target, float weight)
	{
		assert(output != nullptr);
		assert(target != nullptr);
		assert(gradient != nullptr);

		const int first_dim = get_first_dim(shape);
		assert(get_last_dim(shape) == 2);

		float *gradient_ptr = getPointer<float>(gradient);
		const float *output_ptr = getPointer<float>(output);
		const float *target_ptr = getPointer<float>(target);

		const float inv_batch_size = weight / first_dim;

		for (int i = 0; i < first_dim; i++)
		{
			const float mean = output_ptr[i * 2 + 0];
			const float variance = output_ptr[i * 2 + 1];
			const float Q = target_ptr[i];

			gradient_ptr[i * 2 + 0] = inv_batch_size * 2.0f * (mean - Q) / variance;
			gradient_ptr[i * 2 + 1] = inv_batch_size * (variance - square(mean - Q)) / square(variance);
		}
	}

	void cpu_radam_optimize(mlContext_t context, mlShape_t shape, void *weight, const void *update, void *momentum, void *variance,
			float learning_rate, float beta1, float beta2, int step, float weight_decay)
	{
		assert(weight != nullptr);
		assert(update != nullptr);
		assert(momentum != nullptr);
		assert(variance != nullptr);
		assert(step > 0);

		const int elements = volume(shape);

		float *weight_ptr = getPointer<float>(weight);
		const float *update_ptr = getPointer<float>(update);
		float *momentum_ptr = getPointer<float>(momentum);
		float *variance_ptr = getPointer<float>(variance);

		const float pow_beta1 = bounded_pow(beta1, step, 1.0e-8f);
		const float pow_beta2 = bounded_pow(beta2, step, 1.0e-8f);
		const float p_inf = 2.0f / (1.0f - beta2) - 1.0f;
		const float p = p_inf - 2.0f * step * pow_beta2 / (1.0f - pow_beta2);

		float correction = 1.0f;

		for (int i = 0; i < elements; i++)
		{
			momentum_ptr[i] = beta1 * momentum_ptr[i] + (1.0f - beta1) * update_ptr[i];
			variance_ptr[i] = beta2 * variance_ptr[i] + (1.0f - beta2) * square(update_ptr[i]);

			if (p > 4.0f)
			{
				const float l = std::sqrt((1.0f - pow_beta2) / (variance_ptr[i] + 1.0e-8f));
				const float r = std::sqrt((p - 4.0f) * (p - 2.0f) * p_inf / ((p_inf - 4.0f) * (p_inf - 2.0f) * p));
				correction = l * r;
			}

			const float m_dash = momentum_ptr[i] / (1.0f - pow_beta1);
			weight_ptr[i] -= learning_rate * (m_dash * correction + weight_decay * weight_ptr[i]);
			weight_ptr[i] = round_small_to_zero(weight_ptr[i]);
		}
	}

	void cpu_l2_regularization(mlContext_t context, mlShape_t shape, void *gradient, const void *param, float coefficient, float offset)
	{
		assert(gradient != nullptr);
		assert(param != nullptr);

		const int elements = volume(shape);

		float *gradient_ptr = getPointer<float>(gradient);
		const float *param_ptr = getPointer<float>(param);
		for (int i = 0; i < elements; i++)
			gradient_ptr[i] += coefficient * (param_ptr[i] - offset);
	}
} /* namespace ml */

