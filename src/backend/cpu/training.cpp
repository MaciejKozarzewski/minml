/*
 * training.cpp
 *
 *  Created on: Jan 4, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"

#include <cmath>
#include <cassert>
#include <iostream>

namespace
{
	float round_small_to_zero(float x) noexcept
	{
		return (fabsf(x) < 1.0e-7f) ? 0.0f : x;
	}
	float safe_log(float x) noexcept
	{
		return std::log(1.0e-8f + x);
	}
	float cross_entropy(float output, float target) noexcept
	{
		return -target * safe_log(output) - (1.0f - target) * safe_log(1.0f - output);
	}
	float square(float x) noexcept
	{
		return x * x;
	}
	float bounded_pow(float x, float y, float min) noexcept
	{
		assert(0 < x && x < 1);
		assert(min > 0);
		const float max_y = std::log(min) / std::log(x);
		return (y >= max_y) ? 0.0f : std::pow(x, y);
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
	void cpu_add_tensors(mlContext_t context, mlDataType_t dtype, mlShape_t shape, void *dst, const void *src1, const void *src2)
	{
		assert(dst != nullptr);
		assert(src1 != nullptr);
		assert(src2 != nullptr);

		const int elements = volume(shape);

		float *dst_ptr = getPointer<float>(dst);
		const float *src1_ptr = getPointer<float>(src1);
		const float *src2_ptr = getPointer<float>(src2);

		if (dst == src1)
		{ // in place addition
			for (int i = 0; i < elements; i++)
				dst_ptr[i] += src2_ptr[i];
		}
		else
		{
			for (int i = 0; i < elements; i++)
				dst_ptr[i] = src1_ptr[i] + src2_ptr[i];
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

		const float *output_ptr = getPointer<float>(output);
		const float *target_ptr = getPointer<float>(target);

		const float inv_batch_size = 1.0f / get_first_dim(shape);

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
	void cpu_radam_optimize(mlContext_t context, mlShape_t shape, void *weight, const void *update, void *momentum, void *variance,
			float learning_rate, float beta1, float beta2, int step)
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
			weight_ptr[i] -= learning_rate * m_dash * correction;
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

