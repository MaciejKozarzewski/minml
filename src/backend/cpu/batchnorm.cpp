/*
 * batchnorm.cpp
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
	static constexpr float epsilon = 1.0e-6f;

	template<typename T>
	void setzero(T *ptr, int elements) noexcept
	{
		for (int i = 0; i < elements; i++)
			ptr[i] = static_cast<T>(0);
	}
	template<typename T>
	T square(T x)
	{
		return x * x;
	}
	float get_mean(const float *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return ptr[idx];
	}
	float get_stddev(const float *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return std::sqrt(ptr[last_dim + idx] + epsilon);
	}
	float get_gamma(const float *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return ptr[2 * last_dim + idx];
	}
	float get_beta(const float *ptr, int idx, int last_dim)
	{
		assert(idx >= 0 && idx < last_dim);
		return ptr[3 * last_dim + idx];
	}

	/*
	 * Welford's online algorithm for calculating mean and variance
	 */
	template<typename T>
	class AvgVarStats
	{
			T samples = static_cast<T>(0);
			T M = static_cast<T>(0);
			T M2 = static_cast<T>(0);
		public:
			void add(T x) noexcept
			{
				samples += static_cast<T>(1);
				const T delta = x - M;
				M += delta / samples;
				M2 += delta * (x - M);
			}
			T get_average() const noexcept
			{
				return M;
			}
			T get_variance() const noexcept
			{
				assert(samples >= static_cast<T>(2));
				return M2 / (samples - static_cast<T>(1));
			}

			static AvgVarStats merge(const AvgVarStats<T> &lhs, const AvgVarStats<T> &rhs) noexcept
			{
				assert(lhs.samples >= static_cast<T>(0) && rhs.samples >= static_cast<T>(0));
				AvgVarStats result;
				result.samples = lhs.samples + rhs.samples;
				result.M = (lhs.samples * lhs.M + rhs.samples * rhs.M) / result.samples;
				result.M2 = lhs.M2 + rhs.M2 + square(lhs.M - rhs.M) * (lhs.samples * rhs.samples) / result.samples;
				return result;
			}
	};
}

namespace ml
{
	void cpu_batchnorm_inference(mlContext_t context, mlShape_t shape, const void *input, void *output, const void *weights, mlActivationType_t act)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(weights != nullptr);

		const float *input_ptr = getPointer<float>(input);
		float *output_ptr = getPointer<float>(output);
		const float *weights_ptr = getPointer<float>(weights);

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		assert(cpu::Context::getWorkspaceSize(context) >= 2 * last_dim * sizeof(float));

		float *scale_ptr = cpu::Context::getWorkspace<float>(context);
		float *shift_ptr = cpu::Context::getWorkspace<float>(context) + last_dim;

		/* weights rows are:
		 * mean
		 * variance
		 * gamma
		 * beta
		 */
		for (int j = 0; j < last_dim; j++)
		{
			scale_ptr[j] = get_gamma(weights_ptr, j, last_dim) / get_stddev(weights_ptr, j, last_dim); // gamma / sqrt(variance + epsilon)
			shift_ptr[j] = -get_mean(weights_ptr, j, last_dim) * scale_ptr[j] + get_beta(weights_ptr, j, last_dim); // -mean * scale + beta
		}

		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
			{
				float tmp = input_ptr[i * last_dim + j] * scale_ptr[j] + shift_ptr[j];
				if (act == ACTIVATION_RELU)
					tmp = std::max(0.0f, tmp);
				if (act == ACTIVATION_TANH)
					tmp = std::tanh(tmp);
				if (act == ACTIVATION_SIGMOID)
					tmp = 1.0f / (1.0f + std::exp(-tmp));
				output_ptr[i * last_dim + j] = tmp;
			}
	}
	void cpu_batchnorm_forward(mlContext_t context, mlShape_t shape, const void *input, void *output, void *weights, void *running_stats,
			int running_stat_idx, mlActivationType_t act)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(weights != nullptr);
		assert(running_stats != nullptr);
		assert(running_stat_idx >= 0);

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		const float *input_ptr = getPointer<float>(input);
		float *output_ptr = getPointer<float>(output);
		float *weights_ptr = getPointer<float>(weights);
		float *running_stat_ptr = getPointer<float>(running_stats) + 2 * running_stat_idx * last_dim;

		assert(cpu::Context::getWorkspaceSize(context) >= last_dim * sizeof(AvgVarStats<float> ));
		AvgVarStats<float> *stats = cpu::Context::getWorkspace<AvgVarStats<float>>(context);

		for (int j = 0; j < last_dim; j++)
			stats[j] = AvgVarStats<float>();
		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
				stats[j].add(input_ptr[i * last_dim + j]);

		for (int j = 0; j < last_dim; j++)
		{
			running_stat_ptr[j] = stats[j].get_average();
			running_stat_ptr[last_dim + j] = stats[j].get_variance();
		}

		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
			{
				const float gamma = get_gamma(weights_ptr, j, last_dim);
				const float beta = get_beta(weights_ptr, j, last_dim);
				float tmp = gamma * (input_ptr[i * last_dim + j] - get_mean(running_stat_ptr, j, last_dim))
						/ get_stddev(running_stat_ptr, j, last_dim) + beta;
				if (act == ACTIVATION_RELU)
					tmp = std::max(0.0f, tmp);
				if (act == ACTIVATION_TANH)
					tmp = std::tanh(tmp);
				if (act == ACTIVATION_SIGMOID)
					tmp = 1.0f / (1.0f + std::exp(-tmp));
				output_ptr[i * last_dim + j] = tmp;
			}
	}
	void cpu_batchnorm_backward(mlContext_t context, mlShape_t shape, const void *input, const void *output, void *gradient_prev, void *gradient_next,
			const void *weights, void *weights_update, const void *running_stats, int running_stat_idx, mlActivationType_t act)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(gradient_prev != nullptr);
		assert(gradient_next != nullptr);
		assert(weights_update != nullptr);
		assert(running_stats != nullptr);

		const int first_dim = volume_without_last_dim(shape);
		const int last_dim = get_last_dim(shape);

		const float *input_ptr = getPointer<float>(input);
		const float *output_ptr = getPointer<float>(output);
		float *gradient_prev_ptr = getPointer<float>(gradient_prev);
		float *gradient_next_ptr = getPointer<float>(gradient_next);
		float *weights_update_ptr = getPointer<float>(weights_update);
		const float *running_stat_ptr = getPointer<float>(running_stats) + 2 * running_stat_idx * last_dim;

		assert(cpu::Context::getWorkspaceSize(context) >= 2 * last_dim * sizeof(float));

		/* weights rows are:
		 * mean
		 * variance
		 * gamma
		 * beta
		 */
		float *d_sigma = cpu::Context::getWorkspace<float>(context);
		float *d_mu = cpu::Context::getWorkspace<float>(context) + last_dim;
		setzero(d_sigma, last_dim);
		setzero(d_mu, last_dim);

		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
			{
				const int idx = i * last_dim + j;
				if (act == ACTIVATION_RELU and output_ptr[idx] <= 0.0f)
					gradient_next_ptr[idx] *= 0.01f;
				if (act == ACTIVATION_TANH)
					gradient_next_ptr[idx] *= (1.0f - square(output_ptr[idx]));
				if (act == ACTIVATION_SIGMOID)
					gradient_next_ptr[idx] *= output_ptr[idx] * (1.0f - output_ptr[idx]);

				const float avg = get_mean(running_stat_ptr, j, last_dim);
				const float stddev = get_stddev(running_stat_ptr, j, last_dim);
				d_sigma[j] += gradient_next_ptr[idx] * (input_ptr[idx] - avg) / stddev;
				d_mu[j] += gradient_next_ptr[idx];
			}
		for (int j = 0; j < last_dim; j++)
		{
			weights_update_ptr[2 * last_dim + j] += d_sigma[j];
			weights_update_ptr[3 * last_dim + j] += d_mu[j];
		}

		for (int j = 0; j < last_dim; j++)
		{
			const float gamma = get_gamma(getPointer<float>(weights), j, last_dim);
			const float stddev = get_stddev(running_stat_ptr, j, last_dim);
			d_sigma[j] = -gamma / stddev * d_sigma[j] / first_dim;
			d_mu[j] = -gamma / stddev * d_mu[j] / first_dim;
		}
		for (int i = 0; i < first_dim; i++)
			for (int j = 0; j < last_dim; j++)
			{
				const int idx = i * last_dim + j;
				const float gamma = get_gamma(getPointer<float>(weights), j, last_dim);
				const float avg = get_mean(running_stat_ptr, j, last_dim);
				const float stddev = get_stddev(running_stat_ptr, j, last_dim);
				gradient_prev_ptr[idx] = gamma / stddev * gradient_next_ptr[idx] + d_sigma[j] * (input_ptr[idx] - avg) / stddev + d_mu[j];
			}
	}
	void cpu_batchnorm_update(mlContext_t context, mlShape_t shape, const void *running_stat, void *weights, bool use_gamma, bool use_beta)
	{
		assert(running_stat != nullptr);
		assert(weights != nullptr);
		assert(shape.rank == 2);

		const float *running_stat_ptr = getPointer<float>(running_stat);
		float *weights_ptr = getPointer<float>(weights);

		const int first_dim = get_first_dim(shape);
		const int last_dim = get_last_dim(shape) / 2;

		assert(cpu::Context::getWorkspaceSize(context) >= 2 * last_dim * sizeof(float));
		float *mean_ptr = cpu::Context::getWorkspace<float>(context);
		float *variance_ptr = cpu::Context::getWorkspace<float>(context) + last_dim;

		/* weights rows are:
		 * mean
		 * variance
		 * gamma
		 * beta
		 */
		setzero(mean_ptr, last_dim);
		setzero(variance_ptr, last_dim);
		for (int i = 0; i < first_dim; i++)
		{
			for (int j = 0; j < last_dim; j++)
				mean_ptr[j] += running_stat_ptr[i * 2 * last_dim + j];
			for (int j = 0; j < last_dim; j++)
				variance_ptr[j] += running_stat_ptr[(i * 2 + 1) * last_dim + j];
		}

		for (int j = 0; j < last_dim; j++)
			weights_ptr[0 * last_dim + j] = mean_ptr[j] / first_dim;
		for (int j = 0; j < last_dim; j++)
			weights_ptr[1 * last_dim + j] = variance_ptr[j] / first_dim;
		if (not use_gamma)
			for (int j = 0; j < last_dim; j++)
				weights_ptr[2 * last_dim + j] = 1.0f;
		if (not use_beta)
			for (int j = 0; j < last_dim; j++)
				weights_ptr[3 * last_dim + j] = 0.0f;
	}
	void cpu_fold_batchnorm(mlContext_t context, mlShape_t shape, void *layer_weights, void *layer_bias, const void *batchnorm_weights)
	{
		assert(layer_weights != nullptr);
		assert(layer_bias != nullptr);
		assert(batchnorm_weights != nullptr);

		const int channels = get_first_dim(shape);
		const int last_dim = volume_without_first_dim(shape);

		/* weights rows are:
		 * mean
		 * variance
		 * gamma
		 * beta
		 */
		const float *bn_ptr = getPointer<float>(batchnorm_weights);

		for (int i = 0; i < channels; i++)
		{
			const float scale = get_gamma(bn_ptr, i, channels) / get_stddev(bn_ptr, i, channels); // gamma / sqrt(variance + epsilon)
			const float shift = -get_mean(bn_ptr, i, channels) * scale + get_beta(bn_ptr, i, channels); // -mean * scale + beta

			getPointer<float>(layer_bias)[i] = getPointer<float>(layer_bias)[i] * scale + shift;
			for (int j = 0; j < last_dim; j++)
				getPointer<float>(layer_weights)[i * last_dim + j] *= scale;
		}
	}

} /* namespace ml */

