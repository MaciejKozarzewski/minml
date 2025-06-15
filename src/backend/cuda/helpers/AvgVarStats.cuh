/*
 * AvgVarStats.cuh
 *
 *  Created on: Mar 28, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_CUDA_KERNELS_NORMALIZATION_AVGVARSTATS_CUH_
#define BACKEND_CUDA_KERNELS_NORMALIZATION_AVGVARSTATS_CUH_

#include <cuda_runtime_api.h>

#include "misc.cuh"

/*
 * Welford's online algorithm for calculating mean and variance
 */
template<typename T>
struct AvgVarStats
{
		T samples = T{};
		T M = T{}; // mean
		T M2 = T{}; // variance

		AvgVarStats() = default;
		__device__ AvgVarStats(T n, T mean, T var) :
				samples(n),
				M(mean),
				M2(var)
		{
		}
		template<typename U>
		__device__ void add(U x) noexcept
		{
			samples += static_cast<T>(1.0f);
			const T delta = static_cast<T>(x) - M;
			M += delta / samples;
			M2 += delta * (static_cast<T>(x) - M);
		}
		__device__ T get_average() const noexcept
		{
			return M;
		}
		__device__ T get_variance() const noexcept
		{
			assert(samples >= static_cast<T>(2.0f));
			return M2 / (samples - static_cast<T>(1.0f));
		}
		__device__ T get_stddev() const noexcept
		{
			return sqrt(static_cast<T>(1.0e-6f) + get_variance());
		}
		__device__ void merge_with(const AvgVarStats<T> &other) noexcept
		{
			if (other.samples == static_cast<T>(0.0f))
				return;
			if (this->samples == static_cast<T>(0.0f))
			{
				this->samples = other.samples;
				this->M = other.M;
				this->M2 = other.M2;
			}
			else
			{
				const T total_samples = this->samples + other.samples;
				const T total_M = (this->samples * this->M + other.samples * other.M) / total_samples;
				const T total_M2 = this->M2 + other.M2 + square(this->M - other.M) * (this->samples * other.samples) / total_samples;
				this->samples = total_samples;
				this->M = total_M;
				this->M2 = total_M2;
			}
		}
};

#endif /* BACKEND_CUDA_KERNELS_NORMALIZATION_AVGVARSTATS_CUH_ */
