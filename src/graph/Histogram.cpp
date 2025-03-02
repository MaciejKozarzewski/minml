/*
 * Histogram.cpp
 *
 *  Created on: Feb 3, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/graph/Histogram.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>
#include <minml/utils/random.hpp>

#include <numeric>
#include <algorithm>
#include <cmath>

namespace
{
	template<typename T>
	T safe_log(T x) noexcept
	{
		return std::log(std::numeric_limits<T>::epsilon() + x);
	}
	template<typename T>
	T square(T x) noexcept
	{
		return x * x;
	}

	double histogram_l1_diff(const std::vector<double> &P, const std::vector<double> &Q)
	{
		assert(P.size() == Q.size());

		const double area_lhs = 1.0f / std::accumulate(P.begin(), P.end(), 1e-16);
		const double area_rhs = 1.0f / std::accumulate(Q.begin(), Q.end(), 1e-16);

		double diff = 0.0;
		for (size_t i = 0; i < P.size(); i++)
			diff += fabs(P[i] * area_lhs - Q[i] * area_rhs);
		return diff;
	}
	double histogram_kl_divergence(const std::vector<double> &P, const std::vector<double> &Q)
	{
		assert(P.size() == Q.size());
		double result = 0.0;
		for (size_t i = 0; i < P.size(); i++)
			result += P[i] * (safe_log(P[i]) - safe_log(Q[i]));
		return result;
	}

	void resize_histogram(std::vector<double> &result, int from_bin, int to_bin, int target_bins)
	{
		assert(from_bin < to_bin);
		const int span = to_bin - from_bin;
		assert(span % target_bins == 0);
		const int window = span / target_bins; // number of bins merged together

		double outliers_low = 0.0;
		for (int i = 0; i < from_bin; i++)
		{
			outliers_low += result[i];
			result[i] = 0.0;
		}
		result[from_bin] += outliers_low;

		double outliers_high = 0.0;
		for (size_t i = to_bin; i < result.size(); i++)
		{
			outliers_high += result[i];
			result[i] = 0.0;
		}
		result[to_bin - 1] += outliers_high;

		double total_sum = 0.0;
		for (int i = from_bin; i < to_bin; i += window)
		{
			double partial_sum = 0.0;
			int non_zero_count = 0.0;
			for (int j = 0; j < window; j++)
			{
				partial_sum += result[i + j];
				non_zero_count += static_cast<int>(result[i + j] != 0.0);
			}
			total_sum += partial_sum;

			const double avg = partial_sum / non_zero_count;
			for (int j = 0; j < window; j++)
				result[i + j] = (result[i + j] != 0.0) ? avg : 0.0;
		}
		// normalize
		const double scale = 1.0 / total_sum;
		for (int i = from_bin; i < to_bin; i++)
			result[i] *= scale;
	}

	template<typename T>
	void print_dist(const std::vector<T> &dist)
	{
		for (size_t i = 0; i < dist.size(); i++)
			std::cout << dist[i] << ' ';
		std::cout << '\n';
	}
	std::vector<double> normalize(const std::vector<double> &v)
	{
		const double scale = 1.0 / std::accumulate(v.begin(), v.end(), 1e-16);
		std::vector<double> result(v.size(), 0.0);
		for (size_t i = 0; i < result.size(); i++)
			result[i] = v[i] * scale;
		return result;
	}

	std::vector<float> copy_to_host(const ml::Tensor &tensor)
	{
		std::vector<float> result(tensor.volume());
		tensor.copyToHost(result.data(), tensor.sizeInBytes());
		return result;
	}
	double get_quantization_l2_error(float x, ml::AffineTransform scale_and_shift) noexcept
	{
		const int8_t quantized = std::max(-128.0f, std::min(127.0f, x * scale_and_shift.scale() + scale_and_shift.shift()));
		const float recovered = (static_cast<float>(quantized) - scale_and_shift.shift()) / scale_and_shift.scale();
		return square(x - recovered);
	}
}

namespace ml
{
	Histogram::Histogram(int numberOfBins, float accuracy, int targetBins) :
			m_data(numberOfBins, 0),
			m_accuracy(accuracy),
			m_target_bins(targetBins)
	{
	}
	void Histogram::markAsUnused()
	{
		m_is_used = false;
	}
	void Histogram::setHint(ActivationType act)
	{
		switch (act)
		{
			case ActivationType::LINEAR:
				break;
			case ActivationType::SIGMOID:
				m_min_value = 0.0;
				m_max_value = 1.0;
				m_is_ready = true;
				m_is_exact = true;
				break;
			case ActivationType::TANH:
				m_min_value = -1.0;
				m_max_value = 1.0;
				m_is_ready = true;
				m_is_exact = true;
				break;
			case ActivationType::RELU:
				m_min_value = 0.0;
				break;
			case ActivationType::GELU:
				m_min_value = -0.2;
				break;
			case ActivationType::EXP:
				m_min_value = 0.0;
				break;
			case ActivationType::SOFTMAX:
				m_min_value = 0.0;
				m_max_value = 1.0;
				m_is_ready = true;
				m_is_exact = true;
				break;
		}
	}
	void Histogram::setBinary()
	{
		m_min_value = 0.0;
		m_max_value = 1.0;
		m_is_ready = true;
		m_is_exact = true;
		m_is_binary = true;
	}
	void Histogram::collectStatistics(const Tensor &tensor)
	{
		if (not m_is_used)
			return;

		if (not m_is_exact)
		{
			const std::vector<float> runtime_data = copy_to_host(tensor);

			if (not has_enough_samples_for_min_max())
				find_min_max(runtime_data);
			m_collected_samples += runtime_data.size();

			// intentionally not in else block, so after min/max condition is met, we reuse the same tensor for histogram collection
			if (has_enough_samples_for_min_max())
			{
				const std::vector<double> previous_histogram = m_data;
				update_histogram(runtime_data);
				const double diff = histogram_l1_diff(previous_histogram, m_data) / pow(tensor.firstDim(), 0.666f);

//				std::cout << "difference = " << diff << " (" << m_accuracy << ")\n";
				if (diff < m_accuracy)
					m_is_ready = true;
			}

			// now find optimal range for quantization
			if (m_is_ready)
			{
//				std::cout << getInfo() << '\n';
				m_optimal_range = find_coarse_range();
//				std::cout << "found coarse range of [" << m_optimal_range.first << ", " << m_optimal_range.second << "]\n";

//				for (int step = m_target_bins / 4; step >= 16; step /= 4)
//				const int step = m_target_bins / 4;
//				{
//					std::vector<std::pair<int, int>> candidate_ranges;
//
//					const int i0 = std::max(0, m_optimal_range.first - 2 * step);
//					const int i1 = std::min((int) m_data.size(), m_optimal_range.first + 2 * step);
//
//					const int j0 = std::max(0, m_optimal_range.second - 2 * step);
//					const int j1 = std::min((int) m_data.size(), m_optimal_range.second + 2 * step);
//
//					for (int i = i0; i <= i1; i += step)
//						for (int j = j0; j <= j1; j += step)
//							candidate_ranges.emplace_back(i, j);
//
//					m_optimal_range = fine_tune_range(candidate_ranges, runtime_data);
//					std::cout << "tuned range to [" << m_optimal_range.first << ", " << m_optimal_range.second << "]\n";
//				}
				const AffineTransform scale_and_shift = calculate_transform(m_optimal_range);
				int8_to_fp32_scale = scale_and_shift.scale();
				int8_to_fp32_shift = scale_and_shift.shift();

//				std::cout << getInfo() << '\n';
//				exit(0);
			}
		}
	}
	bool Histogram::isReady() const noexcept
	{
		return m_is_ready or not m_is_used;
	}
	std::string Histogram::getInfo() const
	{
		std::string result;
		result += "samples = " + std::to_string(m_collected_samples) + ", is ready " + std::to_string(m_is_ready) + '\n';
		result += "min = " + std::to_string(m_min_value) + ", max = " + std::to_string(m_max_value) + '\n';
		result += "scale = " + std::to_string(int8_to_fp32_scale) + ", shift = " + std::to_string(int8_to_fp32_shift) + '\n';
//		for (size_t i = 0; i < m_data.size(); i++)
//			result += "from " + std::to_string(m_min_value + (m_max_value - m_min_value) * i / m_data.size()) + " to "
//					+ std::to_string(m_min_value + (m_max_value - m_min_value) * (i + 1) / m_data.size()) + " = " + std::to_string(m_data[i]) + '\n';
		return result;
	}
	Json Histogram::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["min"] = m_min_value;
		result["max"] = m_max_value;
		result["collected_samples"] = m_collected_samples;
		result["outliers_count"] = m_outliers_count;
		result["is_used"] = m_is_used;
		result["is_exact"] = m_is_exact;
		result["is_binary"] = m_is_binary;
		result["is_ready"] = m_is_ready;
		result["accuracy"] = m_accuracy;
		result["target_bins"] = m_target_bins;
		result["scale"] = int8_to_fp32_scale;
		result["shift"] = int8_to_fp32_shift;
		result["binary_offset"] = binary_data.size();
		binary_data.save(m_data.data(), sizeof(double) * m_data.size());
		return result;
	}
	void Histogram::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_min_value = json["min"].getDouble();
		m_max_value = json["max"].getDouble();
		m_collected_samples = json["collected_samples"].getLong();
		m_outliers_count = json["outliers_count"].getLong();
		m_is_used = json["is_used"].getBool();
		m_is_exact = json["is_exact"].getBool();
		m_is_binary = json["is_binary"].getBool();
		m_is_ready = json["is_ready"].getBool();
		m_accuracy = json["accuracy"].getDouble();
		m_target_bins = json["target_bins"].getInt();
		int8_to_fp32_scale = json["scale"].getDouble();
		int8_to_fp32_shift = json["shift"].getDouble();
		const size_t offset = json["binary_offset"].getLong();
		binary_data.load(m_data.data(), offset, sizeof(double) * m_data.size());
	}
	AffineTransform Histogram::getTransform() const noexcept
	{
		return AffineTransform(int8_to_fp32_scale, int8_to_fp32_shift);
	}
	void Histogram::find_min_max(const std::vector<float> &data)
	{
		m_outliers_count = 0;
		float min_value = m_min_value;
		float max_value = m_max_value;
		for (size_t i = 0; i < data.size(); i++)
		{
			const float tmp = data[i];
			min_value = std::min(min_value, tmp);
			max_value = std::max(max_value, tmp);
			if (tmp < m_min_value or tmp > m_max_value)
				m_outliers_count++;
		}

		m_min_value = min_value;
		m_max_value = max_value;
	}
	void Histogram::update_histogram(const std::vector<float> &data)
	{
		const float tmp = (m_data.size() - 1) / (m_max_value - m_min_value + 1e-16f);
		for (size_t i = 0; i < data.size(); i++)
		{
			const size_t bin_index = (data[i] - m_min_value) * tmp;
			m_data[std::max((size_t) 0, std::min(m_data.size() - 1, bin_index))]++;
		}
	}
	std::pair<int, int> Histogram::find_coarse_range()
	{
		assert(m_data.size() % m_target_bins == 0);
		if (m_is_exact)
			return std::pair<int, int>(0, m_data.size());

		const std::vector<double> reference = normalize(m_data);
		std::vector<double> candidate(m_data.size());

		std::pair<int, int> result(0, m_data.size());

		double best_value = std::numeric_limits<double>::max();
		for (size_t i = 0; i < m_data.size(); i += m_target_bins)
			for (size_t j = i + m_target_bins; j <= m_data.size(); j += m_target_bins)
			{
				candidate.assign(reference.begin(), reference.end());
				resize_histogram(candidate, i, j, m_target_bins);
				const double kld = histogram_kl_divergence(reference, candidate);

//				std::cout << "[" << i << ", " << j << "] = " << kld << '\n';
				if (kld <= best_value)
				{
					result = std::pair<int, int>(i, j);
					best_value = kld;
//					std::cout << "-->new best range\n";
				}
			}
		return result;
	}
	std::pair<int, int> Histogram::fine_tune_range(const std::vector<std::pair<int, int>> &candidates, const std::vector<float> &data)
	{
		std::vector<AffineTransform> scales_and_shifts;
		for (size_t i = 0; i < candidates.size(); i++)
			scales_and_shifts.push_back(calculate_transform(candidates[i]));

		std::cout << "got following candidates\n";
		for (size_t i = 0; i < candidates.size(); i++)
			std::cout << "[" << candidates[i].first << ", " << candidates[i].second << "]\n";

		std::vector<double> l2_diffs(candidates.size(), 0.0);
		for (size_t i = 0; i < data.size(); i++)
			for (size_t j = 0; j < l2_diffs.size(); j++)
				l2_diffs[j] += get_quantization_l2_error(data[i], scales_and_shifts[j]);

		for (size_t j = 0; j < l2_diffs.size(); j++)
			l2_diffs[j] /= static_cast<double>(data.size());

		std::pair<int, int> result;
		double min_l2_error = std::numeric_limits<double>::max();
		for (size_t j = 0; j < l2_diffs.size(); j++)
		{
			std::cout << "L2 error = " << l2_diffs[j] << " for range [" << candidates[j].first << ", " << candidates[j].second << "]\n";
			if (l2_diffs[j] < min_l2_error)
			{
				min_l2_error = l2_diffs[j];
				result = candidates[j];
				std::cout << "-->new best range\n";
			}
		}
		return result;
	}
	bool Histogram::has_enough_samples_for_min_max() const noexcept
	{
		return (static_cast<double>(m_outliers_count) / m_collected_samples) < m_accuracy;
	}
	AffineTransform Histogram::calculate_transform(std::pair<int, int> range)
	{
		const float step = static_cast<float>(m_max_value - m_min_value) / m_data.size();
		const float start_value = m_min_value + range.first * step;
		const float end_value = m_min_value + range.second * step;

		const float scale = (end_value - start_value) / static_cast<float>(m_target_bins - 1);
		const float shift = 0.5f * (end_value + start_value + scale);

		// we must fine tune the shift to ensure that zero in fp32 is an integer value after quantization
		const int int8_zero = round(-shift / scale); // round to the nearest integer

		return AffineTransform(scale, -int8_zero * scale);
	}

} /* namespace libml */

