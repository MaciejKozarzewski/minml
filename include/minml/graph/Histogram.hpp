/*
 * Histogram.hpp
 *
 *  Created on: Feb 3, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_GRAPH_HISTOGRAM_HPP_
#define MINML_GRAPH_HISTOGRAM_HPP_

#include <vector>
#include <limits>
#include <cinttypes>
#include <string>

class Json;
class SerializedObject;
namespace ml
{
	class Tensor;
	enum class ActivationType
	;
}

namespace ml
{
	class Histogram
	{
		private:
			std::vector<double> m_data;
			float m_min_value = std::numeric_limits<float>::max();
			float m_max_value = std::numeric_limits<float>::lowest();
			int64_t m_collected_samples = 0;
			int64_t m_outliers_count = 0;

			float m_accuracy;
			int m_target_bins;
			bool m_is_used = true;
			bool m_is_exact = false;
			bool m_is_binary = false;
			bool m_is_ready = false;

			std::pair<int, int> m_optimal_range;

			float int8_to_fp32_scale = 1.0f;
			float int8_to_fp32_shift = 0.0f;
		public:
			explicit Histogram(int numberOfBins, float accuracy, int targetBins = 256);
			void markAsUnused();
			/*
			 * \brief Can be used to indicate what kind of activation function is used after a layer.
			 */
			void setHint(ActivationType act);
			void setBinary();
			void collectStatistics(const Tensor &tensor);
			bool isReady() const noexcept;
			std::string getInfo() const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
			float getScale() const noexcept;
			float getShift() const noexcept;
		private:
			void find_min_max(const std::vector<float> &tensor);
			void update_histogram(const std::vector<float> &tensor);
			std::pair<int, int> find_coarse_range();
			std::pair<int, int> fine_tune_range(const std::vector<std::pair<int, int>> &candidates, const std::vector<float> &data);
			bool has_enough_samples_for_min_max() const noexcept;
			std::pair<float, float> get_scale_and_shift(std::pair<int, int> range);
	};

} /* namespace ml */

#endif /* MINML_GRAPH_HISTOGRAM_HPP_ */
