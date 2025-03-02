/*
 * CalibrationTable.hpp
 *
 *  Created on: Feb 3, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_GRAPH_CALIBRATIONTABLE_HPP_
#define MINML_GRAPH_CALIBRATIONTABLE_HPP_

#include <minml/graph/Histogram.hpp>

namespace ml
{
	class CalibrationTable
	{
		private:
			std::vector<Histogram> m_histograms;
			int m_number_of_bins;
			float m_accuracy;
		public:
			explicit CalibrationTable(int numberOfBins = 2048, float accuracy = 1.0e-3f) noexcept;

			void init(int size);
			Histogram& getHistogram(size_t index);
			const Histogram& getHistogram(size_t index) const;
			int size() const noexcept;

			bool isReady() const noexcept;
			double getCompletionFactor() const noexcept;
			Json save(SerializedObject &binary_data) const;
			void load(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace ml */

#endif /* MINML_GRAPH_CALIBRATIONTABLE_HPP_ */
