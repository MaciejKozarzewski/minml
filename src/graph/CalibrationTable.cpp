/*
 * CalibrationTable.cpp
 *
 *  Created on: Feb 4, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/graph/CalibrationTable.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>
#include <minml/utils/random.hpp>

#include <numeric>
#include <algorithm>
#include <omp.h>

namespace ml
{
	CalibrationTable::CalibrationTable(int numberOfBins, float accuracy, int targetBins) noexcept :
			m_number_of_bins(numberOfBins),
			m_accuracy(accuracy),
			m_target_bins(targetBins)
	{
	}
	void CalibrationTable::init(int size)
	{
		m_histograms.clear();
		for (int i = 0; i < size; i++)
			m_histograms.emplace_back(m_number_of_bins, m_accuracy, m_target_bins);
	}
	Histogram& CalibrationTable::getHistogram(size_t index)
	{
		return m_histograms.at(index);
	}
	const Histogram& CalibrationTable::getHistogram(size_t index) const
	{
		return m_histograms.at(index);
	}
	int CalibrationTable::size() const noexcept
	{
		return m_histograms.size();
	}
	bool CalibrationTable::isReady() const noexcept
	{
		if (m_histograms.empty())
			return false;
		else
			return std::all_of(m_histograms.begin(), m_histograms.end(), [](const auto &h)
			{	return h.isReady();});
	}
	double CalibrationTable::getCompletionFactor() const noexcept
	{
		return std::count_if(m_histograms.begin(), m_histograms.end(), [](const auto &h)
		{	return h.isReady();}) / static_cast<double>(m_histograms.size());
	}
	Json CalibrationTable::save(SerializedObject &binary_data) const
	{
		Json result(JsonType::Array);
		for (size_t i = 0; i < m_histograms.size(); i++)
			result[i] = m_histograms[i].serialize(binary_data);
		return result;
	}
	void CalibrationTable::load(const Json &json, const SerializedObject &binary_data)
	{
		for (int i = 0; i < json.size(); i++)
		{
			m_histograms.emplace_back(m_number_of_bins, m_accuracy);
			m_histograms[i].unserialize(json[i], binary_data);
		}
	}

} /* namespace ml */

