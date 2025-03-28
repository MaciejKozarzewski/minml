/*
 * GradientScaler.cpp
 *
 *  Created on: Mar 23, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/training/GradientScaler.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>
#include <minml/layers/Parameter.hpp>

#include <algorithm>

namespace ml
{

	GradientScaler::GradientScaler(float initialScale, int scaleChangeInterval) :
			m_scale(initialScale),
			m_scale_change_interval(scaleChangeInterval)
	{
	}
	float GradientScaler::getScale() const noexcept
	{
		return m_scale;
	}
	float GradientScaler::getInvScale(const Context &context, std::vector<Tensor> &params)
	{
		const std::vector<int> flags = isNanOrInf(context, params);
		const bool are_all_gradients_ok = std::all_of(flags.begin(), flags.end(), [](int i)
		{
			return i == 0;
		});

		if (are_all_gradients_ok)
		{
			m_steps_since_scale_change++;
			const float old_scale = m_scale;
			if (m_steps_since_scale_change >= m_scale_change_interval)
			{
				m_steps_since_scale_change = 0;
				m_scale *= 2.0f;
			}
			return 1.0f / old_scale;
		}
		else
		{
			m_steps_since_scale_change = 0;
			m_scale *= 0.5f;
			return 0.0f;
		}
	}

} /* namespace ml */

