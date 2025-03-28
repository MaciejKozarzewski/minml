/*
 * SpaceToDepth.cpp
 *
 *  Created on: Aug 15, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/SpaceToDepth.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	SpaceToDepth::SpaceToDepth(int patch_size_h, int patch_size_w) :
			Layer(),
			m_patch_size_h(patch_size_h),
			m_patch_size_w(patch_size_w)
	{
	}
	SpaceToDepth::SpaceToDepth(int patch_size) :
			Layer(),
			m_patch_size_h(patch_size),
			m_patch_size_w(patch_size)
	{
	}
	Shape SpaceToDepth::getOutputShape() const
	{
		const int batch_size = getInputShape().dim(0);
		const int height = (getInputShape().dim(1) + m_patch_size_h - 1) / m_patch_size_h;
		const int width = (getInputShape().dim(2) + m_patch_size_w - 1) / m_patch_size_w;
		const int channels = getInputShape().dim(3) * m_patch_size_h * m_patch_size_w;
		return Shape( { batch_size, height, width, channels });
	}
	std::string SpaceToDepth::name() const
	{
		return "SpaceToDepth";
	}
	Json SpaceToDepth::getConfig() const
	{
		Json result = Layer::getConfig();
		result["patch_size_h"] = m_patch_size_h;
		result["patch_size_w"] = m_patch_size_w;
		return result;
	}
	std::unique_ptr<Layer> SpaceToDepth::clone(const Json &config) const
	{
		std::unique_ptr<SpaceToDepth> result = std::make_unique<SpaceToDepth>(config["patch_size_h"].getInt(), config["patch_size_w"].getInt());
		result->loadConfig(config);
		return result;
	}
	void SpaceToDepth::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		spaceToDepth(context(), input[0], output, 0.0f);
	}
	void SpaceToDepth::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
			const std::vector<float> &beta)
	{
		depthToSpace(context(), gradient_next, gradient_prev[0], beta[0]);
	}

} /* namespace ml */

