/*
 * DepthToSpace.cpp
 *
 *  Created on: Aug 15, 2024
 *      Author: Maciej Kozarzewski
 */

#include <minml/layers/DepthToSpace.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/json.hpp>

namespace ml
{
	DepthToSpace::DepthToSpace(int patch_size_h, int patch_size_w, const Shape &output_shape) :
			Layer(),
			m_patch_size_h(patch_size_h),
			m_patch_size_w(patch_size_w),
			m_output_shape(output_shape)
	{
	}
	DepthToSpace::DepthToSpace(int patch_size, const Shape &output_shape) :
			Layer(),
			m_patch_size_h(patch_size),
			m_patch_size_w(patch_size),
			m_output_shape(output_shape)
	{
	}
	Shape DepthToSpace::getOutputShape() const
	{
		const int batch_size = getInputShape().dim(0);
		const int height = m_output_shape.dim(0);
		const int width = m_output_shape.dim(1);
		const int channels = getInputShape().lastDim() / (m_patch_size_h * m_patch_size_w);
		return Shape( { batch_size, height, width, channels });
	}
	std::string DepthToSpace::name() const
	{
		return "DepthToSpace";
	}
	Json DepthToSpace::getConfig() const
	{
		Json result = Layer::getConfig();
		result["patch_size_h"] = m_patch_size_h;
		result["patch_size_w"] = m_patch_size_w;
		result["output_shape"] = m_output_shape.serialize();
		return result;
	}
	std::unique_ptr<Layer> DepthToSpace::clone(const Json &config) const
	{
		std::unique_ptr<DepthToSpace> result = std::make_unique<DepthToSpace>(config["patch_size_h"].getInt(), config["patch_size_w"].getInt(),
				Shape(config["output_shape"]));
		result->m_dtype = typeFromString(config["dtype"].getString());
		return result;
	}
	void DepthToSpace::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		depthToSpace(context(), input[0], output);
	}
	void DepthToSpace::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next)
	{
		spaceToDepth(context(), gradient_next, gradient_prev[0]);
	}

} /* namespace ml */

