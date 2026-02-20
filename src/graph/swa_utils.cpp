/*
 * swa_utils.cpp
 *
 *  Created on: Feb 8, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/graph/swa_utils.hpp>
#include <minml/graph/Graph.hpp>
#include <minml/layers/BatchNormalization.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/math.hpp>

namespace ml
{
	void averageModelWeights(float alpha, const Graph &model, float beta, Graph &averaged)
	{
		for (int i = 0; i < model.numberOfNodes(); i++)
		{
			const Layer &src = model.getNode(i).getLayer();
			Layer &dst = averaged.getNode(i).getLayer();
			if (src.name() != dst.name())
				throw LogicError(__FUNCTION__, "graphs have different layers");
			if (src.getWeightShape() != dst.getWeightShape())
				throw ShapeMismatch(__FUNCTION__, dst.getWeightShape(), src.getWeightShape());
			if (src.getBiasShape() != dst.getBiasShape())
				throw ShapeMismatch(__FUNCTION__, dst.getBiasShape(), src.getBiasShape());

			if (not src.getWeights().getParam().isEmpty())
				addTensors(model.context(), alpha, src.getWeights().getParam(), beta, dst.getWeights().getParam(), 0.0f, dst.getWeights().getParam());
			if (not src.getBias().getParam().isEmpty())
				addTensors(model.context(), alpha, src.getBias().getParam(), beta, dst.getBias().getParam(), 0.0f, dst.getBias().getParam());
		}
	}
	void updateBatchNormStats(Graph &model)
	{
		for (int i = 0; i < model.numberOfNodes(); i++)
		{
			BatchNormalization *layer = dynamic_cast<BatchNormalization*>(&(model.getNode(i).getLayer()));
			if (layer != nullptr)
				layer->updateStatistics();
		}
	}
} /* namespace ml */

