/*
 * graph_optimizers.cpp
 *
 *  Created on: Feb 28, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/graph/Graph.hpp>
#include <minml/graph/GraphNode.hpp>
#include <minml/graph/graph_optimizers.hpp>
#include <minml/layers/Conv2D.hpp>
#include <minml/layers/Dense.hpp>
#include <minml/layers/Input.hpp>
#include <minml/layers/Add.hpp>
#include <minml/layers/BatchNormalization.hpp>
#include <minml/layers/Parameter.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>

#include <memory>
#include <string>
#include <iostream>

namespace
{
	using namespace ml;
	bool can_merge_activations(const ml::GraphNode *prev, const ml::GraphNode *next)
	{
		return (prev->getLayer().getActivationType() == ActivationType::LINEAR)
				or (prev->getLayer().getActivationType() == ActivationType::RELU and next->getLayer().getActivationType() == ActivationType::RELU);
	}
}

namespace ml
{
	bool FoldBatchNorm::optimize(Graph &graph) const
	{
		static BatchNormalization batchnorm;
		static Conv2D conv2d(0, 0);
		static Dense dense(0);

		bool has_anything_changed = false;
		for (int i = 0; i < graph.numberOfNodes(); i++)
			if (graph.getNode(i).getLayer().name() == batchnorm.name())
			{
				GraphNode *next = &(graph.getNode(i));
				GraphNode *prev = next->getInputNode(0); // BatchNorm can have only one input
				if (can_merge_activations(prev, next))
				{
					if (prev->getLayer().name() == conv2d.name())
					{
						static_cast<Conv2D&>(prev->getLayer()).useBias(true);
						static_cast<Conv2D&>(prev->getLayer()).invalidateWeightsCache();
					}
					if (prev->getLayer().name() == dense.name())
						static_cast<Dense&>(prev->getLayer()).useBias(true);
					prev->getLayer().getBias().setTrainable(false);

					if (prev->getLayer().name() == conv2d.name() or prev->getLayer().name() == dense.name())
					{
						const Tensor &batchnorm_weights = next->getLayer().getWeights().getParam();

						Tensor &layer_weights = prev->getLayer().getWeights().getParam();
						Tensor &layer_bias = prev->getLayer().getBias().getParam();

						foldBatchnorm(graph.context(), layer_weights, layer_bias, batchnorm_weights);

						prev->getLayer().setActivationType(next->getLayer().getActivationType());

						GraphNode::link(prev, next->getOutputs());
						graph.remove_node(next);
						has_anything_changed = true;
					}
				}
			}
		return has_anything_changed;
	}

	bool FoldAdd::optimize(Graph &graph) const
	{
		static Conv2D conv2d(0, 0);
		static Dense dense(0);
		static Add add_layer;
		bool has_anything_changed = false;
		for (int i = 0; i < graph.numberOfNodes(); i++)
			if (graph.getNode(i).getLayer().name() == add_layer.name() && graph.getNode(i).numberOfInputs() == 2)
			{
				GraphNode *next = &(graph.getNode(i));
				GraphNodeID input_index = -1;
				if (next->getInputNode(0)->getLayer().name() == conv2d.name() || next->getInputNode(0)->getLayer().name() == dense.name())
					input_index = std::max(input_index, graph.getNodeID(next->getInputNode(0)));
				if (next->getInputNode(1)->getLayer().name() == conv2d.name() || next->getInputNode(1)->getLayer().name() == dense.name())
					input_index = std::max(input_index, graph.getNodeID(next->getInputNode(1)));

				if (input_index != -1 && !next->isOutputNode())
				{
					GraphNode *prev = &(graph.getNode(input_index));
					if (can_merge_activations(prev, next))
					{
						prev->getLayer().setActivationType(next->getLayer().getActivationType());
						GraphNode::removeLink(prev, next);
						GraphNode::link(prev, next->getOutputs());
						GraphNode::link(next->getInputNode(0), prev); // only one input of Add is left now
						graph.remove_node(next);
						has_anything_changed = true;
					}
				}
			}
		return has_anything_changed;
	}

} /* namespace ml */

