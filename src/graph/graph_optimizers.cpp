/*
 * graph_optimizers.cpp
 *
 *  Created on: Feb 28, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/graph/Graph.hpp>
#include <minml/graph/GraphNode.hpp>
#include <minml/graph/graph_optimizers.hpp>
#include <minml/graph/CalibrationTable.hpp>
#include <minml/layers/Conv2D.hpp>
#include <minml/layers/DepthwiseConv2D.hpp>
#include <minml/layers/Dense.hpp>
#include <minml/layers/Input.hpp>
#include <minml/layers/Add.hpp>
#include <minml/layers/BatchNormalization.hpp>
#include <minml/layers/Gelu.hpp>
#include <minml/layers/GlobalBroadcastHW.hpp>
#include <minml/layers/Parameter.hpp>
#include <minml/core/Shape.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/math.hpp>

#include <memory>
#include <string>
#include <iostream>
#include <cmath>

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
		static DepthwiseConv2D depthwise_conv2d(0, 0);
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
					if (prev->getLayer().name() == depthwise_conv2d.name())
						static_cast<DepthwiseConv2D&>(prev->getLayer()).useBias(true);
					if (prev->getLayer().name() == dense.name())
						static_cast<Dense&>(prev->getLayer()).useBias(true);
					prev->getLayer().getBias().setTrainable(false);

					if (prev->getLayer().name() == conv2d.name() or prev->getLayer().name() == depthwise_conv2d.name()
							or prev->getLayer().name() == dense.name())
					{
						const Tensor &batchnorm_weights = next->getLayer().getWeights().getParam();

						Tensor &layer_weights = prev->getLayer().getWeights().getParam();
						Tensor &layer_bias = prev->getLayer().getBias().getParam();

						foldBatchnorm(graph.context(), layer_weights, layer_bias, batchnorm_weights);

						prev->getLayer().setActivationType(next->getLayer().getActivationType());

						while (next->getOutputs().size() > 0)
							GraphNode::replaceInputLink(next, prev, next->getOutputs().at(0));
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
		static GlobalBroadcastHW broadcast;

		bool has_anything_changed = false;
		for (int i = 0; i < graph.numberOfNodes(); i++)
			if (graph.getNode(i).getLayer().name() == add_layer.name() and graph.getNode(i).numberOfInputs() == 2)
			{
				GraphNode *next = &(graph.getNode(i));
				GraphNodeID input_index = -1;
				if (next->getInputNode(0)->getLayer().name() == conv2d.name() or next->getInputNode(0)->getLayer().name() == dense.name())
					input_index = std::max(input_index, graph.getNodeID(next->getInputNode(0)));
				if (next->getInputNode(1)->getLayer().name() == conv2d.name() or next->getInputNode(1)->getLayer().name() == dense.name())
					input_index = std::max(input_index, graph.getNodeID(next->getInputNode(1)));

				if (next->getInputNode(0)->getLayer().name() == broadcast.name() or next->getInputNode(1)->getLayer().name() == broadcast.name())
					input_index = -1;

				if (input_index != -1 and not next->isOutputNode())
				{
					GraphNode *prev = &(graph.getNode(input_index));
					if (can_merge_activations(prev, next))
					{
						prev->getLayer().setActivationType(next->getLayer().getActivationType());
						while (next->getOutputs().size() > 0)
							GraphNode::replaceInputLink(next, prev, next->getOutputs().at(0));
						GraphNode::link(next->getInputNode(0), prev); // only one input of Add is left now
						graph.remove_node(next);
						has_anything_changed = true;
					}
				}
			}
		return has_anything_changed;
	}

	bool FoldGelu::optimize(Graph &graph) const
	{
		static Conv2D conv2d(0, 0);
		static Dense dense(0);
		static Gelu gelu_layer;
		static GlobalBroadcastHW broadcast;

		bool has_anything_changed = false;
		for (int i = 0; i < graph.numberOfNodes(); i++)
			if (graph.getNode(i).getLayer().name() == gelu_layer.name())
			{
				GraphNode *next = &(graph.getNode(i));
				GraphNodeID input_index = -1;
				if (next->getInputNode(0)->getLayer().name() == conv2d.name() or next->getInputNode(0)->getLayer().name() == dense.name())
					input_index = std::max(input_index, graph.getNodeID(next->getInputNode(0)));
				if (next->getInputNode(1)->getLayer().name() == conv2d.name() or next->getInputNode(1)->getLayer().name() == dense.name())
					input_index = std::max(input_index, graph.getNodeID(next->getInputNode(1)));

				if (next->getInputNode(0)->getLayer().name() == broadcast.name() or next->getInputNode(1)->getLayer().name() == broadcast.name())
					input_index = -1;

				if (input_index != -1 and not next->isOutputNode())
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

	bool Quantize::optimize(Graph &graph) const
	{
		for (int n = 0; n < graph.numberOfNodes(); n++)
		{
			GraphNode &node = graph.getNode(n);
			if (node.getLayer().isQuantizable())
			{
				std::vector<AffineTransform> input_transforms(node.numberOfInputs());
				for (int i = 0; i < node.numberOfInputs(); i++)
					input_transforms.at(i) = node.getInputNode(i)->getOutputTransform();
				node.getLayer().setupQuantization(input_transforms, node.getOutputTransform());

				bool all_outputs_are_quantizable = true;
				for (int i = 0; i < node.numberOfOutputs(); i++)
					if (not node.getOutputNode(i)->getLayer().isQuantizable())
						all_outputs_are_quantizable = false;
				if (all_outputs_are_quantizable)
					node.getLayer().convertTo(DataType::INT8);
			}
			node.resolveInputShapes();
		}
		return false;
	}

} /* namespace ml */

