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
#include <minml/layers/Add.hpp>
#include <minml/layers/BatchNormalization.hpp>
#include <minml/layers/ChannelScaling.hpp>
#include <minml/layers/Conv2D.hpp>
#include <minml/layers/Dense.hpp>
#include <minml/layers/DepthwiseConv2D.hpp>
#include <minml/layers/FusedConvBlock.hpp>
#include <minml/layers/SqueezeAndExcitation.hpp>
#include <minml/layers/GlobalAveragePooling.hpp>
#include <minml/layers/Input.hpp>
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

	bool is_linear(const GraphNode *node) noexcept
	{
		return (node == nullptr) ? false : node->getLayer().getActivationType() == ActivationType::LINEAR;
	}
	bool is_relu(const GraphNode *node) noexcept
	{
		return (node == nullptr) ? false : node->getLayer().getActivationType() == ActivationType::RELU;
	}
	bool is_sigmoid(const GraphNode *node) noexcept
	{
		return (node == nullptr) ? false : node->getLayer().getActivationType() == ActivationType::SIGMOID;
	}

	bool is_dense(const GraphNode *node) noexcept
	{
		static const Dense layer(0);
		return (node == nullptr) ? false : node->getLayer().name() == layer.name();
	}
	bool is_conv2d(const GraphNode *node) noexcept
	{
		static const Conv2D layer(0, 0);
		return (node == nullptr) ? false : node->getLayer().name() == layer.name();
	}
	bool is_depthwise_conv2d(const GraphNode *node) noexcept
	{
		static const DepthwiseConv2D layer(0, 0);
		return (node == nullptr) ? false : node->getLayer().name() == layer.name();
	}
	bool is_global_average_pooling(const GraphNode *node) noexcept
	{
		static const GlobalAveragePooling layer;
		return (node == nullptr) ? false : node->getLayer().name() == layer.name();
	}
	bool is_channel_scaling(const GraphNode *node) noexcept
	{
		static const ChannelScaling layer;
		return (node == nullptr) ? false : node->getLayer().name() == layer.name();
	}

	bool can_merge_activations(const GraphNode *prev, const GraphNode *next)
	{
		return is_linear(prev) or (is_relu(prev) and is_relu(next));
	}

	template<int N>
	std::array<GraphNode*, N> get_consecutive_nodes(GraphNode &node)
	{
		std::array<GraphNode*, N> result;
		result.fill(nullptr);
		result[0] = &node;
		for (int i = 1; i < N; i++)
		{
			if (result[i - 1]->numberOfOutputs() == 1)
				result[i] = result[i - 1]->getOutputNode(0);
			else
				break;
		}
		return result;
	}
}

namespace ml
{
	bool FoldBatchNorm::optimize(Graph &graph) const
	{
		static const BatchNormalization batchnorm;

		bool has_anything_changed = false;
		for (int i = 0; i < graph.numberOfNodes(); i++)
			if (graph.getNode(i).getLayer().name() == batchnorm.name())
			{
				GraphNode *next = &(graph.getNode(i));
				GraphNode *prev = next->getInputNode(0); // BatchNorm can have only one input
				if (can_merge_activations(prev, next) and prev->numberOfInputs() == 1)
				{
					if (is_conv2d(prev))
					{
						static_cast<Conv2D&>(prev->getLayer()).useBias(true);
						static_cast<Conv2D&>(prev->getLayer()).invalidateWeightsCache();
					}
					if (is_depthwise_conv2d(prev))
						static_cast<DepthwiseConv2D&>(prev->getLayer()).useBias(true);
					if (is_dense(prev))
						static_cast<Dense&>(prev->getLayer()).useBias(true);

					if (is_conv2d(prev) or is_depthwise_conv2d(prev) or is_dense(prev))
					{
						const Tensor &bn_weights = next->getLayer().getWeights().getParam();
						const Tensor &bn_bias = next->getLayer().getBias().getParam();
						const Tensor &bn_avg_var = static_cast<BatchNormalization&>(next->getLayer()).getStatistics();

						Tensor &layer_weights = prev->getLayer().getWeights().getParam();
						Tensor &layer_bias = prev->getLayer().getBias().getParam();

						foldBatchnorm(graph.context(), layer_weights, layer_bias, bn_weights, bn_bias, bn_avg_var);

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
		static const Conv2D conv2d(0, 0);
		static const Dense dense(0);
		static const Add add_layer;

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

	bool Quantize::optimize(Graph &graph, int bits) const
	{
		for (int n = 0; n < graph.numberOfNodes(); n++)
		{
			GraphNode &node = graph.getNode(n);
			if (node.getLayer().isQuantizable())
			{
				std::vector<AffineTransform> input_transforms(node.numberOfInputs());
				for (int i = 0; i < node.numberOfInputs(); i++)
					input_transforms.at(i) = node.getInputNode(i)->getOutputTransform();
				node.getLayer().setupQuantization(input_transforms, node.getOutputTransform(), bits);

				bool all_outputs_are_quantizable = true;
				for (int i = 0; i < node.numberOfOutputs(); i++)
					if (not node.getOutputNode(i)->getLayer().isQuantizable())
						all_outputs_are_quantizable = false;
				if (all_outputs_are_quantizable)
					node.getLayer().convertTo(get_quantized_dtype(bits));
			}
			node.resolveInputShapes();
		}
		return false;
	}

	bool FuseConvBlock::optimize(Graph &graph) const
	{
		bool has_anything_changed = false;
		for (int i = 0; i < graph.numberOfNodes(); i++)
		{
			const std::array<GraphNode*, 3> nodes = get_consecutive_nodes<3>(graph.getNode(i));

			const bool first_node_match = is_depthwise_conv2d(nodes[0]) and is_linear(nodes[0]);
			const bool second_node_match = is_conv2d(nodes[1]) and is_relu(nodes[1]);
			const bool third_node_match = is_conv2d(nodes[2]) and is_linear(nodes[2]);

			if (first_node_match and second_node_match and third_node_match)
			{
				std::unique_ptr<FusedConvBlock> fused_block = std::make_unique<FusedConvBlock>((DepthwiseConv2D&) nodes[0]->getLayer(),
						(Conv2D&) nodes[1]->getLayer(), (Conv2D&) nodes[2]->getLayer());
				fused_block->setInputShape(nodes[0]->getLayer().getInputShape());
				nodes[0]->replaceLayer(fused_block.release());
				nodes[0]->resolveInputShapes();

				while (nodes[2]->numberOfOutputs() > 0)
					GraphNode::replaceInputLink(nodes[2], nodes[0], nodes[2]->getOutputNode(0));

				graph.remove_node(nodes[1]);
				graph.remove_node(nodes[2]);

				has_anything_changed = true;
			}
		}
		return has_anything_changed;
	}

	bool FuseSEBlock::optimize(Graph &graph) const
	{
		bool has_anything_changed = false;
		for (int i = 0; i < graph.numberOfNodes(); i++)
		{
			const std::array<GraphNode*, 4> nodes = get_consecutive_nodes<4>(graph.getNode(i));

			const bool first_node_match = is_global_average_pooling(nodes[0]);
			const bool second_node_match = is_dense(nodes[1]) and is_relu(nodes[1]);
			const bool third_node_match = is_dense(nodes[2]) and is_sigmoid(nodes[2]);
			const bool fourth_node_match = is_channel_scaling(nodes[3]);

			if (first_node_match and second_node_match and third_node_match and fourth_node_match)
			{
				std::unique_ptr<SqueezeAndExcitation> fused_block = std::make_unique<SqueezeAndExcitation>((Dense&) nodes[1]->getLayer(),
						(Dense&) nodes[2]->getLayer());
				fused_block->setInputShape(nodes[0]->getLayer().getInputShape());
				nodes[0]->replaceLayer(fused_block.release());
				nodes[0]->resolveInputShapes();

				while (nodes[3]->numberOfOutputs() > 0)
					GraphNode::replaceInputLink(nodes[3], nodes[0], nodes[3]->getOutputNode(0));

				graph.remove_node(nodes[1]);
				graph.remove_node(nodes[2]);
				graph.remove_node(nodes[3]);

				has_anything_changed = true;

			}
		}
		return has_anything_changed;
	}

} /* namespace ml */

