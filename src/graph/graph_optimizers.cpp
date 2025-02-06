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
	Tensor quantize_weights(const Tensor &weights, Tensor &quantized)
	{
		const int first_dim = weights.shape().firstDim();
		const int last_dim = weights.shape().volumeWithoutFirstDim();
		const Tensor tmp = weights.view( { first_dim, last_dim });
		Tensor channel_scales( { first_dim }, DataType::FLOAT32, weights.device());
		for (int i = 0; i < first_dim; i++)
		{
			float max_abs_value = 0.0f;
			for (int j = 0; j < last_dim; j++)
				max_abs_value = std::max(max_abs_value, std::fabs(tmp.get( { i, j })));

			const float scale = 127.0f / max_abs_value;
			for (int j = 0; j < last_dim; j++)
			{
				const int8_t q = std::max(-127.0f, std::min(127.0f, tmp.get( { i, j }) * scale));
				quantized.set(q, { i, j });
			}
			channel_scales.set(scale, { i });
		}
		return channel_scales;
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

	Quantize::Quantize(const CalibrationTable &table) :
			m_table(table)
	{
	}
	bool Quantize::optimize(Graph &graph) const
	{
		static Conv2D conv2d(0, 0);
		static DepthwiseConv2D depthwise_conv2d(0, 0);
		static Dense dense(0);
		static Input input;

		for (int n = 0; n < graph.numberOfNodes(); n++)
			if (graph.getNode(n).getLayer().isQuantizable())
			{
				// setup quantization scales
//				GraphNode *next = &(graph.getNode(n));
//				for(int  j=0;j<graph.getNode(n).numberOfInputs();j++)
//					graph.getNode(n).getInputNode(j)->getLayer().
//				GraphNode *prev = next->getInputNode(0); // BatchNorm can have only one input
			}
//				graph.getNode(n).setQuantization(m_table.getHistogram(n).getScale(), m_table.getHistogram(n).getShift());

		bool has_anything_changed = false;
		for (int n = 0; n < graph.numberOfNodes(); n++)
		{
			Layer &old_layer = graph.getNode(n).getLayer();
			if (not old_layer.isQuantizable())
				continue;

			std::cout << "checking layer " << old_layer.name() << '\n';
			if ((old_layer.name() == input.name()) and old_layer.dtype() != DataType::INT8)
			{
//				old_layer.setDataType(DataType::INT8);
//				old_layer.getWeights().getParam() = Tensor(old_layer.getWeightShape(), DataType::INT8, Device::cpu());
				has_anything_changed = true;
			}
//			if ((old_layer.name() == flatten.name()) and old_layer.dtype() != DataType::INT8)
//			{
//				if (graph.getNode(n).getOutputNode(0)->getLayer().dtype() == DataType::INT8)
//				{
//					old_layer.setDataType(DataType::INT8);
//					old_layer.getWeights().getParam() = Tensor(old_layer.getWeightShape(), DataType::INT8, Device::cpu());
//				}
//				has_anything_changed = true;
//			}
//
//			if ((old_layer.name() == conv2d.name() or old_layer.name() == dense.name()) and old_layer.dtype() != DataType::INT8)
//			{
//				std::cout << "   match\n";
//
//				if (old_layer.name() == conv2d.name())
//					dynamic_cast<Conv2D&>(old_layer).useBias(true);
//				if (old_layer.name() == dense.name())
//					dynamic_cast<Dense&>(old_layer).useBias(true);
//
//				old_layer.setDataType(DataType::INT8);
//				const float input_shift = graph.getNode(n).getInputNode(0)->getQuantizationShift();
//
//				const int first_dim = old_layer.getWeightShape().firstDim();
//				const int last_dim = old_layer.getWeightShape().volumeWithoutFirstDim();
//
//				Tensor quantized_weights(old_layer.getWeightShape(), DataType::INT8, Device::cpu());
//				std::cout << old_layer.getWeightShape().toString() << '\n';
//
//				const Tensor tmp_old = old_layer.getWeights().getParam().view( { first_dim, last_dim });
//				Tensor tmp_new = quantized_weights.view( { first_dim, last_dim });
//
//				for (int i = 0; i < first_dim; i++)
//				{
//					float max_abs_value = 0.0f;
//					for (int j = 0; j < last_dim; j++)
//						max_abs_value = std::max(max_abs_value, std::fabs(tmp_old.get<float>( { i, j })));
//
//					// fp32 = sum_over_q[(int8 * scale + shift) * q * scale]
//					const float inv_channel_scale = 127 / max_abs_value;
//					const float channel_scale = max_abs_value / 127;
//					int sum_q = 0;
//					for (int j = 0; j < last_dim; j++)
//					{
//						const int q = std::max(-127, std::min(127, static_cast<int>(std::round(tmp_old.get<float>( { i, j }) * inv_channel_scale))));
//						//							std::cout << i << ", " << j << " : " << tmp_old.get<float>( { i, j }) << " * " << inv_channel_scale << " = " << q << '\n';
//						sum_q += q;
//						tmp_new.set<int8_t>(q, { i, j });
//					}
//					// quantization shift of input tensor is absorbed into the bias of a layer
//					const float new_bias = old_layer.getBias().getParam().get<float>( { i }) + input_shift * sum_q * channel_scale;
//					old_layer.getBias().getParam().set(new_bias, { i });
//					old_layer.getChannelScales().set(channel_scale, { i });
//
//					//						std::cout << "channel " << i << " max abs = " << max_abs_value << ", sum_q = " << sum_q << ", bias = "
//					//								<< old_layer.getBias().getParam().get<float>( { i }) << ", offset = " << (input_shift * sum_q * channel_scale) << '\n';
//					//						exit(0);
//				}
//				old_layer.getWeights().getParam() = quantized_weights;
//				has_anything_changed = true;

//			}
		}
		return has_anything_changed;
	}

} /* namespace ml */

