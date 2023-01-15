/*
 * GraphNode.hpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_GRAPH_GRAPHNODE_HPP_
#define MINML_GRAPH_GRAPHNODE_HPP_

#include <minml/core/DataType.hpp>
#include <minml/core/Shape.hpp>

#include <memory>
#include <vector>

class Json;
namespace ml
{
	class Shape;
	class Layer;
	class Device;
	class Tensor;
	class Context;
} /* namespace ml */

namespace ml
{

	class GraphNode
	{
		private:
			std::unique_ptr<Layer> m_layer;

			std::vector<GraphNode*> m_input_nodes; // non-owning
			std::vector<GraphNode*> m_output_nodes; // non-owning

			std::unique_ptr<Tensor> m_output_tensor;
			std::unique_ptr<Tensor> m_gradient_tensor;
			Shape m_output_shape;

			bool m_done_backward = false;
		public:
			GraphNode(std::unique_ptr<Layer> &layer, const std::vector<GraphNode*> input_nodes);

			bool isInputNode() const;
			bool isOutputNode() const;
			Shape getOutputShape() const;
			void resolveInputShapes();
			int getBackupStorage();

			int numberOfInputs() const noexcept;
			int numberOfOutputs() const noexcept;
			const GraphNode* getInputNode(int index) const;
			GraphNode* getInputNode(int index);
			const GraphNode* getOutputNode(int index) const;
			GraphNode* getOutputNode(int index);
			std::vector<GraphNode*> getInputs() const;
			std::vector<GraphNode*> getOutputs() const;

			void forward(int batch_size);
			void backward(int batch_size, Tensor &backup_tensor);
			void prepareForBackward();

			const Layer& getLayer() const;
			Layer& getLayer();
			const Tensor& getOutputTensor() const;
			Tensor& getOutputTensor();
			const Tensor& getGradientTensor() const;
			Tensor& getGradientTensor();

			void changeContext(std::shared_ptr<Context> &context);
			void convertTo(DataType newType);
			void makeNonTrainable();

			static void link(GraphNode *prev, GraphNode *next);
			static void link(const std::vector<GraphNode*> &prev, GraphNode *next);
			static void link(GraphNode *prev, const std::vector<GraphNode*> &next);
			static void link(const std::vector<GraphNode*> &prev, const std::vector<GraphNode*> &next);

			static void removeLink(GraphNode *prev, GraphNode *next);
			void removeAllLinks();

			Tensor changeBatch(int batch_size, const Tensor &other);
	};

} /* namespace ml */

#endif /* MINML_GRAPH_GRAPHNODE_HPP_ */
