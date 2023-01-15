/*
 * Graph.hpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_GRAPH_HPP_
#define MINML_GRAPH_HPP_

#include <minml/layers/Layer.hpp>
#include <minml/core/DataType.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Shape.hpp>
#include <minml/graph/GraphNode.hpp>

#include <memory>
#include <vector>

class Json;
namespace ml
{
	class Shape;
	class Layer;
	class Device;
	class Tensor;
	class Optimizer;
	class Regularizer;
} /* namespace ml */

namespace ml
{
	typedef int GraphNodeID;

	class Graph
	{
		private:
			std::shared_ptr<Context> m_context;

			std::vector<std::unique_ptr<GraphNode>> m_nodes;
			std::vector<std::unique_ptr<Tensor>> m_targets;

			std::vector<GraphNode*> m_input_nodes; // non-owning
			std::vector<GraphNode*> m_output_nodes; // non-owning

			std::shared_ptr<Tensor> m_workspace;
			std::unique_ptr<Tensor> m_backup_tensor;

			DataType m_datatype = DataType::FLOAT32;
			bool m_is_trainable = true;
		public:
			Graph(Device device = Device::cpu());

			Graph(const Graph &other) = delete;
			Graph& operator=(const Graph &other) = delete;
			Graph(Graph &&other) = delete;
			Graph& operator=(Graph &&other) = delete;

			Device device() const noexcept;
			DataType dtype() const noexcept;
			const Context& context() const noexcept;

			GraphNodeID addInput(const Shape &shape);
			GraphNodeID add(const Layer &layer, GraphNodeID node);
			GraphNodeID add(const Layer &layer, std::initializer_list<GraphNodeID> nodes);
			void addOutput(GraphNodeID node);

			const Tensor& getInput(int index = 0) const;
			const Tensor& getOutput(int index = 0) const;
			const Tensor& getGradient(int index = 0) const;
			const Tensor& getTarget(int index = 0) const;
			Tensor& getInput(int index = 0);
			Tensor& getOutput(int index = 0);
			Tensor& getGradient(int index = 0);
			Tensor& getTarget(int index = 0);

			Shape getInputShape(int index = 0) const;
			Shape getOutputShape(int index = 0) const;

			int numberOfInputs() const noexcept;
			int numberOfOutputs() const noexcept;
			int maxBatchSize() const;
			void moveTo(Device newDevice);
			void convertTo(DataType newType);
			void setInputShape(const Shape &shape);
			void setInputShape(const std::vector<Shape> &list);

			void setRegularizer(const Regularizer &regularizer);
			void setOptimizer(const Optimizer &optimizer);
			void init();
			void forward(int batchSize);
			void backward(int batchSize);
			std::vector<float> getLoss(int batchSize);
			void learn();

			void print() const;
			void makeNonTrainable();
			bool isTrainable() const noexcept;

			int numberOfNodes() const noexcept;
			const GraphNode& getNode(int index) const;
			GraphNode& getNode(int index);
			GraphNodeID getNodeID(const GraphNode *node) const noexcept;
			GraphNodeID getOutputNodeID(int index = 0) const noexcept;

			void clear();
			Json save(SerializedObject &binary_data) const;
			void load(const Json &json, const SerializedObject &binary_data);

			void remove_node(GraphNode *node);
		private:
			GraphNodeID add_node(const Layer &layer, const std::vector<GraphNodeID> &inputs);

			void create_workspace();
			void create_backup_tensor();

			Json save_node(const GraphNode *node, SerializedObject &binary_data) const;
			void load_node(const Json &json, const SerializedObject &binary_data);
			int index_of_node(const GraphNode *node) const noexcept;
			int index_of_layer(const Layer *layer) const noexcept;

			const GraphNode* get_node(GraphNodeID index) const;
			GraphNode* get_node(GraphNodeID index);
	};

} /* namespace ml */

#endif /* MINML_GRAPH_HPP_ */
