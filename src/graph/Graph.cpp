/*
 * Graph.cpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/graph/Graph.hpp>
#include <minml/graph/GraphNode.hpp>
#include <minml/core/Device.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/Input.hpp>
#include <minml/training/CrossEntropyLoss.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>

#include <algorithm>
#include <cmath>

namespace
{
	template<typename T>
	int indexOf(const std::vector<T> &vec, T value)
	{
		for (size_t i = 0; i < vec.size(); i++)
			if (vec[i] == value)
				return i;
		return -1;
	}
	template<typename T>
	void removeByIndex(std::vector<T> &vec, size_t idx)
	{
		if (idx < vec.size())
			vec.erase(vec.begin() + idx);
	}
	template<typename T>
	void removeByValue(std::vector<T> &vec, T value)
	{
		removeByIndex(vec, indexOf(vec, value));
	}
}

namespace ml
{

	Graph::Graph(Device device) :
			m_context(std::make_shared<Context>(device))
	{
	}

	Device Graph::device() const noexcept
	{
		return m_context->device();
	}
	DataType Graph::dtype() const noexcept
	{
		return m_datatype;
	}
	const Context& Graph::context() const noexcept
	{
		return *m_context;
	}

	GraphNodeID Graph::addInput(const Shape &shape)
	{
		return add_node(Input(shape), { });
	}
	GraphNodeID Graph::add(const Layer &layer, GraphNodeID node)
	{
		return add_node(layer, { node });
	}
	GraphNodeID Graph::add(const Layer &layer, std::initializer_list<GraphNodeID> nodes)
	{
		if (nodes.size() == 0)
			throw LogicError(METHOD_NAME, "nodes list must not be empty");
		return add_node(layer, nodes);
	}
	void Graph::addOutput(GraphNodeID node, float weight)
	{
		m_output_nodes.push_back(get_node(node));
		m_targets.push_back(nullptr);
		m_target_weights.push_back(weight);
	}

	const Tensor& Graph::getInput(int index) const
	{
		return m_input_nodes.at(index)->getOutputTensor();
	}
	const Tensor& Graph::getOutput(int index) const
	{
		return m_output_nodes.at(index)->getOutputTensor();
	}
	const Tensor& Graph::getGradient(int index) const
	{
		return m_output_nodes.at(index)->getGradientTensor();
	}
	const Tensor& Graph::getTarget(int index) const
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		if (m_targets.at(index) == nullptr)
			throw UninitializedObject(METHOD_NAME, "target tensor was not initialized");
		return *(m_targets.at(index));
	}
	Tensor& Graph::getInput(int index)
	{
		return m_input_nodes.at(index)->getOutputTensor();
	}
	Tensor& Graph::getOutput(int index)
	{
		return m_output_nodes.at(index)->getOutputTensor();
	}
	Tensor& Graph::getGradient(int index)
	{
		return m_output_nodes.at(index)->getGradientTensor();
	}
	Tensor& Graph::getTarget(int index)
	{
		if (m_targets.at(index) == nullptr)
			m_targets.at(index) = std::make_unique<Tensor>(getOutput(index).shape(), dtype(), device());
		return *(m_targets.at(index));
	}

	Shape Graph::getInputShape(int index) const
	{
		return m_input_nodes.at(index)->getOutputShape();
	}
	Shape Graph::getOutputShape(int index) const
	{
		return m_output_nodes.at(index)->getOutputShape();
	}

	int Graph::numberOfInputs() const noexcept
	{
		return static_cast<int>(m_input_nodes.size());
	}
	int Graph::numberOfOutputs() const noexcept
	{
		return static_cast<int>(m_output_nodes.size());
	}
	int Graph::maxBatchSize() const
	{
		if (numberOfInputs() == 0)
			return 0;
		else
			return getOutputShape().firstDim();
	}
	void Graph::moveTo(Device newDevice)
	{
		if (newDevice == device())
			return;

		m_context = std::make_shared<Context>(newDevice);
		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes[i]->changeContext(m_context);
		if (m_workspace != nullptr)
			m_workspace->moveTo(newDevice);
		if (m_backup_tensor != nullptr)
			m_backup_tensor->moveTo(newDevice);
		for (size_t i = 0; i < m_targets.size(); i++)
			if (m_targets[i] != nullptr)
				m_targets[i]->moveTo(newDevice);
	}
	void Graph::convertTo(DataType newType)
	{
		m_datatype = newType;
		m_workspace = nullptr;
		m_backup_tensor = nullptr;
		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).convertTo(newType);
	}
	void Graph::setInputShape(const Shape &shape)
	{
		setInputShape(std::vector<Shape>( { shape }));
	}
	void Graph::setInputShape(const std::vector<Shape> &list)
	{
		for (int i = 0; i < numberOfInputs(); i++)
			m_input_nodes[i]->getLayer().setInputShape(list[i]);

		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes[i]->resolveInputShapes();
		m_backup_tensor = nullptr;
		m_workspace = nullptr;
	}

	void Graph::setOptimizer(const Optimizer &optimizer)
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).getLayer().setOptimizer(optimizer);
	}
	void Graph::setRegularizer(const Regularizer &regularizer)
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).getLayer().setRegularizer(regularizer);
	}
	void Graph::init()
	{
		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).getLayer().init();
	}
	void Graph::forward(int batchSize)
	{
		if (m_workspace == nullptr)
			create_workspace();
		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes[i]->forward(batchSize);
	}
	void Graph::backward(int batchSize)
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		if (m_backup_tensor == nullptr)
			create_backup_tensor();
		if (m_workspace == nullptr)
			create_workspace();
		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes[i]->prepareForBackward();

		for (size_t i = 0; i < m_targets.size(); i++)
		{
			CrossEntropyLoss loss(m_target_weights.at(i));
			Shape tmp(getTarget(i).shape());
			tmp[0] = batchSize;
			Tensor gradient = getGradient(i).view(tmp);
			Tensor output = getOutput(i).view(tmp);
			Tensor target = getTarget(i).view(tmp);
			loss.getGradient(context(), gradient, output, target);

//			if (std::isnan(testing::normForTest(gradient)))
//			{
//				std::cout << "Loss at output " << i << "is NaN\n";
//				exit(9);
//			}

//			if (i == 2)
//			{
//				for (int j = 0; j < 225; j++)
//				{
//					std::cout << "output =";
//					for (int k = 0; k < 3; k++)
//						std::cout << ' ' << output.get( { 0, j / 15, j % 15, k });
//					std::cout << ", target =";
//					for (int k = 0; k < 3; k++)
//						std::cout << ' ' << target.get( { 0, j / 15, j % 15, k });
//					std::cout << ", gradient =";
//					for (int k = 0; k < 3; k++)
//						std::cout << ' ' << gradient.get( { 0, j / 15, j % 15, k });
//					std::cout << '\n';
//				}
//			}
		}

		for (int i = static_cast<int>(m_nodes.size()) - 1; i >= 0; i--)
		{
			m_nodes[i]->backward(batchSize, *m_backup_tensor);
//			if (std::isnan(testing::normForTest(m_nodes[i]->getGradientTensor())))
//			{
//				std::cout << "Loss at layer " << m_nodes[i]->getLayer().name() << " is NaN\n";
//				exit(9);
//			}
//			else
//				std::cout << "successfully backwarded layer " << m_nodes[i]->getLayer().name() << '\n';
		}
	}
	std::vector<float> Graph::getLoss(int batchSize)
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		if (m_workspace == nullptr)
			create_workspace();

		std::vector<float> result(numberOfOutputs());
		for (size_t i = 0; i < m_targets.size(); i++)
		{
			CrossEntropyLoss loss(m_target_weights.at(i));
			Shape tmp(getTarget(i).shape());
			tmp[0] = batchSize;
			Tensor output = getOutput(i).view(tmp);
			Tensor target = getTarget(i).view(tmp);
			result[i] = loss.getLoss(context(), output, target);
		}
		return result;
	}
	void Graph::learn()
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		if (m_workspace == nullptr)
			create_workspace();

		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).getLayer().learn();
	}
	void Graph::setLearningRate(float lr)
	{
		for (int i = 0; i < numberOfNodes(); i++)
		{
			getNode(i).getLayer().getWeights().getOptimizer().setLearningRate(lr);
			getNode(i).getLayer().getBias().getOptimizer().setLearningRate(lr);
		}
	}

	void Graph::print() const
	{
		for (size_t i = 0; i < m_nodes.size(); i++)
		{
			GraphNode *node = m_nodes[i].get();
			std::cout << i << ' ' << m_nodes[i]->getLayer().name() << " (" << toString(m_nodes[i]->getLayer().getActivationType()) << ") : "
					<< node->getOutputShape() << " : " << node->getLayer().dtype() << " : {";
			for (int j = 0; j < node->numberOfInputs(); j++)
			{
				if (j != 0)
					std::cout << ',';
				std::cout << index_of_node(node->getInputNode(j));
			}
			std::cout << "} -> {";
			for (int j = 0; j < node->numberOfOutputs(); j++)
			{
				if (j != 0)
					std::cout << ',';
				std::cout << index_of_node(node->getOutputNode(j));
			}
			std::cout << "}\n";
		}
		for (size_t i = 0; i < m_output_nodes.size(); i++)
			std::cout << "Output:" << i << " : {" << index_of_node(m_output_nodes[i]) << "} : " << m_output_nodes[i]->getOutputShape() << std::endl;
	}
	void Graph::makeNonTrainable()
	{
		m_targets.clear();
		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).makeNonTrainable();
		m_is_trainable = false;
	}
	bool Graph::isTrainable() const noexcept
	{
		return m_is_trainable;
	}

	int Graph::numberOfNodes() const noexcept
	{
		return static_cast<int>(m_nodes.size());
	}
	const GraphNode& Graph::getNode(int index) const
	{
		return *(m_nodes.at(index));
	}
	GraphNode& Graph::getNode(int index)
	{
		return *(m_nodes.at(index));
	}
	GraphNodeID Graph::getNodeID(const GraphNode *node) const noexcept
	{
		return index_of_node(node);
	}
	GraphNodeID Graph::getOutputNodeID(int index) const noexcept
	{
		return getNodeID(m_output_nodes.at(index));
	}

	void Graph::clear()
	{
		m_context = std::make_shared<Context>();
		m_nodes.clear();
		m_targets.clear();

		m_input_nodes.clear();
		m_output_nodes.clear();

		m_workspace.reset();
		m_backup_tensor.reset();

		m_datatype = DataType::FLOAT32;
	}
	Json Graph::save(SerializedObject &binary_data) const
	{
		Json result(JsonType::Array);
		for (int i = 0; i < static_cast<int>(m_nodes.size()); i++)
			result[i] = save_node(m_nodes[i].get(), binary_data);
		return result;
	}
	void Graph::load(const Json &json, const SerializedObject &binary_data)
	{
		clear();
		for (int i = 0; i < json.size(); i++)
			load_node(json[i], binary_data);

		m_is_trainable = false;
		for (int i = 0; i < numberOfNodes(); i++)
		{
			getNode(i).getLayer().loadParameters(json[i]["layer"], binary_data);
			m_is_trainable |= getNode(i).getLayer().isTrainable();
		}
		if (isTrainable())
			m_targets = std::vector<std::unique_ptr<Tensor>>(numberOfOutputs());
	}

	GraphNodeID Graph::add_node(const Layer &layer, const std::vector<GraphNodeID> &inputs)
	{
		std::unique_ptr<Layer> cloned_layer = layer.clone(layer.getConfig());

		std::vector<GraphNode*> tmp(inputs.size());
		for (size_t i = 0; i < inputs.size(); i++)
			tmp[i] = get_node(inputs[i]);
		m_nodes.push_back(std::make_unique<GraphNode>(cloned_layer, tmp));
		m_nodes.back()->getLayer().changeContext(m_context);

		if (m_nodes.back()->isInputNode())
			m_input_nodes.push_back(m_nodes.back().get());
		return static_cast<GraphNodeID>(m_nodes.size() - 1);
	}
	void Graph::remove_node(GraphNode *node)
	{
		auto index_in_input_nodes = std::find(m_input_nodes.begin(), m_input_nodes.end(), node);
		auto index_in_output_nodes = std::find(m_output_nodes.begin(), m_output_nodes.end(), node);

		if (index_in_input_nodes != m_input_nodes.end())
		{
			if (node->numberOfOutputs() > 1)
				throw LogicError(METHOD_NAME, "trying to remove input node");
			else
				*index_in_input_nodes = node->getOutputNode(0);
		}
		if (index_in_output_nodes != m_output_nodes.end())
		{
			if (node->numberOfInputs() > 1)
				throw LogicError(METHOD_NAME, "trying to remove output node");
			else
				*index_in_output_nodes = node->getInputNode(0);
		}
		node->removeAllLinks();
		removeByIndex(m_nodes, index_of_node(node));
	}

	void Graph::create_workspace()
	{
		int tmp = 0;
		for (int i = 0; i < numberOfNodes(); i++)
			tmp = std::max(tmp, getNode(i).getLayer().getWorkspaceSize());
		m_workspace = std::make_shared<Tensor>(Shape( { tmp }), dtype(), device());

		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).getLayer().setWorkspace(m_workspace);
	}
	void Graph::create_backup_tensor()
	{
		int tmp = 0;
		for (size_t i = 0; i < m_nodes.size(); i++)
			tmp = std::max(tmp, m_nodes[i]->getBackupStorage());
		m_backup_tensor = std::make_unique<Tensor>(Shape( { tmp }), dtype(), device());
	}

	Json Graph::save_node(const GraphNode *node, SerializedObject &binary_data) const
	{
		Json result;
		result["is_input_node"] = node->isInputNode();
		result["is_output_node"] = node->isOutputNode();
		if(node->isOutputNode())
		{
			result["loss_weight"] = m_target_weights.at(0); // TODO
		}
		result["input_nodes"] = Json(JsonType::Array);
		for (int i = 0; i < node->numberOfInputs(); i++)
			result["input_nodes"][i] = index_of_node(node->getInputNode(i));

		Json cfg = node->getLayer().getConfig();
		const Json tmp = node->getLayer().saveParameters(binary_data);
		cfg.append(tmp);
		result["layer"] = cfg;
		return result;
	}
	void Graph::load_node(const Json &json, const SerializedObject &binary_data)
	{
		std::unique_ptr<Layer> layer = loadLayer(json["layer"], binary_data);
		layer->changeContext(m_context);

		std::vector<GraphNode*> inputs;
		for (int i = 0; i < json["input_nodes"].size(); i++)
			inputs.push_back(m_nodes[static_cast<int>(json["input_nodes"][i])].get());
		m_nodes.push_back(std::make_unique<GraphNode>(layer, inputs));

		if (json["is_input_node"])
			m_input_nodes.push_back(m_nodes.back().get());
		if (json["is_output_node"])
			m_output_nodes.push_back(m_nodes.back().get());
	}
	int Graph::index_of_node(const GraphNode *node) const noexcept
	{
		for (size_t i = 0; i < m_nodes.size(); i++)
			if (m_nodes[i].get() == node)
				return i;
		return -1;
	}
	const GraphNode* Graph::get_node(GraphNodeID index) const
	{
		if (index < 0 || index >= numberOfNodes())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfNodes());
		return m_nodes[index].get();
	}
	GraphNode* Graph::get_node(GraphNodeID index)
	{
		if (index < 0 || index >= numberOfNodes())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfNodes());
		return m_nodes[index].get();
	}

} /* namespace ml */

