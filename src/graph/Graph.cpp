/*
 * Graph.cpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/graph/Graph.hpp>
#include <minml/graph/GraphNode.hpp>
#include <minml/graph/CalibrationTable.hpp>
#include <minml/core/Device.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/math.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/layers/Input.hpp>
#include <minml/layers/BatchNormalization.hpp>
#include <minml/training/LossFunction.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/utils/file_util.hpp>

#include <algorithm>
#include <thread>
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
	void Graph::addOutput(GraphNodeID node, const LossFunction &loss, float weight)
	{
		m_output_nodes.push_back(get_node(node));
		m_targets.push_back(Tensor());
		m_masks.push_back(Tensor());
		m_losses.push_back(loss.clone());
		m_loss_weights.push_back(weight);
	}
	void Graph::addOutput(GraphNodeID node)
	{
		addOutput(node, CrossEntropyLoss());
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
		return m_targets.at(index);
	}
	const Tensor& Graph::getMask(int index) const
	{
		return m_masks.at(index);
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
		if (m_targets.at(index).isEmpty())
			m_targets.at(index) = zeros_like(getOutput(index));
		return m_targets.at(index);
	}
	Tensor& Graph::getMask(int index)
	{
		if (m_masks.at(index).isEmpty())
			m_masks.at(index) = ones_like(getOutput(index));
		return m_masks.at(index);
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
			return getInputShape().firstDim();
	}
	void Graph::moveTo(Device newDevice)
	{
		m_context = std::make_shared<Context>(newDevice);
		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes[i]->changeContext(m_context);
		for (size_t i = 0; i < m_targets.size(); i++)
			m_targets[i].moveTo(newDevice);
		for (size_t i = 0; i < m_masks.size(); i++)
			m_masks[i].moveTo(newDevice);
		m_workspace = nullptr;
	}
	void Graph::convertTo(DataType newType)
	{
		m_datatype = newType;
		m_workspace = nullptr;
		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).convertTo(newType);
		for (size_t i = 0; i < m_targets.size(); i++)
			m_targets[i] = Tensor();
		for (size_t i = 0; i < m_masks.size(); i++)
			m_masks[i] = Tensor();
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
		m_workspace = nullptr;
	}

	void Graph::init()
	{
		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).getLayer().init();
	}
	void Graph::setOptimizer(const RAdam &opt)
	{
		m_optimizer = opt;
	}
	void Graph::setRegularizer(const RegularizerL2 &reg)
	{
		m_regularizer = reg;
	}
	void Graph::setGradientScaler(const GradientScaler &scaler)
	{
		m_gradient_scaler = scaler;
	}
	RAdam& Graph::getOptimizer()
	{
		return m_optimizer;
	}
	RegularizerL2& Graph::getRegularizer()
	{
		return m_regularizer;
	}
	GradientScaler& Graph::getGradientScaler()
	{
		return m_gradient_scaler;
	}

	void Graph::predict(int batchSize)
	{
		if (m_workspace == nullptr)
			create_workspace();

		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes[i]->forward(batchSize);
	}
	void Graph::train(int batchSize)
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");

		predict(batchSize);

		const float gradient_scale = (m_datatype == DataType::FLOAT16) ? m_gradient_scaler.getScale() : 1.0f;
		for (size_t i = 0; i < m_losses.size(); i++)
		{
			const Shape tmp = change_dim<0>(getTarget(i).shape(), batchSize);
			Tensor gradient = getGradient(i).view(tmp);
			Tensor output = getOutput(i).view(tmp);
			Tensor target = getTarget(i).view(tmp);
			Tensor mask = getMask(i).view(tmp);
			m_losses.at(i)->getGradient(context(), m_loss_weights[i] * gradient_scale / batchSize, gradient, output, target, mask);
		}

		if (m_workspace == nullptr)
			create_workspace();
		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes[i]->prepareForBackward();

		for (int i = static_cast<int>(m_nodes.size()) - 1; i >= 0; i--)
			m_nodes[i]->backward(batchSize);

		std::vector<Tensor> params = getParameters();
		std::vector<Tensor> param_gradients = getParameterGradients();
		m_regularizer.apply(context(), gradient_scale, params, param_gradients);

//		{
//			Json json = Json::array();
//			SerializedObject so;
//			for (size_t i = 0; i < param_gradients.size(); i++)
//			{
//				std::cout << i << " " << ml::testing::normForTest(param_gradients[i]) << '\n';
//				json[i] = param_gradients[i].serialize(so);
//			}
//			FileSaver fs("/home/maciek/alphagomoku/dump.bin");
//			fs.save(json, so);
//			exit(0);
//		}

//		FileLoader fl("/home/maciek/alphagomoku/dump.bin");
//
//		for (size_t i = 0; i < param_gradients.size() / 2; i++)
//		{
//			const Tensor loaded_dw(fl.getJson()[2 * i + 0], fl.getBinaryData());
//			const Tensor loaded_db(fl.getJson()[2 * i + 1], fl.getBinaryData());
//
//			Tensor current_dw = zeros_like(param_gradients[2 * i + 0]);
//			current_dw.copyFrom(context(), param_gradients[2 * i + 0]);
//			current_dw.convertTo(context(), DataType::FLOAT32);
//
//			Tensor current_db = zeros_like(param_gradients[2 * i + 1]);
//			current_db.copyFrom(context(), param_gradients[2 * i + 1]);
//			current_db.convertTo(context(), DataType::FLOAT32);
//			std::cout << i << " weights = " << ml::testing::diffForTest(current_dw, loaded_dw) << " (" << ml::testing::normForTest(loaded_dw) << ", "
//					<< ml::testing::normForTest(current_dw) / gradient_scale << "), bias = " << ml::testing::diffForTest(current_db, loaded_db)
//					<< " (" << ml::testing::normForTest(loaded_db) << ", " << ml::testing::normForTest(current_db) / gradient_scale << ")  "
//					<< getNode(i).getLayer().name() << "\n";
//
////			std::cout << i << " " << ml::testing::normForTest(param_gradients[2 * i + 0]) / gradient_scale << " "
////					<< ml::testing::normForTest(param_gradients[2 * i + 1]) / gradient_scale << " " << getNode(i).getLayer().name() << '\n';
//		}

		const float inv_gradient_scale = (m_datatype == DataType::FLOAT16) ? m_gradient_scaler.getInvScale(context(), param_gradients) : 1.0f;
		if (inv_gradient_scale != 0.0f)
		{
//			std::cout << "---gradients ok, updating weights with scale " << m_gradient_scaler.getScale() << "\n";
			m_optimizer.apply(context(), params, param_gradients, inv_gradient_scale);
		}
		else
		{
//			std::cout << "---some gradients are NaN or Inf, reducing scale to " << m_gradient_scaler.getScale() << " and skipping update\n";
		}
	}
	std::vector<float> Graph::getLoss(int batchSize)
	{
		std::vector<float> result(m_losses.size());
		for (size_t i = 0; i < m_losses.size(); i++)
		{
			const Shape tmp = change_dim<0>(getTarget(i).shape(), batchSize);
			Tensor output = getOutput(i).view(tmp);
			Tensor target = getTarget(i).view(tmp);
			Tensor mask = getMask(i).view(tmp);
			result[i] = m_losses.at(i)->getLoss(context(), output, target, mask) * m_loss_weights[i] / batchSize;
		}
		return result;
	}

	std::vector<Tensor> Graph::getParameters()
	{
		std::vector<Tensor> result(m_nodes.size() * 2);
		for (size_t i = 0; i < m_nodes.size(); i++)
		{
			result[2 * i + 0] = m_nodes[i]->getLayer().getWeights().getParam().view();
			result[2 * i + 1] = m_nodes[i]->getLayer().getBias().getParam().view();
		}
		return result;
	}
	std::vector<Tensor> Graph::getParameterGradients()
	{
		std::vector<Tensor> result(m_nodes.size() * 2);
		for (size_t i = 0; i < m_nodes.size(); i++)
		{
			result[2 * i + 0] = m_nodes[i]->getLayer().getWeights().getGradient().view();
			result[2 * i + 1] = m_nodes[i]->getLayer().getBias().getGradient().view();
		}
		return result;
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
	void Graph::makeTrainable(bool b)
	{
		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).makeTrainable(b);
		m_is_trainable = b;
	}
	bool Graph::isTrainable() const noexcept
	{
		return m_is_trainable;
	}
	void Graph::calibrate(CalibrationTable &table)
	{
		for (size_t i = 0; i < m_nodes.size(); i++)
		{
			if (m_nodes[i]->getLayer().isQuantizable())
			{
				if (table.getHistogram(i).isReady())
					m_nodes[i]->setOutputTransform(table.getHistogram(i).getTransform());
				else
					table.getHistogram(i).collectStatistics(m_nodes[i]->getOutputTensor());
			}
			else
				table.getHistogram(i).markAsUnused();
		}
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

		m_input_nodes.clear();
		m_output_nodes.clear();

		m_workspace.reset();

		m_datatype = DataType::FLOAT32;
	}
	Json Graph::save(SerializedObject &binary_data) const
	{
		Json result;
		result["dtype"] = toString(m_datatype);
		result["is_trainable"] = m_is_trainable;
		result["nodes"] = Json::array();
		for (size_t i = 0; i < m_nodes.size(); i++)
			result["nodes"][i] = save_node(m_nodes[i].get(), binary_data);
		if (isTrainable())
		{
			result["optimizer"] = m_optimizer.serialize(binary_data);
			result["regularizer"] = m_regularizer.serialize(binary_data);
			result["gradient_scaler"] = m_gradient_scaler.serialize(binary_data);
			result["loss_function"] = Json::array();
			for (size_t i = 0; i < m_loss_weights.size(); i++)
				result["loss_function"][i] = m_losses[i]->serialize(binary_data);
			result["loss_weights"] = Json::array();
			for (size_t i = 0; i < m_loss_weights.size(); i++)
				result["loss_weights"][i] = m_loss_weights[i];
		}
		return result;
	}
	void Graph::load(const Json &json, const SerializedObject &binary_data)
	{
		clear();

		if (json.isArray())
		{ // old format
			for (int i = 0; i < json.size(); i++)
				load_node(json[i], binary_data);
			for (int i = 0; i < numberOfNodes(); i++)
				getNode(i).getLayer().loadParameters(json[i]["layer"], binary_data);
			m_is_trainable = false;
			return;
		}

		m_datatype = typeFromString(json["dtype"]);
		m_is_trainable = json["is_trainable"].getBool();
		for (int i = 0; i < json.size(); i++)
			load_node(json["nodes"][i], binary_data);

		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).getLayer().loadParameters(json[i]["nodes"]["layer"], binary_data);

		if (isTrainable())
		{
			m_optimizer.unserialize(json["optimizer"], binary_data);
			m_regularizer.unserialize(json["regularizer"], binary_data);
			m_gradient_scaler.unserialize(json["gradient_scaler"], binary_data);
			if (json["loss_function"]["name"].getString() == "CrossEntropyLoss")
				m_losses.push_back(std::make_unique<CrossEntropyLoss>());
			if (json["loss_function"]["name"].getString() == "MeanSquaredLoss")
				m_losses.push_back(std::make_unique<MeanSquaredLoss>());
			m_losses.back()->unserialize(json["loss_function"], binary_data);
			for (int i = 0; i < json["loss_weights"].size(); i++)
				m_loss_weights.push_back(json["loss_weights"][i]);
		}
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

	Json Graph::save_node(const GraphNode *node, SerializedObject &binary_data) const
	{
		Json result;
		result["is_input_node"] = node->isInputNode();
		result["is_output_node"] = node->isOutputNode();
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
		if (index < 0 or index >= numberOfNodes())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfNodes());
		return m_nodes[index].get();
	}
	GraphNode* Graph::get_node(GraphNodeID index)
	{
		if (index < 0 or index >= numberOfNodes())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfNodes());
		return m_nodes[index].get();
	}

} /* namespace ml */

