/*
 * GraphNode.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/graph/GraphNode.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>

#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/testing_util.hpp>

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
	void removeByIndex(std::vector<T> &vec, int idx)
	{
		if (idx < static_cast<int>(vec.size()) && idx >= 0)
			vec.erase(vec.begin() + idx);
	}
	template<typename T>
	void removeByValue(std::vector<T> &vec, T value)
	{
		const int tmp = indexOf(vec, value);
		if (tmp == -1)
			throw std::logic_error("no such value");
		removeByIndex(vec, tmp);
	}
}

namespace ml
{
	std::string TimedStat::toString() const
	{
		double time = (m_total_count == 0) ? 0.0 : (m_total_time * 1.0e-9 / m_total_count);
		char unit = ' ';
		if (time < 1.0e-3)
		{
			time *= 1.0e6;
			unit = 'u';
		}
		else
		{
			if (time < 1.0)
			{
				time *= 1.0e3;
				unit = 'm';
			}
		}
		return m_name + " = " + std::to_string(m_total_time * 1.0e-9) + "s : " + std::to_string(m_total_count) + " : " + std::to_string(time) + ' '
				+ unit + 's';
	}

	GraphNode::GraphNode(std::unique_ptr<Layer> &layer, const std::vector<GraphNode*> input_nodes) :
			m_layer(std::move(layer))
	{
		GraphNode::link(input_nodes, this);
		resolveInputShapes();
//		m_timer = TimedStat(
//				m_layer->name() + " " + m_layer->getInputShape() + " x " + m_layer->getWeightShape() + " -> " + m_layer->getOutputShape());
	}
	GraphNode::~GraphNode()
	{
//		std::cout << m_timer.toString() << '\n';
	}
	bool GraphNode::isInputNode() const
	{
		return numberOfInputs() == 0;
	}
	bool GraphNode::isOutputNode() const
	{
		return numberOfOutputs() == 0;
	}
	Shape GraphNode::getOutputShape() const
	{
		return m_output_shape;
	}
	void GraphNode::resolveInputShapes()
	{
		if (not isInputNode())
		{
			std::vector<Shape> input_shapes(numberOfInputs());
			for (int i = 0; i < numberOfInputs(); i++)
				input_shapes[i] = getInputNode(i)->getOutputShape();
			getLayer().setInputShape(input_shapes);
		}
		m_output_shape = getLayer().getOutputShape();
		m_output_tensor = nullptr;
		m_gradient_tensor = nullptr;
	}
	int GraphNode::getBackupStorage()
	{
		int tmp_size = 0;
		for (int i = 0; i < numberOfInputs(); i++)
		{
			if (getInputNode(i)->m_done_backward == true)
				tmp_size += getInputNode(i)->m_output_shape.volume();
			else
				getInputNode(i)->m_done_backward = true;
		}
		return tmp_size;
	}
	AffineTransform GraphNode::getOutputTransform() const noexcept
	{
		return m_output_transform;
	}

	int GraphNode::numberOfInputs() const noexcept
	{
		return static_cast<int>(m_input_nodes.size());
	}
	int GraphNode::numberOfOutputs() const noexcept
	{
		return static_cast<int>(m_output_nodes.size());
	}
	const GraphNode* GraphNode::getInputNode(int index) const
	{
		if (index < 0 || index >= numberOfInputs())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfInputs());
		return m_input_nodes[index];
	}
	GraphNode* GraphNode::getInputNode(int index)
	{
		if (index < 0 || index >= numberOfInputs())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfInputs());
		return m_input_nodes[index];
	}
	const GraphNode* GraphNode::getOutputNode(int index) const
	{
		if (index < 0 || index >= numberOfOutputs())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfOutputs());
		return m_output_nodes[index];
	}
	GraphNode* GraphNode::getOutputNode(int index)
	{
		if (index < 0 || index >= numberOfOutputs())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfOutputs());
		return m_output_nodes[index];
	}
	std::vector<GraphNode*> GraphNode::getInputs() const
	{
		return m_input_nodes;
	}
	std::vector<GraphNode*> GraphNode::getOutputs() const
	{
		return m_output_nodes;
	}

	void GraphNode::forward(int batchSize)
	{
		if (this->isInputNode())
			return;

		std::vector<Tensor> input(numberOfInputs());
		for (int i = 0; i < numberOfInputs(); i++)
			input[i] = change_batch(batchSize, getInputNode(i)->getOutputTensor());
		Tensor output = change_batch(batchSize, this->getOutputTensor());

//		m_timer.start();
		getLayer().forward(input, output);
//		getLayer().context().synchronize();
//		std::cout << m_layer->name() + " " + m_layer->getInputShape() + " x " + m_layer->getWeightShape() + " -> " + m_layer->getOutputShape()
//				<< " = " << testing::normForTest(output) << '\n';
//		m_timer.stop();

//		const bool emulate_low_precision = false; // getLayer().isTrainable() and getLayer().dtype() == DataType::FLOAT32;
//		if (emulate_low_precision)
//			emulateLowPrecision(getLayer().context(), output, output);
	}
	void GraphNode::backward(int batchSize, Tensor &backup_tensor)
	{
		if (isInputNode())
			return;

		std::vector<Tensor> input(numberOfInputs());
		std::vector<Tensor> gradient_prev(numberOfInputs());
		size_t offset = 0;
		for (int i = 0; i < numberOfInputs(); i++)
		{
			input[i] = change_batch(batchSize, getInputNode(i)->getOutputTensor());
			if (getInputNode(i)->m_done_backward == true)
			{
				Shape tmp_shape(getInputNode(i)->getOutputShape());
				tmp_shape[0] = batchSize;
				gradient_prev[i] = backup_tensor.view(tmp_shape, offset);
				offset += gradient_prev[i].volume();
			}
			else
				gradient_prev[i] = change_batch(batchSize, getInputNode(i)->getGradientTensor());
		}
		Tensor output = change_batch(batchSize, this->getOutputTensor());
		Tensor gradient_next = change_batch(batchSize, this->getGradientTensor());

		m_layer->backward(input, output, gradient_prev, gradient_next);

		for (int i = 0; i < numberOfInputs(); i++)
		{
			if (getInputNode(i)->m_done_backward == true)
			{
				Tensor tmp = getInputNode(i)->getGradientTensor().view(gradient_prev[i].shape());
				addTensors(m_layer->context(), tmp, tmp, gradient_prev[i]); // in-place addition
			}
			else
				getInputNode(i)->m_done_backward = true;
		}
	}
	void GraphNode::prepareForBackward()
	{
		m_done_backward = false;
	}

	const Layer& GraphNode::getLayer() const
	{
		assert(m_layer != nullptr);
		return *m_layer;
	}
	Layer& GraphNode::getLayer()
	{
		assert(m_layer != nullptr);
		return *m_layer;
	}
	const Tensor& GraphNode::getOutputTensor() const
	{
		assert(m_output_tensor != nullptr);
		return *m_output_tensor;
	}
	Tensor& GraphNode::getOutputTensor()
	{
		if (m_output_tensor == nullptr)
			m_output_tensor = std::make_unique<Tensor>(getOutputShape(), getLayer().dtype(), getLayer().device());
		return *m_output_tensor;
	}
	const Tensor& GraphNode::getGradientTensor() const
	{
		assert(m_gradient_tensor != nullptr);
		return *m_gradient_tensor;
	}
	Tensor& GraphNode::getGradientTensor()
	{
		if (m_gradient_tensor == nullptr)
			m_gradient_tensor = std::make_unique<Tensor>(getOutputShape(), getLayer().dtype(), getLayer().device());
		return *m_gradient_tensor;
	}

	void GraphNode::changeContext(std::shared_ptr<Context> &context)
	{
		getLayer().changeContext(context);
		if (m_output_tensor != nullptr)
			m_output_tensor->moveTo(context->device());
		if (m_gradient_tensor != nullptr)
			m_gradient_tensor->moveTo(context->device());
	}
	void GraphNode::convertTo(DataType newType)
	{
		getLayer().convertTo(newType);
		m_output_tensor = nullptr;
		m_gradient_tensor = nullptr;
	}
	void GraphNode::makeTrainable(bool b)
	{
		if (not b)
			m_gradient_tensor = nullptr;
		getLayer().getWeights().setTrainable(b);
		getLayer().getBias().setTrainable(b);
	}
	void GraphNode::setOutputTransform(const AffineTransform &t) noexcept
	{
		m_output_transform = t;
	}
	void GraphNode::replaceLayer(Layer *layer)
	{
		m_layer.reset(layer);
	}

	void GraphNode::link(GraphNode *prev, GraphNode *next)
	{
		prev->m_output_nodes.push_back(next);
		next->m_input_nodes.push_back(prev);
	}
	void GraphNode::link(const std::vector<GraphNode*> &prev, GraphNode *next)
	{
		for (size_t i = 0; i < prev.size(); i++)
		{
			prev[i]->m_output_nodes.push_back(next);
			next->m_input_nodes.push_back(prev[i]);
		}
	}
	void GraphNode::link(GraphNode *prev, const std::vector<GraphNode*> &next)
	{
		for (size_t i = 0; i < next.size(); i++)
		{
			prev->m_output_nodes.push_back(next[i]);
			next[i]->m_input_nodes.push_back(prev);
		}
	}
	void GraphNode::link(const std::vector<GraphNode*> &prev, const std::vector<GraphNode*> &next)
	{
		for (size_t i = 0; i < prev.size(); i++)
			for (size_t j = 0; j < next.size(); j++)
			{
				prev[i]->m_output_nodes.push_back(next[j]);
				next[j]->m_input_nodes.push_back(prev[i]);
			}
	}
	void GraphNode::removeLink(GraphNode *prev, GraphNode *next)
	{
		removeByValue(prev->m_output_nodes, next);
		removeByValue(next->m_input_nodes, prev);
	}
	void GraphNode::replaceInputLink(GraphNode *old_prev, GraphNode *new_prev, GraphNode *next)
	{
		const int idx = indexOf(next->m_input_nodes, old_prev);
		next->m_input_nodes.at(idx) = new_prev; // replace input link old_prev->next into new_prev->next
		removeByValue(old_prev->m_output_nodes, next); // remove link old_prev->next
		new_prev->m_output_nodes.push_back(next); // create link new_prev->next
	}
	void GraphNode::replaceOutputLink(GraphNode *prev, GraphNode *old_next, GraphNode *new_next)
	{
		const int idx = indexOf(prev->m_output_nodes, old_next);
		prev->m_output_nodes.at(idx) = new_next; // replace link prev->old_next into prev->new_next
		removeByValue(prev->m_output_nodes, old_next); // remove link prev->old_next
		new_next->m_input_nodes.push_back(prev); // create link prev->new_next
	}
	void GraphNode::removeAllLinks()
	{
		while (m_input_nodes.size() > 0)
			removeLink(m_input_nodes[0], this);
		while (m_output_nodes.size() > 0)
			removeLink(this, m_output_nodes[0]);
	}
	/*
	 * private
	 */
	Tensor GraphNode::change_batch(int batch_size, const Tensor &other)
	{
		Shape tmp(other.shape());
		tmp[0] = batch_size;
		return other.view(tmp);
	}

} /* namespace ml */

