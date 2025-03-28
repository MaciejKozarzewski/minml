/*
 * Trainer.cpp
 *
 *  Created on: Mar 24, 2025
 *      Author: Maciej Kozarzewski
 */

#include <minml/training/Trainer.hpp>
#include <minml/graph/Graph.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/math.hpp>
#include <minml/utils/testing_util.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/file_util.hpp>

#include <thread>

namespace ml
{
//	Trainer::Trainer(Graph &graph, DataType dtype) :
//			m_graph_ptr(&graph),
//			m_targets(graph.numberOfOutputs()),
//			m_masks(graph.numberOfOutputs()),
//			m_losses(graph.numberOfOutputs()),
//			m_loss_weights(graph.numberOfOutputs()),
//			m_training_dtype(dtype)
//	{
//		switch (dtype)
//		{
//			case DataType::FLOAT16:
//			{
//				const std::vector<Tensor> params = graph.getParameters();
//				m_fp32_weights_copy.resize(params.size());
//				for (size_t i = 0; i < params.size(); i++)
//				{
//					m_fp32_weights_copy[i] = zeros_like(params[i]);
//					m_fp32_weights_copy[i].copyFrom(graph.context(), params[i]);
//				}
//				graph.convertTo(dtype);
//				break;
//			}
//			case DataType::FLOAT32:
//			{
//				graph.convertTo(dtype);
//				break;
//			}
//			default:
//				throw NotImplemented(METHOD_NAME, "training in types other than fp32 or fp16 are not supported");
//		}
//	}
//
//	Graph& Trainer::graph()
//	{
//		assert(m_graph_ptr != nullptr);
//		return *m_graph_ptr;
//	}
//	void Trainer::setLossFunction(int index, const LossFunction &loss, float weight)
//	{
//		m_losses.at(index) = loss.clone();
//		m_loss_weights.at(index) = weight;
//	}
//	void Trainer::setOptimizer(const RAdam &opt)
//	{
//		m_optimizer = opt;
//	}
//	void Trainer::setRegularizer(const RegularizerL2 &reg)
//	{
//		m_regularizer = reg;
//	}
//
//	const Tensor& Trainer::getTarget(int index) const
//	{
//		return m_targets.at(index);
//	}
//	const Tensor& Trainer::getMask(int index) const
//	{
//		return m_masks.at(index);
//	}
//	Tensor& Trainer::getTarget(int index)
//	{
//		if (m_targets.at(index).isEmpty())
//			m_targets.at(index) = zeros_like(graph().getOutput(index));
//		return m_targets.at(index);
//	}
//	Tensor& Trainer::getMask(int index)
//	{
//		if (m_masks.at(index).isEmpty())
//			m_masks.at(index) = ones_like(graph().getOutput(index));
//		return m_masks.at(index);
//	}
//
//	void Trainer::moveTo(Device newDevice)
//	{
//		for (size_t i = 0; i < m_targets.size(); i++)
//			m_targets[i].moveTo(newDevice);
//		for (size_t i = 0; i < m_masks.size(); i++)
//			m_masks[i].moveTo(newDevice);
//	}
//	void Trainer::train(int batchSize, GradientScaler &gradientScaler)
//	{
//		graph().forward(batchSize);
//
//		const float gradient_scale = gradientScaler.getScale();
//		for (size_t i = 0; i < m_targets.size(); i++)
//		{
//			const Shape tmp = change_dim<0>(getTarget(i).shape(), batchSize);
//			Tensor gradient = graph().getGradient(i).view(tmp);
//			Tensor output = graph().getOutput(i).view(tmp);
//			Tensor target = getTarget(i).view(tmp);
//			Tensor mask = getMask(i).view(tmp);
//			m_losses.at(i)->getGradient(graph().context(), gradient_scale / batchSize, gradient, output, target, mask);
//		}
//
//		std::this_thread::sleep_for(std::chrono::milliseconds(100));
//		graph().backward(batchSize);
//
//		std::vector<Tensor> params = graph().getParameters();
//		std::vector<Tensor> param_gradients = graph().getParameterGradients();
//		m_regularizer.apply(graph().context(), gradient_scale, params, param_gradients);
//
////		{
////			Json json = Json::array();
////			SerializedObject so;
////			for (size_t i = 0; i < param_gradients.size(); i++)
////				json[i] = param_gradients[i].serialize(so);
////			FileSaver fs("/home/maciek/alphagomoku/dump.bin");
////			fs.save(json, so);
////			exit(0);
////		}
//
////		FileLoader fl("/home/maciek/alphagomoku/dump.bin");
////
////		for (size_t i = 0; i < param_gradients.size() / 2; i++)
////		{
////			const Tensor dw(fl.getJson()[2 * i + 0], fl.getBinaryData());
////			const Tensor db(fl.getJson()[2 * i + 1], fl.getBinaryData());
////
////			Tensor pw = zeros_like(param_gradients[2 * i + 0]);
////			pw.copyFrom(graph().context(), param_gradients[2 * i + 0]);
////			pw.convertTo(graph().context(), DataType::FLOAT32);
////
////			Tensor pb = zeros_like(param_gradients[2 * i + 1]);
////			pb.copyFrom(graph().context(), param_gradients[2 * i + 1]);
////			pb.convertTo(graph().context(), DataType::FLOAT32);
////			std::cout << i << " " << ml::testing::diffForTest(pw, dw) << " " << ml::testing::diffForTest(pb, db) << "   "
////					<< graph().getNode(i).getLayer().name() << '\n';
////
//////			std::cout << i << " " << ml::testing::normForTest(param_gradients[2 * i + 0]) / gradient_scale << " "
//////					<< ml::testing::normForTest(param_gradients[2 * i + 1]) / gradient_scale << " " << graph().getNode(i).getLayer().name() << '\n';
////		}
//		switch (m_training_dtype)
//		{
//			case DataType::FLOAT16:
//			{
//				const float inv_gradient_scale = gradientScaler.getInvScale(graph().context(), params);
//				if (inv_gradient_scale != 0.0f)
//				{
//					m_optimizer.apply(graph().context(), m_fp32_weights_copy, param_gradients, inv_gradient_scale);
//					for (size_t i = 0; i < params.size(); i++)
//						convertType(graph().context(), params[i].data(), params[i].dtype(), m_fp32_weights_copy[i].data(), DataType::FLOAT32,
//								params[i].volume());
//				}
//				break;
//			}
//			case DataType::FLOAT32:
//			{
//				m_optimizer.apply(graph().context(), params, param_gradients, 1.0f);
//				break;
//			}
//			default:
//				throw NotImplemented(METHOD_NAME, "training in types other than fp32 or fp16 are not supported");
//		}
//	}
//	std::vector<float> Trainer::getLoss(int batchSize)
//	{
//		std::vector<float> result(m_targets.size());
//		for (size_t i = 0; i < m_targets.size(); i++)
//		{
//			const Shape tmp = change_dim<0>(getTarget(i).shape(), batchSize);
//			Tensor output = graph().getOutput(i).view(tmp);
//			Tensor target = getTarget(i).view(tmp);
//			Tensor mask = getMask(i).view(tmp);
//			result[i] = m_losses.at(i)->getLoss(graph().context(), output, target, mask) * m_loss_weights[i] / batchSize;
//		}
//		return result;
//	}
} /* namespace ml */

