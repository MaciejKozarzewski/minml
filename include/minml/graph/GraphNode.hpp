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
#include <minml/layers/quantization.hpp>

#include <memory>
#include <vector>
#include <chrono>

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
	class TimedStat
	{
		private:
			using time_point = std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds>;

			std::string m_name;
			time_point m_timer_start;
			time_point m_timer_stop;
			int64_t m_total_time = 0;
			int64_t m_total_count = 0;
		public:
			TimedStat() = default;
			TimedStat(const std::string &name) :
					m_name(name)
			{
			}
			void start() noexcept
			{
				m_timer_start = std::chrono::steady_clock::now();
				m_timer_stop = m_timer_start;
			}
			void stop() noexcept
			{
				m_timer_stop = std::chrono::steady_clock::now();
				m_total_time += std::chrono::duration<int64_t, std::nano>(m_timer_stop - m_timer_start).count();
				m_total_count++;
			}
			std::string toString() const;
	};

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
			AffineTransform m_output_transform;
			TimedStat m_timer;
		public:
			GraphNode(std::unique_ptr<Layer> &layer, const std::vector<GraphNode*> input_nodes);
			~GraphNode();

			bool isInputNode() const;
			bool isOutputNode() const;
			Shape getOutputShape() const;
			void resolveInputShapes();
			int getBackupStorage();
			AffineTransform getOutputTransform() const noexcept;

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
			void makeTrainable(bool b);
			void setOutputTransform(const AffineTransform &t) noexcept;

			void replaceLayer(Layer *layer);

			static void link(GraphNode *prev, GraphNode *next);
			static void link(const std::vector<GraphNode*> &prev, GraphNode *next);
			static void link(GraphNode *prev, const std::vector<GraphNode*> &next);
			static void link(const std::vector<GraphNode*> &prev, const std::vector<GraphNode*> &next);

			static void removeLink(GraphNode *prev, GraphNode *next);
			static void replaceInputLink(GraphNode *old_prev, GraphNode *new_prev, GraphNode *next);
			static void replaceOutputLink(GraphNode *prev, GraphNode *old_next, GraphNode *new_next);
			void removeAllLinks();
		private:
			Tensor change_batch(int batch_size, const Tensor &other);
	};

} /* namespace ml */

#endif /* MINML_GRAPH_GRAPHNODE_HPP_ */
