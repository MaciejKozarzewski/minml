/*
 * test_moe.cpp
 *
 *  Created on: Feb 3, 2026
 *      Author: Maciej Kozarzewski
 */

#ifndef BACKEND_TEST_MOE_CPP_
#define BACKEND_TEST_MOE_CPP_

#include <minml/core/math.hpp>
#include <minml/core/Context.hpp>
#include <minml/core/Tensor.hpp>
#include <minml/core/Shape.hpp>
#include <minml/layers/Layer.hpp>
#include <minml/layers/Router.hpp>
#include <minml/layers/GatherTopK.hpp>
#include <minml/layers/ScatterTopK.hpp>
#include <minml/layers/MixtureOfExperts.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>

#include <cmath>
#include <gtest/gtest.h>

namespace
{
	using namespace ml;

	template<typename T>
	void baseline_gemm(char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, T alpha, T beta)
	{
		assert(A.device().isCPU());
		assert(B.device().isCPU());
		assert(C.device().isCPU());
		for (int m = 0; m < C.dim(0); m++)
			for (int n = 0; n < C.dim(1); n++)
			{
				T tmp = (T) 0;
				if (opA == 'n')
				{
					if (opB == 'n')
					{ // NN
						for (int k = 0; k < A.dim(1); k++)
							tmp += (T) A.at( { m, k }) * (T) B.at( { k, n });
					}
					else
					{ // NT
						for (int k = 0; k < A.dim(1); k++)
							tmp += (T) A.at( { m, k }) * (T) B.at( { n, k });
					}
				}
				else
				{
					if (opB == 'n')
					{ // TN
						for (int k = 0; k < A.dim(0); k++)
							tmp += (T) A.at( { k, m }) * (T) B.at( { k, n });
					}
					else
					{ // TT
						for (int k = 0; k < A.dim(0); k++)
							tmp += (T) A.at( { k, m }) * (T) B.at( { n, k });
					}
				}
				tmp = alpha * tmp;
				if (beta != (T) 0)
					tmp += beta * (T) C.at( { m, n });
				C.at( { m, n }) = tmp;
			}
	}

	template<typename T>
	void baseline_softmax_forward(Tensor &output, const Tensor &input)
	{
		assert(input.rank() == 2);
		assert(output.shape() == input.shape());
		const int first_dim = input.dim(0);
		const int last_dim = input.dim(1);

		for (int i = 0; i < first_dim; i++)
		{
			T max_value = std::numeric_limits<T>::lowest();
			for (int j = 0; j < last_dim; j++)
				max_value = std::max(max_value, (T) input.at( { i, j }));

			T sum = 0;
			for (int j = 0; j < last_dim; j++)
			{
				const T tmp = std::exp((T) input.at( { i, j }) - max_value);
				sum += tmp;
				output.at( { i, j }) = tmp;
			}

			for (int j = 0; j < last_dim; j++)
				output.at( { i, j }) = (T) output.at( { i, j }) / sum;
		}
	}
	template<typename T>
	void baseline_softmax_backward(Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &output)
	{
		assert(output.rank() == 2);
		assert(output.shape() == gradient_next.shape());
		assert(gradient_prev.shape() == gradient_next.shape());
		const int first_dim = output.dim(0);
		const int last_dim = output.dim(1);

		for (int i = 0; i < first_dim; i++)
		{
			T tmp = 0;
			for (int j = 0; j < last_dim; j++)
			{
				const T y = output.at( { i, j });
				const T dy = gradient_next.at( { i, j });
				tmp += dy * y;
			}
			for (int j = 0; j < last_dim; j++)
			{
				const T y = output.at( { i, j });
				const T dy = gradient_next.at( { i, j });
				const T dx = y * (dy - tmp);
				gradient_prev.at( { i, j }) = dx;
			}
		}
	}
	template<typename T>
	void scale_by_beta(T beta, Tensor &x)
	{
		if (beta == static_cast<T>(0))
		{
			x.zeroall();
			return;
		}
		else
		{
			Tensor tmp = x.view().flatten();
			for (int i = 0; i < tmp.volume(); i++)
				tmp.at( { i }) = (T) tmp.at( { i }) * beta;
		}
	}

	template<typename T>
	Tensor top_k_indices(const Tensor &t, int top_k)
	{
		assert(t.lastDim() >= top_k);
		Tensor result(Shape { t.dim(0), t.dim(1), top_k }, "int32", t.device());
		std::vector<std::pair<T, int>> storage(t.lastDim());

		for (int b = 0; b < t.dim(0); b++)
			for (int e = 0; e < t.dim(1); e++)
			{
				for (int i = 0; i < t.dim(2); i++)
					storage[i] = std::make_pair((T) t.at( { b, e, i }), i);
				// first sort by values to select top k
				std::sort(storage.begin(), storage.end(), [](const std::pair<T, int> &lhs, const std::pair<T, int> &rhs)
				{	return lhs.first > rhs.first;});
				// now sort those top k by indices
				std::sort(storage.begin(), storage.begin() + top_k, [](const std::pair<T, int> &lhs, const std::pair<T, int> &rhs)
				{	return lhs.second < rhs.second;});
				// now write to resulting tensor
				for (int i = 0; i < top_k; i++)
					result.at( { b, e, i }) = storage[i].second;
			}
		return result;
	}
	template<typename T>
	Tensor get_scales_forward(const Tensor &router_output, const Tensor &indices)
	{
		Tensor result(indices.shape(), router_output.dtype(), router_output.device());
		for (int b = 0; b < indices.dim(0); b++)
			for (int e = 0; e < indices.dim(1); e++)
				for (int k = 0; k < indices.dim(2); k++)
					result.at( { b, e, k }) = router_output.at( { b, e, (int) indices.at( { b, e, k }) });
		Tensor tmp = result.view().flatten( { 0, 1 });
		baseline_softmax_forward<T>(tmp, tmp);
		return result;
	}
	template<typename T>
	void get_scales_backward(float beta, Tensor &router_gradient, const Tensor &scales_gradient, const Tensor &indices, const Tensor &scales)
	{
		Tensor tmp1 = scales_gradient.view().flatten( { 0, 1 });
		Tensor tmp2 = scales.view().flatten( { 0, 1 });
		baseline_softmax_backward<T>(tmp1, tmp1, tmp2);

		scale_by_beta<T>(beta, router_gradient);
		Tensor tmp_prev = router_gradient.view().flatten( { 2, 3 });
		for (int b = 0; b < scales_gradient.dim(0); b++)
			for (int e = 0; e < scales_gradient.dim(1); e++)
				for (int k = 0; k < scales_gradient.dim(2); k++)
					tmp_prev.at( { b, e, (int) indices.at( { b, e, k }) }) = (T) tmp_prev.at( { b, e, (int) indices.at( { b, e, k }) })
							+ (T) scales_gradient.at( { b, e, k });
	}

	void gather_tokens_forward(Tensor &output, const Tensor &input, const Tensor &indices)
	{
		assert(input.rank() == 4);
		assert(indices.rank() == 3);

		const int batch_size = input.dim(0);
		const int channels = input.dim(3);
		assert(batch_size == indices.dim(0));
		const int experts = indices.dim(1);
		const int top_k = indices.dim(2);

		const Tensor flattened_input = input.view().flatten( { 1, 2 });
		assert(output.dim(0) == batch_size);
		assert(output.dim(1) == top_k);
		assert(output.dim(2) == experts);
		assert(output.dim(3) == channels);

		output.zeroall();
		for (int e = 0; e < experts; e++)
			for (int b = 0; b < batch_size; b++)
				for (int k = 0; k < top_k; k++)
				{
					const int token_index = (int) indices.at( { b, e, k });
					for (int c = 0; c < channels; c++)
						output.at( { b, k, e, c }) = flattened_input.at( { b, token_index, c });
				}
	}
	template<typename T>
	void gather_tokens_backward(float beta, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &indices)
	{
		assert(gradient_prev.rank() == 4);
		assert(indices.rank() == 3);

		const int batch_size = gradient_prev.dim(0);
		const int channels = gradient_prev.dim(3);
		assert(batch_size == indices.dim(0));
		const int experts = indices.dim(1);
		const int top_k = indices.dim(2);

		Tensor flattened_prev = gradient_prev.view().flatten( { 1, 2 });
		assert(gradient_next.dim(0) == batch_size);
		assert(gradient_next.dim(1) == top_k);
		assert(gradient_next.dim(2) == experts);
		assert(gradient_next.dim(3) == channels);

		scale_by_beta<T>(beta, gradient_prev);
		for (int e = 0; e < experts; e++)
			for (int b = 0; b < batch_size; b++)
				for (int k = 0; k < top_k; k++)
				{
					const int token_index = (int) indices.at( { b, e, k });
					for (int c = 0; c < channels; c++)
					{
						const T tmp = (T) gradient_next.at( { b, k, e, c }) + (T) flattened_prev.at( { b, token_index, c });
						flattened_prev.at( { b, token_index, c }) = tmp;
					}
				}
	}
	template<typename T>
	void scatter_tokens_forward(Tensor &output, const Tensor &input, const Tensor &indices, const Tensor &scales)
	{
		assert(output.rank() == 4);
		assert(input.rank() == 4);
		assert(indices.rank() == 3);

		const int batch_size = output.dim(0);
		const int channels = output.dim(3);
		assert(batch_size == indices.dim(0));
		const int experts = indices.dim(1);
		const int top_k = indices.dim(2);

		assert(batch_size == input.dim(0));
		assert(top_k == input.dim(1));
		assert(experts == input.dim(2));
		assert(channels == input.dim(3));

		Tensor flattened_output = output.view().flatten( { 1, 2 });
		output.zeroall();

		for (int e = 0; e < experts; e++)
			for (int b = 0; b < batch_size; b++)
				for (int k = 0; k < top_k; k++)
				{
					const int token_index = (int) indices.at( { b, e, k });
					const T scale = scales.isEmpty() ? static_cast<T>(1) : static_cast<T>(scales.at( { b, e, k }));
					for (int c = 0; c < channels; c++)
					{
						const T tmp = (T) flattened_output.at( { b, token_index, c }) + (T) input.at( { b, k, e, c }) * scale;
						flattened_output.at( { b, token_index, c }) = tmp;
					}
				}
	}
	template<typename T>
	void scatter_tokens_backward(float beta_prev, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &input, const Tensor &indices,
			const Tensor &scales, Tensor &scales_gradient)
	{
		assert(gradient_prev.rank() == 4);
		assert(gradient_next.rank() == 4);
		assert(indices.rank() == 3);

		const int batch_size = gradient_next.dim(0);
		const int channels = gradient_next.dim(3);
		assert(batch_size == indices.dim(0));
		const int experts = indices.dim(1);
		const int top_k = indices.dim(2);
		assert(batch_size == gradient_prev.dim(0));
		assert(top_k == gradient_prev.dim(1));
		assert(experts == gradient_prev.dim(2));
		assert(channels == gradient_prev.dim(3));

		Tensor flattened_next = gradient_next.view().flatten( { 1, 2 });

		scale_by_beta<T>(beta_prev, gradient_prev);

		for (int e = 0; e < experts; e++)
			for (int b = 0; b < batch_size; b++)
				for (int k = 0; k < top_k; k++)
				{
					const int token_index = (int) indices.at( { b, e, k });
					const T scale = (T) scales.at( { b, e, k });
					T scale_gradient = static_cast<T>(0);
					for (int c = 0; c < channels; c++)
					{
						scale_gradient += (T) input.at( { b, k, e, c }) * (T) flattened_next.at( { b, token_index, c });
						gradient_prev.at( { b, k, e, c }) = (T) gradient_prev.at( { b, k, e, c })
								+ (T) flattened_next.at( { b, token_index, c }) * scale;
					}
					scales_gradient.at( { b, e, k }) = scale_gradient;
				}
	}

	Tensor transpose_experts_last(const Tensor &t)
	{
		assert(t.rank() == 4);
		const int experts = t.dim(0);
		const int batch_size = t.dim(1);
		const int top_k = t.dim(2);
		const int channels = t.dim(3);

		Tensor result( { batch_size, top_k, experts, channels }, t.dtype(), t.device());
		for (int e = 0; e < experts; e++)
			for (int b = 0; b < batch_size; b++)
				for (int k = 0; k < top_k; k++)
					for (int c = 0; c < channels; c++)
						result.at( { b, k, e, c }) = t.at( { e, b, k, c });
		return result;
	}
	Tensor transpose_experts_first(const Tensor &t)
	{
		assert(t.rank() == 4);
		const int batch_size = t.dim(0);
		const int top_k = t.dim(1);
		const int experts = t.dim(2);
		const int channels = t.dim(3);

		Tensor result( { experts, batch_size, top_k, channels }, t.dtype(), t.device());
		for (int b = 0; b < batch_size; b++)
			for (int k = 0; k < top_k; k++)
				for (int e = 0; e < experts; e++)
					for (int c = 0; c < channels; c++)
						result.at( { e, b, k, c }) = t.at( { b, k, e, c });
		return result;
	}

	/*
	 * input:	[NHWC]
	 * weights:	[EC]	E - number of experts
	 * output:	[NEHW]
	 */
	class BaselineRouter: public Layer
	{
			int m_experts = 0;
		public:
			BaselineRouter(int experts) :
					Layer(),
					m_experts(experts)
			{
			}
			Shape getWeightShape() const
			{
				return Shape( { m_experts, getInputShape().lastDim() });
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().dim(0);
				const int height = getInputShape().dim(1);
				const int width = getInputShape().dim(2);
				return Shape( { batch_size, m_experts, height, width });
			}
			std::string name() const
			{
				return "BaselineRouter";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["experts"] = m_experts;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineRouter> result = std::make_unique<BaselineRouter>(config["experts"].getInt());
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				assert(input.size() == 1);
				const int batch_size = input[0].dim(0);
				const int tokens = input[0].dim(1) * input[0].dim(2);
				const int channels = input[0].dim(3);

				for (int b = 0; b < batch_size; b++)
				{
					const Tensor x = input[0].view( { tokens, channels }, b * tokens * channels);
					Tensor y = output.view( { m_experts, tokens }, b * m_experts * tokens);
					if (dtype() == DataType::FLOAT32)
						baseline_gemm<float>('n', 't', y, getWeights().getParam(), x, 1.0f, 0.0f);
					else
						baseline_gemm<double>('n', 't', y, getWeights().getParam(), x, 1.0f, 0.0f);
				}
//				Tensor y = output.view().flatten( { 0, 1 }, { 2, 3 });
//				if (dtype() == DataType::FLOAT32)
//					baseline_softmax_forward<float>(y, y);
//				else
//					baseline_softmax_forward<double>(y, y);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				assert(input.size() == 1);
				assert(gradient_prev.size() == 1);

//				{ /* artificial scope for variables */
//					Tensor dy = gradient_next.view().flatten( { 0, 1 }, { 2, 3 });
//					const Tensor y = output.view().flatten( { 0, 1 }, { 2, 3 });
//					if (dtype() == DataType::FLOAT32)
//						baseline_softmax_backward<float>(dy, dy, y);
//					else
//						baseline_softmax_backward<double>(dy, dy, y);
//				}

				const int batch_size = input[0].dim(0);
				const int tokens = input[0].dim(1) * input[0].dim(2);
				const int channels = input[0].dim(3);
				getWeights().getGradient().zeroall();
				for (int b = 0; b < batch_size; b++)
				{
					const Tensor x = input[0].view( { tokens, channels }, b * tokens * channels);
					Tensor dx = gradient_prev[0].view( { tokens, channels }, b * tokens * channels);
					Tensor dy = gradient_next.view( { m_experts, tokens }, b * m_experts * tokens);

					if (dtype() == DataType::FLOAT32)
					{
						baseline_gemm<float>('t', 'n', dx, dy, getWeights().getParam(), 1.0f, beta[0]);
						baseline_gemm<float>('n', 'n', getWeights().getGradient(), dy, x, 1.0f, 1.0f);
					}
					else
					{
						baseline_gemm<double>('t', 'n', dx, dy, getWeights().getParam(), 1.0f, beta[0]);
						baseline_gemm<double>('n', 'n', getWeights().getGradient(), dy, x, 1.0f, 1.0f);
					}
				}
			}
	};
	/*
	 * 					router
	 * inputs:	[NHWC]	[NEHW]
	 * weights:
	 * output:	[ENKC]	K - selected top K out of HW tokens
	 */
	class BaselineGatherTopK: public Layer
	{
			int m_top_k = 0;
		public:
			BaselineGatherTopK(int top_k) :
					Layer(),
					m_top_k(top_k)
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				assert(shapes.size() == 2);
				assert(shapes[0].rank() == 4);
				assert(shapes[1].rank() == 4);
				assert(shapes[0].dim(0) == shapes[1].dim(0)); // batch size match
				assert(shapes[0].dim(1) == shapes[1].dim(2)); // height match
				assert(shapes[0].dim(2) == shapes[1].dim(3)); // width match
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape(0).dim(0);
				const int height = getInputShape(0).dim(1);
				const int width = getInputShape(0).dim(2);
				const int channels = getInputShape(0).dim(3);
				assert(getInputShape(1).dim(0) == batch_size);
				const int experts = getInputShape(1).dim(1);
				assert(getInputShape(1).dim(2) == height);
				assert(getInputShape(1).dim(3) == width);
				return Shape( { batch_size, m_top_k, experts, channels });
			}
			std::string name() const
			{
				return "BaselineGatherTopK";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["m_top_k"] = m_top_k;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineGatherTopK> result = std::make_unique<BaselineGatherTopK>(config["m_top_k"].getInt());
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				assert(input.size() == 2);

				const Tensor flattened_router_output = input[1].view().flatten( { 2, 3 });
				if (dtype() == DataType::FLOAT32)
				{
					const Tensor indices = top_k_indices<float>(flattened_router_output, m_top_k);
					gather_tokens_forward(output, input[0], indices);
				}
				else
				{
					const Tensor indices = top_k_indices<double>(flattened_router_output, m_top_k);
					gather_tokens_forward(output, input[0], indices);
				}
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				assert(input.size() == 2);
				assert(gradient_prev.size() == 2);

				const Tensor flattened_router_output = input[1].view().flatten( { 2, 3 });
				if (dtype() == DataType::FLOAT32)
				{
					const Tensor indices = top_k_indices<float>(flattened_router_output, m_top_k);
					gather_tokens_backward<float>(beta[0], gradient_prev[0], gradient_next, indices);
				}
				else
				{
					const Tensor indices = top_k_indices<double>(flattened_router_output, m_top_k);
					gather_tokens_backward<double>(beta[0], gradient_prev[0], gradient_next, indices);
				}
			}
	};
	/*
	 * 			MoE		router
	 * inputs:	[ENKC]	[NEHW]	K - selected top K out of HW tokens
	 * weights:
	 * output:	[NHWC]
	 */
	class BaselineScatter: public Layer
	{
		public:
			BaselineScatter() :
					Layer()
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				assert(shapes.size() == 2);
				assert(shapes[0].rank() == 4);
				assert(shapes[1].rank() == 4);
				assert(shapes[0].dim(0) == shapes[1].dim(0)); // experts match
				assert(shapes[0].dim(2) == shapes[1].dim(1)); // batch size match
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape(0).dim(0);
				const int channels = getInputShape(0).dim(3);
				const int height = getInputShape(1).dim(2);
				const int width = getInputShape(1).dim(3);
				return Shape( { batch_size, height, width, channels });
			}
			std::string name() const
			{
				return "BaselineScatter";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineScatter> result = std::make_unique<BaselineScatter>();
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				assert(input.size() == 2);
				const int top_k = input[0].dim(1);

				const Tensor flattened_router_output = input[1].view().flatten( { 2, 3 });
				if (dtype() == DataType::FLOAT32)
				{
					const Tensor indices = top_k_indices<float>(flattened_router_output, top_k);
					Tensor scales = get_scales_forward<float>(flattened_router_output, indices);
					scatter_tokens_forward<float>(output, input[0], indices, scales);
				}
				else
				{
					const Tensor indices = top_k_indices<double>(flattened_router_output, top_k);
					Tensor scales = get_scales_forward<double>(flattened_router_output, indices);
					scatter_tokens_forward<double>(output, input[0], indices, scales);
				}
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				assert(input.size() == 2);
				assert(gradient_prev.size() == 2);

				const int top_k = input[0].dim(1);

				const Tensor flattened_router_output = input[1].view().flatten( { 2, 3 });
				if (dtype() == DataType::FLOAT32)
				{
					const Tensor indices = top_k_indices<float>(flattened_router_output, top_k);
					Tensor scales = get_scales_forward<float>(flattened_router_output, indices);
					Tensor scales_gradient = zeros_like(scales);
					scatter_tokens_backward<float>(beta[0], gradient_prev[0], gradient_next, input[0], indices, scales, scales_gradient);

					get_scales_backward<float>(beta[1], gradient_prev[1], scales_gradient, indices, scales);
				}
				else
				{
					const Tensor indices = top_k_indices<double>(flattened_router_output, top_k);
					Tensor scales = get_scales_forward<double>(flattened_router_output, indices);
					Tensor scales_gradient = zeros_like(scales);
					scatter_tokens_backward<double>(beta[0], gradient_prev[0], gradient_next, input[0], indices, scales, scales_gradient);

					get_scales_backward<double>(beta[1], gradient_prev[1], scales_gradient, indices, scales);
				}
			}
	};
	/*
	 * 			gather
	 * inputs:	[ENKC]	K - selected top K out of HW tokens
	 * weights:	[EOC]	O - number of output neurons
	 * output:	[ENKC]
	 */
	class BaselineMoE: public Layer
	{
			int m_experts = 0;
			int m_neurons = 0;
		public:
			BaselineMoE(int experts, int neurons, const std::string &activation) :
					Layer(activation),
					m_experts(experts),
					m_neurons(neurons)
			{
			}
			Shape getWeightShape() const
			{
				return Shape( { m_experts, m_neurons, getInputShape().lastDim() });
			}
			Shape getBiasShape() const
			{
				return Shape( { m_experts, m_neurons });
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				assert(shapes.size() == 1);
				assert(shapes[0].dim(0) == m_experts);
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().dim(0);
				const int top_k = getInputShape().dim(1);
				assert(getInputShape().dim(2) == m_experts);
				return Shape( { batch_size, top_k, m_experts, m_neurons });
			}
			std::string name() const
			{
				return "BaselineMoE";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["experts"] = m_experts;
				result["neurons"] = m_neurons;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineMoE> result = std::make_unique<BaselineMoE>(config["experts"].getInt(), config["neurons"].getInt(),
						config["nonlinearity"]);
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				assert(input.size() == 1);
				const int batch_size = input[0].dim(0);
				const int top_k = input[0].dim(1);
				const int experts = input[0].dim(2);
				const int in_channels = input[0].dim(3);
				const int out_channels = output.dim(3);

				const Tensor tmp_input = transpose_experts_first(input[0]);
				const Tensor tmp_output = transpose_experts_first(output);

				for (int e = 0; e < experts; e++)
				{
					const Tensor x = tmp_input.view( { batch_size * top_k, in_channels }, { e, 0, 0, 0 });
					const Tensor w = getWeights().getParam().view( { out_channels, in_channels }, { e, 0, 0 });
					Tensor y = tmp_output.view( { batch_size * top_k, out_channels }, { e, 0, 0, 0 });
					if (dtype() == DataType::FLOAT32)
						baseline_gemm<float>('n', 't', y, x, w, 1.0f, 0.0f);
					else
						baseline_gemm<double>('n', 't', y, x, w, 1.0f, 0.0f);
					const Tensor b = getBias().getParam().view( { out_channels }, { e, 0 });
					addBiasAct(context(), 1.0f, y, b, 0.0f, y, m_activation);
				}
				output = transpose_experts_last(tmp_output);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				assert(input.size() == 1);
				assert(gradient_prev.size() == 1);
				const int batch_size = input[0].dim(0);
				const int top_k = input[0].dim(1);
				const int experts = input[0].dim(2);
				const int in_channels = input[0].dim(3);
				const int out_channels = output.dim(3);

				activationBackward(context(), 1.0f, gradient_next, output, 0.0f, gradient_next, m_activation);

				const Tensor tmp_next = transpose_experts_first(gradient_next);
				const Tensor tmp_input = transpose_experts_first(input[0]);
				Tensor tmp_prev = transpose_experts_first(gradient_prev[0]);
				for (int e = 0; e < experts; e++)
				{
					Tensor db = getBias().getGradient().view( { out_channels }, { e, 0 });
					const Tensor dy = tmp_next.view( { batch_size * top_k, out_channels }, { e, 0, 0, 0 });
					sumOverFirstDim(context(), 1.0f, dy, 0.0f, db);

					const Tensor x = tmp_input.view( { batch_size * top_k, in_channels }, { e, 0, 0, 0 });
					Tensor dx = tmp_prev.view( { batch_size * top_k, in_channels }, { e, 0, 0, 0 });
					const Tensor w = getWeights().getParam().view( { out_channels, in_channels }, { e, 0, 0 });
					Tensor dw = getWeights().getGradient().view( { out_channels, in_channels }, { e, 0, 0 });

					if (dtype() == DataType::FLOAT32)
					{
						baseline_gemm<float>('n', 'n', dx, dy, w, 1.0f, beta[0]);
						baseline_gemm<float>('t', 'n', dw, dy, x, 1.0f, 0.0f);
					}
					else
					{
						baseline_gemm<double>('n', 'n', dx, dy, w, 1.0f, beta[0]);
						baseline_gemm<double>('t', 'n', dw, dy, x, 1.0f, 0.0f);
					}
				}
				gradient_prev[0] = transpose_experts_last(tmp_prev);
			}
	};
}

namespace ml
{
//	TEST(TestRouter, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineRouter(5) };
//		gradcheck.setInputShape(Shape( { 3, 8, 10, 37 }));
//
//		gradcheck.check(100, 1.0e-3, "all", true);
//
//		exit(0);
//	}
//	TEST(TestGatherTopK, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineGatherTopK(8) };
//		const Shape input( { 3, 11, 12, 37 });
//		const Shape router_output( { 3, 5, 11, 12 });
//		gradcheck.setInputShape( { input, router_output });
//
//		gradcheck.check(100, 1.0e-3, "all", true);
//
//		exit(0);
//	}
//	TEST(TestScatterTopK, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineScatter() };
//		const Shape moe_output( { 3, 8, 5, 37 });
//		const Shape router_output( { 3, 5, 11, 12 });
//		gradcheck.setInputShape( { moe_output, router_output });
//
//		gradcheck.check(100, 1.0e-4, "input", true);
//
//		exit(0);
//	}
//	TEST(TestMoE, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineMoE(11, 100, "tanh") };
//		const Shape input( { 3, 8, 11, 37 });
//		gradcheck.setInputShape(input);
//
//		gradcheck.check(100, 1.0e-3, "all", true);
//
//		exit(0);
//	}

	TEST(TestRouter, forward_and_backward)
	{
		const int batch_size = 3;
		const int height = 13;
		const int width = 14;
		const int channels = 56;
		const int experts = 7;
		const bool verbose = false;

		const Shape input_shape( { batch_size, height, width, channels });

		testing::LayerCheck baseline { BaselineRouter(experts) };
		testing::LayerCheck under_test { Router(experts) };

		baseline.setInputShape(input_shape);
		under_test.setInputShape(input_shape);

		baseline.setup(Device::cpu(), DataType::FLOAT32);
		under_test.setup(Device::cuda(), DataType::FLOAT32);

		baseline.init();
		under_test.initFrom(baseline);

		baseline.forward(verbose);
		under_test.forward(verbose);

		EXPECT_LE(output_diff(baseline, under_test), 1.0e-4f);

		baseline.backward(verbose);
		under_test.backward(verbose);

		EXPECT_LE(gradient_prev_diff(baseline, under_test, 0), 1.0e-4f);
		EXPECT_LE(weight_gradient_diff(baseline, under_test), 1.0e-4f);
		EXPECT_LE(bias_gradient_diff(baseline, under_test), 1.0e-4f);
	}

	TEST(TestGatherTopK, forward_and_backward)
	{
		const int batch_size = 5;
		const int height = 11;
		const int width = 12;
		const int channels = 56;
		const int experts = 17;
		const int top_k = 27;
		const bool verbose = false;

		const Shape input_shape( { batch_size, height, width, channels });
		const Shape router_output_shape( { batch_size, experts, height, width });

		testing::LayerCheck baseline { BaselineGatherTopK(top_k) };
		testing::LayerCheck under_test { GatherTopK(top_k) };

		baseline.setInputShape( { input_shape, router_output_shape });
		under_test.setInputShape( { input_shape, router_output_shape });

		baseline.setup(Device::cpu(), DataType::FLOAT32);
		under_test.setup(Device::cuda(), DataType::FLOAT32);

		baseline.init();
		under_test.initFrom(baseline);

		baseline.forward(verbose);
		under_test.forward(verbose);

		EXPECT_LE(output_diff(baseline, under_test), 1.0e-4f);

		baseline.backward(verbose);
		under_test.backward(verbose);

		EXPECT_LE(gradient_prev_diff(baseline, under_test, 0), 1.0e-4f);
		EXPECT_LE(gradient_prev_diff(baseline, under_test, 1), 1.0e-4f);
		EXPECT_LE(weight_gradient_diff(baseline, under_test), 1.0e-4f);
		EXPECT_LE(bias_gradient_diff(baseline, under_test), 1.0e-4f);
	}

	TEST(TestScatterTopK, forward_and_backward)
	{
		const int batch_size = 5;
		const int height = 11;
		const int width = 12;
		const int channels = 56;
		const int experts = 17;
		const int top_k = 27;
		const bool verbose = true;

		const Shape input_shape( { batch_size, top_k, experts, channels });
		const Shape router_output_shape( { batch_size, experts, height, width });

		testing::LayerCheck baseline { BaselineScatter() };
		testing::LayerCheck under_test { ScatterTopK() };

		baseline.setInputShape( { input_shape, router_output_shape });
		under_test.setInputShape( { input_shape, router_output_shape });

		baseline.setup(Device::cpu(), DataType::FLOAT32);
		under_test.setup(Device::cuda(), DataType::FLOAT32);

		baseline.init();
		under_test.initFrom(baseline);

		baseline.forward(verbose);
		under_test.forward(verbose);

		EXPECT_LE(output_diff(baseline, under_test), 1.0e-4f);

		baseline.backward(verbose);
		under_test.backward(verbose);

		EXPECT_LE(gradient_prev_diff(baseline, under_test, 0), 1.0e-4f);
		EXPECT_LE(gradient_prev_diff(baseline, under_test, 1), 1.0e-4f);
		EXPECT_LE(weight_gradient_diff(baseline, under_test), 1.0e-4f);
		EXPECT_LE(bias_gradient_diff(baseline, under_test), 1.0e-4f);
	}

	TEST(TestMixtureOfExperts, forward_and_backward)
	{
		const int batch_size = 5;
		const int top_k = 17;
		const int channels = 56;
		const int neurons = 80;
		const int experts = 11;
		const bool verbose = false;

		const Shape input_shape( { batch_size, top_k, experts, channels });

		testing::LayerCheck baseline { BaselineMoE(experts, neurons, "relu") };
		testing::LayerCheck under_test { MixtureOfExperts(experts, neurons, "relu") };

		baseline.setInputShape(input_shape);
		under_test.setInputShape(input_shape);

		baseline.setup(Device::cpu(), DataType::FLOAT32);
		under_test.setup(Device::cuda(), DataType::FLOAT32);

		baseline.init();
		under_test.initFrom(baseline);

		baseline.forward(verbose);
		under_test.forward(verbose);

		EXPECT_LE(output_diff(baseline, under_test), 1.0e-4f);

		baseline.backward();
		under_test.backward();

		EXPECT_LE(gradient_prev_diff(baseline, under_test, 0), 1.0e-4f);
		EXPECT_LE(weight_gradient_diff(baseline, under_test), 1.0e-4f);
		EXPECT_LE(bias_gradient_diff(baseline, under_test), 1.0e-4f);
	}

} /* namespace ml */

#endif /* BACKEND_TEST_MOE_CPP_ */
