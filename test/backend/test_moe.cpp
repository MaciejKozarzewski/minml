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
#include <minml/layers/MixtureOfExperts.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/testing_util.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <minml/layers/GatherTokens.hpp>
#include <minml/layers/ScatterTokens.hpp>

namespace
{
	using namespace ml;

	struct ExpertTokenValue
	{
			int expert = -1;
			int token = 0;
			float value = 0.0f;

			ExpertTokenValue() noexcept = default;
			ExpertTokenValue(int e, int t, float v) noexcept :
					expert(e),
					token(t),
					value(v)
			{
			}
			friend bool operator<(const ExpertTokenValue &lhs, const ExpertTokenValue &rhs) noexcept
			{
				return (lhs.expert == rhs.expert) ? (lhs.value < rhs.value) : (lhs.expert < rhs.expert);
			}
	};

	template<typename T>
	DataType type_of()
	{
		if (std::is_same<T, float>::value)
			return DataType::FLOAT32;
		if (std::is_same<T, double>::value)
			return DataType::FLOAT64;
		if (std::is_same<T, int>::value)
			return DataType::INT32;
		return DataType::UNKNOWN;
	}

	template<typename T>
	void baseline_gemm(char opA, char opB, Tensor &C, const Tensor &A, const Tensor &B, T alpha, T beta)
	{
		assert(A.device().isCPU() && A.dtype() == type_of<T>());
		assert(B.device().isCPU() && B.dtype() == type_of<T>());
		assert(C.device().isCPU() && C.dtype() == type_of<T>());
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
		assert(input.dtype() == type_of<T>());
		assert(output.dtype() == type_of<T>());
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
		assert(gradient_prev.dtype() == type_of<T>());
		assert(gradient_next.dtype() == type_of<T>());
		assert(output.dtype() == type_of<T>());
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
			assert(x.dtype() == type_of<T>());
			Tensor tmp = x.view().flatten();
			for (int i = 0; i < tmp.volume(); i++)
				tmp.at( { i }) = (T) tmp.at( { i }) * beta;
		}
	}

	template<typename T>
	Tensor top_k_indices(const Tensor &tensor, int top_k)
	{
		assert(tensor.dtype() == type_of<T>());
		const int batch_size = tensor.dim(0);
		const int tokens = tensor.dim(1);
		assert(tokens >= top_k);
		const int experts = tensor.dim(2);

		Tensor result(Shape { batch_size, 2, experts, top_k }, tensor.dtype(), tensor.device());
		std::vector<std::pair<T, int>> storage(tokens);

		for (int b = 0; b < batch_size; b++)
			for (int e = 0; e < experts; e++)
			{
				for (int t = 0; t < tokens; t++)
					storage[t] = std::make_pair((T) tensor.at( { b, t, e }), t);
				// first sort by values to select top k
				std::sort(storage.begin(), storage.end(), [](const std::pair<T, int> &lhs, const std::pair<T, int> &rhs)
				{	return lhs.first > rhs.first;});
				// now sort those top k by indices
				std::sort(storage.begin(), storage.begin() + top_k, [](const std::pair<T, int> &lhs, const std::pair<T, int> &rhs)
				{	return lhs.second < rhs.second;});
				// now write to resulting tensor
				for (int k = 0; k < top_k; k++)
				{
					result.at( { b, 0, e, k }) = static_cast<T>(storage[k].second); // index
					result.at( { b, 1, e, k }) = storage[k].first; // value
				}
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

	void gather_tokens_forward(Tensor &output, const Tensor &input, const Tensor &indices_and_values)
	{
		assert(input.rank() == 4);
		assert(indices_and_values.rank() == 4);

		const int batch_size = input.dim(0);
		const int channels = input.dim(3);
		assert(batch_size == indices_and_values.dim(0));
		assert(indices_and_values.dim(1) == 2);
		const int experts = indices_and_values.dim(2);
		const int top_k = indices_and_values.dim(3);

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
					const int token_index = (int) indices_and_values.at( { b, 0, e, k });
					if (token_index >= 0)
						for (int c = 0; c < channels; c++)
							output.at( { b, k, e, c }) = flattened_input.at( { b, token_index, c });
				}
	}
	template<typename T>
	void gather_tokens_backward(float beta, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &indices_and_values)
	{
		assert(gradient_prev.rank() == 4);
		assert(indices_and_values.rank() == 4);

		const int batch_size = gradient_prev.dim(0);
		const int channels = gradient_prev.dim(3);
		assert(batch_size == indices_and_values.dim(0));
		assert(indices_and_values.dim(1) == 2);
		const int experts = indices_and_values.dim(2);
		const int top_k = indices_and_values.dim(3);

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
					const int token_index = (int) indices_and_values.at( { b, 0, e, k });
					if (token_index >= 0)
						for (int c = 0; c < channels; c++)
						{
							const T tmp = (T) gradient_next.at( { b, k, e, c }) + (T) flattened_prev.at( { b, token_index, c });
							flattened_prev.at( { b, token_index, c }) = tmp;
						}
				}
	}
	template<typename T>
	void scatter_tokens_forward(Tensor &output, const Tensor &input, const Tensor &indices_and_values)
	{
		assert(output.rank() == 4);
		assert(input.rank() == 4);
		assert(indices_and_values.rank() == 4);

		const int batch_size = output.dim(0);
		const int channels = output.dim(3);
		assert(batch_size == indices_and_values.dim(0));
		assert(indices_and_values.dim(1) == 2);
		const int experts = indices_and_values.dim(2);
		const int top_k = indices_and_values.dim(3);

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
					const int token_index = (int) indices_and_values.at( { b, 0, e, k });
					const T scale = static_cast<T>(indices_and_values.at( { b, 1, e, k }));
					if (token_index >= 0)
						for (int c = 0; c < channels; c++)
						{
							const T tmp = (T) flattened_output.at( { b, token_index, c }) + (T) input.at( { b, k, e, c }) * scale;
							flattened_output.at( { b, token_index, c }) = tmp;
						}
				}
	}
	template<typename T>
	void scatter_tokens_backward(float beta_prev, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &input,
			const Tensor &indices_and_values, float beta_scales, Tensor &scales_gradient)
	{
		assert(gradient_prev.rank() == 4);
		assert(gradient_next.rank() == 4);
		assert(indices_and_values.rank() == 4);

		const int batch_size = gradient_next.dim(0);
		const int channels = gradient_next.dim(3);
		assert(batch_size == indices_and_values.dim(0));
		assert(indices_and_values.dim(1) == 2);
		const int experts = indices_and_values.dim(2);
		const int top_k = indices_and_values.dim(3);
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
					const int token_index = (int) indices_and_values.at( { b, 0, e, k });
					const T scale = (T) indices_and_values.at( { b, 1, e, k });
					T scale_gradient = static_cast<T>(0);
					if (token_index >= 0)
						for (int c = 0; c < channels; c++)
						{
							scale_gradient += (T) input.at( { b, k, e, c }) * (T) flattened_next.at( { b, token_index, c });
							gradient_prev.at( { b, k, e, c }) = (T) gradient_prev.at( { b, k, e, c })
									+ (T) flattened_next.at( { b, token_index, c }) * scale;
						}
					scales_gradient.at( { b, 0, e, k }) = static_cast<T>(0.0f);
					if (beta_scales != 0.0f)
						scale_gradient += (T) scales_gradient.at( { b, 1, e, k }) * beta_scales;
					scales_gradient.at( { b, 1, e, k }) = scale_gradient;
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
	 * input:	[NHWE]
	 * output:	[N2EK]	indices along with values
	 */
	class BaselineRouter: public Layer
	{
			std::string m_algorithm;
			float m_capacity_factor = 1.0f;
		public:
			BaselineRouter(const std::string &algo, float capacityFactor) :
					Layer(),
					m_algorithm(algo),
					m_capacity_factor(capacityFactor)
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape().dim(0);
				const int tokens = getInputShape().dim(1) * getInputShape().dim(2);
				const int experts = getInputShape().dim(3);
				return Shape( { batch_size, 2, experts, static_cast<int>(m_capacity_factor * tokens / experts + 0.5f) });
			}
			std::string name() const
			{
				return "BaselineRouter";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["algorithm"] = m_algorithm;
				result["capacity_factor"] = m_capacity_factor;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineRouter> result = std::make_unique<BaselineRouter>(config["algorithm"].getString(),
						config["capacity_factor"].getDouble());
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				assert(input.size() == 1);
				const int batch_size = input[0].dim(0);
				const int tokens = input[0].dim(1) * input[0].dim(2);
				const int experts = input[0].dim(3);
				const int capacity = m_capacity_factor * tokens / experts + 0.5f;

				if (m_algorithm == "hash")
				{
					for (int b = 0; b < batch_size; b++)
						for (int e = 0; e < experts; e++)
							for (int k = 0; k < capacity; k++)
							{
								const int token_index = k * experts + e;
								if (token_index < tokens)
								{
									output.at( { b, 0, e, k }) = token_index;
									output.at( { b, 1, e, k }) = 1.0f;
								}
								else
								{
									output.at( { b, 0, e, k }) = -1;
									output.at( { b, 1, e, k }) = 0.0f;
								}
							}
				}
				else
				{
					const Tensor logits = input[0].view().flatten( { 0, 1, 2 });
					Tensor values = zeros_like(logits);

					if (m_algorithm == "token_choice")
					{
						if (dtype() == DataType::FLOAT32)
							baseline_softmax_forward<float>(values, logits);
						else
							baseline_softmax_forward<double>(values, logits);

						values.reshape( { batch_size, tokens, experts });

						for (int b = 0; b < batch_size; b++)
						{
							std::vector<std::tuple<int, int, double>> workspace;
							for (int t = 0; t < tokens; t++)
							{
								int expert_idx = -1;
								double max_value = std::numeric_limits<double>::lowest();
								for (int e = 0; e < experts; e++)
									if ((double) values.at( { b, t, e }) > max_value)
									{
										max_value = (double) values.at( { b, t, e });
										expert_idx = e;
									}
								workspace.emplace_back(expert_idx, t, (double) values.at( { b, t, expert_idx }));
							}

							std::sort(workspace.begin(), workspace.end(),
									[](const std::tuple<int, int, double> &lhs,
											const std::tuple<int, int, double> &rhs)
											{	return (std::get<0>(lhs) == std::get<0>(rhs)) ? (std::get<2>(lhs) > std::get<2>(rhs)) : (std::get<0>(lhs) < std::get<0>(rhs));});

							for (int t = 0; t < tokens; t++)
								for (int e = 0; e < experts; e++)
									for (int k = 0; k < capacity; k++)
									{
										output.at( { b, 0, e, k }) = -1;
										output.at( { b, 1, e, k }) = 0.0f;
									}

							std::vector<int> counter(experts, 0);
							for (int t = 0; t < tokens; t++)
							{
								const int expert_idx = std::get<0>(workspace[t]);
								const int token_idx = std::get<1>(workspace[t]);
								const double value = std::get<2>(workspace[t]);
								if (counter[expert_idx] < capacity)
								{
									output.at( { b, 0, expert_idx, counter[expert_idx] }) = token_idx;
									output.at( { b, 1, expert_idx, counter[expert_idx] }) = value;
									counter[expert_idx]++;
								}
							}

//							for (size_t i = 0; i < workspace.size(); i++)
//								std::cout << std::get<0>(workspace[i]) << " " << std::get<2>(workspace[i]) << " " << std::get<1>(workspace[i])
//										<< '\n';
//							std::cout << '\n';
//							exit(0);
						}

						if (m_algorithm == "expert_choice")
						{
							if (dtype() == DataType::FLOAT32)
								output.copyFrom(context(), top_k_indices<float>(logits, capacity));
							else
								output.copyFrom(context(), top_k_indices<double>(logits, capacity));

							for (int b = 0; b < batch_size; b++)
							{
								Tensor values = output.view( { experts, capacity }, { b, 1, 0, 0 });
								if (dtype() == DataType::FLOAT32)
									baseline_softmax_forward<float>(values, values);
								else
									baseline_softmax_forward<double>(values, values);
							}
						}
					}
				}
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				assert(input.size() == 1);
				assert(gradient_prev.size() == 1);
				const int batch_size = input[0].dim(0);
				const int tokens = input[0].dim(1) * input[0].dim(2);
				const int experts = input[0].dim(3);
				const int capacity = m_capacity_factor * tokens / experts + 0.5f;

				if (m_algorithm == "hash")
				{
				}
				else
				{
					Tensor grad = zeros_like(gradient_prev[0]);
					if (m_algorithm == "token_choice")
					{
						grad.reshape( { batch_size, tokens, experts });
						for (int b = 0; b < batch_size; b++)
							for (int e = 0; e < experts; e++)
								for (int k = 0; k < capacity; k++)
								{
									const int index = (int) output.at( { b, 0, e, k });
									if (index >= 0)
										grad.at( { b, index, e }) = gradient_next.at( { b, 1, e, k });
								}

						grad.reshape( { batch_size * tokens, experts });
						const Tensor logits = input[0].view().flatten( { 0, 1, 2 });
						Tensor values = zeros_like(logits);

						if (dtype() == DataType::FLOAT32)
						{
							baseline_softmax_forward<float>(values, logits);
							baseline_softmax_backward<float>(grad, grad, values);
						}
						else
						{
							baseline_softmax_forward<double>(values, logits);
							baseline_softmax_backward<double>(grad, grad, values);
						}

					}
					if (m_algorithm == "expert_choice")
					{
						for (int b = 0; b < batch_size; b++)
						{
							Tensor values = output.view( { experts, capacity }, { b, 1, 0, 0 });
							Tensor gradients = gradient_next.view( { experts, capacity }, { b, 1, 0, 0 });
							if (dtype() == DataType::FLOAT32)
								baseline_softmax_backward<float>(gradients, gradients, values);
							else
								baseline_softmax_backward<double>(gradients, gradients, values);
						}

						grad.flatten( { 1, 2 });
						for (int b = 0; b < batch_size; b++)
							for (int e = 0; e < experts; e++)
								for (int k = 0; k < capacity; k++)
								{
									const int index = (int) output.at( { b, 0, e, k });
									grad.at( { b, index, e }) = gradient_next.at( { b, 1, e, k });
								}
					}

					grad.reshape(gradient_prev[0].shape());
					addTensors(context(), 1.0f, grad, beta[0], gradient_prev[0], 0.0f, gradient_prev[0]);
				}
			}
	};
	/*
	 * 					router
	 * inputs:	[NHWC]	[N2EK]
	 * weights:
	 * output:	[NKEC]	K - selected top K out of HW tokens
	 */
	class BaselineGatherTopK: public Layer
	{
		public:
			BaselineGatherTopK() :
					Layer()
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				assert(shapes.size() == 2);
				assert(shapes[0].rank() == 4);
				assert(shapes[1].rank() == 4);
				assert(shapes[0].dim(0) == shapes[1].dim(0)); // batch size match
				assert(shapes[1].dim(1) == 2);
				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape(0).dim(0);
				const int channels = getInputShape(0).dim(3);
				assert(getInputShape(1).dim(0) == batch_size);
				assert(getInputShape(1).dim(1) == 2);
				const int experts = getInputShape(1).dim(2);
				const int top_k = getInputShape(1).dim(3);
				return Shape( { batch_size, top_k, experts, channels });
			}
			std::string name() const
			{
				return "BaselineGatherTopK";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineGatherTopK> result = std::make_unique<BaselineGatherTopK>();
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				assert(input.size() == 2);

				if (dtype() == DataType::FLOAT32)
					gather_tokens_forward(output, input[0], input[1]);
				else
					gather_tokens_forward(output, input[0], input[1]);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				assert(input.size() == 2);
				assert(gradient_prev.size() == 2);

				if (dtype() == DataType::FLOAT32)
					gather_tokens_backward<float>(beta[0], gradient_prev[0], gradient_next, input[1]);
				else
					gather_tokens_backward<double>(beta[0], gradient_prev[0], gradient_next, input[1]);
			}
	};
	/*
	 * 			MoE		router
	 * inputs:	[NKEC]	[N2EK]	K - selected top K out of HW tokens
	 * weights:
	 * output:	[NHWC]
	 */
	class BaselineScatter: public Layer
	{
			int m_height = 0;
			int m_width = 0;
		public:
			BaselineScatter(int height, int width) :
					Layer(),
					m_height(height),
					m_width(width)
			{
			}
			void setInputShape(const std::vector<Shape> &shapes)
			{
				assert(shapes.size() == 2);
				assert(shapes[0].rank() == 4);
				assert(shapes[1].rank() == 4);
				assert(shapes[0].dim(0) == shapes[1].dim(0)); // batch size match
				assert(shapes[0].dim(1) == shapes[1].dim(3)); // top k match
				assert(shapes[0].dim(2) == shapes[1].dim(2)); // experts match

				m_input_shapes = shapes;
			}
			Shape getOutputShape() const
			{
				const int batch_size = getInputShape(0).dim(0);
				const int channels = getInputShape(0).dim(3);
				return Shape( { batch_size, m_height, m_width, channels });
			}
			std::string name() const
			{
				return "BaselineScatter";
			}
			Json getConfig() const
			{
				Json result = Layer::getConfig();
				result["height"] = m_height;
				result["width"] = m_width;
				return result;
			}
			std::unique_ptr<Layer> clone(const Json &config) const
			{
				std::unique_ptr<BaselineScatter> result = std::make_unique<BaselineScatter>(config["height"].getInt(), config["width"].getInt());
				result->m_dtype = typeFromString(config["dtype"].getString());
				return result;
			}

			void forward(const std::vector<Tensor> &input, Tensor &output)
			{
				assert(input.size() == 2);

				if (dtype() == DataType::FLOAT32)
					scatter_tokens_forward<float>(output, input[0], input[1]);
				else
					scatter_tokens_forward<double>(output, input[0], input[1]);
			}
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next,
					const std::vector<float> &beta)
			{
				assert(input.size() == 2);
				assert(gradient_prev.size() == 2);

				if (dtype() == DataType::FLOAT32)
					scatter_tokens_backward<float>(beta[0], gradient_prev[0], gradient_next, input[0], input[1], beta[1], gradient_prev[1]);
				else
					scatter_tokens_backward<double>(beta[0], gradient_prev[0], gradient_next, input[0], input[1], beta[1], gradient_prev[1]);
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
//		testing::GradientCheck gradcheck { BaselineRouter("token_choice", 2.0f) };
//		gradcheck.setInputShape(Shape( { 1, 11, 12, 8 }));
//
//		gradcheck.check(100, 1.0e-3, "all", true);
//
//		exit(0);
//	}
//	TEST(TestGatherTopK, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineGatherTopK() };
//		const Shape input( { 3, 11, 12, 37 });
//		const Shape router_output( { 3, 2, 4, 8 });
//		gradcheck.setInputShape( { input, router_output });
//
//		gradcheck.check(1000, 1.0e-3, "all", true);
//
//		exit(0);
//	}
//	TEST(TestScatterTopK, baseline)
//	{
//		testing::GradientCheck gradcheck { BaselineScatter(15,15) };
//		const Shape moe_output( { 3, 12, 11, 37 });
//		const Shape router_output( { 3, 2, 11, 12 });
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
		const int height = 7;
		const int width = 8;
		const int experts = 2;
		const bool verbose = true;

		const Shape input_shape( { batch_size, height, width, experts });

		testing::LayerCheck baseline { BaselineRouter("token_choice", 1.25f) };
		testing::LayerCheck under_test { Router(RoutingAlgorithm::TOKEN_CHOICE, 1.25f) };

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
	}

	TEST(TestGatherTopK, forward_and_backward)
	{
		const int batch_size = 3;
		const int height = 11;
		const int width = 12;
		const int channels = 56;
		const int experts = 8;
		const int top_k = 12;
		const bool verbose = false;

		const Shape input_shape( { batch_size, height, width, channels });
		const Shape router_output_shape( { batch_size, 2, experts, top_k });

		testing::LayerCheck baseline { BaselineGatherTopK() };
		testing::LayerCheck under_test { GatherTokens() };

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
		const int batch_size = 3;
		const int height = 11;
		const int width = 12;
		const int channels = 56;
		const int experts = 8;
		const int top_k = 16;
		const bool verbose = false;

		const Shape input_shape( { batch_size, top_k, experts, channels });
		const Shape router_output_shape( { batch_size, 2, experts, top_k });

		testing::LayerCheck baseline { BaselineScatter(height, width) };
		testing::LayerCheck under_test { ScatterTokens(height, width) };

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
		const int batch_size = 32;
		const int top_k = 16;
		const int channels = 128;
		const int neurons = 64;
		const int experts = 32;
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
