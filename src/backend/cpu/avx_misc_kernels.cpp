/*
 * avx_misc_kernels.cpp
 *
 *  Created on: Oct 20, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/backend_types.h>
#include <minml/backend/backend_utils.hpp>

#include "utils.hpp"
#include "misc_kernels.hpp"
#include "common_math.hpp"
#include "fp16.hpp"

#include <cstddef>
#include <cstring>
#include <cmath>
#include <cassert>
#include <x86intrin.h>

#include "assembly_macros.hpp"

namespace
{
	using namespace ml::cpu;

	template<typename DstT, typename SrcT>
	DstT convert_to(SrcT x) noexcept
	{
		return static_cast<DstT>(x);
	}
	template<>
	ml::cpu::float16 convert_to(float x) noexcept
	{
#if defined(__AVX__) && defined(__F16C__)
		return _cvtss_sh(x, (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC));
#else
		return 0;
#endif
	}
	template<>
	float convert_to(ml::cpu::float16 x) noexcept
	{
#if defined(__AVX__) && defined(__F16C__)
		return _cvtsh_ss(x.m_data);
#else
		return 0.0f;
#endif
	}

	template<typename T>
	void kernel_softmax_3_channels(void *dst, const void *src, int first_dim)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const T *src_ptr = ml::getPointer<T>(src);
		T *dst_ptr = ml::getPointer<T>(dst);

		for (int i = 0; i < first_dim; i++)
		{
			float x0 = convert_to<float>(src_ptr[0]);
			float x1 = convert_to<float>(src_ptr[1]);
			float x2 = convert_to<float>(src_ptr[2]);

			const float max_value = std::max(x0, std::max(x1, x2));
			x0 = std::exp(x0 - max_value);
			x1 = std::exp(x1 - max_value);
			x2 = std::exp(x2 - max_value);

			const float inv_sum = 1.0f / (x0 + x1 + x2);
			dst_ptr[0] = convert_to<T>(x0 * inv_sum);
			dst_ptr[1] = convert_to<T>(x1 * inv_sum);
			dst_ptr[2] = convert_to<T>(x2 * inv_sum);

			src_ptr += 3;
			dst_ptr += 3;
		}
	}
	template<typename T>
	void kernel_softmax(void *dst, const void *src, int first_dim, int last_dim, void *workspace)
	{
		assert(dst != nullptr);
		assert(src != nullptr);
		const T *src_ptr = ml::getPointer<T>(src);
		T *dst_ptr = ml::getPointer<T>(dst);
		float *workspace_ptr = ml::getPointer<float>(workspace);

		for (int i = 0; i < first_dim; i++)
		{
			for (int j = 0; j < last_dim; j++)
				workspace_ptr[j] = convert_to<float>(src_ptr[j]);

			float max_value = workspace_ptr[0];
			for (int j = 0; j < last_dim; j++)
				max_value = std::max(max_value, workspace_ptr[j]);

			float sum = 0.0f;
			for (int j = 0; j < last_dim; j++)
			{
				const float tmp = std::exp(workspace_ptr[j] - max_value);
				sum += tmp;
				workspace_ptr[j] = tmp;
			}

			const float scale = 1.0f / sum;
			for (int j = 0; j < last_dim; j++)
				dst_ptr[j] = convert_to<T>(workspace_ptr[j] * scale);
			src_ptr += last_dim;
			dst_ptr += last_dim;
		}
	}
	template<typename T>
	void kernel_activation_forward(void *dst, const void *src, size_t elements, ml::mlActivationType_t activation)
	{
		T *dst_ptr = ml::getPointer<T>(dst);
		const T *src_ptr = ml::getPointer<T>(src);
		switch (activation)
		{
			case ml::ACTIVATION_LINEAR:
				if (dst != src)
					std::memcpy(dst, src, sizeof(float) * elements);
				break;
			case ml::ACTIVATION_SIGMOID:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert_to<T>(sigmoid(convert_to<float>(src_ptr[i])));
				break;
			case ml::ACTIVATION_TANH:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert_to<T>(std::tanh(convert_to<float>(src_ptr[i])));
				break;
			case ml::ACTIVATION_RELU:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert_to<T>(relu(convert_to<float>(src_ptr[i])));
				break;
				break;
			case ml::ACTIVATION_EXP:
				for (size_t i = 0; i < elements; i++)
					dst_ptr[i] = convert_to<T>(std::exp(convert_to<float>(src_ptr[i])));
				break;
			default:
				break;
		}
	}

	template<typename T, ml::mlActivationType_t ACT>
	void kernel_add_bias_act(void *output, const void *input, const void *bias, int first_dim, int last_dim)
	{
		T *output_ptr = ml::getPointer<T>(output);
		const T *input_ptr = ml::getPointer<T>(input);
		const T *bias_ptr = ml::getPointer<T>(bias);

		for (int i = 0; i < first_dim; i++)
		{
			for (int j = 0; j < last_dim; j++)
			{
				float tmp = convert_to<float>(input_ptr[j]) + convert_to<float>(bias_ptr[j]);
				switch (ACT)
				{
					default:
					case ml::ACTIVATION_LINEAR:
						break;
					case ml::ACTIVATION_SIGMOID:
						tmp = sigmoid(tmp);
						break;
					case ml::ACTIVATION_TANH:
						tmp = std::tanh(tmp);
						break;
					case ml::ACTIVATION_RELU:
						tmp = relu(tmp);
						break;
					case ml::ACTIVATION_EXP:
						tmp = std::exp(tmp);
						break;
				}
				output_ptr[j] = convert_to<T>(tmp);
			}
			input_ptr += last_dim;
			output_ptr += last_dim;
		}
	}

	template<int Size>
	class Line
	{
			constexpr static int num_registers = Size / 8;
			__m256 data[num_registers];
		public:
			constexpr int size() const noexcept
			{
				return Size;
			}
			void setzero() noexcept
			{
				for (int i = 0; i < num_registers; i++)
					data[i] = _mm256_setzero_ps();
			}
			void load(const float *src) noexcept
			{
				for (int i = 0; i < num_registers; i++)
					data[i] = _mm256_loadu_ps(src + 8 * i);
			}
			void load(const ml::cpu::float16 *src) noexcept
			{
				for (int i = 0; i < num_registers; i++)
					data[i] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*) (src + 8 * i)));
			}
			Line& operator+=(const Line<Size> &other) noexcept
			{
				for (int i = 0; i < num_registers; i++)
					data[i] = _mm256_add_ps(data[i], other.data[i]);
				return *this;
			}
			Line& operator*=(float x) noexcept
			{
				const __m256 tmp = _mm256_set1_ps(x);
				for (int i = 0; i < num_registers; i++)
					data[i] = _mm256_mul_ps(data[i], tmp);
				return *this;
			}
			Line& max(const Line<Size> &other) noexcept
			{
				for (int i = 0; i < num_registers; i++)
					data[i] = _mm256_max_ps(data[i], other.data[i]);
				return *this;
			}
			void store(float *dst) const noexcept
			{
				for (int i = 0; i < num_registers; i++)
					_mm256_storeu_ps(dst + 8 * i, data[i]);
			}
			void store(ml::cpu::float16 *dst) const noexcept
			{
				for (int i = 0; i < num_registers; i++)
				{
					const __m128i tmp = _mm256_cvtps_ph(data[i], (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC));
					_mm_storeu_si128((__m128i*) (dst + 8 * i), tmp);
				}
			}
			Line& relu() noexcept
			{
				const __m256 zero = _mm256_setzero_ps();
				for (int i = 0; i < num_registers; i++)
					data[i] = _mm256_max_ps(data[i], zero);
				return *this;
			}
	};

	template<typename T>
	void kernel_pooling(ml::mlShape_t shape, const void *input, void *output, float *workspace)
	{
		const int batch_size = shape.dim[0];
		const int hw = shape.dim[1] * shape.dim[2];
		const int channels = shape.dim[3];

		constexpr int Length = 32;
		const int k_iter = channels - (channels % Length);

		const T *input_ptr = reinterpret_cast<const T*>(input);
		T *output_ptr = reinterpret_cast<T*>(output);

		Line<Length> avg;
		Line<Length> max;
		Line<Length> tmp;

		const float inv = 1.0f / hw;
		for (int i = 0; i < batch_size; i++)
		{
			for (int k = 0; k < channels; k++)
			{
				workspace[k] = 0.0f; // avg
				workspace[k + channels] = convert_to<float>(input_ptr[k]); // max
			}

			for (int j = 0; j < hw; j++)
			{
				for (int k = 0; k < k_iter; k += Length)
				{
					avg.load(workspace + k);
					max.load(workspace + channels + k);

					tmp.load(input_ptr + k);
					avg += tmp;
					max = max.max(tmp);

					avg.store(workspace + k);
					max.store(workspace + channels + k);
				}
				for (int k = k_iter; k < channels; k++)
				{
					const float tmp = convert_to<float>(input_ptr[k]);
					workspace[k] += tmp;
					workspace[k + channels] = std::max(workspace[k + channels], tmp);
				}
				input_ptr += channels;
			}

			for (int k = 0; k < channels; k++)
			{
				output_ptr[k] = convert_to<T>(convert_to<float>(workspace[k]) * inv);
				output_ptr[k + channels] = convert_to<T>(workspace[k + channels]);
			}
			output_ptr += 2 * channels;

//			for (int k = 0; k < k_iter; k += length)
//			{
//				Line<length> avg;
//				Line<length> max;
//				Line<length> tmp;
//				avg.setzero();
//				max.load(input_ptr + k);
//				for (int j = 0; j < hw; j++)
//				{
//					tmp.load(input_ptr + k);
//					avg += tmp;
//					max = max.max(tmp);
//					input_ptr += channels;
//				}
//				avg *= inv;
//				avg.store(output_ptr + k);
//				max.store(output_ptr + channels + k);
//				input_ptr -= hw * channels;
//			}
//
//			for (int k = k_iter; k < channels; k++)
//			{
//				output_ptr[k] = 0.0f; // avg
//				output_ptr[k + channels] = input_ptr[k]; // max
//			}
//			for (int j = 0; j < hw; j++)
//			{
//				for (int k = k_iter; k < channels; k++)
//				{
//					const float tmp = convert_to<float>(input_ptr[k]);
//					output_ptr[k] = convert_to<T>(convert_to<float>(output_ptr[k]) + tmp);
//					output_ptr[k + channels] = convert_to<T>(std::max(convert_to<float>(output_ptr[k + channels]), tmp));
//				}
//				input_ptr += channels;
//			}
//			for (int k = k_iter; k < channels; k++)
//				output_ptr[k] = convert_to<T>(convert_to<float>(output_ptr[k]) * inv);
//			output_ptr += 2 * channels;
		}
	}
	template<typename T, ml::mlActivationType_t ACT>
	void kernel_broadcasting(ml::mlShape_t shape, void *output, const void *input, const void *bias, float *workspace)
	{
		const int batch_size = shape.dim[0];
		const int hw = shape.dim[1] * shape.dim[2];
		const int channels = shape.dim[3];

		const T *input_ptr = reinterpret_cast<const T*>(input);
		const T *bias_ptr = reinterpret_cast<const T*>(bias);
		T *output_ptr = reinterpret_cast<T*>(output);

		constexpr int length = 32;
		const int k_iter = channels - (channels % length);

		Line<length> in, b;
		for (int i = 0; i < batch_size; i++)
		{
			for (int k = 0; k < channels; k++)
				workspace[k] = convert_to<float>(bias_ptr[k]);
			for (int j = 0; j < hw; j++)
			{
				for (int k = 0; k < k_iter; k += length)
				{
					in.load(input_ptr + k);
					b.load(workspace + k);
					in += b;
					switch (ACT)
					{
						default:
						case ml::ACTIVATION_LINEAR:
							break;
						case ml::ACTIVATION_SIGMOID:
							for (int l = 0; l < length; l++)
								reinterpret_cast<float*>(&in)[l] = sigmoid(reinterpret_cast<float*>(&in)[l]);
							break;
						case ml::ACTIVATION_TANH:
							for (int l = 0; l < length; l++)
								reinterpret_cast<float*>(&in)[l] = std::tanh(reinterpret_cast<float*>(&in)[l]);
							break;
						case ml::ACTIVATION_RELU:
							in.relu();
							break;
					}
					in.store(output_ptr + k);
				}
				for (int k = k_iter; k < channels; k++)
				{
					float tmp = convert_to<float>(input_ptr[k]) + workspace[k];
					switch (ACT)
					{
						default:
						case ml::ACTIVATION_LINEAR:
							break;
						case ml::ACTIVATION_SIGMOID:
							tmp = sigmoid(tmp);
							break;
						case ml::ACTIVATION_TANH:
							tmp = std::tanh(tmp);
							break;
						case ml::ACTIVATION_RELU:
							tmp = relu(tmp);
							break;
					}
					output_ptr[k] = convert_to<T>(tmp);
				}
				input_ptr += channels;
				output_ptr += channels;
			}
			bias_ptr += channels;
		}
	}

//	template<typename T>
//	void kernel_pooling(ml::mlShape_t shape, const void *input, void *output)
//	{
//		const int batch_size = shape.dim[0];
//		const int hw = shape.dim[1] * shape.dim[2];
//		const int channels = shape.dim[3];
//
//		const T *input_ptr = reinterpret_cast<const T*>(input);
//		T *output_ptr = reinterpret_cast<T*>(output);
//
//		const float inv = 1.0f / hw;
//		for (int i = 0; i < batch_size; i++)
//		{
//			for (int k = 0; k < channels; k++)
//			{
//				output_ptr[k] = 0.0f; // avg
//				output_ptr[k + channels] = input_ptr[k]; // max
//			}
//			for (int j = 0; j < hw; j++)
//			{
//				for (int k = 0; k < channels; k++)
//				{
//					const float tmp = convert_to<float>(input_ptr[k]);
//					output_ptr[k] = convert_to<T>(convert_to<float>(output_ptr[k]) + tmp);
//					output_ptr[k + channels] = convert_to<T>(std::max(convert_to<float>(output_ptr[k + channels]), tmp));
//				}
//				input_ptr += channels;
//			}
//			for (int k = 0; k < channels; k++)
//				output_ptr[k] = convert_to<T>(convert_to<float>(output_ptr[k]) * inv);
//			output_ptr += 2 * channels;
//		}
//	}
//	template<typename T, ml::mlActivationType_t ACT>
//	void kernel_broadcasting(ml::mlShape_t shape, void *output, const void *input, const void *bias)
//	{
//		const int batch_size = shape.dim[0];
//		const int hw = shape.dim[1] * shape.dim[2];
//		const int channels = shape.dim[3];
//
//		const T *input_ptr = reinterpret_cast<const T*>(input);
//		const T *bias_ptr = reinterpret_cast<const T*>(bias);
//		T *output_ptr = reinterpret_cast<T*>(output);
//
//		for (int i = 0; i < batch_size; i++)
//		{
//			for (int j = 0; j < hw; j++)
//			{
//				for (int k = 0; k < channels; k++)
//				{
//					float tmp = convert_to<float>(input_ptr[k]) + convert_to<float>(bias_ptr[k]);
//					switch (ACT)
//					{
//						default:
//						case ml::ACTIVATION_LINEAR:
//							break;
//						case ml::ACTIVATION_SIGMOID:
//							tmp = sigmoid(tmp);
//							break;
//						case ml::ACTIVATION_TANH:
//							tmp = std::tanh(tmp);
//							break;
//						case ml::ACTIVATION_RELU:
//							tmp = relu(tmp);
//							break;
//					}
//					output_ptr[k] = convert_to<T>(tmp);
//				}
//				input_ptr += channels;
//				output_ptr += channels;
//			}
//			bias_ptr += channels;
//		}
//	}

}

namespace ml
{
	namespace cpu
	{
		void avx_kernel_convert_fp32_to_fp16(void *dst, const void *src, size_t elements)
		{
			const float *src_ptr = getPointer<float>(src);
			float16 *dst_ptr = getPointer<float16>(dst);
			const uint64_t k_iter = elements / 16;
			const uint64_t k_left = elements % 16;

			begin_asm()
			movq(var(src_ptr), rax)
			movq(var(dst_ptr), rbx)

			movq(var(k_iter), r15)
			test(r15, r15)
			je(FINALLOOP)

			label(UNROLLED_x16)
			vmovups(mem(rax), ymm0)
			vmovups(mem(rax, 4*8), ymm1)
			vcvtps2ph(imm(0x03), ymm0, xmm0)
			vcvtps2ph(imm(0x03), ymm1, xmm1)
			vmovups(xmm0, mem(rbx))
			vmovups(xmm1, mem(rbx, 2*8))

			add(imm(4*16), rax)
			add(imm(2*16), rbx)
			dec(r15)
			jne(UNROLLED_x16)

			label(FINALLOOP)
			movq(var(k_left), r15)
			test(r15, r15)
			je(EPILOGUE)

			label(UNROLLED_x1)
			vmovss(mem(rax), xmm0)
			vcvtps2ph(imm(0x03), xmm0, xmm0)
			vmovq(xmm0, rcx)
			mov(cx, mem(rbx))

			add(imm(4), rax)
			add(imm(2), rbx)
			dec(r15)
			jne(UNROLLED_x1)

			label(EPILOGUE)
			vzeroupper()
			end_asm(
					: // outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left)
					:// clobbers
					"cc", "memory", "%ymm0", "%ymm1", "%rax", "%rbx", "%rcx", "%r15")
		}
		void avx_kernel_convert_fp16_to_fp32(void *dst, const void *src, size_t elements)
		{
			const float *src_ptr = getPointer<float>(src);
			float16 *dst_ptr = getPointer<float16>(dst);
			const uint64_t k_iter = elements / 16;
			const uint64_t k_left = elements % 16;

			begin_asm()
			movq(var(src_ptr), rax)
			movq(var(dst_ptr), rbx)

			movq(var(k_iter), r15)
			test(r15, r15)
			je(FINALLOOP)

			label(UNROLLED_x16)
			vmovups(mem(rax), xmm0)
			vmovups(mem(rax, 2*8), xmm1)
			vcvtph2ps(xmm0, ymm0)
			vcvtph2ps(xmm1, ymm1)
			vmovups(ymm0, mem(rbx))
			vmovups(ymm1, mem(rbx, 4*8))

			add(imm(2*16), rax)
			add(imm(4*16), rbx)
			dec(r15)
			jne(UNROLLED_x16)

			label(FINALLOOP)
			movq(var(k_left), r15) // load the number of 1-unrolled iterations
			test(r15, r15)
			je(EPILOGUE)

			label(UNROLLED_x1)
			mov(mem(rax), cx)
			vmovq(rcx, xmm0)
			vcvtph2ps(xmm0, xmm0)
			vmovss(xmm0, mem(rbx))

			add(imm(2), rax)
			add(imm(4), rbx)
			dec(r15)
			jne(UNROLLED_x1)

			label(EPILOGUE)
			vzeroupper()
			end_asm(
					:// outputs
					:// inputs
					[src_ptr] "m"(src_ptr),
					[dst_ptr] "m"(dst_ptr),
					[k_iter] "m"(k_iter),
					[k_left] "m"(k_left)
					:// clobbers
					"cc", "memory", "%ymm0", "%ymm1", "%rax", "%rbx", "%rcx", "%r15")
		}

		void avx_kernel_softmax_3_channels_fp16(void *dst, const void *src, int first_dim)
		{
			kernel_softmax_3_channels<float16>(dst, src, first_dim);
		}
		void avx_kernel_softmax_fp16(void *dst, const void *src, int first_dim, int last_dim, void *workspace)
		{
			kernel_softmax<float16>(dst, src, first_dim, last_dim, workspace);
		}

		void avx_kernel_activation_forward_fp16(void *dst, const void *src, size_t elements, mlActivationType_t activation)
		{
			kernel_activation_forward<float16>(dst, src, elements, activation);
		}

		void avx_kernel_add_bias_act_fp16(void *output, const void *input, const void *bias, int first_dim, int last_dim, mlActivationType_t act)
		{
			switch (act)
			{
				default:
				case ACTIVATION_LINEAR:
					kernel_add_bias_act<float16, ACTIVATION_LINEAR>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_SIGMOID:
					kernel_add_bias_act<float16, ACTIVATION_SIGMOID>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_TANH:
					kernel_add_bias_act<float16, ACTIVATION_TANH>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_RELU:
					kernel_add_bias_act<float16, ACTIVATION_RELU>(output, input, bias, first_dim, last_dim);
					break;
				case ACTIVATION_EXP:
					kernel_add_bias_act<float16, ACTIVATION_EXP>(output, input, bias, first_dim, last_dim);
					break;
			}
		}

		void avx_kernel_global_avg_and_max_pooling_forward_fp32(mlContext_t context, mlShape_t shape, const void *input, void *output)
		{
			kernel_pooling<float>(shape, input, output, Context::getWorkspace<float>(context));
//			const int batch_size = shape.dim[0];
//			const int hw = shape.dim[1] * shape.dim[2];
//			const int channels = shape.dim[3];
//
//			constexpr int length = 8;
//			const int k_iter = channels - (channels % length);
//
//			const float *input_ptr = reinterpret_cast<const float*>(input);
//			float *output_ptr = reinterpret_cast<float*>(output);
//
//			const float inv = 1.0f / hw;
//			for (int i = 0; i < batch_size; i++)
//			{
//				for (int k = 0; k < channels; k++)
//				{
//					output_ptr[k] = 0.0f; // avg
//					output_ptr[k + channels] = input_ptr[k]; // max
//				}
//
//				for (int j = 0; j < hw; j++)
//				{
//					for (int k = 0; k < k_iter; k += length)
//					{
//						const __m256 in = _mm256_loadu_ps(input_ptr + k);
//						__m256 avg = _mm256_loadu_ps(output_ptr + k);
//						__m256 max = _mm256_loadu_ps(output_ptr + k + channels);
//
//						avg = _mm256_add_ps(avg, in);
//						max = _mm256_max_ps(max, in);
//
//						_mm256_storeu_ps(output_ptr + k, avg);
//						_mm256_storeu_ps(output_ptr + k + channels, max);
//					}
//					for (int k = k_iter; k < channels; k++)
//					{
//						output_ptr[k] += input_ptr[k];
//						output_ptr[k + channels] = std::max(output_ptr[k + channels], input_ptr[k]);
//					}
//					input_ptr += channels;
//				}
//
//				for (int k = 0; k < channels; k++)
//					output_ptr[k] *= inv;
//				output_ptr += 2 * channels;
//			}
		}
		void avx_kernel_global_avg_and_max_pooling_forward_fp16(mlContext_t context, mlShape_t shape, const void *input, void *output)
		{
			kernel_pooling<float16>(shape, input, output, Context::getWorkspace<float>(context));
		}
		void avx_kernel_global_broadcasting_forward_fp32(mlContext_t context, mlShape_t shape, void *output, const void *input, const void *bias,
				mlActivationType_t act)
		{
			float *workspace = Context::getWorkspace<float>(context);
			switch (act)
			{
				case ACTIVATION_LINEAR:
					kernel_broadcasting<float, ACTIVATION_LINEAR>(shape, output, input, bias, workspace);
					break;
				case ACTIVATION_SIGMOID:
					kernel_broadcasting<float, ACTIVATION_SIGMOID>(shape, output, input, bias, workspace);
					break;
				case ACTIVATION_TANH:
					kernel_broadcasting<float, ACTIVATION_TANH>(shape, output, input, bias, workspace);
					break;
				case ACTIVATION_RELU:
					kernel_broadcasting<float, ACTIVATION_RELU>(shape, output, input, bias, workspace);
					break;
				default:
					break;
			}
		}
		void avx_kernel_global_broadcasting_forward_fp16(mlContext_t context, mlShape_t shape, void *output, const void *input, const void *bias,
				mlActivationType_t act)
		{
			float *workspace = Context::getWorkspace<float>(context);
			switch (act)
			{
				case ACTIVATION_LINEAR:
					kernel_broadcasting<float16, ACTIVATION_LINEAR>(shape, output, input, bias, workspace);
					break;
				case ACTIVATION_SIGMOID:
					kernel_broadcasting<float16, ACTIVATION_SIGMOID>(shape, output, input, bias, workspace);
					break;
				case ACTIVATION_TANH:
					kernel_broadcasting<float16, ACTIVATION_TANH>(shape, output, input, bias, workspace);
					break;
				case ACTIVATION_RELU:
					kernel_broadcasting<float16, ACTIVATION_RELU>(shape, output, input, bias, workspace);
					break;
				default:
					break;
			}
		}

	} /* namespace cpu */
} /* namespace ml */

