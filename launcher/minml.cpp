//============================================================================
// Name        : minml.cpp
// Author      : Maciej Kozarzewski
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <minml/core/Tensor.hpp>
#include <minml/core/ml_memory.hpp>
#include <minml/core/math.hpp>
#include <minml/core/Event.hpp>
#include <minml/graph/Graph.hpp>
#include <minml/graph/graph_optimizers.hpp>
#include <minml/layers/Dense.hpp>
#include <minml/layers/DepthwiseConv2D.hpp>
#include <minml/layers/Conv2D.hpp>
#include <minml/layers/BatchNormalization.hpp>
#include <minml/layers/Gelu.hpp>
#include <minml/layers/GlobalBroadcastHW.hpp>
#include <minml/layers/GlobalPooling.hpp>
#include <minml/layers/Add.hpp>
#include <minml/layers/Softmax.hpp>
#include <minml/layers/Multiply.hpp>
#include <minml/layers/MultiHeadAttention.hpp>
#include <minml/layers/PositionalEncoding.hpp>
#include <minml/layers/RMSNormalization.hpp>
#include <minml/layers/LayerNormalization.hpp>
#include <minml/layers/DepthToSpace.hpp>
#include <minml/layers/SpaceToDepth.hpp>
#include <minml/training/Optimizer.hpp>
#include <minml/training/LossFunction.hpp>
#include <minml/utils/random.hpp>
#include <minml/utils/time_util.hpp>
#include <minml/utils/file_util.hpp>
#include <minml/utils/selfcheck.hpp>

#include <minml/backend/cpu_backend.h>
#include <minml/backend/opencl_backend.h>

#include <iomanip>
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
#include <fstream>
#include <memory>
#include <cmath>
#include <x86intrin.h>
#include <omp.h>

#ifndef M_LN2
#define M_LN2 0.69314718056
#endif

using namespace ml;

class MNIST
{
	public:
		Tensor train_images;
		Tensor test_images;
		Tensor train_labels;
		Tensor test_labels;
		MNIST()
		{
			train_images = load_images("/home/maciek/Downloads/mnist/train-images-idx3-ubyte", 60000);
			test_images = load_images("/home/maciek/Downloads/mnist/t10k-images-idx3-ubyte", 10000);

			train_labels = load_labels("/home/maciek/Downloads/mnist/train-labels-idx1-ubyte", 60000);
			test_labels = load_labels("/home/maciek/Downloads/mnist/t10k-labels-idx1-ubyte", 10000);
		}
		void printSample(int index)
		{
			std::cout << "label = " << train_labels.get( { index }) << '\n';
			for (int i = 0; i < 28; i++)
			{
				for (int j = 0; j < 28; j++)
					if (train_images.get( { index, i * 28 + j }) < 0.5f)
						std::cout << "..";
					else
						std::cout << "##";
				std::cout << '\n';
			}
		}
		void packSamples(Tensor &input, Tensor &target, int sample_index = -1)
		{
			assert(input.firstDim() == target.firstDim());
			const int sample_size = sizeof(float) * 28 * 28;
			Tensor storage(input.shape(), input.dtype(), Device::cpu());
			for (int i = 0; i < input.firstDim(); i++)
			{
				const int index = (sample_index == -1) ? randInt(train_images.firstDim()) : sample_index;

//				for (int h = 0; h < 28; h++)
//					for (int w = 0; w < 28; w++)
//					{
//						const int h0 = h / 4;
//						const int h1 = h % 4;
//						const int w0 = w / 4;
//						const int w1 = w % 4;
//						storage.set(train_images.get( { index, h * 28 + w }), { i, h0, w0, h1 * 4 + w1 });
//					}

				if (input.dtype() == DataType::FLOAT32)
					ml::memcpy(input.device(), input.data(), i * sample_size, Device::cpu(), train_images.data(), index * sample_size, sample_size);
				else
				{
					for (int h = 0; h < 28; h++)
						for (int w = 0; w < 28; w++)
							input.set(train_images.get( { index, h * 28 + w }), { i, h, w, 0 });
				}
				for (int j = 0; j < 10; j++)
					target.set(0.0f, { i, j });
				const int label = train_labels.get( { index });
				target.set(1.0f, { i, label });
			}
//			input.copyFromHost(storage.data(), input.sizeInBytes());
		}
	private:
		Tensor load_images(const std::string &path, int n)
		{
			std::fstream stream(path, std::fstream::in);
			std::unique_ptr<char[]> buffer = std::make_unique<char[]>(n * 28 * 28);
			stream.read(buffer.get(), 16); // skip header
			stream.read(buffer.get(), n * 28 * 28);
			Tensor images( { n, 28 * 28 }, "float32", Device::cpu());
			for (int i = 0; i < n; i++)
				for (int j = 0; j < 28 * 28; j++)
				{
					const uint8_t tmp = reinterpret_cast<uint8_t*>(buffer.get())[i * 28 * 28 + j];
					images.set(tmp / 255.0f, { i, j });
				}
			return images;
		}
		Tensor load_labels(const std::string &path, int n)
		{
			std::fstream stream(path, std::fstream::in);
			std::unique_ptr<char[]> buffer = std::make_unique<char[]>(n);
			stream.read(buffer.get(), 8); // skip header

			Tensor labels( { n }, "float32", Device::cpu());
			stream.read(buffer.get(), n);
			for (int i = 0; i < n; i++)
				labels.set(buffer[i], { i });
			return labels;
		}
};

double get_accuracy(const Tensor &output, const Tensor &target)
{
	assert(output.firstDim() == target.firstDim());
	double correct = 0;
	for (int i = 0; i < output.firstDim(); i++)
	{
		int output_idx = 0;
		float output_max = output.get( { i, 0 });
		for (int j = 0; j < output.lastDim(); j++)
			if (output.get( { i, j }) > output_max)
			{
				output_max = output.get( { i, j });
				output_idx = j;
			}

		int target_idx = 0;
		for (int j = 0; j < target.lastDim(); j++)
			if (target.get( { i, j }) == 1.0f)
				target_idx = j;

		correct += static_cast<int>(target_idx == output_idx);
//		if (target_idx != output_idx)
//		{
//			std::cout << "output = ";
//			for (int j = 0; j < 10; j++)
//				std::cout << output.get( { i, j }) << ' ';
//			std::cout << '\n';
//			std::cout << "target = ";
//			for (int j = 0; j < 10; j++)
//				std::cout << target.get( { i, j }) << ' ';
//			std::cout << '\n' << '\n';
//			exit(0);
//		}
	}
	return correct;
}

uint32_t as_uint(const float x)
{
	return *(uint32_t*) &x;
}
float as_float(const uint32_t x)
{
	return *(float*) &x;
}

void print_bits(const uint16_t x)
{
	for (int i = 15; i >= 0; i--)
	{
		std::cout << ((x >> i) & 1);
		if (i == 15 || i == 10)
			std::cout << " ";
	}
	std::cout << '\n';
}
void print_bits(const uint32_t x)
{
	for (int i = 31; i >= 0; i--)
		std::cout << ((x >> i) & 1);
	std::cout << '\n';
}
void print_bits(const float x)
{
	uint32_t b = *(uint32_t*) &x;
	for (int i = 31; i >= 0; i--)
	{
		std::cout << ((b >> i) & 1);
		if (i == 31 or i == 23)
			std::cout << " ";
	}
	int e = (b & 0x7F800000u) >> 23u;
	std::cout << " : exponent = " << e << " (" << e - 127 << "), mantissa = " << (b & 0x007FFFFFu) << '\n';
}

// IEEE-754 16-bit floating-point format
float half_to_float(const uint16_t x)
{
//	uint32_t exponent = (x & 0x7C00) >> 10; // exponent
//	uint32_t mantissa = (x & 0x03FF) << 13; // mantissa
//
//	const uint32_t sign = (x & 0x8000) << 16;
//	if (exponent == 31)
//	{
//		exponent = 0x7F800000; // +/- Inf or +/- NaN
//		if (mantissa != 0)
//			mantissa |= (1 << 22);
//	}
//	else
//	{
//		if (exponent != 0)
//			exponent = (exponent + 112) << 23; // normalized
//		else
//		{
//			if (exponent == 0 and mantissa != 0)
//			{ // denormalized
//				const uint32_t v = _lzcnt_u32(mantissa);
//				exponent = (121 - v) << 23;
//				mantissa = (mantissa << (v - 8)) & 0x007FE000;
//			}
//		}
//	}
	uint32_t exponent = x & 0x7C00; // '0 11111 0000000000'
	uint32_t mantissa = x & 0x03FF; // '0 00000 1111111111'

	const uint32_t sign = (x & 0x8000) << 16; // '1 00000 0000000000'
	if (exponent == 0x7C00)
	{
		exponent = 0x3FC00; // +/- Inf or +/- NaN (it's 0x7F800000 >> 13)
		mantissa |= ((mantissa != 0) << 9); // set first bit of the mantissa in case of NaN, but preserve other bits
	}
	else
	{
		if (exponent != 0) // normalized
			exponent += (112 << 10);
		else
		{
			if (mantissa != 0)
			{ // denormalized
//				const uint32_t v = _bit_scan_forward(mantissa); //_lzcnt_u32(mantissa);
//				exponent = (134 - v) << 10;
//				mantissa = (mantissa << (v - 21)) & 0x000003FF;

				const uint32_t v = as_uint((float) mantissa) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
				exponent = (v - 24) << 10;
				mantissa = (mantissa << (137 - v)) & 0x03FF;
			}
		}
	}
	return as_float(sign | ((exponent | mantissa) << 13));

//	const uint32_t e = (x & 0x7C00) >> 10; // exponent
//	const uint32_t m = (x & 0x03FF) << 13; // mantissa
//
//	const uint32_t sign = (x & 0x8000) << 16;
//	if (e > 30)
//		return as_float(sign | 0x7F800000 | m); // +/- Inf or +/- NaN
//	if (e != 0)
//		return as_float(sign | ((e + 112) << 23 | m)); // normalized
//	if (e == 0 and m != 0)
//	{ // denormalized
////		std::cout << "m = ";
////		print_bits(m);
////		std::cout << "    ";
////		print_bits(as_uint((float) m));
////		const uint32_t v = as_uint((float) m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
////		std::cout << "v = " << v << '\n';
////		std::cout << "lzcnt = " << _lzcnt_u32(m) << '\n';
////		std::cout << "e = ";
////		print_bits((v - 37) << 23);
////		std::cout << "e'= ";
////		print_bits((121 - _lzcnt_u32(m)) << 23);
////		std::cout << "    ";
////		print_bits(m << (150 - v));
////		std::cout << "m'= ";
////		print_bits(m << (_lzcnt_u32(m) - 8)); // new
////		std::cout << "    ";
////		print_bits((uint32_t) 0x007FE000);
////		return as_float(sign | ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));
//
//		std::cout << "m = ";
//		print_bits(m);
//
//		std::cout << "v = ";
//		print_bits(as_uint((float) m));
//		std::cout << "v = " << as_uint((float) m) << '\n';
//
//		const uint32_t v = _lzcnt_u32(m);
//		std::cout << "v'= ";
//		print_bits(v);
//		std::cout << "v'= " << _lzcnt_u32(m) << '\n';
//
//		std::cout << "e = ";
//		print_bits((v - 37) << 23);
//		std::cout << "e'= ";
//		print_bits((121 - _lzcnt_u32(m)) << 23);
//		std::cout << "    ";
//		print_bits(m << (150 - v));
//		std::cout << "m'= ";
//		print_bits(m << (_lzcnt_u32(m) - 8)); // new
//		std::cout << "    ";
//		print_bits((uint32_t) 0x007FE000);
//		return as_float(sign | ((121 - v) << 23 | ((m << (v - 8)) & 0x007FE000)));
//	}
//	return as_float(sign);
}
// IEEE-754 16-bit floating-point format
enum class RoundingMode
{
	ROUND_TO_NEAREST_INT, // round to nearest
	ROUND_TO_NEG_INF,    // round down
	ROUND_TO_POS_INF,   // round up
	ROUND_TO_ZERO,  // truncate
};
template<RoundingMode RM = RoundingMode::ROUND_TO_NEAREST_INT>
uint16_t float_to_half(const float x)
{
	const uint32_t original = as_uint(x);
	const uint32_t rounded = original + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
	uint32_t exponent = (rounded & 0x7F800000) >> 23; // exponent
	uint32_t mantissa = rounded & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding

	const uint32_t sign = (original & 0x80000000) >> 16;

	if ((original & 0x7FFFFFFF) > 0x7F800000)
	{ // check NaN
		exponent = 0x7C00;
		mantissa = ((original & 0x007FFFFF) >> 13) | 0x200; // set first mantissa bit but preserve others
	}
	else
	{
		if (exponent > 142)
		{ // +/- Inf
			exponent = 0x7C00;
			mantissa = 0;
		}
		else
		{
			if (exponent > 112)
			{ // normalized
				exponent = ((exponent - 112) << 10) & 0x7C00;
				mantissa >>= 13;
			}
			else
			{ // denormalized
				mantissa += 0x007FF000; // TODO figure out why it is here
				mantissa >>= std::min(125u - exponent, 31u);
				mantissa = (mantissa + 1) >> 1;
				exponent = 0;
			}
		}
	}
	return sign | exponent | mantissa;

//	const uint32_t b = as_uint(x); // + 0x00001000u; // round-to-nearest-even: add last bit after truncated mantissa
//	const uint32_t e = (b & 0x7F800000u) >> 23u; // exponent
//	const uint32_t m = b & 0x007FFFFFu; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
//
//	const uint32_t sign = (b & 0x80000000u) >> 16u;
//
//	if (e > 142)
//	{
//		if (e != 255)
//			m = 0;
//		return sign | 0x7c00 | (m >> 13); // +/- Inf or +/- NaN
//	}
//	if (e > 112)
//	{ // normalized
//		const uint32_t shifted_exponent = (e - 112) << 10;
//		return sign | (shifted_exponent & 0x7C00) | (m >> 13);
//	}
//	if (101 < e and e < 113)
//	{ // denormalized
//		uint32_t tmp = 0x007FF000 + m;
//		tmp = tmp >> (125 - e);
//		tmp = tmp + 1;
//		tmp = tmp >> 1;
//		return sign | tmp;
//	}
//	return sign; // +/- 0
}

void test_mnist()
{
////	std::cout << Device::hardwareInfo();
//
//	for (int i = 0; i < 65536; i++)
//	{
//		const uint32_t x_native_fp32 = as_uint(_cvtsh_ss(i));
//		const uint32_t x_emulated_fp32 = as_uint(half_to_float(i));
//		if (x_native_fp32 != x_emulated_fp32)
//		{
//			std::cout << "mismatch at i = " << i << '\n';
//			const float x_native_fp32 = _cvtsh_ss(i);
//			const float x_emulated_fp32 = half_to_float(i);
//
//			std::cout << "bits (fp16)     = ";
//			print_bits((uint16_t) i);
//			std::cout << "bits (native)   = ";
//			print_bits(x_native_fp32);
//			std::cout << "bits (emulated) = ";
//			print_bits(x_emulated_fp32);
//
//			std::cout << "recovered " << x_native_fp32 << " " << x_emulated_fp32 << '\n';
//			return;
//		}
//	}
//	std::cout << "all correct\n";
//	return;
//
//	int64_t last_bit_mismatch = 0;
//	double start = getTime();
////	for (int64_t i = 0; i < 4294967296; i++)
//	for (int64_t i = 0; i < 1073741824; i++)
//	{
////		last_bit_mismatch += _cvtsh_ss(i & 65535);
//		last_bit_mismatch += half_to_float(i & 65535);
//
////		const float x = as_float(i);
////		last_bit_mismatch += _cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT);
////		last_bit_mismatch += float_to_half(x);
//	}
//	double stop = getTime();
//	std::cout << 4294967296.0 / (stop - start) << '\n';
//	std::cout << last_bit_mismatch << '\n';
//	return;
//
//	for (int64_t i = 0; i < 4294967296; i++)
////	int64_t i = 2139095041;
//	{
//		if (i % 100000000 == 0)
//			std::cout << "checked " << i / 1000000 << "M ...\n";
//		const float x = as_float(i);
//		const uint16_t x_native_fp16 = _cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT);
//		const uint16_t x_emulated_fp16 = float_to_half(x);
//		if (x_native_fp16 != x_emulated_fp16)
//		{
//			if ((x_native_fp16 & 0xFFFE) != (x_emulated_fp16 & 0xFFFE))
//			{
//				std::cout << "mismatch at i = " << i << '\n';
//
//				std::cout << "float = " << x << '\n';
//				std::cout << "bits (fp32)     = ";
//				print_bits((uint32_t) i);
//				std::cout << "bits (native)   = ";
//				print_bits(x_native_fp16);
//				std::cout << "bits (emulated) = ";
//				print_bits(x_emulated_fp16);
//
//				std::cout << "recovered fp32 " << _cvtsh_ss(x_native_fp16) << " " << _cvtsh_ss(x_emulated_fp16) << '\n';
//				return;
//			}
//			else
//				last_bit_mismatch++;
//		}
//	}
//	std::cout << "last bit mismatched " << last_bit_mismatch << '\n';
//	std::cout << "all correct\n";
//	return;
//
//	{
//		const float x = 1.23e-6;
//		print_bits(x);
//		const uint16_t x_native_fp16 = _cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
//		const uint16_t x_emulated_fp16 = float_to_half(x);
//
//		std::cout << "original  " << x << '\n';
//		std::cout << "bits (native)   = ";
//		print_bits(x_native_fp16);
//		std::cout << "bits (emulated) = ";
//		print_bits(x_emulated_fp16);
//
//		std::cout << _cvtsh_ss(x_native_fp16) << " " << _cvtsh_ss(x_emulated_fp16) << '\n';
//
//		const float x_native_fp32 = _cvtsh_ss(x_native_fp16);
//		const float x_emulated_fp32 = half_to_float(x_native_fp16);
//
//		std::cout << "original  " << x << '\n';
//		std::cout << "bits (native)   = ";
//		print_bits(x_native_fp32);
//		std::cout << "bits (emulated) = ";
//		print_bits(x_emulated_fp32);
//
//		std::cout << "recovered " << x_native_fp32 << " " << x_emulated_fp32 << '\n';
//
////		const float x_sw_fp32 = half_to_float(x_sw_fp16);
////		const float x_hw_fp32 = _cvtsh_ss(x_hw_fp16);
////
//
////		std::cout << "recovered " << x_decompressed << '\n';
//	}
//	return;

	Device::setNumberOfThreads(1);
	MNIST dataset;

	const int batch_size = 128;

	Graph model;
	auto x = model.addInput( { batch_size, 28, 28, 1 });
//	x = model.add(Conv2D(8, 3, "linear"), x);
//	x = model.add(BatchNormalization("relu").useGamma(false), x);
//
//	auto y = model.add(Conv2D(8, 3, "linear"), x);
//	y = model.add(BatchNormalization("linear").useGamma(false), y);
//	x = model.add(Add("relu"), { x, y });
//	x = model.add(Conv2D(31, 1, "relu"), x);

	x = model.add(Conv2D(8, 1, "relu"), x);
//	x = model.add(LayerNormalization().useGamma(false), x);
	x = model.add(Dense(10, "linear"), x);
	x = model.add(Softmax( { 1 }), x);
	model.addOutput(x);

//	const int embedding = 512;
//
//	auto x = model.addInput( { batch_size, 7, 7, 16 });
//	x = model.add(Conv2D(embedding, 1).useBias(false), x);
//
//	for (int i = 0; i < 1; i++)
//	{
//		auto y = model.add(LayerNormalization(), x);
//		y = model.add(Conv2D(3 * embedding, 1).useBias(false), y);
//		y = model.add(MultiHeadAttention(16, 7), y);
//		y = model.add(Conv2D(embedding, 1).useBias(false), y);
//		x = model.add(Add(), { x, y });
//
//		y = model.add(LayerNormalization(), x);
//		y = model.add(Conv2D(embedding, 1, "relu"), y);
//		y = model.add(Conv2D(embedding, 1), y);
//		x = model.add(Add(), { x, y });
//	}
//
//	x = model.add(GlobalPooling(), x);
//	x = model.add(Dense(10), x);
//	x = model.add(Softmax( { 1 }), x);
//	model.addOutput(x);

	const float learning_rate = 1.0e-3f;
	model.init();
	model.setOptimizer(Optimizer(learning_rate));
//	model.setRegularizer(Regularizer(1.0e-4f));
//	model.moveTo(Device::cpu());
	model.moveTo(Device::cuda(0));
	model.print();

//	model.forward(1);

////	x = model.add(Conv2D(32, 3, "linear"), x);
//	auto x = model.addInput( { batch_size, 28 * 28 * 1 });
//	x = model.add(Dense(64, "relu"), x);
//	auto y = model.add(Dense(64, "linear").useBias(false), x);
//	y = model.add(BatchNormalization("relu").useGamma(false), y);
//	y = model.add(Dense(64, "linear").useBias(false), y);
//	y = model.add(BatchNormalization("linear").useGamma(false), y);
//	x = model.add(Add("relu"), { x, y });
////	x = model.add(BatchNormalization("relu"), x);
////	x = model.add(Dense(64, "linear"), x);
////	x = model.add(BatchNormalization("relu"), x);
////	x = model.add(Conv2D(32, 3, "linear"), x);
////	x = model.add(BatchNormalization("relu"), x);
////	x = model.add(Flatten(), x);
//	x = model.add(Dense(10, "linear").quantizable(false), x);
//	x = model.add(Softmax().quantizable(false), x);
//	model.addOutput(x, CrossEntropyLoss());
//	model.init();
//	model.setOptimizer(ADAM());
//	model.setRegularizer(RegularizerL2(0.0001f));
//	model.moveTo(Device::cuda(1));

	const int steps = 1000;
	for (int e = 0; e < 150; e++)
	{
		if (e == 100)
			model.setLearningRate(learning_rate / 10.0f);
		double loss = 0.0;
		double acc = 0.0;
		for (int s = 0; s < steps; s++)
		{
			dataset.packSamples(model.getInput(), model.getTarget());
			model.forward(batch_size);
			model.backward(batch_size);
			model.learn();
			model.context().synchronize();
			loss += model.getLoss(batch_size).at(0);
			acc += get_accuracy(model.getOutput(), model.getTarget());
			if (loss != loss)
				break;
		}
		std::cout << "epoch " << e << ", loss = " << loss / steps << ", accuracy = " << acc / (steps * batch_size) << '\n';

//		SerializedObject so;
//		Json json = model.save(so);
//		FileSaver fs("/home/maciek/cpp_workspace/libml/mnist.bin");
//		fs.save(json, so);
		if (loss != loss)
			break;
	}
	return;
	{
		SerializedObject so;
		Json json = model.save(so);
		FileSaver fs("mnist_network.bin");
		fs.save(json, so, 2);
	}
	return;
	model.makeNonTrainable();
	{
		SerializedObject so;
		Json json = model.save(so);
		FileSaver fs("mnist_network_opt.bin");
		fs.save(json, so, 2);
	}
	return;

	dataset.packSamples(model.getInput(), model.getTarget(), 0);
	model.getOutput().zeroall(model.context());
	model.print();
	model.forward(1);
	model.context().synchronize();

	std::cout << "output = ";
	for (int j = 0; j < 10; j++)
		std::cout << model.getOutput().get( { 0, j }) << ' ';
	std::cout << '\n';
//	std::cout << "target = ";
//	for (int j = 0; j < 10; j++)
//		std::cout << model.getTarget().get( { 0, j }) << ' ';
//	std::cout << '\n' << '\n';

//	model.moveTo(Device::cpu());
//	model.getOutput().zeroall(model.context());
//	model.print();
//	model.forward(1);
//	model.context().synchronize();
//
//	std::cout << "output = ";
//	for (int j = 0; j < 10; j++)
//		std::cout << model.getOutput().get( { 0, j }) << ' ';
//	std::cout << '\n';
//	std::cout << "target = ";
//	for (int j = 0; j < 10; j++)
//		std::cout << model.getTarget().get( { 0, j }) << ' ';
//	std::cout << '\n' << '\n';

	bool result = FoldBatchNorm().optimize(model);
	std::cout << "changed anything = " << result << '\n';
	result = FoldAdd().optimize(model);
	std::cout << "changed anything = " << result << '\n';
	model.print();

	dataset.packSamples(model.getInput(), model.getTarget(), 0);
	model.getOutput().zeroall(model.context());
	model.forward(1);
	model.context().synchronize();

	std::cout << "output = ";
	for (int j = 0; j < 10; j++)
		std::cout << model.getOutput().get( { 0, j }) << ' ';
	std::cout << '\n';
	return;
//	std::cout << "target = ";
//	for (int j = 0; j < 10; j++)
//		std::cout << model.getTarget().get( { 0, j }) << ' ';
//	std::cout << '\n' << '\n';
//	return;

//	for (int i = 0; i < 28; i++)
//	{
//		for (int j = 0; j < 28; j++)
//			printf("%f ", model.getInput().get( { 0, i, j, 0 }));
//		printf("\n");
//	}
//	printf("----------------------------------------\n");

	model.convertTo(DataType::FLOAT16);
	model.print();

	dataset.packSamples(model.getInput(), model.getTarget(), 0);
//	for (int i = 0; i < 28; i++)
//	{
//		for (int j = 0; j < 28; j++)
//			printf("%f ", model.getInput().get( { 0, i, j, 0 }));
//		printf("\n");
//	}
//	printf("----------------------------------------\n");
	model.getOutput().zeroall(model.context());
	model.forward(1);
	model.context().synchronize();

	std::cout << "output = ";
	for (int j = 0; j < 10; j++)
		std::cout << model.getOutput().get( { 0, j }) << ' ';
	std::cout << '\n';
	std::cout << "target = ";
	for (int j = 0; j < 10; j++)
		std::cout << model.getTarget().get( { 0, j }) << ' ';
	std::cout << '\n' << '\n';
}

namespace gemm
{

	void* aligned_new(size_t count, size_t alignment)
	{
		if (count == 0)
			return nullptr;
		else
			return ::operator new[](count, std::align_val_t(alignment));
	}
	void aligned_free(void *ptr, size_t alignment)
	{
		if (ptr != nullptr)
			::operator delete[](ptr, std::align_val_t(alignment));
	}
	template<typename T>
	constexpr bool is_power_of_2(T x) noexcept
	{
		return (x > 0) and not (x & (x - 1));
	}

	template<size_t Alignment>
	class AlignedPointer
	{
			static_assert(is_power_of_2(Alignment), "Alignment must be a power of 2");
			void *__restrict__ ptr = nullptr;
		public:
			AlignedPointer() noexcept = default;
			AlignedPointer(size_t size) :
					ptr(aligned_new(size, Alignment))
			{
			}
			AlignedPointer(const AlignedPointer &other) = delete;
			AlignedPointer(AlignedPointer &&other) noexcept :
					ptr(other.ptr)
			{
				other.ptr = nullptr;
			}
			AlignedPointer& operator==(const AlignedPointer &other) = delete;
			AlignedPointer& operator==(AlignedPointer &&other) noexcept
			{
				std::swap(this->ptr, other.ptr);
				return *this;
			}
			~AlignedPointer()
			{
				aligned_free(ptr, Alignment);
			}

			void* get() noexcept
			{
				return ptr;
			}
			const void* get() const noexcept
			{
				return ptr;
			}
	};

	enum class Use
	{
		MATRIX_A,
		MATRIX_B,
		ACCUMULATOR
	};

	template<typename T>
	class Fragment
	{
			AlignedPointer<4096> m_ptr;
			int m_rows = 0;
			int m_columns = 0;
		public:
			Fragment() noexcept = default;
			Fragment(int rows, int columns) :
					m_ptr(sizeof(T) * rows * columns),
					m_rows(rows),
					m_columns(columns)
			{
			}
			int rows() const noexcept
			{
				return m_rows;
			}
			int columns() const noexcept
			{
				return m_columns;
			}
			int size() const noexcept
			{
				return rows() * columns();
			}
			T* data() noexcept
			{
//				return reinterpret_cast<T*>(m_ptr);
			}
			const T* data() const noexcept
			{
//				return reinterpret_cast<const T*>(m_ptr);
			}
			T& operator[](int index) noexcept
			{
				assert(0 <= index && index < size());
				return data()[index];
			}
			const T& operator[](int index) const noexcept
			{
				assert(0 <= index && index < size());
				return data()[index];
			}
			T& at(int row, int col) noexcept
			{
				assert(0 <= row && row < rows());
				assert(0 <= col && col < columns());
				return data()[row * columns() + col];
			}
			const T& at(int row, int col) const noexcept
			{
				assert(0 <= row && row < rows());
				assert(0 <= col && col < columns());
				return data()[row * columns() + col];
			}
	};

	template<int TileM, int TileN, typename DataType, typename ComputeType = DataType>
	class GemmRuntime
	{
		public:
			void packA(const DataType *__restrict__ A, Fragment<ComputeType> &fragmentA) const
			{
			}
			void packB(const DataType *__restrict__ A, Fragment<ComputeType> &fragmentB) const
			{
			}
			void packC(const DataType *__restrict__ A, Fragment<ComputeType> &fragmentC) const
			{

			}
			void run(Fragment<ComputeType> &fragmentC, const Fragment<ComputeType> &fragmentA, const Fragment<ComputeType> &fragmentB, int K) const
			{
			}
	};

	template<>
	class GemmRuntime<6, 16, float, float>
	{
		public:
			void packA(const float *__restrict__ A, Fragment<float> &fragmentA) const
			{
			}
			void packB(const float *__restrict__ A, Fragment<float> &fragmentB) const
			{
			}
			void packC(const float *__restrict__ A, Fragment<float> &fragmentC) const
			{
			}
			void run(Fragment<float> &fragmentC, const Fragment<float> &fragmentA, const Fragment<float> &fragmentB, int K) const
			{
				__m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
				__m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
				__m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
				__m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();
				__m256 acc40 = _mm256_setzero_ps(), acc41 = _mm256_setzero_ps();
				__m256 acc50 = _mm256_setzero_ps(), acc51 = _mm256_setzero_ps();

				float *ptrC = fragmentC.data();
				const float *ptrA = fragmentA.data();
				const float *ptrB = fragmentB.data();

				for (int k = 0; k < K; k++)
				{
					__m256 b0 = _mm256_load_ps(ptrB);
					__m256 b1 = _mm256_load_ps(ptrB + 8);
					__m256 a0 = _mm256_broadcast_ss(ptrA);
					__m256 a1 = _mm256_broadcast_ss(ptrA + 1);

					acc00 = _mm256_fmadd_ps(a0, b0, acc00);
					acc01 = _mm256_fmadd_ps(a0, b1, acc01);
					acc10 = _mm256_fmadd_ps(a1, b0, acc10);
					acc11 = _mm256_fmadd_ps(a1, b1, acc11);

					a0 = _mm256_broadcast_ss(ptrA + 2);
					a1 = _mm256_broadcast_ss(ptrA + 3);

					acc20 = _mm256_fmadd_ps(a0, b0, acc20);
					acc21 = _mm256_fmadd_ps(a0, b1, acc21);
					acc30 = _mm256_fmadd_ps(a1, b0, acc30);
					acc31 = _mm256_fmadd_ps(a1, b1, acc31);

					a0 = _mm256_broadcast_ss(ptrA + 3);
					a1 = _mm256_broadcast_ss(ptrA + 4);

					acc40 = _mm256_fmadd_ps(a0, b0, acc40);
					acc41 = _mm256_fmadd_ps(a0, b1, acc41);
					acc50 = _mm256_fmadd_ps(a1, b0, acc50);
					acc51 = _mm256_fmadd_ps(a1, b1, acc51);
				}

				_mm256_store_ps(ptrC + 0, acc00);
				_mm256_store_ps(ptrC + 8, acc01);
				_mm256_store_ps(ptrC + 16, acc10);
				_mm256_store_ps(ptrC + 24, acc11);
				_mm256_store_ps(ptrC + 32, acc20);
				_mm256_store_ps(ptrC + 40, acc21);
				_mm256_store_ps(ptrC + 48, acc30);
				_mm256_store_ps(ptrC + 56, acc31);
				_mm256_store_ps(ptrC + 64, acc40);
				_mm256_store_ps(ptrC + 72, acc41);
				_mm256_store_ps(ptrC + 80, acc50);
				_mm256_store_ps(ptrC + 88, acc51);
			}
	};

	template<typename T, typename U>
	T cast_to(U x) noexcept
	{
		return static_cast<T>(x);
	}

	void gemm_def_MxN_fp32(int M, int N, int K, const void *alpha_ptr, const void *__restrict__ lhs_ptr, const void *__restrict__ rhs_ptr,
			const void *beta_ptr, void *__restrict__ dst_ptr, int dst_stride)
	{
		std::unique_ptr<float[]> acc = std::make_unique<float[]>(M * N);
		for (int i = 0; i < M * N; i++)
			acc[i] = 0.0f;

		for (int k = 0; k < K; k++)
			for (int m = 0; m < M; m++)
			{
				const float tmp = reinterpret_cast<const float*>(lhs_ptr)[k * M + m];
				for (int n = 0; n < N; n++)
					acc[m * N + n] += tmp * reinterpret_cast<const float*>(rhs_ptr)[k * N + n];
			}

		const float alpha = reinterpret_cast<const float*>(alpha_ptr)[0];
		if (alpha != 1.0f)
		{
			for (int i = 0; i < M * N; i++)
				acc[i] *= alpha;
		}

		const float beta = reinterpret_cast<const float*>(beta_ptr)[0];
		if (beta == 0.0f)
		{
			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					reinterpret_cast<float*>(dst_ptr)[m * dst_stride + n] = acc[m * N + n];
		}
		else
		{
			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					reinterpret_cast<float*>(dst_ptr)[m * dst_stride + n] = beta * reinterpret_cast<float*>(dst_ptr)[m * dst_stride + n]
							+ acc[m * N + n];
		}
	}

	__attribute__((noinline)) void gemm_avx2_fma_6x16_fp32(int M, int N, int K, const void *alpha_ptr, const void *__restrict__ lhs_ptr,
			const void *__restrict__ rhs_ptr, const void *beta_ptr, void *__restrict__ dst_ptr, int dst_stride)
	{
		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(lhs_ptr != nullptr);
		assert(rhs_ptr != nullptr);
		assert(dst_ptr != nullptr);
		assert(M == 6);
		assert(N == 16);
		assert(dst_stride > 0);

		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		uint64_t stride = dst_stride;

		asm volatile(
				"movq %[lhs_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[rhs_ptr], %%rbx \n\t"// rhs pointer is in rbx
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				// Set accumulators to zero.
				"vpxor %%ymm4, %%ymm4, %%ymm4 \n\t"
				"vpxor %%ymm5, %%ymm5, %%ymm5 \n\t"
				"vpxor %%ymm6, %%ymm6, %%ymm6 \n\t"
				"vpxor %%ymm7, %%ymm7, %%ymm7 \n\t"
				"vpxor %%ymm8, %%ymm8, %%ymm8 \n\t"
				"vpxor %%ymm9, %%ymm9, %%ymm9 \n\t"
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0

				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x40(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x60(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x18(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x20(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x28(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x80(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xA0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x38(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x40(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0xC0(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xE0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x48(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x50(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x54(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x58(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x5C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				"add $0x60, %%rax \n\t"
				"add $0x100, %%rbx \n\t"
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"
				"add $0x18, %%rax \n\t"
				"add $0x40, %%rbx \n\t"

				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm0, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm0, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm0, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm0, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm0, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"movq %[stride], %%r12 \n\t"// stride is r12
				"shlq $2, %%r12 \n\t"// multiply stride by sizeof(float)
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"ucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm4 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm6 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm7 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm8 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm9 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm10 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm11 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups 0x00(%%rcx), %%ymm2 \n\t"
				"vmovups 0x20(%%rcx), %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm14 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm15 \n\t"
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"BETAZERO%=: \n\t"
				"vmovups %%ymm4, 0x00(%%rcx) \n\t"
				"vmovups %%ymm5, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm6, 0x00(%%rcx) \n\t"
				"vmovups %%ymm7, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm8, 0x00(%%rcx) \n\t"
				"vmovups %%ymm9, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm10, 0x00(%%rcx) \n\t"
				"vmovups %%ymm11, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm12, 0x00(%%rcx) \n\t"
				"vmovups %%ymm13, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm14, 0x00(%%rcx) \n\t"
				"vmovups %%ymm15, 0x20(%%rcx) \n\t"

				"vzeroupper \n\t"

				:// outputs
				:// inputs
				[lhs_ptr] "m"(lhs_ptr),
				[rhs_ptr] "m"(rhs_ptr),
				[dst_ptr] "m"(dst_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[stride] "m"(stride),
				[alpha_ptr] "m"(alpha_ptr),
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
				"%r12", "%r13", "%r14");
	}

	__attribute__((noinline)) void gemm_avx_8x8_fp32(int M, int N, int K, const void *alpha_ptr, const void *__restrict__ lhs_ptr,
			const void *__restrict__ rhs_ptr, const void *beta_ptr, void *__restrict__ dst_ptr, int dst_stride)
	{
		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(lhs_ptr != nullptr);
		assert(rhs_ptr != nullptr);
		assert(dst_ptr != nullptr);
		assert(M == 8);
		assert(N == 8);
		assert(dst_stride > 0);

		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		uint64_t stride = dst_stride;

		asm volatile(
				"movq %[lhs_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[rhs_ptr], %%rbx \n\t"// rhs pointer is in rbx

				// Set accumulators to zero.
				"vpxor %%ymm8, %%ymm8, %%ymm8 \n\t"
				"vpxor %%ymm9, %%ymm9, %%ymm9 \n\t"
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rax), %%ymm0 \n\t"
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x20(%%rax), %%ymm0 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x40(%%rax), %%ymm0 \n\t"
				"vmovaps 0x40(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0x60(%%rax), %%ymm0 \n\t"
				"vmovaps 0x60(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"

				"add $0x80, %%rax \n\t"
				"add $0x80, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rax), %%ymm0 \n\t"
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"

				"vpermilps $0x00, %%ymm0, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm0, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm0, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm0, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"
				"vperm2f128 $0x03, %%ymm0, %%ymm0, %%ymm1 \n\t"

				"vaddps %%ymm4, %%ymm8, %%ymm8 \n\t"
				"vaddps %%ymm5, %%ymm9, %%ymm9 \n\t"
				"vaddps %%ymm6, %%ymm10, %%ymm10 \n\t"
				"vaddps %%ymm7, %%ymm11, %%ymm11 \n\t"

				"vpermilps $0x00, %%ymm1, %%ymm4 \n\t"
				"vpermilps $0x55, %%ymm1, %%ymm5 \n\t"
				"vpermilps $0xAA, %%ymm1, %%ymm6 \n\t"
				"vpermilps $0xFF, %%ymm1, %%ymm7 \n\t"

				"vmulps %%ymm4, %%ymm2, %%ymm4 \n\t"
				"vmulps %%ymm5, %%ymm2, %%ymm5 \n\t"
				"vmulps %%ymm6, %%ymm2, %%ymm6 \n\t"
				"vmulps %%ymm7, %%ymm2, %%ymm7 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm12 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm13 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm14 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm15 \n\t"
				"add $0x20, %%rax \n\t"
				"add $0x20, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				// permute back to row-najor storage
				"vperm2f128 $0x12, %%ymm8, %%ymm12, %%ymm0 \n\t"
				"vperm2f128 $0x30, %%ymm8, %%ymm12, %%ymm4 \n\t"
				"vperm2f128 $0x12, %%ymm9, %%ymm13, %%ymm1 \n\t"
				"vperm2f128 $0x30, %%ymm9, %%ymm13, %%ymm5 \n\t"
				"vperm2f128 $0x12, %%ymm10, %%ymm14, %%ymm2 \n\t"
				"vperm2f128 $0x30, %%ymm10, %%ymm14, %%ymm6 \n\t"
				"vperm2f128 $0x12, %%ymm11, %%ymm15, %%ymm3 \n\t"
				"vperm2f128 $0x30, %%ymm11, %%ymm15, %%ymm7 \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm8 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm9 \n\t"

				// scale by alpha
				"vmulps %%ymm8, %%ymm0, %%ymm0 \n\t"
				"vmulps %%ymm8, %%ymm1, %%ymm1 \n\t"
				"vmulps %%ymm8, %%ymm2, %%ymm2 \n\t"
				"vmulps %%ymm8, %%ymm3, %%ymm3 \n\t"
				"vmulps %%ymm8, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm8, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm8, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm8, %%ymm7, %%ymm7 \n\t"

				// load destination pointer and stride
				"movq %[stride], %%r12 \n\t"// stride is r12
				"shlq $2, %%r12 \n\t"// multiply stride by sizeof(float)
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"
				"ucomiss %%xmm9, %%xmm15 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// beta != 0 case
				"vmovups 0x00(%%rcx), %%ymm12 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm14 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm15 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmulps %%ymm9, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm9, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm9, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm9, %%ymm15, %%ymm15 \n\t"

				"vaddps %%ymm0, %%ymm12, %%ymm0 \n\t"
				"vaddps %%ymm1, %%ymm13, %%ymm1 \n\t"
				"vaddps %%ymm2, %%ymm14, %%ymm2 \n\t"
				"vaddps %%ymm3, %%ymm15, %%ymm3 \n\t"

				"vmovups 0x00(%%rcx), %%ymm12 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm14 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups 0x00(%%rcx), %%ymm15 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmulps %%ymm9, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm9, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm9, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm9, %%ymm15, %%ymm15 \n\t"

				"vaddps %%ymm4, %%ymm12, %%ymm4 \n\t"
				"vaddps %%ymm5, %%ymm13, %%ymm5 \n\t"
				"vaddps %%ymm6, %%ymm14, %%ymm6 \n\t"
				"vaddps %%ymm7, %%ymm15, %%ymm7 \n\t"

				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"BETAZERO%=: \n\t"
				// beta == 0 case
				"vmovups %%ymm0, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups %%ymm1, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups %%ymm2, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups %%ymm3, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups %%ymm4, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups %%ymm5, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups %%ymm6, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"vmovups %%ymm7, 0x00(%%rcx) \n\t"

				"vzeroupper \n\t"

				:// outputs
				:// inputs
				[lhs_ptr] "m"(lhs_ptr),
				[rhs_ptr] "m"(rhs_ptr),
				[dst_ptr] "m"(dst_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[stride] "m"(stride),
				[alpha_ptr] "m"(alpha_ptr),
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
				"%r12", "%r13", "%r14");
	}

	__attribute__((noinline)) void gemm_sse2_8x4_fp32(int M, int N, int K, const void *alpha_ptr, const void *__restrict__ lhs_ptr,
			const void *__restrict__ rhs_ptr, const void *beta_ptr, void *__restrict__ dst_ptr, int dst_stride)
	{
		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(lhs_ptr != nullptr);
		assert(rhs_ptr != nullptr);
		assert(dst_ptr != nullptr);
		assert(M == 8);
		assert(N == 4);
		assert(dst_stride > 0);

		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		uint64_t stride = dst_stride;

		asm volatile(
				"movq %[lhs_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[rhs_ptr], %%rbx \n\t"// rhs pointer is in rbx

				// Set accumulators to zero.
				"pxor %%xmm8, %%xmm8 \n\t"
				"pxor %%xmm9, %%xmm9 \n\t"
				"pxor %%xmm10, %%xmm10 \n\t"
				"pxor %%xmm11, %%xmm11 \n\t"
				"pxor %%xmm12, %%xmm12 \n\t"
				"pxor %%xmm13, %%xmm13 \n\t"
				"pxor %%xmm14, %%xmm14 \n\t"
				"pxor %%xmm15, %%xmm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"movaps 0x00(%%rax), %%xmm0 \n\t"
				"movaps 0x10(%%rax), %%xmm1 \n\t"
				"movaps 0x00(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				// iteration 1
				"movaps 0x20(%%rax), %%xmm0 \n\t"
				"movaps 0x30(%%rax), %%xmm1 \n\t"
				"movaps 0x10(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				// iteration 2
				"movaps 0x40(%%rax), %%xmm0 \n\t"
				"movaps 0x50(%%rax), %%xmm1 \n\t"
				"movaps 0x20(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				// iteration 3
				"movaps 0x60(%%rax), %%xmm0 \n\t"
				"movaps 0x70(%%rax), %%xmm1 \n\t"
				"movaps 0x30(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				"add $0x80, %%rax \n\t"
				"add $0x40, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"movaps 0x00(%%rax), %%xmm0 \n\t"
				"movaps 0x10(%%rax), %%xmm1 \n\t"
				"movaps 0x00(%%rbx), %%xmm2 \n\t"

				"pshufd $0x00, %%xmm0, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm0, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm0, %%xmm7 \n\t"

				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"pshufd $0x00, %%xmm1, %%xmm4 \n\t"
				"pshufd $0x55, %%xmm1, %%xmm5 \n\t"
				"pshufd $0xAA, %%xmm1, %%xmm6 \n\t"
				"pshufd $0xFF, %%xmm1, %%xmm7 \n\t"
				"mulps %%xmm2, %%xmm4 \n\t"
				"mulps %%xmm2, %%xmm5 \n\t"
				"mulps %%xmm2, %%xmm6 \n\t"
				"mulps %%xmm2, %%xmm7 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"

				"add $0x20, %%rax \n\t"
				"add $0x10, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"movss 0x00(%%rax), %%xmm0 \n\t"
				"movss 0x00(%%rbx), %%xmm1 \n\t"
				"pshufd $0x00, %%xmm0, %%xmm0 \n\t"
				"pshufd $0x00, %%xmm1, %%xmm1 \n\t"

				// scale by alpha
				"mulps %%xmm0, %%xmm8 \n\t"
				"mulps %%xmm0, %%xmm9 \n\t"
				"mulps %%xmm0, %%xmm10 \n\t"
				"mulps %%xmm0, %%xmm11 \n\t"
				"mulps %%xmm0, %%xmm12 \n\t"
				"mulps %%xmm0, %%xmm13 \n\t"
				"mulps %%xmm0, %%xmm14 \n\t"
				"mulps %%xmm0, %%xmm15 \n\t"

				// load destination pointer and stride
				"movq %[stride], %%r12 \n\t"// stride is r12
				"shlq $2, %%r12 \n\t"// multiply stride by sizeof(float)
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"pxor %%xmm0, %%xmm0 \n\t"
				"ucomiss %%xmm1, %%xmm0 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// beta != 0 case
				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm7 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"mulps %%xmm1, %%xmm4 \n\t"
				"mulps %%xmm1, %%xmm5 \n\t"
				"mulps %%xmm1, %%xmm6 \n\t"
				"mulps %%xmm1, %%xmm7 \n\t"

				"addps %%xmm4, %%xmm8 \n\t"
				"addps %%xmm5, %%xmm9 \n\t"
				"addps %%xmm6, %%xmm10 \n\t"
				"addps %%xmm7, %%xmm11 \n\t"

				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm7 \n\t"

				"mulps %%xmm1, %%xmm4 \n\t"
				"mulps %%xmm1, %%xmm5 \n\t"
				"mulps %%xmm1, %%xmm6 \n\t"
				"mulps %%xmm1, %%xmm7 \n\t"

				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm6, %%xmm14 \n\t"
				"addps %%xmm7, %%xmm15 \n\t"
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"BETAZERO%=: \n\t"
				// beta == 0 case
				"movups %%xmm8, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm9, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm10, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm11, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm12, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm13, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm14, 0x00(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups %%xmm15, 0x00(%%rcx) \n\t"

				:// outputs
				:// inputs
				[lhs_ptr] "m"(lhs_ptr),
				[rhs_ptr] "m"(rhs_ptr),
				[dst_ptr] "m"(dst_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[stride] "m"(stride),
				[alpha_ptr] "m"(alpha_ptr),
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
				"%r12", "%r13", "%r14");
	}

	__attribute__((noinline)) void gemm_avx2_fma_6x16_fp16_fp32(int M, int N, int K, const void *alpha_ptr, const void *__restrict__ lhs_ptr,
			const void *__restrict__ rhs_ptr, const void *beta_ptr, void *__restrict__ dst_ptr, int dst_stride, void *__restrict__ workspace)
	{
		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(lhs_ptr != nullptr);
		assert(rhs_ptr != nullptr);
		assert(dst_ptr != nullptr);
		assert(M == 6);
		assert(N == 16);
		assert(dst_stride > 0);

		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		uint64_t stride = dst_stride;
		float alpha = reinterpret_cast<const float*>(alpha_ptr)[0];
		float beta = reinterpret_cast<const float*>(beta_ptr)[0];
		if (beta != 0.0f)
			alpha /= beta;
		const float *_alpha = &alpha;

		asm volatile(
				"movq %[beta_ptr], %%rbx \n\t" // load address of beta
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"
				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"ucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je ZEROACC%= \n\t"
				// load and convert dst
				"movq %[stride], %%r12 \n\t"// stride is r12
				"shlq $1, %%r12 \n\t"// multiply stride by sizeof(float16)
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"movups 0x00(%%rcx), %%xmm4 \n\t"
				"movups 0x10(%%rcx), %%xmm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm6 \n\t"
				"movups 0x10(%%rcx), %%xmm7 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm8 \n\t"
				"movups 0x10(%%rcx), %%xmm9 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm10 \n\t"
				"movups 0x10(%%rcx), %%xmm11 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm12 \n\t"
				"movups 0x10(%%rcx), %%xmm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride
				"movups 0x00(%%rcx), %%xmm14 \n\t"
				"movups 0x10(%%rcx), %%xmm15 \n\t"

				"vcvtph2ps %%xmm4, %%ymm4 \n\t"
				"vcvtph2ps %%xmm5, %%ymm5 \n\t"
				"vcvtph2ps %%xmm6, %%ymm6 \n\t"
				"vcvtph2ps %%xmm7, %%ymm7 \n\t"
				"vcvtph2ps %%xmm8, %%ymm8 \n\t"
				"vcvtph2ps %%xmm9, %%ymm9 \n\t"
				"vcvtph2ps %%xmm10, %%ymm10 \n\t"
				"vcvtph2ps %%xmm11, %%ymm11 \n\t"
				"vcvtph2ps %%xmm12, %%ymm12 \n\t"
				"vcvtph2ps %%xmm13, %%ymm13 \n\t"
				"vcvtph2ps %%xmm14, %%ymm14 \n\t"
				"vcvtph2ps %%xmm15, %%ymm15 \n\t"

				"vmulps %%ymm1, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm1, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm1, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm1, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm1, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm1, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm1, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm1, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm1, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm1, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm1, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm1, %%ymm15, %%ymm15 \n\t"
				"jmp LOOPSTART%= \n\t"

				"ZEROACC%=: \n\t"
				// Set accumulators to zero.
				"vpxor %%ymm4, %%ymm4, %%ymm4 \n\t"
				"vpxor %%ymm5, %%ymm5, %%ymm5 \n\t"
				"vpxor %%ymm6, %%ymm6, %%ymm6 \n\t"
				"vpxor %%ymm7, %%ymm7, %%ymm7 \n\t"
				"vpxor %%ymm8, %%ymm8, %%ymm8 \n\t"
				"vpxor %%ymm9, %%ymm9, %%ymm9 \n\t"
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"LOOPSTART%=: \n\t"

				"movq %[lhs_ptr], %%rax \n\t"// lhs pointer is in rax
				"movq %[rhs_ptr], %%rbx \n\t"// rhs pointer is in rbx
				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0

				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 1
				"vmovaps 0x40(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x60(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x18(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x20(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x28(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 2
				"vmovaps 0x80(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xA0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x38(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x40(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				// iteration 3
				"vmovaps 0xC0(%%rbx), %%ymm2 \n\t"
				"vmovaps 0xE0(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x48(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x50(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x54(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x58(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x5C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"

				"add $0x60, %%rax \n\t"
				"add $0x100, %%rbx \n\t"
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm4 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm5 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm7 \n\t"

				"vbroadcastss 0x08(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm11 \n\t"

				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm0, %%ymm2, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm3, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm2, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm3, %%ymm15 \n\t"
				"add $0x18, %%rax \n\t"
				"add $0x40, %%rbx \n\t"

				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm4, %%ymm4 \n\t"
				"vmulps %%ymm0, %%ymm5, %%ymm5 \n\t"
				"vmulps %%ymm0, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm0, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm0, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm0, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"movq %[stride], %%r12 \n\t"// stride is r12
				"shlq $1, %%r12 \n\t"// multiply stride by sizeof(float16)
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"vcvtps2ph $0x03, %%ymm4, %%xmm4 \n\t"
				"vcvtps2ph $0x03, %%ymm5, %%xmm5 \n\t"
				"vcvtps2ph $0x03, %%ymm6, %%xmm6 \n\t"
				"vcvtps2ph $0x03, %%ymm7, %%xmm7 \n\t"
				"vcvtps2ph $0x03, %%ymm8, %%xmm8 \n\t"
				"vcvtps2ph $0x03, %%ymm9, %%xmm9 \n\t"
				"vcvtps2ph $0x03, %%ymm10, %%xmm10 \n\t"
				"vcvtps2ph $0x03, %%ymm11, %%xmm11 \n\t"
				"vcvtps2ph $0x03, %%ymm12, %%xmm12 \n\t"
				"vcvtps2ph $0x03, %%ymm13, %%xmm13 \n\t"
				"vcvtps2ph $0x03, %%ymm14, %%xmm14 \n\t"
				"vcvtps2ph $0x03, %%ymm15, %%xmm15 \n\t"

				"movups %%xmm4, 0x00(%%rcx) \n\t"
				"movups %%xmm5, 0x10(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm6, 0x00(%%rcx) \n\t"
				"movups %%xmm7, 0x10(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm8, 0x00(%%rcx) \n\t"
				"movups %%xmm9, 0x10(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm10, 0x00(%%rcx) \n\t"
				"movups %%xmm11, 0x10(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm12, 0x00(%%rcx) \n\t"
				"movups %%xmm13, 0x10(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups %%xmm14, 0x00(%%rcx) \n\t"
				"movups %%xmm15, 0x10(%%rcx) \n\t"

				"vzeroupper \n\t"

				:// outputs
				:// inputs
				[lhs_ptr] "m"(lhs_ptr),
				[rhs_ptr] "m"(rhs_ptr),
				[dst_ptr] "m"(dst_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[stride] "m"(stride),
				[alpha_ptr] "m"(_alpha),
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
				"%r12", "%r13", "%r14");
	}

	__attribute__((noinline)) void gemm_avx2_fma_5x16_fp32(int M, int N, int K, const void *alpha_ptr, const void *__restrict__ lhs_ptr,
			const void *__restrict__ rhs_ptr, const void *beta_ptr, void *__restrict__ dst_ptr, int dst_stride)
	{
		assert(alpha_ptr != nullptr);
		assert(beta_ptr != nullptr);
		assert(lhs_ptr != nullptr);
		assert(rhs_ptr != nullptr);
		assert(dst_ptr != nullptr);
		assert(M == 5);
		assert(N == 16);
		assert(dst_stride > 0);

		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		uint64_t stride = dst_stride;

		asm volatile(
				"movq %[lhs_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq %[rhs_ptr], %%rbx \n\t"// rhs pointer is in rbx

				// Set accumulators to zero.
				"vpxor %%ymm6, %%ymm6, %%ymm6 \n\t"
				"vpxor %%ymm7, %%ymm7, %%ymm7 \n\t"
				"vpxor %%ymm8, %%ymm8, %%ymm8 \n\t"
				"vpxor %%ymm9, %%ymm9, %%ymm9 \n\t"
				"vpxor %%ymm10, %%ymm10, %%ymm10 \n\t"
				"vpxor %%ymm11, %%ymm11, %%ymm11 \n\t"
				"vpxor %%ymm12, %%ymm12, %%ymm12 \n\t"
				"vpxor %%ymm13, %%ymm13, %%ymm13 \n\t"
				"vpxor %%ymm14, %%ymm14, %%ymm14 \n\t"
				"vpxor %%ymm15, %%ymm15, %%ymm15 \n\t"

				"movq %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je FINALLOOP%= \n\t"

				"UNROLLED4%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm4 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm5 \n\t"

				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vfmadd231ps %%ymm0, %%ymm4, %%ymm6 \n\t"
				"vfmadd231ps %%ymm0, %%ymm5, %%ymm7 \n\t"
				"vfmadd231ps %%ymm1, %%ymm4, %%ymm8 \n\t"
				"vfmadd231ps %%ymm1, %%ymm5, %%ymm9 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x14(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm2, %%ymm4, %%ymm10 \n\t"
				"vfmadd231ps %%ymm2, %%ymm5, %%ymm11 \n\t"
				"vfmadd231ps %%ymm3, %%ymm4, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm5, %%ymm13 \n\t"
				"vbroadcastss 0x18(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x1C(%%rax), %%ymm3 \n\t"
				"vfmadd231ps %%ymm0, %%ymm4, %%ymm14 \n\t"
				"vfmadd231ps %%ymm0, %%ymm5, %%ymm15 \n\t"

				"vmovaps 0x40(%%rbx), %%ymm4 \n\t"
				"vmovaps 0x60(%%rbx), %%ymm5 \n\t"

				"vfmadd231ps %%ymm1, %%ymm4, %%ymm6 \n\t"
				"vfmadd231ps %%ymm1, %%ymm5, %%ymm7 \n\t"
				"vbroadcastss 0x20(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x24(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm2, %%ymm4, %%ymm8 \n\t"
				"vfmadd231ps %%ymm2, %%ymm5, %%ymm9 \n\t"
				"vfmadd231ps %%ymm3, %%ymm4, %%ymm10 \n\t"
				"vfmadd231ps %%ymm3, %%ymm5, %%ymm11 \n\t"
				"vbroadcastss 0x28(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x2C(%%rax), %%ymm3 \n\t"

				"vfmadd231ps %%ymm0, %%ymm4, %%ymm12 \n\t"
				"vfmadd231ps %%ymm0, %%ymm5, %%ymm13 \n\t"
				"vfmadd231ps %%ymm1, %%ymm4, %%ymm14 \n\t"
				"vfmadd231ps %%ymm1, %%ymm5, %%ymm15 \n\t"

				"vbroadcastss 0x30(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x34(%%rax), %%ymm1 \n\t"
				"vmovaps 0x80(%%rbx), %%ymm4 \n\t"
				"vmovaps 0xA0(%%rbx), %%ymm5 \n\t"

				"vfmadd231ps %%ymm2, %%ymm4, %%ymm6 \n\t"
				"vfmadd231ps %%ymm2, %%ymm5, %%ymm7 \n\t"
				"vfmadd231ps %%ymm3, %%ymm4, %%ymm8 \n\t"
				"vfmadd231ps %%ymm3, %%ymm5, %%ymm9 \n\t"

				"vbroadcastss 0x38(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x3C(%%rax), %%ymm3 \n\t"
				"vfmadd231ps %%ymm0, %%ymm4, %%ymm10 \n\t"
				"vfmadd231ps %%ymm0, %%ymm5, %%ymm11 \n\t"
				"vfmadd231ps %%ymm1, %%ymm4, %%ymm12 \n\t"
				"vfmadd231ps %%ymm1, %%ymm5, %%ymm13 \n\t"
				"vbroadcastss 0x40(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x44(%%rax), %%ymm1 \n\t"
				"vfmadd231ps %%ymm2, %%ymm4, %%ymm14 \n\t"
				"vfmadd231ps %%ymm2, %%ymm5, %%ymm15 \n\t"

				"vmovaps 0xC0(%%rbx), %%ymm4 \n\t"
				"vmovaps 0xE0(%%rbx), %%ymm5 \n\t"

				"vfmadd231ps %%ymm3, %%ymm4, %%ymm6 \n\t"
				"vfmadd231ps %%ymm3, %%ymm5, %%ymm7 \n\t"

				"vbroadcastss 0x48(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x4C(%%rax), %%ymm3 \n\t"
				"vfmadd231ps %%ymm0, %%ymm4, %%ymm8 \n\t"
				"vfmadd231ps %%ymm0, %%ymm5, %%ymm9 \n\t"
				"vfmadd231ps %%ymm1, %%ymm4, %%ymm10 \n\t"
				"vfmadd231ps %%ymm1, %%ymm5, %%ymm11 \n\t"
				"vfmadd231ps %%ymm2, %%ymm4, %%ymm12 \n\t"
				"vfmadd231ps %%ymm2, %%ymm5, %%ymm13 \n\t"
				"vfmadd231ps %%ymm3, %%ymm4, %%ymm14 \n\t"
				"vfmadd231ps %%ymm3, %%ymm5, %%ymm15 \n\t"

				"add $0x50, %%rax \n\t"
				"add $0x100, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED4%= \n\t"

				"FINALLOOP%=: \n\t"
				"movq %[k_left], %%r14 \n\t"// load the number of 1-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je EPILOGUE%= \n\t"

				"UNROLLED1%=: \n\t"
				// iteration 0
				"vmovaps 0x00(%%rbx), %%ymm2 \n\t"
				"vmovaps 0x20(%%rbx), %%ymm3 \n\t"

				"vbroadcastss 0x00(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x04(%%rax), %%ymm1 \n\t"
				"vbroadcastss 0x08(%%rax), %%ymm2 \n\t"
				"vbroadcastss 0x0C(%%rax), %%ymm3 \n\t"
				"vfmadd231ps %%ymm0, %%ymm4, %%ymm6 \n\t"
				"vfmadd231ps %%ymm0, %%ymm5, %%ymm7 \n\t"
				"vbroadcastss 0x10(%%rax), %%ymm0 \n\t"
				"vfmadd231ps %%ymm1, %%ymm4, %%ymm8 \n\t"
				"vfmadd231ps %%ymm1, %%ymm5, %%ymm9 \n\t"
				"vfmadd231ps %%ymm2, %%ymm4, %%ymm10 \n\t"
				"vfmadd231ps %%ymm2, %%ymm5, %%ymm11 \n\t"
				"vfmadd231ps %%ymm3, %%ymm4, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm5, %%ymm13 \n\t"
				"vfmadd231ps %%ymm0, %%ymm4, %%ymm14 \n\t"
				"vfmadd231ps %%ymm0, %%ymm5, %%ymm15 \n\t"

				"add $0x14, %%rax \n\t"
				"add $0x40, %%rbx \n\t"
				"dec %%r14 \n\t"
				"jne UNROLLED1%= \n\t"

				"EPILOGUE%=: \n\t"

				"movq %[alpha_ptr], %%rax \n\t"// load address of alpha
				"movq %[beta_ptr], %%rbx \n\t"// load address of beta
				"vbroadcastss 0x0(%%rax), %%ymm0 \n\t"
				"vbroadcastss 0x0(%%rbx), %%ymm1 \n\t"

				// scale by alpha
				"vmulps %%ymm0, %%ymm6, %%ymm6 \n\t"
				"vmulps %%ymm0, %%ymm7, %%ymm7 \n\t"
				"vmulps %%ymm0, %%ymm8, %%ymm8 \n\t"
				"vmulps %%ymm0, %%ymm9, %%ymm9 \n\t"
				"vmulps %%ymm0, %%ymm10, %%ymm10 \n\t"
				"vmulps %%ymm0, %%ymm11, %%ymm11 \n\t"
				"vmulps %%ymm0, %%ymm12, %%ymm12 \n\t"
				"vmulps %%ymm0, %%ymm13, %%ymm13 \n\t"
				"vmulps %%ymm0, %%ymm14, %%ymm14 \n\t"
				"vmulps %%ymm0, %%ymm15, %%ymm15 \n\t"

				// load destination pointer and stride
				"movq %[stride], %%r12 \n\t"// stride is r12
				"shlq $2, %%r12 \n\t"// multiply stride by sizeof(float)
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"ucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// beta != 0 case
				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm6 \n\t"
				"vmovups %%ymm6, 0x00(%%rcx) \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm7 \n\t"
				"vmovups %%ymm7, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm8 \n\t"
				"vmovups %%ymm8, 0x00(%%rcx) \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm9 \n\t"
				"vmovups %%ymm9, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm10 \n\t"
				"vmovups %%ymm10, 0x00(%%rcx) \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm11 \n\t"
				"vmovups %%ymm11, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm12 \n\t"
				"vmovups %%ymm12, 0x00(%%rcx) \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm13 \n\t"
				"vmovups %%ymm13, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm14 \n\t"
				"vmovups %%ymm14, 0x00(%%rcx) \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm15 \n\t"
				"vmovups %%ymm15, 0x20(%%rcx) \n\t"

				"jmp END%= \n\t"// jump to end.

				"BETAZERO%=: \n\t"
				// beta == 0 case
				"vmovups %%ymm6, 0x00(%%rcx) \n\t"
				"vmovups %%ymm7, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm8, 0x00(%%rcx) \n\t"
				"vmovups %%ymm9, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm10, 0x00(%%rcx) \n\t"
				"vmovups %%ymm11, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm12, 0x00(%%rcx) \n\t"
				"vmovups %%ymm13, 0x20(%%rcx) \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vmovups %%ymm14, 0x00(%%rcx) \n\t"
				"vmovups %%ymm15, 0x20(%%rcx) \n\t"

				"END%=: \n\t"
				"vzeroupper \n\t"

				:// outputs
				:// inputs
				[lhs_ptr] "m"(lhs_ptr),
				[rhs_ptr] "m"(rhs_ptr),
				[dst_ptr] "m"(dst_ptr),
				[k_iter] "m"(k_iter),
				[k_left] "m"(k_left),
				[stride] "m"(stride),
				[alpha_ptr] "m"(alpha_ptr),
				[beta_ptr] "m"(beta_ptr)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%rax", "%rbx", "%rcx",
				"%r12", "%r13", "%r14");
	}

	template<int TileM, int TileN, typename DataType, typename ComputeType = DataType>
	struct GemmKernel
	{
			int total_fma_count = 0;
			~GemmKernel()
			{
				std::cout << "total fma count    = " << total_fma_count << '\n';
			}
			void run(DataType *__restrict__ C, const DataType *__restrict__ A, const DataType *__restrict__ B, int M, int N, int K)
			{
				ComputeType acc[M * N];
				for (int i = 0; i < M * N; i++)
					acc[i] = cast_to<ComputeType>(0.0);

				for (int k = 0; k < K; k++)
					for (int m = 0; m < M; m++)
					{
						const DataType tmp = cast_to<ComputeType>(A[k * M + m]);
						for (int n = 0; n < N; n++, total_fma_count++)
							acc[m * N + n] += tmp * cast_to<ComputeType>(B[k * N + n]);
					}

				for (int m = 0; m < M; m++)
					for (int n = 0; n < N; n++)
						C[m * N + n] += acc[m * N + n];
			}
	};

	template<typename T>
	struct MicroKernel
	{
			int total_fma_count = 0;
			~MicroKernel()
			{
				std::cout << "total fma count    = " << total_fma_count << '\n';
			}
			void run(T *__restrict__ C, const T *__restrict__ A, const T *__restrict__ B, const int M, const int N, const int K)
			{
				T acc[M * N];
				for (int i = 0; i < M * N; i++)
					acc[i] = static_cast<T>(0.0);

				for (int k = 0; k < K; k++)
					for (int m = 0; m < M; m++)
					{
						const T tmp = A[k * M + m];
						for (int n = 0; n < N; n++, total_fma_count++)
							acc[m * N + n] += tmp * B[k * N + n];
					}

				for (int m = 0; m < M; m++)
					for (int n = 0; n < N; n++)
						C[m * N + n] += acc[m * N + n];
			}
	};
}

#include "../src/backend/cpu/gemm/utilities.hpp"
#include "../src/backend/cpu/gemm/Fragment.hpp"
#include "../src/backend/cpu/gemm/Matrix.hpp"
#include "../src/backend/cpu/gemm/gemm_kernels.hpp"
#include "../src/backend/cpu/indexers.hpp"
#include "../src/backend/cpu/winograd/winograd_kernels.hpp"
#include "../src/backend/cpu/fp16.hpp"
#include "minml/utils/testing_util.hpp"

void test_packing(int rows, int columns, ml::MatrixOp op, mlDataType_t dtype,
		std::function<void(Fragment&, const Matrix&, const Position2D&, MatrixOp)> kernel, bool benchmark)
{
	const int M = columns;
	const int K = rows;

	std::unique_ptr<uint8_t[]> src = std::make_unique<uint8_t[]>(4 * 1024 * 1024);
	for (int i = 0; i < 1024 * 1024; i++)
		if (dtype == DTYPE_FLOAT32)
			reinterpret_cast<float*>(src.get())[i] = (i / 1024) + (i % 1024) * 0.01f;
		else
			reinterpret_cast<uint16_t*>(src.get())[i] = float_to_half((i / 1024) + (i % 1024) * 0.01f);

	const size_t size_in_bytes = sizeof(float) * M * K;
	void *dst1 = gemm::aligned_new(size_in_bytes, 4096);
	void *dst2 = gemm::aligned_new(size_in_bytes, 4096);
	std::memset(dst1, 0, size_in_bytes);
	std::memset(dst2, 0, size_in_bytes);

	Fragment correct(dst1, DTYPE_FLOAT32, M);
	Fragment fragment(dst2, DTYPE_FLOAT32, M);
	correct.mark_as_packed_with_size( { K, M });
	fragment.mark_as_packed_with_size( { K, M });

	Matrix matrix(src.get(), dtype, 1024, 1024, 1024);

	pack_def_MxK(correct, matrix, { 0, 0 }, op);

	kernel(fragment, matrix, { 0, 0 }, op);

	if (benchmark)
	{
		const double repeats = 1.0e8;
		const double start = getTime();
		int i = 0;
		for (; i < repeats; i++)
		{
			kernel(fragment, matrix, { 0, 0 }, op);
			if ((getTime() - start) > 10.0)
				break;
		}
		const double stop = getTime();
		std::cout << M << "x" << K << " : " << 1.0e6 * (stop - start) / i << " us (" << i << " repeats)\n";
	}
	else
	{
		double diff = 0.0;
		double max_diff = 0.0;
		for (int k = 0; k < K; k++)
			for (int m = 0; m < M; m++)
			{
				double tmp = std::fabs(correct.at<float>(k, m) - fragment.at<float>(k, m));
				diff += tmp;
				max_diff = std::max(max_diff, tmp);
			}
		if (diff > 1.0e-3 or max_diff > 1.0e-3)
		{
			std::cout << "rows = " << rows << ", columns = " << columns << ", transpose = " << (int) op << ": FAILED\n";
			std::cout << "diff = " << diff << '\n';
			std::cout << "max_diff = " << max_diff << '\n';
			std::cout << "Correct\n";
			for (int k = 0; k < K; k++)
			{
				for (int m = 0; m < M; m++)
					std::cout << correct.at<float>(k, m) << ' ';
				std::cout << '\n';
			}
			std::cout << "-------------------------------------------\n";
			std::cout << "Actual\n";
			for (int k = 0; k < K; k++)
			{
				for (int m = 0; m < M; m++)
					std::cout << fragment.at<float>(k, m) << ' ';
				std::cout << '\n';
			}
			exit(255);
		}
		else
			std::cout << "rows = " << rows << ", columns = " << columns << ", transpose = " << (int) op << " : OK\n";
	}

	gemm::aligned_free(dst1, 4096);
	gemm::aligned_free(dst2, 4096);
}

void test_microkernel(const int M, const int N, const int K, mlDataType_t dtype,
		std::function<void(Fragment&, const Fragment&, const Fragment&, const Fragment&, const void*, const Fragment&, const Fragment&, bool)> kernel)
{
	std::unique_ptr<float[]> matrix_c = std::make_unique<float[]>(1024 * 1024);
	std::unique_ptr<float[]> matrix_d = std::make_unique<float[]>(1024 * 1024);
	std::unique_ptr<float[]> correct_d = std::make_unique<float[]>(1024 * 1024);

	const size_t size_in_bytes_lhs = sizeof(float) * M * K;
	const size_t size_in_bytes_rhs = sizeof(float) * N * K;
	void *lhs = gemm::aligned_new(size_in_bytes_lhs, 4096);
	void *rhs = gemm::aligned_new(size_in_bytes_rhs, 4096);
	void *b_ptr = gemm::aligned_new(sizeof(float) * 1024, 4096);
	void *alpha_ptr = gemm::aligned_new(sizeof(float) * 1024, 4096);
	std::memset(lhs, 0, size_in_bytes_lhs);
	std::memset(rhs, 0, size_in_bytes_rhs);
	std::memset(b_ptr, 0, sizeof(float) * 1024);
	std::memset(alpha_ptr, 0, sizeof(float) * 1024);

	Fragment fragment_a(lhs, DTYPE_FLOAT32, M);
	Fragment fragment_b(rhs, DTYPE_FLOAT32, N);
	Fragment fragment_c(matrix_c.get(), dtype, 1024);
	Fragment fragment_d(matrix_d.get(), dtype, 1024);
	Fragment correct_fragment_d(correct_d.get(), dtype, 1024);
	Fragment fragment_bias(b_ptr, DTYPE_FLOAT32, N);
	Fragment fragment_alpha(alpha_ptr, DTYPE_FLOAT32, 1);
	fragment_a.mark_as_packed_with_size( { K, M });
	fragment_b.mark_as_packed_with_size( { K, N });
	fragment_c.mark_as_packed_with_size( { M, N });
	fragment_d.mark_as_packed_with_size( { M, N });
	correct_fragment_d.mark_as_packed_with_size( { M, N });
//	fragment_bias.mark_as_packed_with_size( { 1, N });
	fragment_alpha.mark_as_packed_with_size( { M, 1 });

	if (fragment_c.dtype() == DTYPE_FLOAT16)
		for (int m = 0; m < M; m++)
			for (int n = 0; n < N; n++)
				fragment_c.at<uint16_t>(m, n) = float_to_half(randFloat() - 0.5f);
	else
		for (int m = 0; m < M; m++)
			for (int n = 0; n < N; n++)
				fragment_c.at<float>(m, n) = randFloat() - 0.5f;

	for (int k = 0; k < K; k++)
	{
		for (int m = 0; m < M; m++)
			fragment_a.at<float>(k, m) = randFloat() - 0.5f;
		for (int n = 0; n < N; n++)
			fragment_b.at<float>(k, n) = randFloat() - 0.5f;
	}

	for (int n = 0; n < N; n++)
		fragment_bias.at<float>(0, n) = randFloat() - 0.5f;

	const float alpha = 1.0f;
	const float beta = 0.0f;
	const bool use_relu = false;
	if (fragment_alpha.rows() == 1)
		fragment_alpha.at<float>(0, 0) = alpha;
	else
	{
		for (int m = 0; m < M; m++)
			fragment_alpha.at<float>(m, 0) = m + randFloat();
	}

//	if (fragment_c.dtype() == DTYPE_FLOAT32)
//		gemm_def_MxN_fp32(correct_fragment_d, &alpha, fragment_a, fragment_b, &beta, fragment_c, fragment_bias, use_relu);
//	if (fragment_c.dtype() == DTYPE_FLOAT16)
//		gemm_def_MxN_fp32_fp16(correct_fragment_d, &alpha, fragment_a, fragment_b, &beta, fragment_c, fragment_bias, use_relu);

//	gemm_def_MxN(correct_fragment_d, &alpha, fragment_a, fragment_b, &beta, fragment_c, fragment_bias, use_relu);
//	kernel(fragment_d, &alpha, fragment_a, fragment_b, &beta, fragment_c, fragment_bias, use_relu);

	gemm_def_MxN(correct_fragment_d, fragment_alpha, fragment_a, fragment_b, &beta, fragment_c, fragment_bias, use_relu);
	kernel(fragment_d, fragment_alpha, fragment_a, fragment_b, &beta, fragment_c, fragment_bias, use_relu);

//	double diff = 0.0;
//	for (int m = 0; m < M; m++)
//		for (int n = 0; n < N; n++)
//			if (fragment_d.dtype() == DTYPE_FLOAT16)
//				diff += std::fabs(half_to_float(correct_fragment_d.at<uint16_t>(m, n)) - half_to_float(fragment_d.at<uint16_t>(m, n)));
//			else
//				diff += std::fabs(correct_fragment_d.at<float>(m, n) - fragment_d.at<float>(m, n));
//	diff /= (M * N);
//	if (diff > 1.0e-4)
//	{
//		std::cout << "Correct\n";
//		for (int m = 0; m < M; m++)
//		{
//			for (int n = 0; n < N; n++)
//				if (correct_fragment_d.dtype() == DTYPE_FLOAT16)
//					std::cout << half_to_float(correct_fragment_d.at<uint16_t>(m, n)) << ' ';
//				else
//					std::cout << correct_fragment_d.at<float>(m, n) << ' ';
//			std::cout << '\n';
//		}
//		std::cout << "-------------------------------------------\n";
//
//		std::cout << "Actual\n";
//		for (int m = 0; m < M; m++)
//		{
//			for (int n = 0; n < N; n++)
//				if (fragment_d.dtype() == DTYPE_FLOAT16)
//					std::cout << half_to_float(fragment_d.at<uint16_t>(m, n)) << ' ';
//				else
//					std::cout << fragment_d.at<float>(m, n) << ' ';
//			std::cout << '\n';
//		}
//
//		std::cout << "\ndiff = " << diff << '\n';
//		exit(255);
//	}
//	else
//		std::cout << "gemm kernel = " << M << "x" << N << "x" << K << " : OK\n";

	const double repeats = 1.0e8;
	const double start = getTime();
	int i = 0;
	for (; i < repeats; i++)
	{
		kernel(fragment_d, fragment_alpha, fragment_a, fragment_b, &beta, fragment_c, fragment_bias, use_relu);
		if ((getTime() - start) > 10.0)
			break;
	}
	const double stop = getTime();
	const double flops = (double) i * (M * N * K) / (stop - start);
	std::cout << M << "x" << N << "x" << K << " : " << flops / 1.0e9 << " GFLOPS\n";
	std::cout << "time = " << (stop - start) / (i / 1.0e6) << "us\n";

	gemm::aligned_free(lhs, 4096);
	gemm::aligned_free(rhs, 4096);
	gemm::aligned_free(b_ptr, 4096);
	gemm::aligned_free(alpha_ptr, 4096);
}

void test_mha_kernel(const int M, const int N, const int K,
		std::function<void(Fragment&, const void*, const Fragment&, const Fragment&, const Fragment&, Fragment&)> kernel)
{
	void *matrix_qk = gemm::aligned_new(1024 * 1024, 4096);
	void *correct_qk = gemm::aligned_new(1024 * 1024, 4096);

	const size_t size_in_bytes_lhs = sizeof(float) * M * K;
	const size_t size_in_bytes_rhs = sizeof(float) * N * K;
	void *lhs = gemm::aligned_new(size_in_bytes_lhs, 4096);
	void *rhs = gemm::aligned_new(size_in_bytes_rhs, 4096);
	void *bias_ptr = gemm::aligned_new(sizeof(float) * 1024, 4096);
	void *softmax_ptr = gemm::aligned_new(sizeof(float) * 1024, 4096);
	void *correct_softmax_ptr = gemm::aligned_new(sizeof(float) * 1024, 4096);
	std::memset(lhs, 0, size_in_bytes_lhs);
	std::memset(rhs, 0, size_in_bytes_rhs);
	std::memset(bias_ptr, 0, sizeof(float) * 1024);
	std::memset(softmax_ptr, 0, sizeof(float) * 1024);
	std::memset(correct_softmax_ptr, 0, sizeof(float) * 1024);

	Fragment fragment_a(lhs, DTYPE_FLOAT32, M);
	Fragment fragment_b(rhs, DTYPE_FLOAT32, N);
	Fragment fragment_qk(matrix_qk, DTYPE_FLOAT32, M);
	Fragment correct_fragment_qk(correct_qk, DTYPE_FLOAT32, M);
	Fragment fragment_bias(bias_ptr, DTYPE_FLOAT32, N);
	Fragment fragment_softmax(softmax_ptr, DTYPE_FLOAT32, 1);
	Fragment correct_fragment_softmax(correct_softmax_ptr, DTYPE_FLOAT32, 1);

	fragment_a.mark_as_packed_with_size( { K, M });
	fragment_b.mark_as_packed_with_size( { K, N });
	fragment_qk.mark_as_packed_with_size( { N, M });
	correct_fragment_qk.mark_as_packed_with_size( { N, M });
	fragment_bias.mark_as_packed_with_size( { M, N });
	fragment_softmax.mark_as_packed_with_size( { M, 1 });
	correct_fragment_softmax.mark_as_packed_with_size( { M, 1 });

	for (int k = 0; k < K; k++)
	{
		for (int m = 0; m < M; m++)
			fragment_a.at<float>(k, m) = randFloat() - 0.5f;
		for (int n = 0; n < N; n++)
			fragment_b.at<float>(k, n) = randFloat() - 0.5f;
	}
	for (int m = 0; m < M; m++)
	{
		const float r = randFloat() - 0.5f;
		fragment_softmax.at<float>(0, m) = r;
		correct_fragment_softmax.at<float>(0, m) = r;
		for (int n = 0; n < N; n++)
			fragment_bias.at<float>(m, n) = randFloat() - 0.5f;
	}

	const float alpha = 1.0f / std::sqrt(K);

	mha_qk_def_MxN(correct_fragment_qk, &alpha, fragment_a, fragment_b, fragment_bias, correct_fragment_softmax);

	kernel(fragment_qk, &alpha, fragment_a, fragment_b, fragment_bias, fragment_softmax);

//	std::cout << "Correct\n";
//	for (int n = 0; n < N; n++)
//	{
//		for (int m = 0; m < M; m++)
//			std::cout << correct_fragment_qk.at<float>(n, m) << ' ';
//		std::cout << '\n';
//	}
//	std::cout << "-------------------------------------------\n";
//
//	std::cout << "Actual\n";
//	for (int n = 0; n < N; n++)
//	{
//		for (int m = 0; m < M; m++)
//			std::cout << fragment_qk.at<float>(n, m) << ' ';
//		std::cout << '\n';
//	}
//
//	std::cout << "\nCorrect softmax sum\n";
//	for (int m = 0; m < M; m++)
//		std::cout << correct_fragment_softmax.at<float>(m, 0) << ' ';
//	std::cout << "\n-------------------------------------------\n";
//	std::cout << "Actual softmax sum\n";
//	for (int m = 0; m < M; m++)
//		std::cout << fragment_softmax.at<float>(m, 0) << ' ';
//	std::cout << '\n';
//
//	double diff = 0.0;
//	for (int n = 0; n < N; n++)
//		for (int m = 0; m < M; m++)
//			diff += std::fabs(correct_fragment_qk.at<float>(n, m) - fragment_qk.at<float>(n, m));
//	if (diff / (M * K) > 1.0e-3)
//		std::cout << "\ndiff = " << diff / (M * K) << '\n';

	const double repeats = 1.0e8;
	const double start = getTime();
	int i = 0;
	for (; i < repeats; i++)
	{
		kernel(fragment_qk, &alpha, fragment_a, fragment_b, fragment_bias, fragment_softmax);
		if ((getTime() - start) > 10.0)
			break;
	}
	const double stop = getTime();
//	std::cout << 1.0e6 * (stop - start) / i << " us (" << i << " repeats)\n";
	const double flops = (double) i * (M * N * K) / (stop - start);
	std::cout << M << "x" << N << "x" << K << " : " << flops / 1.0e9 << " GFLOPS\n";

	gemm::aligned_free(matrix_qk, 4096);
	gemm::aligned_free(correct_qk, 4096);
	gemm::aligned_free(lhs, 4096);
	gemm::aligned_free(rhs, 4096);
	gemm::aligned_free(bias_ptr, 4096);
	gemm::aligned_free(softmax_ptr, 4096);
	gemm::aligned_free(correct_softmax_ptr, 4096);
}

void test_depthwise_conv_kernel(const int M, const int N, const int K, mlDataType_t dtype,
		std::function<void(Fragment&, const Fragment&, const Fragment&, const Fragment&)> kernel)
{
	std::unique_ptr<float[]> matrix_c = std::make_unique<float[]>(1024 * 1024);
	std::unique_ptr<float[]> correct_c = std::make_unique<float[]>(1024 * 1024);

	const size_t size_in_bytes_lhs = sizeof(float) * M * N * K;
	const size_t size_in_bytes_rhs = sizeof(float) * N * K;
	void *lhs = gemm::aligned_new(size_in_bytes_lhs, 4096);
	void *rhs = gemm::aligned_new(size_in_bytes_rhs, 4096);
	void *alpha_ptr = gemm::aligned_new(sizeof(float) * 1024, 4096);
	std::memset(lhs, 0, size_in_bytes_lhs);
	std::memset(rhs, 0, size_in_bytes_rhs);
	std::memset(alpha_ptr, 0, sizeof(float) * 1024);

	Fragment fragment_a(lhs, DTYPE_FLOAT32, M * N);
	Fragment fragment_b(rhs, DTYPE_FLOAT32, N);
	Fragment fragment_c(matrix_c.get(), dtype, 1024);
	Fragment correct_fragment_c(correct_c.get(), dtype, 1024);
	Fragment fragment_alpha(alpha_ptr, DTYPE_FLOAT32, 1);
	fragment_a.mark_as_packed_with_size( { K, M * N });
	fragment_b.mark_as_packed_with_size( { K, N });
	fragment_c.mark_as_packed_with_size( { M, N });
	correct_fragment_c.mark_as_packed_with_size( { M, N });
	fragment_alpha.mark_as_packed_with_size( { 1, N });

	for (int k = 0; k < K; k++)
	{
		for (int mn = 0; mn < M * N; mn++)
			fragment_a.at<float>(k, mn) = randFloat() - 0.5f;
		for (int n = 0; n < N; n++)
			fragment_b.at<float>(k, n) = randFloat() - 0.5f;
	}

	const float alpha = 1.0f;
	if (fragment_alpha.rows() == 1)
		fragment_alpha.at<float>(0, 0) = alpha;
	else
	{
		for (int n = 0; n < N; n++)
			fragment_alpha.at<float>(0, n) = n + randFloat();
	}

	depthwise_conv_def_MxN(correct_fragment_c, fragment_alpha, fragment_a, fragment_b);
	kernel(fragment_c, fragment_alpha, fragment_a, fragment_b);

	double diff = 0.0;
	for (int m = 0; m < M; m++)
		for (int n = 0; n < N; n++)
			if (fragment_c.dtype() == DTYPE_FLOAT16)
				diff += std::fabs(half_to_float(correct_fragment_c.at<uint16_t>(m, n)) - half_to_float(fragment_c.at<uint16_t>(m, n)));
			else
				diff += std::fabs(correct_fragment_c.at<float>(m, n) - fragment_c.at<float>(m, n));
	diff /= (M * N);
	if (diff > 1.0e-4)
	{
		std::cout << "Correct\n";
		for (int m = 0; m < M; m++)
		{
			for (int n = 0; n < N; n++)
				if (correct_fragment_c.dtype() == DTYPE_FLOAT16)
					std::cout << half_to_float(correct_fragment_c.at<uint16_t>(m, n)) << ' ';
				else
					std::cout << correct_fragment_c.at<float>(m, n) << ' ';
			std::cout << '\n';
		}
		std::cout << "-------------------------------------------\n";

		std::cout << "Actual\n";
		for (int m = 0; m < M; m++)
		{
			for (int n = 0; n < N; n++)
				if (fragment_c.dtype() == DTYPE_FLOAT16)
					std::cout << half_to_float(fragment_c.at<uint16_t>(m, n)) << ' ';
				else
					std::cout << fragment_c.at<float>(m, n) << ' ';
			std::cout << '\n';
		}

		std::cout << "\ndiff = " << diff << '\n';
		exit(255);
	}
	else
		std::cout << "depthwise conv kernel = " << M << "x" << N << "x" << K << " : OK\n";

	const double repeats = 1.0e8;
	const double start = getTime();
	int i = 0;
	for (; i < repeats; i++)
	{
		kernel(fragment_c, fragment_alpha, fragment_a, fragment_b);
		if ((getTime() - start) > 10.0)
			break;
	}
	const double stop = getTime();
	const double flops = (double) i * (M * N * K) / (stop - start);
	std::cout << M << "x" << N << "x" << K << " : " << flops / 1.0e9 << " GFLOPS\n";
	std::cout << "time = " << (stop - start) / (i / 1.0e6) << "us\n";

	gemm::aligned_free(lhs, 4096);
	gemm::aligned_free(rhs, 4096);
	gemm::aligned_free(alpha_ptr, 4096);
}

void test_depthwise_conv_kernel_v2(const int outputs, const int channels, mlDataType_t dtype,
		std::function<void(Fragment&, const Fragment&, const Fragment&, const Fragment&)> kernel)
{
	std::unique_ptr<float[]> matrix_c = std::make_unique<float[]>(1024 * 1024);
	std::unique_ptr<float[]> correct_c = std::make_unique<float[]>(1024 * 1024);
	const int kernel_size = 7;
	const int inputs = outputs + kernel_size - 1;

	const size_t size_in_bytes_lhs = sizeof(float) * kernel_size * inputs * channels;
	const size_t size_in_bytes_rhs = sizeof(float) * kernel_size * kernel_size * channels;
	void *lhs = gemm::aligned_new(size_in_bytes_lhs, 4096);
	void *rhs = gemm::aligned_new(size_in_bytes_rhs, 4096);
	void *bias_ptr = gemm::aligned_new(sizeof(float) * 1024, 4096);
	std::memset(lhs, 0, size_in_bytes_lhs);
	std::memset(rhs, 0, size_in_bytes_rhs);
	std::memset(bias_ptr, 0, sizeof(float) * 1024);

	Fragment fragment_a(lhs, DTYPE_FLOAT32, inputs * channels);
	Fragment fragment_b(rhs, DTYPE_FLOAT32, channels);
	Fragment fragment_c(matrix_c.get(), dtype, 1024);
	Fragment correct_fragment_c(correct_c.get(), dtype, 1024);
	Fragment fragment_bias(bias_ptr, DTYPE_FLOAT32, 0);
	fragment_a.mark_as_packed_with_size( { kernel_size, inputs * channels });
	fragment_b.mark_as_packed_with_size( { kernel_size * kernel_size, channels });
	fragment_c.mark_as_packed_with_size( { outputs, channels });
	correct_fragment_c.mark_as_packed_with_size( { outputs, channels });
	fragment_bias.mark_as_packed_with_size( { 1, channels });

	for (int k = 0; k < kernel_size; k++)
		for (int mn = 0; mn < inputs * channels; mn++)
			fragment_a.at<float>(k, mn) = randFloat() - 0.5f;

	for (int k = 0; k < kernel_size * kernel_size; k++)
		for (int n = 0; n < channels; n++)
			fragment_b.at<float>(k, n) = randFloat() - 0.5f;

	for (int n = 0; n < channels; n++)
		fragment_bias.at<float>(0, n) = randFloat() - 0.5f;

//	depthwise_conv_def_MxN_v2(correct_fragment_c, fragment_a, fragment_b, fragment_bias);
	kernel(correct_fragment_c, fragment_a, fragment_b, fragment_bias);

//	double diff = 0.0;
//	for (int m = 0; m < outputs; m++)
//		for (int n = 0; n < channels; n++)
//			if (fragment_c.dtype() == DTYPE_FLOAT16)
//				diff += std::fabs(half_to_float(correct_fragment_c.at<uint16_t>(m, n)) - half_to_float(fragment_c.at<uint16_t>(m, n)));
//			else
//				diff += std::fabs(correct_fragment_c.at<float>(m, n) - fragment_c.at<float>(m, n));
//	diff /= (outputs * channels);
//	if (diff > 1.0e-4)
//	{
//		std::cout << "Correct\n";
//		for (int m = 0; m < outputs; m++)
//		{
//			for (int n = 0; n < channels; n++)
//				if (correct_fragment_c.dtype() == DTYPE_FLOAT16)
//					std::cout << half_to_float(correct_fragment_c.at<uint16_t>(m, n)) << ' ';
//				else
//					std::cout << correct_fragment_c.at<float>(m, n) << ' ';
//			std::cout << '\n';
//		}
//		std::cout << "-------------------------------------------\n";
//
//		std::cout << "Actual\n";
//		for (int m = 0; m < outputs; m++)
//		{
//			for (int n = 0; n < channels; n++)
//				if (fragment_c.dtype() == DTYPE_FLOAT16)
//					std::cout << half_to_float(fragment_c.at<uint16_t>(m, n)) << ' ';
//				else
//					std::cout << fragment_c.at<float>(m, n) << ' ';
//			std::cout << '\n';
//		}
//
//		std::cout << "\ndiff = " << diff << '\n';
//		exit(255);
//	}
//	else
//		std::cout << "depthwise conv kernel = " << outputs << "x" << channels << "x" << kernel_size << " : OK\n";

	const double repeats = 1.0e8;
	const double start = getTime();
	int i = 0;
	for (; i < repeats; i++)
	{
		kernel(fragment_c, fragment_a, fragment_b, fragment_bias);
		if ((getTime() - start) > 10.0)
			break;
	}
	const double stop = getTime();
	const double flops = (double) i * (outputs * kernel_size * kernel_size * channels) / (stop - start);
	std::cout << outputs << "x" << channels << "x" << kernel_size << " : " << flops / 1.0e9 << " GFLOPS\n";
	std::cout << "time = " << (stop - start) / (i / 1.0e6) << "us\n";

	gemm::aligned_free(lhs, 4096);
	gemm::aligned_free(rhs, 4096);
	gemm::aligned_free(bias_ptr, 4096);
}

//__m128 BetterFastExpSse(__m128 x) noexcept
//{
//	const __m128 a = _mm_set1_ps((1 << 22) / float(M_LN2));
//	const __m128i b = _mm_set1_epi32(127 * (1 << 23) - 139157);
//	__m128i r = _mm_cvtps_epi32(_mm_mul_ps(a, x));
//	__m128i s = _mm_add_epi32(b, r);
//	__m128i t = _mm_sub_epi32(b, r);
//	return _mm_div_ps(_mm_castsi128_ps(s), _mm_castsi128_ps(t));
//}
//__m256 BetterFastExpAvx(__m256 x) noexcept
//{
//	const __m256 a = _mm256_set1_ps((1 << 22) / float(M_LN2));
//	const __m256i b = _mm256_set1_epi32(127 * (1 << 23) - 139160);
//	__m256i r = _mm256_cvtps_epi32(_mm256_mul_ps(a, x));
//	__m256i s = _mm256_add_epi32(b, r);
//	__m256i t = _mm256_sub_epi32(b, r);
//	return _mm256_div_ps(_mm256_castsi256_ps(s), _mm256_castsi256_ps(t));
//}
//__m256 fast_tanh_avx(__m256 x) noexcept
//{
//	const __m256 a = _mm256_set1_ps((1 << 22) / float(M_LN2));  // to get exp(x/2)
//	const __m256i b = _mm256_set1_epi32(127 * (1 << 23));       // NB: zero shift!
//
//	const __m256i r = _mm256_cvtps_epi32(_mm256_mul_ps(a, x));
//	const __m256 s = _mm256_castsi256_ps(_mm256_add_epi32(b, r));
//	const __m256 t = _mm256_castsi256_ps(_mm256_sub_epi32(b, r));
//
//	return _mm256_div_ps(_mm256_sub_ps(s, t), _mm256_add_ps(s, t));
//}
//__m256 fast_sigmoid_avx(__m256 x) noexcept
//{
//	const __m256 a = _mm256_set1_ps((1 << 22) / float(M_LN2));
//	const __m256i b = _mm256_set1_epi32(127 * (1 << 23) - 139002);
//	__m256i r = _mm256_cvtps_epi32(_mm256_mul_ps(a, x));
//	__m256 s = _mm256_castsi256_ps(_mm256_add_epi32(b, r));
//	__m256 t = _mm256_castsi256_ps(_mm256_sub_epi32(b, r));
//	return _mm256_div_ps(s, _mm256_add_ps(s, t));
//}
//__m256 fast_gelu_avx(__m256 x) noexcept
//{
//	const __m256 a = _mm256_set1_ps((1 << 22) * (1.6849f / float(M_LN2)));
//	const __m256i b = _mm256_set1_epi32(127 * (1 << 23) - 329698);
//	__m256i r = _mm256_cvtps_epi32(_mm256_mul_ps(a, x));
//	__m256 s = _mm256_castsi256_ps(_mm256_add_epi32(b, r));
//	__m256 t = _mm256_castsi256_ps(_mm256_sub_epi32(b, r));
//	return _mm256_div_ps(_mm256_mul_ps(x, s), _mm256_add_ps(s, t));
//}
//
//__m128 fast_exp_sse(__m128 x) noexcept
//{
//	__m128 f, p, r;
//	__m128i t, j;
//	const __m128 a = _mm_set1_ps(12102203.0f); /* (1 << 23) / log(2) */
//	const __m128i m = _mm_set1_epi32(0xff800000); /* mask for integer bits */
//	const __m128 ttm23 = _mm_set1_ps(1.1920929e-7f); /* exp2(-23) */
//	const __m128 c0 = _mm_set1_ps(0.3371894346f);
//	const __m128 c1 = _mm_set1_ps(0.657636276f);
//	const __m128 c2 = _mm_set1_ps(1.00172476f);
//
//	t = _mm_cvtps_epi32(_mm_mul_ps(a, x));
//	j = _mm_and_si128(t, m); /* j = (int)(floor (x/log(2))) << 23 */
//	t = _mm_sub_epi32(t, j);
//	f = _mm_mul_ps(ttm23, _mm_cvtepi32_ps(t)); /* f = (x/log(2)) - floor (x/log(2)) */
//	p = c0; /* c0 */
//	p = _mm_mul_ps(p, f); /* c0 * f */
//	p = _mm_add_ps(p, c1); /* c0 * f + c1 */
//	p = _mm_mul_ps(p, f); /* (c0 * f + c1) * f */
//	p = _mm_add_ps(p, c2); /* p = (c0 * f + c1) * f + c2 ~= 2^f */
//	r = _mm_castsi128_ps(_mm_add_epi32(j, _mm_castps_si128(p))); /* r = p * 2^i*/
//	return r;
//}

int32_t to_int(float x) noexcept
{
	return reinterpret_cast<const int32_t*>(&x)[0];
}
float to_float(int32_t x) noexcept
{
	return reinterpret_cast<const float*>(&x)[0];
}
float best_expf(float x, int32_t shift = -139160) noexcept
{
	// maximum relative error = 0.628981%
	const float a = (1 << 22) / float(M_LN2);
	const int32_t b = 127 * (1 << 23) + shift;

//	std::stringstream stream;
//	stream << std::hex << to_int(a) << " " << b;
//	std::string result(stream.str());
//	std::cout << result << '\n';
//	exit(0);

	const int32_t r = static_cast<int32_t>(a * x);
	const float s = to_float(b + r);
	const float t = to_float(b - r);
	return s / t;
}
float best_sigmf(float x, int32_t shift = -139002) noexcept
{
	// maximum relative error = 0.628656%
	const float a = (1 << 22) / float(M_LN2);
	const int32_t b = 127 * (1 << 23) + -139002;
	const int32_t r = static_cast<int32_t>(a * x);
	const float s = to_float(b + r);
	const float t = to_float(b - r);
	return s / (s + t);
}
float best_tanhf(float x, int32_t shift = 0) noexcept
{
	// maximum relative error = 4.049%
	if (std::fabs(x) < 0.347f)
//		return 0.9805f * x;
		return x * (1.0f - 0.0924f * x);
	const float a = (1 << 23) / float(M_LN2);
	const int32_t b = 127 * (1 << 23);
	const int32_t r = static_cast<int32_t>(a * x);
	const float s = to_float(b + r);
	const float t = to_float(b - r);
	return (s - t) / (s + t);
}
float best_geluf(float x, int32_t shift = -329698, float scale = 1.6849f) noexcept
{
	// maximum relative error =
	const float a = (1 << 22) * (scale / float(M_LN2)); // reg0
	const int32_t b = 127 * (1 << 23) + shift; // reg1
	const int32_t r = static_cast<int32_t>(a * x); // reg2
	const float s = to_float(b + r); // reg3
	const float t = to_float(b - r); // reg2
	return (s * x) / (s + t);
}

float fastExp3(float x) noexcept
{
	union
	{
			float f;
			int32_t i;
	} reinterpreter;

	reinterpreter.i = (int32_t) (12102203.0f * x) + 127 * (1 << 23);
	int32_t m = (reinterpreter.i >> 7) & 0xFFFF;  // copy mantissa
	// empirical values for small maximum relative error (8.34e-5):
//	reinterpreter.i += ((((((((1277 * m) >> 14) + 14825) * m) >> 14) - 79749) * m) >> 11) - 626;
	const float p0 = 1277.0f / (16384.0f * 16384.0f * 2048.0f);
	const float p1 = 14825.0f / (16384.0f * 2048.0f);
	const float p2 = -79749.0f / 2048.0f;

//	reinterpreter.i += (p0 * m * m * m + p1 * m * m + p2 * m) - 626;
	reinterpreter.i += m * 0;
	return reinterpreter.f;
}

float test_accuracy(int32_t shift, bool print = false)
{
	float max_error = 0.0f;
	float avg_error = 0.0f;

	const int samples = 1600000;
	for (int i = 0; i < samples; i++)
	{
		const float x = -80.0 + 160.0f * i / samples;
//		const float target = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
//		const float target = std::tanh(x);
//		const float target = 1.0f / (1.0f + std::exp(-x));
		const float target = std::exp(x);
		const float pred = fastExp3(x);
		const float relative_error = std::fabs(pred - target) / (1.0e-8f + std::fabs(target));
		const float absolute_error = std::fabs(pred - target);
		if (relative_error > max_error)
//		if (pred > target)
		{
//			std::cout << x << " : " << pred << " vs " << target << ", error = " << 100 * relative_error << "%\n";
//			exit(0);
		}
		max_error = std::max(max_error, relative_error);
		avg_error += relative_error;
	}
	if (print)
	{
		std::cout << "max error = " << 100 * max_error << "%\n";
		std::cout << "avg error = " << 100 * avg_error / samples << "%\n";
	}
	return max_error;
}

int main()
{
	std::cout << "BEGIN" << std::endl;
	std::cout << ml::Device::hardwareInfo();
//	{
//		Tensor input( { 100, 15, 15, 100 }, "float32", Device::cpu());
//		for (int i = 0; i < input.dim(1); i++)
//			for (int j = 0; j < input.dim(2); j++)
//				input.at( { 0, i, j, 0 }) = (i + 0.01f * j);
//
//		for (int j = 0; j < input.dim(1); j++)
//		{
//			for (int k = 0; k < input.dim(2); k++)
//				std::cout << (float) input.at( { 0, j, k, 0 }) << ' ';
//			std::cout << "\n";
//		}
//
//		Context context(Device::cuda(0));
//		input.moveTo(context.device());
//
//		Tensor output( { 100, 15, 15, 100 }, "float32", context.device());
//
//		windowPartitioning(context, input, output, { 2, 2 });
//		context.synchronize();
//
////		for (int i = 0; i < output.dim(0); i++)
////		{
////			std::cout << "\n---Window " << i << "---\n";
////			for (int j = 0; j < output.dim(1); j++)
////			{
////				for (int k = 0; k < output.dim(2); k++)
////					std::cout << (float) output.at( { i, j, k, 0 }) << ' ';
////				std::cout << "\n";
////			}
////		}
//
//		Tensor recovered = zeros_like(input);
//		windowMerging(context, output, recovered, { 2, 2 });
//		context.synchronize();
//
//		std::cout << "\n\n\n";
//		for (int j = 0; j < recovered.dim(1); j++)
//		{
//			for (int k = 0; k < recovered.dim(2); k++)
//				std::cout << (float) recovered.at( { 0, j, k, 0 }) << ' ';
//			std::cout << "\n";
//		}
//
//		std::cout << "\ndiff = " << testing::diffForTest(recovered, input) << '\n';
//		return 0;
//	}

//	return 0;
	{
//		Graph graph;
//		FileLoader fl("/home/maciek/alphagomoku/new_runs_2024/supervised/conv_pvum_8x128_v2/network_opt.bin", false);
//		graph.load(fl.getJson()["model"], fl.getBinaryData());
//		graph.setInputShape( { 1, 15, 15, 32 });
//		graph.moveTo(Device::cpu());
////		graph.init();
//		graph.convertTo(DataType::FLOAT16);
//		graph.print();
//
//		Tensor input(graph.getInputShape(), graph.dtype(), Device::cpu());
//		input.zeroall();
//		for (int i = 0; i < input.shape().volumeWithoutLastDim(); i++)
//		{
//			if (input.dtype() == DataType::FLOAT32)
//			{
//				float *ptr = reinterpret_cast<float*>(input.data()) + i * 32;
//				ptr[0] = 1.0f;
//				ptr[3] = 1.0f;
//				ptr[4] = 1.0f;
//			}
//			if (input.dtype() == DataType::FLOAT16)
//			{
//				uint16_t *ptr = reinterpret_cast<uint16_t*>(input.data()) + i * 32;
//				ptr[0] = 0x3c00;
//				ptr[3] = 0x3c00;
//				ptr[4] = 0x3c00;
//			}
//		}
//
//		graph.getInput().copyFrom(graph.context(), input);
//		graph.forward(1);
//		graph.context().synchronize();
//
//		std::cout << "value = " << graph.getOutput(1).get( { 0, 0 }) << '\n';
//		std::cout << "uncertainty = " << graph.getOutput(2).get( { 0, 0 }) << '\n';
//
//		for (int i = 0; i < 50; i++)
//			std::cout << graph.getOutput(3).get( { 0, i }) << ' ';
//		std::cout << '\n';
//
//		for (int i = 0; i < 15; i++)
//		{
//			for (int j = 0; j < 15; j++)
//				std::cout << graph.getOutput(0).get( { 0, i * 15 + j }) << ' ';
//			std::cout << '\n';
//		}
//
//		return 0;

//		best_expf(0.0f);
		const mlDataType_t dtype = DTYPE_FLOAT32;
//		for (int i = 1; i <= 256; i++)
//			test_packing(i, 6, ml::MatrixOp::TRANSPOSE, dtype, pack_avx2_6xK, false);
//		for (int i = 1; i <= 256; i *= 2)
//			test_packing(i, 8, ml::MatrixOp::NORMAL, dtype, pack_avx_8xK, true);
//		for (int i = 1; i <= 128; i++)
//			test_packing(i, 24, ml::MatrixOp::TRANSPOSE, dtype, pack_avx512f_24xK, false);
//		for (int i = 1; i <= 128; i++)
//			test_packing(i, 16, ml::MatrixOp::NORMAL, dtype, pack_avx512f_16xK, false);
//		for (int i = 1; i <= 128; i++)
//			test_packing(i, 16, ml::MatrixOp::TRANSPOSE, dtype, pack_avx512f_16xK, false);

//		test_packing(256, 12, ml::MatrixOp::TRANSPOSE, DTYPE_FLOAT32, pack_avx2_fma_12xK, true);
//		std::cout << "MHA fused softmax(QK)\n";
//		for (int i = 1; i <= 512; i *= 2)
//			test_mha_kernel(12, 8, i, mha_qk_avx2_12x8);
//		std::cout << "Base GEMM\n";
//		for (int i = 1; i <= 512; i *= 2)
//			test_microkernel(12, 8, i, dtype, gemm_avx2_12x8);

		std::cout << "fp32 kernel\n";
		for (int i = 1; i <= 512; i *= 2)
			test_microkernel(12, 8, i, dtype, gemm_avx2_12x8);

		std::cout << "int8 kernel\n";
		for (int i = 1; i <= 512; i *= 2)
			test_microkernel(12, 8, i, dtype, intgemm_avx2_12x8);

//		test_depthwise_conv_kernel_v2(6, 8, dtype, depthwise_conv_avx2_6x8_v2);
//		test_depthwise_conv_kernel(12, 8, 49, dtype, depthwise_conv_avx2_12x8);
//		for (int i = 1; i <= 512; i *= 2)
//			test_depthwise_conv_kernel(12, 8, i, dtype, depthwise_conv_avx2_12x8);
		return 0;

//		int32_t center = 0;
//		int32_t range = 0;
//		int32_t step = 1;
//		float best_accuracy = 1.0f;
//		int best_shift = 0;
//		for (int i = center - range; i <= center + range; i += step)
//		{
//			const float acc = test_accuracy(i);
//
//			if (acc < best_accuracy)
//			{
//				best_accuracy = acc;
//				best_shift = i;
//				std::cout << i << " : " << 100 * acc << "%\n";
//			}
//		}
//		std::cout << "\nbest shift : best accuracy\n";
//		std::cout << best_shift << " : " << 100 * best_accuracy << "%\n";
//		test_accuracy(0, true);
//		test_accuracy(best_shift, true);
//		for (int i = 0; i < 30; i++)
//		{
//			const float x = i * 1.0e-1f;
//			std::cout << x << " " << best_tanhf(x) << " " << std::tanh(x) << '\n';
//		}
	}
//	return 0;

//	{
//		float input[8] = { -2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f };
//		__m256 x = _mm256_loadu_ps(input);
//		volatile __m256 y = BetterFastExpAvx(x);
//		float output[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
//		_mm256_storeu_ps(output, y);
//		for (int i = 0; i < 8; i++)
//			std::cout << output[i] << " vs " << std::exp(input[i]) << '\n';
//
//		double t0 = getTime();
//		for (int i = 0; i < 1000000000; i++)
//		{
//			y = BetterFastExpAvx(x);
//			y = BetterFastExpAvx(x);
//			y = BetterFastExpAvx(x);
//			y = BetterFastExpAvx(x);
//		}
//		double t1 = getTime();
//		std::cout << "sse exp " << (t1 - t0) / 4 << "ns\n";
//
//		__m256 x_avx = _mm256_loadu_ps(input);
//		volatile __m256 y_avx = fast_gelu_avx(x_avx);
//		_mm256_storeu_ps(output, y_avx);
//		for (int i = 0; i < 8; i++)
//			std::cout << output[i] << " vs " << input[i] / (1.0f + std::exp(-1.702f * input[i])) << '\n';
//		t0 = getTime();
//		for (int i = 0; i < 1000000000; i++)
//		{
//			y_avx = fast_gelu_avx(x_avx);
//			y_avx = fast_gelu_avx(x_avx);
//			y_avx = fast_gelu_avx(x_avx);
//			y_avx = fast_gelu_avx(x_avx);
//		}
//		t1 = getTime();
//		std::cout << "avx sigm " << (t1 - t0) / 4 << "ns\n";
//
//		const double t2 = getTime();
//		volatile float stdx = 1.0f, stdy = 0.0f;
//		for (int i = 0; i < 100000000; i++)
//		{
//			stdy = std::tanh(stdx);
//			stdy = std::tanh(stdx);
//			stdy = std::tanh(stdx);
//			stdy = std::tanh(stdx);
//		}
//		const double t3 = getTime();
//		std::cout << "std::tanh " << 10 * (t3 - t2) / 4 << "ns\n";
//		std::cout << stdy << '\n';
//
//	}
//	return 0;

//	test_mnist();
//	return 0;

	if (true)
	{
		Graph graph;
		const bool symmetric = false;
		const int batch_size = 256;
		const int board_size = 15;
		int embedding = 256;
		const int patch_size = 1;
		const int head_dim = 32;
		const int pos_encoding_range = (board_size + patch_size - 1) / patch_size;

		auto x = graph.addInput( { batch_size, board_size, board_size, 8 });
		x = graph.add(Conv2D(embedding / (patch_size * patch_size), 3, "relu"), x);
//		x = graph.add(Conv2D(embedding / (patch_size * patch_size), 5).useBias(false), x);
//		x = graph.add(BatchNormalization("relu").useGamma(false), x);
//		x = graph.add(ml::SpaceToDepth(patch_size), x);

//		const int size_2 = (board_size + 1) / 2;
//
//		x = graph.add(ml::Conv2D(embedding, 3, "relu"), x);
//		x = graph.add(ml::Conv2D(embedding, 3, "relu"), x);
//		auto level0 = graph.add(ml::Conv2D(embedding, 3, "relu"), x);
//		x = graph.add(ml::SpaceToDepth(2), level0);
//
//		x = graph.add(ml::Conv2D(embedding * 2, 3, "relu"), x);
//		x = graph.add(ml::Conv2D(embedding * 2, 3, "relu"), x);
//		auto level1 = graph.add(ml::Conv2D(embedding * 2, 3, "relu"), x);
//		x = graph.add(ml::SpaceToDepth(2), level1);
//
//		x = graph.add(ml::Conv2D(embedding * 4, 3, "relu"), x);
//		x = graph.add(ml::Conv2D(embedding * 4, 3, "relu"), x);
//		x = graph.add(ml::Conv2D(embedding * 4, 3, "relu"), x);
//		x = graph.add(ml::Conv2D(embedding * 4, 3, "relu"), x);
//
//		x = graph.add(ml::DepthToSpace(2, { size_2, size_2 }), x);
//		x = graph.add(ml::Conv2D(embedding * 2, 1, "linear"), x);
//		x = graph.add(ml::Add(), { x, level1 });
//		x = graph.add(ml::Conv2D(embedding * 2, 3, "relu"), x);
//		x = graph.add(ml::Conv2D(embedding * 2, 3, "relu"), x);
//		x = graph.add(ml::Conv2D(embedding * 2, 3, "relu"), x);
//
//		x = graph.add(ml::DepthToSpace(2, { board_size, board_size }), x);
//		x = graph.add(ml::Conv2D(embedding, 1, "linear"), x);
//		x = graph.add(ml::Add(), { x, level0 });
//		x = graph.add(ml::Conv2D(embedding, 3, "relu"), x);
//		x = graph.add(ml::Conv2D(embedding, 3, "relu"), x);
//		x = graph.add(ml::Conv2D(embedding, 3, "relu"), x);

//		for (int i = 0; i < 4; i++)
//		{
////			auto y = graph.add(ml::RMSNormalization(), x);
////			y = graph.add(ml::Conv2D((3 - symmetric) * embedding, 1).useBias(false), y);
////			y = graph.add(ml::MultiHeadAttention(embedding / head_dim, pos_encoding_range, symmetric), y);
////			y = graph.add(ml::Conv2D(embedding, 1).useBias(false), y);
////			x = graph.add(ml::Add(), { x, y });
////
////			y = graph.add(ml::RMSNormalization(), x);
////			y = graph.add(ml::Conv2D(embedding, 1, "relu"), y);
////			y = graph.add(ml::Conv2D(embedding, 1), y);
////			x = graph.add(ml::Add(), { x, y });
//
////			auto y = graph.add(ml::Conv2D(embedding, 3, "relu"), x);
////			y = graph.add(ml::Conv2D(embedding, 3, "linear"), y);
////			x = graph.add(ml::Add("relu"), { x, y });
//
//			auto y = graph.add(ml::Conv2D(embedding, 1, "relu"), x);
////			y = graph.add(ml::Conv2D(embedding, 3, "relu"), y);
//			y = graph.add(ml::Conv2D(embedding, 1, "linear"), y);
//			x = graph.add(ml::Add(), { x, y });
//		}
//		embedding *= 2;
//		x = graph.add(ml::Conv2D(embedding, 1, "relu"), x);

		for (int i = 0; i < 8; i++)
		{
//			auto y = graph.add(ml::RMSNormalization(false), x);
//			y = graph.add(ml::Conv2D((3 - symmetric) * embedding, 1).useBias(false), y);
//			y = graph.add(ml::MultiHeadAttention(embedding / head_dim, pos_encoding_range, symmetric), y);
//			y = graph.add(ml::Conv2D(embedding, 1).useBias(false), y);
//			x = graph.add(ml::Add(), { x, y });
//
//			y = graph.add(ml::RMSNormalization(false), x);
//			y = graph.add(ml::Conv2D(embedding, 1, "relu"), y);
//			y = graph.add(ml::Conv2D(embedding, 1), y);
//			x = graph.add(ml::Add(), { x, y });

			auto y = graph.add(ml::DepthwiseConv2D(embedding, 7), x);
			y = graph.add(ml::Conv2D(embedding, 1, "relu"), y);
			y = graph.add(ml::Conv2D(embedding, 1, "linear"), y);
			x = graph.add(ml::Add(), { x, y });

//			auto y = graph.add(ml::Conv2D(embedding / 2, 1, "relu"), x);
//			y = graph.add(ml::Conv2D(embedding / 2, 3, "relu"), y);
//			y = graph.add(ml::Conv2D(embedding, 3, "linear"), y);
//			x = graph.add(ml::Add("relu"), { x, y });
		}
//		embedding *= 2;
//		x = graph.add(ml::Conv2D(embedding, 1, "relu"), x);

//		for (int i = 0; i < 4; i++)
//		{
//			auto y = graph.add(ml::Conv2D(embedding, 3, "relu"), x);
//			y = graph.add(ml::Conv2D(embedding, 3, "linear"), y);
//			x = graph.add(ml::Add("relu"), { x, y });
//		}
//		embedding *= 2;
//		x = graph.add(ml::Conv2D(embedding, 1, "relu"), x);

//		for (int i = 0; i < 2; i++)
//		{
//			auto y = graph.add(ml::RMSNormalization(), x);
//			y = graph.add(ml::Conv2D((3 - symmetric) * embedding, 1).useBias(false), y);
//			y = graph.add(ml::MultiHeadAttention(embedding / head_dim, pos_encoding_range, symmetric), y);
//			y = graph.add(ml::Conv2D(embedding, 1).useBias(false), y);
//			x = graph.add(ml::Add(), { x, y });

//			auto y = graph.add(ml::Conv2D(embedding / 2, 1, "relu"), x);
//			y = graph.add(ml::Conv2D(embedding / 2, 3, "relu"), y);
//			y = graph.add(ml::Conv2D(embedding, 3, "linear"), y);
//			x = graph.add(ml::Add("relu"), { x, y });

//			auto y = graph.add(ml::Conv2D(embedding, 3, "relu"), x);
//			y = graph.add(ml::Conv2D(embedding, 3, "linear"), y);
//			x = graph.add(ml::Add("relu"), { x, y });

//			auto y = graph.add(ml::RMSNormalization(), x);
//			y = graph.add(ml::PositionalEncoding(), y);
//			y = graph.add(ml::Conv2D((3 - symmetric) * embedding, 1).useBias(false), y);
//			y = graph.add(ml::MultiHeadAttention(embedding / head_dim, pos_encoding_range, symmetric), y);
//			y = graph.add(ml::Conv2D(embedding, 1).useBias(false), y);
//			x = graph.add(ml::Add(), { x, y });

//			y = graph.add(ml::RMSNormalization(), x);
//			y = graph.add(ml::Conv2D(embedding, 1, "relu"), y);
//			y = graph.add(ml::Conv2D(embedding, 1), y);
//			x = graph.add(ml::Add(), { x, y });
//		}

		// policy head
		auto p = graph.add(ml::DepthwiseConv2D(embedding, 7), x);
		p = graph.add(ml::Conv2D(embedding, 1, "relu"), p);
		p = graph.add(ml::Conv2D(1, 1, "linear"), p);
		p = graph.add(ml::Softmax( { 1, 2, 3 }), p);
		graph.addOutput(p, ml::CrossEntropyLoss(1.0f));
//		auto p = graph.add(ml::RMSNormalization(), x);
//		p = graph.add(ml::Conv2D(2 * embedding, 1).useBias(false), p);
//		p = graph.add(ml::MultiHeadAttention(1, pos_encoding_range, true), p);
//		p = graph.add(ml::Conv2D(1, 1), p);
//		p = graph.add(ml::Softmax( { 1, 2, 3 }), p);
//		graph.addOutput(p);

//		auto common = graph.add(ml::GlobalPooling(), x);
//		common = graph.add(ml::Dense(256, "relu"), common);

		// value head
		auto v = graph.add(ml::Conv2D(embedding, 1, "relu"), x);
		v = graph.add(ml::GlobalPooling(), v);
		v = graph.add(ml::Dense(256, "relu"), v);
//		auto v = graph.add(ml::Dense(256).useBias(false), common);
//		v = graph.add(BatchNormalization("relu").useGamma(false), v);
		v = graph.add(ml::Dense(3), v);
		v = graph.add(ml::Softmax( { 1 }), v);
		graph.addOutput(v, ml::CrossEntropyLoss(1.0f));

		// uncertainty head
//		auto unc = graph.add(ml::Dense(128, "relu"), common);
//		unc = graph.add(ml::Dense(1, "sigmoid"), unc);
//		graph.addOutput(unc, ml::MeanSquaredLoss(1.0f));

		// moves left head
//		auto mlh = graph.add(ml::Dense(128, "relu"), common);
//		mlh = graph.add(ml::Dense(50), mlh);
//		mlh = graph.add(ml::Softmax( { 1 }), mlh);
//		graph.addOutput(mlh, ml::CrossEntropyLoss(0.1f));

//		auto p = graph.add(ml::RMSNormalization(), x);
////		auto p = graph.add(ml::Conv2D(embedding, 3, "relu"), x);
////		p = graph.add(ml::Conv2D(1, 3, "linear"), p);
//		p = graph.add(ml::Conv2D(3 * embedding, 1).useBias(false), p);
//		p = graph.add(ml::MultiHeadAttention(embedding / head_dim, pos_encoding_range), p);
//		p = graph.add(ml::Conv2D(patch_size * patch_size, 1), p);
////		p = graph.add(ml::DepthToSpace(patch_size, { board_size, board_size }), p);
//		p = graph.add(ml::Softmax( { 1, 2, 3 }), p);
//		graph.addOutput(p);
//
////		auto v = graph.add(ml::Conv2D(4, 1, "relu"), x);
////		v = graph.add(ml::Dense(std::min(256, 2 * embedding), "relu"), v);
////		v = graph.add(ml::Dense(3, "linear"), v);
////		v = graph.add(ml::Softmax( { 1 }), v);
//
//		auto v = graph.add(ml::GlobalPooling(), x);
//		v = graph.add(ml::Dense(embedding, "relu"), v);
//		v = graph.add(ml::Dense(embedding, "relu"), v);
//		v = graph.add(ml::Dense(3), v);
//		v = graph.add(ml::Softmax( { 1 }), v);
//		graph.addOutput(v);

//		auto p = graph.add(ml::Conv2D(embedding, 1, "relu"), x);
//		p = graph.add(ml::DepthToSpace(patch_size, { board_size, board_size }), p);
//		p = graph.add(ml::Conv2D(1, 1), p);
//		p = graph.add(ml::Softmax( { 1, 2, 3 }), p);
//		graph.addOutput(p);
//
//		// value head
//		auto v = graph.add(ml::GlobalPooling(), x);
//		v = graph.add(ml::Dense(embedding, "relu"), v);
//		v = graph.add(ml::Dense(3), v);
//		v = graph.add(ml::Softmax( { 1 }), v);
//		graph.addOutput(v);

//		graph.print();

		graph.init();
//		graph.moveTo(Device::cpu());
		graph.moveTo(Device::cuda(0));
		graph.convertTo(DataType::FLOAT16);

		Tensor input(graph.getInputShape(), graph.dtype(), Device::cpu());
		for (int i = 0; i < input.shape().volumeWithoutLastDim(); i++)
		{
			if (input.dtype() == DataType::FLOAT32)
			{
				float *ptr = reinterpret_cast<float*>(input.data()) + i * graph.getInputShape().lastDim();
				ptr[0] = 1.0f;
				ptr[3] = 1.0f;
				ptr[4] = 1.0f;
			}
			if (input.dtype() == DataType::FLOAT16)
			{
				uint16_t *ptr = reinterpret_cast<uint16_t*>(input.data()) + i * graph.getInputShape().lastDim();
				ptr[0] = 0x3c00;
				ptr[3] = 0x3c00;
				ptr[4] = 0x3c00;
			}
		}
		graph.getInput().copyFrom(graph.context(), input);
//		graph.forward(batch_size);
//		graph.context().synchronize();
//		graph.backward(batch_size);
//		graph.learn();
//		graph.context().synchronize();
//
//		std::this_thread::sleep_for(std::chrono::milliseconds(200));
//
//		graph.forward(batch_size);
//		graph.context().synchronize();
//		std::this_thread::sleep_for(std::chrono::milliseconds(100));
//
//		graph.backward(batch_size);
//		graph.learn();
//		graph.context().synchronize();
//
		graph.makeNonTrainable();
		ml::FoldAdd().optimize(graph);
		graph.print();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
//
		graph.forward(batch_size);
		graph.context().synchronize();

		std::cout << "starting benchmark\n";
		const double start = getTime();
		int repeats = 0;
		for (; repeats < 100000; repeats++)
		{
			graph.forward(batch_size);
			graph.context().synchronize();
			if ((getTime() - start) > 30.0)
				break;
		}
		const double stop = getTime();
		const double time = stop - start;
		std::cout << "time = " << time << "s, repeats = " << repeats << ", n/s = " << batch_size * repeats / time << "\n";

		for (int i = 0; i < graph.numberOfOutputs(); i++)
			std::cout << testing::normForTest(graph.getOutput(i)) / graph.getOutputShape(i).volume() << '\n';
		return 0;
	}

//	std::cout << "Detailed properties" << std::endl;
//	opencl_print_device_features(0);

//	{
//		Graph graph;
//
//		auto x = graph.addInput(Shape( { 32, 15, 15, 8 }));
//		x = graph.add(Conv2D(128, 5, "relu"), x);
//		x = graph.add(Conv2D(64, 1, "relu"), x);
//		x = graph.add(Conv2D(64, 3, "relu"), x);
//		x = graph.add(Conv2D(128, 3, "relu"), x);
//		x = graph.add(GlobalPooling("linear"), x);
//		graph.addOutput(x);
//		graph.init();
//		graph.makeNonTrainable();
////		graph.convertTo(DataType::FLOAT16);
//		graph.moveTo(Device::opencl(0));
//		graph.forward(32);
//		graph.context().synchronize();
//
//		std::cout << "starting benchmark\n";
//		const double start = getTime();
//		ml::Event t0 = graph.context().createEvent();
//		int repeats = 0;
//		for (; repeats < 1000000; repeats++)
//		{
//			graph.forward(32);
//			if ((getTime() - start) > 30.0)
//				break;
//		}
//		ml::Event t1 = graph.context().createEvent();
//		graph.context().synchronize();
//		const double stop = getTime();
//		std::cout << "event time = " << ml::Event::getElapsedTime(t0, t1) << '\n';
//
//		const double time = stop - start;
//		std::cout << "time = " << time << "s, repeats = " << repeats << ", n/s = " << 12 * repeats / time << "\n";
//
////		graph.forward(12);
////		graph.backward(12);
////		graph.context().synchronize();
//		std::cout << "END" << '\n';
//	}
//	return 0;

	if (false)
	{
		Context context(Device::opencl(1));
		Tensor input( { 10000, 256 }, "float32", Device::opencl(1));
		Tensor output( { 10000, 256 }, "float32", Device::opencl(1));

//		std::vector<int> tmp(input.volume(), -1);
//		input.copyFromHost(context, tmp.data(), input.sizeInBytes());

		mlShape_t shape = ml::make_shape( { 1000, 256 });

//		opencl_unpack_input(context.backend(), shape, DTYPE_FLOAT32, output.data(), input.data());
//		context.synchronize();

		const double start = getTime();
		for (;;)
		{
//			std::cout << "in the loop " << i << '\n';
			const double t0 = getTime();
			for (int j = 0; j < 1000; j++)
			{
				softmaxForward(context, output, input);
				context.synchronize();
			}
			const double t1 = getTime();
			std::cout << "time = " << 1000 * (t1 - t0) << " ms\n";
//			opencl_unpack_input(context.backend(), shape, DTYPE_FLOAT32, output.data(), input.data());
//			context.synchronize();
		}
		std::cout << "out of the loop" << '\n';
//		context.synchronize();
		const double stop = getTime();
		std::cout << "time = " << 1000 * (stop - start) << "ms\n";
		context.synchronize();

//		for (int i = 0; i < 1000; i++)
//			for (int j = 0; j < 1000; j++)
//				for (int l = 0; l < 32; l++)
//					if (output.get( { i, j, l }) != 1.0f)
//					{
//						std::cout << i << "," << j << "," << l << " = " << output.get( { i, j, l }) << '\n';
//						return 0;
//					}

//		output.convertTo(context, DataType::FLOAT32);
		context.synchronize();
		for (int i = 0; i < 10; i++)
			std::cout << output.get( { 0, 0, i }) << '\n';

//		Context context(Device::opencl(1));
//		context.synchronize();
//
//		Tensor t( { 10, 10 }, "float32", Device::opencl(1));
//		std::cout << t.info() << '\n';
//		t.setall(context, 1.234f);
//		std::cout << "value = " << t.get( { 5, 5 }) << '\n';
//
//		Tensor other( { 10, 10 }, "float32", Device::opencl(0));
//		other.copyFrom(context, t);
//		std::cout << "other = " << other.get( { 5, 5 }) << '\n';
		std::cout << "END" << std::endl;
		return 0;
	}

//	for (int i = 1; i <= 1024; i++)
//		test_packing(i, 16, ml::MatrixOp::TRANSPOSE);

//	for (int i = 1; i <= 1024; i *= 2)
//		benchmark_winograd_transform(i);
//	return 0;

//	{
//		Context context;
//		const int filters = 1;
//		const Shape weight_shape( { filters, 3, 3, filters });
//		Tensor weights(weight_shape, "float32", Device::cpu());
//		Tensor matrices_old( { 49, filters, filters }, "float32", Device::cpu());
//		Tensor matrices_new(matrices_old.shape(), "float32", Device::cpu());
//
//		testing::initForTest(weights, 0.0, 1.0);
//		weights.setall(context, 1.0f);
//
//		{
//			const double repeats = 1.0e8;
//			const double start = getTime();
//			int i = 1;
//			for (; i < repeats; i++)
//			{
//				winogradWeightTransform(context, weights, matrices_old, false);
////				if ((getTime() - start) > 30.0)
//				break;
//			}
//			const double stop = getTime();
//			std::cout << i << " repeats, " << 1.0e3 * (stop - start) / i << "ms\n";
//		}
//
//		void *workspace = gemm::aligned_new(1024 * 1024, 4096);
//
//		omp_set_num_threads(1);
//		//		matrices_new.zeroall(context);
//		{
//			const double repeats = 1.0e8;
//			const double start = getTime();
//			int i = 1;
//			for (; i < repeats; i++)
//			{
//				cpu_winograd_weight_transform_v2(context.backend(), 5, DTYPE_FLOAT32, make_shape( { filters, 3, 3, filters }), weights.data(),
//						matrices_new.data(), false);
////				if ((getTime() - start) > 30.0)
//				break;
//			}
//			const double stop = getTime();
//			std::cout << i << " repeats, " << 1.0e3 * (stop - start) / i << "ms\n";
//		}
//		std::cout << "diff = " << testing::diffForTest(matrices_old, matrices_new) << '\n';
//
//		for (int i = 0; i < matrices_old.firstDim(); i++)
//			std::cout << i << " : " << matrices_old.get( { i, 0, 0 }) << " vs " << matrices_new.get( { i, 0, 0 }) << '\n';
//
//		gemm::aligned_free(workspace, 4096);
//	}
//	return 0;

//	if (true)
//	{
//		Context context;
//		const int filters = 128;
//		const Shape weight_shape( { filters, 3, 3, filters });
//		Tensor input( { 12, 15, 15, filters }, "float16", Device::cpu());
//		Tensor matrices_old( { 49, 12 * 3 * 3, filters }, "float16", Device::cpu());
//		Tensor matrices_new(matrices_old.shape(), "float16", Device::cpu());
//
//		testing::initForTest(input, 0.0, 1.0);
////		matrices_new.setall(context, 1.0f);
////		input.setall(context, 1.0f);
//
//		{
//			const double repeats = 1.0e8;
//			const double start = getTime();
//			int i = 1;
//			for (; i < repeats; i++)
//			{
//				winogradInputTransform(context, weight_shape, input, matrices_old);
//				if ((getTime() - start) > 30.0)
//					break;
//			}
//			const double stop = getTime();
//			std::cout << i << " repeats, " << 1.0e3 * (stop - start) / i << "ms\n";
//		}
//
//		void *workspace = gemm::aligned_new(1024 * 1024, 4096);
//
//		omp_set_num_threads(1);
////		matrices_new.zeroall(context);
//		{
//			const double repeats = 1.0e8;
//			const double start = getTime();
//			int i = 1;
//			for (; i < repeats; i++)
//			{
////				kernel_transform_input<float16, float, 3, 5>(matrices_new.data(), input.data(), input.dim(0), input.dim(1), input.dim(2),
////						input.dim(3), workspace);
//				kernel_transform_input_v2(matrices_new.data(), input.data(), input.dim(0), input.dim(1), input.dim(2), input.dim(3), workspace, 3, 5,
//						DataType::FLOAT16);
//				if ((getTime() - start) > 30.0)
//					break;
//			}
//			const double stop = getTime();
//			std::cout << i << " repeats, " << 1.0e3 * (stop - start) / i << "ms\n";
//		}
//		std::cout << "diff = " << testing::diffForTest(matrices_old, matrices_new) << '\n';
//
////		for (int i = 0; i < matrices_old.firstDim(); i++)
////			std::cout << matrices_old.get( { i, 0, 0 }) << " vs " << matrices_new.get( { i, 0, 0 }) << '\n';
//
//		for (int i = 0; i < matrices_old.dim(0); i++)
//			for (int j = 0; j < matrices_old.dim(1); j++)
//				for (int k = 0; k < matrices_old.dim(2); k++)
//				{
//					const float old_value = matrices_old.get( { i, j, k });
//					const float new_value = matrices_new.get( { i, j, k });
//					if (std::fabs(old_value - new_value) > 0.1f)
//					{
//						std::cout << "difference : ";
//						std::cout << old_value << " vs " << new_value << '\n';
//						std::cout << "at " << i << " " << j << " " << k << '\n';
//						return 0;
//					}
//				}
//
//		gemm::aligned_free(workspace, 4096);
//	}
//	else
//	{
//		Context context;
//		const int filters = 23;
//		const Shape weight_shape( { filters, 3, 3, filters });
//		Tensor matrices( { 36, 12 * 4 * 4, filters }, "float16", Device::cpu());
//		Tensor ext( { 12, 15, 15, filters }, "float16", Device::cpu());
//		Tensor bias( { filters }, "float16", Device::cpu());
//
//		Tensor output_old( { 12, 15, 15, filters }, "float16", Device::cpu());
//		Tensor output_new( { 12, 15, 15, filters }, "float16", Device::cpu());
//
//		testing::initForTest(matrices, 0.0, 1.0);
//		testing::initForTest(ext, 0.0, 1.0);
//		testing::initForTest(bias, 0.0, 1.0);
////		matrices.setall(context, 1.0f);
//
//		{
//			const double repeats = 1.0e8;
//			const double start = getTime();
//			int i = 0;
//			for (; i < repeats; i++)
//			{
//				winogradOutputTransform(context, weight_shape, matrices, output_old, bias, ext, ActivationType::RELU);
////				if ((getTime() - start) > 30.0)
//				break;
//			}
//			const double stop = getTime();
//			std::cout << i << " repeats, " << 1.0e3 * (stop - start) / i << "ms\n";
//		}
//
//		void *workspace = gemm::aligned_new(1024 * 1024, 4096);
//
//		output_new.zeroall(context);
//		{
//			const double repeats = 1.0e8;
//			const double start = getTime();
//			int i = 0;
//			for (; i < repeats; i++)
//			{
//				kernel_transform_output<float16, float, 3, 4>(matrices.data(), output_new.data(), ext.data(), bias.data(), output_new.dim(0),
//						output_new.dim(1), output_new.dim(2), output_new.dim(3), workspace, true);
////				if ((getTime() - start) > 30.0)
//				break;
//			}
//			const double stop = getTime();
//			std::cout << i << " repeats, " << 1.0e3 * (stop - start) / i << "ms\n";
//		}
//		std::cout << "diff = " << testing::diffForTest(output_old, output_new) << '\n';
//
//		for (int i = 0; i < 4; i++)
//			for (int j = 0; j < 4; j++)
//				std::cout << output_old.get( { 0, i, j, 0 }) << " vs " << output_new.get( { 0, i, j, 0 }) << '\n';
//
//		for (int i = 0; i < output_old.dim(0); i++)
//			for (int j = 0; j < output_old.dim(1); j++)
//				for (int k = 0; k < output_old.dim(2); k++)
//					for (int l = 0; l < output_old.dim(3); l++)
//					{
//						const float old_value = output_old.get( { i, j, k, l });
//						const float new_value = output_new.get( { i, j, k, l });
//						if (std::fabs(old_value - new_value) > 0.01f)
//						{
//							std::cout << old_value << " " << new_value << '\n';
//							std::cout << "at " << i << " " << j << " " << k << '\n';
//							return 0;
//						}
//					}
//
//		gemm::aligned_free(workspace, 4096);
//	}
////		test_winograd_transform();
//	std::cout << "END" << std::endl;
//	return 0;

	return 0;

//	test_microkernel(24, 4, 64);
//	test_microkernel(24, 4, 128);
//	test_microkernel(24, 4, 256);
//	test_microkernel(24, 4, 512);
//	test_microkernel(24, 4, 1024);
//	return 0;

//	{
//		Device::setNumberOfThreads(1);
//		const char op_a = 'n';
//		const char op_b = 't';
//
//		const int B = 36;
//		const int M = 128;
//		const int N = 128;
//		const int K = 64;
//
//		mlShape_t shape_a = (op_a == 'n') ? create_shape(B, M, K) : create_shape(B, K, M);
//		mlShape_t shape_b = (op_b == 'n') ? create_shape(B, K, N) : create_shape(B, N, K);
//		mlShape_t shape_c = create_shape(B, M, N);
//		mlShape_t shape_d = create_shape(B, M, N);
//
//		std::unique_ptr<float[]> matrix_a = std::make_unique<float[]>(B * M * K);
//		std::unique_ptr<float[]> matrix_b = std::make_unique<float[]>(B * K * N);
//		std::unique_ptr<float[]> matrix_c = std::make_unique<float[]>(B * M * N);
//		std::unique_ptr<float[]> matrix_d = std::make_unique<float[]>(B * M * N);
//
//		for (int i = 0; i < B * M * K; i++)
////			matrix_a[i] = float_to_half(randFloat() - 0.5f);
//			matrix_a[i] = randFloat() - 0.5f;
//		for (int i = 0; i < B * K * N; i++)
////			matrix_b[i] = float_to_half(randFloat() - 0.5f);
//			matrix_b[i] = randFloat() - 0.5f;
//
//		const float alpha = 1.0f;
//		const float beta = 0.0f;
//
////		for (int row = 0; row < shape_a.dim[0]; row++)
////		{
////			for (int col = 0; col < shape_a.dim[1]; col++)
////				std::cout << matrix_a[row * shape_a.dim[1] + col] << ' ';
////			std::cout << '\n';
////		}
//
//		std::unique_ptr<float[]> correct_d = std::make_unique<float[]>(B * M * N);
//		baseline_gemm_batched<float, float>(shape_d, correct_d.get(), alpha, op_a, shape_a, matrix_a.get(), op_b, shape_b, matrix_b.get(), beta,
//				shape_c, matrix_c.get());
//
//		mlContext_t context = cpu_create_context();
//		cpu_gemm_batched(context, DTYPE_FLOAT32, shape_c, matrix_c.get(), shape_a, matrix_a.get(), shape_b, matrix_b.get(), op_a, op_b, alpha, beta);
////		cpu_gemm(context, DTYPE_FLOAT32, shape_d, matrix_d.get(), shape_a, matrix_a.get(), shape_b, matrix_b.get(), op_a, op_b, alpha, beta);
//
//		const double repeats = 1.0e7;
//		const double start = getTime();
//		int i = 0;
//		for (; i < repeats; i++)
//		{
//			cpu_gemm_batched(context, DTYPE_FLOAT32, shape_c, matrix_c.get(), shape_a, matrix_a.get(), shape_b, matrix_b.get(), op_a, op_b, alpha,
//					beta);
////			cpu_gemm_v2(context, DTYPE_FLOAT16, shape_d, matrix_d.get(), alpha, op_a, shape_a, matrix_a.get(), op_b, shape_b, matrix_b.get(), beta,
////					shape_d, matrix_d.get());
////			cpu_gemm(context, DTYPE_FLOAT32, shape_d, matrix_d.get(), shape_a, matrix_a.get(), shape_b, matrix_b.get(), op_a, op_b, alpha, beta);
//			if ((getTime() - start) > 10.0)
//				break;
//		}
//		const double stop = getTime();
//		std::cout << 1.0e6 * (stop - start) / i << " us (" << i << " repeats)\n";
//		const double flops = (double) i * ((double) B * (double) M * (double) N * (double) K) / (stop - start);
//		std::cout << flops / 1.0e9 << " GFLOPS\n";
//
////		cpu_gemm_v2(context, DTYPE_FLOAT32, shape_d, matrix_d.get(), alpha, op_a, shape_a, matrix_a.get(), op_b, shape_b, matrix_b.get(), beta,
////				shape_c, matrix_c.get());
//
////		std::cout << "Correct\n";
////		for (int row = 0; row < M; row++)
////		{
////			for (int col = 0; col < N; col++)
////				std::cout << half_to_float(correct_d[row * N + col]) << ' ';
////			std::cout << '\n';
////		}
////		std::cout << "-------------------------------------------\n";
////		std::cout << "Actual\n";
////		for (int row = 0; row < M; row++)
////		{
////			for (int col = 0; col < N; col++)
////				std::cout << half_to_float(matrix_c[row * N + col]) << ' ';
////			std::cout << '\n';
////		}
//
//		double diff = 0.0;
//		float max_diff = 0.0;
//		for (int i = 0; i < B * M * N; i++)
//		{
//			diff += std::fabs(matrix_c[i] - correct_d[i]);
//			max_diff = std::max(max_diff, std::fabs(matrix_c[i] - correct_d[i]));
////			diff += std::fabs(half_to_float(matrix_c[i]) - half_to_float(correct_d[i]));
////			max_diff = std::max(max_diff, std::fabs(half_to_float(matrix_c[i]) - half_to_float(correct_d[i])));
////			if (std::fabs(matrix_c[i] - correct_d[i]) > 1.0e-3)
////			{
////				std::cout << "correct = " << correct_d[i] << ", got = " << matrix_c[i] << " at " << (i / N) << ", " << (i % N) << '\n';
////				break;
////			}
//		}
//		std::cout << "\ndiff = " << diff / (B * M * N) << " (max = " << max_diff << ")\n";
//
//		cpu_destroy_context(context);
//		std::cout << "END" << std::endl;
//		return 0;
//	}

//	{
//		Device::setNumberOfThreads(1);
//		const int batch_size = 12;
//		const int filters = 128;
//
//		Graph graph;
//		auto x = graph.addInput( { batch_size, 15, 15, filters });
//		for (int i = 0; i < 20; i++)
//			x = graph.add(GlobalBroadcastHW("relu", true), x);
//		graph.addOutput(x);
//
////		graph.convertTo(DataType::FLOAT16);
//		graph.moveTo(Device::cpu());
//		graph.forward(batch_size);
//
//		std::cout << "starting benchmark\n";
//		const double start = getTime();
//		int repeats = 0;
//		for (; repeats < 100000; repeats++)
//		{
//			graph.forward(batch_size);
//			if ((getTime() - start) > 20.0)
//				break;
//		}
//		const double stop = getTime();
//		const double time = stop - start;
//
//		std::cout << "time = " << time << "s, repeats = " << repeats << '\n';
//		std::cout << "time per convolution = " << 1.0e3 * time / (10 * repeats * batch_size) << "ms\n";
//		return 0;
//	}
//	std::cout << "END" << std::endl;

//	{
//		constexpr int m = 6;
//		constexpr int n = 16;
//		constexpr int max_k = 320;
//		float *A = reinterpret_cast<float*>(gemm::aligned_new(2880 * m * max_k * 4, 4096));
//		float *B = reinterpret_cast<float*>(gemm::aligned_new(2880 * n * max_k * 4, 4096));
//		float *C = reinterpret_cast<float*>(gemm::aligned_new(2880 * m * n * 4, 4096));
//		float *workspace = reinterpret_cast<float*>(gemm::aligned_new(4096, 4096));
//
//		for (int i = 0; i < 2880 * m * max_k; i++)
//			A[i] = randFloat();
//		for (int i = 0; i < 2880 * n * max_k; i++)
//			B[i] = randFloat();
//		for (int i = 0; i < 2880 * m * n; i++)
//			C[i] = randFloat();
//
//		const float alpha = 1.1f;
//		const float beta = 0.0f;
//
//		float C2[m * n];
//		for (int i = 0; i < m * n; i++)
//			C2[i] = C[i];
//		gemm::gemm_def_MxN_fp32(m, n, max_k, &alpha, A + 0 * m * max_k, B + 0 * n * max_k, &beta, C2, n);
//
//		const double repeats = 1.0e7;
//		const double start = getTime();
//		for (int i = 0; i < repeats; i++)
//		{
//			const int tmp = 0; // i % 384;
////			gemm::gemm_avx2_fma_6x16_fp32(m, n, max_k, &alpha, A + tmp * m * max_k, B + tmp * n * max_k, &beta, C + tmp * m * n, n);
//			gemm::gemm_avx2_fma_6x16_fp16_fp32(m, n, max_k, &alpha, A + tmp * m * max_k, B + tmp * n * max_k, &beta, C + tmp * m * n, n, workspace);
//
////			gemm::gemm_avx_8x8_fp32(m, n, max_k, &alpha, A + tmp * m * max_k, B + tmp * n * max_k, &beta, C + tmp * m * n, n);
//
////			gemm::gemm_sse2_8x4_fp32(m, n, max_k, &alpha, A + tmp * m * max_k, B + tmp * n * max_k, &beta, C + tmp * m * n, n);
//
////			gemm::gemm_def_MxN_fp32(m, n, max_k, &alpha, A + 0 * m * max_k, B + 0 * n * max_k, &beta, C, n);
//
////			gemm::gemm_avx2_fma_5x16_fp32(m, n, max_k, &alpha, A + tmp * m * max_k, B + tmp * n * max_k, &beta, C + tmp * m * n, n);
//		}
//		const double stop = getTime();
//		std::cout << 1.0e6 * (stop - start) / repeats << " us" << '\n';
//		const double flops = repeats * (m * n * max_k) / (stop - start);
//		std::cout << flops / 1.0e9 << " GFLOPS\n";
//
//		double diff = 0.0;
//		for (int i = 0; i < m * n; i++)
//			diff += std::fabs(C[i] - C2[i]);
//		std::cout << "diff = " << diff / (m * n) << '\n';
//
//		for (int i = 0; i < m; i++)
//		{
//			for (int j = 0; j < n; j++)
//				std::cout << C2[i * n + j] << ' ';
//			std::cout << '\n';
//		}
//		std::cout << '\n';
//
//		for (int i = 0; i < m; i++)
//		{
//			for (int j = 0; j < n; j++)
//				std::cout << C[0 * m * n + i * n + j] << ' ';
//			std::cout << '\n';
//		}
//
//		std::cout << "END" << std::endl;
//
//		gemm::aligned_free(A, 4096);
//		gemm::aligned_free(B, 4096);
//		gemm::aligned_free(C, 4096);
//		gemm::aligned_free(workspace, 4096);
//		return 0;
//	}

//	ml::cpu::cpu_x86 prop;
//	prop.print();

	if (false)
	{
		Device::setNumberOfThreads(1);
		const int batch_size = 32;

		FileLoader fl("/home/maciek/Desktop/AlphaGomoku561/networks/standard_conv_8x128.bin");

		Graph graph;
		graph.load(fl.getJson()["model"], fl.getBinaryData());
		graph.setInputShape(Shape( { batch_size, 15, 15, 8 }));
		graph.moveTo(Device::opencl(0));
//		graph.convertTo(DataType::FLOAT16);

		Tensor input(graph.getInputShape(), graph.dtype(), Device::cpu());
		for (int i = 0; i < batch_size * 15 * 15; i++)
		{
			if (graph.dtype() == DataType::FLOAT32)
			{
				float *ptr = reinterpret_cast<float*>(input.data()) + i * 8;
				ptr[0] = 1.0f;
				ptr[3] = 1.0f;
				ptr[4] = 1.0f;
			}
			if (graph.dtype() == DataType::FLOAT16)
			{
				uint16_t *ptr = reinterpret_cast<uint16_t*>(input.data()) + i * 8;
				ptr[0] = 0x3c00;
				ptr[3] = 0x3c00;
				ptr[4] = 0x3c00;
			}
		}
		graph.getInput().copyFrom(graph.context(), input);
		graph.forward(batch_size);
		graph.context().synchronize();

		std::cout << "starting benchmark\n";
		const double start = getTime();
		int repeats = 0;
		for (; repeats < 10000; repeats++)
		{
			graph.forward(batch_size);
			graph.context().synchronize();
			if ((getTime() - start) > 30.0)
				break;
		}
		const double stop = getTime();
		const double time = stop - start;
		std::cout << "time = " << time << "s, repeats = " << repeats << ", n/s = " << batch_size * repeats / time << "\n";

		for (int i = 0; i < graph.getOutputShape(1).dim(1); i++)
			std::cout << "output " << i << " = " << graph.getOutput(1).get( { 0, i }) << '\n';

//		ml::cpu::ComputeConfig cfg(ml::cpu::Type::BF16, ml::cpu::Type::FP16);
//		CREATE_KERNEL_TABLE(some_kernel);

//		REGISTER_KERNEL(some_kernel, float, float);
//		REGISTER_KERNEL(some_kernel, float16, float);
//		REGISTER_KERNEL(some_kernel, sw_float16, float);

//		CALL_KERNEL(some_kernel, cfg)(nullptr, nullptr, 0);

//		ADD_FUNCTION(tmp, some_kernel, float, float);

//		tmp.call(cfg, nullptr, nullptr, 0);

//		float dst[16];
//		std::memset(dst, 0, 16 * sizeof(float));
//
//		float src[16];
//		for (int i = 0; i < 16; i++)
//			src[i] = 1 + i;
//
//		Vector<float, AUTO> vector(src);
//
//		vector = _mm256_unpacklo_ps(vector, vector);
//		vector.store(dst);
//		std::cout << vector.size() << '\n';
//		for (int i = 0; i < 16; i++)
//			std::cout << i << " : " << src[i] << " vs " << dst[i] << '\n';

	}
	std::cout << "END" << std::endl;

	return 0;
	{
		Graph graph;
		FileLoader fl("/home/maciek/alphagomoku/minml_test/minml3v7_10x128_opt.bin");
		graph.load(fl.getJson(), fl.getBinaryData());
		graph.moveTo(Device::cpu());
		graph.setInputShape(Shape( { 1, 15, 15, 32 }));
		graph.forward(1);
	}
	std::cout << "END" << std::endl;
//	return 0;

	const int blocks = 10;
	const int filters = 128;

	Graph graph;
	auto x = graph.addInput( { 256, 15, 15, 32 });
	x = graph.add(Conv2D(filters, 3, "linear").useBias(false), x);
	for (int i = 0; i < blocks; i++)
		x = graph.add(Conv2D(filters, 3, "linear").useBias(false), x);
	x = graph.add(GlobalBroadcastHW(), x);
	graph.addOutput(x);

//	x = graph.add(Conv2D(filters, 3, "linear").useBias(false), x);
////	x = graph.add(BatchNormalization("relu").useGamma(false), x);
//
//	for (int i = 0; i < blocks; i++)
//	{
//		auto y = graph.add(Conv2D(filters, 3, "linear").useBias(false), x);
////		y = graph.add(BatchNormalization("relu").useGamma(false), y);
//
//		y = graph.add(Conv2D(filters, 3, "linear").useBias(false), y);
////		y = graph.add(BatchNormalization("linear").useGamma(false), y);
//
//		x = graph.add(Add("relu"), { x, y });
//	}
//
//	// policy head
//	auto p = graph.add(Conv2D(filters, 3, "linear").useBias(false), x);
////	p = graph.add(BatchNormalization("relu").useGamma(false), p);
//	p = graph.add(ml::Conv2D(1, 1, "softmax"), p);
//	graph.addOutput(p);
//
//	// value head
//	auto v = graph.add(ml::Conv2D(2, 1, "linear").useBias(false), x);
////	v = graph.add(ml::BatchNormalization("relu").useGamma(false), v);
//
//	v = graph.add(ml::Dense(std::min(256, 2 * filters), "linear").useBias(false), v);
////	v = graph.add(ml::BatchNormalization("relu").useGamma(false), v);
//	v = graph.add(ml::Dense(3, "softmax"), v);
//	graph.addOutput(v);

	graph.init();
	graph.setOptimizer(Optimizer(1.0e-3f));
	graph.setRegularizer(Regularizer(1.0e-5f));
	graph.moveTo(Device::cuda(0));

	graph.makeNonTrainable();
	graph.convertTo(DataType::FLOAT16);

	graph.print();

	graph.forward(256);
	graph.context().synchronize();
//	return 0;

	double start = getTime();
	for (int i = 0; i < 1000; i++)
	{
		graph.forward(256);
		graph.context().synchronize();
	}
	double stop = getTime();
	std::cout << (stop - start) << '\n';

	std::cout << "END" << std::endl;
	return 0;
}
