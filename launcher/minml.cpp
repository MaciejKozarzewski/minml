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
#include <minml/graph/Graph.hpp>
#include <minml/graph/graph_optimizers.hpp>
#include <minml/layers/Dense.hpp>
#include <minml/layers/Conv2D.hpp>
#include <minml/layers/BatchNormalization.hpp>
#include <minml/layers/GlobalBroadcastHW.hpp>
#include <minml/layers/Add.hpp>
#include <minml/layers/Softmax.hpp>
#include <minml/training/Optimizer.hpp>
#include <minml/utils/random.hpp>
#include <minml/utils/time_util.hpp>
#include <minml/utils/file_util.hpp>

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>
#include <x86intrin.h>

#include "../src/backend/cpu/vectors/vectors.hpp"

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
			for (int i = 0; i < input.firstDim(); i++)
			{
				const int index = (sample_index == -1) ? randInt(train_images.firstDim()) : sample_index;
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

	x = model.add(Dense(32, "linear").useBias(true), x);
	x = model.add(BatchNormalization("relu").useGamma(true), x);
	x = model.add(Dense(10, "linear"), x);
	x = model.add(Softmax( { 1 }), x);
	model.addOutput(x);

	model.init();
	model.setOptimizer(Optimizer(1.0e-3f));
	model.setRegularizer(Regularizer(1.0e-4f));
//	model.moveTo(Device::cuda(1));
	model.moveTo(Device::cpu());
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

	const int steps = 100;
	for (int e = 0; e < 100; e++)
	{
		if (e == 75)
			model.setLearningRate(1.0e-4f);
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
				// beta != 0 case
				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm4 \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm6 \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm7 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm8 \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm9 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm10 \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm11 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm12 \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"vfmadd231ps 0x00(%%rcx), %%ymm1, %%ymm14 \n\t"
				"vfmadd231ps 0x20(%%rcx), %%ymm1, %%ymm15 \n\t"
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"BETAZERO%=: \n\t"
				// beta == 0 case
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
				"movaps 0x00(%%rbx), %%xmm6 \n\t"
				"movaps 0x10(%%rbx), %%xmm7 \n\t"

				"pshufd $0x55, %%xmm0, %%xmm1 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm2 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"mulps %%xmm6, %%xmm1 \n\t"
				"mulps %%xmm7, %%xmm2 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm1, %%xmm14 \n\t"
				"addps %%xmm2, %%xmm15 \n\t"

				"pshufd $0xFF, %%xmm0, %%xmm3 \n\t"
				"pshufd $0x00, %%xmm0, %%xmm0 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"mulps %%xmm6, %%xmm0 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm0, %%xmm14 \n\t"
				"addps %%xmm3, %%xmm15 \n\t"

				// iteration 1
				"movaps 0x10(%%rax), %%xmm0 \n\t"
				"movaps 0x20(%%rbx), %%xmm6 \n\t"
				"movaps 0x30(%%rbx), %%xmm7 \n\t"

				"pshufd $0x55, %%xmm0, %%xmm1 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm2 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"mulps %%xmm6, %%xmm1 \n\t"
				"mulps %%xmm7, %%xmm2 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm1, %%xmm14 \n\t"
				"addps %%xmm2, %%xmm15 \n\t"

				"pshufd $0xFF, %%xmm0, %%xmm3 \n\t"
				"pshufd $0x00, %%xmm0, %%xmm0 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"mulps %%xmm6, %%xmm0 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm0, %%xmm14 \n\t"
				"addps %%xmm3, %%xmm15 \n\t"

				// iteration 2
				"movaps 0x20(%%rax), %%xmm0 \n\t"
				"movaps 0x40(%%rbx), %%xmm6 \n\t"
				"movaps 0x50(%%rbx), %%xmm7 \n\t"

				"pshufd $0x55, %%xmm0, %%xmm1 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm2 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"mulps %%xmm6, %%xmm1 \n\t"
				"mulps %%xmm7, %%xmm2 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm1, %%xmm14 \n\t"
				"addps %%xmm2, %%xmm15 \n\t"

				"pshufd $0xFF, %%xmm0, %%xmm3 \n\t"
				"pshufd $0x00, %%xmm0, %%xmm0 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"mulps %%xmm6, %%xmm0 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm0, %%xmm14 \n\t"
				"addps %%xmm3, %%xmm15 \n\t"

				// iteration 3
				"movaps 0x30(%%rax), %%xmm0 \n\t"
				"movaps 0x60(%%rbx), %%xmm6 \n\t"
				"movaps 0x70(%%rbx), %%xmm7 \n\t"

				"pshufd $0x55, %%xmm0, %%xmm1 \n\t"
				"pshufd $0xAA, %%xmm0, %%xmm2 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"mulps %%xmm6, %%xmm1 \n\t"
				"mulps %%xmm7, %%xmm2 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm1, %%xmm14 \n\t"
				"addps %%xmm2, %%xmm15 \n\t"

				"pshufd $0xFF, %%xmm0, %%xmm3 \n\t"
				"pshufd $0x00, %%xmm0, %%xmm0 \n\t"
				"mulps %%xmm6, %%xmm4 \n\t"
				"mulps %%xmm7, %%xmm5 \n\t"
				"mulps %%xmm6, %%xmm0 \n\t"
				"mulps %%xmm7, %%xmm3 \n\t"
				"addps %%xmm4, %%xmm12 \n\t"
				"addps %%xmm5, %%xmm13 \n\t"
				"addps %%xmm0, %%xmm14 \n\t"
				"addps %%xmm3, %%xmm15 \n\t"

				"add $0x40, %%rax \n\t"
				"add $0x80, %%rbx \n\t"
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
				"shlq $1, %%r12 \n\t"// multiply stride by sizeof(float16)
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"vpxor %%ymm0, %%ymm0, %%ymm0 \n\t"
				"ucomiss %%xmm0, %%xmm1 \n\t"// set ZF if beta == 0.
				"je BETAZERO%= \n\t"
				// beta != 0 case
				"movups 0x00(%%rcx), %%xmm2 \n\t"
				"movups 0x10(%%rcx), %%xmm3 \n\t"
				"vcvtph2ps %%xmm2, %%ymm2 \n\t"
				"vcvtph2ps %%xmm3, %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm4 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm5 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups 0x00(%%rcx), %%xmm2 \n\t"
				"movups 0x10(%%rcx), %%xmm3 \n\t"
				"vcvtph2ps %%xmm2, %%ymm2 \n\t"
				"vcvtph2ps %%xmm3, %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm6 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm7 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups 0x00(%%rcx), %%xmm2 \n\t"
				"movups 0x10(%%rcx), %%xmm3 \n\t"
				"vcvtph2ps %%xmm2, %%ymm2 \n\t"
				"vcvtph2ps %%xmm3, %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm8 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm9 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups 0x00(%%rcx), %%xmm2 \n\t"
				"movups 0x10(%%rcx), %%xmm3 \n\t"
				"vcvtph2ps %%xmm2, %%ymm2 \n\t"
				"vcvtph2ps %%xmm3, %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm10 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm11 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups 0x00(%%rcx), %%xmm2 \n\t"
				"movups 0x10(%%rcx), %%xmm3 \n\t"
				"vcvtph2ps %%xmm2, %%ymm2 \n\t"
				"vcvtph2ps %%xmm3, %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm12 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm13 \n\t"
				"add %%r12, %%rcx \n\t"// add stride

				"movups 0x00(%%rcx), %%xmm2 \n\t"
				"movups 0x10(%%rcx), %%xmm3 \n\t"
				"vcvtph2ps %%xmm2, %%ymm2 \n\t"
				"vcvtph2ps %%xmm3, %%ymm3 \n\t"
				"vfmadd231ps %%ymm2, %%ymm1, %%ymm14 \n\t"
				"vfmadd231ps %%ymm3, %%ymm1, %%ymm15 \n\t"
				"movq %[dst_ptr], %%rcx \n\t"// dst pointer is in rcx

				"BETAZERO%=: \n\t"
				// beta == 0 case
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

#include "../src/backend/cpu/utils.hpp"

template<typename T>
std::string print_type()
{
	if (std::is_same<T, sw_bfloat16>::value)
		return "sw_bf16";
	if (std::is_same<T, bfloat16>::value)
		return "bf16";
	if (std::is_same<T, sw_float16>::value)
		return "sw_fp16";
	if (std::is_same<T, float16>::value)
		return "fp16";
	if (std::is_same<T, float>::value)
		return "fp32";
	if (std::is_same<T, int>::value)
		return "int32";
	return "none";
}

template<typename DT, typename CT>
void some_kernel(const void *input, void *output, int size)
{
	std::cout << "data = " << print_type<DT>() << ", compute = " << print_type<CT>() << '\n';
}

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)

//#define CREATE_KERNEL_TABLE(name) static auto name##_table = createFunctionTable(name<float, float>);	\
//		name##_table.get(ml::cpu::Type::SW_BF16, ml::cpu::Type::FP32) = name<sw_bfloat16, float>;		\
//		name##_table.get(ml::cpu::Type::BF16, ml::cpu::Type::FP32) = name<bfloat16, float>;				\
//		name##_table.get(ml::cpu::Type::BF16, ml::cpu::Type::BF16) = name<bfloat16, bfloat16>;			\
//		name##_table.get(ml::cpu::Type::SW_FP16, ml::cpu::Type::FP32) = name<sw_float16, float>;		\
//		name##_table.get(ml::cpu::Type::FP16, ml::cpu::Type::FP32) = name<float16, float>;				\
//		name##_table.get(ml::cpu::Type::FP16, ml::cpu::Type::FP16) = name<float16, float16>;			\
//		name##_table.get(ml::cpu::Type::FP32, ml::cpu::Type::FP32) = name<float, float>
//
//#define REGISTER_KERNEL(name, dtype, ctype) name##_table.get(get_type<dtype>(), get_type<ctype>()) = name<dtype, ctype>
//#define DISABLE_KERNEL(name, dtype, ctype) name##_table.get(get_type<dtype>(), get_type<ctype>()) = nullptr
//
//#define CALL_KERNEL(name, cfg) name##_table.get(cfg.data_type, cfg.compute_type)

#include "../src/backend/cpu/cpu_x86.hpp"
#include "../src/backend/cpu/vectors/vectors.hpp"

using namespace SIMD_NAMESPACE;
int main()
{
	std::cout << "BEGIN" << std::endl;
	{
		constexpr int m = 8;
		constexpr int n = 4;
		constexpr int max_k = 1;
		float *A = reinterpret_cast<float*>(gemm::aligned_new(2880 * m * max_k * 4, 4096));
		float *B = reinterpret_cast<float*>(gemm::aligned_new(2880 * n * max_k * 4, 4096));
		float *C = reinterpret_cast<float*>(gemm::aligned_new(2880 * m * n * 4, 4096));
		float *workspace = reinterpret_cast<float*>(gemm::aligned_new(4096, 4096));

		for (int i = 0; i < 2880 * m * max_k; i++)
			A[i] = randFloat();
		for (int i = 0; i < 2880 * n * max_k; i++)
			B[i] = randFloat();
		for (int i = 0; i < 2880 * m * n; i++)
			C[i] = randFloat();

		const float alpha = 1.0f;
		const float beta = 0.1f;

		float C2[m * n];
		for (int i = 0; i < m * n; i++)
			C2[i] = C[i];
		gemm::gemm_def_MxN_fp32(m, n, max_k, &alpha, A + 0 * m * max_k, B + 0 * n * max_k, &beta, C2, n);

		const double repeats = 1.0e7;
		const double start = getTime();
//		for (int i = 0; i < repeats; i++)
		{
			const int tmp = 0; //i % 384;
//			gemm::gemm_avx2_fma_6x16_fp32(m, n, max_k, &alpha, A + tmp * m * max_k, B + tmp * n * max_k, &beta, C + tmp * m * n, n);
//			gemm::gemm_avx2_fma_6x16_fp16_fp32(m, n, max_k, &alpha, A + tmp * m * max_k, B + tmp * n * max_k, &beta, C + tmp * m * n, n);

//			gemm::gemm_avx_8x8_fp32(m, n, max_k, &alpha, A + tmp * m * max_k, B + tmp * n * max_k, &beta, C + tmp * m * n, n);

			gemm::gemm_sse2_8x4_fp32(m, n, max_k, &alpha, A + tmp * m * max_k, B + tmp * n * max_k, &beta, C + tmp * m * n, n);

//			gemm::gemm_avx2_fma_5x16_fp32(5, 16, max_k, &alpha, A + tmp * 6 * max_k, B + tmp * 16 * max_k, &beta, C + tmp * 6 * 16, 16);
		}
		const double stop = getTime();
		std::cout << 1.0e6 * (stop - start) / repeats << " us" << '\n';
		const double flops = repeats * (m * n * max_k) / (stop - start);
		std::cout << flops / 1.0e9 << " GFLOPS\n";

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
				std::cout << C2[i * n + j] << ' ';
			std::cout << '\n';
		}
		std::cout << '\n';

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
				std::cout << C[0 * m * n + i * n + j] << ' ';
			std::cout << '\n';
		}

		std::cout << "END" << std::endl;

		gemm::aligned_free(A, 4096);
		gemm::aligned_free(B, 4096);
		gemm::aligned_free(C, 4096);
		gemm::aligned_free(workspace, 4096);
		return 0;
	}

//	ml::cpu::cpu_x86 prop;
//	prop.print();

	{
		Device::setNumberOfThreads(1);
		const int batch_size = 12;
		const int filters = 128;

		Graph graph;
		auto x = graph.addInput( { batch_size, 15, 15, filters });
		for (int i = 0; i < 10; i++)
			x = graph.add(Conv2D(filters, 3, "linear"), x);
		graph.addOutput(x);

		graph.convertTo(DataType::FLOAT16);
		graph.moveTo(Device::cpu());
		graph.forward(batch_size);

		std::cout << "starting benchmark\n";
		const double start = getTime();
		int repeats = 0;
		for (; repeats < 10000; repeats++)
		{
			graph.forward(batch_size);
			if ((getTime() - start) > 10.0)
				break;
		}
		const double stop = getTime();
		const double time = stop - start;

		std::cout << "time = " << time << "s, repeats = " << repeats << '\n';
		std::cout << "time per convolution = " << 1.0e3 * time / (10 * repeats * batch_size) << "ms\n";

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
