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
#include <minml/layers/Add.hpp>
#include <minml/training/Optimizer.hpp>
#include <minml/utils/random.hpp>
#include <minml/utils/time_util.hpp>
#include <minml/utils/file_util.hpp>

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>
#include <x86intrin.h>

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

	const int batch_size = 32;

	Graph model;
	auto x = model.addInput( { batch_size, 28, 28, 1 });
	x = model.add(Conv2D(8, 3, "relu"), x);
//	x = model.add(BatchNormalization("relu").useGamma(false), x);

//	auto y = model.add(Conv2D(32, 3, "linear"), x);
//	y = model.add(BatchNormalization("linear").useGamma(false), y);
//	x = model.add(Add("relu"), { x, y });
//	x = model.add(Conv2D(31, 1, "relu"), x);

//	x = model.add(Dense(32, "linear").useBias(false), x);
//	x = model.add(BatchNormalization("relu").useGamma(false), x);
	x = model.add(Dense(10, "softmax"), x);
	model.addOutput(x);

	model.init();
//	model.setOptimizer(Optimizer(1.0e-3f));
//	model.setRegularizer(Regularizer(1.0e-5f));
//	model.moveTo(Device::cuda(1));
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

int main()
{
	std::cout << "BEGIN" << std::endl;
//	test_mnist();
	{
		Graph graph;
		FileLoader fl("/home/maciek/alphagomoku/minml_test/minml3v7_10x128_opt.bin");
		graph.load(fl.getJson(), fl.getBinaryData());
		graph.moveTo(Device::cpu());
		graph.setInputShape(Shape( { 1, 15, 15, 32 }));
		graph.forward(1);
	}
	std::cout << "END" << std::endl;
	return 0;

	const int blocks = 10;
	const int filters = 128;

	Graph graph;
	auto x = graph.addInput( { 256, 15, 15, 32 });
	x = graph.add(Conv2D(filters, 3, "linear").useBias(false), x);
	for (int i = 0; i < blocks; i++)
		x = graph.add(Conv2D(filters, 3, "linear").useBias(false), x);
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
