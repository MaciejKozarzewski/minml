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
					reinterpret_cast<float*>(dst_ptr)[m * dst_stride + n] += acc[m * N + n];
		}
		else
		{
			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					reinterpret_cast<float*>(dst_ptr)[m * dst_stride + n] = beta * reinterpret_cast<float*>(dst_ptr)[m * dst_stride + n]
							+ acc[m * N + n];
		}
	}

	void gemm_avx2_fma_6x16_fp32(int M, int N, int K, const void *alpha_ptr, const void *__restrict__ lhs_ptr, const void *__restrict__ rhs_ptr,
			const void *beta_ptr, void *__restrict__ dst_ptr, int dst_stride)
	{
		assert(M == 6);
		assert(N == 16);

		uint64_t k_iter = K / 4;
		uint64_t k_left = K % 4;
		uint64_t stride = dst_stride;
		asm volatile(
				"movq  %[stride], %%r12 \n\t" // stride is r12
				"movq  %[lhs_ptr], %%rax \n\t" // lhs pointer is in rax
				"movq  %[rhs_ptr], %%rbx \n\t" // rhs pointer is in rbx
				"shlq $2, %%r12 \n\t"// multiply stride by sizeof(float)

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

				"movq  %[k_iter], %%r14 \n\t"// load the number of 4-unrolled iterations
				"test %%r14, %%r14 \n\t"
				"je unrolled4 \n\t"

				"unrolled4: \n\t"
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

				:// outputs
				[lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [dst_ptr] "+r"(dst_ptr)
				:// inputs
				[k_iter] "r"(k_iter), [k_left] "r"(k_left), [stride] "r"(stride)
				:// clobbers
				"cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
				"%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%r12",
				"%r13", "%r14");
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
