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

#include <iostream>
#include <fstream>
#include <memory>

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

void test_mnist()
{
	std::cout << Device::hardwareInfo();
//	return;

	Device::setNumberOfThreads(1);
	MNIST dataset;

	const int batch_size = 32;

	Graph model;
	auto x = model.addInput( { batch_size, 28, 28, 1 });
	x = model.add(Conv2D(32, 5, "relu"), x);
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
	model.setOptimizer(Optimizer(1.0e-3f));
	model.setRegularizer(Regularizer(1.0e-5f));
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

	for (int e = 0; e < 10; e++)
	{
		double loss = 0.0;
		double acc = 0.0;
		for (int step = 0; step < 1000; step++)
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
		std::cout << "epoch " << e << ", loss = " << loss / 1000 << ", accuracy = " << acc / (1000 * batch_size) << '\n';

//		SerializedObject so;
//		Json json = model.save(so);
//		FileSaver fs("/home/maciek/cpp_workspace/libml/mnist.bin");
//		fs.save(json, so);
		if (loss != loss)
			break;
	}
	return;
	model.makeNonTrainable();

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
	test_mnist();
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
