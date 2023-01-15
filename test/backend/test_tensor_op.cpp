/*
 * test_tensor_op.cpp
 *
 *  Created on: Sep 20, 2020
 *      Author: Maciej Kozarzewski
 */

//#include <libml/math/tensor_op.hpp>
//#include <libml/math/activations.hpp>
//#include <libml/hardware/DeviceContext.hpp>
//#include <libml/Tensor.hpp>
//#include <libml/Scalar.hpp>
//#include <libml/utils/testing_util.hpp>
//
//#include <gtest/gtest.h>
//
//namespace
//{
//	using namespace ml;
//
//	template<typename T>
//	void baseline_add_to_tensor(Tensor &dst, const Scalar &src)
//	{
//		const T value = src.get<T>();
//		for (int i = 0; i < dst.volume(); i++)
//			dst.data<T>()[i] += value;
//	}
//	template<typename T>
//	void baseline_add_to_last_dim(Tensor &dst, const Tensor &src, NonlinearityType act)
//	{
//		assert(same_device(dst, src));
//		assert(dst.device().isCPU());
//		const int first_dim = dst.shape().volumeWithoutLastDim();
//		const int last_dim = dst.shape().lastDim();
//		for (int i = 0; i < first_dim; i++)
//			for (int j = 0; j < last_dim; j++)
//				dst.data<T>()[i * last_dim + j] += src.data<T>()[j];
//		math::nonlinearityForwardInPlace(DeviceContext(), dst, act);
//	}
//	template<typename T>
//	void baseline_sum_over_first_dim(Tensor &dst, const Tensor &src, T beta)
//	{
//		assert(same_device(dst, src));
//		assert(dst.device().isCPU());
//		const int first_dim = src.shape().volumeWithoutLastDim();
//		const int last_dim = src.shape().lastDim();
//		for (int j = 0; j < last_dim; j++)
//		{
//			T acc = static_cast<T>(0);
//			for (int i = 0; i < first_dim; i++)
//				acc += src.data<T>()[i * last_dim + j];
//			if (beta == 0)
//				dst.data<T>()[j] = acc;
//			else
//				dst.data<T>()[j] = dst.data<T>()[j] * beta + acc;
//		}
//	}
//	template<typename T>
//	void baseline_add_tensors(Tensor &dst, const std::vector<Tensor> &src, NonlinearityType act)
//	{
//		for (int i = 0; i < dst.volume(); i++)
//		{
//			T acc = 0;
//			for (size_t j = 0; j < src.size(); j++)
//				acc += src[j].data<T>()[i];
//			dst.data<T>()[i] = acc;
//		}
//		math::nonlinearityForwardInPlace(DeviceContext(), dst, act);
//	}
//	template<typename T>
//	void baseline_concat_tensors(Tensor &dst, const std::vector<Tensor> &src)
//	{
//		const int first_dim = dst.shape().volumeWithoutLastDim();
//		const int last_dim = dst.shape().lastDim();
//
//		int offset = 0;
//		for (size_t i = 0; i < src.size(); i++)
//		{
//			for (int j = 0; j < first_dim; j++)
//				for (int k = 0; k < src[i].lastDim(); k++)
//					dst.data<T>()[j * last_dim + offset + k] = src[i].data<T>()[j * src[i].lastDim() + k];
//			offset += src[i].lastDim();
//		}
//	}
//	template<typename T>
//	void baseline_split_tensors(std::vector<Tensor> &dst, const Tensor &src)
//	{
//		const int first_dim = src.shape().volumeWithoutLastDim();
//		const int last_dim = src.shape().lastDim();
//
//		int offset = 0;
//		for (size_t i = 0; i < dst.size(); i++)
//		{
//			for (int j = 0; j < first_dim; j++)
//				for (int k = 0; k < dst[i].lastDim(); k++)
//					dst[i].data<T>()[j * dst[i].lastDim() + k] = src.data<T>()[j * last_dim + offset + k];
//			offset += dst[i].lastDim();
//		}
//	}
//}
//
//namespace ml
//{
//	TEST(TestTensorOp, setall_cpu)
//	{
//		Tensor t( { 12 }, "float32", Device::cpu());
//		Scalar s(1.23);
//
//		math::setall(DeviceContext(), t, s);
//		for (int i = 0; i < t.volume(); i++)
//			EXPECT_EQ(t.get<float>( { i }), 1.23f);
//	}
//	TEST(TestTensorOp, setall_cuda)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		Tensor t( { 12 }, "float32", Device::cuda(0));
//		Scalar s(1.23);
//
//		math::setall(DeviceContext(Device::cuda(0)), t, s);
//		for (int i = 0; i < t.volume(); i++)
//			EXPECT_EQ(t.get<float>( { i }), 1.23f);
//	}
//
//	TEST(TestTensorOp, addToLastDim)
//	{
//		Tensor correct( { 123, 14 }, DataType::FLOAT32, Device::cpu());
//		Tensor dst( { 123, 14 }, DataType::FLOAT32, Device::cpu());
//		Tensor src( { 14 }, DataType::FLOAT32, Device::cpu());
//		testing::initForTest(src, 0.0f);
//		testing::initForTest(dst, 1.57f);
//		testing::initForTest(correct, 1.57f);
//
//		baseline_add_to_last_dim<float>(correct, src, NonlinearityType::RELU);
//
//		math::addToLastDim(DeviceContext(), dst, src, NonlinearityType::RELU);
//		EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			testing::initForTest(dst, 1.57f);
//			src.moveTo(Device::cuda(0));
//			dst.moveTo(Device::cuda(0));
//			math::addToLastDim(DeviceContext(Device::cuda(0)), dst, src, NonlinearityType::RELU);
//			EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//		}
//	}
//	TEST(TestTensorOp, sumOverLastDim_no_workspace)
//	{
//		double beta = 2.1;
//		Tensor src( { 123, 34 }, DataType::FLOAT32, Device::cpu());
//		Tensor correct( { 34 }, DataType::FLOAT32, Device::cpu());
//		Tensor dst( { 34 }, DataType::FLOAT32, Device::cpu());
//		Tensor empty;
//		testing::initForTest(src, 0.0f);
//		testing::initForTest(correct, 1.0f);
//		testing::initForTest(dst, 1.0f);
//
//		baseline_sum_over_first_dim<float>(correct, src, beta);
//
//		math::sumOverFirstDim(DeviceContext(), dst, src, empty, beta);
//		EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//
//		if (Device::numberOfCudaDevices() > 10)
//		{
//			testing::initForTest(dst, 1.0f);
//			src.moveTo(Device::cuda(0));
//			dst.moveTo(Device::cuda(0));
//			math::sumOverFirstDim(DeviceContext(Device::cuda(0)), dst, src, empty, beta);
//			EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//		}
//	}
//	TEST(TestTensorOp, sumOverLastDim_with_workspace)
//	{
//		double beta = 2.1;
//		Tensor src( { 567, 34 }, DataType::FLOAT32, Device::cpu());
//		Tensor correct( { 34 }, DataType::FLOAT32, Device::cpu());
//		Tensor dst( { 34 }, DataType::FLOAT32, Device::cpu());
//		Tensor storage( { 16, 34 }, DataType::FLOAT32, Device::cpu());
//		testing::initForTest(src, 0.0f);
//		testing::initForTest(correct, 1.0f);
//		testing::initForTest(dst, 1.0f);
//
//		baseline_sum_over_first_dim<float>(correct, src, beta);
//
//		math::sumOverFirstDim(DeviceContext(), dst, src, storage, beta);
//		EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			testing::initForTest(dst, 1.0f);
//			src.moveTo(Device::cuda(0));
//			dst.moveTo(Device::cuda(0));
//			storage.moveTo(Device::cuda(0));
//			math::sumOverFirstDim(DeviceContext(Device::cuda(0)), dst, src, storage, beta);
//			EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//		}
//	}
//	TEST(TestTensorOp, addTensors)
//	{
//		Tensor correct( { 123, 14 }, DataType::FLOAT32, Device::cpu());
//		Tensor dst( { 123, 14 }, DataType::FLOAT32, Device::cpu());
//		std::vector<Tensor> src = { Tensor(Shape( { 123, 14 }), DataType::FLOAT32, Device::cpu()), Tensor(Shape( { 123, 14 }), DataType::FLOAT32,
//				Device::cpu()), Tensor(Shape( { 123, 14 }), DataType::FLOAT32, Device::cpu()) };
//		testing::initForTest(src[0], 0.0f);
//		testing::initForTest(src[1], 1.0f);
//		testing::initForTest(src[2], 2.0f);
//
//		baseline_add_tensors<float>(correct, src, NonlinearityType::RELU);
//		math::addTensors(DeviceContext(), dst, src, NonlinearityType::RELU);
//		EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			DeviceContext context(Device::cuda(0));
//			src[0].moveTo(Device::cuda(0));
//			src[1].moveTo(Device::cuda(0));
//			src[2].moveTo(Device::cuda(0));
//			dst.moveTo(Device::cuda(0));
//			dst.zeroall(context);
//			math::addTensors(context, dst, src, NonlinearityType::RELU);
//			EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//		}
//	}
//	TEST(TestTensorOp, addToTensor)
//	{
//		Shape ts( { 12, 34 });
//		Tensor correct(ts, DataType::FLOAT32, Device::cpu());
//		testing::initForTest(correct, 0.0f);
//		baseline_add_to_tensor<float>(correct, 1.234f);
//
//		Tensor dst(ts, DataType::FLOAT32, Device::cpu());
//		testing::initForTest(dst, 0.0f);
//		math::addScalarToTensor(DeviceContext(), dst, 1.234f);
//		EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			testing::initForTest(dst, 0.0f);
//			dst.moveTo(Device::cuda(0));
//			math::addScalarToTensor(DeviceContext(Device::cuda(0)), dst, 1.234f);
//			EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//		}
//	}
//
//	TEST(TestTensorOp, concatTensors)
//	{
//		Tensor correct( { 123, 40 }, DataType::FLOAT32, Device::cpu());
//		Tensor dst( { 123, 40 }, DataType::FLOAT32, Device::cpu());
//		std::vector<Tensor> src = { Tensor(Shape( { 123, 10 }), DataType::FLOAT32, Device::cpu()), Tensor(Shape( { 123, 13 }), DataType::FLOAT32,
//				Device::cpu()), Tensor(Shape( { 123, 17 }), DataType::FLOAT32, Device::cpu()) };
//		testing::initForTest(src[0], 0.0f);
//		testing::initForTest(src[1], 1.0f);
//		testing::initForTest(src[2], 2.0f);
//
//		baseline_concat_tensors<float>(correct, src);
//		math::concatTensors(DeviceContext(), dst, src);
//		EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			DeviceContext context(Device::cuda(0));
//			src[0].moveTo(Device::cuda(0));
//			src[1].moveTo(Device::cuda(0));
//			src[2].moveTo(Device::cuda(0));
//			dst.moveTo(Device::cuda(0));
//			dst.zeroall(context);
//			math::concatTensors(context, dst, src);
//			EXPECT_LE(testing::diffForTest(correct, dst), 1.0e-6);
//		}
//	}
//	TEST(TestTensorOp, splitTensors)
//	{
//		Tensor src( { 123, 30 }, DataType::FLOAT32, Device::cpu());
//		testing::initForTest(src, 0.0f);
//		std::vector<Tensor> correct = { Tensor(Shape( { 123, 17 }), DataType::FLOAT32, Device::cpu()), Tensor(Shape( { 123, 13 }), DataType::FLOAT32,
//				Device::cpu()) };
//		std::vector<Tensor> dest = { Tensor(Shape( { 123, 17 }), DataType::FLOAT32, Device::cpu()), Tensor(Shape( { 123, 13 }), DataType::FLOAT32,
//				Device::cpu()) };
//
//		baseline_split_tensors<float>(correct, src);
//		math::splitTensors(DeviceContext(), dest, src);
//		EXPECT_LE(testing::diffForTest(correct[0], dest[0]), 1.0e-6);
//		EXPECT_LE(testing::diffForTest(correct[1], dest[1]), 1.0e-6);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			DeviceContext context(Device::cuda(0));
//			dest[0].moveTo(Device::cuda(0));
//			dest[1].moveTo(Device::cuda(0));
//			dest[0].zeroall(context);
//			dest[1].zeroall(context);
//			src.moveTo(Device::cuda(0));
//			math::splitTensors(context, dest, src);
//			EXPECT_LE(testing::diffForTest(correct[0], dest[0]), 1.0e-6);
//			EXPECT_LE(testing::diffForTest(correct[1], dest[1]), 1.0e-6);
//		}
//	}
//
//	TEST(TestTensorOp, transpose2D)
//	{
//		Tensor src( { 10, 23 }, "float32", Device::cpu());
//		Tensor dst( { 23, 10 }, "float32", Device::cpu());
//		testing::initForTest(src, 0.0f);
//		math::transpose(DeviceContext(), dst, src, { 1, 0 });
//
//		float diff = 0.0f;
//		for (int i = 0; i < src.shape(0); i++)
//			for (int j = 0; j < src.shape(1); j++)
//				diff += fabsf(src.get<float>( { i, j }) - dst.get<float>( { j, i }));
//		EXPECT_FLOAT_EQ(diff, 0.0f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			src.moveTo(Device::cuda(0));
//			Tensor dst2(dst.shape(), "float32", Device::cuda(0));
//			math::transpose(DeviceContext(Device::cuda(0)), dst2, src, { 1, 0 });
//			EXPECT_FLOAT_EQ(testing::diffForTest(dst, dst2), 0.0f);
//		}
//	}
//	TEST(TestTensorOp, transpose3D)
//	{
//		Tensor src( { 10, 15, 23 }, "float32", Device::cpu());
//		Tensor dst( { 23, 10, 15 }, "float32", Device::cpu());
//		testing::initForTest(src, 0.0f);
//		math::transpose(DeviceContext(), dst, src, { 2, 0, 1 });
//
//		float diff = 0.0f;
//		for (int i = 0; i < src.shape(0); i++)
//			for (int j = 0; j < src.shape(1); j++)
//				for (int k = 0; k < src.shape(2); k++)
//					diff += fabsf(src.get<float>( { i, j, k }) - dst.get<float>( { k, i, j }));
//		EXPECT_FLOAT_EQ(diff, 0.0f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			src.moveTo(Device::cuda(0));
//			Tensor dst2(dst.shape(), "float32", Device::cuda(0));
//			math::transpose(DeviceContext(Device::cuda(0)), dst2, src, { 2, 0, 1 });
//			EXPECT_FLOAT_EQ(testing::diffForTest(dst, dst2), 0.0f);
//		}
//	}
//	TEST(TestTensorOp, transpose4D)
//	{
//		Tensor src( { 10, 15, 20, 23 }, "float32", Device::cpu());
//		Tensor dst( { 20, 10, 23, 15 }, "float32", Device::cpu());
//		testing::initForTest(src, 0.0f);
//		math::transpose(DeviceContext(), dst, src, { 2, 0, 3, 1 });
//
//		float diff = 0.0f;
//		for (int i = 0; i < src.shape(0); i++)
//			for (int j = 0; j < src.shape(1); j++)
//				for (int k = 0; k < src.shape(2); k++)
//					for (int l = 0; l < src.shape(3); l++)
//						diff += fabsf(src.get<float>( { i, j, k, l }) - dst.get<float>( { k, i, l, j }));
//		EXPECT_FLOAT_EQ(diff, 0.0f);
//
//		if (Device::numberOfCudaDevices() > 0)
//		{
//			src.moveTo(Device::cuda(0));
//			Tensor dst2(dst.shape(), "float32", Device::cuda(0));
//			math::transpose(DeviceContext(Device::cuda(0)), dst2, src, { 2, 0, 3, 1 });
//			EXPECT_FLOAT_EQ(testing::diffForTest(dst, dst2), 0.0f);
//		}
//	}
//} /* namespace ml */

