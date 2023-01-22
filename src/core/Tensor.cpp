/*
 * Tensor.cpp
 *
 *  Created on: Aug 18, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Tensor.hpp>
#include <minml/core/Context.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>
#include <minml/core/ml_memory.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/core/math.hpp>

#include <minml/backend/cuda_backend.h>

#include <cassert>
#include <memory>
#include <algorithm>

namespace ml
{

	Tensor::Tensor(const Shape &shape, DataType dtype, Device device) :
			m_device(device),
			m_shape(shape),
			m_dtype(dtype),
			m_is_owning(true)
	{
		m_data = ml::malloc(device, sizeInBytes());
		ml::memzero(device, m_data, 0, sizeInBytes());
		create_stride();
	}
	Tensor::Tensor(const Shape &shape, const std::string &dtype, Device device) :
			Tensor(shape, typeFromString(dtype), device)
	{
	}
	Tensor::Tensor(const Json &json, const SerializedObject &binary_data)
	{
		unserialize(json, binary_data);
	}

	Tensor::Tensor(const Tensor &other) :
			m_device(other.m_device),
			m_shape(other.m_shape),
			m_dtype(other.m_dtype),
			m_is_owning(other.m_is_owning)
	{
		create_stride();
		if (other.isOwning())
		{
			m_data = ml::malloc(device(), sizeInBytes());
			ml::memcpy(this->device(), this->data(), 0, other.device(), other.data(), 0, sizeInBytes());
		}
		else
			this->m_data = other.m_data;
	}
	Tensor::Tensor(Tensor &&other) noexcept :
			m_data(other.m_data),
			m_device(other.m_device),
			m_shape(other.m_shape),
			m_dtype(other.m_dtype),
			m_is_owning(other.m_is_owning)
	{
		create_stride();
		other.m_data = nullptr;
		other.m_is_owning = false;
	}
	Tensor::~Tensor() noexcept
	{
		deallocate_if_owning();
	}
	Tensor& Tensor::operator=(const Tensor &other)
	{
		if (this != &other)
		{
			if (other.isOwning()) // make a full copy
			{
				if (this->isOwning())
				{
					if (this->sizeInBytes() != other.sizeInBytes() or this->device() != other.device())
					{ // reallocate if different size or device
						deallocate_if_owning();
						m_data = ml::malloc(other.device(), other.sizeInBytes());
					}
				}
				else
					m_data = ml::malloc(other.device(), other.sizeInBytes());
				ml::memcpy(this->device(), this->data(), 0, other.device(), other.data(), 0, sizeInBytes());
			}
			else
			{
				deallocate_if_owning(); // assign other[tensor view] to this[owning tensor] -> produces tensor view
				this->m_data = other.m_data;
			}
			this->m_device = other.device();
			this->m_shape = other.shape();
			this->m_dtype = other.dtype();
			this->m_is_owning = other.m_is_owning;
			if (other.isPageLocked())
				pageLock();
			create_stride();
		}
		return *this;
	}
	Tensor& Tensor::operator=(Tensor &&other) noexcept
	{
		if (this != &other)
		{
			std::swap(this->m_data, other.m_data);
			std::swap(this->m_device, other.m_device);
			std::swap(this->m_shape, other.m_shape);
			std::swap(this->m_dtype, other.m_dtype);
			std::swap(this->m_is_owning, other.m_is_owning);
			std::swap(this->m_is_page_locked, other.m_is_page_locked);
			create_stride();
		}
		return *this;
	}

	std::string Tensor::info(bool full) const
	{
		if (full)
		{
			std::string result;
			result += std::string("device     : ") + device() + '\n';
			result += std::string("data type  : ") + dtype() + '\n';
			result += std::string("size       : ") + std::to_string(sizeInBytes()) + '\n';
			result += std::string("volume     : ") + std::to_string(volume()) + '\n';
			result += std::string("is owning  : ") + std::to_string(isOwning()) + '\n';
			result += std::string("is view    : ") + std::to_string(isView()) + '\n';
			result += std::string("rank       : ") + std::to_string(rank()) + '\n';
			result += std::string("is locked  : ") + std::to_string(isPageLocked()) + '\n';
//			result += "data       : " + std::to_string(data()) + '\n';
			return result;
		}
		else
		{
			if (isOwning())
				return std::string("Tensor<") + dtype() + ">" + m_shape.toString() + " on " + device();
			else
				return std::string("TensorView<") + dtype() + ">" + m_shape.toString() + " on " + device();
		}
	}

	Device Tensor::device() const
	{
		return m_device;
	}
	DataType Tensor::dtype() const
	{
		return m_dtype;
	}
	size_t Tensor::sizeInBytes() const
	{
		return sizeOf(dtype()) * volume();
	}

	bool Tensor::isOwning() const noexcept
	{
		return m_is_owning;
	}
	bool Tensor::isView() const noexcept
	{
		return not isOwning() and not isEmpty();
	}
	bool Tensor::isEmpty() const noexcept
	{
		return rank() == 0;
	}

	int Tensor::rank() const noexcept
	{
		return m_shape.rank();
	}
	int Tensor::firstDim() const noexcept
	{
		return m_shape.firstDim();
	}
	int Tensor::lastDim() const noexcept
	{
		return m_shape.lastDim();
	}
	int Tensor::dim(int idx) const noexcept
	{
		return m_shape[idx];
	}
	int Tensor::stride(int idx) const noexcept
	{
		return m_stride[idx];
	}
	const Shape& Tensor::shape() const noexcept
	{
		return m_shape;
	}
	int Tensor::volume() const noexcept
	{
		return m_shape.volume();
	}

	void Tensor::moveTo(Device newDevice)
	{
		if (isView())
			throw LogicError(METHOD_NAME, "Tensor view cannot be moved to another device");
		if (device() == newDevice)
			return;

		if (not isEmpty())
		{
			void *tmp = ml::malloc(newDevice, sizeInBytes());
			ml::memcpy(newDevice, tmp, 0, device(), data(), 0, sizeInBytes());
			deallocate_if_owning();
			m_data = tmp;
		}
		m_device = newDevice;
	}
	void Tensor::reshape(const Shape &newShape)
	{
		if (this->m_shape.volume() != newShape.volume())
			throw ShapeMismatch(METHOD_NAME, "");

		this->m_shape = newShape;
		create_stride();
	}

	void Tensor::convertTo(const Context &context, DataType newType)
	{
		if (isView())
			throw LogicError(METHOD_NAME, "Tensor view cannot be converted to another type");
		if (dtype() == newType)
			return;
		if (context.device() != this->device())
			throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());

		if (not isEmpty())
		{
			if (sizeOf(dtype()) == sizeOf(newType)) // no reallocation needed
				convertType(context, data(), newType, data(), dtype(), volume());
			else
			{
				void *tmp = ml::malloc(device(), volume() * sizeof(newType));
				convertType(context, tmp, newType, data(), dtype(), volume());
				deallocate_if_owning();
				m_data = tmp;
			}
		}
		m_dtype = newType;
	}
	void Tensor::zeroall(const Context &context)
	{
		if (context.device() != this->device())
			throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());
		ml::memzero(device(), data(), 0, sizeInBytes());
	}
	void Tensor::setall(const Context &context, float value)
	{
		if (context.device() != this->device())
			throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());
		uint16_t tmp[2] = { 0, 0 };
		switch (dtype())
		{
			case DataType::BFLOAT16:
				tmp[0] = convert_fp32_to_bf16(value);
				break;
			case DataType::FLOAT16:
				tmp[0] = convert_fp32_to_fp16(value);
				break;
			case DataType::FLOAT32:
				std::memcpy(tmp, &value, sizeof(float));
				break;
			default:
				throw DataTypeMismatch(METHOD_NAME, "unknown data type");
		}
		ml::memset(device(), data(), 0, sizeInBytes(), tmp, sizeOf(dtype()));
	}
	void Tensor::copyToHost(const Context &context, void *ptr, size_t bytes) const
	{
		if (bytes > static_cast<size_t>(this->sizeInBytes()))
			throw IllegalArgument(METHOD_NAME, "bytes", "must be lower than tensor size in bytes", bytes);
		if (bytes == 0)
			return; // no elements copied

		if (context.device().isCPU())
			ml::memcpy(Device::cpu(), ptr, 0, this->device(), this->data(), 0, bytes);
		else
		{
			if (context.device() != this->device())
				throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());
			ml::memcpy_async(context, Device::cpu(), ptr, 0, this->device(), this->data(), 0, bytes);
		}
	}
	void Tensor::copyFromHost(const Context &context, const void *ptr, size_t bytes)
	{
		if (bytes > static_cast<size_t>(this->sizeInBytes()))
			throw IllegalArgument(METHOD_NAME, "bytes", "must be lower than tensor size in bytes", bytes);
		if (bytes == 0)
			return; // no elements copied

		if (context.device().isCPU())
			ml::memcpy(this->device(), this->data(), 0, Device::cpu(), ptr, 0, bytes);
		else
		{
			if (context.device() != this->device())
				throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());
			ml::memcpy_async(context, this->device(), this->data(), 0, Device::cpu(), ptr, 0, bytes);
		}
	}
	void Tensor::copyFrom(const Context &context, const Tensor &other)
	{
		this->copyFrom(context, other, other.volume());
	}
	void Tensor::copyFrom(const Context &context, const Tensor &other, size_t elements)
	{
		if (elements > static_cast<size_t>(std::min(this->volume(), other.volume())))
			throw IllegalArgument(METHOD_NAME, "elements", "must be lower than tensor size", elements);
		if (elements == 0)
			return; // no elements copied
		if (this->m_shape != other.m_shape)
			throw ShapeMismatch(METHOD_NAME, this->m_shape, other.m_shape);
		if (this->dtype() != other.dtype())
			throw DataTypeMismatch(METHOD_NAME, this->dtype(), other.dtype());

		if (context.device().isCPU())
			ml::memcpy(this->device(), this->data(), 0, other.device(), other.data(), 0, sizeInBytes());
		else
		{
			if (context.device() != this->device() and context.device() != other.device())
				throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());
			ml::memcpy_async(context, device(), data(), 0, other.device(), other.data(), 0, sizeInBytes());
		}
	}

	bool Tensor::isPageLocked() const
	{
		return m_is_page_locked;
	}
	void Tensor::pageLock()
	{
		if (isView())
			throw LogicError(METHOD_NAME, "tensor view cannot be page locked");
		if (isPageLocked())
			throw LogicError(METHOD_NAME, "tensor already is page locked");
		if (device().isCPU())
		{
			cuda_page_lock(data(), sizeInBytes());
			m_is_page_locked = true;
		}
	}
	void Tensor::pageUnlock()
	{
		if (isView())
			throw LogicError(METHOD_NAME, "tensor view cannot be page unlocked");
		if (not isPageLocked())
			throw LogicError(METHOD_NAME, "tensor is not page locked");
		if (device().isCPU())
		{
			cuda_page_unlock(data());
			m_is_page_locked = false;
		}
	}

	Tensor Tensor::view() const
	{
		return view(shape(), 0);
	}
	Tensor Tensor::view(const Shape &shape, size_t offsetInElements) const
	{
		if (offsetInElements + shape.volume() > static_cast<size_t>(this->volume()))
			throw ShapeMismatch(METHOD_NAME, "view would extend beyond the original tensor");

		void *tmp = ml::view(device(), const_cast<void*>(data()), sizeOf(dtype()) * offsetInElements, sizeOf(dtype()) * shape.volume());
		if (tmp == nullptr)
			throw LogicError(METHOD_NAME, "could not create tensor view");

		Tensor result;
		result.m_data = tmp;
		result.m_device = device();
		result.m_shape = shape;
		result.m_dtype = dtype();
		result.m_is_owning = false;
		result.m_is_page_locked = isPageLocked();
		result.create_stride();
		return result;
	}

	const void* Tensor::data() const
	{
		return m_data;
	}
	void* Tensor::data()
	{
		return m_data;
	}
	float Tensor::get(std::initializer_list<int> idx) const
	{
		uint16_t tmp[2] = { 0, 0 };
		ml::memcpy(Device::cpu(), tmp, 0, device(), data(), sizeOf(dtype()) * get_index(idx.begin(), idx.size()), sizeOf(dtype()));
		switch (dtype())
		{
			case DataType::BFLOAT16:
				return convert_bf16_to_fp32(tmp[0]);
			case DataType::FLOAT16:
				return convert_fp16_to_fp32(tmp[0]);
			case DataType::FLOAT32:
			{
				float x;
				std::memcpy(&x, tmp, sizeof(float));
				return x;
			}
			case DataType::INT32:
			{
				int x;
				std::memcpy(&x, tmp, sizeof(int));
				return x;
			}
			default:
				throw DataTypeMismatch(METHOD_NAME, "unknown data type");
		}
	}
	void Tensor::set(float value, std::initializer_list<int> idx)
	{
		uint16_t tmp[2] = { 0, 0 };
		switch (dtype())
		{
			case DataType::BFLOAT16:
				tmp[0] = convert_fp32_to_bf16(value);
				break;
			case DataType::FLOAT16:
				tmp[0] = convert_fp32_to_fp16(value);
				break;
			case DataType::FLOAT32:
				std::memcpy(tmp, &value, sizeof(float));
				break;
			case DataType::INT32:
				std::memcpy(tmp, &value, sizeof(int));
				break;
			default:
				throw DataTypeMismatch(METHOD_NAME, "unknown data type");
		}
		ml::memcpy(device(), data(), sizeOf(dtype()) * get_index(idx.begin(), idx.size()), Device::cpu(), tmp, 0, sizeOf(dtype()));
	}

	Json Tensor::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["shape"] = m_shape.serialize();
		result["dtype"] = toString(dtype());
		result["binary_offset"] = binary_data.size();

		if (not isEmpty())
		{
			if (device().isCPU())
				binary_data.save(data(), sizeInBytes());
			else
			{
				std::unique_ptr<int8_t[]> buffer_on_cpu = std::make_unique<int8_t[]>(sizeInBytes());
				ml::memcpy(Device::cpu(), buffer_on_cpu.get(), 0, device(), data(), 0, sizeInBytes());
				binary_data.save(buffer_on_cpu.get(), sizeInBytes());
			}
		}

		return result;
	}
	void Tensor::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		if (isEmpty())
		{
			m_shape = Shape(json["shape"]);
			m_dtype = typeFromString(json["dtype"]);
			m_device = Device::cpu();
			m_is_owning = true;
			m_is_page_locked = false;
			m_data = ml::malloc(device(), sizeInBytes());
		}
		else
		{
			if (shape() != Shape(json["shape"]))
				throw ShapeMismatch(METHOD_NAME, shape(), Shape(json["shape"]));
			if (dtype() != typeFromString(json["dtype"]))
				throw DataTypeMismatch(METHOD_NAME, dtype(), typeFromString(json["dtype"]));
		}

		if (device().isCPU())
			binary_data.load(data(), static_cast<size_t>(json["binary_offset"]), sizeInBytes());
		else
		{
			std::unique_ptr<int8_t[]> buffer_on_cpu = std::make_unique<int8_t[]>(sizeInBytes());
			binary_data.load(buffer_on_cpu.get(), static_cast<size_t>(json["binary_offset"]), sizeInBytes());
			ml::memcpy(device(), data(), 0, Device::cpu(), buffer_on_cpu.get(), 0, sizeInBytes());
		}
	}

	size_t Tensor::get_index(const int *ptr, size_t size) const
	{
		if (static_cast<int>(size) != rank())
			throw ShapeMismatch(METHOD_NAME, rank(), static_cast<int>(size));

		assert(ptr != nullptr);
		size_t result = 0;
		for (int i = 0; i < rank(); i++)
		{
#ifndef NDEBUG
			if (ptr[i] < 0 || ptr[i] > m_shape[i])
				throw IndexOutOfBounds(METHOD_NAME, std::string("index:") + std::to_string(i), ptr[i], m_shape[i]);
#endif
			result += m_stride[i] * static_cast<uint32_t>(ptr[i]);
		}
		return result;
	}
	void Tensor::create_stride() noexcept
	{
		uint32_t tmp = 1;
		for (int i = Shape::max_dimension - 1; i >= m_shape.rank(); i--)
			m_stride[i] = 0;
		for (int i = m_shape.rank() - 1; i >= 0; i--)
		{
			m_stride[i] = tmp;
			tmp *= static_cast<uint32_t>(m_shape[i]);
		}
	}
	void Tensor::deallocate_if_owning()
	{
		if (isOwning())
		{
			if (isPageLocked())
				pageUnlock();
			ml::free(device(), data());
		}
	}

	Tensor toTensor(std::initializer_list<float> data)
	{
		Tensor result( { static_cast<int>(data.size()) }, "float32", Device::cpu());
		std::memcpy(result.data(), data.begin(), sizeof(float) * data.size());
		return result;
	}
	Tensor toTensor(std::initializer_list<std::initializer_list<float>> data)
	{
		Tensor result( { static_cast<int>(data.size()), static_cast<int>((data.begin()[0]).size()) }, "float32", Device::cpu());
		for (int i = 0; i < result.dim(0); i++)
		{
			assert(result.dim(1) == static_cast<int>((data.begin()[i]).size()));
			std::memcpy(reinterpret_cast<float*>(result.data()) + i * result.dim(1), (data.begin()[i]).begin(), sizeof(float) * result.dim(1));
		}
		return result;
	}

} /* namespace ml */

