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

namespace
{
	template<typename T>
	T load(uint64_t data)
	{
		static_assert(sizeof(T) <= sizeof(uint64_t), "");
		T result;
		std::memcpy(&result, &data, sizeof(T));
		return result;
	}
	template<typename T>
	uint64_t store(T value)
	{
		static_assert(sizeof(T) <= sizeof(uint64_t), "");
		uint64_t result = 0u;
		std::memcpy(&result, &value, sizeof(T));
		return result;
	}

	class Data
	{
			uint64_t m_data = 0u;
			ml::DataType m_dtype = ml::DataType::UNKNOWN;
		public:
			Data(ml::DataType dtype) noexcept :
					m_dtype(dtype)
			{
			}
			Data(ml::DataType dtype, float value) :
					m_dtype(dtype)
			{
				switch (m_dtype)
				{
					case ml::DataType::FLOAT16:
						m_data = store(ml::convert_fp32_to_fp16(value));
						break;
					case ml::DataType::FLOAT32:
						m_data = store(value);
						break;
					case ml::DataType::FLOAT64:
						m_data = store(static_cast<double>(value));
						break;
					case ml::DataType::UINT8:
						m_data = store(static_cast<uint8_t>(value));
						break;
					case ml::DataType::INT8:
						m_data = store(static_cast<int8_t>(value));
						break;
					case ml::DataType::INT32:
						m_data = store(static_cast<int32_t>(value));
						break;
					default:
						throw ml::DataTypeMismatch(METHOD_NAME, "unknown data type");
				}
			}
			float get() const
			{
				switch (m_dtype)
				{
					case ml::DataType::FLOAT16:
						return ml::convert_fp16_to_fp32(load<uint16_t>(m_data));
					case ml::DataType::FLOAT32:
						return load<float>(m_data);
					case ml::DataType::FLOAT64:
						return load<double>(m_data);
					case ml::DataType::UINT8:
						return load<uint8_t>(m_data);
					case ml::DataType::INT8:
						return load<int8_t>(m_data);
					case ml::DataType::INT32:
						return load<int32_t>(m_data);
					default:
						throw ml::DataTypeMismatch(METHOD_NAME, "unknown data type");
				}
			}
			void* data() noexcept
			{
				return &m_data;
			}
			const void* data() const noexcept
			{
				return &m_data;
			}
			size_t size() const noexcept
			{
				return ml::sizeOf(m_dtype);
			}
	};

	template<typename T>
	T load(const void *ptr, ml::DataType dtype)
	{
		switch (dtype)
		{
			case ml::DataType::FLOAT16:
				return static_cast<T>(ml::convert_fp16_to_fp32(reinterpret_cast<const uint16_t*>(ptr)[0]));
			case ml::DataType::FLOAT32:
			{
				float x;
				std::memcpy(&x, ptr, sizeof(float));
				return static_cast<T>(x);
			}
			case ml::DataType::FLOAT64:
			{
				double x;
				std::memcpy(&x, ptr, sizeof(double));
				return static_cast<T>(x);
			}
			case ml::DataType::UINT8:
			{
				uint8_t x;
				std::memcpy(&x, ptr, sizeof(uint8_t));
				return static_cast<T>(x);
			}
			case ml::DataType::INT8:
			{
				int8_t x;
				std::memcpy(&x, ptr, sizeof(int8_t));
				return static_cast<T>(x);
			}
			case ml::DataType::INT32:
			{
				int32_t x;
				std::memcpy(&x, ptr, sizeof(int32_t));
				return static_cast<T>(x);
			}
			default:
				throw ml::DataTypeMismatch(METHOD_NAME, "unknown data type");
		}
	}
	template<typename T>
	void store(T x, void *ptr, ml::DataType dtype)
	{
		switch (dtype)
		{
			case ml::DataType::FLOAT16:
			{
				const float value = static_cast<float>(x);
				reinterpret_cast<uint16_t*>(ptr)[0] = ml::convert_fp32_to_fp16(value);
				break;
			}
			case ml::DataType::FLOAT32:
			{
				const float value = static_cast<float>(x);
				std::memcpy(ptr, &value, sizeof(float));
				break;
			}
			case ml::DataType::FLOAT64:
			{
				const double value = static_cast<double>(x);
				std::memcpy(ptr, &value, sizeof(double));
				break;
			}
			case ml::DataType::UINT8:
			{
				const uint8_t value = static_cast<uint8_t>(x);
				std::memcpy(ptr, &value, sizeof(uint8_t));
				break;
			}
			case ml::DataType::INT8:
			{
				const int8_t value = static_cast<int8_t>(x);
				std::memcpy(ptr, &value, sizeof(int8_t));
				break;
			}
			case ml::DataType::INT32:
			{
				const int value = static_cast<int>(x);
				std::memcpy(ptr, &value, sizeof(int));
				break;
			}
			default:
				throw ml::DataTypeMismatch(METHOD_NAME, "unknown data type");
		}
	}
}

namespace ml
{
	Tensor::reference::reference(void *ptr, size_t offset, Device d, DataType dtype) :
			m_ptr(reinterpret_cast<uint8_t*>(ptr) + offset * sizeOf(dtype)),
			m_device(d),
			m_dtype(dtype)
	{
	}
	Tensor::reference& Tensor::reference::operator=(float x)
	{
		uint64_t tmp;
		store(x, &tmp, m_dtype);
		ml::memcpy(m_device, m_ptr, 0, Device::cpu(), &tmp, 0, sizeOf(m_dtype));
		return *this;
	}
	Tensor::reference& Tensor::reference::operator=(double x)
	{
		uint64_t tmp;
		store(x, &tmp, m_dtype);
		ml::memcpy(m_device, m_ptr, 0, Device::cpu(), &tmp, 0, sizeOf(m_dtype));
		return *this;
	}
	Tensor::reference& Tensor::reference::operator=(int x)
	{
		uint64_t tmp;
		store(x, &tmp, m_dtype);
		ml::memcpy(m_device, m_ptr, 0, Device::cpu(), &tmp, 0, sizeOf(m_dtype));
		return *this;
	}
	Tensor::reference::operator float() const
	{
		uint64_t tmp;
		ml::memcpy(Device::cpu(), &tmp, 0, m_device, m_ptr, 0, sizeOf(m_dtype));
		return load<float>(&tmp, m_dtype);
	}
	Tensor::reference::operator double() const
	{
		uint64_t tmp;
		ml::memcpy(Device::cpu(), &tmp, 0, m_device, m_ptr, 0, sizeOf(m_dtype));
		return load<float>(&tmp, m_dtype);
	}
	Tensor::reference::operator int() const
	{
		uint64_t tmp;
		ml::memcpy(Device::cpu(), &tmp, 0, m_device, m_ptr, 0, sizeOf(m_dtype));
		return load<float>(&tmp, m_dtype);
	}

	Tensor::const_reference::const_reference(void *ptr, size_t offset, Device d, DataType dtype) :
			m_dtype(dtype)
	{
		ml::memcpy(Device::cpu(), &m_data, 0, d, ptr, sizeOf(dtype) * offset, sizeOf(dtype));
	}
	Tensor::const_reference::operator float() const
	{
		return load<float>(&m_data, m_dtype);
	}
	Tensor::const_reference::operator double() const
	{
		return load<double>(&m_data, m_dtype);
	}
	Tensor::const_reference::operator int() const
	{
		return load<int>(&m_data, m_dtype);
	}

	Tensor::Tensor() noexcept
	{
		std::memset(m_stride, 0, sizeof(m_stride));
	}
	Tensor::Tensor(const Shape &shape) :
			Tensor(shape, DataType::FLOAT32, Device::cpu())
	{
	}
	Tensor::Tensor(const Shape &shape, DataType dtype, Device device) :
			m_device(device),
			m_shape(shape),
			m_dtype(dtype),
			m_is_owning(true)
	{
		m_data = ml::malloc(device, sizeInBytes());
		zeroall();
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
				ml::memcpy(this->device(), this->data(), 0, other.device(), other.data(), 0, other.sizeInBytes());
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
			result += std::string("device    : ") + device() + '\n';
			result += std::string("data type : ") + dtype() + '\n';
			result += std::string("shape     : ") + shape().toString() + '\n';
			result += std::string("volume    : ") + std::to_string(volume()) + '\n';
			result += std::string("bytes     : ") + std::to_string(sizeInBytes()) + '\n';
			result += std::string("is owning : ") + std::to_string(isOwning()) + '\n';
			result += std::string("is view   : ") + std::to_string(isView()) + '\n';
			result += std::string("is locked : ") + std::to_string(isPageLocked()) + '\n';
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

	Device Tensor::device() const noexcept
	{
		return m_device;
	}
	DataType Tensor::dtype() const noexcept
	{
		return m_dtype;
	}
	size_t Tensor::sizeInBytes() const noexcept
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
			throw ShapeMismatch(METHOD_NAME, "trying to reshape " + shape().toString() + " into " + newShape.toString());

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
				void *tmp = ml::malloc(device(), volume() * sizeOf(newType));
				convertType(context, tmp, newType, data(), dtype(), volume());
				deallocate_if_owning();
				m_data = tmp;
			}
		}
		m_dtype = newType;
	}
	void Tensor::zeroall()
	{
		ml::memzero(device(), data(), 0, sizeInBytes());
	}
	void Tensor::zeroall(const Context &context)
	{
		if (context.device() != this->device())
			throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());
		ml::memzero_async(context, device(), data(), 0, sizeInBytes());
	}
	void Tensor::setall(float value)
	{
		const Data tmp(dtype(), value);
		ml::memset(device(), data(), 0, sizeInBytes(), tmp.data(), tmp.size());
	}
	void Tensor::setall(const Context &context, float value)
	{
		if (context.device() != this->device())
			throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());

		const Data tmp(dtype(), value);
		ml::memset_async(context, device(), data(), 0, sizeInBytes(), tmp.data(), tmp.size());
	}
	void Tensor::copyToHost(void *dst, size_t bytes) const
	{
		if (bytes > static_cast<size_t>(this->sizeInBytes()))
			throw IllegalArgument(METHOD_NAME, "bytes", "must be lower than tensor size in bytes", bytes);
		if (bytes == 0)
			return; // no elements copied

		ml::memcpy(Device::cpu(), dst, 0, this->device(), this->data(), 0, bytes);
	}
	void Tensor::copyToHost(const Context &context, void *dst, size_t bytes) const
	{
		if (bytes > static_cast<size_t>(this->sizeInBytes()))
			throw IllegalArgument(METHOD_NAME, "bytes", "must be lower than tensor size in bytes", bytes);
		if (bytes == 0)
			return; // no elements copied
		if (context.device() != this->device())
			throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());

		ml::memcpy_async(context, Device::cpu(), dst, 0, this->device(), this->data(), 0, bytes);
	}
	void Tensor::copyFromHost(const void *src, size_t bytes)
	{
		if (bytes > static_cast<size_t>(this->sizeInBytes()))
			throw IllegalArgument(METHOD_NAME, "bytes", "must be lower than tensor size in bytes", bytes);
		if (bytes == 0)
			return; // no elements copied

		ml::memcpy(this->device(), this->data(), 0, Device::cpu(), src, 0, bytes);
	}
	void Tensor::copyFromHost(const Context &context, const void *src, size_t bytes)
	{
		if (bytes > static_cast<size_t>(this->sizeInBytes()))
			throw IllegalArgument(METHOD_NAME, "bytes", "must be lower than tensor size in bytes", bytes);
		if (bytes == 0)
			return; // no elements copied
		if (context.device() != this->device())
			throw DeviceMismatch(METHOD_NAME, "context on " + context.device().toString() + ", Tensor on " + device().toString());

		ml::memcpy_async(context, this->device(), this->data(), 0, Device::cpu(), src, 0, bytes);
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

	bool Tensor::isPageLocked() const noexcept
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

		void *tmp = ml::create_view(device(), const_cast<void*>(data()), sizeOf(dtype()) * offsetInElements, sizeOf(dtype()) * shape.volume());
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

	const void* Tensor::data() const noexcept
	{
		return m_data;
	}
	void* Tensor::data() noexcept
	{
		return m_data;
	}
	float Tensor::get(std::initializer_list<int> idx) const
	{
		Data tmp(dtype());
		ml::memcpy(Device::cpu(), tmp.data(), 0, device(), data(), sizeOf(dtype()) * get_index(idx.begin(), idx.size()), tmp.size());
		return tmp.get();
	}
	void Tensor::set(float value, std::initializer_list<int> idx)
	{
		const Data tmp(dtype(), value);
		ml::memcpy(device(), data(), sizeOf(dtype()) * get_index(idx.begin(), idx.size()), Device::cpu(), tmp.data(), 0, tmp.size());
	}

	Tensor::const_reference Tensor::at(std::initializer_list<int> idx) const
	{
		return const_reference(m_data, get_index(idx.begin(), idx.size()), device(), dtype());
	}
	Tensor::reference Tensor::at(std::initializer_list<int> idx)
	{
		return reference(m_data, get_index(idx.begin(), idx.size()), device(), dtype());
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
			binary_data.load(data(), json["binary_offset"].getLong(), sizeInBytes());
		else
		{
			std::unique_ptr<int8_t[]> buffer_on_cpu = std::make_unique<int8_t[]>(sizeInBytes());
			binary_data.load(buffer_on_cpu.get(), json["binary_offset"].getLong(), sizeInBytes());
			ml::memcpy(device(), data(), 0, Device::cpu(), buffer_on_cpu.get(), 0, sizeInBytes());
		}
	}
	size_t Tensor::getIndexOf(std::initializer_list<int> idx) const
	{
		return get_index(idx.begin(), idx.size());
	}
	/*
	 * private
	 */
	size_t Tensor::get_index(const int *ptr, size_t size) const
	{
		if (static_cast<int>(size) != rank())
			throw ShapeMismatch(METHOD_NAME, rank(), static_cast<int>(size));

		assert(ptr != nullptr);
		size_t result = 0;
		for (int i = 0; i < rank(); i++)
		{
#ifndef NDEBUG
			if (ptr[i] < 0 or ptr[i] >= m_shape[i])
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
		else
			ml::destroy_view(device(), data());
	}

	Tensor zeros_like(const Tensor &t)
	{
		return Tensor(t.shape(), t.dtype(), t.device());
	}
	Tensor ones_like(const Tensor &t)
	{
		Tensor result(t.shape(), t.dtype(), t.device());
		result.setall(1.0f);
		return result;
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

