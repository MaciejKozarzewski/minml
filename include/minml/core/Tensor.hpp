/*
 * Tensor.hpp
 *
 *  Created on: Jan 3, 2023
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_CORE_TENSOR_HPP_
#define MINML_CORE_TENSOR_HPP_

#include <minml/core/Shape.hpp>
#include <minml/core/Device.hpp>
#include <minml/core/DataType.hpp>

#include <cstring>
#include <cassert>

class Json;
class SerializedObject;

namespace ml /* forward declarations */
{
	class Context;
}

namespace ml
{

	class Tensor
	{
		private:
			void *m_data = nullptr;
			Device m_device = Device::cpu();
			Shape m_shape;
			DataType m_dtype = DataType::UNKNOWN;
			bool m_is_owning = false;
			bool m_is_page_locked = false;

			uint32_t m_stride[Shape::max_dimension];
		public:
			class const_reference
			{
					friend class Tensor;
					uint64_t m_data;
					DataType m_dtype = DataType::UNKNOWN;
					const_reference(void *ptr, size_t offset, Device d, DataType dtype);
				public:
					operator float() const;
					operator double() const;
					operator uint8_t() const;
					operator int8_t() const;
					operator int16_t() const;
					operator int32_t() const;
			};
			class reference
			{
					friend class Tensor;
					void *m_ptr = nullptr;
					Device m_device = Device::cpu();
					DataType m_dtype = DataType::UNKNOWN;
					reference(void *ptr, size_t offset, Device d, DataType dtype);
				public:
					reference& operator=(const_reference x);
					reference& operator=(reference x);
					reference& operator=(float x);
					reference& operator=(double x);
					reference& operator=(uint8_t x);
					reference& operator=(int8_t x);
					reference& operator=(int16_t x);
					reference& operator=(int32_t x);
					operator float() const;
					operator double() const;
					operator uint8_t() const;
					operator int8_t() const;
					operator int16_t() const;
					operator int32_t() const;
			};

			Tensor() noexcept;
			Tensor(const Shape &shape);
			Tensor(const Shape &shape, DataType dtype, Device device);
			Tensor(const Shape &shape, const std::string &dtype, Device device);
			Tensor(const Json &json, const SerializedObject &binary_data);

			Tensor(const Tensor &other);
			Tensor(Tensor &&other) noexcept;

			~Tensor() noexcept;

			Tensor& operator=(const Tensor &other);
			Tensor& operator=(Tensor &&other) noexcept;

			std::string info(bool full = false) const;

			Device device() const noexcept;
			DataType dtype() const noexcept;
			size_t sizeInBytes() const noexcept;

			bool isOwning() const noexcept;
			bool isView() const noexcept;
			bool isEmpty() const noexcept;

			int rank() const noexcept;
			int dim(int idx) const noexcept;
			int stride(int idx) const noexcept;
			int firstDim() const noexcept;
			int lastDim() const noexcept;
			int volume() const noexcept;
			const Shape& shape() const noexcept;

			void moveTo(Device newDevice);
			void reshape(const Shape &newShape);

			void convertTo(const Context &context, DataType newType);
			void reinterpretAs(DataType newType);
			void zeroall();
			void zeroall(const Context &context);
			void setall(float value);
			void setall(const Context &context, float value);
			void copyToHost(void *dst, size_t bytes) const;
			void copyToHost(const Context &context, void *dst, size_t bytes) const;
			void copyFromHost(const void *src, size_t bytes);
			void copyFromHost(const Context &context, const void *src, size_t bytes);
			void copyFrom(const Context &context, const Tensor &other);
			void copyFrom(const Context &context, const Tensor &other, size_t elements);

			bool isPageLocked() const noexcept;
			void pageLock();
			void pageUnlock();

			Tensor view() const;
			Tensor view(const Shape &shape) const;
			Tensor view(const Shape &shape, size_t offsetInElements) const;
			Tensor view(const Shape &shape, std::initializer_list<int> position) const;

			const void* data() const noexcept;
			void* data() noexcept;

			float get(std::initializer_list<int> idx) const;
			void set(float value, std::initializer_list<int> idx);

			const_reference at(std::initializer_list<int> idx) const;
			reference at(std::initializer_list<int> idx);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);

			size_t getIndexOf(std::initializer_list<int> idx) const;
		private:
			size_t get_index(const int *ptr, size_t size) const;
			void create_stride() noexcept;
			void deallocate_if_owning();

	};

	Tensor zeros_like(const Tensor &t);
	Tensor ones_like(const Tensor &t);

	Tensor toTensor(std::initializer_list<float> data);
	Tensor toTensor(std::initializer_list<std::initializer_list<float>> data);

	template<class T, class U>
	bool same_device(const T &lhs, const U &rhs)
	{
		return lhs.device() == rhs.device();
	}
	template<class T, class U, class ... ARGS>
	bool same_device(const T &lhs, const U &rhs, const ARGS &... args)
	{
		if (lhs.device() == rhs.device())
			return same_device(lhs, args...);
		else
			return false;
	}

	template<class T, class U>
	bool same_type(const T &lhs, const U &rhs)
	{
		return lhs.dtype() == rhs.dtype();
	}
	template<class T, class U, class ... ARGS>
	bool same_type(const T &lhs, const U &rhs, const ARGS &... args)
	{
		if (lhs.dtype() == rhs.dtype())
			return same_type(lhs, args...);
		else
			return false;
	}

	template<class T, class U>
	bool same_shape(const T &lhs, const U &rhs)
	{
		return lhs.shape() == rhs.shape();
	}
	template<class T, class U, class ... ARGS>
	bool same_shape(const T &lhs, const U &rhs, const ARGS &... args)
	{
		if (lhs.shape() == rhs.shape())
			return same_shape(lhs, args...);
		else
			return false;
	}

}
/* namespace ml */

#endif /* MINML_CORE_TENSOR_HPP_ */
