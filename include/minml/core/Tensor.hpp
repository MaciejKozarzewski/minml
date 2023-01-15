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
			Tensor() = default;
			Tensor(const Shape &shape, DataType dtype, Device device);
			Tensor(const Shape &shape, const std::string &dtype, Device device);
			Tensor(const Json &json, const SerializedObject &binary_data);

			Tensor(const Tensor &other);
			Tensor(Tensor &&other) noexcept;

			~Tensor() noexcept;

			Tensor& operator=(const Tensor &other);
			Tensor& operator=(Tensor &&other) noexcept;

			std::string info(bool full = false) const;

			Device device() const;
			DataType dtype() const;
			size_t sizeInBytes() const;

			bool isOwning() const noexcept;
			bool isView() const noexcept;
			bool isEmpty() const noexcept;

			int rank() const noexcept;
			int dim(int idx) const noexcept;
			int firstDim() const noexcept;
			int lastDim() const noexcept;
			int volume() const noexcept;
			const Shape& shape() const noexcept;

			void moveTo(Device newDevice);
			void reshape(const Shape &newShape);

			void convertTo(const Context &context, DataType newType);
			void zeroall(const Context &context);
			void setall(const Context &context, float value);
			void copyFrom(const Context &context, const Tensor &other);
			void copyFrom(const Context &context, const Tensor &other, size_t elements);

			bool isPageLocked() const;
			void pageLock();
			void pageUnlock();

			Tensor view() const;
			Tensor view(const Shape &shape, size_t offsetInElements = 0) const;

			const void* data() const;
			void* data();

			float get(std::initializer_list<int> idx) const;
			void set(float value, std::initializer_list<int> idx);

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);

		private:
			size_t get_index(const int *ptr, size_t size) const;
			void create_stride() noexcept;

	};

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
