/*
 * Shape.cpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/core/Shape.hpp>
#include <minml/core/ml_exceptions.hpp>
#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>

#include <cstring>

namespace ml
{
	Shape::Shape()
	{
		std::memset(m_dim, 0, sizeof(m_dim));
	}
	Shape::Shape(const Json &json) :
			m_rank(json.size())
	{
		std::memset(m_dim, 0, sizeof(m_dim));
		if (m_rank < 0)
			throw IllegalArgument(METHOD_NAME, "length", "must be greater or equal 0", m_rank);
		if (m_rank > max_dimension)
			throw IllegalArgument(METHOD_NAME, "length", "must not exceed " + std::to_string(max_dimension), m_rank);

		for (int i = 0; i < m_rank; i++)
			m_dim[i] = json[i];
	}
	Shape::Shape(std::initializer_list<int> dims) :
			m_rank(dims.size())
	{
		std::memset(m_dim, 0, sizeof(m_dim));
		std::memcpy(m_dim, dims.begin(), sizeof(int) * m_rank);
	}

	std::string Shape::toString() const
	{
		std::string result = "[";
		for (int i = 0; i < m_rank; i++)
		{
			if (i != 0)
				result += 'x';
			result += std::to_string(m_dim[i]);
		}
		result += ']';
		return result;
	}
	int Shape::rank() const noexcept
	{
		return m_rank;
	}
	int Shape::dim(int index) const
	{
		if (index < 0 or index >= m_rank)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_rank);
		return m_dim[index];
	}
	int& Shape::dim(int index)
	{
		if (index < 0 or index >= m_rank)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_rank);
		return m_dim[index];
	}
	int Shape::operator[](int index) const
	{
		if (index < 0 or index >= m_rank)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_rank);
		return m_dim[index];
	}
	int& Shape::operator[](int index)
	{
		if (index < 0 or index >= m_rank)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_rank);
		return m_dim[index];
	}
	const int* Shape::data() const noexcept
	{
		return m_dim;
	}

	int Shape::firstDim() const noexcept
	{
		if (m_rank == 0)
			return 0;
		else
			return m_dim[0];
	}
	int Shape::lastDim() const noexcept
	{
		if (m_rank == 0)
			return 0;
		else
			return m_dim[m_rank - 1];
	}
	int Shape::volume() const noexcept
	{
		if (m_rank == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < m_rank; i++)
				result *= m_dim[i];
			return result;
		}
	}
	int Shape::volumeWithoutFirstDim() const noexcept
	{
		if (m_rank <= 1)
			return 0;
		else
		{
			int result = 1;
			for (int i = 1; i < m_rank; i++)
				result *= m_dim[i];
			return result;
		}
	}
	int Shape::volumeWithoutLastDim() const noexcept
	{
		if (m_rank <= 1)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < m_rank - 1; i++)
				result *= m_dim[i];
			return result;
		}
	}
	int Shape::volume(const std::vector<int> &dims) const
	{
		if (m_rank == 0 or dims.size() == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < static_cast<int>(dims.size()); i++)
			{
				int index = dims.begin()[i];
				if (index < 0 or index >= m_rank)
					throw IndexOutOfBounds(METHOD_NAME, "index" + std::to_string(i), index, m_rank);
				result *= m_dim[index];
			}
			return result;
		}
	}

	void Shape::removeDim(int index) noexcept
	{
		if (index < 0 or index >= m_rank)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_rank);

		for (int i = index; i < rank() - 1; i++)
			m_dim[i] = m_dim[i + 1];
		m_rank--;
	}

	bool operator==(const Shape &lhs, const Shape &rhs) noexcept
	{
		if (lhs.m_rank != rhs.m_rank)
			return false;
		for (int i = 0; i < lhs.m_rank; i++)
			if (lhs.m_dim[i] != rhs.m_dim[i])
				return false;
		return true;
	}
	bool operator!=(const Shape &lhs, const Shape &rhs) noexcept
	{
		return !(lhs == rhs);
	}

	//serialization interface
	Json Shape::serialize() const
	{
		return Json(m_dim, m_rank);
	}

	//common to all classes
	size_t Shape::getMemory() const noexcept
	{
		return sizeof(this);
	}

	std::ostream& operator<<(std::ostream &stream, const Shape &s)
	{
		stream << s.toString();
		return stream;
	}
	std::string operator+(const std::string &str, const Shape &shape)
	{
		return str + shape.toString();
	}
	std::string operator+(const Shape &shape, const std::string &str)
	{
		return shape.toString() + str;
	}

	ShapeMismatch::ShapeMismatch(const char *function, const std::string &what_arg) :
			logic_error(std::string(function) + " : " + what_arg)
	{
	}
	ShapeMismatch::ShapeMismatch(const char *function, int expected_rank, int actual_rank) :
			logic_error(std::string(function) + " : expected " + std::to_string(expected_rank) + "D shape, got " + std::to_string(actual_rank) + "D")
	{
	}
	ShapeMismatch::ShapeMismatch(const char *function, const Shape &expected, const Shape &got) :
			logic_error(std::string(function) + " : expected shape " + expected + ", got " + got)
	{
	}

} /* namespace ml */
