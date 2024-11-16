/*
 * Shape.hpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_CORE_SHAPE_HPP_
#define MINML_CORE_SHAPE_HPP_

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Json;

namespace ml
{
	class Shape
	{
		public:
			static const int max_dimension = 6;
		private:
			int m_dim[max_dimension];
			int m_rank = 0;
		public:
			Shape();
			Shape(const Json &json);
			Shape(std::initializer_list<int> dims);
			Shape(const std::vector<int> &dims);

			std::string toString() const;

			int rank() const noexcept;
			int dim(int index) const;
			int& dim(int index);
			int operator[](int index) const;
			int& operator[](int index);
			const int* data() const noexcept;

			int firstDim() const noexcept;
			int lastDim() const noexcept;
			int volume() const noexcept;
			int volumeWithoutFirstDim() const noexcept;
			int volumeWithoutLastDim() const noexcept;
			int volume(const std::vector<int> &dims) const;

			void removeDim(int index);

			friend bool operator==(const Shape &lhs, const Shape &rhs) noexcept;
			friend bool operator!=(const Shape &lhs, const Shape &rhs) noexcept;

			// serialization
			Json serialize() const;

			// common to all classes
			size_t getMemory() const noexcept;
	};

	std::ostream& operator<<(std::ostream &stream, const Shape &s);
	std::string operator+(const std::string &lhs, const Shape &rhs);
	std::string operator+(const Shape &lhs, const std::string &rhs);

	class ShapeMismatch: public std::logic_error
	{
		public:
			ShapeMismatch(const char *function, const std::string &what_arg);
			ShapeMismatch(const char *function, int expected_rank, int actual_rank);
			ShapeMismatch(const char *function, const Shape &expected, const Shape &got);
	};

} /* namespace ml */

#endif /* MINML_CORE_SHAPE_HPP_ */
