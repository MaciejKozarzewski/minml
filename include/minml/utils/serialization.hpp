/*
 * SerializedObject.hpp
 *
 *  Created on: May 6, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_UTILS_SERIALIZATION_HPP_
#define MINML_UTILS_SERIALIZATION_HPP_

#include <cstring>
#include <string>
#include <vector>

class SerializedObject
{
	private:
		std::vector<uint8_t> m_data;
	public:
		explicit SerializedObject(int64_t size = 0);
		SerializedObject(const std::string &path);

		int64_t size() const noexcept;
		int64_t capacity() const noexcept;
		void clear() noexcept;

		void saveToFile(const std::string &path) const;
		void loadFromFile(const std::string &path);

		const uint8_t* data() const noexcept;
		uint8_t* data() noexcept;

		void save(const void *src, int64_t size_in_bytes);
		void load(void *dst, int64_t offset, int64_t size_in_bytes) const;

		template<typename T>
		void save(const T &value)
		{
			this->save(&value, sizeof(T));
		}
		template<typename T>
		T load(int64_t offset) const
		{
			T result;
			this->load(&result, offset, sizeof(T));
			return result;
		}

		//common to all classes
		int64_t getMemory() const noexcept;
};

#endif /* MINML_UTILS_SERIALIZATION_HPP_ */
