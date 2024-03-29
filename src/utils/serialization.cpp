/*
 * SerializedObject.cpp
 *
 *  Created on: May 6, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/serialization.hpp>

#include <cstddef>
#include <fstream>
#include <cassert>

SerializedObject::SerializedObject(int64_t size)
{
	m_data.reserve(size);
}
SerializedObject::SerializedObject(const std::string &path)
{
	loadFromFile(path);
}

int64_t SerializedObject::size() const noexcept
{
	return m_data.size();
}
int64_t SerializedObject::capacity() const noexcept
{
	return m_data.capacity();
}
void SerializedObject::clear() noexcept
{
	m_data.clear();
}

void SerializedObject::saveToFile(const std::string &path) const
{
	std::ofstream myFile(path.data(), std::ios::out | std::ios::binary);
	const int64_t length = size();
	myFile.write(reinterpret_cast<const char*>(&length), sizeof(length));
	myFile.write(reinterpret_cast<const char*>(data()), length);
	myFile.close();
}
void SerializedObject::loadFromFile(const std::string &path)
{
	std::ifstream myFile(path.data(), std::ios::in | std::ios::binary);
	if (myFile.good() == false)
		throw std::runtime_error("file not found");
	int64_t length;
	myFile.read(reinterpret_cast<char*>(&length), sizeof(length));
	m_data.resize(length);
	myFile.read(reinterpret_cast<char*>(data()), length);
	myFile.close();
}

const uint8_t* SerializedObject::data() const noexcept
{
	return m_data.data();
}
uint8_t* SerializedObject::data() noexcept
{
	return m_data.data();
}

void SerializedObject::save(const void *src, int64_t size_in_bytes)
{
	if (size_in_bytes == 0)
		return;

	assert(src != nullptr);
	const uint8_t *ptr = reinterpret_cast<const uint8_t*>(src);
	m_data.insert(m_data.end(), ptr, ptr + size_in_bytes);
}
void SerializedObject::load(void *dst, int64_t offset, int64_t size_in_bytes) const
{
	if (size_in_bytes == 0)
		return;

	assert(dst != nullptr);
	assert(offset + size_in_bytes <= size());
	uint8_t *ptr = reinterpret_cast<uint8_t*>(dst);
	std::memcpy(ptr, m_data.data() + offset, size_in_bytes);
}

//common to all classes
int64_t SerializedObject::getMemory() const noexcept
{
	return sizeof(this) + sizeof(uint8_t) * capacity();
}

