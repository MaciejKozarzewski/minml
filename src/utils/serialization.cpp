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

SerializedObject::SerializedObject(size_t size)
{
	m_data.reserve(size);
}
SerializedObject::SerializedObject(const std::string &path)
{
	loadFromFile(path);
}

size_t SerializedObject::size() const noexcept
{
	return m_data.size();
}
size_t SerializedObject::capacity() const noexcept
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
	const uint64_t length = size();
	myFile.write(reinterpret_cast<const char*>(&length), sizeof(length));
	myFile.write(reinterpret_cast<const char*>(data()), length);
	myFile.close();
}
void SerializedObject::loadFromFile(const std::string &path)
{
	std::ifstream myFile(path.data(), std::ios::in | std::ios::binary);
	if (myFile.good() == false)
		throw std::runtime_error("file not found");
	uint64_t length;
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

void SerializedObject::save(const void *src, size_t size_in_bytes)
{
	assert(src != nullptr);
	const char *ptr = reinterpret_cast<const char*>(src);
	m_data.insert(m_data.end(), ptr, ptr + size_in_bytes);
}
void SerializedObject::load(void *dst, size_t offset, size_t size_in_bytes) const
{
	assert(dst != nullptr);
	assert(offset + size_in_bytes <= m_data.size());
	char *ptr = reinterpret_cast<char*>(dst);
	std::memcpy(ptr, m_data.data() + offset, size_in_bytes);
}

//common to all classes
size_t SerializedObject::getMemory() const noexcept
{
	return sizeof(this) + sizeof(char) * capacity();
}

