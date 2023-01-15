/*
 * json.cpp
 *
 *  Created on: Oct 2, 2020
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/json.hpp>
#include <minml/utils/string_util.hpp>
#include <minml/core/ml_exceptions.hpp>

#include <cstring>
#include <iostream>
#include <variant>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cassert>

namespace
{
	class JsonSerializer
	{
			std::string m_indent_string;
			std::vector<char> m_data;
			const int m_indent_step;
			const bool m_pretty_print;
		public:
			explicit JsonSerializer(int indent) :
					m_indent_string(128, ' '),
					m_data(),
					m_indent_step(indent),
					m_pretty_print(m_indent_step != -1)
			{
				m_data.reserve(1024);
			}
			void dump(const Json &json, int current_indent = 0)
			{
				if (json.isNull())
				{
					write("null", 4);
					return;
				}

				if (json.isBool())
				{
					write_bool(static_cast<bool>(json));
					return;
				}

				if (json.isNumber())
				{
					write_double(static_cast<double>(json));
					return;
				}

				if (json.isString())
				{
					write_string(static_cast<std::string>(json));
					return;
				}

				const int new_indent = current_indent + m_indent_step;
				if (static_cast<int>(m_indent_string.size()) < new_indent) // increase buffer for indentation
					m_indent_string.resize(m_indent_string.size() * 2, ' ');

				if (json.isArray())
				{
					if (json.size() == 0)
					{
						write("[]", 2);
						return;
					}

					if (json.isArrayOfPrimitives())
					{
						write('[');
						for (int i = 0; i < json.size(); i++)
						{
							if (i != 0)
							{
								write(',');
								if (m_pretty_print)
									write(' ');
							}
							dump(json[i]);
						}
						write(']');
					}
					else
					{
						write('[');
						if (m_pretty_print)
							write('\n');
						for (int i = 0; i < json.size(); i++)
						{
							if (m_pretty_print)
								write(m_indent_string.c_str(), new_indent);
							dump(json[i], new_indent);
							if (i != json.size() - 1)
								write(',');
							if (m_pretty_print)
								write('\n');
						}
						if (m_pretty_print)
							write(m_indent_string.c_str(), current_indent);
						write(']');
					}
					return;
				}

				if (json.isObject())
				{
					if (json.size() == 0)
					{
						write("{}", 2);
						return;
					}

					write('{');
					if (m_pretty_print)
						write('\n');
					for (int i = 0; i < json.size(); i++)
					{
						if (m_pretty_print)
							write(m_indent_string.c_str(), new_indent);
						dump(json.entry(i).first, new_indent);
						write(':');
						if (m_pretty_print)
							write(' ');
						dump(json.entry(i).second, new_indent);

						if (i != json.size() - 1)
							write(',');
						if (m_pretty_print)
							write('\n');
					}
					if (m_pretty_print)
						write(m_indent_string.c_str(), current_indent);
					write('}');
					return;
				}
			}
			std::string getString() const
			{
				return std::string(m_data.begin(), m_data.end());
			}
		private:
			void write(const char *str, int length)
			{
				m_data.insert(m_data.end(), str, str + length);
			}
			void write(char c)
			{
				m_data.push_back(c);
			}
			void write_bool(bool b)
			{
				if (b)
					write("true", 4);
				else
					write("false", 5);
			}
			void write_double(double d)
			{
				std::string tmp = std::to_string(d);
				int zeros = 0;
				for (int i = static_cast<int>(tmp.size()) - 1; i >= 0; i--)
					if (tmp[i] == '0')
						zeros++;
					else
						break;
				if (tmp[tmp.size() - zeros - 1] == '.')
					zeros++;
				m_data.insert(m_data.end(), tmp.begin(), tmp.end() - zeros);
			}
			void write_string(const std::string &str)
			{
				write('\"');
				m_data.insert(m_data.end(), str.begin(), str.end());
				write('\"');
			}
	};

	class JsonDeserializer
	{
			std::string m_data;
			size_t offset = 0;
		public:
			explicit JsonDeserializer(const std::string &str)
			{
				remove_spaces(str);
			}
			Json load()
			{
				if (m_data[offset] == '{') // start object
				{
					offset++;
					Json result(JsonType::Object);
					while (m_data[offset] != '}')
					{
						std::string key = load_string();
						if (m_data[offset] != ':')
							throw JsonParsingError(METHOD_NAME, "invalid json");
						offset++;
						result[key] = load();
						if (m_data[offset] == ',')
							offset++;
						else
						{
							if (m_data[offset] != '}')
								throw JsonParsingError(METHOD_NAME, "invalid json");
						}
					}
					offset++;
					return result;
				}
				if (m_data[offset] == '[') // start array
				{
					offset++;
					Json result(JsonType::Array);
					while (m_data[offset] != ']')
					{
						result[result.size()] = load();
						if (m_data[offset] == ',')
							offset++;
						else
						{
							if (m_data[offset] != ']')
								throw JsonParsingError(METHOD_NAME, "invalid json");
						}
					}
					offset++;
					return result;
				}
				if (m_data[offset] == '\"') // load string
				{
					std::string str = load_string();
					return Json(str);
				}
				return load_primitive_type();
			}
		private:
			void remove_spaces(const std::string &str)
			{
				m_data.reserve(1024);
				bool is_string = false;
				bool is_escape_character = false;
				for (size_t i = 0; i < str.size(); i++)
				{
					if ((!isspace(str[i]) || is_string) && str[i] != '\n')
						m_data.push_back(str[i]);

					if (is_escape_character)
						is_escape_character = false;
					else
					{
						if (str[i] == '\"')
							is_string = !is_string;
					}

					if (str[i] == '\\')
						is_escape_character = true;
				}
			}

			Json load_primitive_type()
			{
				const char *string_to_load = m_data.c_str() + offset;
				if (m_data.size() - offset >= 4 && memcmp(string_to_load, "null", 4) == 0)
				{
					offset += 4;
					return Json();
				}
				if (m_data.size() - offset >= 4 && memcmp(string_to_load, "true", 4) == 0)
				{
					offset += 4;
					return Json(true);
				}
				if (m_data.size() - offset >= 5 && memcmp(string_to_load, "false", 5) == 0)
				{
					offset += 5;
					return Json(false);
				}

				char *end = nullptr;
				double val = strtod(string_to_load, &end);

				if ((end != string_to_load) && (val != HUGE_VAL))
				{
					offset += static_cast<size_t>(end - string_to_load);
					return val;
				}
				else
					throw JsonParsingError(METHOD_NAME, "not a primitive type");
			}
			std::string load_string()
			{
				if (m_data[offset] != '\"')
					throw JsonParsingError(METHOD_NAME, "string never started");
				offset++;
				size_t start_index = offset;
				bool is_escape_character = false;
				for (; offset < m_data.size(); offset++)
				{
					if (is_escape_character)
						is_escape_character = false;
					else
					{
						if (m_data.at(offset) == '\"')
						{
							offset++;
							return m_data.substr(start_index, offset - 1 - start_index);
						}
					}

					if (m_data.at(offset) == '\\')
						is_escape_character = true;
				}
				throw JsonParsingError(METHOD_NAME, "string never ended");
			}
	};
}

Json::Json(JsonType type) noexcept
{
	switch (type)
	{
		default:
		case JsonType::Null:
			break;
		case JsonType::Bool:
			m_data = false;
			break;
		case JsonType::Number:
			m_data = 0.0;
			break;
		case JsonType::String:
			m_data = json_string();
			break;
		case JsonType::Array:
			m_data = json_array();
			break;
		case JsonType::Object:
			m_data = json_object();
			break;
	}
}

Json::Json() noexcept :
		m_data(NullObject())
{
}
Json::Json(bool b) noexcept :
		m_data(b)
{
}
Json::Json(int i) noexcept :
		m_data(static_cast<json_number>(i))
{
}
Json::Json(size_t i) noexcept :
		m_data(static_cast<json_number>(i))
{
	assert(i <= (static_cast<size_t>(1) << 53));
}
Json::Json(float f) noexcept :
		m_data(static_cast<json_number>(f))
{
}
Json::Json(double d) noexcept :
		m_data(d)
{
}
Json::Json(const std::string &str) :
		m_data(json_string(str))
{
}
Json::Json(const char *str) :
		Json(json_string(str))
{
}
Json::Json(const std::initializer_list<Json> &list)
{
	bool is_object = std::all_of(list.begin(), list.end(), [](const Json &element)
	{
		return element.isArray() && element.size() == 2 && element[0].isString();
	});

	if (is_object)
	{
		m_data = json_object(list.size());
		for (size_t i = 0; i < list.size(); i++)
			as_object()[i] = key_value_pair(list.begin()[i][0], list.begin()[i][1]);
	}
	else
		m_data = json_array(list);
}
Json::Json(const bool *list, size_t length) :
		m_data(json_array(list, list + length))
{
}
Json::Json(const int *list, size_t length) :
		m_data(json_array(list, list + length))
{
}
Json::Json(const double *list, size_t length) :
		m_data(json_array(list, list + length))
{
}

bool Json::isNull() const noexcept
{
	return std::holds_alternative<json_null>(m_data);
}
bool Json::isBool() const noexcept
{
	return std::holds_alternative<json_bool>(m_data);
}
bool Json::isNumber() const noexcept
{
	return std::holds_alternative<json_number>(m_data);
}
bool Json::isString() const noexcept
{
	return std::holds_alternative<json_string>(m_data);
}
bool Json::isArray() const noexcept
{
	return std::holds_alternative<json_array>(m_data);
}
bool Json::isArrayOfPrimitives() const noexcept
{
	if (not isArray())
		return false;
	for (auto element = as_array().begin(); element < as_array().end(); element++)
		if ((element->isObject() || element->isArray()) && element->size() > 0)
			return false;
	return true;
}
bool Json::isObject() const noexcept
{
	return std::holds_alternative<json_object>(m_data);
}

Json::operator bool() const
{
	if (not isBool())
		throw JsonTypeError(METHOD_NAME, storedType());
	return std::get<json_bool>(m_data);
}
Json::operator int() const
{
	if (not isNumber())
		throw JsonTypeError(METHOD_NAME, storedType());
	return static_cast<int>(std::get<json_number>(m_data));
}
Json::operator size_t() const
{
	if (not isNumber())
		throw JsonTypeError(METHOD_NAME, storedType());
	return static_cast<size_t>(std::get<json_number>(m_data));
}
Json::operator float() const
{
	if (not isNumber())
		throw JsonTypeError(METHOD_NAME, storedType());
	return static_cast<float>(std::get<json_number>(m_data));
}
Json::operator double() const
{
	if (not isNumber())
		throw JsonTypeError(METHOD_NAME, storedType());
	return std::get<json_number>(m_data);
}
Json::operator std::string() const
{
	if (not isString())
		throw JsonTypeError(METHOD_NAME, storedType());
	return std::get<std::string>(m_data);
}

bool Json::getBool() const
{
	if (not isBool())
		throw JsonTypeError(METHOD_NAME, storedType());
	return std::get<json_bool>(m_data);
}
int Json::getInt() const
{
	if (not isNumber())
		throw JsonTypeError(METHOD_NAME, storedType());
	return static_cast<int>(std::get<json_number>(m_data));
}
int64_t Json::getLong() const
{
	if (not isNumber())
		throw JsonTypeError(METHOD_NAME, storedType());
	return static_cast<int64_t>(std::get<json_number>(m_data));
}
double Json::getDouble() const
{
	if (not isNumber())
		throw JsonTypeError(METHOD_NAME, storedType());
	return std::get<json_number>(m_data);
}
std::string Json::getString() const
{
	if (not isString())
		throw JsonTypeError(METHOD_NAME, storedType());
	return std::get<std::string>(m_data);
}

const Json& Json::operator[](int idx) const
{
	if (not this->isArray())
		throw JsonTypeError(METHOD_NAME, storedType());
	return as_array()[idx];
}
Json& Json::operator[](int idx)
{
	if (isNull()) // null json can be turned into an array
		m_data = json_array(1 + idx);
	if (not this->isArray())
		throw JsonTypeError(METHOD_NAME, storedType());

	if (static_cast<size_t>(idx) >= as_array().size()) // if index is beyond range, create missing elements as null json
		as_array().insert(as_array().end(), 1 + idx - as_array().size(), Json());
	return as_array()[idx];
}

const Json& Json::operator[](const std::string &key) const
{
	if (not this->isObject())
		throw JsonTypeError(METHOD_NAME, storedType());

	const Json *element = find(key);
	if (element == nullptr)
		throw JsonKeyError(METHOD_NAME, key);
	else
		return *element;
}
Json& Json::operator[](const std::string &key)
{
	if (this->isNull()) // null json can be turned into an object
	{
		m_data = json_object( { key_value_pair(key, Json()) });
		return as_object()[0].second;
	}
	if (not this->isObject())
		throw JsonTypeError(METHOD_NAME, storedType());
	Json *element = find(key);
	if (element == nullptr)
	{
		as_object().push_back(key_value_pair(key, Json()));
		return as_object().back().second;
	}
	else
		return *element;
}
const Json& Json::operator[](const char *key) const
{
	return this->operator [](std::string(key));
}
Json& Json::operator[](const char *key)
{
	return this->operator [](std::string(key));
}

const Json* Json::find(const std::string &key) const noexcept
{
	if (this->isObject())
	{
		for (size_t i = 0; i < as_object().size(); i++)
			if (as_object()[i].first == key)
				return &(as_object()[i].second);
	}
	return nullptr;
}
Json* Json::find(const std::string &key) noexcept
{
	if (this->isObject())
	{
		for (size_t i = 0; i < as_object().size(); i++)
			if (as_object()[i].first == key)
				return &(as_object()[i].second);
	}
	return nullptr;
}
bool Json::hasKey(const std::string &key) const
{
	if (not this->isObject())
		throw JsonTypeError(METHOD_NAME, storedType());
	return find(key) != nullptr;
}
const std::pair<std::string, Json>& Json::entry(int idx) const
{
	if (not isObject())
		throw JsonTypeError(METHOD_NAME, storedType());
	return as_object()[idx];
}
std::pair<std::string, Json>& Json::entry(int idx)
{
	if (not isObject())
		throw JsonTypeError(METHOD_NAME, storedType());
	return as_object()[idx];
}
void Json::append(const Json &other)
{
	if (other.isNull())
		return;
	if (this->isObject() and other.isObject())
		this->as_object().insert(this->as_object().end(), other.as_object().begin(), other.as_object().end());
	else
	{
		if (this->isArray() and other.isArray())
			this->as_array().insert(this->as_array().end(), other.as_array().begin(), other.as_array().end());
		else
			throw JsonTypeError(METHOD_NAME, this->storedType());
	}
}

// misc methods
int Json::size() const
{
	switch (get_type())
	{
		case JsonType::Null:
			return 0;
		case JsonType::Array:
			return static_cast<int>(std::get<json_array>(m_data).size());
		case JsonType::Object:
			return static_cast<int>(std::get<json_object>(m_data).size());
		default:
			return 1;
	}
}
void Json::clear() noexcept
{
	switch (get_type())
	{
		case JsonType::Null:
			break;
		case JsonType::Bool:
			m_data = false;
			break;
		case JsonType::Number:
			m_data = 0.0;
			break;
		case JsonType::String:
			std::get<json_string>(m_data).clear();
			break;
		case JsonType::Array:
			std::get<json_array>(m_data).clear();
			break;
		case JsonType::Object:
			std::get<json_object>(m_data).clear();
			break;
	}
}
bool Json::isEmpty() const noexcept
{
	switch (get_type())
	{
		case JsonType::Null:
			return true;
		case JsonType::Array:
			return std::get<json_array>(m_data).empty();
		case JsonType::Object:
			return std::get<json_object>(m_data).empty();
		default:
			return false;
	}
}
const char* Json::storedType() const noexcept
{
	switch (get_type())
	{
		case JsonType::Null:
			return "null";
		case JsonType::Bool:
			return "bool";
		case JsonType::Number:
			return "number";
		case JsonType::String:
			return "string";
		case JsonType::Array:
			return "array";
		case JsonType::Object:
			return "object";
		default:
			return "unknown";
	}
}

std::string Json::dump(int indent) const
{
	JsonSerializer serializer(indent);
	serializer.dump(*this);
	return serializer.getString();
}
Json Json::load(const std::string &str)
{
	JsonDeserializer deserializer(str);
	return deserializer.load();
}

//private
JsonType Json::get_type() const noexcept
{
	return static_cast<JsonType>(m_data.index()); // @suppress("Method cannot be resolved")
}
const Json::json_array& Json::as_array() const noexcept
{
	assert(this->isArray());
	return std::get<json_array>(m_data);
}
Json::json_array& Json::as_array() noexcept
{
	assert(this->isArray());
	return std::get<json_array>(m_data);
}
const Json::json_object& Json::as_object() const noexcept
{
	assert(this->isObject());
	return std::get<json_object>(m_data);
}
Json::json_object& Json::as_object() noexcept
{
	assert(this->isObject());
	return std::get<json_object>(m_data);
}

JsonKeyError::JsonKeyError(const char *method, const std::string &key) :
		std::logic_error(std::string(method) + " : key '" + key + "' not found")
{
}
JsonTypeError::JsonTypeError(const char *method, const char *current_type) :
		std::logic_error(std::string(method) + " : currently stored type is " + current_type)
{
}
JsonParsingError::JsonParsingError(const char *method, const std::string &message) :
		std::logic_error(std::string(method) + " : " + message)
{

}

