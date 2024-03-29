/*
 * file_util.hpp
 *
 *  Created on: Jun 17, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_UTILS_FILE_UTIL_HPP_
#define MINML_UTILS_FILE_UTIL_HPP_

#include <minml/utils/json.hpp>
#include <minml/utils/serialization.hpp>
#include <cstddef>
#include <fstream>
#include <string>

namespace ml
{
	class FileSaver
	{
			std::string path;
			std::ofstream stream;
		public:
			FileSaver(const std::string &path);
			std::string getPath() const;
			void save(const Json &json, const SerializedObject &binary_data, int indent = -1, bool compress = false);
			void close();
	};

	class FileLoader
	{
			Json json;
			SerializedObject binary_data;
			std::vector<char> loaded_data;
			size_t split_point;
		public:
			FileLoader(const std::string &path, bool uncompress = false);
			const Json& getJson() const noexcept;
			Json& getJson() noexcept;
			const SerializedObject& getBinaryData() const noexcept;
			SerializedObject& getBinaryData() noexcept;
		private:
			void load_all_data();
			size_t find_split_point() const noexcept;
	};

} /* namespace ml */

#endif /* MINML_UTILS_FILE_UTIL_HPP_ */
