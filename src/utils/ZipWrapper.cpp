/*
 * ZipWrapper.cpp
 *
 *  Created on: Mar 7, 2021
 *      Author: Maciej Kozarzewski
 */

#include <minml/utils/ZipWrapper.hpp>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <iostream>

#include <zlib.h>

std::vector<char> ZipWrapper::compress(const std::vector<char> &data, int level)
{
	z_stream strm;
	std::vector<char> result;
	unsigned char input_buffer[CHUNK];
	unsigned char output_buffer[CHUNK];

	// allocate deflate state
	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	int ret = deflateInit(&strm, level);
	assert(ret == Z_OK);

	// compress until end of file
	for (size_t i = 0; i < data.size(); i += CHUNK)
	{
		int tmp = (CHUNK < data.size() - i) ? CHUNK : data.size() - i;
		strm.avail_in = tmp;
		strm.next_in = input_buffer;
		std::memcpy(input_buffer, data.data() + i, tmp);
		int flush = (data.size() - i <= CHUNK);
		do
		{
			strm.avail_out = CHUNK;
			strm.next_out = output_buffer;
			ret = deflate(&strm, flush); // no bad return value
			assert(ret != Z_STREAM_ERROR); // state not clobbered
			auto have = CHUNK - strm.avail_out;
			result.insert(result.end(),output_buffer, output_buffer + have);
		}
		while (strm.avail_out == 0);
	}
	(void) deflateEnd(&strm);
	return result;
}
std::vector<char> ZipWrapper::uncompress(const std::vector<char> &data)
{
	z_stream strm;
	std::vector<char> result;
	unsigned char input_buffer[CHUNK];
	unsigned char output_buffer[CHUNK];
	// allocate inflate state
	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	strm.avail_in = 0;
	strm.next_in = Z_NULL;
	int ret = inflateInit(&strm);
	assert(ret == Z_OK);

	// decompress until deflate stream ends or end of file
	for (size_t i = 0; i < data.size(); i += CHUNK)
	{
		int tmp = (CHUNK < data.size() - i) ? CHUNK : data.size() - i;
		strm.avail_in = tmp;
		strm.next_in = input_buffer;
		std::memcpy(input_buffer, data.data() + i, tmp);
		do
		{
			strm.avail_out = CHUNK;
			strm.next_out = output_buffer;
			ret = inflate(&strm, Z_NO_FLUSH);
			assert(ret != Z_STREAM_ERROR); // state not clobbered
			auto have = CHUNK - strm.avail_out;
			result.insert(result.end(), output_buffer, output_buffer + have);
		}
		while (strm.avail_out == 0); // done when inflate() says it's done
	}
	(void) inflateEnd(&strm); // clean up and return
	return result;
}

