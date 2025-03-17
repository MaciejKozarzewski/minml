/*
 * conv_im2row.cpp
 *
 *  Created on: May 29, 2023
 *      Author: Maciej Kozarzewski
 */

#include <minml/backend/cpu_backend.h>
#include <minml/backend/backend_utils.hpp>

#include "indexers.hpp"

#include <cstring>
#include <cinttypes>

namespace
{
	using namespace ml;

	template<typename T>
	void create_receptive_fields(const T *input, T *matrix, mlShape_t weights_shape, mlShape_t input_shape)
	{
		const int batch_size = input_shape.dim[0];
		const int height = input_shape.dim[1];
		const int width = input_shape.dim[2];
		const int filters = input_shape.dim[3];

		const int kernel_height = weights_shape.dim[1];
		const int kernel_width = weights_shape.dim[2];
		const int pad_h = (kernel_height - 1) / 2;
		const int pad_w = (kernel_width - 1) / 2;

		const Indexer<4> input_indexer(batch_size, height, width, filters);
		const Indexer<4> matrix_indexer(batch_size, height, width, kernel_height * kernel_width * filters);

		const size_t block_length = sizeof(T) * filters;
		for (int b = 0; b < batch_size; b++)
		{
			for (int h = 0; h < height; h++)
				for (int w = 0; w < width; w++)
				{
					T *ptr_dst = matrix + matrix_indexer.at(b, h, w, 0);
					if (h >= pad_h and h < height - pad_h and w >= pad_w and w < width - pad_w) // center of the image
					{
						for (int i = -pad_h; i <= pad_h; i++, ptr_dst += kernel_width * filters)
							std::memcpy(ptr_dst, input + input_indexer.at(b, h + i, w - pad_w, 0), kernel_width * block_length);
					}
					else // borders of the image
					{
						for (int i = -pad_h; i <= pad_h; i++)
							for (int j = -pad_w; j <= pad_w; j++, ptr_dst += filters)
								if ((h + i) >= 0 and (h + i) < height and (w + j) >= 0 and (w + j) < width)
									std::memcpy(ptr_dst, input + input_indexer.at(b, h + i, w + j, 0), block_length);
								else
									std::memset(ptr_dst, 0, block_length);
					}
				}
		}
	}
}

namespace ml
{

	void cpu_im2row(mlContext_t context, mlDataType_t dtype, mlShape_t input_shape, void *output, const void *input, int kernel_size, bool invert,
			const void *padding)
	{
		mlShape_t weights_shape = make_shape( { 0, kernel_size, kernel_size, 0 });
		switch (size_of(dtype))
		{
			case 1:
				create_receptive_fields(getPointer<int8_t>(input), getPointer<int8_t>(output), weights_shape, input_shape);
				break;
			case 2:
				create_receptive_fields(getPointer<int16_t>(input), getPointer<int16_t>(output), weights_shape, input_shape);
				break;
			case 4:
				create_receptive_fields(getPointer<int32_t>(input), getPointer<int32_t>(output), weights_shape, input_shape);
				break;
			case 8:
				create_receptive_fields(getPointer<int64_t>(input), getPointer<int64_t>(output), weights_shape, input_shape);
				break;
		}
	}

} /* namespace ml */
