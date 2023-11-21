R"(

/*
* Winograd transform 2x2 for kernel 3x3
*/
float weight_transform_3x3_2x2(int row, struct Line3 column)
{
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0;
		case 1:
			return 0.5f * (column.x0 + column.x1 + column.x2);
		case 2:
			return 0.5f * (column.x0 - column.x1 + column.x2);
		case 3:
			return column.x2;
	}
}
float input_transform_3x3_2x2(int row, struct Line4 column)
{
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0 - column.x2;
		case 1:
			return column.x1 + column.x2;
		case 2:
			return column.x2 - column.x1;
		case 3:
			return column.x3 - column.x1;
	}
}
float output_transform_3x3_2x2(int row, struct Line4 column)
{
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0 + column.x1 + column.x2;
		case 1:
			return column.x1 - column.x2 + column.x3;
	}
}
float gradient_transform_3x3_2x2(int row, struct Line2 column)
{
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0;
		case 1:
			return column.x0 + column.x1;
		case 2:
			return column.x0 - column.x1;
		case 3:
			return column.x1;
	}
}
float update_transform_3x3_2x2(int row, struct Line4 column)
{
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0 + 0.5f * (column.x1 + column.x2);
		case 1:
			return 0.5f * (column.x1 - column.x2);
		case 2:
			return 0.5f * (column.x1 + column.x2) + column.x3;
	}
}

/*
* Winograd transform 4x4 for kernel 3x3
*/
float weight_transform_3x3_4x4(int row, struct Line3 column)
{
	const float c13 = 1.0f / 3.0f;
	const float c23 = 2.0f / 3.0f;
	const float c2 = 2.0f;
	const float c4 = 4.0f;
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0;
		case 1:
			return c23 * (column.x0 + column.x1 + column.x2);
		case 2:
			return c23 * (column.x0 - column.x1 + column.x2);
		case 3:
			return c13 * (column.x0 + c2 * column.x1 + c4 * column.x2);
		case 4:
			return c13 * (column.x0 - c2 * column.x1 + c4 * column.x2);
		case 5:
			return c2 * column.x2;
	}
}
float input_transform_3x3_4x4(int row, struct Line6 column)
{
	const float c025 = 0.25f;
	const float c05 = 0.5f;
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0 - column.x2 + c025 * (column.x4 - column.x2);
		case 1:
			return column.x1 + column.x2 - c025 * (column.x3 + column.x4);
		case 2:
			return column.x2 - column.x1 + c025 * (column.x3 - column.x4);
		case 3:
			return column.x3 - column.x1 + c05 * (column.x4 - column.x2);
		case 4:
			return column.x1 - column.x3 + c05 * (column.x4 - column.x2);
		case 5:
			return column.x1 - column.x3 + c025 * (column.x5 - column.x3);
	}
}
float output_transform_3x3_4x4(int row, struct Line6 column)
{
	const float c025 = 0.25f;
	const float c05 = 0.5f;
	const float c2 = 2.0f;
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0 + column.x1 + column.x2 + c025 * (column.x3 + column.x4);
		case 1:
			return column.x1 - column.x2 + c05 * (column.x3 - column.x4);
		case 2:
			return column.x1 + column.x2 + column.x3 + column.x4;
		case 3:
			return column.x1 - column.x2 + c2 * (column.x3 - column.x4 + column.x5);
	}
}
float gradient_transform_3x3_4x4(int row, struct Line4 column)
{
	const float c13 = 1.0f / 3.0f;
	const float c23 = 2.0f / 3.0f;
	const float c2 = 2.0f;
	const float c4 = 4.0f;
	const float c8 = 8.0f;
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0;
		case 1:
			return c23 * (column.x0 + column.x1 + column.x2 + column.x3);
		case 2:
			return c23 * (column.x0 - column.x1 + column.x2 - column.x3);
		case 3:
			return c13 * (column.x0 + c2 * column.x1 + c4 * column.x2 + c8 * column.x3);
		case 4:
			return c13 * (column.x0 - c2 * column.x1 + c4 * column.x2 - c8 * column.x3);
		case 5:
			return c2 * column.x3;
	}
}
float update_transform_3x3_4x4(int row, struct Line6 column)
{
	const float c025 = 0.25f;
	const float c05 = 0.5f;
	const float c2 = 2.0f;
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0 + column.x1 + column.x2 + c025 * (column.x3 + column.x4);
		case 1:
			return column.x1 - column.x2 + c05 * (column.x3 - column.x4);
		case 2:
			return column.x1 + column.x2 + column.x3 + column.x4 + c2 * column.x5;
	}
}

/*
* Winograd transform 2x2 for kernel 5x5
*/
float weight_transform_5x5_2x2(int row, struct Line5 column)
{
	const float c016 = 1.0f / 6.0f;
	const float c066 = 2.0f / 3.0f;
	const float c2 = 2.0f;
	const float c4 = 4.0f;
	const float c8 = 8.0f;
	const float c16 = 16.0f;
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0;
		case 1:
			return c066 * (column.x0 + column.x1 + column.x2 + column.x3 + column.x4);
		case 2:
			return c066 * (column.x0 - column.x1 + column.x2 - column.x3 + column.x4);
		case 3:
			return c016 * (column.x0 + c2 * column.x1 + c4 * column.x2 + c8 * column.x3 + c16 * column.x4);
		case 4:
			return c016 * (column.x0 - c2 * column.x1 + c4 * column.x2 - c8 * column.x3 + c16 * column.x4);
		case 5:
			return c2 * column.x4;
	}
}
float input_transform_5x5_2x2(int row, struct Line6 column)
{
	return input_transform_3x3_4x4(row, column); // 2x2 input transform for 5x5 kernel is the same as 4x4 transform for 3x3 kernel
}
float output_transform_5x5_2x2(int row, struct Line6 column)
{
	const float c05 = 0.5f;
	const float c2 = 2.0f;
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0 + column.x1 + column.x2 + c05 * (column.x3 + column.x4);
		case 1:
			return column.x1 - column.x2 + column.x3 - column.x4 + c2 * column.x5;
	}
}
float gradient_transform_5x5_2x2(int row, struct Line2 column)
{
	const float c13 = 1.0f / 3.0f;
	const float c23 = 2.0f / 3.0f;
	const float c2 = 2.0f;
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0;
		case 1:
			return c23 * (column.x0 + column.x1);
		case 2:
			return c23 * (column.x0 - column.x1);
		case 3:
			return c13 * (column.x0 + c2 * column.x1);
		case 4:
			return c13 * (column.x0 - c2 * column.x1);
		case 5:
			return column.x1;
	}
}
float update_transform_5x5_2x2(int row, struct Line6 column)
{
	const float c025 = 0.25f;
	const float c05 = 0.5f;
	const float c2 = 2.0f;
	const float c4 = 4.0f;
	switch (row)
	{
		default:
			return 0.0f;
		case 0:
			return column.x0 + column.x1 + column.x2 + c025 * (column.x3 + column.x4);
		case 1:
			return column.x1 - column.x2 + c05 * (column.x3 - column.x4);
		case 2:
			return column.x1 + column.x2 + column.x3 + column.x4;
		case 3:
			return column.x1 - column.x2 + c2 * (column.x3 - column.x4);
		case 4:
			return column.x1 + column.x2 + c4 * (column.x3 + column.x4 + column.x5);
	}
}

/*
* Weight transform
*/

#ifndef KERNEL_SIZE
	#error "KERNEL_SIZE is not defined"
#endif
#ifndef TRANSFORM_SIZE
	#error "TRANSFORM_SIZE is not defined"
#endif

#if KERNEL_SIZE == 3 && TRANSFORM_SIZE == 2
	#define TILE_SIZE 4
#elif KERNEL_SIZE == 3 && TRANSFORM_SIZE == 4
	#define TILE_SIZE 6
#elif KERNEL_SIZE == 5 && TRANSFORM_SIZE == 2
	#define TILE_SIZE 6
#endif

#if KERNEL_SIZE == 3 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile3x3
	#define LINE_TYPE Line3
	#define TRANSFORM_FUNCTION weight_transform_3x3_2x2
	#define SET_TILE_AT_FUNCTION set_tile3x3_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile3x3
	#define SET_LINE_AT_FUNCTION set_line3_at
#elif KERNEL_SIZE == 3 && TRANSFORM_SIZE == 4
	#define TILE_TYPE Tile3x3
	#define LINE_TYPE Line3
	#define TRANSFORM_FUNCTION weight_transform_3x3_4x4
	#define SET_TILE_AT_FUNCTION set_tile3x3_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile3x3
	#define SET_LINE_AT_FUNCTION set_line3_at
#elif KERNEL_SIZE == 5 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile5x5
	#define LINE_TYPE Line5
	#define TRANSFORM_FUNCTION weight_transform_5x5_2x2
	#define SET_TILE_AT_FUNCTION set_tile5x5_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile5x5
	#define SET_LINE_AT_FUNCTION set_line5_at
#endif

__kernel void transform_weights(__global float * matrices, const __global float * weights, int output_filters, int input_filters, int invert)
{
	struct TILE_TYPE tile;
	for (int f = get_local_id(0); f < input_filters; f += get_local_size(0))
	{
		struct Indexer4D indexer = create_indexer_4D(output_filters, KERNEL_SIZE, KERNEL_SIZE, input_filters);
		for (int col = 0; col < KERNEL_SIZE; col++)
			for (int row = 0; row < KERNEL_SIZE; row++)
			{
				int index;
				if (invert)
					index = get_pos_4D(indexer, get_group_id(1), KERNEL_SIZE - 1 - row, KERNEL_SIZE - 1 - col, f);
				else
					index = get_pos_4D(indexer, get_group_id(1), row, col, f);
				const float tmp = weights[index];
				SET_TILE_AT_FUNCTION(&tile, col, row, tmp);
			}

		indexer = create_indexer_4D(TILE_SIZE, TILE_SIZE, output_filters, input_filters);

		for (int row = 0; row < TILE_SIZE; row++)
		{
			struct LINE_TYPE line;
			for (int col = 0; col < KERNEL_SIZE; col++)
			{	
				const struct LINE_TYPE tile_row = GET_ROW_FROM_TILE_FUNCTION(tile, col);
				const float tmp = TRANSFORM_FUNCTION(row, tile_row); // tile is stored as transposed (column-major) 
				SET_LINE_AT_FUNCTION(&line, col, tmp); 
			}

			for (int col = 0; col < TILE_SIZE; col++)
			{
				const float tmp = TRANSFORM_FUNCTION(col, line);
				matrices[get_pos_4D(indexer, row, col, get_group_id(1), f)] = tmp;
			}
		}
	}
}
#undef TILE_TYPE
#undef LINE_TYPE
#undef TRANSFORM_FUNCTION
#undef SET_TILE_AT_FUNCTION
#undef GET_ROW_FROM_TILE_FUNCTION
#undef SET_LINE_AT_FUNCTION


/*
* Input transform
*/
#if KERNEL_SIZE == 3 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile4x4
	#define LINE_TYPE Line4
	#define TRANSFORM_FUNCTION input_transform_3x3_2x2
	#define SET_TILE_AT_FUNCTION set_tile4x4_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile4x4
	#define SET_LINE_AT_FUNCTION set_line4_at
#elif KERNEL_SIZE == 3 && TRANSFORM_SIZE == 4
	#define TILE_TYPE Tile6x6
	#define LINE_TYPE Line6
	#define TRANSFORM_FUNCTION input_transform_3x3_4x4
	#define SET_TILE_AT_FUNCTION set_tile6x6_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile6x6
	#define SET_LINE_AT_FUNCTION set_line6_at
#elif KERNEL_SIZE == 5 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile6x6
	#define LINE_TYPE Line6
	#define TRANSFORM_FUNCTION input_transform_5x5_2x2
	#define SET_TILE_AT_FUNCTION set_tile6x6_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile6x6
	#define SET_LINE_AT_FUNCTION set_line6_at
#endif

__kernel void transform_input(__global float * matrices, const __global float * input, int batch_size, int height, int width, int input_filters)
{
	const int Padding = KERNEL_SIZE / 2;

	struct TILE_TYPE tile;
	for (int f = get_local_id(0); f < input_filters; f += get_local_size(0))
	{
		struct Indexer4D indexer = create_indexer_4D(batch_size, height, width, input_filters);
		for (int col = 0; col < TILE_SIZE; col++)
			for (int row = 0; row < TILE_SIZE; row++)
			{
				const int h = TRANSFORM_SIZE * get_group_id(0) - Padding + row;
				const int w = TRANSFORM_SIZE * get_group_id(1) - Padding + col;
				float tmp = 0.0f;
				if (0 <= h && h < height && 0 <= w && w < width)
					tmp = input[get_pos_4D(indexer, get_group_id(2), h, w, f)];
				SET_TILE_AT_FUNCTION(&tile, col, row, tmp);
			}

		const int tile_index = (get_group_id(2) * get_num_groups(0) + get_group_id(0)) * get_num_groups(1) + get_group_id(1);
		indexer = create_indexer_4D(TILE_SIZE, TILE_SIZE, get_num_groups(0) * get_num_groups(1) * get_num_groups(2), input_filters);

		for (int row = 0; row < TILE_SIZE; row++)
		{
			struct LINE_TYPE line;
			for (int col = 0; col < TILE_SIZE; col++)
			{	
				const struct LINE_TYPE tile_row = GET_ROW_FROM_TILE_FUNCTION(tile, col);
				const float tmp = TRANSFORM_FUNCTION(row, tile_row); // tile is stored as transposed (column-major) 
				SET_LINE_AT_FUNCTION(&line, col, tmp); 
			}

			for (int col = 0; col < TILE_SIZE; col++)
			{
				const float tmp = TRANSFORM_FUNCTION(col, line);
				matrices[get_pos_4D(indexer, row, col, tile_index, f)] = tmp;
			}
		}

	}
}
#undef TILE_TYPE
#undef LINE_TYPE
#undef TRANSFORM_FUNCTION
#undef SET_TILE_AT_FUNCTION
#undef GET_ROW_FROM_TILE_FUNCTION
#undef SET_LINE_AT_FUNCTION

/*
* Output transform
*/
#if KERNEL_SIZE == 3 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile4x4
	#define LINE_TYPE Line4
	#define TRANSFORM_FUNCTION output_transform_3x3_2x2
	#define SET_TILE_AT_FUNCTION set_tile4x4_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile4x4
	#define SET_LINE_AT_FUNCTION set_line4_at
#elif KERNEL_SIZE == 3 && TRANSFORM_SIZE == 4
	#define TILE_TYPE Tile6x6
	#define LINE_TYPE Line6
	#define TRANSFORM_FUNCTION output_transform_3x3_4x4
	#define SET_TILE_AT_FUNCTION set_tile6x6_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile6x6
	#define SET_LINE_AT_FUNCTION set_line6_at
#elif KERNEL_SIZE == 5 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile6x6
	#define LINE_TYPE Line6
	#define TRANSFORM_FUNCTION output_transform_5x5_2x2
	#define SET_TILE_AT_FUNCTION set_tile6x6_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile6x6
	#define SET_LINE_AT_FUNCTION set_line6_at
#endif

__kernel void transform_output(const __global float * matrices, __global float * output, const __global float * add, int use_add,
		const __global float * bias, int use_bias, int activation, int batch_size, int height, int width, int output_filters)
{
	struct TILE_TYPE tile;
	for (int f = get_local_id(0); f < output_filters; f += get_local_size(0))
	{
		const float bias_value = use_bias ? bias[f] : 0.0f;

		const int tile_index = (get_group_id(2) * get_num_groups(0) + get_group_id(0)) * get_num_groups(1) + get_group_id(1);
		struct Indexer4D indexer = create_indexer_4D(TILE_SIZE, TILE_SIZE, get_num_groups(0) * get_num_groups(1) * get_num_groups(2), output_filters);
		for (int col = 0; col < TILE_SIZE; col++)
			for (int row = 0; row < TILE_SIZE; row++)
			{
				const float tmp = matrices[get_pos_4D(indexer, row, col, tile_index, f)];
				SET_TILE_AT_FUNCTION(&tile, col, row, tmp);
			}

		indexer = create_indexer_4D(batch_size, height, width, output_filters);
		for (int row = 0; row < TRANSFORM_SIZE; row++)
		{
			const int h = TRANSFORM_SIZE * get_group_id(0) + row;
			if (h < height)
			{
				struct LINE_TYPE line;
				for (int col = 0; col < TILE_SIZE; col++)
				{
					const struct LINE_TYPE tile_row = GET_ROW_FROM_TILE_FUNCTION(tile, col);
					const float tmp = TRANSFORM_FUNCTION(row, tile_row); // tile is stored as transposed (column-major) 
					SET_LINE_AT_FUNCTION(&line, col, tmp); 
				}

				for (int col = 0; col < TRANSFORM_SIZE; col++)
				{
					const int w = TRANSFORM_SIZE * get_group_id(1) + col;
					if (w < width)
					{
						const int index = get_pos_4D(indexer, get_group_id(2), h, w, f);
						float tmp = TRANSFORM_FUNCTION(col, line) + bias_value;

						if (use_add)
							tmp += add[index];
						
						switch(activation)
						{
							case 0: // linear
								break;
							case 1: // sigmoid
								tmp = sigmoid(tmp);
								break;
							case 2: // tanh
								tmp = tanh(tmp);
								break;
							case 3: // relu
								tmp = relu(tmp);
								break;
							case 4: // softmax
								break;
						}

						output[index] = tmp; 
					}
				}
			}
		}
	}
}
#undef TILE_TYPE
#undef LINE_TYPE
#undef TRANSFORM_FUNCTION
#undef SET_TILE_AT_FUNCTION
#undef GET_ROW_FROM_TILE_FUNCTION
#undef SET_LINE_AT_FUNCTION


/*
* Gradient transform
*/
#if KERNEL_SIZE == 3 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile2x2
	#define LINE_TYPE Line2
	#define TRANSFORM_FUNCTION gradient_transform_3x3_2x2
	#define SET_TILE_AT_FUNCTION set_tile2x2_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile2x2
	#define SET_LINE_AT_FUNCTION set_line2_at
#elif KERNEL_SIZE == 3 && TRANSFORM_SIZE == 4
	#define TILE_TYPE Tile4x4
	#define LINE_TYPE Line4
	#define TRANSFORM_FUNCTION gradient_transform_3x3_4x4
	#define SET_TILE_AT_FUNCTION set_tile4x4_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile4x4
	#define SET_LINE_AT_FUNCTION set_line4_at
#elif KERNEL_SIZE == 5 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile2x2
	#define LINE_TYPE Line2
	#define TRANSFORM_FUNCTION gradient_transform_5x5_2x2
	#define SET_TILE_AT_FUNCTION set_tile2x2_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile2x2
	#define SET_LINE_AT_FUNCTION set_line2_at
#endif
__kernel void transform_gradient(__global float * matrices, const __global float * gradient, int batch_size, int height, int width, int filters)
{
	struct TILE_TYPE tile;
	for (int f = get_local_id(0); f < filters; f += get_local_size(0))
	{
		struct Indexer4D indexer = create_indexer_4D(batch_size, height, width, filters);
		for (int col = 0; col < TRANSFORM_SIZE; col++)
			for (int row = 0; row < TRANSFORM_SIZE; row++)
			{
				const int h = TRANSFORM_SIZE * get_group_id(0) + row;
				const int w = TRANSFORM_SIZE * get_group_id(1) + col;
				float tmp = 0.0f;
				if (0 <= h && h < height && 0 <= w && w < width)
					tmp = gradient[get_pos_4D(indexer, get_group_id(2), h, w, f)];
				SET_TILE_AT_FUNCTION(&tile, col, row, tmp);
			}

		const int tile_index = (get_group_id(2) * get_num_groups(0) + get_group_id(0)) * get_num_groups(1) + get_group_id(1);
		indexer = create_indexer_4D(TILE_SIZE, TILE_SIZE, get_num_groups(0) * get_num_groups(1) * get_num_groups(2), filters);
		for (int row = 0; row < TILE_SIZE; row++)
		{
			struct LINE_TYPE line;
			for (int col = 0; col < TRANSFORM_SIZE; col++)
			{
				const struct LINE_TYPE tile_row = GET_ROW_FROM_TILE_FUNCTION(tile, col);
				const float tmp = TRANSFORM_FUNCTION(row, tile_row); // tile is stored as transposed (column-major) 
				SET_LINE_AT_FUNCTION(&line, col, tmp); 
			}

			for (int col = 0; col < TILE_SIZE; col++)
			{
				const float tmp = TRANSFORM_FUNCTION(col, line);
				matrices[get_pos_4D(indexer, row, col, tile_index, f)] = tmp;
			}
		}
	}
}
#undef TILE_TYPE
#undef LINE_TYPE
#undef TRANSFORM_FUNCTION
#undef SET_TILE_AT_FUNCTION
#undef GET_ROW_FROM_TILE_FUNCTION
#undef SET_LINE_AT_FUNCTION

/*
* Update transform
*/
#if KERNEL_SIZE == 3 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile4x4
	#define LINE_TYPE Line4
	#define TRANSFORM_FUNCTION update_transform_3x3_2x2
	#define SET_TILE_AT_FUNCTION set_tile4x4_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile4x4
	#define SET_LINE_AT_FUNCTION set_line4_at
#elif KERNEL_SIZE == 3 && TRANSFORM_SIZE == 4
	#define TILE_TYPE Tile6x6
	#define LINE_TYPE Line6
	#define TRANSFORM_FUNCTION update_transform_3x3_4x4
	#define SET_TILE_AT_FUNCTION set_tile6x6_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile6x6
	#define SET_LINE_AT_FUNCTION set_line6_at
#elif KERNEL_SIZE == 5 && TRANSFORM_SIZE == 2
	#define TILE_TYPE Tile6x6
	#define LINE_TYPE Line6
	#define TRANSFORM_FUNCTION update_transform_5x5_2x2
	#define SET_TILE_AT_FUNCTION set_tile6x6_at
	#define GET_ROW_FROM_TILE_FUNCTION get_row_from_tile6x6
	#define SET_LINE_AT_FUNCTION set_line6_at
#endif
__kernel void transform_update(const __global float * matrices, __global float * update, int output_filters, int input_filters)
{
	struct TILE_TYPE tile;
	for (int f = get_local_id(0); f < input_filters; f += get_local_size(0))
	{
		struct Indexer4D indexer = create_indexer_4D(TILE_SIZE, TILE_SIZE, output_filters, input_filters);
		for (int col = 0; col < TILE_SIZE; col++)
			for (int row = 0; row < TILE_SIZE; row++)
			{
				const float tmp = matrices[get_pos_4D(indexer, row, col, get_group_id(1), f)];
				SET_TILE_AT_FUNCTION(&tile, col, row, tmp);
			}

		indexer = create_indexer_4D(output_filters, KERNEL_SIZE, KERNEL_SIZE, input_filters);
		for (int row = 0; row < KERNEL_SIZE; row++)
		{
			struct LINE_TYPE line;
			for (int col = 0; col < TILE_SIZE; col++)
			{
				const struct LINE_TYPE tile_row = GET_ROW_FROM_TILE_FUNCTION(tile, col);
				const float tmp = TRANSFORM_FUNCTION(row, tile_row); // tile is stored as transposed (column-major) 
				SET_LINE_AT_FUNCTION(&line, col, tmp); 
			}

			for (int col = 0; col < KERNEL_SIZE; col++)
			{
				const float tmp = TRANSFORM_FUNCTION(col, line);
				update[get_pos_4D(indexer, get_group_id(1), row, col, f)] = tmp;
			}
		}
	}
}
#undef TILE_TYPE
#undef LINE_TYPE
#undef TRANSFORM_FUNCTION
#undef SET_TILE_AT_FUNCTION
#undef GET_ROW_FROM_TILE_FUNCTION
#undef SET_LINE_AT_FUNCTION

)"