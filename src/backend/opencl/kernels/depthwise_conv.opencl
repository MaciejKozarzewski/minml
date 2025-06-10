R"(

bool is_inside(int idx, int range)
{
	return 0 <= idx && idx < range;
}

float conditional_load(bool cond, __global const float *ptr, int offset, float default_value)
{
	if (cond)
		return ptr[offset];
	else
		return default_value;
}

void set_line4(__private struct Line4 *line, float x)
{
	line->x0 = x;
	line->x1 = x;
	line->x2 = x;
	line->x3 = x;
}

void load_line3(__private struct Line3 *line, const __local float *src, int row)
{
	const int tid = (row * 3 + 0) * 32 + get_local_id(0);
	line->x0 = src[tid + 0 * 32];
	line->x1 = src[tid + 1 * 32];
	line->x2 = src[tid + 2 * 32];
}
void load_line5(__private struct Line5 *line, const __local float *src, int row)
{
	const int tid = (row * 5 + 0) * 32 + get_local_id(0);
	line->x0 = src[tid + 0 * 32];
	line->x1 = src[tid + 1 * 32];
	line->x2 = src[tid + 2 * 32];
	line->x3 = src[tid + 3 * 32];
	line->x4 = src[tid + 4 * 32];
}
void load_line7(__private struct Line7 *line, const __local float *src, int row)
{
	const int tid = (row * 7 + 0) * 32 + get_local_id(0);
	line->x0 = src[tid + 0 * 32];
	line->x1 = src[tid + 1 * 32];
	line->x2 = src[tid + 2 * 32];
	line->x3 = src[tid + 3 * 32];
	line->x4 = src[tid + 4 * 32];
	line->x5 = src[tid + 5 * 32];
	line->x6 = src[tid + 6 * 32];
}

void store_line4(float *dst, int row, struct Line4 line)
{
	const int tid = (row * 4 + 0) * 32 + get_local_id(0);
	dst[tid + 0 * 32] = line.x0;
	dst[tid + 1 * 32] = line.x1;
	dst[tid + 2 * 32] = line.x2;
	dst[tid + 3 * 32] = line.x3;
}

void accumulate_4x6x3(__private struct Line4 *acc, struct Line6 input, struct Line3 filter)
{
	acc->x0 += input.x0 * filter.x0 + input.x1 * filter.x1 + input.x2 * filter.x2;
	acc->x1 += input.x1 * filter.x0 + input.x2 * filter.x1 + input.x3 * filter.x2;
	acc->x2 += input.x2 * filter.x0 + input.x3 * filter.x1 + input.x4 * filter.x2;
	acc->x3 += input.x3 * filter.x0 + input.x4 * filter.x1 + input.x5 * filter.x2;
}
void accumulate_4x8x5(__private struct Line4 *acc, struct Line8 input, struct Line5 filter)
{
	acc->x0 += input.x0 * filter.x0 + input.x1 * filter.x1 + input.x2 * filter.x2 + input.x3 * filter.x3 + input.x4 * filter.x4;
	acc->x1 += input.x1 * filter.x0 + input.x2 * filter.x1 + input.x3 * filter.x2 + input.x4 * filter.x3 + input.x5 * filter.x4;
	acc->x2 += input.x2 * filter.x0 + input.x3 * filter.x1 + input.x4 * filter.x2 + input.x5 * filter.x3 + input.x6 * filter.x4;
	acc->x3 += input.x3 * filter.x0 + input.x4 * filter.x1 + input.x5 * filter.x2 + input.x6 * filter.x3 + input.x7 * filter.x4;
}
void accumulate_4x10x7(__private struct Line4 *acc, struct Line10 input, struct Line7 filter)
{
	acc->x0 += input.x0 * filter.x0 + input.x1 * filter.x1 + input.x2 * filter.x2 + input.x3 * filter.x3 + input.x4 * filter.x4 + input.x5 * filter.x5 + input.x6 * filter.x6;
	acc->x1 += input.x1 * filter.x0 + input.x2 * filter.x1 + input.x3 * filter.x2 + input.x4 * filter.x3 + input.x5 * filter.x4 + input.x6 * filter.x5 + input.x7 * filter.x6;
	acc->x2 += input.x2 * filter.x0 + input.x3 * filter.x1 + input.x4 * filter.x2 + input.x5 * filter.x3 + input.x6 * filter.x4 + input.x7 * filter.x5 + input.x8 * filter.x6;
	acc->x3 += input.x3 * filter.x0 + input.x4 * filter.x1 + input.x5 * filter.x2 + input.x6 * filter.x3 + input.x7 * filter.x4 + input.x8 * filter.x5 + input.x9 * filter.x6;
}

#if KERNEL_SIZE == 7
	#define INPUT_SIZE 10
	#define TILE_SIZE 4
	#define INPUT_LINE_TYPE Line10 
	#define FILTER_LINE_TYPE Line7
	#define TRANSFORM_FUNCTION accumulate_4x10x7
	#define LOAD_FILTER_LINE_FUNCTION load_line7
	#define STORE_OUTPUT_LINE_FUNCTION store_line4
#elif KERNEL_SIZE == 5
	#define INPUT_SIZE 8
	#define TILE_SIZE 4
	#define INPUT_LINE_TYPE Line8 
	#define FILTER_LINE_TYPE Line5
	#define TRANSFORM_FUNCTION accumulate_4x8x5
	#define LOAD_FILTER_LINE_FUNCTION load_line5
	#define STORE_OUTPUT_LINE_FUNCTION store_line4
#elif KERNEL_SIZE == 3
	#define INPUT_SIZE 6
	#define TILE_SIZE 4
	#define INPUT_LINE_TYPE Line6 
	#define FILTER_LINE_TYPE Line3
	#define TRANSFORM_FUNCTION accumulate_4x6x3
	#define LOAD_FILTER_LINE_FUNCTION load_line3
	#define STORE_OUTPUT_LINE_FUNCTION store_line4
#endif

__kernel void depthwise_conv_forward(float beta, __global float *y_ptr, float alpha, const __global float *x_ptr, const __global float *w_ptr, const __global float *b_ptr, int height,
		int width, int channels, int invert_filter)
{
	const int Padding = (KERNEL_SIZE - 1) / 2;
	local float filter_tile[KERNEL_SIZE * KERNEL_SIZE * 32];
	local float bias_tile[32];

	const int f = get_global_id(0);

	if (f < channels)
	{
		if (get_local_id(1) == 0)
			bias_tile[get_local_id(0)] = conditional_load(b_ptr != NULL, b_ptr, f, 0.0f);
		for (int i = get_local_id(1); i < KERNEL_SIZE * KERNEL_SIZE; i += get_local_size(1))
		{
			int tmp = i;
			if (invert_filter)
				tmp = KERNEL_SIZE * KERNEL_SIZE - 1 - i;
			filter_tile[tmp * 32 + get_local_id(0)] = w_ptr[i * channels + f];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (f < channels)
	{
		const float bias = bias_tile[get_local_id(0)];

		const int h_stride = channels * width;
		const int w_stride = channels;

		for (int origin_w = 0; origin_w < width; origin_w += TILE_SIZE)
		{
			const int origin_h = TILE_SIZE * get_group_id(1);
			struct Indexer4D indexer = create_indexer_4D(get_num_groups(2), height, width, channels);
			const int input_offset = get_pos_4D(indexer, get_group_id(2), 0, 0, f);
			struct Line4 acc;
			set_line4(&acc, 0.0f);
			for (int k = 0; k < KERNEL_SIZE; k++)
			{
				const int h = origin_h + get_local_id(1) + k - Padding;

				if (is_inside(h, height))
				{
					struct INPUT_LINE_TYPE inp;
					for (int i = 0; i < INPUT_SIZE; i++)
					{
						const int w = origin_w + i - Padding;
						((float*)(&inp))[i] = conditional_load(is_inside(w, width), x_ptr, input_offset + h * h_stride + w * w_stride, 0.0f);
					}

					struct FILTER_LINE_TYPE fil;
					LOAD_FILTER_LINE_FUNCTION(&fil, filter_tile, k);
					TRANSFORM_FUNCTION(&acc, inp, fil);
				}
			}

			for (int i = 0; i < TILE_SIZE; i++)
			{
				const int h = origin_h + get_local_id(1);
				const int w = origin_w + i;
				if (is_inside(h, height) && is_inside(w, width))
				{
					const int idx = input_offset + h * h_stride + w * w_stride;
					float tmp = ((float*)(&acc))[i] * alpha + bias;
					if (beta != 0.0f)
						tmp += beta * y_ptr[idx];
					y_ptr[idx] = tmp;
				}
			}
		}
	}
}
)"