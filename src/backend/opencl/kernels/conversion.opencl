R"(

__kernel void unpack_input_fp32(__global float *output, const __global uint *input, int first_dim, int last_dim)
{
	const int stride = (last_dim + 31) / 32;
	for (int i = get_group_id(0); i < first_dim; i += get_num_groups(0))
		for (int j = get_local_id(1); j < last_dim; j += get_local_size(1))
		{
			const int int_idx = j / 32;
			const int bit_idx = j % 32;
			const uint32_t value = input[i * stride + int_idx] >> bit_idx;
			output[i * last_dim + j] = (value & 1) ? 1.0f : 0.0f;
		}
}

__kernel void unpack_input_fp16(__global ushort *output, const __global uint *input, int first_dim, int last_dim)
{
	const int stride = (last_dim + 31) / 32;
	for (int i = get_group_id(0); i < first_dim; i += get_num_groups(0))
		for (int j = get_local_id(1); j < last_dim; j += get_local_size(1))
		{
			const int int_idx = j / 32;
			const int bit_idx = j % 32;
			const uint32_t value = input[i * stride + int_idx] >> bit_idx;
			output[i * last_dim + j] = (value & 1) ? 0x3c00 : 0x0000;
		}
}

/*
* Conversion fp32 <-> fp16
*/
__kernel void convert_fp32_to_fp16(__global half *output, const __global float *input, int length)
{
	for (int i = get_global_id(0); i < length; i += get_global_size(0))
		vstore_half(input[i], i, output);
}

__kernel void convert_fp16_to_fp32(__global float *output, const __global half *input, int length)
{
	for (int i = get_global_id(0); i < length; i += get_global_size(0))
		output[i] = vload_half(i, input);
}

)"
