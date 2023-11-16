R"(

__kernel void unpack_input_fp32(__global float *output, const __global uint *input, int first_dim, int last_dim)
{
	for (int i = get_group_id(0); i < first_dim; i += get_num_groups(0))
	{
		const uint value = input[i] >> get_local_id(1);
		output[i * last_dim + get_local_id(1)] = (value & 1) ? 1.0f : 0.0f;
	}
}

__kernel void unpack_input_fp16(__global ushort *output, const __global uint *input, int first_dim, int last_dim)
{
	for (int i = get_group_id(0); i < first_dim; i += get_num_groups(0))
	{
		const uint value = input[i] >> get_local_id(1);
		output[i * last_dim + get_local_id(1)] = (value & 1) ? 0x3c00 : 0x0000;
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
