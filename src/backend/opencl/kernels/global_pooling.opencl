R"(

__kernel void pooling_avg_max_forward(__global storage_type *output, const __global storage_type *input, int dim0, int dim1, int dim2)
{
	local compute_type shared_avg[32 * 32];
	local compute_type shared_max[32 * 32];

	const int last_dim_index = get_group_id(0) * 32 + get_local_id(0);
	const int idx = get_local_id(1) * 32 + get_local_id(0);

	struct Indexer3D input_indexer = create_indexer_3D(dim0, dim1, dim2);

	if (last_dim_index < dim2)
	{
		compute_type local_avg = zero();
		compute_type local_max = load(input, get_pos_3D(input_indexer, get_group_id(2), 0, last_dim_index));
		for (int i = get_local_id(1); i < dim1; i += 32)
		{
			const compute_type tmp = load(input, get_pos_3D(input_indexer, get_group_id(2), i, last_dim_index));
			local_avg += tmp;
			local_max = max(local_max, tmp);
		}

		shared_avg[idx] = local_avg;
		shared_max[idx] = local_max;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 16; i >= 1; i /= 2)
	{
		if (get_local_id(1) < i)
		{
			shared_avg[idx] += shared_avg[idx + i * 32];
			shared_max[idx] = max(shared_max[idx], shared_max[idx + i * 32]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (get_local_id(1) == 0 && last_dim_index < dim2)
	{
		struct Indexer3D output_indexer = create_indexer_3D(dim0, 2, dim2);
		const compute_type inv = 1.0f / (float)dim1;
		const compute_type local_avg = shared_avg[get_local_id(0)] * inv;
		const compute_type local_max = shared_max[get_local_id(0)];

		store(local_avg, output, get_pos_3D(output_indexer, get_group_id(2), 0, last_dim_index));
		store(local_max, output, get_pos_3D(output_indexer, get_group_id(2), 1, last_dim_index));
	}
}
__kernel void pooling_avg_max_backward(__global float *gradient_prev, const __global float *gradient_next, const __global float *input,
		const __global float *output, int dim0, int dim1, int dim2)
{
	const int last_dim_index = get_global_id(0);

	if (last_dim_index < dim2)
	{
		struct Indexer3D input_indexer = create_indexer_3D(dim0, dim1, dim2);
		struct Indexer3D output_indexer = create_indexer_3D(dim0, 2, dim2);

		const float gradient_avg = gradient_next[get_pos_3D(output_indexer, get_group_id(2), 0, last_dim_index)] / (float)dim1;
		const float gradient_max = gradient_next[get_pos_3D(output_indexer, get_group_id(2), 1, last_dim_index)];
		const float local_max = output[get_pos_3D(output_indexer, get_group_id(2), 1, last_dim_index)];

		for (int i = get_group_id(1); i < dim1; i += get_num_groups(1))
		{
			const int index = get_pos_3D(input_indexer, get_group_id(2), i, last_dim_index);
			const float d_max = (input[index] == local_max) ? gradient_max : 0.0f;
			gradient_prev[index] = gradient_avg + d_max;
		}
	}
}

__kernel void global_broadcast_forward(__global storage_type *output, const __global storage_type *input, const __global storage_type *bias,
		int dim0, int dim1, int dim2, int act)
{
	struct Indexer2D bias_indexer = create_indexer_2D(dim0, dim2);

	struct Indexer3D input_indexer = create_indexer_3D(dim0, dim1, dim2);
	for (int j = get_global_id(0); j < dim2; j += get_global_size(0))
	{
		const compute_type _bias = load(bias, get_pos_2D(bias_indexer, get_group_id(2), j));
		for (int i = get_group_id(1); i < dim1; i += get_num_groups(1))
		{
			const int index = get_pos_3D(input_indexer, get_group_id(2), i, j);
			compute_type tmp = load(input, index) + _bias;
			switch(act)
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
			store(tmp, output, index);
		}
	}
}
__kernel void global_broadcast_backward(__global float *gradient_prev, __global float *gradient_next, const __global float *output,
		int dim0, int dim1, int dim2, int act)
{
	local float workspace[32 * 32];

	const int last_dim_index = get_group_id(0) * 32 + get_local_id(0);
	const int idx = get_local_id(1) * 32 + get_local_id(0);

	if (last_dim_index < dim2)
	{
		struct Indexer3D next_indexer = create_indexer_3D(dim0, dim1, dim2);

		float local_sum = 0.0f;
		for (int i = get_local_id(1); i < dim1; i += 32)
		{
			const int index = get_pos_3D(next_indexer, get_group_id(2), i, last_dim_index);
			if (act == 1)
				gradient_next[index] *= output[index] * (1.0f - output[index]);
			if (act == 2)
				gradient_next[index] *= (1.0f - output[index]) * (1.0f + output[index]);
			if (act == 3 && output[index] == 0.0f)
				gradient_next[index] = 0.0f;
			local_sum += gradient_next[index];
		}
		workspace[idx] = local_sum;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 16; i >= 1; i /= 2)
	{
		if (get_local_id(1) < i)
			workspace[idx] += workspace[idx + i * 32];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (get_local_id(1) == 0 && last_dim_index < dim2)
	{
		struct Indexer2D prev_indexer = create_indexer_2D(dim0, dim2);
		gradient_prev[get_pos_2D(prev_indexer, get_group_id(2), last_dim_index)] = workspace[get_local_id(0)];
	}
}

)"