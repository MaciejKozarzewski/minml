R"(

__kernel void sigmoid_forward_fp32(__global compute_type *output, const __global compute_type *input, int first_dim, int last_dim)
{
	const int elements = first_dim * last_dim;
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
	{
		compute_type x = load(input, i);
		x = sigmoid(x);
		store(x, output, i);
	}
}

__kernel void tanh_forward_fp32(__global storage_type *output, const __global storage_type *input, int first_dim, int last_dim)
{
	const int elements = first_dim * last_dim;
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
	{
		compute_type x = load(input, i);
		x = tanh(x);
		store(x, output, i);
	}
}

__kernel void relu_forward_fp32(__global storage_type *output, const __global storage_type *input, int first_dim, int last_dim)
{
	const int elements = first_dim * last_dim;
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
	{
		compute_type x = load(input, i);
		x = relu(x);
		store(x, output, i);
	}
}

__kernel void softmax_3_channels_fp32(__global storage_type *output, const __global storage_type *input, int first_dim, int last_dim)
{
	const int idx = get_global_id(0);
	if (idx < first_dim)
	{
		compute_type x0 = load(input, idx * 3 + 0);
		compute_type x1 = load(input, idx * 3 + 1);
		compute_type x2 = load(input, idx * 3 + 2);

		const compute_type max_value = max(x0, max(x1, x2));
		x0 = exp(x0 - max_value);
		x1 = exp(x1 - max_value);
		x2 = exp(x2 - max_value);

		const compute_type inv_sum = one() / (x0 + x1 + x2);

		store(x0 * inv_sum, output, idx * 3 + 0);
		store(x1 * inv_sum, output, idx * 3 + 1);
		store(x2 * inv_sum, output, idx * 3 + 2);
	}
}

__kernel void softmax_generic_fp32(__global storage_type *output, const __global storage_type *input, int first_dim, int last_dim)
{
	local float workspace[1024];
	local float reduction_storage[256];
	const int thread_idx = get_local_id(0); 	
	for (int i = get_group_id(0); i < first_dim; i += get_num_groups(0))
	{
		float max_value = -1e+32f;
		for (int j = thread_idx; j < last_dim; j += get_local_size(0))
		{
			workspace[j] = load(input, i * last_dim + j);
			max_value = max(max_value, workspace[j]);
		}
		reduction_storage[thread_idx] = max_value;

		max_value = reduce_max(reduction_storage);

		float partial_sum = 0.0f;
		for (int j = thread_idx; j < last_dim; j += get_local_size(0))
		{
			workspace[j] = exp(workspace[j] - max_value);
			partial_sum += workspace[j];
		}
		reduction_storage[thread_idx] = partial_sum;

		partial_sum = reduce_add(reduction_storage);

		const float inv_sum = one() / reduction_storage[0];
		for (int j = thread_idx; j < last_dim; j += get_local_size(0))
			store(workspace[j] * inv_sum, output, i * last_dim + j);
	}
}

)"