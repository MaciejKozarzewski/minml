R"(

__kernel void add_bias_act_fp32(__global storage_type *output, const __global storage_type *input, const __global storage_type *bias, int first_dim, int last_dim, int act)
{
	for (int j = get_global_id(0); j < last_dim; j += get_global_size(0))
	{
		const compute_type _bias = load(bias, j);
		for (int i = get_group_id(1); i < first_dim; i += get_num_groups(1))
		{
			compute_type tmp = load(input, i * last_dim + j) + _bias;
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
				case 4: // leaky_relu
					tmp = leaky_relu(tmp);
					break;
				case 5: // softmax
					break;
			}
			store(tmp, output, i * last_dim + j);	
		}
	}
}

)"