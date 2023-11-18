R"(

float get_running_mean(const __global float *ptr, int idx, int last_dim)
{
	return ptr[idx];
}
float get_running_stddev(const __global float *ptr, int idx, int last_dim)
{
	return sqrt(ptr[last_dim + idx] + 1.0e-6f);
}
float get_running_gamma(const __global float *ptr, int idx, int last_dim)
{
	return ptr[2 * last_dim + idx];
}
float get_running_beta(const __global float *ptr, int idx, int last_dim)
{
	return ptr[3 * last_dim + idx];
}

/*
* Welford's online algorithm for calculating mean and variance
*/
struct AvgVarStats
{
	float samples, M, M2;
};

struct AvgVarStats create_stats()
{
	struct AvgVarStats result;
	result.samples = 0.0f;
	result.M = 0.0f;
	result.M2 = 0.0f;
	return result;
}
struct AvgVarStats add(struct AvgVarStats dst, float x)
{
	dst.samples += 1.0f;
	const float delta = x - dst.M;
	dst.M += delta / dst.samples;
	dst.M2 += delta * (x - dst.M);
	return dst;
}
float get_average(struct AvgVarStats dst)
{
	return dst.M;
}
float get_variance(struct AvgVarStats dst)
{
	return dst.M2 / (dst.samples - 1.0f);
}
float get_stddev(struct AvgVarStats dst)
{
	return sqrt(1.0e-6f + get_variance(dst));
}
struct AvgVarStats merge(struct AvgVarStats dst, struct AvgVarStats src)
{
	if (src.samples == 0.0f)
		return dst;
	if (dst.samples == 0.0f)
		return src;
	else
	{
		const float total_samples = dst.samples + src.samples;
		const float total_M = (dst.samples * dst.M + src.samples * src.M) / total_samples;
		const float total_M2 = dst.M2 + src.M2 + square(dst.M - src.M) * (dst.samples * src.samples) / total_samples;

		struct AvgVarStats result;
		result.samples = total_samples;
		result.M = total_M;
		result.M2 = total_M2;
		return result;
	}
}

/*
* Actual batchnorm code
*/

void combine_stats(__local struct AvgVarStats *stats)
{
	for (int i = 16; i >= 1; i /= 2)
	{
		const int dst_idx = get_local_id(1) * 32 + get_local_id(0);
		const int src_idx = dst_idx + 32 * i; 
		if (get_local_id(1) < i)
			stats[dst_idx] = merge(stats[dst_idx], stats[src_idx]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
__kernel void batchnorm_forward_avg_var_1(__global struct AvgVarStats *workspace, const __global float *input, int first_dim, int last_dim)
{
	local struct AvgVarStats shared_stats[32 * 32]; // 32 x 3 layout will be perfectly interleaved with no bank conflicts

	const int tid = get_global_id(0);

	struct AvgVarStats thread_stat = create_stats();
	if (tid < last_dim)
		for (int i = get_global_id(1); i < first_dim; i += get_global_size(1))
			thread_stat = add(thread_stat, input[i * last_dim + tid]);

	shared_stats[get_local_id(1) * 32 + get_local_id(0)] = thread_stat;
	barrier(CLK_LOCAL_MEM_FENCE);

	combine_stats(shared_stats);
	if (get_local_id(1) == 0 && tid < last_dim)
		workspace[get_group_id(1) * last_dim + tid] = shared_stats[get_local_id(0)];
}
__kernel void batchnorm_forward_avg_var_2(__global struct AvgVarStats *running_stat, int running_id, const __global struct AvgVarStats *workspace, int first_dim, int last_dim)
{
	local struct AvgVarStats shared_stats[32 * 32]; // 32 x 3 layout will be perfectly interleaved with no bank conflicts

	const int tid = get_global_id(0);

	struct AvgVarStats thread_stat = create_stats();
	if (tid < last_dim)
		for (int i = get_local_id(1); i < first_dim; i += 32)
			thread_stat = merge(thread_stat, workspace[i * last_dim + tid]);

	shared_stats[get_local_id(1) * 32 + get_local_id(0)] = thread_stat;
	barrier(CLK_LOCAL_MEM_FENCE);

	combine_stats(shared_stats);
	if (get_local_id(1) == 0 && tid < last_dim)
		running_stat[running_id * last_dim + tid] = shared_stats[get_local_id(0)];
}
__kernel void batchnorm_forward(const __global float *weights, const __global float *input, __global float *output, const __global struct AvgVarStats *running_stats,
		int running_id, int first_dim, int last_dim, int act)
{
	const int tid = get_global_id(0);
	if (tid < last_dim)
	{
		const float mean = get_average(running_stats[running_id * last_dim + tid]);
		const float stddev = get_stddev(running_stats[running_id * last_dim + tid]);
		const float gamma = get_running_gamma(weights, tid, last_dim);
		const float beta = get_running_beta(weights, tid, last_dim);

		const float scale = gamma / stddev;
		const float shift = -mean * scale + beta;

		for (int i = get_global_id(1); i < first_dim; i += get_global_size(1))
		{
			float tmp = input[i * last_dim + tid] * scale + shift;
			if (act == 1)
				tmp = sigmoid(tmp);
			if (act == 2)
				tmp = tanh(tmp);
			if (act == 3)
				tmp = relu(tmp);
			
			output[i * last_dim + tid] = tmp;
		}
	}
}
__kernel void batchnorm_inference(const __global float *weights, const __global float *input, __global float *output, int first_dim, int last_dim, int act)
{
	const int tid = get_global_id(0);
	if (tid < last_dim)
	{
		const float mean = get_running_mean(weights, tid, last_dim);
		const float stddev = get_running_stddev(weights, tid, last_dim);
		const float gamma = get_running_gamma(weights, tid, last_dim);
		const float beta = get_running_beta(weights, tid, last_dim);

		const float scale = gamma / stddev;
		const float shift = -mean * scale + beta;

		for (int i = get_global_id(1); i < first_dim; i += get_global_size(1))
		{
			float tmp = input[i * last_dim + tid] * scale + shift;
			if (act == 1)
				tmp = sigmoid(tmp);
			if (act == 2)
				tmp = tanh(tmp);
			if (act == 3)
				tmp = relu(tmp);
			output[i * last_dim + tid] = tmp;
		}
	}
}
void reduce_add_32x32_dual(__local float *ptr1, __local float *ptr2)
{
	for (int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
	{
		if (get_local_id(1) < i)
		{
			ptr1[get_local_id(1) * 32 + get_local_id(0)] += ptr1[(i + get_local_id(1)) * 32 + get_local_id(0)];
			ptr2[get_local_id(1) * 32 + get_local_id(0)] += ptr2[(i + get_local_id(1)) * 32 + get_local_id(0)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
__kernel void batchnorm_backward_delta_1(__global float *workspace, const __global float *input, const __global float *output,
		__global float *gradient_next, const __global struct AvgVarStats *running_stats, int running_id, int first_dim, int last_dim, int act)
{
	local float d_sigma[32 * 32];
	local float d_mu[32 * 32];
	const int tid = get_global_id(0);

	float d_sigma_acc = 0.0f, d_mu_acc = 0.0f;
	if (tid < last_dim)
	{
		const float mean = get_average(running_stats[running_id * last_dim + tid]);
		const float stddev = get_stddev(running_stats[running_id * last_dim + tid]);
		for (int i = get_global_id(1); i < first_dim; i += get_global_size(1))
		{
			const int tmp_idx = i * last_dim + tid;
			if (act == 1)
				gradient_next[tmp_idx] *= output[tmp_idx] * (1.0f - output[tmp_idx]);
			if (act == 2)
				gradient_next[tmp_idx] *= (1.0f - output[tmp_idx]) * (1.0f + output[tmp_idx]);
			if (act == 3 && output[tmp_idx] == 0.0f)
				gradient_next[tmp_idx] = 0.0f;
			
			d_sigma_acc += gradient_next[tmp_idx] * (input[tmp_idx] - mean) / stddev;
			d_mu_acc += gradient_next[tmp_idx];
		}
	}
	d_sigma[get_local_id(1) * 32 + get_local_id(0)] = d_sigma_acc;
	d_mu[get_local_id(1) * 32 + get_local_id(0)] = d_mu_acc;

	barrier(CLK_LOCAL_MEM_FENCE);
	reduce_add_32x32_dual(d_sigma, d_mu);
	if (get_local_id(1) == 0 && tid < last_dim)
	{
		workspace[2 * get_group_id(1) * last_dim + tid] = d_sigma[get_local_id(0)];
		workspace[(2 * get_group_id(1) + 1) * last_dim + tid] = d_mu[get_local_id(0)];
	}
}
__kernel void batchnorm_backward_delta_2(__global float *workspace, int first_dim, int last_dim)
{
	local float storage_d_sigma[32 * 32];
	local float storage_d_mu[32 * 32];
	const int tid = get_global_id(0);
	float d_sigma = 0.0f, d_mu = 0.0f;
	if (tid < last_dim)
		for (int i = get_global_id(1); i < first_dim; i += get_global_size(1))
		{
			d_sigma += workspace[i * 2 * last_dim + tid];
			d_mu += workspace[(i * 2 + 1) * last_dim + tid];
		}
	storage_d_sigma[get_local_id(1) * 32 + get_local_id(0)] = d_sigma;
	storage_d_mu[get_local_id(1) * 32 + get_local_id(0)] = d_mu;

	barrier(CLK_LOCAL_MEM_FENCE);
	reduce_add_32x32_dual(storage_d_sigma, storage_d_mu);
	if (get_local_id(1) == 0 && tid < last_dim)
	{
		workspace[tid] = storage_d_sigma[get_local_id(0)];
		workspace[last_dim + tid] = storage_d_mu[get_local_id(0)];
	}
}
__kernel void batchnorm_backward(const __global float *workspace, const __global float *input, __global float *gradient_prev, const __global float *gradient_next,
		const __global float *weights, __global float *weight_update, const __global struct AvgVarStats *running_stats, int running_id, int first_dim, int last_dim)
{
	// avg, stddev, gamma, d_sigma, d_mu
	const int tid = get_global_id(0);
	if (tid < last_dim)
	{
		const float mean = get_average(running_stats[running_id * last_dim + tid]);
		const float stddev = get_stddev(running_stats[running_id * last_dim + tid]);
		const float gamma = get_running_gamma(weights, tid, last_dim);

		float d_sigma = workspace[tid];
		float d_mu = workspace[last_dim + tid];
		if (get_group_id(1) == 0 && get_local_id(1) == 0)
		{ // only single line can update this
			weight_update[2 * last_dim + tid] += d_sigma; // gamma
			weight_update[3 * last_dim + tid] += d_mu; // beta
		}

		d_sigma = -gamma / stddev * d_sigma / (float)first_dim;
		d_mu = -gamma / stddev * d_mu / (float)first_dim;
		for (int i = get_global_id(1); i < first_dim; i += get_global_size(1))
			gradient_prev[i * last_dim + tid] = gamma / stddev * gradient_next[i * last_dim + tid]
					+ d_sigma * (input[i * last_dim + tid] - mean) / stddev + d_mu;
	}
}
__kernel void batchnorm_update(const __global struct AvgVarStats *running_stat, __global float *weights, int first_dim, int last_dim, int use_gamma,
		int use_beta)
{
	const int tid = get_global_id(0);
	if (tid < last_dim)
	{
		struct AvgVarStats stats;
		for (int i = 0; i < first_dim; i++)
			stats = merge(stats, running_stat[i * last_dim + tid]);
		weights[0 * last_dim + tid] = get_average(stats);
		weights[1 * last_dim + tid] = get_variance(stats);

		if (!use_gamma)
			weights[2 * last_dim + tid] = 1.0f; // gamma
		if (!use_beta)
			weights[3 * last_dim + tid] = 0.0f; // beta
	}
}
__kernel void fold_batchnorm(int first_dim, int last_dim, __global float *layer_weights, __global float *layer_bias, const __global float *batchnorm_weights)
{
	const float mean = get_running_mean(batchnorm_weights, get_group_id(0), first_dim);
	const float stddev = get_running_stddev(batchnorm_weights, get_group_id(0), first_dim);
	const float gamma = get_running_gamma(batchnorm_weights, get_group_id(0), first_dim);
	const float beta = get_running_beta(batchnorm_weights, get_group_id(0), first_dim);

	const float scale = gamma / stddev;
	const float shift = -mean * scale + beta;
	for (int i = get_local_id(0); i < last_dim; i += get_local_size(0))
		layer_weights[get_group_id(0) * last_dim + i] *= scale;

	if (get_local_id(0) == 0)
		layer_bias[get_group_id(0)] = layer_bias[get_group_id(0)] * scale + shift;
}

)"