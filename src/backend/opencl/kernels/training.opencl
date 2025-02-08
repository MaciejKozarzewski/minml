R"(

float round_small_to_zero(float x)
{
	return (fabs(x) < 1.0e-6f) ? 0.0f : x;
}
float cross_entropy(float output, float target)
{
	return -target * safe_log(output) - (1.0f - target) * safe_log(1.0f - output);
}

__kernel void loss_gradient(__global float *gradient, const __global float *output, const __global float *target, int elements, float inv_batch_size)
{
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
		gradient[i] = inv_batch_size * (output[i] - target[i]);
}
__kernel void CE_loss_step1(__global float *workspace, const __global float *output, const __global float *target, int elements)
{
	local float reduction_storage[256]; 

	float acc = 0.0f;
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
		acc += max(0.0f, cross_entropy(output[i], target[i]) - cross_entropy(target[i], target[i]));
	reduction_storage[get_local_id(0)] = acc;

	const float sum = reduce_add(reduction_storage);
	if (get_local_id(0) == 0)
		workspace[get_group_id(0)] = sum;
}
__kernel void MSE_loss_step1(__global float *workspace, const __global float *output, const __global float *target, int elements)
{
	local float reduction_storage[256]; 

	float acc = 0.0f;
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
		acc += square(output[i] - target[i]);
	reduction_storage[get_local_id(0)] = acc;

	const float sum = reduce_add(reduction_storage);
	if (get_local_id(0) == 0)
		workspace[get_group_id(0)] = sum;
}
__kernel void reduce_loss_step2(__global float *workspace, int elements)
{
	local float reduction_storage[256]; 

	float acc = 0.0f;
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
		acc += workspace[i];
	reduction_storage[get_local_id(0)] = acc;

	const float sum = reduce_add(reduction_storage);
	if (get_local_id(0) == 0)
		workspace[get_group_id(0)] = sum;
}

__kernel void learn_radam(__global float *weight, const __global float *gradient, __global float *momentum, __global float *variance, int elements,
		float learning_rate, float beta1, float beta2, int step)
{
	const float pow_beta1 = bounded_pow(beta1, step, 1.0e-8f);
	const float pow_beta2 = bounded_pow(beta2, step, 1.0e-8f);
	const float p_inf = 2.0f / (1.0f - beta2) - 1.0f;
	const float p = p_inf - 2.0f * step * pow_beta2 / (1.0f - pow_beta2);

	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
	{
		momentum[i] = beta1 * momentum[i] + (1.0f - beta1) * gradient[i];
		variance[i] = beta2 * variance[i] + (1.0f - beta2) * square(gradient[i]);

		float correction = 1.0f;
		if (p > 4.0f)
		{
			const float l = std::sqrt((1.0f - pow_beta2) / (variance[i] + 1.0e-8f));
			const float r = std::sqrt((p - 4.0f) * (p - 2.0f) * p_inf / ((p_inf - 4.0f) * (p_inf - 2.0f) * p));
			correction = l * r;
		}

		const float m_dash = momentum[i] / (1.0f - pow_beta1);
		const float tmp = -learning_rate * m_dash * correction;
		weight[i] = round_small_to_zero(weight[i] + tmp);
	}
}

__kernel void regularizer_l2(__global float *gradient, const __global float *param, float scale, float offset, int elements)
{
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
		gradient[i] += scale * (param[i] - offset);
}

__kernel void sum_over_first_dim(__global float *dst, const __global float *src, int first_dim, int last_dim, float beta, int step)
{
	local float tmp[32][32];

	const int tid = get_global_id(0);
	if (tid < last_dim)
	{
		float result = 0.0f;
		for (int i = get_global_id(1); i < first_dim; i += get_global_size(1))
			result += src[i * last_dim + tid];
		tmp[get_local_id(1)][get_local_id(0)] = result;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
	{
		if (get_local_id(1) < i)
			tmp[get_local_id(1)][get_local_id(0)] += tmp[i + get_local_id(1)][get_local_id(0)];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(1) == 0 && tid < last_dim)
	{
		if (step == 1) // write to temporary storage array
			dst[get_group_id(1) * last_dim + tid] = tmp[0][get_local_id(0)];
		if (step == 2) // write to final destination
		{
			if (beta == 0.0f)
				dst[tid] = tmp[0][get_local_id(0)];
			else
				dst[tid] = beta * dst[tid] + tmp[0][get_local_id(0)];
		}
	}
}


__kernel void multiply_tensors(__global storage_type *dst, const __global storage_type *src0, const __global storage_type *src1, int elements)
{
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
	{
		const compute_type x0 = load(src0, i);
		const compute_type x1 = load(src1, i);
		store(x0 * x1, dst, i);
	}
}
__kernel void add_tensors(__global storage_type *dst, float alpha1, const __global storage_type *src0, float alpha2, const __global storage_type *src1, int elements)
{
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
	{
		const compute_type x0 = load(src0, i);
		const compute_type x1 = load(src1, i);
		store(alpha1 * x0 + alpha2 * x1, dst, i);
	}
}

__kernel void emulate_low_precision(__global uint *dst, const __global uint *src, int elements)
{
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
		dst[i] = src[i] & 0xFFFFF000u;
}

)"