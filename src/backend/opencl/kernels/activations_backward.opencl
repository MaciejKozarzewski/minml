R"(

__kernel void sigmoid_backward_fp32(__global float *gradient_prev, const __global float *gradient_next, const __global float *output, int elements)
{
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
		gradient_prev[i] = gradient_next[i] * output[i] * (1.0f - output[i]);
}
__kernel void tanh_backward_fp32(__global float *gradient_prev, const __global float *gradient_next, const __global float *output, int elements)
{
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
		gradient_prev[i] = gradient_next[i] * (1.0f - output[i]) * (1.0f + output[i]);
}
__kernel void relu_backward_fp32(__global float *gradient_prev, const __global float *gradient_next, const __global float *output, int elements)
{
	for (int i = get_global_id(0); i < elements; i += get_global_size(0))
		gradient_prev[i] = (output[i] == 0.0f) ? 0.0f : gradient_next[i];
}

)"