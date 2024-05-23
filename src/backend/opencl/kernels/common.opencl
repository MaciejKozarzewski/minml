R"(

#ifndef STORAGE_PRECISION
  #define STORAGE_PRECISION 32
#endif
#ifndef COMPUTE_PRECISION
  #define COMPUTE_PRECISION 32
#endif

#if COMPUTE_PRECISION == 32
  typedef float compute_type;
  float zero()
  {
	  return 0.0f;
  }
  float one()
  {
  	  return 1.0f;
  }
  #if STORAGE_PRECISION == 16
	typedef half storage_type;
	float load(const __global half *ptr, int offset)
	{
		return vload_half(offset, ptr);
	}
	void store(half value, __global half *ptr, int offset)
	{
		vstore_half(value, offset, ptr);
	}
  #elif STORAGE_PRECISION == 32
    typedef float storage_type;
	float load(const __global float *ptr, int offset)
	{
		return ptr[offset];
	}
	void store(float value, __global float *ptr, int offset)
	{
		ptr[offset] = value;
	}
  #endif
#elif COMPUTE_PRECISION == 16
  typedef half compute_type;
  half zero()
  {
	  return 0.0h;
  }
  half one()
  {
	  return 1.0h;
  }
  #if STORAGE_PRECISION == 16
  	typedef half storage_type;
	half load(const __global half *ptr, int offset)
	{
		return half[offset];
	}
	void store(half value, __global half *ptr, int offset)
	{
		ptr[offset] = value;
	}
  #endif
#endif

compute_type sigmoid(compute_type x)
{
	return one() / (one() + exp(-x));
}
compute_type relu(compute_type x)
{
	return max(zero(), x);
}
float square(float x)
{
	return x * x;
}
float safe_log(float x)
{
	return log(1.0e-8f + x);
}

)"