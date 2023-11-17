R"(

compute_type reduce_max(local compute_type *workspace)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int j = get_local_size(0) / 2; j >= 1; j /= 2)
	{
		if (get_local_id(0) < j)
			workspace[get_local_id(0)] = max(workspace[get_local_id(0)], workspace[get_local_id(0) + j]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	return workspace[0];
}

compute_type reduce_add(local compute_type *workspace)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int j = get_local_size(0) / 2; j >= 1; j /= 2)
	{
		if (get_local_id(0) < j)
			workspace[get_local_id(0)] += workspace[get_local_id(0) + j];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	return workspace[0];
}

)"