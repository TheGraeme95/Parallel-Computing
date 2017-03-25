__kernel void Sum2(__global int* input, __local int* localData, __global int* output)
{

	size_t global_ID = get_global_id(0);
	size_t local_Size = get_local_size(0);
	size_t local_ID = get_local_id(0);

	localData[local_ID] = input[global_ID];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = local_Size >> 1; i > 0; i >>= 1)
	{ 
		if (local_ID < i)
			localData[local_ID] += localData[local_ID + i];

		barrier(CLK_LOCAL_MEM_FENCE);

	}

	if (local_ID == 0)
		output[get_group_id(0)] = localData[0];
 }