  __kernel void floatSum ( __global const int* input, __local int* scratch, __global float* output)
 {
  uint local_id = get_local_id(0);
  uint group_size = get_local_size(0);

  // Copy from global memory to local memory
  scratch[local_id] = input[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);

      // Divide WorkGroup into 2 parts and add elements 2 by 2
      // between local_id and local_id + stride

	  for (int i = 1; i < group_size; i *= 2) {
		if (!(local_id % (i * 2)) && ((local_id + i) < group_size)){
			//printf("scratch[%d] += scratch[%d] (%d += %d)\n", local_id, local_id + i, scratch[local_id], scratch[local_id + i]);
			scratch[local_id] += scratch[local_id + i];
			}
		barrier(CLK_LOCAL_MEM_FENCE);
	}           

  // Write result into partialSums[nWorkGroups]
  if (local_id == 0)
    output[get_group_id(0)] = scratch[0];	
  
 }

__kernel void Sum(__global const int* input, __local int* scratch, __global int* output) 
{
	int global_ID = get_global_id(0);
	int local_ID = get_local_id(0);
	int group_Size = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[local_ID] = input[global_ID];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < group_Size; i *= 2) 
	{
		if (!(local_ID % (i * 2)) && ((local_ID + i) < group_Size))
		{
			//printf("scratch[%d] += scratch[%d] (%d += %d)\n", lid, lid + i, scratch[lid], scratch[lid + i]);
			scratch[local_ID] += scratch[local_ID + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!local_ID) 
	{
		atomic_add(&output[0],scratch[local_ID]);
	}
}

__kernel void Minimum(__global const int* input, __local int* localMem, __global int* output, int choice)
{
	int global_ID = get_global_id(0);
	int local_ID = get_local_id(0);
	int group_Size = get_local_size(0);

	localMem[local_ID] = input[global_ID];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < group_Size; i *= 2)
	{ 
		if (!(local_ID % (i * 2)) && ((local_ID + i) < group_Size))
		{
			if (choice == 1)
				{
					if (localMem[local_ID] > localMem[local_ID + i])
						localMem[local_ID] = localMem[local_ID + i];			
				}
			else
				{
					if (localMem[local_ID] < localMem[local_ID + i])
						localMem[local_ID] = localMem[local_ID + i];
				}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

	}

	if (choice == 1)
				{
					if (!local_ID)
						atomic_min(&output[0], localMem[local_ID]);			
				}
			else
				{
					if (!local_ID)
						atomic_max(&output[0], localMem[local_ID]);	
				}

}
