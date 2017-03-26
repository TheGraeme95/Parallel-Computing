  __kernel void sumGPU ( __global const float* input, __local int* scratch, __global float* output)
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
			printf("scratch[%d] += scratch[%d] (%d += %d)\n", local_id, local_id + i, scratch[local_id], scratch[local_id + i]);
			scratch[local_id] += scratch[local_id + i];
			}
		barrier(CLK_LOCAL_MEM_FENCE);
	}      
	
     

  // Write result into partialSums[nWorkGroups]
  if (local_id == 0)
    output[get_group_id(0)] = scratch[0];	
  
 }	

 __kernel void reduce(__global uint4* input, __local uint4* sdata, __global uint4* output)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    sdata[tid] = input[stride] + input[stride + 1];

    barrier(CLK_LOCAL_MEM_FENCE);
    // do reduction in shared mem
    for(unsigned int s = localSize >> 1; s > 0; s >>= 1) 
    {
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) output[bid] = sdata[0];
}