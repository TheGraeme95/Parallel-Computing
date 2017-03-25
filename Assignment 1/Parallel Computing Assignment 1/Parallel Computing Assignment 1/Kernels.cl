__kernel
void reduce(__global float* buffer,
            __local float* scratch,
            __const int length,
            __global float* result) {

  int global_index = get_global_id(0);
  float accumulator = INFINITY;
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float element = buffer[global_index];
    accumulator = (accumulator < element) ? accumulator : element;
    global_index += get_global_size(0);
  }

  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratch[local_index] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = (mine < other) ? mine : other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}

__kernel void Sum(__global int* input, __local int* scratch, __global int* output) {

	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = input[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
		}

		//copy the cache to output array
	output[id] = scratch[lid];
	barrier(CLK_GLOBAL_MEM_FENCE);

 }

 __kernel void Sum2(__global int* input, __local int* localData, __global int* output)
{

	size_t global_ID = get_global_id(0);
	size_t local_Size = get_local_size(0);
	size_t local_ID = get_local_id(0);

	localData[local_ID] = input[global_ID];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = local_Size >> i; i > 0; i >>= 1)
	{ 
		if (local_ID < i)
			localData[local_ID] += localData[local_ID + i];

		barrier(CLK_LOCAL_MEM_FENCE);

	}

	if (local_ID == 0)
		output[get_group_id(0)] = localData[0];




 }