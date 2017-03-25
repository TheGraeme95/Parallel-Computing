//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

__kernel void reduce_add_4(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

__kernel void Reduction(__global const int* A, __global const int* B, __local int* localData)
{ 
	size_t globalID = get_global_id(0);
	size_t localSize = get_local_size(0);
	size_t localID = get_local_id(0);

	localData[localID] = A[globalID];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = localSize >> 1; i > 0; i >>= 1)
	{
		if (localID < i)
			localData[localID] += localData[localID + i];
			
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(localID == 0)
		B[get_group_id(0)] = localData[0];

}
