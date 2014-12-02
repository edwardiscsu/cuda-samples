
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MAXBLOCKS 1
#define MAXTHREADS 10

__global__ void HelloWorld()
{
	printf("core block %d\'s thread no. %d says: Hello World!\n", blockIdx.x, threadIdx.x);
}

int main()
{
	int *d_a;
	cudaMalloc(&d_a, MAXTHREADS*sizeof(int));

	HelloWorld<<<MAXBLOCKS, MAXTHREADS>>>();

	cudaFree(d_a);
	return 0;
}