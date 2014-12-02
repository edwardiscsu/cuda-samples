
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MAXBLOCKS 10
#define MAXTHREADS 1

//__global__ (paralellized method)
__global__ void VectorAdd(int *a, int *b, int*c, int n)
{
	int i = blockIdx.x; //Assign each c element to a single block
	c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b, *c;      //CPU
	int *d_a, *d_b, *d_c;//GPU

	//Allocate CPU memory
	a = (int*)malloc(MAXBLOCKS*sizeof(int));
	b = (int*)malloc(MAXBLOCKS*sizeof(int));
	c = (int*)malloc(MAXBLOCKS*sizeof(int));

	//Allocate GPU memory
	cudaMalloc(&d_a, MAXBLOCKS*sizeof(int));
	cudaMalloc(&d_b, MAXBLOCKS*sizeof(int));
	cudaMalloc(&d_c, MAXBLOCKS*sizeof(int));

	for (int i = 0; i < MAXBLOCKS; ++i) //Populate array
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	//Copy data to GPU
	cudaMemcpy(d_a, a, MAXBLOCKS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, MAXBLOCKS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, MAXBLOCKS*sizeof(int), cudaMemcpyHostToDevice);

	VectorAdd<<< MAXBLOCKS, MAXTHREADS >>>(d_a, d_b, d_c, MAXBLOCKS); //Run GPU using MAXBLOCK number of blocks and MAXTHREADS number of threads

	//Copy result back to CPU
	cudaMemcpy(c, d_c, MAXBLOCKS*sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nMAXBLOCKS (%d) VECTOR ADDITION USING CUDA\n\n", MAXBLOCKS);
	printf("c[i] = a[i] + b[i]\n");
	printf("======================================\n");
	for (int i = 0; i < MAXBLOCKS; ++i)
		printf("a[%d] = %d, b[%d] = %d, c[%d] = %d\n", i, a[i], i, b[i], i, c[i]);

	//Free CPU memory
	free(a);
	free(b);
	free(c);

	//Free GPU memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;
}