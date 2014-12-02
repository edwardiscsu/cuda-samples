
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MAXBLOCKS 1
#define MAXTHREADS 10

//__global__ (paralellized method)
__global__ void VectorAdd(int *a, int *b, int*c, int n)
{
	int i = threadIdx.x; //Assign each c element to a single processor
	if (i < n) //Make sure there are no processing overlap
		c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b, *c;      //CPU
	int *d_a, *d_b, *d_c;//GPU

	//Allocate CPU memory
	a = (int*)malloc(MAXTHREADS*sizeof(int));
	b = (int*)malloc(MAXTHREADS*sizeof(int));
	c = (int*)malloc(MAXTHREADS*sizeof(int));

	//Allocate GPU memory
	cudaMalloc(&d_a, MAXTHREADS*sizeof(int));
	cudaMalloc(&d_b, MAXTHREADS*sizeof(int));
	cudaMalloc(&d_c, MAXTHREADS*sizeof(int));

	for (int i = 0; i < MAXTHREADS; ++i) //Populate array
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	//Copy data to GPU
	cudaMemcpy(d_a, a, MAXTHREADS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, MAXTHREADS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, MAXTHREADS*sizeof(int), cudaMemcpyHostToDevice);

	VectorAdd<<< MAXBLOCKS, MAXTHREADS >>>(d_a, d_b, d_c, MAXTHREADS); //Run GPU using 1 block and MAXTHREADS number of threads

	//Copy result back to CPU
	cudaMemcpy(c, d_c, MAXTHREADS*sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nMAXTHREADS (%d) VECTOR ADDITION USING CUDA\n\n", MAXTHREADS);
	printf("c[i] = a[i] + b[i]\n");
	printf("======================================\n");
	for (int i = 0; i < MAXTHREADS; ++i)
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