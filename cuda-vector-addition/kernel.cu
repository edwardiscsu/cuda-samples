
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define SIZE 10

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
	a = (int*)malloc(SIZE*sizeof(int));
	b = (int*)malloc(SIZE*sizeof(int));
	c = (int*)malloc(SIZE*sizeof(int));

	//Allocate GPU memory
	cudaMalloc(&d_a, SIZE*sizeof(int));
	cudaMalloc(&d_b, SIZE*sizeof(int));
	cudaMalloc(&d_c, SIZE*sizeof(int));

	for (int i = 0; i < SIZE; ++i) //Populate array
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	//Copy data to GPU
	cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, SIZE*sizeof(int), cudaMemcpyHostToDevice);

	VectorAdd<<< 1, SIZE >>>(d_a, d_b, d_c, SIZE); //Run GPU using 1 block and SIZE number of threads

	//Copy result back to CPU
	cudaMemcpy(c, d_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nSIZE (%d) VECTOR ADDITION USING CUDA\n\n", SIZE);
	printf("c[i] = a[i] + b[i]\n");
	printf("======================================\n");
	for (int i = 0; i < SIZE; ++i)
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