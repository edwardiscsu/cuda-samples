
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MAXBLOCKS 1
#define MAXTHREADS 1

__global__ void SimpleAddition(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main()
{
    int a, b, c;         //CPU
	int *d_a, *d_b, *d_c;//GPU

	//Allocate GPU memory
	cudaMalloc((void **)&d_a, sizeof(int));
	cudaMalloc((void **)&d_b, sizeof(int));
	cudaMalloc((void **)&d_c, sizeof(int));

	a = 1;
	b = 2;
	c = 0;

	//Copy data to GPU
	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);

	SimpleAddition<<<MAXBLOCKS, MAXTHREADS>>>(d_a, d_b, d_c);

	//Copy result back to CPU
	cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("%d + %d = %d\n", a, b, c);

	//Free GPU memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

    return 0;
}