
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MAXBLOCKS 1
#define MAXTHREADS 10

//Helper method
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//__global__ (paralellized method)
__global__ void VectorAdd(int *c, const int *a, const int *b)
{
    int i = threadIdx.x; //Assign each c element to a single thread
	c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b, *c;      //CPU

	//Allocate CPU memory
	a = (int*)malloc(MAXTHREADS*sizeof(int));
	b = (int*)malloc(MAXTHREADS*sizeof(int));
	c = (int*)malloc(MAXTHREADS*sizeof(int));

	for (int i = 0; i < MAXTHREADS; ++i) //Populate array
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

    //Call "surrogate" method
	cudaError_t cudaStatus = addWithCuda(c, a, b, MAXTHREADS);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	//Display result
    printf("\nMAXTHREADS (%d) VECTOR ADDITION USING CUDA\n\n", MAXTHREADS);
	printf("c[i] = a[i] + b[i]\n");
	printf("======================================\n");
	for (int i = 0; i < MAXTHREADS; ++i)
		printf("a[%d] = %d, b[%d] = %d, c[%d] = %d\n", i, a[i], i, b[i], i, c[i]);

	//Free CPU memory
	free(a);
	free(b);
	free(c);

    //cudaDeviceReset must be called before exiting in order for profiling and
    //tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

//Helper/"surrogate" method for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *d_a = 0;//GPU
    int *d_b = 0;//GPU
    int *d_c = 0;//GPU
    cudaError_t cudaStatus;

    //Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //Allocate GPU memory
    cudaStatus = cudaMalloc((void**)&d_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //Copy data to GPU
    cudaStatus = cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	//Run GPU using MAXBLOCK number of blocks and size number of threads
	VectorAdd<<<MAXBLOCKS, size>>>(d_c, d_a, d_b); 

    //Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    //cudaDeviceSynchronize waits for the kernel to finish, and returns
    //any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    //Copy result back to CPU
    cudaStatus = cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	//Free GPU memory
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);
    
    return cudaStatus;
}
