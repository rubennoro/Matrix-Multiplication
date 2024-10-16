#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#define LEN 1024
#define BLOCK_SIZE 16
/*
#define CUDA_CHECK(err) { if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); }}
*/

//this math is correct
__global__ void gpumatrixmulti(double *a, double *b, double *c, int N){
	//printf("here\n");
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < N && col < N){

	       float sum = 0;
	
		for(int i = 0; i < N; i++){
			sum += a[row*N + i] * b[i*N + col];
			}
			//printf("%f\n", sum);
			c[row*N +col] = sum;
	}
}

//this function works
void cpu_mm(double *a, double *b, double *c){
	float sum = 0;
	for(int i = 0; i < LEN; i++){
		for(int j = 0; j < LEN; j++){
			sum = 0;
			for(int k = 0; k < LEN; k++){
					sum += a[i * LEN + k] * b[k * LEN + j];	
			}
		c[i*LEN+j] = sum;
		}
	}
}

double CLOCK() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC,  &t);
  return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

int main(int argc, char* argv[]){
	printf("Running");
	int N = LEN;
    	double *h_a;
	double *h_b;
	double *h_c;
	
	cudaMallocHost((void **) &h_a, sizeof(double)*LEN*LEN);
    	cudaMallocHost((void **) &h_b, sizeof(double)*LEN*LEN);
    	cudaMallocHost((void **) &h_c, sizeof(double)*LEN*LEN);

	for(int i = 0; i < LEN*LEN; i++){
		h_a[i] = 1.0;
		h_b[i] = 2.0;
		h_c[i] = 5.0;
	}
	
	float gpu_time_ms, total_time;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//double clock_start, finish;
	
	
	//cudaEventRecord(start, 0);
	
	//copies data from the host to the GPU device
	double *d_a;
	double *d_b;
	double *d_c;
	cudaMalloc((void **) &d_a, sizeof(double)*LEN*LEN);
	cudaMalloc((void **) &d_b, sizeof(double)*LEN*LEN);
	cudaMalloc((void **) &d_c, sizeof(double)*LEN*LEN);
	
	cudaMemcpy(d_a, h_a, sizeof(double)*LEN*LEN, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(double)*LEN*LEN, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, sizeof(double)*LEN*LEN, cudaMemcpyHostToDevice);
	
	unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	printf("HERE\n");
	
	for(int iterations = 0; iterations < 200; iterations++){
		
		memset(h_c, 0.0, sizeof(double)*LEN*LEN);
		cudaEventRecord(start, 0);	
		gpumatrixmulti<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
	
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
    	   	   fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    	   	   //return -1; // Handle the error as appropriate
	   	   }

		 cudaDeviceSynchronize();
		 //printf("DONE\n");
		 //cudaMemcpy(h_c, d_c, sizeof(float)*LEN*LEN, cudaMemcpyDeviceToHost);
		 cudaEventRecord(stop, 0);
		 cudaEventSynchronize(stop);
		 cudaMemcpy(h_c, d_c, sizeof(double)*LEN*LEN, cudaMemcpyDeviceToHost);
	
		cudaEventElapsedTime(&gpu_time_ms, start, stop);
		total_time += gpu_time_ms;
		
		
	}				  
	printf("Time elapsed: %f \n", total_time);
	//double diff = finish - clock_start;	
	//printf("Clock time elapse: %f \n", diff);
	printf("Result at (0,0): %f\n", h_c[5]);
	
	//for (int i = 0; i < 10; i++) {
    	//    for (int j = 0; j < 10; j++) {
        //    	printf("Result at (%d,%d): %f\n", i, j, h_c[i * LEN + j]);
    	//	}
	//}
	float cpu_time_ms;
	cudaEventRecord(start, 0);
	
	cpu_mm(h_a, h_b, h_c);
	
	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&cpu_time_ms, start, stop);
	
	printf("CPU Time: %f\n", cpu_time_ms);
	printf("Result is %f\n", h_c[5]);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFreeHost(h_a);
    	cudaFreeHost(h_b);
    	cudaFreeHost(h_c);
	return 0;
}
