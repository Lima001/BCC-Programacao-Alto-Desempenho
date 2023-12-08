/*
	Baseado em:
	-- https://alexminnaar.com/2019/07/12/implementing-convolutions-in-cuda.html
	-- https://github.com/pietrobongini/CUDA-ImageConvolution/tree/master

	CUDA 12.0
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MASK_DIM 3

__constant__ float mask[MASK_DIM*MASK_DIM];

__global__ void convolution_constant_memory(float *src, float *dst, int width, int height, int mask_dim){
	
	float value;
	int col = threadIdx.x + blockIdx.x * blockDim.x;   //col index
	int row = threadIdx.y + blockIdx.y * blockDim.y;   //row index
	
	int maskRowsRadius = mask_dim/2;
	int maskColsRadius = mask_dim/2;


	if(row < height && col < width){
		value = 0;
		int startRow = row - maskRowsRadius;  //row index shifted by mask radius
		int startCol = col - maskColsRadius;  //col index shifted by mask radius

		for(int i=0; i<mask_dim; i++){ //cycle on mask rows

			for(int j=0; j<mask_dim; j++){ //cycle on mask columns

				int currentRow = startRow + i; // row index to fetch data from input image
				int currentCol = startCol + j; // col index to fetch data from input image

				if(currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width){
					value += src[currentRow * width + currentCol] * mask[i * mask_dim + j];
				}
			}

		}
		dst[(row* width + col)] = value;
	}

}

int main(int argc, char *argv[]){

	if (argc != 3)
        return 1;

    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    //int mask_dim = atoi(argv[3]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	//device input and output
	float *d_src = 0;
	float *d_dst = 0;

	cudaMalloc(&d_src, (width*height)*sizeof(float));
	cudaMalloc(&d_dst, (width*height)*sizeof(float));


	//host input and output
	float *h_src = (float*) malloc((width*height)*sizeof(float));
	float *h_dst = (float*) malloc((width*height)*sizeof(float));
	float *h_kernel = (float*) malloc((MASK_DIM*MASK_DIM)*sizeof(float));

	//initialize input on host
	for (int i=0; i<(width*height); i++)
		h_src[i] = (float) (std::rand()%256);

    //initialize mask on host
	for(int i=0; i<(MASK_DIM*MASK_DIM); i++)
		h_kernel[i] = (float) (std::rand()%256);

	//transfer input to device
	cudaMemcpy(d_src, h_src, (width*height)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, h_dst, (width*height)*sizeof(float), cudaMemcpyHostToDevice);

	//transfer mask to constant memory
	cudaMemcpyToSymbol(mask,h_kernel,(MASK_DIM*MASK_DIM)*sizeof(float));

    //call convolution kernel
	dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    cudaEventRecord(start);
	convolution_constant_memory<<<numBlocks, threadsPerBlock>>>(d_src,d_dst,width,height,MASK_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Shared Memory: %f ms", milliseconds);

	//retrieve result from device
	cudaMemcpy(h_dst,d_dst,(width*height)*sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_dst);
	cudaFree(mask);

	free(h_src);
	free(h_dst);
	free(h_kernel);

    return 0;
}