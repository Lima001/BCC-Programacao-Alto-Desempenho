/*
	Baseado em:
	-- https://alexminnaar.com/2019/07/12/implementing-convolutions-in-cuda.html
	-- https://github.com/pietrobongini/CUDA-ImageConvolution/tree/master

	CUDA 12.0
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convolution_global_memory(float *src, float *dst, float *mask, int width, int height, int mask_dim){
	
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
		dst[row* width + col] = value;
	}

}

int main(int argc, char *argv[]){

	if (argc != 4)
        return 1;

    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    int mask_dim = atoi(argv[3]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	//device input and output
	float *d_src = 0;
	float *d_dst = 0;
	float *d_mask = 0;

	cudaMalloc(&d_src, (width*height)*sizeof(float));
	cudaMalloc(&d_dst, (width*height)*sizeof(float));
	cudaMalloc(&d_mask, (mask_dim*mask_dim)*sizeof(float));

	//host input and output
	float *h_src = (float*) malloc((width*height)*sizeof(float));
	float *h_dst = (float*) malloc((width*height)*sizeof(float));
	float *h_mask = (float*) malloc((mask_dim*mask_dim)*sizeof(float));

	//initialize input on host
	for (int i=0; i<(width*height); i++)
		h_src[i] = (float) (std::rand()%256);

    //initialize mask on host
	for(int i=0; i<(mask_dim*mask_dim); i++)
		h_mask[i] = (float) (float) (std::rand()%256);

	//transfer input to device
	cudaMemcpy(d_src, h_src, (width*height)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, h_dst, (width*height)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, h_mask, (mask_dim*mask_dim)*sizeof(float), cudaMemcpyHostToDevice);

    //call convolution mask
	dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    cudaEventRecord(start);
	convolution_global_memory<<<numBlocks, threadsPerBlock>>>(d_src,d_dst,d_mask,width,height,mask_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Global Memory: %f ms\n", milliseconds);

	//retrieve result from device
	cudaMemcpy(h_dst,d_dst,(width*height)*sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_dst);
	cudaFree(d_mask);

	free(h_src);
	free(h_dst);
	free(h_mask);

    return 0;
}