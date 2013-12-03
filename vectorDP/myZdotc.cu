#include <math.h>
#include <cuComplex.h>
#include <iostream>
#include <stdio.h>

__global__ void Map(const cuDoubleComplex *A, cuDoubleComplex *B, int numElements) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;

  if(index < numElements){
    B[index] = cuCmul(cuConj(A[index]),B[index]);
  }
}

__global__ void reduce(cuDoubleComplex *B_idata, cuDoubleComplex *B_odata) {

  extern __shared__ cuDoubleComplex SM[];

  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  SM[tid] = B_idata[index];
  __syncthreads();

    
  for(int s = 1; s < blockDim.x; s *= 2) {
    if(tid % (2*s) == 0) {
      SM[tid] = cuCadd(SM[tid],SM[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    B_odata[blockIdx.x] = SM[0];
  }
}

__global__ void myZdotc(cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex *O, int numElements) {

  int blocksize = 128;
  int gridsize = ((numElements - 1) / blocksize) + 1;
  int SMsize = blocksize * sizeof(cuDoubleComplex);

  Map <<< gridsize, blocksize >>> (A,B,numElements);
  cudaDeviceSynchronize();
  cuDoubleComplex *temp;
  while (gridsize > 0) {
    reduce <<< gridsize, blocksize, SMsize >>>(B,A);
    cudaDeviceSynchronize();
    temp = A;
    A = B;
    B = temp;
    gridsize >>= 7;
  }
  O[0] = B[0];
}

int main() 
{
  int numElements = pow(2,21);
  size_t size = numElements * sizeof(cuDoubleComplex);
  size_t size1 = sizeof(cuDoubleComplex);

  cuDoubleComplex *h_A, *h_B, *h_O;
  cuDoubleComplex *d_A, *d_B, *d_O;

  cudaMallocHost((cuDoubleComplex**)&h_A, size);
  cudaMallocHost((cuDoubleComplex**)&h_B, size);
  cudaMallocHost((cuDoubleComplex**)&h_O, size1);

  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_O, size1);

  for(int i = 0; i < numElements; i++)
    {
      h_A[i] = make_cuDoubleComplex(2.25,2.25);
      h_B[i] = make_cuDoubleComplex(2.25,2.25);
    }
  
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_O, h_O, size1, cudaMemcpyHostToDevice);
  
  myZdotc <<< 1, 1 >>> (d_A,d_B,d_O,numElements);
  
  cudaMemcpy(h_O, d_O, size1, cudaMemcpyDeviceToHost);

  std::cout << cuCreal(h_O[0]) << ":" << cuCimag(h_O[0]) <<'\n';

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_O);

  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_O);

  cudaDeviceReset();

  return 0;
}
