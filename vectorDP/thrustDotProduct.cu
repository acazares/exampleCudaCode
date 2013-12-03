#include <cuComplex.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <math.h>
#include <thrust/device_vector.h>
#include "cublas_v2.h"

struct dot_functor {
  
  dot_functor() {}
  
  __host__ __device__
  
  cuDoubleComplex operator()(const cuDoubleComplex& A, const cuDoubleComplex& B) const {
    return cuCmul(cuConj(A),B);
  }
};

struct sum_functor {

  sum_functor() {}

  __host__ __device__

  cuDoubleComplex operator()(const cuDoubleComplex& A, const cuDoubleComplex& B) const {
    return cuCadd(A,B);
  }
};


int main() {
  
  int numElements = pow(2,21);
  
  thrust::device_vector<cuDoubleComplex> A(numElements);
  thrust::device_vector<cuDoubleComplex> B(numElements);
  cuDoubleComplex CDouble = make_cuDoubleComplex(2.25,2.25);
  
  thrust::fill(A.begin(), A.end(), CDouble);
  thrust::fill(B.begin(), B.end(), CDouble);
  
  thrust::transform(A.begin(), A.end(), B.begin(), B.begin(), dot_functor());
  
  cuDoubleComplex result = thrust::reduce(B.begin(), B.end(), (cuDoubleComplex) make_cuDoubleComplex(0.0,0.0), sum_functor());
  
  return 0;
}
