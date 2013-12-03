#include <iostream>
#include <cuComplex.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/system_error.h>
#include <math.h>

struct zaxpy_functor {
  const cuDoubleComplex x;

  zaxpy_functor(cuDoubleComplex _x) : x(_x) {}

  __host__ __device__
  cuDoubleComplex operator()(const cuDoubleComplex& A, const cuDoubleComplex& B) const {
    return cuCadd(cuCmul(A,x),B);
  }
};

int main() {

  int numElements = pow(2,20);
  
  thrust::device_vector<cuDoubleComplex> A(numElements);
  thrust::device_vector<cuDoubleComplex> B(numElements);
  cuDoubleComplex CDouble = make_cuDoubleComplex(2.25,2.25);

  thrust::fill(A.begin(), A.end(), CDouble);
  thrust::fill(B.begin(), B.end(), CDouble);
    
  thrust::transform(A.begin(), A.end(), B.begin(), B.begin(), zaxpy_functor(CDouble));

  return 0;
}
