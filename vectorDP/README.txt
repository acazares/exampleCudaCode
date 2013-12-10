========
vectorDP
========

This directory contains three files that perform the
dot product operation.

=====
Files
=====
	
	myZdotc
	=======

	This program performs the dot product operation
	using three kernels. The first kernel is the 
	first version of vectorAdd that is in the other
	directory in this depository. The first version
	was used because if one compiles the different
	versions, they will notice that the first has the
	best performance. The second kernel performs the
	reduce operation. This kernel does all the work
	with shared memory. This kernel was borrowed from
	Mark Harris from NVIDIA. The third kernel calls
	the other two kernels on the GPU. This dot 
	product is done with dynamic parallelism.

	thrustDotProduct
	================

	This program performs the dot product operation
	using the Thrust template library. The algorithms
	that are used are transform and reduce. Transform
	is used in the same manner that it was used in 
	the vectorAdd directory. Reduce is used also in
	the same manner, where reduction is done by
	by a functor that tells reduce how to compute.

	cublasZdotc
	===========

	This program is straight forward, it simply calls
	an already defined cuda dot product function.

=========
Compiling
=========

Each of these files are compiled differently.

     	myZdotc.cu
     	==========

	nvcc -arch=sm_35 -rdc=true myZdotc.cu -lcudadevrt
	
	thrustDotProduct	
	================
	
	nvcc thrustDotProduct.cu

	cublasZdotc
	===========

	nvcc -lcublas cublasZdotc
