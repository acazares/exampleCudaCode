=========
vectorAdd
=========

This directory has cuda code that is working with vector
addition. There are a total of five files. The thrust
file is using the thrust template library as opposed to 
the rest of the files that aren't using any.

=====
Files
=====

	vectorCD
	========
	
	This program is simply doing vector addition with
	complex doubles. The operation that is done is
	Zaxpy. Assuming one has two vectors A and B and
	any complex double D, Zaxpy is B = (A*D) + B.
	Nothing is out of the ordinary, each thread is
	doing a single operation.

	vectorCDv2
	==========
	
	This program is almost all the same as the 
	previous version except the kernel was changed
	so that this time each thread does two 
	operations. Perhaps adding more work to each warp
	may increase performance.
	
	vectorCDv3
	==========
	
	For this program, unlike the second version the
	opposite was done. Now only every other thread
	will perform the operation. Perhaps performance
	may increase if less work is done per warp.
	
	vectorCDv4
	==========

	This program goes back to the first version 
	where every thread is doing a single operation
	only this time all the work is done with shared 
	memory. Perhaps if less time is spent reading
	and writing date, performance might increase.

	
	vectorThrust
	============

	This program does pretty much what the first 
	version of my vectorAdd does only everything
	is done with the Thrust template library.
	The main algorithm that is used is transform,
	which applies a functor to the vectors that
	are given. The functor performs the Zaxpy
	operation.

=========
Compiling
=========

All of these files are compiled with the NVDIA CUDA 
compiler driver NVCC.

	 Examples
	 =======

	 nvcc vectorCD.cu
	 nvcc vectorThrust.cu

