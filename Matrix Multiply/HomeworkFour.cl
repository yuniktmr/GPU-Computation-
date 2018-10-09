/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 // OpenCL Kernel Function for element by element matrix multiplication
__kernel void HomeworkFour(__global const float* A, __global const float* B, __global float* C,int wA,int wB,int wC)
{
   int iGIDx = get_global_id(0);
   int iGIDy = get_global_id(1);
   float value=0;
   for(int k=0;k<wA;k++){
      float eleA=A[iGIDx*wA+k];
	  float eleB=B[k*wB+iGIDy];
	  value+=eleA*eleB;
	}
   	C[iGIDx*wC+iGIDy]=value;      
}

