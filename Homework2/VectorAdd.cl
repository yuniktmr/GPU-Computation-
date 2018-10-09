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
 
 // OpenCL Kernel Function for element by element vector addition
__kernel void VectorAdd(__global const float* a, __global const float* b, __global float* c, int iNumElements)
{
// get index into global data array

    int iGIDx = get_global_id(0);
   int iGIDy=get_global_id(1);
    int width=get_global_size(0);
   int tid=iGIDx+width*iGIDy;

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
   // if (iGID >= iNumElements)
   // {   
     //   return; 
   // }
    
    // add the vector elements
    c[tid] = a[tid] + b[tid];
}

