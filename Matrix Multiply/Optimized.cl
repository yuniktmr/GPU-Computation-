; __kernel void Optimized(__global const float* A, __global const float* B, __local float* C,int wA,int wB,int wC)
; {
; 	#define BLOCK_SIZE 16
; 	float csub = 0.0f;
; 	int by = get_group_id(1);
; 	int bx = get_group_id(0);
; 	int tx = get_local_id(0);
; 	int ty = get_local_id(1);
; 	int aBegin = wA * BLOCK_SIZE * by;
; 	int bBegin = BLOCK_SIZE * bx;
; 	int b;
; 	__global uint * AS = (__global uint*) A;
; 	__global uint * BS = (__global uint*) B;
; 	for( int a = aBegin, b = bBegin; a <= aBegin + wA - 1, a += BLOCK_SIZE, b += BLOCK_SIZE * wB ){
; 		AS[ty * BLOCK_SIZE + tx] = A[a + ty * wA + tx];
; 		BS[ty * BLOCK_SIZE + tx] = B[b + ty * wB + tx];
; 		barrier(CLK_LOCAL_MEM_FENCE);
		
; 		for (int k = 0; k < BLOCK_SIZE; k++){
; 			csub += AS[ty * BLOCK_SIZE + k] * BS[k * BLOCK_SIZE + tx];
; 		}
; 		barrier(CLK_LOCAL_MEM_FENCE);
; 	}
; 	C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = csub; 
; }


/* Matrix multiplication: C = A * B.
 * Device code.
 */
 
// Thread block size
#define BLOCK_SIZE 16
  
//////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
//////////////////////////////////////////////////////
__kernel void
Optimized(__global float* A, 
          __global float* B, 
          __local float* C, int wA, int wB, int wC)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
 
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
 
    // Index of the first sub-matrix of A processed 
    // by the block
    int aBegin = wA * BLOCK_SIZE * by;
 
    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd   = aBegin + wA - 1;
 
    // Step size used to iterate through the 
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;
 
    // Index of the first sub-matrix of B processed 
    // by the block
    int bBegin = BLOCK_SIZE * bx;
 
    // Step size used to iterate through the 
    // sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;
 
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) 
    {

        // Declaration of the local memory array As 
        // used to store the sub-matrix of A
        __local float As[BLOCK_SIZE][BLOCK_SIZE];
 
        // Declaration of the local memory array Bs 
        // used to store the sub-matrix of B
        __local float Bs[BLOCK_SIZE][BLOCK_SIZE];
 
        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty * BLOCK_SIZE + tx] = A[a + wA * ty + tx];
        Bs[ty * BLOCK_SIZE + tx] = B[b + wB * ty + tx];
 
        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx];
 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;

}
