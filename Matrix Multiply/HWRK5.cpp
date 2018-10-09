#include <stdio.h>
#include <iostream>
#include <string.h>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <CL/opencl.h>
#include<time.h>
#include<sys/time.h>
//Inline your kernel or use a separate .cl file

//const char *kernelSource =
  //"__kernel void VectorAdd(  __global float *a,                       \n" \
  //"                       __global float *b,                       \n" \
  //"                       __global float *c,                       \n" \
  //"                       int n)                    \n"		\
  //"{                                                               \n" \
  //"    //Get our global thread ID                                  \n" \
  //"    int iGIDx = get_global_id(0);                                  \n" \
  //"    int iGIDy=get_global_id(1);
  // int width=get_global_id(0);
  // int tid=iGIDy*width+iGIDx;                                 \n"	\
  //"                                                                \n" \
  //"    //Make sure we do not go out of bounds                      \n" \
// "    if (id < n)                                                 \n"	\
// "        c[id] = a[id] + b[id];                                  \n"	\
  ///"}                                                               \n" \
  //                                                               "\n" ;

void randomMemInit(float* data, int size)
{
   int i;

   for (i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char*  str;

    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);

        str = new char[size+1];
        if(!str)
        {
            f.close();
            return 1;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
    
        s = str;
        delete[] str;
        return 0;
    }
    printf("Error: Failed to open file %s\n", filename);
    return 1;
}



float *srcA, *srcB, *srcC, *dst;        // Host buffers for OpenCL test
float *Golden;                   // Host buffer for host golden processing cross check

cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmDevSrcA;               // OpenCL device source buffer A
cl_mem cmDevSrcB;               // OpenCL device source buffer B 
cl_mem cmDevSrcC;                // OpenCL device destination buffer 
size_t szGlobalWorkSize[2];        // 1D var for Total # of work items
size_t szLocalWorkSize[2];		// 1D var for # of work items in the work group	
size_t szParmDataBytes;		// Byte size of context information
size_t szKernelLength;		// Byte size of kernel code
cl_int ciErr1, ciErr2;		// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation

int iNumElements = 2560;        //Number of elements on host machine
void MatrixMultiply(const float* A, const float* B, float* C,int wA,int wB,int wC);
void Cleanup (int argc, char **argv, int iExitCode);

int main(int argc, char **argv)
{

    const char * filename  = "VecMultiply.cl";
    std::string  sourceStr;
    ciErr1 = convertToString(filename, sourceStr);
    if(ciErr1 != CL_SUCCESS) {
        printf("Error in convertToString, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        return EXIT_FAILURE;
    }

    const char * source    = sourceStr.c_str();
    size_t sourceSize[]    = { strlen(source) };   

//=========================================================================================
       //Matrix A
    int WA=1024;
    int HA=1024 ; 
    unsigned int size_A=WA*HA;
    srcA = (float *)malloc(sizeof(cl_float) * size_A);
    int WB=1024;
    int HB=1024;
    unsigned int size_B=WB*HB;
    srcB = (float *)malloc(sizeof(cl_float) * size_B);
    int WC=1024;
    int HC=1024;
    unsigned int size_C=WC*HC;
    srcC = (float *)malloc(sizeof(cl_float) * size_C);
    randomMemInit(srcA, size_A);
    randomMemInit(srcB, size_B);
    // dst = (float *)malloc(sizeof(cl_float) * size_C);
    Golden = (float *)malloc(sizeof(cl_float) * size_C);
    // for(int idx = 0; idx < iNumElements; idx++) {
    //   srcA[idx] = (float)idx;
    //   srcB[idx] = (float)idx;
    //   srcC[idx]=(float)idx;
    // }
//c=========================================================================================
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got platform\n");
//=========================================================================================
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got device\n");
//=========================================================================================
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got context\n");
//=========================================================================================
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice,CL_QUEUE_PROFILING_ENABLE, &ciErr1);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got commandqueue\n");
//=========================================================================================
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, &source, sourceSize, &ciErr1);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got createprogramwithsource\n");
//=========================================================================================
    ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clBuildProgram, Line %u in file %s !!! Error code = %d\n\n", __LINE__, __FILE__, ciErr1);
        size_t length;
        char buffer[2048];
        clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        //printf("--- Build log ---\n%s\n", buffer);
	std::cout << "--- Build log ---\n" << buffer << std::endl;
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got buildprogram\n");
//=========================================================================================
    ckKernel = clCreateKernel(cpProgram, "VecMultiply", &ciErr1);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got createkernel\n");
//=========================================================================================
     cmDevSrcC = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * size_C, NULL, &ciErr2);
    cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * size_A, srcA, &ciErr1);
    cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * size_B, srcB, &ciErr2);
    
    // ciErr1 |= ciErr2;
    if (!cmDevSrcA||!cmDevSrcB||!cmDevSrcC) {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got createbuffer\n");
//=========================================================================================
    printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n", 
   WA,HA,WB,HB); 

    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA);
    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevSrcB);
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevSrcC);
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&WA);
    ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_int), (void*)&WB);
     ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_int), (void*)&WC);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got setkernelarg\n");
//=========================================================================================
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * size_A, srcA, 0, NULL, NULL);//THIS IS DATA COPY FROM HOST TO DEVICE
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float) * size_B, srcB, 0, NULL, NULL);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got enqueuewritebuffer\n");
//=========================================================================================


    szGlobalWorkSize[0] = 1024;
    szGlobalWorkSize[1] = 1024; 
    szLocalWorkSize[0] = 16;
    szLocalWorkSize[1] = 16;

    cl_event event;
    ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL,szGlobalWorkSize,szLocalWorkSize, 0, NULL,&event);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got enqueuendrangekernel\n");
    clWaitForEvents(1,&event);
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(time_start),&time_start,NULL); 
      clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(time_end),&time_end,NULL);
      double nano=(time_end-time_start);
      printf("OpenCL Execution time is: %0.3f microsecs \n", nano/1000000.0);
      printf("\n\nMatrix C (Results)\n");
      
      
//=========================================================================================
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmDevSrcC, CL_TRUE, 0, sizeof(cl_float) *size_C , srcC, 0, NULL, NULL);//copy result from device to host
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got enqueuereadbuffer\n");
//=========================================================================================

    printf("\n");
    printf("*** Got result from GPU \n");

    for(int idx = 0; idx < 5; idx++) {
      printf("%f ", srcC[idx]);
     }
     printf(" ... ...");
//=========================================================================================
    struct timeval begin, end;
    gettimeofday (&begin, NULL);
    MatrixMultiply((const float*) srcA, (const float*) srcB,(float*) Golden,WA,WB,WC);
    gettimeofday (&end, NULL);
    int time = 1e6 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec);
    printf("\nCPU Execution time is %d microsecs",time);
    printf("\n\n*** Got result from HOST\n");

    for(int idx = 0; idx < 5; idx++) {
      printf("%f ", Golden[idx]);
    }
     printf(" ... ...");

    bool bMatch = true;
    for(int idx = 0; idx < iNumElements; idx++) {
      if(Golden[idx] != srcC[idx]) {
        bMatch = false;
	printf("idx %d doesn't match. %f vs %f", idx, Golden[idx], dst[idx]);
	break;
      }
    }

    printf("\n\n*** Got checkMatch");
    Cleanup (argc, argv, (bMatch == true) ? EXIT_SUCCESS : EXIT_FAILURE);
}
//=========================================================================================



//Helper functions
void Cleanup (int argc, char **argv, int iExitCode)
{
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
    if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cmDevSrcA)clReleaseMemObject(cmDevSrcA);
    if(cmDevSrcB)clReleaseMemObject(cmDevSrcB);
    if(cmDevSrcC)clReleaseMemObject(cmDevSrcC);

    free(srcA); 
    free(srcB);
    free(srcC);
    free(Golden);

    if(iExitCode == EXIT_SUCCESS)
      printf("\n******* PASSed\n");
    else
      printf("\n******* FAILed\n");
}
void MatrixMultiply(const float* A, const float* B, float* C,int wA,int wB,int wC)
{
  for(int i=0;i<wA;i++){
    for(int j = 0; j<wB; j++){
      C[i*wC + j] = 0.0f;
      for(int k = 0; k<wC; k++){
	C[i*wC+j] = C[i*wC+j] + A[i*wA+k] * B[k*wB+j];
      }
    }
  }
}
