#ifndef CUDACONLY
#define CUDACONLY
/* Includes, cuda */
#include "stdafx.h"
//#include "cuPrintf.cu"

extern int iDebugLvl, iDevDebug, iTrace;
extern float gElapsedTime, gKernelTimeTally;

// device-global variables to facilitate data transfer
//__device__ __align__(16) __constant__ struct rohanContext devSes;
//__device__ __align__(16) __constant__ struct rohanLearningSet devLearn;
//__device__ __align__(16) struct rohanNetwork devNet;
//__device__ __align__(16) const cuDoubleComplex gpuZero = { 0, 0 };
//__device__ __align__(16) double devdReturn[1024*1024];
//__device__ __align__(16) double devdRMSE=0;
//__device__ __align__(16) int devlReturn[1024*1024];
//__device__ __align__(16) int dTrainable=0;
//__device__ __align__(16) int iDevDebug=0;

#endif
