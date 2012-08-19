#ifndef ROHAN_KERNEL_H
#define ROHAN_KERNEL_H

extern"C"
int SetDevDebug(int iDevDebug);

extern"C"
int knlBackProp(struct rohanContext& rSes, int o, char Option, int iBlocks, int iThreads, char cModel);

__global__ void mtkBackPropMT( int lSampleQtyReq, int o, char Option, char cModel);

__device__ void subkBackPropMT(int lSample, int o);

__device__ void subkBackPropRBoptMT(int lSampleQtyReq, int o);

__device__ void subkBackPropRGoptMT(int lSampleQtyReq, int o);

__device__ void subkBackPropSBoptMT(int s, int o, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval );

__device__ void subkBackPropSGoptMT(int s, int o, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval );

__device__ void subkBackPropEoptMT(int lSampleQtyReq, int o);

extern"C"
double knlFFeRmseOpt(struct rohanContext& rSes, int o, char Option, int iBlocks, int iThreads);

__global__ void mtkFFeRmseOptMT( int lSampleQtyReq, int o, char Option);

__device__ void subkRmseMTBeta(int lSampleQtyReq, int o, int OUTROWLEN, double * dSqrErr);

__device__ void subkRmseMTGamma(int lSampleQtyReq, int o, int OUTROWLEN, double * dSqrErr);

__device__ void subkRmseMTDelta(int lSampleQtyReq, int o, int OUTROWLEN, double * dSqrErr);

__global__ void mtkRmseFinalMT(int limit);

__device__ void subkEvalSampleBetaMT(rohanContext& Ses, int s, rohanNetwork& Net, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval, double * dSqrErr);

__device__ void subkEvalSampleSingleThread(int s, char Option, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval, double * dSqrErr);

__device__ double FUnitCxUT(const cuDoubleComplex A);

__device__ cuDoubleComplex CxAddCxUT(const cuDoubleComplex A, const cuDoubleComplex B);

__device__ cuDoubleComplex CxMultiplyCxUT(const cuDoubleComplex A, const cuDoubleComplex B);

__device__ cuDoubleComplex CxActivateUT(const cuDoubleComplex Z, rohanNetwork& Net);

__device__ cuDoubleComplex CxMultiplyRlUT(const cuDoubleComplex A, const double Rl);

__device__ cuDoubleComplex CxDivideRlUT(const cuDoubleComplex A, const double Rl);

__device__ double CxAbsUT(const cuDoubleComplex Z);

__device__ cuDoubleComplex CxSubtractCxUT(const cuDoubleComplex A, const cuDoubleComplex B);

__device__ cuDoubleComplex CxDivideCxUT(const cuDoubleComplex A, const cuDoubleComplex B);

__device__ cuDoubleComplex CxConjugateUT(const cuDoubleComplex Z);

//__device__ int d_answer;

extern "C" 
long knlCRC32Buf(char * buffer, unsigned int length);

__global__ __device__ void mtkCRC32Buf(char * buffer, unsigned int length);

__device__ long subkCrc32buf(char *buf, size_t len);

__device__ double atomicAdd(double* address, double val);

__device__ void __checksum(char * sLabel);

#endif
