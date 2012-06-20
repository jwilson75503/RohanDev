#ifndef ROHAN_H
#define ROHAN_H
//#define VERSION "0.7" included kernels
//#define VERSION "0.8" //included object model
//#define VERSION "0.9" // accurately working
//#define VERSION "0.9.1" // functional learning, continuous and discrete activation
//#define VERSION "0.9.2" // passes eval, classification, and learning tests
#define VERSION "0.9.3" // included shared device memory, logging

/*! Includes, cuda */
#include "stdafx.h"
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))

// Project defines and constants
#define MAXLAYERS 4
#define MAXWEIGHTS 1024
#define MAXNEURONS 256
#define _USE_MATH_DEFINES
#define ONE_PI 3.14159265358979323846264338327950288
#define TWO_PI 6.283185307179586476925286766558
#define ONE_OVER_TWO_PI 0.15915494309189533576888376337254

const cuDoubleComplex cdcZero = { 0, 0 }; 
const cuDoubleComplex cdcIdentity = { 1, 0 }; 

// allocated memory structure flags
#define RLEARNcdc 1
#define RLEARNd	2
#define RNETlayers 4
#define RNETbdry 8


// major data structures
typedef struct rohanLayer
{
	int iNeuronQty /*! # neurons hosted on this layer; 1-based, but neuron zero is wholly virtual */;
	int iDendriteQty /*! # neurons or inputs on previous layer; 1-based, with virtual signal source zero */;
	cuDoubleComplex *Weights /*! 2D neuron cx weight matrix */;
	cuDoubleComplex *XInputs /*! 1D layer's cx input signals */;
	cuDoubleComplex *ZOutputs /*! 1D layer's cx weighted sums/activated signals */;
	cuDoubleComplex *Deltas /*! 1D cx error correction values per neuron */;
	cuDoubleComplex *gpuWeights /*! 2D neuron cx weight matrix */;
	cuDoubleComplex *gpuXInputs /*! 1D layer's cx input signals */;
	cuDoubleComplex *gpuZOutputs /*! 1D layer's cx weighted sums/activated signals */;
	cuDoubleComplex *gpuDeltas /*! 1D cx error correction values per neuron */;
} rohanLayer;

typedef struct rohanNetwork
{// everything frequently acessed needs to be on 64-byte boundaries, 128 bytes if possible
	cuDoubleComplex Wt[MAXWEIGHTS]; // 16 x 1024 = 2^4 * 2^10 = 16384
	cuDoubleComplex Signals[MAXNEURONS]; // 16 x 256 = 2^4 * 2^8 = 4196
	cuDoubleComplex Deltas[MAXNEURONS]; // 16 x 256
	cuDoubleComplex Zs[MAXNEURONS]; // 16 x 256 = 4196, x 3 = 12288
	double dINV_S[MAXLAYERS]; // 8 x 16 = 128
	int iDendrtQTY[MAXLAYERS]; // 4 x 16 = 64
	int iDendrtOfst[MAXLAYERS];
	int iNeuronQTY[MAXLAYERS];
	int iNeuronOfst[MAXLAYERS];
	int iWeightQTY[MAXLAYERS];
	int iWeightOfst[MAXLAYERS];
	int iSectorQty; // 4
	int kdiv2; 
	int iLayerQty; // 4
	cuDoubleComplex * gWt;
	cuDoubleComplex * gSignals;
	cuDoubleComplex * gDeltas;
	//int iWeightMode; // 4
	int bContActivation /*! Use Continuous activation function (or not). */;
	double dK_DIV_TWO_PI; // 8
	double two_pi_div_sect_qty; // 8
	cuDoubleComplex *cdcSectorBdry; // 4
	cuDoubleComplex *cdcAltSectorBdry;
	cuDoubleComplex *gpuSectorBdry;
	rohanLayer *rLayer;
	rohanLayer *gpuLayer;
	char* sWeightSet;
	FILE* fileInput; // 4
	void* hostNetPtr; // 4 
} rohanNetwork;

struct rohanLearningSet /*! Learning sets contain a short preamble of parameters followed by delimited data */
{
	/// scalar value members
	long iEvalMode /*! Defaults to discrete outputs but a value of 0 denotes Continuous outputs. */;
	long lSampleQty /*! # samples to be read from the learning set file; defaults to # samples found*/;
	long iValuesPerLine /*! Numerical values per line-sample. Inputs + outputs should equal values per line in most cases but some inputs may be intentionally excluded from learning. */;
	long iInputQty /*! # inputs are per line; defaults to the number of values per line, minus the number of outputs, read from left to right. See bRInJMode. */;
	long iOutputQty /*! # outputs are per line; defaults to 1. Read from left to right among the rightmost columns. */;
	//rohanSample *rSample /*! Array of rohanSample structures. */; //removed 10/24/11
	FILE *fileInput /*! The filehandle used for reading the learning set from the given file-like object. */; 
	char *sLearnSet /*! Filename of learning set. */;
	long bContInputs /*! Inputs have decimals. This may go away in favor of a variety of input types.*/;
	long iContOutputs /*! Outputs have decimals. */;
	long lSampleIdxReq /*! index of sample currently or last under consideration. */;
	void* hostLearnPtr;
	/// host space arrays
	double *dXInputs /*! 1BA, learnable real tuples on host. */;
	double *dDOutputs /*! 1BA, desired real ouputs only on host. */;
	double *dYEval /*! 1BA generated scalar outputs on host. Activated. */;
	double *dAltYEval  /*! comparison values */;
	double *dSqrErr /* 2D array for intermediate RMSE totals */ ;
	cuDoubleComplex *cdcXInputs /*! 1D complex input tuple on host. */;
	cuDoubleComplex *cdcDOutputs/*! desired cx output(s) on host. */;
	cuDoubleComplex *cdcYEval /*! 1BA, generated cx outputs on host. Unactivated. */;
	cuDoubleComplex *cdcAltYEval  /*! comparison values */;
	/// dev space arrays added 9/20/2011 to accommodate cDevTeam
	cuDoubleComplex *gpuXInputs   /*! 2D complex input tuples in GPU. */ ;
	cuDoubleComplex *gpuDOutputs  /*! 2D desired complex outputs in GPU. */ ;
	cuDoubleComplex *gpuYEval /*! 1BA, final yielded cx outputs. Never discrete. */;
	cuDoubleComplex *gpuAltYEval /*! 1BA, final cx outputs generated by alt method. Never discrete. */;
	double *gpudXInputs /*! 1BA, learnable real tuples on GPU. */;
	double *gpudDOutputs /*! 1BA, desired real ouputs on GPU. */;
	double *gpudYEval /*! generated scalar outputs in GPU. Activated. */;
	double *gpudAltYEval /*! generated scalar outputs in GPU by alternate method. Activated. */;
	double *gpudSqrErr /* 2Darray for intermediate RMSE totals */ ;
	/// array measurements
	long IQTY;
	long OQTY;
	long INSIZED;
	long OUTSIZED;
	long INSIZECX;
	long OUTSIZECX;
};

typedef struct rohanContext
{
	long gDebugLvl, iEvalMode, iWarnings, iErrors;
	long lMemStructAlloc /* tracks allocated memory structures */ ;
	long bCUDAavailable /*! CUDA-compatible hardware available for use with CUBLAS libraries. */;
	long bConfigFileUsed /*! Session parameters taken from a config file. XX */;
	//long bContActivation /*! Use Continuous activation function (or not). */;
	long bCLargsUsed /*! Command line arguments taken. XX */;
	long bLearnSetSpecUsed /*! Nonstandard learning set guidelines. XX */;
	long bConsoleUsed /*! Session is being directed via console input. XX */;
	long bRInJMode /*! Reverse Input Justification: When active, will read inputs from the same columns but in right to left order to maintain backwards compatibility with some older simulators. */;
	long bRMSEon /*! diables RMSE tracking for classification problems. XX */;
	double dMAX /*! Maximum allowable error in sample output reproduction without flagging for backprop learning. */;
	double dHostRMSE /*! The evaluated Root Mean Square Error over the working sample subset, equivalent to the standard deviation or sigma. */;
	double dDevRMSE /*! The evaluated Root Mean Square Error over the working sample subset, equivalent to the standard deviation or sigma. */;
	double dTargetRMSE /*! Acceptable RMSE value for stopping learninig when achieved. */;
	long iEpochLength /*! iterations per epoch; learning will pause to check once per epoch for further input */;
	long lSampleQtyReq /*! Size of requested working subset of samples, counted from the top. */;
	long lSamplesTrainable /*! Number of samples that exceed dMAX criterion. */;
	long iSaveInputs /* include inputs when saving evaluations */;
	long iSaveOutputs /* include desired outputs when saving evaluations */;
	long iSaveSampleIndex /* includes sample serials when saving evaluations */;
	long iEvalBlocks /*! multithread param for evaluation */;
	long iEvalThreads /*! multithread param for evaluation */;
	long iBpropBlocks /*! multithread param for backward propagation */;
	long iBpropThreads /*! multithread param for backward propagtion */;
	long iMasterCalcHw /*! master calc hardware -1=CPU, 0-1=CUDA Device */;
	double dMasterCalcVer /*! master calc hardware Compute Capability */;
	struct rohanNetwork * rNet /*! Active network currently in use for session. */;
	struct rohanLearningSet * rLearn /*! Active learning set currently in use for session. */;
	struct rohanNetwork * devNet /*! dev space network currently in use for session. */;
	struct rohanLearningSet * devLearn /*! dev space learning set currently in use for session. */;
	struct rohanContext * devSes /*! dev space learning set currently in use for session. */;
	class cDeviceTeam * ctDraftTeam /*! The calculating "engine" currently in use. */;
	FILE *debugHandle /*! handle used for writing large volumes of diagnostic information */;
	std::ofstream * ofsRLog /*! handle used for writing terse updates to RohanLog.txt. */;
	char sLogPath[256] /*! String for path to directory where RohanLog.txt is to be written. */;
	char sConfigFileOpt[256] /*! String to parse for config file options. XX */;
	char sCLargsOpt[256] /*! String to parse for command line arguments. XX  */;
	char sLearnSetSpecOpt[256] /*! String to parse for special learning set guidelines. XX */;
	char sConsoleOpt[256] /*! String to parse for console options. XX */;
	char sSesName[256] /*! Name of .roh file for session. XX */;
	int deviceCount /*! number of CUDA devices attached to host */ ;
	cudaDeviceProp deviceProp /*! capabilities of the CUDA device currently in use. */ ;
} rohanContext;

#define mCheckMallocWorked(X) if (X == NULL) { printf("%s: malloc fail for x in line %d\n", __FILE__, __LINE__); return 0; } 
#define mCuMsg(X, Y) if (gDebugLvl || X!=CUBLAS_STATUS_SUCCESS ) cuMessage(X, Y, __FILE__, __LINE__, __FUNCTION__);
#define mIDfunc if (gTrace) printf("FUNCTION: %s\n", __FUNCTION__);
#define mDebug(X, Y) if (Y) printf("%s line %d: ",__FILE__, __LINE__); if ( (gDebugLvl & X) == X) 
#define mExitKeystroke _getch();
#define mCheckCudaWorked {cudaError_t cet=cudaGetLastError(); while (cet!=cudaSuccess) { printf(" C U D A : %s %s on or before line %d\n", __FILE__, cudaGetErrorString(cet), __LINE__); cet=cudaGetLastError(); } }
#define mSafeFree(X) if ( X != NULL ) free(X); X = NULL;
#define mNANOIdev(X, Y) if ( X != X) cuPrintf("%s %f %d\n", __FUNCTION__, X, Y);
#define mNANOIhost(X, Y) if ( X != X) printf("%s %f %d\n", __FUNCTION__, X, Y);

#define dbprintf(X) fprintf(rSes.debugHandle, X)
#define conPrintf if(rSes.bConsoleUsed)printf
#define errPrintf(X,Y) fprintf(stderr, X, Y)

/*! end redundant-include protection */
#endif