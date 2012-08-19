#include "stdafx.h"

#include <boost/timer/timer.hpp>
using namespace boost::timer;

#if defined (_INTEGRAL_MAX_BITS) &&  _INTEGRAL_MAX_BITS >= 64
typedef signed __int64 int64;
typedef unsigned __int64 uint64;
#else
#error __int64 type not supported
#endif

extern int gDebugLvl, gDevDebug, gTrace;
extern int bCUDAavailable;

//////////////// class cTeam begins ////////////////

cTeam::cTeam( struct rohanContext& rSes)//, struct rohanNetwork& rNet, rohanLearningSet& rLearn)
{mIDfunc
	rSes.Team=this;
	bHitched=FALSE;
	bTaut=FALSE;
	// end ctor
}

int cTeam::SetContext( rohanContext& rC)
{/// enables pointer access to internal objects
	rSes = &rC;
	rLearn = rC.rLearn;
	rNet = rC.rNet;
	Barge=rC.Barge;
	Drover=rC.Drover;
	Ramp=rC.Ramp;
	//Team=rC.Team;

	return 0;
}

void cTeam::ShowMe()
{mIDfunc
	//ShowMeSes(* rSes, false);
	printf("I'm a mighty stallion!\n");
}

char cTeam::GetHitched()
{	
	return bHitched;
}

char cTeam::GetTaut()
{	
	return bTaut;
}


int cTeam::LetEvalSet( rohanContext& rSes, char chMethod)
{mIDfunc/// Submits a subset of the samples available forevaluation, returns samples submitted.
	// choose host submission loop or device
	// currently only called from console menu, and is considered "boarded up" or not actively used - 7/16/12
	if(chMethod=='H' || chMethod=='h'){
		for (int i=0; i< rSes.lSampleQtyReq; ++i){
			cuEvalSingleSampleBeta(rSes, i, *rSes.rNet, *rSes.rLearn, 0, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval);
		}
	}
	else {// D for GPU device XX
		if(!bTaut){
			LetTaut(rSes);
			knlFFeRmseOpt( rSes, 1, 'R', rSes.iEvalBlocks, rSes.iEvalThreads);
			LetSlack(rSes);
		} else
		knlFFeRmseOpt( rSes, 1, 'R', rSes.iEvalBlocks, rSes.iEvalThreads);
	}
	char sLog[255];
	sprintf(sLog,"%s(rSes, chMethod %c ) returns %d", __FUNCTION__, chMethod, rSes.lSampleQtyReq);
	Barge->RLog(rSes, 0, sLog);

	return rSes.lSampleQtyReq;
}

int cTeam::LetTrainNNThresh( rohanContext& rSes, int o, char chMethod, double dTargetRMSE, int iEpochLength, int iEpochQty, char cModel)
{mIDfunc/// Submits samples available for backprop learning.
/// o indictaes which output(s)
// chMethod controls option used:
// Option S - single sample correction only XX unimplemented?
// Option E - keep existing weights, count trainable samples only
// Option R - perform corrections for all trainable samples
	// cModel indicates which calculation model is desired to be used for the duration of the hitching.
	// S - serial, no acceleration
	// 3 - Hecht-Neilsen trilayer with indefinitely width
	// B - fixed-size "block" implementation meant to accomodate shared memory
	// P - polylayer original implementation, currently limited to 3 layers + 0

	int lReturn, count=1, ECount=1;
	double RMSE; //=dTargetRMSE;
	char sLog[255];
	bool bInternalTaut;

	sprintf(sLog, "Learn input %d, method %c, target %f, epoch %d, epqty %d, cModel %c: BEGIN", o, chMethod, dTargetRMSE, iEpochLength, iEpochQty, cModel);
	Barge->RLog(rSes, 0, sLog);

	if(cModel=='B'){ 
		bInternalTaut=false;
		if(!bTaut){
			LetTaut(rSes);
			bInternalTaut=true;
		}
	}

	RMSE=GetRmseNN(rSes, o, 'R', 'B'); // check RMSE anew
	
	if(cModel=='B') lReturn=knlBackProp( rSes, o, 'E', rSes.iBpropBlocks, rSes.iBpropThreads, cEngagedModel); // check for trainable samples
	if(cModel=='S') lReturn=TrainNNThresh( rSes, false);

	if(chMethod=='R') {
		while( dTargetRMSE<RMSE && lReturn && ECount<=iEpochQty){ // target not met, trainable samples left, iterations yet to go
		
			if(cModel=='B') lReturn=knlBackProp( rSes, o, chMethod, rSes.iBpropBlocks, rSes.iBpropThreads, cEngagedModel); // do more training
			if(cModel=='S') lReturn=TrainNNThresh( rSes, true); // do more training

			RMSE=GetRmseNN(rSes, o, 'R', cModel); // check RMSE anew
			sprintf(sLog, "Epoch %d, iteration %d returns RMSE %f", ECount, count, RMSE); Barge->RLog(rSes, 0, sLog);
			if(++count>iEpochLength){ // increment counter
				if(iEpochLength>1){
					sprintf(sLog, "EPOCH %d COMPLETE: rmse %f", ECount, RMSE); Barge->RLog(rSes, ADMINF, sLog);
					LetSlack(rSes); Ramp->SaveNNWeights(rSes, cModel); LetTaut(rSes);
				}
				++ECount; count=1;
			}
		}
	}

	if(cModel=='B') // || cModel=='D' || cModel=='d'){
		if(bInternalTaut)
			LetSlack(rSes);	

	if(dTargetRMSE>=RMSE) Barge->RLog(rSes, 0, "Target RMSE achieved");
	if(lReturn==0) Barge->RLog(rSes, 0, "No trainable samples");
	if(iEpochLength>1 && ECount>iEpochQty) Barge->RLog(rSes, 0, "CHILIAD COMPLETE");
	int iTTL=(count-1)+(ECount-1)*iEpochLength;
	if(chMethod=='R') sprintf(sLog, "Iterations=%d", iTTL);
	Barge->RLog(rSes, USERF, sLog);
	sprintf(sLog, "Learn input %d, method %c, target %f, epoch %d, epqty %d, cModel %c: RMSE= %f END", o, chMethod, dTargetRMSE, iEpochLength, iEpochQty, cModel, RMSE);
	Barge->RLog(rSes, 0, sLog);
	Barge->RLog(rSes, 0, __FUNCTION__);
	return lReturn;
}

int cTeam::LetTrainNNThreshGGG( rohanContext& rSes, int o, char chMethod, double dTargetRMSE, int iEpochLength, int iEpochQty, char cModel)
{mIDfunc/// Submits samples available for backprop learning.
/// o indictaes which output(s)
// chMethod controls option used:
// Option S - single sample correction only XX unimplemented?
// Option E - keep existing weights, count trainable samples only
// Option R - perform corrections for all trainable samples
	// cModel indicates which calculation model is desired to be used for the duration of the hitching.
	// S - serial, no acceleration
	// 3 - Hecht-Neilsen trilayer with indefinitely width
	// B - fixed-size "block" implementation meant to accomodate shared memory
	// P - polylayer original implementation, currently limited to 3 layers + 0

	int lReturn, count=1, ECount=1;
	double RMSE; //=dTargetRMSE;
	char sLog[255];

	sprintf(sLog, "Learn input %d, method %c, target %f, epoch %d, epqty %d, cModel %c: BEGIN", o, chMethod, dTargetRMSE, iEpochLength, iEpochQty, cModel);
	Barge->RLog(rSes, 0, sLog);

	if(cModel=='B'){ // || cModel=='D' || cModel=='d'){
		bool bInternalTaut=false;
		if(!bTaut){
			LetTaut(rSes);
			bInternalTaut=true;
		}
		RMSE=GetRmseNN(rSes, o, 'R', 'B'); // check RMSE anew
		lReturn=knlBackProp( rSes, o, 'E', rSes.iBpropBlocks, rSes.iBpropThreads, cEngagedModel); // check for trainable samples
		if(chMethod=='R') {
			while( dTargetRMSE<RMSE && lReturn && ECount<=iEpochQty){ // target not met, trainable samples left, iterations yet to go
				lReturn=knlBackProp( rSes, o, chMethod, rSes.iBpropBlocks, rSes.iBpropThreads, cEngagedModel); // do more training
				RMSE=GetRmseNN(rSes, o, 'R', 'B'); // check RMSE anew
				sprintf(sLog, "Epoch %d, iteration %d returns RMSE %f", ECount, count, RMSE); Barge->RLog(rSes, 0, sLog);
				if(++count>iEpochLength){ // increment counter
					sprintf(sLog, "EPOCH %d COMPLETE: %f", ECount, RMSE);
					Barge->RLog(rSes, ADMINF, sLog);
					++ECount; count=1;
				}
			}
			if(bInternalTaut)
				LetSlack(rSes);
			//UpdateRmseRecord(rSes, RMSE, RMSE, 'B'); // update device-achieved RMSE
		}
		else{ // for method E
			if(bInternalTaut)
				LetSlack(rSes);
		}	
	}
	if(cModel=='S' || cModel=='H' || cModel=='h'){
		//host training goes here
		RMSE=GetRmseNN(rSes, o, 'R', 'S'); // check RMSE anew
		lReturn=TrainNNThresh( rSes, false);
		if(chMethod=='R') {
			while(dTargetRMSE<RMSE && lReturn && ECount<=iEpochQty ){ // target not met, trainable samples left
				lReturn=TrainNNThresh( rSes, true); // do more training
				RMSE=GetRmseNN(rSes, o, 'R', 'S'); // check RMSE anew
				sprintf(sLog, "Epoch %d, iteration %d returns RMSE %f", ECount, count, RMSE); Barge->RLog(rSes, 0, sLog);
				if(++count>iEpochLength){ // increment counter
					sprintf(sLog, "EPOCH %d COMPLETE: %f", ECount, RMSE);
					Barge->RLog(rSes, ADMINF, sLog);
					++ECount; count=1;
				}
			}
			//UpdateRmseRecord(rSes, RMSE, RMSE, 'S'); // update device-achieved RMSE
		}				
	}

	if(dTargetRMSE>=RMSE) Barge->RLog(rSes, 0, "Target RMSE achieved");
	if(lReturn==0) Barge->RLog(rSes, 0, "No trainable samples");
	if(ECount>iEpochQty) Barge->RLog(rSes, 0, "CHILIAD COMPLETE");
	int iTTL=(count-1)+(ECount-1)*iEpochLength;
	sprintf(sLog, "Iterations=%d", iTTL);
	Barge->RLog(rSes, USERF, sLog);
	sprintf(sLog, "Learn input %d, method %c, target %f, epoch %d, epqty %d, cModel %c: RMSE= %f END", o, chMethod, dTargetRMSE, iEpochLength, iEpochQty, cModel, RMSE);
	Barge->RLog(rSes, 0, sLog);
	Barge->RLog(rSes, 0, __FUNCTION__);
	return lReturn;
}

double cTeam::GetRmseNN(struct rohanContext& rSes, int o, char Option, char cModel)
{mIDfunc/*! checks sampled outputs vs evaluated outputs and calculates root mean squared error. */
	// o controls which output is used (zero for all)
	// Option will determine if existing data is used (E, not fully implemented) or refreshed (R) XX 6/23/12
	// cModel controls which mechanism (cpu H, GPU D) will do the calulation
		// cModel indicates which calculation model is desired to be used for the duration of the hitching.
	// S - serial, no acceleration
	// 3 - Hecht-Neilsen trilayer with indefinite width
	// B - fixed-size "block" implementation meant to accomodate shared memory
	// P - polylayer original implementation, currently limited to 3 layers + 0

	double dAnswer=777.777;
	char sLog[255];
	sprintf(sLog, "\t%s(rSes, o=%d, opt=%c, model=%c) called", __FUNCTION__, o, Option, cModel); 
	//if (cModel=='S')fprintf(rSes.hostBucket,"%s\n", sLog); else fprintf(rSes.deviceBucket,"%s\n", sLog);
	Barge->RLog(rSes, 0, sLog); 


	if (cModel=='B'){ 
		if(bTaut) // already taut? no problem!
			dAnswer=knlFFeRmseOpt( rSes, o, Option, rSes.iEvalBlocks, rSes.iEvalThreads );
		else{ //must make taut then return to previous state
			LetTaut(rSes);
			dAnswer=knlFFeRmseOpt( rSes, o, Option, rSes.iEvalBlocks, rSes.iEvalThreads );
			LetSlack(rSes);
		}
	} 
	if (cModel=='S'){ 
		if (Option=='R')
			cuEvalNNLearnSet(rSes, o); // refresh YEvals
		dAnswer=CalcRmseSerial(rSes, o); 
	} 
	
	if(Option=='R'){
		UpdateRmseRecord(rSes, dAnswer, dAnswer, cModel); // update achieved RMSE
		sprintf(sLog, "GetRmseNN(rSes, o=%d, Option %c, %c) updates %f RMSE", o, Option, cModel, dAnswer); 
	}
	else // no feed-forward evaluation, so no RMSE update 7/16/12
		sprintf(sLog, "GetRmseNN(rSes, o=%d, Option %c, %c) returns %f without updating RMSE", o, Option, cModel, dAnswer); 
	Barge->RLog(rSes, 0, sLog); 

	//if (cModel=='S')fprintf(rSes.hostBucket,"%s\n", sLog); else fprintf(rSes.deviceBucket,"%s\n", sLog);
		
	return dAnswer;
}

int cTeam::LetHitch(struct rohanContext& rSes, char cModel)   
{mIDfunc/*! \callgraph \callergraph copy data to device memory space and attach structures to Team */
	// cModel indicates which calculation model is desired to be used for the duration of the hitching.
	// S - serial, no acceleration
	// 3 - Hecht-Neilsen trilayer with indefinitely width
	// B - fixed-size "block" implementation meant to accomodate shared memory
	// G - fixed-size "block" implementation without shared memory
	// P - polylayer original implementation, currently limited to 3 layers + 0
	char sLog[255];

	if (bHitched){
		Barge->RLog(rSes, WARNINGF, "cTeam already hitched.");
		return FALSE;
	}
	if (cModel!='S' && rSes.dMasterCalcVer <2.0){
		Barge->RLog(rSes, WARNINGF, "Insufficient CUDA Compute Capability value.");
		return FALSE;
	}
	else{
		rSes.cEngagedModel=cEngagedModel=cModel;
		sprintf(sLog, "Engaging model %c, %s network...", cEngagedModel, rSes.sNetString); Barge->RLog(rSes, 0+DEBUGF, sLog);

		// BEGIN context data structure members (multimodel 4)
 		TransferContext(rSes, 'D');
		
		// BEGIN network architecture structure members (need multimodel ZZ)
		CopyNet( rSes, 'D' );
		
		// BEGIN learning set structure members (need multimodel ZZ)
		CopyLearnSet(rSes, 'D');
		
		/// end members and structures
		Barge->RLog(rSes, 0, "H+T-->");
		bHitched=TRUE; bTaut=TRUE;
		Barge->RLog(rSes, 0, __FUNCTION__);
		return TRUE;
	}
}


int cTeam::TransferContext(struct rohanContext& rSes, char Direction)
{mIDfunc/*! copy rSes and members to dev mem */
	int iReturn=1;
	if(cEngagedModel!='S'){//non-serial models do this
		int SIZE = sizeof(rSes);
		
		if(Direction=='D'||Direction=='d') {
			// publish Context structure to device
				mCheckCudaWorked
			cudaMemcpyToSymbol( "devSes", &rSes, sizeof(rSes) );
				mCheckCudaWorked
			if(crc32buf( (char*)&rSes, SIZE )!=knlCRC32Buf( (char*)rSes.devSes, SIZE )){
				iReturn=0;
				Barge->RLog(rSes, ERRORF, "FAILURE copying context H->D");
			}
		}
		else {
			// retrieve rSes from device
			cudaMemcpyFromSymbol( &rSes, "devSes", sizeof(rSes) );
				mCheckCudaWorked
		}
	}
	char sLog[255];
	sprintf(sLog,"%s(rSes, Direction %c )", __FUNCTION__, Direction);
	Barge->RLog(rSes, 0, sLog);

	return iReturn;
}

int cTeam::CopyNet(struct rohanContext& rSes, char Direction)
{mIDfunc/*! copy NN arch and layer structures, their contents, and member values to dev memory */
// layer structures removed 3/31/12 JAW
	cudaMemcpyKind kind; //cuDoubleComplex * dummy;
	struct rohanNetwork * rnSrc, * rnDest ;
	//struct rohanLayer * rlSrc;
	//int LQTY, LLAST, LSIZE, 
	int SECSIZE;

	//printf("=> Copy Network %c =>\n", Direction);
	rnSrc=(rSes.rNet);
	int SIZE = sizeof(*rSes.rNet);
	//printf("%08lX %7ld %s\n", crc32buf( (char*)rnSrc, SIZE ), SIZE, "rNet");
		
	cudaGetSymbolAddress( (void**)&rnDest, "devNet" ); /*! get ptr into devspace for network structure */
		mCheckCudaWorked
	//LQTY = rnSrc->iLayerQTY ; 
	//	LLAST = LQTY - 1 ;
	//LSIZE = sizeof(rohanLayer) * LQTY ;
	kind=cudaMemcpyHostToDevice;
		
	// mold dev memspace for sector table and copy contents
	SECSIZE = rnSrc->iSectorQty * sizeof(cuDoubleComplex);
	cudaMalloc( (void**)&rnSrc->gpuSectorBdry, SECSIZE );
		mCheckCudaWorked
		rSes.lMemStructAlloc = rSes.lMemStructAlloc || RNETbdry;
	cudaMemcpy( rnSrc->gpuSectorBdry, rnSrc->cdcSectorBdry, SECSIZE , kind );
		mCheckCudaWorked
	if(crc32buf( (char*)rnSrc->cdcSectorBdry, SECSIZE )!=knlCRC32Buf( (char*)rnSrc->gpuSectorBdry, SECSIZE ))
		Barge->RLog(rSes, ERRORF, "FAILURE copying sector table");
	
	TransferNet(rSes, 'D');

	char sLog[255];
	sprintf(sLog,"%s(rSes, Direction %c )", __FUNCTION__, Direction);
	Barge->RLog(rSes, 0, sLog);

	return 0;
}


int cTeam::TransferNet(struct rohanContext& rSes, char Direction)
{mIDfunc/*! copy rNet params to dev mem */
	int SIZE = sizeof(rohanNetwork);
	//printf("=> Transfer Net %c =>\n", Direction);
	if(Direction=='D'||Direction=='d') {
		cudaMemcpyToSymbol( "devNet", rSes.rNet, SIZE ); //, 0, kind );
			mCheckCudaWorked
	}
	else {
		// retrieve net params from device
		cudaMemcpyFromSymbol( rSes.rNet, "devNet", SIZE ); //, 0, kind );
			mCheckCudaWorked
	}
	if(crc32buf( (char*)rSes.rNet, SIZE )!=knlCRC32Buf( (char*)rSes.devNet, SIZE ))
			Barge->RLog(rSes, ERRORF, "FAILURE copying Net params");
	
	char sLog[255];
	sprintf(sLog,"%s(rSes, Direction %c )", __FUNCTION__, Direction);
	Barge->RLog(rSes, 0, sLog);

	return 0;
}


int cTeam::CopyLearnSet(struct rohanContext& rSes, char Direction)
{mIDfunc/*! copies learning set structures, contents, andvmember values to device memory space */
	cudaMemcpyKind kind; //cuDoubleComplex * dummy;
	struct rohanLearningSet * rlSrc;
	int IQTY, OQTY, INSIZED, OUTSIZED, INSIZECX, OUTSIZECX;
	
	//setup dimension values
	IQTY = rSes.rLearn->iInputQty+1 ;
	INSIZED = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(double) ;
	INSIZECX = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(cuDoubleComplex) ;
	OQTY = rSes.rLearn->iOutputQty+1; 
	OUTSIZED = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(double);
	OUTSIZECX = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(cuDoubleComplex);
	
	//printf("=> Copy Learning Set %c =>\n", Direction);
	rlSrc=(rSes.rLearn);
	int SIZE = sizeof(*rSes.rLearn);
	//printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc, SIZE ), SIZE, "rLearn");
	
	if(Direction=='D'||Direction=='d') {
		kind=cudaMemcpyHostToDevice;
		// gpudXInputs
		cudaMalloc( (void**)&rlSrc->gpudXInputs, INSIZED ); /*! mold devspace for scalar input signals */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudXInputs, rlSrc->dXInputs, INSIZED, kind); /*! fill the devspace mold with hostspace values */
			mCheckCudaWorked
		// gpudDOutputs
		cudaMalloc( (void**)&rlSrc->gpudDOutputs, OUTSIZED ); /*! 2D desired scalar outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudDOutputs, rlSrc->dDOutputs, OUTSIZED, kind); 
			mCheckCudaWorked
		// gpudYEval
		cudaMalloc( (void**)&rlSrc->gpudYEval   , OUTSIZED ); /*! 2D yielded scalar outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudYEval   , rlSrc->dYEval   , OUTSIZED, kind); 
			mCheckCudaWorked
		// gpudAltYEval
		cudaMalloc( (void**)&rlSrc->gpudAltYEval, OUTSIZED ); /*! 2D yielded scalar outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudAltYEval, rlSrc->dAltYEval, OUTSIZED, kind); 
			mCheckCudaWorked
		// gpudSqrErr
		cudaMalloc( (void**)&rlSrc->gpudSqrErr, OUTSIZED ); /* array for intermediate RMSE totals, changed from 1024 to OUTSIZED on 1/8/12 */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpudSqrErr, rlSrc->dSqrErr, OUTSIZED , kind );
			mCheckCudaWorked
		// gpuXInputs
		cudaMalloc( (void**)&rlSrc->gpuXInputs, INSIZECX ); /*! mold devspace for complex input signals */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpuXInputs, rlSrc->cdcXInputs, INSIZECX, kind); /*! fill the devspace mold with hostspace values */
			mCheckCudaWorked
		// gpuDOutputs
		cudaMalloc( (void**)&rlSrc->gpuDOutputs, OUTSIZECX ); /*! 2D desired complex outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpuDOutputs, rlSrc->cdcDOutputs, OUTSIZECX, kind); 
			mCheckCudaWorked
		// gpuYEval
		cudaMalloc( (void**)&rlSrc->gpuYEval   , OUTSIZECX ); /*! 2D yielded complex outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpuYEval   , rlSrc->cdcYEval   , OUTSIZECX, kind); 
			mCheckCudaWorked
		// gpuAltYEval
		cudaMalloc( (void**)&rlSrc->gpuAltYEval, OUTSIZECX ); /*! 2D yielded complex outputs in GPU. */
			mCheckCudaWorked
		cudaMemcpy( rlSrc->gpuAltYEval, rlSrc->cdcAltYEval, OUTSIZECX, kind); 
			mCheckCudaWorked

#ifdef _DEBUG		
		if(0){ // checksums to verify faithful transfer
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dXInputs, INSIZED ), INSIZED, "dXInputs");
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudXInputs, INSIZED ), INSIZED, "gpudXInputs");
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dDOutputs, OUTSIZED ), OUTSIZED, "dDOutputs");
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudDOutputs, OUTSIZED ), OUTSIZED, "gpudDOutputs");
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dYEval, OUTSIZED ), OUTSIZED, "dYEval");
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudYEval, OUTSIZED ), OUTSIZED, "gpudYEval");
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dAltYEval, OUTSIZED ), OUTSIZED, "dAltYEval");
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudAltYEval, OUTSIZED ), OUTSIZED, "gpudAltYEval");
			printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc->dSqrErr, OUTSIZED ), OUTSIZED, "dSqrErr"); // made OUTSIZED by JAW 1/8/12
			printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rlSrc->gpudSqrErr, OUTSIZED ), OUTSIZED, "gpudSE102");
		}
#endif
		// store array dimensions
		rSes.rLearn->IQTY=IQTY;
		rSes.rLearn->OQTY=OQTY;
		rSes.rLearn->INSIZED=INSIZED;
		rSes.rLearn->OUTSIZED=OUTSIZED;
		rSes.rLearn->INSIZECX=INSIZECX;
		rSes.rLearn->OUTSIZECX=OUTSIZECX;
		
		// publish Learn struct to device
		rSes.rLearn->lSampleIdxReq=77777;
		
		//printf("%08lX %7ld %s\n", crc32buf( (char*)rlSrc, SIZE ), SIZE, "rLearn");
		//printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rSes.devLearn, SIZE ), SIZE, "devLearn");
		
		//ShowMeLS(rSes, 0);
		
		cudaMemcpyToSymbol( "devLearn", rSes.rLearn, sizeof(*rSes.rLearn) ); //, 0, kind );
			mCheckCudaWorked
		//printf("%08lX %7ld %s\n", knlCRC32Buf( (char*)rSes.devLearn, SIZE ), SIZE, "devLearn");
	}
	else {
		kind=cudaMemcpyDeviceToHost;
	}
	if(crc32buf( (char*)rlSrc, SIZE )!=knlCRC32Buf( (char*)rSes.devLearn, SIZE ))
		Barge->RLog(rSes, ERRORF, "FAILURE copying samples");
	
	char sLog[255];
	sprintf(sLog,"%s(rSes, Direction %c )", __FUNCTION__, Direction);
	Barge->RLog(rSes, 0, sLog);
	
	return 0;
}

int cTeam::LetUnHitch(struct rohanContext& rSes)
{mIDfunc/*! \callgraph \callergraph free device memory structures to Team */
	// cModel indicates which calculation model is desired to be used for the duration of the hitching.
	// S - serial, no acceleration
	// 3 - Hecht-Neilsen trilayer with indefinitely width
	// B - fixed-size "block" implementation meant to accomodate shared memory
	// P - polylayer original implementation, currently limited to 3 layers + 0                                                                                                                                                           

	char sLog[255];
	if(bHitched){ // check for hitched state
		if(bTaut) // check for taut/slack state
			LetSlack(rSes); // if taut, go slack before unhitching
		// BEGIN free nw data structures
		struct rohanNetwork * rnSrc=(rSes.rNet);

		cudaFree( rnSrc->gpuSectorBdry);
			mCheckCudaWorked
		Barge->RLog(rSes, 0, "cdt: Net Arch clear");	

		//BEGIN freeing rdLearn structures
		struct rohanLearningSet * rlSrc = rSes.rLearn;
			cudaFree( rlSrc->gpudXInputs);
				mCheckCudaWorked
			cudaFree( rlSrc->gpudDOutputs);
				mCheckCudaWorked
			cudaFree( rlSrc->gpudYEval);
				mCheckCudaWorked
			cudaFree( rlSrc->gpudAltYEval);
				mCheckCudaWorked
			cudaFree( rlSrc->gpudSqrErr);
				mCheckCudaWorked
			cudaFree( rlSrc->gpuXInputs);
				mCheckCudaWorked
			cudaFree( rlSrc->gpuDOutputs);
				mCheckCudaWorked
			cudaFree( rlSrc->gpuYEval);
				mCheckCudaWorked
			cudaFree( rlSrc->gpuAltYEval);
				mCheckCudaWorked
		Barge->RLog(rSes, 0, "cdt: LearnSet clear");

		/// end members and structures
		Barge->RLog(rSes, 0, "cTeam unhitched.");
		bHitched=FALSE;
		sprintf(sLog, "Model %c network disengaged.", cEngagedModel); Barge->RLog(rSes, 0+DEBUGF, sLog);
		rSes.cEngagedModel=cEngagedModel=' '; // reset to space
		return TRUE;
	}
	else{
		Barge->RLog(rSes, WARNINGF, "cTeam is already unhitched!");
		return FALSE;
	}
}


int cTeam::LetTaut(struct rohanContext& rSes)
{mIDfunc/*! \callgraph \callergraph update dev mem from host for epoch */;
	if(bHitched && bTaut==FALSE){ // check taut state first
		TransferContext(rSes, 'D');
		//TransferLayers(rSes, 'D');
		TransferNet(rSes, 'D');
		TransferOutputs(rSes, 'D');
		Barge->RLog(rSes, 0, "T-->");
		bTaut=TRUE;
	}
	else{
		if(bTaut)
			Barge->RLog(rSes, WARNINGF, "cTeam already taut!");
		if(bHitched==FALSE)
			Barge->RLog(rSes, WARNINGF, "cTeam is not hitched!");
	}
	Barge->RLog(rSes, 0, __FUNCTION__);
	return 0;
}


int cTeam::TransferLayers(struct rohanContext& rSes, char Direction)
{mIDfunc/*! \callgraph \callergraph copy rNet layer data to dev mem */
	cudaMemcpyKind kind; //cuDoubleComplex * dummy;
	struct rohanNetwork * rnSrc, * rnDest ;
	struct rohanLayer * rlSrc;
	int LQTY, LLAST, LSIZE;

	rnSrc=(rSes.rNet);
	int SIZE = sizeof(*rSes.rNet);
	//printf("%08lX %7ld %s\n", crc32buf( (char*)rnSrc, SIZE ), SIZE, "rNet");
		
	cudaGetSymbolAddress( (void**)&rnDest, "devNet" ); /*! get ptr into devspace for network structure */
		mCheckCudaWorked
	LQTY = rnSrc->iLayerQTY ; 
		LLAST = LQTY - 1 ;
	LSIZE = sizeof(rohanLayer) * LQTY ;

	if (Direction=='D' || Direction=='D'){
		kind=cudaMemcpyHostToDevice;
		for (int L=1; L<=LLAST; ++L){
			//printf("----> Copy Layer %d %c ---->\n", L, Direction);
			int DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE;
			//setup dimension values
			DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = weights
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
			rlSrc=&(rSes.rNet->rLayer[L]);
			//gpuWeights
			cudaMemcpy( rlSrc->gpuWeights, rlSrc->Weights, WSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuXInputs
			cudaMemcpy( rlSrc->gpuXInputs, rlSrc->XInputs, DSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuZOutputs
			cudaMemcpy( rlSrc->gpuZOutputs, rlSrc->ZOutputs, NSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuDeltas 
			cudaMemcpy( rlSrc->gpuDeltas, rlSrc->Deltas, NSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//printf("-> layer %d transfered?\n", L);
		}
	}
	else{
		kind=cudaMemcpyDeviceToHost;
		for (int L=1; L<=LLAST; ++L){
			//printf("----> Copy Layer %d %c ---->\n", L, Direction);
			int DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE;
			//setup dimension values
			DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = weights
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
			rlSrc=&(rSes.rNet->rLayer[L]);
			//gpuWeights
			cudaMemcpy( rlSrc->Weights, rlSrc->gpuWeights, WSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuXInputs
			cudaMemcpy( rlSrc->XInputs, rlSrc->gpuXInputs, DSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuZOutputs
			cudaMemcpy( rlSrc->ZOutputs, rlSrc->gpuZOutputs, NSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//gpuDeltas 
			cudaMemcpy( rlSrc->Deltas, rlSrc->gpuDeltas, NSIZE, kind); /*! fill the devspace mold with hostspace values */
				mCheckCudaWorked
			//printf("-> layer %d transfered?\n", L);
		}
	}
	char sLog[255];
	sprintf(sLog,"%s(rSes, Direction %c )", __FUNCTION__, Direction);
	Barge->RLog(rSes, 0, sLog);
	Barge->RLog(rSes, 0, __FUNCTION__);
	return 0;
}


int cTeam::LetSlack(struct rohanContext& rSes)
{mIDfunc/*! \callgraph \callergraph update dev mem from host for epoch */;
	if(bHitched && bTaut){ // check taut state first
		//TransferContext(rSes, 'H');
		//TransferLayers(rSes, 'H');
		TransferNet(rSes, 'H');
		TransferOutputs(rSes, 'H');
		Barge->RLog(rSes, 0, "<--S");
		bTaut=FALSE;
	}
	else{
		if(bTaut==FALSE)
			Barge->RLog(rSes, WARNINGF, "cTeam already slack!");
		if(bHitched==FALSE)
			Barge->RLog(rSes, WARNINGF, "cTeam is not hitched!");
	}
	Barge->RLog(rSes, 0, __FUNCTION__);
	return 0;
}


int cTeam::TransferOutputs(struct rohanContext& rSes, char Direction)
{mIDfunc/*! transfers contents of yielded output data strctures between memory spaces, usually dev to host */
	cudaMemcpyKind kind; //cuDoubleComplex * dummy;
	struct rohanLearningSet * rlSrc;
	int IQTY, OQTY, INSIZED, OUTSIZED, INSIZECX, OUTSIZECX;
	
	//setup dimension values
	IQTY = rSes.rLearn->iInputQty+1 ;
	INSIZED   = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(double) ;
	INSIZECX  = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(cuDoubleComplex) ;
	OQTY = rSes.rLearn->iOutputQty+1; 
	OUTSIZED  = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(double);
	OUTSIZECX = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(cuDoubleComplex);
	
	rlSrc=(rSes.rLearn);

	if(Direction=='D'||Direction=='d') {
		kind=cudaMemcpyHostToDevice; // CPU plain evals go to GPU alt evals, and vice versa
		// gpudYEval
		cudaMemcpy( rlSrc->gpudYEval   , rlSrc->dAltYEval   , OUTSIZED, kind); 
			mCheckCudaWorked
		// gpudAltYEval 
		cudaMemcpy( rlSrc->gpudAltYEval, rlSrc->dYEval      , OUTSIZED, kind); 
			mCheckCudaWorked
		// gpuYEval 
		cudaMemcpy( rlSrc->gpuYEval   , rlSrc->cdcAltYEval, OUTSIZECX, kind); 
			mCheckCudaWorked
		// gpuAltYEval 
		cudaMemcpy( rlSrc->gpuAltYEval, rlSrc->cdcYEval   , OUTSIZECX, kind); 
			mCheckCudaWorked
	}
	else {
		kind=cudaMemcpyDeviceToHost; // GPU plain evals go to CPU alt evals, and vice versa
		cudaMemcpy( rlSrc->dAltYEval   , rlSrc->gpudYEval   , OUTSIZED, kind); 
			mCheckCudaWorked
		cudaMemcpy( rlSrc->dYEval      , rlSrc->gpudAltYEval, OUTSIZED, kind); 
			mCheckCudaWorked
		cudaMemcpy( rlSrc->cdcAltYEval, rlSrc->gpuYEval   , OUTSIZECX, kind); 
			mCheckCudaWorked
		cudaMemcpy( rlSrc->cdcYEval   , rlSrc->gpuAltYEval, OUTSIZECX, kind); 
			mCheckCudaWorked
	}
	char sLog[255];
	sprintf(sLog,"%s(rSes, Direction %c )", __FUNCTION__, Direction);
	Barge->RLog(rSes, 0, sLog);
	return 0;
}

int cTeam::TeamTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTestType, int iTrials, int iBlocks, int iThreads, int iSampleQtyReq, int o)
{mIDfunc /// runs tests on both host and GPU
	double dDifferent=0.0, dCtrlDiff, dExpDiff; float fExpTime=0.0;
	int iDeviceTrainable, iHostTrainable, iMargin=0, oldThreads=rSes.iBpropThreads, 
		iOldSampleQtyReq=rSes.lSampleQtyReq, iDifferent=0, iPassed=true; 
	char sTestType[80], sLog0[255], sLog1[255], sLog2[255], sLog3[255];
	boost::timer::cpu_timer tCtrl, tExp;
	boost::timer::cpu_times elapCtrl, elapExp;
	tCtrl.start();	tCtrl.stop();	tExp.start();	tExp.stop();
	if(iTestType==EVALUATEF) sprintf(sTestType, "%s", "EVALUATE");
	if(iTestType==CLASSIFYF) sprintf(sTestType, "%s", "CLASSIFY");
	if(iTestType==BACKPROPF) sprintf(sTestType, "%s", "BACKPROP");
	
	if(iSampleQtyReq>0 && iSampleQtyReq < rSes.lSampleQty)
		rSes.lSampleQtyReq=iSampleQtyReq; // change sample set size just for this function
	else 
		rSes.lSampleQtyReq=rSes.lSampleQty;
	if(iThreads>0)
		if(iTestType==BACKPROPF)
			rSes.iBpropThreads=iThreads;

	// perform a warm-up to eliminate the always-longer first one, return true number of samples, prepare timer to resume @ 0.0;
	GetRmseNN(rSes, rSes.iOutputFocus, 'R', cControlModel);
	GetRmseNN(rSes, rSes.iOutputFocus, 'R', cExpModel);// evaluation is now included in classification tests
			 
	sprintf(sLog0, "BEGIN %s TEST: %d TRIALS %d THREADS %d SAMPLES %c/%c", sTestType, iTrials, iThreads, rSes.lSampleQtyReq, cControlModel, cExpModel);
	Barge->RLog( rSes, 0, sLog0);
	if(iTestType==BACKPROPF) LetTaut(rSes);
	for(int i=1; i<=iTrials; ++i){
		if(iTestType==EVALUATEF) {
			tExp.resume();
			GetRmseNN(rSes, rSes.iOutputFocus, 'R', cExpModel); // run on device with refreshed values
			fExpTime+=gElapsedTime; // device times are roughly equal to serial overhead; kernel launchers record time in global variable for later pickup
			tExp.stop();
			tCtrl.resume();
			GetRmseNN(rSes, rSes.iOutputFocus, 'R', cControlModel); // update dHostRMSE
			tCtrl.stop();
			dDifferent += dExpDiff = GetLastRmse(rSes, cExpModel);
			dDifferent -= dCtrlDiff = GetLastRmse(rSes, cControlModel);
			sprintf(sLog0, "%s TRIAL %d: Exp RMSE %f, Ctrl RMSE %f", sTestType, i, dExpDiff, dCtrlDiff);
			Barge->RLog(rSes, 0, sLog0);
		}
		if(iTestType==CLASSIFYF) {
			LetTaut(rSes);
			tExp.resume();
			gKernelTimeTally=0.0; //reset global kernel time tally
			iMargin += iDeviceTrainable=  LetTrainNNThresh(rSes, 0, 'E', rSes.dTargetRMSE, 1, 1, cExpModel);
			fExpTime += gKernelTimeTally; // device times are roughly equal to serial overhead; kernel launchers record time in global variable for later pickup
			tExp.stop();
			LetSlack(rSes);
			tCtrl.resume();
			cuEvalNNLearnSet(rSes, o); // evaluation is now included separately in host classification tests
			iMargin -= iHostTrainable = TrainNNThresh(rSes, false);
			tCtrl.stop();
			sprintf(sLog0, "%s TRIAL %d: Exp trainable %d, Ctrl trainable %d", sTestType, i, iDeviceTrainable, iHostTrainable);
			Barge->RLog(rSes, 0, sLog0);
		}
		if(iTestType==BACKPROPF) {
			gKernelTimeTally=0.0; //reset global kernel time tally
			// evaluation is now integrated in device classification tests
				iMargin += iDeviceTrainable = LetTrainNNThresh( rSes, rSes.iOutputFocus, 'R', rSes.dTargetRMSE, 1, 1, cExpModel); // backprop all samples, output usual, revise wts, target RMSE, epoch=single iteration YY change E to R
				dDifferent += dExpDiff = GetLastRmse(rSes, cExpModel); //GetRmseNN(rSes, rSes.iOutputFocus, 'R', cExpModel);
			fExpTime+=gKernelTimeTally; // device times are roughly equal to serial overhead; kernel launchers record time in global variable for later pickup
			tCtrl.resume();
				cuEvalNNLearnSet(rSes, o); // evaluation is now included separately in host classification tests
				//iMargin -= iHostTrainable = TrainNNThresh( rSes, true);
				iMargin -= iHostTrainable = TrainNNThresh( rSes, true);
				dDifferent -= dCtrlDiff = GetRmseNN(rSes, rSes.iOutputFocus, 'R', cControlModel);
			tCtrl.stop();
			sprintf(sLog0, "%s TRIAL %d: %c Exp trainable %d, %c Ctrl trainable %d", sTestType, i, cExpModel, iDeviceTrainable, cControlModel, iHostTrainable);
			Barge->RLog(rSes, 0, sLog0);
			sprintf(sLog0, "%s TRIAL %d: %c Exp RMSE %f, %c Ctrl RMSE %f", sTestType, i, cExpModel, dExpDiff, cControlModel, dCtrlDiff);
			Barge->RLog(rSes, 0, sLog0);
		}
	}
	if(iTestType==BACKPROPF) LetSlack(rSes);
	if(iTestType==EVALUATEF) iDifferent += OutputValidate(rSes);
			
	elapCtrl=tCtrl.elapsed();
	elapExp =tExp.elapsed();
	int64 denominator = iTrials*100000; // convert to tenths of milliseconds
	int64 quotientCtrl = elapCtrl.wall / denominator;
	int64 quotientExp  = elapExp.wall  / denominator;
	double dAvgTimeHost = (double)quotientCtrl; 
	double dAvgTimeDev = (double)quotientExp; 
	double su = ((dAvgTimeHost/10)/(fExpTime/iTrials));
	int cWt=crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS );int eWt=knlCRC32Buf( (char*)rSes.devNet->Wt, MAXWEIGHTS * 16);
	sprintf(sLog0, "final Ctrl wt state: %08lX %s final Expr wt state: %08lX", cWt, (cWt==eWt ? "==" : "!="), eWt ); // weight check
	sprintf(sLog1, "Control    mean performance over %d runs: %.1f ms", iTrials, dAvgTimeHost/10);
	sprintf(sLog2, "Experiment mean performance over %d runs: %.1f ms", iTrials, fExpTime/iTrials);
	if(iTestType==EVALUATEF) {
		if(iDifferent || abs(dDifferent)>.001 ) iPassed=false;
		sprintf(sLog3, "%s\t%s\t%c s%d\td/%d,%f/\tsu %.1fx", ( iPassed ? "Evaluate Pass" :"EVALUATE FAIL" ), rSes.sNetString, rSes.cEngagedModel, iSampleQtyReq, iDifferent, dDifferent, su );
	}
	if(iTestType==CLASSIFYF) {
		if(iMargin) iPassed=false;
		sprintf(sLog3, "%s\t%s\t%c\td/%d/\tsu %.1fx", ( iPassed ? "Classify Pass" : "CLASSIFY FAIL"), rSes.sNetString, rSes.cEngagedModel, iMargin, su );
	}
	if(iTestType==BACKPROPF) {
		if(iMargin || abs(dDifferent)>.01 ) iPassed=false;
		sprintf(sLog3, "%s\t%s\t%c %d %d\td/%d,%f/\tsu %.1fx", ( iPassed ? "Backprop Pass" : "BACKPROP FAIL"), rSes.sNetString, rSes.cEngagedModel, iBlocks, iThreads, iMargin, dDifferent, su );
	}
	Barge->RLog(rSes, 0, sLog0);
	Barge->RLog(rSes, 0, sLog1);
	Barge->RLog(rSes, 0, sLog2);
	Barge->RLog(rSes, 0+ADMINF, sLog3);
	
	rSes.lSampleQtyReq=iOldSampleQtyReq; // restore original sample set size just before exit
	rSes.iBpropThreads=oldThreads; // restore original threads and blocks
	Barge->RLog(rSes, 0, __FUNCTION__);
	
	return (iPassed);
}


double cTeam::CUDAverify(struct rohanContext& rSes)
{mIDfunc/// Checks for prsence of CUDA-enabled hardware
	double compCap=1.0, check; rSes.dMasterCalcVer=0.0;
	
	cudaGetDeviceCount(&(rSes.deviceCount) ); 
	if(rSes.deviceCount){
		for (int device = 0; device < rSes.deviceCount; ++device) { 
			//CUDAShowProperties(rSes, device, rSes.debugHandle);
			cudaGetDeviceProperties(&rSes.deviceProp, device); 
			check=rSes.deviceProp.major + rSes.deviceProp.minor * 0.1;
			if (check>compCap){
				compCap=check;
				if(check>rSes.dMasterCalcVer){
					rSes.dMasterCalcVer=check;
					rSes.iMasterCalcHw=device;
				}
			}
		}
		Barge->RLog(rSes, 0, __FUNCTION__);
		return compCap;
	}
	else 
		Barge->RLog(rSes, 0, __FUNCTION__);
		return 0; // not available
}


// following section borrowed from CUDA SDK, (C) Nvidia
// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
	CUresult error_result = cuDeviceGetAttribute( attribute, device_attribute, device );
    if (error_result != CUDA_SUCCESS) {
        shrLog( "cuDeviceGetAttribute returned %d\n-> %s\n", (int)error_result, getCudaDrvErrorString(error_result) );
		exit(0);
    }
}
// end borrowed section


void cTeam::CUDAShowProperties(struct rohanContext& rSes, int device, FILE* fShow)
{
	if(fShow!=NULL){
		// following section borrowed from CUDA SDK, (C) Nvidia
		fprintf(fShow,"\nDevice %d has compute capability %d.%d.\n", device, rSes.deviceProp.major, rSes.deviceProp.minor);
		fprintf(fShow,"  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", 
			(float)rSes.deviceProp.totalGlobalMem/1048576.0f, (unsigned int long) rSes.deviceProp.totalGlobalMem);
		fprintf(fShow,"  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n", 
			rSes.deviceProp.multiProcessorCount, ConvertSMVer2Cores(rSes.deviceProp.major, rSes.deviceProp.minor),
			ConvertSMVer2Cores(rSes.deviceProp.major, rSes.deviceProp.minor) * rSes.deviceProp.multiProcessorCount);
		//int L2CacheSize;
		//getCudaAttribute<int>( &L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device );
		//if (L2CacheSize) {fprintf(fShow,"  L2 Cache Size:                                 %d bytes\n", L2CacheSize);}
		fprintf(fShow,"  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
			rSes.deviceProp.maxTexture1D, rSes.deviceProp.maxTexture2D[0], rSes.deviceProp.maxTexture2D[1],
			rSes.deviceProp.maxTexture3D[0], rSes.deviceProp.maxTexture3D[1], rSes.deviceProp.maxTexture3D[2]);
		fprintf(fShow,"  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
			rSes.deviceProp.maxTexture1DLayered[0], rSes.deviceProp.maxTexture1DLayered[1],
			rSes.deviceProp.maxTexture2DLayered[0], rSes.deviceProp.maxTexture2DLayered[1], rSes.deviceProp.maxTexture2DLayered[2]);
		fprintf(fShow,"  Total amount of constant memory:               %u bytes\n", rSes.deviceProp.totalConstMem); 
		fprintf(fShow,"  Total amount of shared memory per block:       %u bytes\n", rSes.deviceProp.sharedMemPerBlock);
		fprintf(fShow,"  Total number of registers available per block: %d\n", rSes.deviceProp.regsPerBlock);
		fprintf(fShow,"  Warp size:                                     %d\n", rSes.deviceProp.warpSize);
		fprintf(fShow,"  Maximum number of threads per block:           %d\n", rSes.deviceProp.maxThreadsPerBlock);
		fprintf(fShow,"  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
			rSes.deviceProp.maxThreadsDim[0],
			rSes.deviceProp.maxThreadsDim[1],
			rSes.deviceProp.maxThreadsDim[2]);
		fprintf(fShow,"  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
			rSes.deviceProp.maxGridSize[0],
			rSes.deviceProp.maxGridSize[1],
			rSes.deviceProp.maxGridSize[2]);
		// end borrowed section
		fprintf(fShow,"  sizeof devSes, devLearn, devNet:               %d, %d, %d\n", sizeof(rohanContext), sizeof(rohanLearningSet), sizeof(rohanNetwork) );
	}
	Barge->RLog(rSes, 0, __FUNCTION__);
}

void cTeam::TeamLog(struct rohanContext& rSes, int iRank, char * sLogEntry) 
{mIDfunc /// wrapper to call cBarge's RLog from CUDA C kernel launchers
	Barge->RLog(rSes, iRank, sLogEntry);
}

double cTeam::GetLastRmse(struct rohanContext& rSes, char model)
{mIDfunc /// Get latest recorded Rmse values
	double dReturn=888.888;
	char sLog[255];
	if(model=='B'){
		dReturn=rSes.dLastRmseB; // model B Block
	}
	if(model=='G'){
		dReturn=rSes.dLastRmseG; // model G global memory
	}
	if(model=='S'){
		dReturn=rSes.dLastRmseS; // model S serial
	}
	if(model=='3'){
		dReturn=rSes.dLastRmse3; // model 3 wide HN3
	}
	sprintf(sLog, "%s %c returns %f.", __FUNCTION__, model, dReturn);
	Barge->RLog(rSes, 0, sLog);
	return dReturn;
}

void cTeam::UpdateRmseRecord(struct rohanContext& rSes, double lastRmse, double bestRmse, char cModel)
{mIDfunc /// update latest Rmse values

	if(cModel=='B'){
		rSes.dLastRmseB=lastRmse; // model B Block
		rSes.dBestRmseB=bestRmse;
	}
	if(cModel=='G'){
		rSes.dLastRmseG=lastRmse; // model G global memory
		rSes.dBestRmseG=bestRmse;
	}
	if(cModel=='S'){
		rSes.dLastRmseS=lastRmse; // model S serial
		rSes.dBestRmseS=bestRmse;
	}
	if(cModel=='3'){
		rSes.dLastRmse3=lastRmse; // model 3 wide HN3
		rSes.dBestRmse3=bestRmse;
	}
	
	rSes.dPrevLastRMSE=rSes.dLastRMSE;
	rSes.dPrevBestRMSE=rSes.dBestRMSE;
	rSes.dLastRMSE=lastRmse;
	rSes.dBestRMSE=bestRmse;
	char sLog[255];
	sprintf(sLog,"%s(rSes, %3.2f, %3.2f, %c) replaces %3.2f, %3.2f", __FUNCTION__, lastRmse, bestRmse, cModel, rSes.dPrevLastRMSE, rSes.dPrevBestRMSE);
	Barge->RLog(rSes, 0, sLog);
}

double cTeam::CalcRmseSerial(struct rohanContext& rSes, int o)
{mIDfunc /*! checks sampled outputs vs evaluated outputs and calculates root mean squared error. */
	// Y evaluated outputs must be made ready before calling this function.
	// o controls which output is used for evaluation (0=all)
	double AccumSquareError=0.0;
	double dReturn;
	char sLog[255];

	sprintf(sLog, "\t%s(rSes, o=%d) called", __FUNCTION__, o); 
	//fprintf(rSes.hostBucket,"%s\n", sLog);
	Barge->RLog(rSes, 0, sLog); 

	for(int s=0; s<rSes.lSampleQtyReq; ++s){  // loop over all requested samples and documented outputs
		if(o){
			AccumSquareError+=rSes.rLearn->dSqrErr[IDX2C( o, s, (rSes.rLearn->iOutputQty+1))];
		}
		else {
			for(int i=1; i<=rSes.rLearn->iOutputQty; ++i){
				AccumSquareError+=rSes.rLearn->dSqrErr[IDX2C( o, s, (rSes.rLearn->iOutputQty+1))];
			}
		}
		//fprintf(rSes.hostBucket, "s%3d: D%6.3f - Y%6.3f -> SE%7.3f\n", s, rSes.rLearn->dDOutputs[IDX2C( 1, s, 2 )], rSes.rLearn->dYEval[IDX2C( 1, s, 2 )], rSes.rLearn->dSqrErr[IDX2C( 1, s, 2 )]);
	}
	// take the root of the mean of the accumulated square error
	if (o){
		dReturn=sqrt( (AccumSquareError/(rSes.lSampleQtyReq )) ); 
		sprintf(sLog, "\t%s(rSes, o=%d) returns %.6f=sqrt(%2.2f/%d)", __FUNCTION__, o , dReturn, AccumSquareError, rSes.lSampleQtyReq); 
	}
	else{
		dReturn=sqrt( (AccumSquareError/(rSes.lSampleQtyReq * rSes.rLearn->iOutputQty)) ); 
		sprintf(sLog, "\t%s(rSes, o=%d) returns %.6f=sqrt(%2.2f/(%d * %d))", __FUNCTION__, o , dReturn, AccumSquareError, rSes.lSampleQtyReq, rSes.rLearn->iOutputQty); 
	}
	
	//fprintf(rSes.hostBucket,"%s\n", sLog);
	Barge->RLog(rSes, 0, sLog); 
	return dReturn;
}

void cTeam::SetTestModels(char xp, char control)
{
	cExpModel=xp;
	cControlModel=control;
}

int cTeam::TrainNNThresh(struct rohanContext& rSes, int bChangeWeights)
{mIDfunc 
/*! checks sampled outputs vs evaluated outputs, and returns number of samples that exceed threshold
 *  excessive samples are submitted for backpropagation if bChangeWeights is true.
 */
	// needs multiple output support? XX
	int iReturn=0;
	double dDelta=0;
	int ROWLEN = rSes.rLearn->iOutputQty+1 ;
	//adjust requested amount to available values
	for(int s=0; s<rSes.lSampleQtyReq; ++s){  // loop over samples.
		int iOverMAX=0;
		for(int i=0; i<=rSes.rLearn->iOutputQty; ++i){  // loop over outputs.
			dDelta = (double) abs( rSes.rLearn->dDOutputs[ IDX2C( i, s, ROWLEN ) ] - rSes.rLearn->dYEval[ IDX2C( i, s, ROWLEN ) ] );
			if((dDelta*2)>rSes.rNet->iSectorQty)
				dDelta=rSes.rNet->iSectorQty-dDelta;
			if( dDelta > rSes.dMAX)  // if effective error exceeds MAX, make a note
				++iOverMAX;
		}
		if (iOverMAX!=0) {	 // if a note has been made. 
			++iReturn; // increment the number of excessive samples.
			if (bChangeWeights) {  // and correct weights if that is desired.
				cuBackpropSingleSample(rSes, s, *rSes.rNet, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rNet->Deltas, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval);
			}
		}
	}
	char sLog[255];
	sprintf(sLog, "\tTrainNNThresh(rSes, bChangeWeights=%d) returns %d submitted samples",  bChangeWeights, iReturn); 
	Barge->RLog(rSes, 0, sLog); 
	Barge->RLog(rSes, 0, __FUNCTION__);
	return (iReturn);
}

int cTeam::TrainNNThreshGGG(struct rohanContext& rSes, int bChangeWeights)
{mIDfunc 
/*! checks sampled outputs vs evaluated outputs, and returns number of samples that exceed threshold
 *  excessive samples are submitted for backpropagation if bChangeWeights is true.
 */
	// needs multiple output support? XX
	int iReturn=0;
	double dDelta=0;
	int ROWLEN = rSes.rLearn->iOutputQty+1 ;
	//adjust requested amount to available values
	for(int s=0; s<rSes.lSampleQtyReq; ++s){  // loop over samples.
		int iOverMAX=0;
		for(int i=0; i<=rSes.rLearn->iOutputQty; ++i){  // loop over outputs.
			dDelta = (double) abs( rSes.rLearn->dDOutputs[ IDX2C( i, s, ROWLEN ) ] - rSes.rLearn->dYEval[ IDX2C( i, s, ROWLEN ) ] );
			if((dDelta*2)>rSes.rNet->iSectorQty)
				dDelta=rSes.rNet->iSectorQty-dDelta;
			if( dDelta > rSes.dMAX)  // if effective error exceeds MAX, make a note
				++iOverMAX;
		}
		if (iOverMAX!=0) {	 // if a note has been made. 
			++iReturn; // increment the number of excessive samples.
			if (bChangeWeights) {  // and correct weights if that is desired.
				cuBackpropSingleSampleGGG(rSes, s, *rSes.rNet, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rNet->Deltas, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval);
			}
		}
	}
	char sLog[255];
	sprintf(sLog, "\tTrainNNThresh(rSes, bChangeWeights=%d) returns %d submitted samples",  bChangeWeights, iReturn); 
	Barge->RLog(rSes, 0, sLog); 
	Barge->RLog(rSes, 0, __FUNCTION__);
	return (iReturn);
}

