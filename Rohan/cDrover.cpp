/* Includes, cuda */
#include "stdafx.h"

#include <boost/timer/timer.hpp>
using namespace boost::timer;

#include <boost/date_time/posix_time/posix_time.hpp> //include all types plus i/o
using namespace boost::posix_time;

extern int gDebugLvl, gDevDebug, gTrace;
extern float gElapsedTime, gKernelTimeTally;

//////////////// class cDrover begins ////////////////

cDrover::cDrover( rohanContext& rC)
{/// begin constructor
	rC.Drover=this;
} // end ctor

void cDrover::ShowMe()
{
	printf("Am Volga boatman.\n");
}

int cDrover::SetContext( rohanContext& rC, int argc, _TCHAR * argv[])
{/// enables pointer access to internal objects
	rSes = &rC;
	rLearn = rC.rLearn;
	rNet = rC.rNet;
	Barge=rC.Barge;
	//Drover=rC.Drover;
	Ramp=rC.Ramp;
	Team=rC.Team;

	Barge->SetContext(rC);
	Ramp->SetContext(rC);
	Team->SetContext(rC);

	rSes->argc=argc;
	rSes->argv=argv;

	// pointers to object copies in device memory space are recorded
	cudaGetSymbolAddress( (void**)&rC.devSes, "devSes" );
		mCheckCudaWorked
	cudaGetSymbolAddress( (void**)&rC.devLearn, "devLearn" );
		mCheckCudaWorked
	cudaGetSymbolAddress( (void**)&rC.devNet, "devNet" );
		mCheckCudaWorked

	return 0;
}


int cDrover::DoAnteLoop(struct rohanContext& rSes, int argc, _TCHAR * argv[])
{mIDfunc /// This function prepares all parameters and data structures necesary for learning and evaluation.
	int iReturn=1; char sLog[255];
	
	if ( !Barge->PrepareAllSettings(rSes) )
		return 0;

	sprintf(sLog, "rohan=%s", argv[0]); Barge->RLog(rSes, GUIF, sLog );

	if ( !Barge->ObtainSampleSet(rSes) )
		return 0;
	if ( !Barge->DoPrepareNetwork(rSes) )
		return 0;
	if( ShowDiagnostics( rSes, *(rSes.rNet) ) )
		return 0;
	if( !Barge->ShowDiagnostics(rSes) )
		return 0;
	
	Barge->RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}


int cDrover::ShowDiagnostics(struct rohanContext& rSes, struct rohanNetwork& rNet)
{mIDfunc /// run tests, display accumulated warnongs and errors if any, return # of fatal errors
	double devdAccumSE=0.0, hostdRMSE=0.0;
	int pass=0, tt=0; char sLog[255];
	cuDoubleComplex keepsakeWts[MAXWEIGHTS]; // 16 x 2048
	
#ifdef _DEBUG
		cudaMemcpy( keepsakeWts, rSes.rNet->Wt, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // backup weights for post-test restoration
		sprintf(sLog, "(backup wt %08lX)\n", crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS ) ); // weight check
		Barge->RLog(rSes, 0, sLog);

		Team->SetTestModels('B','S');

		Team->LetHitch(rSes, 'B');
		Team->LetSlack(rSes);
	
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 3, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 5, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 7, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 11, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 13, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 17, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 19, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 23, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 29, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 31, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 32, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 37, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 41, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 43, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 47, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 53, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 59, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 61, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 64, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 67, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 71, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 73, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 79, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 83, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 89, 1); ++tt;
		pass+=Team->TeamTest(rSes, rNet, EVALUATEF, 2 , 0, 0, 96, 1); ++tt;
		
		// some illustrative default values
		rSes.dTargetRMSE=floor((sqrt(rSes.dLastRmseS/10)*10)-1.0)+1.0;
		rSes.dMAX=rSes.dTargetRMSE-2;
		//rSes.dTargetRMSE=0.0;
		//rSes.dMAX=0.0;
		// more tests
		pass+=Team->TeamTest(rSes, rNet, CLASSIFYF, 2 , 0, 0, 0, 1); ++tt;
		rSes.dTargetRMSE=0.0;
		rSes.dMAX=0.0;
		//	SetDevDebug(1); gDevDebug=1;
		pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 32, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 32, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 32, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 32, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 32, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 32, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 32, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 32, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 64, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 96, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 128, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 160, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 192, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 224, 0, 1); ++tt;
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 2 , 1, 256, 0, 1); ++tt;
		//
		//pass+=Team->TeamTest(rSes, rNet, BACKPROPF, 400 , 1, 32, 0, 1); ++tt;
		
		cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//printf("(restrd wt %08lX)\n", crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS ) ); // weight check
		sprintf(sLog, "%s: %d passed of %d.", __FUNCTION__, pass, tt);
		Barge->RLog(rSes, USERF+ADMINF, sLog);
		Barge->RLog(rSes, 0, "END TESTING");

#endif	
	
		if (rSes.iWarnings || rSes.iErrors){
			sprintf(sLog, "Diagnosis: %d warning(s), %d operational error(s).\n", rSes.iWarnings, rSes.iErrors );
			Barge->RLog(rSes, USERF, sLog);
		}
	Barge->RLog(rSes, 0, __FUNCTION__);
	return rSes.iErrors;
}


int cDrover::DoMainLoop(struct rohanContext& rSes)
{mIDfunc /// Trains a weight set to more closely reproduce the sampled outputs from the sampled inputs, and other options.
	int iReturn=0, iSelect=1;
	
	Barge->RLog(rSes, USERF, "Main duty loop begin.");
	if(rSes.bConsoleUsed){
			while(iSelect ){
				//iSelect=DisplayMenu(0, rSes);
				//if (iSelect==1) iReturn=AskSessionName(rSes); // new or resume session
				//if (iSelect==2) iReturn=GetNNTop(rSes);
				//if (iSelect==3) iReturn=ReGetSampleSet(rSes); XX
				//if (iSelect==4) iReturn=GetWeightSet(rSes);
				//if (iSelect==5) iReturn=LetInteractiveEvaluation(rSes);
				//if (iSelect==6) iReturn=LetInteractiveLearning(rSes);
				if (iSelect==7) iReturn=Ramp->SaveNNWeights(rSes, 'B');
				if (iSelect==8) {iReturn=Ramp->cuRandomizeWeightsBlock(rSes); 
								Team->LetEvalSet(rSes, 'S'); // this is performed on the host
								//CalcRmseSerial(rSes, 0);
				}
				//if (iSelect==9) iReturn=LetUtilities(rSes);
			}
	}
	else {
		
		if (Barge->vm.count("conty")) {
			DoContyOpt(rSes);
        }

		if (Barge->vm.count("learn")) {
			DoLearnOpt(rSes);
			++iReturn;
        }

        if (Barge->vm.count("eval")) {
			DoEvalOpt(rSes);
			++iReturn;
        }

		if (iReturn==0){
			Barge->RLog(rSes, WARNINGF+ADMINF, "No directives given.");
		}

	}
	Barge->RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}

void cDrover::DoContyOpt(struct rohanContext& rSes)
{mIDfunc///parses Conty directive program_option 
	char sLog[255];
	vector<int> v;
	
	if(Barge->VectorFromOption("conty", v, CONTYPMQTY) ){ // all params present, hopefully valid
		// perform Contying here, other activities at cBarge::GetContyOpt
		rSes.rLearn->iContInputs= v.at(0);
		rSes.rLearn->iContOutputs= v.at(1);
		rSes.iContActivation = v.at(2);
	}
	else{ // missing or extra parameters
		Barge->RLog(rSes, GUIF, "conty=fail");
		sprintf(sLog, "bad conty specification: %s", Barge->vm["conty"].as<string>().c_str());
		Barge->RLog(rSes, WARNINGF, sLog);
	}
	Barge->RLog(rSes, 0, __FUNCTION__);
}

void cDrover::DoLearnOpt(struct rohanContext& rSes)
{mIDfunc///parses learn directive program_option 
	char sLog[255];
	vector<double> v;
	int before, after;
	
	if(Barge->VectorFromOption("learn", v, LEARNPMQTY) ){ // all params present, hopefully valid
		// perform learning here, other activities at cBarge::GetLearnOpt
		char cModel;
		Barge->RLog(rSes, 0, "BEGIN LEARN DIRECTIVE");
		if(rSes.iBpropThreads) //printf("AUTOMATED DEVICE LEARNING HERE\n"); XX
			cModel='B';
		else
			cModel='S';

			Team->LetSlack(rSes);
			before=crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS );
		rSes.lSamplesTrainable=Team->LetTrainNNThresh( rSes, rSes.iOutputFocus, 'R', rSes.dTargetRMSE, rSes.iEpochLength, rSes.iEpochQty, cModel);
			Team->LetSlack(rSes);
			after=crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS );
		
		if(before!=after){
			Ramp->SaveNNWeights(rSes, cModel);
		}
		else
			Barge->RLog(rSes, USERF, "NO CHANGE IN WEIGHTS");

		sprintf(sLog, "Learning terminates with %2.2f RMSE achieved towards %2.2f %c target RMSE", rSes.dLastRMSE, rSes.dTargetRMSE, cModel);
		Barge->RLog(rSes, USERF, sLog);
		Barge->RLog(rSes, GUIF, "learn=pass");
		sprintf(sLog, "RMSE=%f", rSes.dLastRMSE);
		Barge->RLog(rSes, GUIF, sLog);
		Barge->RLog(rSes, 0, "END LEARN DIRECTIVE");
	}
	else{ // missing or extra parameters
		Barge->RLog(rSes, GUIF, "learn=fail");
		sprintf(sLog, "bad learn directive: %s", Barge->vm["learn"].as<string>().c_str());
		Barge->RLog(rSes, WARNINGF, sLog);
	}
	Barge->RLog(rSes, 0, __FUNCTION__);
}


void cDrover::DoEvalOpt(struct rohanContext& rSes)
{mIDfunc///parses eval directive program_option 
	char sLog[255], sFileAscii[255];
	vector<int> v;
	double RMSE;

	//other acrtivities at cBarge::GetEvalOpt
	if(Barge->VectorFromOption("eval", v, EVALPMQTY) ) {
		Barge->RLog(rSes, 0, "BEGIN EVAL DIRECTIVE");
		rSes.iSaveSampleIndex = v.at(0);
		rSes.iSaveInputs = v.at(1);
		rSes.iSaveOutputs = v.at(2);
		rSes.lSampleQtyReq = v.at(3);
		rSes.iEvalSkip = v.at(4);
		if( rSes.lSampleQtyReq==0) // if specifying 0, set to all w/o comment
			rSes.lSampleQtyReq=rSes.lSampleQty;
		if( rSes.lSampleQtyReq<0 || rSes.lSampleQtyReq>rSes.lSampleQty ) {// if specifying less than none or more than all
			rSes.lSampleQtyReq=rSes.lSampleQty;// set request to all
			sprintf(sLog, "%s: Bad sample req value, set to %d", __FUNCTION__, rSes.iEvalSkip );
			Barge->RLog(rSes, WARNINGF+USERF, sLog);
		}
		if( rSes.iEvalSkip<0 || (rSes.iEvalSkip+rSes.lSampleQtyReq)>rSes.lSampleQty ){ // if asking to skip less than none or enough to push request past end
			rSes.iEvalSkip=rSes.lSampleQty-rSes.lSampleQtyReq; // set skip to allow request before end
			sprintf(sLog, "%s: Bad sample skip value, set to %d", __FUNCTION__, rSes.iEvalSkip );
			Barge->RLog(rSes, WARNINGF+USERF, sLog);
		}
		
		sprintf(sLog, "%s: first %d samples requested", rSes.sWeightSet, rSes.lSampleQtyReq);
		Barge->RLog(rSes, USERF, sLog);
		RMSE=Team->GetRmseNN( rSes, rSes.iOutputFocus, 'R', rSes.cEngagedModel );		
		// write evaluation report
		sprintf(sFileAscii, "%s%03d%s", rSes.sSesName, (int)RMSE*100, "Evals.txt"); // do not exceed 254 char file name
		Ramp->LetWriteEvals(rSes, *rSes.rLearn, sFileAscii, rSes.cEngagedModel);
		Barge->RLog(rSes, 0, "END EVAL DIRECTIVE");
	}
	else{
		Barge->RLog(rSes, GUIF, "eval=fail");
		sprintf(sLog, "bad eval directive: %s", Barge->vm["eval"].as<string>().c_str());
		Barge->RLog(rSes, WARNINGF, sLog);
	}
	Barge->RLog(rSes, 0, __FUNCTION__);
}


int cDrover::DoPostLoop(struct rohanContext& rSes) 
{mIDfunc /// Final operations including freeing of dynamically allocated memory are called from here. 
	int iReturn=0, iSelect=0;
	char sLog[255];

	sprintf(sLog, "Simulation terminated after %d warning(s), %d operational error(s).\n", rSes.iWarnings, rSes.iErrors);
	Barge->RLog(rSes, USERF, sLog);
	Barge->RLog(rSes, GUIF, "end=end");
	Barge->LogFlush(rSes);
	DoEndItAll(rSes);
	
#ifdef _DEBUG
	printf("Waiting on keystroke...\n");
	mExitKeystroke
#endif
	// call to source tracking here? 6/23/12
	Barge->RLog(rSes, 0, __FUNCTION__);
	return 0;
}


int cDrover::DoEndItAll(struct rohanContext& rSes)
{mIDfunc /// prepares for graceful ending of program
	int iReturn=0;

	Team->LetUnHitch(rSes);
	iReturn=Barge->DoCuFree(rSes);
	Barge->LogFlush(rSes);
	Barge->RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}

