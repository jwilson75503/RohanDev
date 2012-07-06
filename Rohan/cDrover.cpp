/* Includes, cuda */
#include "stdafx.h"

#include <boost/timer/timer.hpp>
using namespace boost::timer;

#include <boost/date_time/posix_time/posix_time.hpp> //include all types plus i/o
using namespace boost::posix_time;

extern int gDebugLvl, gDevDebug, gTrace;
extern float gElapsedTime, gKernelTimeTally;

//////////////// class cDrover begins ////////////////

cDrover::cDrover( rohanContext& rC, rohanLearningSet& rL, rohanNetwork& rN, cBarge& cB, cTeam& cT)
{/// begin constructor
	SetContext(rC, rL, rN); 
	SetDroverBargeAndTeam(cB, cT); 
	//ShowMe();
} // end ctor

void cDrover::ShowMe()
{
	//ShowMeSes(* rSes, false);
	printf("Am Volga boatman.\n");
}


int cDrover::SetContext( rohanContext& rC, rohanLearningSet& rL, rohanNetwork& rN)
{/// enables pointer access to master context struct, sets up pointers to toher objects of interest
	rSes = &rC;
	// pointers to learning set and network object copies in host memory space are recorded
	rC.rLearn = &rL;
	rLearn = &rL;
	rC.rNet = &rN; 
	rNet = &rN;

	//rC.dDevRMSE=9.9; rSes->dDevRMSE=8.8; printf("<<<%f - %f>>>\n", rC.dDevRMSE, rSes->dDevRMSE);
	// pointers to learning set and network object copies in dev memory space are recorded
	cudaGetSymbolAddress( (void**)&rC.devSes, "devSes" );
		mCheckCudaWorked
	cudaGetSymbolAddress( (void**)&rC.devLearn, "devLearn" );
		mCheckCudaWorked
	cudaGetSymbolAddress( (void**)&rC.devNet, "devNet" );
		mCheckCudaWorked

	return 0;
}


int cDrover::SetDroverBargeAndTeam( class cBarge& cB, class cTeam& cT)
{mIDfunc /// sets pointers to hitch barge to team and mount driver on barge
	Barge = &cB;
	Team = &cT;
	Barge->SetDrover(this);
	Barge->SetTeam(Team);
	Team->SetDrover(this);
	Team->SetBarge(Barge);
	rSes->Team=&cT;
	
	return 0;
}


int cDrover::DoAnteLoop(struct rohanContext& rSes, int argc, char * argv[])
{mIDfunc /// This function prepares all parameters and data structures necesary for learning and evaluation.
	int iReturn=1; char sLog[255];
	
	fprintf(stdout, "Rohan %s neural net simulator\n", VERSION);
	fprintf(stdout, "%s\n", AUTHORCREDIT);
	if ( Barge->SetProgOptions( rSes, argc, argv ) < 2 ) // narch and samples are required
		return 0;
	iReturn=Barge->ObtainGlobalSettings(rSes);
	sprintf(sLog, "rohan=%s", argv[0]); Barge->RLog(rSes, GUIF, sLog );
	iReturn=Barge->ObtainSampleSet(rSes);
	iReturn=Barge->DoPrepareNetwork(rSes);
	if(iReturn*=ShowDiagnostics( rSes, *(rSes.rNet) ))
			iReturn*=Barge->ShowDiagnostics(rSes); 
	
	return iReturn;
}


int cDrover::ShowDiagnostics(struct rohanContext& rSes, struct rohanNetwork& rNet)
{mIDfunc /// show some statistics, dump weights, and display warning and error counts
	double devdRMSE=0.0, hostdRMSE=0.0;
	int iReturn=1;
	cuDoubleComplex keepsakeWts[MAXWEIGHTS]; // 16 x 2048
	
#ifdef _DEBUG
		cudaMemcpy( keepsakeWts, rSes.rNet->Wt, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // backup weights for post-test restoration
		//printf("(backup wt %08lX)\n", crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS ) ); // weight check
	
		Team->LetHitch(rSes);
		Team->LetSlack(rSes);
	
		Team->RmseEvaluateTest(rSes, rNet, 2 , 0);
		// some illustrative default values
		rSes.dTargetRMSE=floor((sqrt(rSes.dHostRMSE/10)*10)-1.0)+1.0;
		rSes.dMAX=rSes.dTargetRMSE-2;
		// more tests
		Team->ClassifyTest(rSes, rNet, 2, 0 );
		//rSes.dTargetRMSE=0.0;
		//rSes.dMAX=0.0;
		//	SetDevDebug(1); gDevDebug=1;
		Team->BackPropTest(rSes, rNet, 2, 128, 0);
		//	SetDevDebug(0);  gDevDebug=0;
		//cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//BackPropTest(rSes, rNet, 1024, 64, 0);
		//cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//BackPropTest(rSes, rNet, 10, 128, 0);
		//cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//BackPropTest(rSes, rNet, 32, 256, 0);
		//cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//BackPropTest(rSes, rNet, 10, 512, 0);
		//RmseEvaluateTest(rSes, rNet, 1 , 1);
		
		cudaMemcpy( rSes.rNet->Wt, keepsakeWts, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // post-test restoration of weights
		//printf("(restrd wt %08lX)\n", crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS ) ); // weight check
#endif	
	
	if (rSes.iWarnings) fprintf(stderr, "Drover Diagnosis: %d warnings.\n", rSes.iWarnings);
	if (rSes.iErrors) fprintf(stderr, "Drover Diagnosis: %d operational errors.\n", rSes.iErrors);

	return iReturn;
}


int cDrover::AskSampleSetName(struct rohanContext& rSes)
{mIDfunc /// chooses the learning set to be worked with Ante-Loop
	int iReturn=0; 
	//rSes.rLearn->bContInputs=false;
	//rSes.rLearn->iContOutputs=(int)false;
	//cout << "Samples treated as discrete or continuous by fractionality. XX" << endl;

	printf("Enter 0 for 10K-set, 9-36-1 weights\n\t 1 for 3-set, 331 weights\n\t 2 for 150-set, no wgts\n\t 3 for 4-set, 2x21 wgts\n\t 4 for 2-1 rand weights");
	printf("\n\t 5 for 416 samples x 200 inputs\n\t 6 for 10k-set, 9-45-1 weights\n\t 7 for 10k-set, 9-54-1 weights\n");
	std::cin >> gDebugLvl;
	switch ( gDebugLvl % 10) {
		case 0:
		  strcpy(rSes.sLearnSet, "AirplanePsDN1W3S10k.txt");
		  break;
		case 1:
		  strcpy(rSes.sLearnSet, "trivial3331.txt");
		  break;
		case 2:
		  strcpy(rSes.sLearnSet, "iris.txt");
		  break;
		case 3:
		  strcpy(rSes.sLearnSet, "trivial221.txt");
		  break;
		case 4:
		  strcpy(rSes.sLearnSet, "trivial3.txt");	
		  break;
		case 5:
		  strcpy(rSes.sLearnSet, "PC-63-32-200-LearnSet.txt");
		  break;
		case 6:
		  strcpy(rSes.sLearnSet, "LenaPsDN2W3S10k.txt");
		  break;
		case 7:
		  strcpy(rSes.sLearnSet, "LenaPsUN2W3S10k.txt");
		  break;
		default:
		  strcpy(rSes.sLearnSet, "iris.txt");
		  break;
	}
	gDebugLvl/=10; // drop final digit
	if (gDebugLvl) fprintf(stderr, "Debug level is %d.\n", gDebugLvl);
	return iReturn;
}


int cDrover::DoMainLoop(struct rohanContext& rSes)
{mIDfunc /// Trains a weight set to more closely reproduce the sampled outputs from the sampled inputs, and other options.
	int iReturn=0, iSelect=1;
	
	Barge->RLog(rSes, USERF, "Main duty loop begin.");
	if(rSes.bConsoleUsed){
			while(iSelect ){
				iSelect=DisplayMenu(0, rSes);
				if (iSelect==1) iReturn=AskSessionName(rSes); // new or resume session
				if (iSelect==2) iReturn=GetNNTop(rSes);
				//if (iSelect==3) iReturn=ReGetSampleSet(rSes); XX
				if (iSelect==4) iReturn=GetWeightSet(rSes);
				if (iSelect==5) iReturn=LetInteractiveEvaluation(rSes);
				if (iSelect==6) iReturn=LetInteractiveLearning(rSes);
				if (iSelect==7) iReturn=Barge->cuPreSaveNNWeights(rSes, 'D');
				if (iSelect==8) {iReturn=cuRandomizeWeightsBlock(rSes); 
								Team->LetEvalSet(rSes, 'H'); // this is performed on the host
								RmseNN(rSes, 0);
				}
				if (iSelect==9) iReturn=LetUtilities(rSes);
			}
	}
	else {
		
		if (Barge->vm.count("learn")) {
			DoLearnOpt(rSes);
        }

        if (Barge->vm.count("eval")) {
			DoEvalOpt(rSes);
        }
	}
	return iReturn;
}

void cDrover::DoLearnOpt(struct rohanContext& rSes)
{mIDfunc///parses learn directive program_option 
	char sLog[255];
	vector<double> v;
	
	if(Barge->VectorFromOption("learn", v, 4) ){ // all params present, hopefully valid
		rSes.dTargetRMSE = v.at(0);
		rSes.dMAX = v.at(1);
		rSes.lSampleQtyReq = (int)v.at(2);
		rSes.iBpropThreads = (int)v.at(3) * 32; // Warp factor 1, Mr Sulu!
		// perform learning
		char cVenue;
		if(rSes.iBpropThreads) //printf("AUTOMATED DEVICE LEARNING HERE\n");
			cVenue='D';
		else
			cVenue='H';
		rSes.lSamplesTrainable=Team->LetTrainNNThresh( rSes, rSes.iOutputFocus, 'R', rSes.dTargetRMSE, rSes.iEpochLength, cVenue);
		Barge->cuPreSaveNNWeights(rSes, cVenue);
		sprintf(sLog, "Learning terminates with %2.2f RMSE achieved towards %2.2f %c target RMSE", rSes.dRMSE, rSes.dTargetRMSE, cVenue);
		Barge->RLog(rSes, USERF, sLog);
		Barge->RLog(rSes, GUIF, "learn=pass");
		sprintf(sLog, "RMSE=%f", rSes.dRMSE);
		Barge->RLog(rSes, GUIF, sLog);
	}
	else{ // missing or extra parameters
		Barge->RLog(rSes, GUIF, "learn=fail");
		sprintf(sLog, "bad learn directive: %s", Barge->vm["learn"].as<string>().c_str());
		Barge->RLog(rSes, WARNINGF, sLog);
	}
}


void cDrover::DoEvalOpt(struct rohanContext& rSes)
{mIDfunc///parses eval directive program_option 
	char sLog[255];
	vector<int> v;

	if(Barge->VectorFromOption("eval", v, 4) ) {
		rSes.iSaveSampleIndex = v.at(0);
		rSes.iSaveInputs = v.at(1);
		rSes.iSaveOutputs = v.at(2);
		rSes.lSampleQtyReq = v.at(3);
		Team->GetRmseNN(rSes, rSes.iOutputFocus, 'R', 'D');
		//Team->LetSlack(rSes); // make sure device outputs are transferred back to host
		printf("%s: first %d samples requested\nRMSE= %f\n", rSes.sWeightSet, rSes.lSampleQtyReq, rSes.dDevRMSE);		
		// write evaluation report
		char sLog[255], sFileAscii[255];
		sprintf(sFileAscii,"%s%d%s",rSes.sSesName, (int)(rSes.dRMSE*100), "Evals.txt"); // do not exceed 254 char file name
		int lReturn=Barge->LetWriteEvals(rSes, *rSes.rLearn);
		// Log event
		sprintf(sLog, "%d evals writen to %s", lReturn, sFileAscii ); // document success and filename
		printf("%s\n", sLog);
		Barge->RLog(rSes, USERF, sLog);
		Barge->RLog(rSes, GUIF, "eval=pass");
		// include report for GUI
		sprintf(sLog, "report=%s", sFileAscii ); // document success and filename
		Barge->RLog(rSes, GUIF, sLog);
	}
	else{
		Barge->RLog(rSes, GUIF, "eval=fail");
		sprintf(sLog, "bad eval directive: %s", Barge->vm["eval"].as<string>().c_str());
		Barge->RLog(rSes, WARNINGF, sLog);
	}
}


int cDrover::DisplayMenu(int iMenuNum, struct rohanContext& rSes)
{mIDfunc
	char a='.';
	//refresh RMSE
	cuEvalNNLearnSet(rSes);
	RmseNN(rSes, rSes.iOutputFocus);
	
	//list menu items
	if(iMenuNum==0){
		printf("\n1 - Label session \"%s\"", rSes.sSesName);
		printf("\n2 X Network topology setup");
		printf("\n3 / Sample set load");
		printf("\n4 - Weight set load");
		printf("\n5 - Evaluation Feed Forward");
		printf("\n6 - Learning");
		printf("\n7 - Save weights");
		printf("\n8 - Randomize weights");
		printf("\n9 - Utilities");
		printf("\n0 - Quit");
		MenuBase(rSes);
	}
	if(iMenuNum==50){
		printf("\n1 - Include inputs: %d", rSes.iSaveInputs);
		printf("\n2 - Include outputs: %d", rSes.iSaveOutputs);
		printf("\n3 - Include sample index: %d", rSes.iSaveSampleIndex);
		printf("\n4 - change Samples used: %d", rSes.lSampleQtyReq);
		printf("\n5 - host serial evaluation");
		printf("\n6 - Asynchronous device commence");
		printf("\n7 - change Blocks per kernel: %d", rSes.iEvalBlocks);
		printf("\n8 - change Threads per block: %d", rSes.iEvalThreads);
		printf("\n9 - Save evals");
		printf("\n0 - Quit");
		MenuBase(rSes);
	}
	if(iMenuNum==60){
		printf("\n1 - change Target RMSE: % #3.3g", rSes.dTargetRMSE);
		printf("\n2 - change MAX error: % #3.3g", rSes.dMAX);
		printf("\n3 - change Epoch length: %d", rSes.iEpochLength);
		printf("\n4 - change Samples used: %d", rSes.lSampleQtyReq);
		printf("\n5 X Synchronous commence");
		printf("\n6 - Asynchronous commence");
		printf("\n7 - change Blocks per kernel: %d", rSes.iBpropBlocks);
		printf("\n8 - change Threads per block: %d", rSes.iBpropThreads);
		printf("\n9 X ");
		printf("\n0 - Quit");
		MenuBase(rSes);
	}
	if(iMenuNum==90){
		printf("\n1 - convert .txt list of weights to .wgt");
		printf("\n2 X RMSE/evaluate test");
		printf("\n3 X classification test");
		printf("\n4 X backpropagation test (weights are restored)");
		printf("\n5 - Show CUDA properties ");
		printf("\n6 - change Epoch length: %d", rSes.iEpochLength);
		printf("\n7 - change Samples used: %d", rSes.lSampleQtyReq);
		printf("\n8 - change Blocks per kernel: %d", rSes.iBpropBlocks);
		printf("\n9 - change Threads per block: %d", rSes.iBpropThreads);
		printf("\n0 - Quit");
		MenuBase(rSes);
	}
	printf("\n");
	// http://www.cplusplus.com/doc/ascii/
	while(a<'0'||a>'9')
		a=_getch();
	return ((int)a)-48;
}


int cDrover::MenuBase(struct rohanContext& rSes)
{mIDfunc /// displays the base information common to each/most menus
		printf("\n %s %d samples MAX %f, %d trainable", rSes.sLearnSet, rSes.rLearn->lSampleQty, rSes.dMAX, 
			TrainNNThresh(rSes, false));
		printf("\nRMSE: D %f, Y %f/%f ", rSes.dTargetRMSE, rSes.dHostRMSE, rSes.dDevRMSE);
		for(int i=0;i<rSes.rNet->iLayerQTY;++i)
			printf("L%d %d; ", i, rSes.rNet->rLayer[i].iNeuronQty);
		printf("%d sectors ", rSes.rNet->iSectorQty);
		if (rSes.bRInJMode) printf("ReverseInput "); 
	return 1;
}

int cDrover::AskSessionName(struct rohanContext& rSes)
{mIDfunc /// accepts keyboard input to define the name of the session, which will be used to name certain output files.
	cout << "\nEnter a session name: ";
	cin >> rSes.sSesName; 

	return 1;
}


int cDrover::GetNNTop(struct rohanContext& rSes)
{mIDfunc /// sets up network poperties and data structures for use
	char sNeuronsPerLayer[254];
	int iSectorQty, iInputQty;

	cout << "Enter # of sectors (0 to return): ";
	cin >> iSectorQty;
	if(iSectorQty){
		cout << "Enter # of inputs (0 to return): ";
		cin >> iInputQty; // last chance to quit
	}
	if(iSectorQty && iInputQty) {
		Barge->cuFreeNNTop(rSes); // release old network structures
		rSes.rNet->iSectorQty=iSectorQty; // update sector qty
		rSes.rNet->kdiv2=iSectorQty/2; // update sector qty
		rSes.rLearn->iInputQty=iInputQty; // upsdate input qty
		cout << "Enter numbers of neurons per layer separated by commas, \ne.g. 63,18,1 : ";
		cin >> sNeuronsPerLayer;
		Barge->cuMakeLayers(iInputQty, sNeuronsPerLayer, rSes); // make new layers
		rSes.rNet->dK_DIV_TWO_PI = rSes.rNet->iSectorQty / TWO_PI; // Prevents redundant conversion operations
		Barge->cuMakeNNStructures(rSes); // allocates memory and populates network structural arrays
		cuRandomizeWeightsBlock(rSes); // populate newtork with random weight values
		printf("Random weights loaded.\n");
		printf("%d-valued logic sector table made.\n", cuSectorTableMake(rSes));
		printf("\n");
		return rSes.rNet->iLayerQTY;
	}
	else
		return 999;
}

int cDrover::GetWeightSet(struct rohanContext& rSes)
{mIDfunc /// chooses and loads the weight set to be worked with
	int iReturn=0; 
	char sWeightSet[254];
	FILE *fileInput;
	
	cout << "Enter name of binary weight set: ";
	std::cin.clear();
	std::cin >> sWeightSet;
	strcat(sWeightSet, ".wgt");

	// File handle for input
	iReturn=Barge->BinaryFileHandleRead(sWeightSet, &fileInput);
	if (iReturn==0) // unable to open file
		++rSes.iErrors;
	else{ // file opened normally
		// file opening and reading are separated to allow for streams to be added later
		iReturn=Barge->cuNNLoadWeights(rSes, fileInput); // reads weights into layered structures
		if (iReturn) {
			printf("%d weights read.\n", iReturn);
			Barge->LayersToBlocks(rSes); //, *rSes.rNet);
		}
		else {
			printf("No Weights Read\n");
			iReturn=0;
		}
	}
	printf("\n");
	return iReturn;
}


int cDrover::LetInteractiveEvaluation(struct rohanContext& rSes)
{mIDfunc /// allows user to ask for different number of samples to be evaluated
	int iReturn=0, iSelect=1;
	
	while(iSelect){
		iSelect=DisplayMenu(50, rSes);
		if (iSelect==1) {rSes.iSaveInputs=(rSes.iSaveInputs ? false: true); }
		if (iSelect==2) {rSes.iSaveOutputs=(rSes.iSaveOutputs ? false: true); }
		if (iSelect==3) {rSes.iSaveSampleIndex=(rSes.iSaveSampleIndex ? false: true); }
		if (iSelect==4) {printf("Enter requested sample qty\n");std::cin >> rSes.lSampleQtyReq;
							if(rSes.lSampleQtyReq<1 || rSes.lSampleQtyReq > rSes.rLearn->lSampleQty)
								rSes.lSampleQtyReq=rSes.rLearn->lSampleQty;} 
		if (iSelect==5) { // serial values are computed and then displayed
					++iReturn; 
					boost::timer::auto_cpu_timer t;
					cuEvalNNLearnSet(rSes);
					RmseNN(rSes, rSes.iOutputFocus);
					printf("%s: first %d samples requested\nRMSE= %f", rSes.sWeightSet, rSes.lSampleQtyReq, rSes.dHostRMSE);		
		}
		if (iSelect==6) { // asynchronous kernel launch
					++iReturn;
					// device values are computed and then displayed
					//Team->LetEvalSet(rSes, 'D'); // eval on device
					Team->GetRmseNN(rSes, rSes.iOutputFocus, 'R', 'D');
					printf("%s: first %d samples requested\nRMSE= %f", rSes.sWeightSet, rSes.lSampleQtyReq, rSes.dDevRMSE);		
		}
		if (iSelect==7) {printf("Enter blocks per kernel\n");std::cin >> rSes.iEvalBlocks;}
		if (iSelect==8) {printf("Enter threads per block\n");std::cin >> rSes.iEvalThreads;}
		if (iSelect==9) {Barge->LetWriteEvals(rSes, *rSes.rLearn);} 
		if (iSelect==0) {} // quit
	}
	return iReturn;
}


int cDrover::LetInteractiveLearning(struct rohanContext& rSes)
{mIDfunc /// allows user to select learning thresholds
	int iReturn=0, iSelect=1;
	
	while(iSelect){
		iSelect=DisplayMenu(60, rSes);
		if (iSelect==1) {printf("Enter desired RMSE for learning\n");std::cin >> rSes.dTargetRMSE;}
		if (iSelect==2) {printf("Enter MAX allowable error per sample\n");std::cin >> rSes.dMAX;}
		if (iSelect==3) {printf("Enter iterations per epoch\n");std::cin >> rSes.iEpochLength;}
		if (iSelect==4) {printf("Enter samples requested\n");std::cin >> rSes.lSampleQtyReq;	 
							if(rSes.lSampleQtyReq<2 || rSes.lSampleQtyReq > rSes.rLearn->lSampleQty)
								rSes.lSampleQtyReq=rSes.rLearn->lSampleQty;} 
		//if (iSelect==5) {} // synchronous kernel launch
		if (iSelect==6) { // asynchronous kernel launch
					++iReturn;
							Team->LetTaut(rSes);
							rSes.lSamplesTrainable=Team->LetTrainNNThresh( rSes, rSes.iOutputFocus, 'R', rSes.dTargetRMSE, rSes.iEpochLength, 'D');
		}
		if (iSelect==7) {printf("Enter blocks per kernel\n");std::cin >> rSes.iBpropBlocks;}
		if (iSelect==8) {printf("Enter threads per block\n");std::cin >> rSes.iBpropThreads;}
		//if (iSelect==9) {} //
		if (iSelect==0) {} // quit
	}
	return iReturn;
}


int cDrover::LetUtilities(struct rohanContext& rSes)
{mIDfunc /// allows user to select learning thresholds
	int iReturn=0, iSelect=1;
	int iEpoch=rSes.iEpochLength;
	int iSampleQtyReq=rSes.lSampleQtyReq;
	int iBpropBlocks=rSes.iBpropBlocks;
	int iBpropThreads=rSes.iBpropThreads;
	
	while(iSelect){
		iSelect=DisplayMenu(90, rSes);
		if (iSelect==1) {
			char afname[100]="", bfname[100]="", 
				lineIn[255], *tok, *cSample;
			FILE * ASCIN, * BINOUT; cuDoubleComplex way;
			cout << "Enter name of .txt file to convert to .wgt:" << endl;
			cin >> bfname;
			strcat(afname, bfname); strcat(afname, ".txt"); strcat(bfname, ".wgt");
			Barge->AsciiFileHandleRead( afname, &ASCIN );
			Barge->BinaryFileHandleWrite( bfname, &BINOUT );
			#define MAX_REC_LEN 65536 /* Maximum size of input buffer */
			while(fgets(lineIn, MAX_REC_LEN, ASCIN)) { //each line is read in turn
				cSample = _strdup(lineIn);
				printf("%s", cSample); 
				tok=strtok( cSample , " ,\t" ); way.x=atof( tok ); 
				tok=strtok( NULL, " ,\t" ); way.y=atof( tok );
				printf("%f+%f\n", way.x, way.y);
				fwrite( &(way.x) , sizeof(double), 1, BINOUT);
				fwrite( &(way.y) , sizeof(double), 1, BINOUT);
			}
			fclose(ASCIN);
			fclose(BINOUT);
		}
		if (iSelect==2) {printf("Enter MAX allowable error per sample\n");std::cin >> rSes.dMAX;}
		if (iSelect==5) {Team->CUDAShowProperties(rSes, rSes.iMasterCalcHw, stdout);}
		if (iSelect==6) {printf("Enter iterations per epoch\n");std::cin >> rSes.iEpochLength;}
		if (iSelect==7) {printf("Enter samples requested\n");std::cin >> rSes.lSampleQtyReq;} //
		if (iSelect==8) {printf("Enter blocks per kernel\n");std::cin >> rSes.iBpropBlocks;}
		if (iSelect==9) {printf("Enter threads per block\n");std::cin >> rSes.iBpropThreads;}
		if (iSelect==0) {} // quit
	}
	return iReturn;
}


int cDrover::DoPostLoop(struct rohanContext& rSes) 
{mIDfunc /// Final operations including freeing of dynamically allocated memory are called from here. 
	int iReturn=0, iSelect=0;
	char sLog[255];

	sprintf(sLog, "Simulation terminated after %d warning(s), %d operational error(s).\n", rSes.iWarnings, rSes.iErrors);
	Barge->RLog(rSes, USERF, sLog);
	Barge->RLog(rSes, GUIF, "end=end");
	DoEndItAll(rSes);
	
#ifdef _DEBUG
	printf("Waiting on keystroke...\n");
	mExitKeystroke
#endif
	// call to source tracking here? 6/23/12

	return 0;
}


int cDrover::DoEndItAll(struct rohanContext& rSes)
{mIDfunc /// prepares for graceful ending of program
	int iReturn=0;

	Team->LetUnHitch(rSes);
	iReturn=Barge->DoCuFree(rSes);
	
	return iReturn;
}

