/* Includes, cuda */
#include "stdafx.h"

#include <boost/timer/timer.hpp>
using namespace boost::timer;

#include <boost/date_time/posix_time/posix_time.hpp> //include all types plus i/o
using namespace boost::posix_time;

extern int gDebugLvl, gDevDebug, gTrace;
extern float gElapsedTime, gKernelTimeTally;

//////////////// class cDrover begins ////////////////

cDrover::cDrover( rohanContext& rC, rohanLearningSet& rL, rohanNetwork& rN, cBarge& cB, cDeviceTeam& cdT)
{/// begin constructor
	SetContext(rC, rL, rN); 
	SetDroverBargeAndTeam(cB, cdT); 
	/*ShowMe();*/ 
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


int cDrover::SetDroverBargeAndTeam( class cBarge& cbB, class cDeviceTeam& cdtT)
{mIDfunc /// sets pointers to hitch barge to team and mount driver on barge
	Barge = &cbB;
	Team = &cdtT;
	Barge->SetDrover(this);
	Barge->SetTeam(Team);
	Team->SetDrover(this);
	Team->SetBarge(Barge);
	return 0;
}


int cDrover::DoAnteLoop(struct rohanContext& rSes, int argc, char * argv[])
{mIDfunc /// This function prepares all parameters and data structures necesary for learning and evaluation.
	int iReturn=1;
	
	if ( Barge->SetProgOptions( rSes, argc, argv ) < 2 ) // narch and samples are required
		return 0;
	iReturn=Barge->ObtainGlobalSettings(rSes);
	iReturn=Barge->ObtainSampleSet(rSes);
	iReturn=Barge->DoPrepareNetwork(rSes);
	fprintf(stdout, "Rohan v%s Neural Network Simulator\n", VERSION);
	if(iReturn*=ShowDiagnostics( rSes, *(rSes.rNet) ))
			iReturn*=Barge->ShowDiagnostics(); 
	
	return iReturn;
}


int cDrover::ShowDiagnostics(struct rohanContext& rSes, struct rohanNetwork& rNet)
{mIDfunc /// show some statistics, dump weights, and display warning and error counts
	double devdRMSE=0.0, hostdRMSE=0.0;
	int iReturn=1;
	cuDoubleComplex keepsakeWts[MAXWEIGHTS]; // 16 x 2048
	
	//if(iReturn=Barge->ObtainSampleSet(rSes)){
	//	iReturn=Barge->DoPrepareNetwork(rSes);
	if(true){
		cudaMemcpy( keepsakeWts, rSes.rNet->Wt, 16*MAXWEIGHTS, cudaMemcpyHostToHost); // backup weights for post-test restoration
		printf("(backup wt %08lX)\n", crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS ) ); // weight check
	
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
		printf("(restrd wt %08lX)\n", crc32buf( (char*)rSes.rNet->Wt, 16*MAXWEIGHTS ) ); // weight check
	}
	
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
	
	cout << "Main duty loop begin." << endl;
	if(rSes.bConsoleUsed){
			while(iSelect ){
				//Team->LetSlack(rSes);
				//Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
				//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
				iSelect=DisplayMenu(0, rSes);
				if (iSelect==1) iReturn=BeginSession(rSes); // new or resume session
				if (iSelect==2) iReturn=GetNNTop(rSes);
				//if (iSelect==3) iReturn=ReGetSampleSet(rSes); XX
				if (iSelect==4) iReturn=GetWeightSet(rSes);
				if (iSelect==5) iReturn=this->LetInteractiveEvaluation(rSes);
				if (iSelect==6) iReturn=this->LetInteractiveLearning(rSes);
				if (iSelect==7) iReturn=cuPreSaveNNWeights(rSes, 'D');
				if (iSelect==8) {iReturn=cuRandomizeWeightsBlock(rSes); 
								Team->LetEvalSet(rSes, 'H'); // this is performed on the host
								RmseNN(rSes, 0);
				}
				if (iSelect==9) iReturn=this->LetUtilities(rSes);
			}
	}
	else {
		
		if (Barge->vm.count("learn")) {
            vector<double> v;
			cout << "learn " << Barge->vm["learn"].as<string>() << "\n";
			Barge->OptionToDoubleVector("learn", v);
			if(v.size()==4){ // all params present, hopefully valid
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
				cuPreSaveNNWeights(rSes, cVenue);
				printf("Training terminates with %2.2f RMSE achieved towards %2.2f %c target RMSE\n", rSes.dRMSE, rSes.dTargetRMSE, cVenue);
			}
			else{ // missing or extra parameters
				if(v.size()<4){

				}
				else {
				}
			}
        }

        if (Barge->vm.count("eval")) {
			vector<int> v;
            cout << "eval " << Barge->vm["eval"].as<string>() << "\n";
			Barge->OptionToIntVector("eval", v);
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
			Barge->RLog(rSes, sLog);
			// include report for GUI
			sprintf(sLog, "report=%s\n", sFileAscii);
			Barge->HanReport(rSes, sLog);
        }
	}
	return iReturn;
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
		CLIbase(rSes);
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
		CLIbase(rSes);
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
		CLIbase(rSes);
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
		CLIbase(rSes);
	}
	printf("\n");
	// http://www.cplusplus.com/doc/ascii/
	while(a<'0'||a>'9')
		a=_getch();
	return ((int)a)-48;
}


int cDrover::CLIbase(struct rohanContext& rSes)
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

int BeginSession(struct rohanContext& rSes)
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
		cuMakeLayers(iInputQty, sNeuronsPerLayer, rSes); // make new layers
		rSes.rNet->dK_DIV_TWO_PI = rSes.rNet->iSectorQty / TWO_PI; // Prevents redundant conversion operations
		cuMakeNNStructures(rSes); // allocates memory and populates network structural arrays
		cuRandomizeWeightsBlock(rSes); // populate newtork with random weight values
		printf("Random weights loaded.\n");
		printf("%d-valued logic sector table made.\n", cuSectorTableMake(rSes));
		printf("\n");
		return rSes.rNet->iLayerQTY;
	}
	else
		return 999;
}

int cuMakeNNStructures(struct rohanContext &rSes)
{mIDfunc
/*! Initializes a neural network structure of the given number of layers and
 *  layer populations, allocates memory, and populates the set of weight values randomly.
 *
 * iLayerQTY = 3 means Layer 1 and Layer 2 are "full" neurons, with output-only neurons on layer 0.
 * 0th neuron on each layer is a stub with no inputs and output is alawys 1+0i, to accomodate internal weights of next layer.
 * This allows values to be efficiently calculated by referring to all layers and neurons identically.
 * 
 * rLayer[1].iNeuronQty is # of neurons in Layer 1, not including 0
 * rLayer[2].iNeuronQty is # of neurons in Layer 2, not including 0
 * rLayer[0].iNeuronQty is # of inputs in Layer 0 
 * iNeuronQTY[1] is # of neurons in Layer 1, including 0
 * iNeuronQTY[2] is # of neurons in Layer 2, including 0 */

	int lReturn=0;
//const cuDoubleComplex cdcZero = { 0, 0 }, 
	const cuDoubleComplex cdcInit = { -999.0, 999.0 };
	//cdcInit.x=-999.0; cdcInit.y=999.0;
	for (int i=0; i < rSes.rNet->iLayerQTY; ++i){  //Layer Zero has no need of weights! 8/13/2010
		struct rohanLayer& lay = rSes.rNet->rLayer[i];
		struct rohanNetwork * rnSrc = rSes.rNet;
		int DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE, L=i;
		//setup dimension values
		DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
		DSIZE = DQTY * sizeof(cuDoubleComplex) ;
		NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
		NSIZE = NQTY * sizeof(cuDoubleComplex) ;
		WQTY = DQTY * NQTY ; // weights = neurons * dendrites
		WSIZE = WQTY * sizeof(cuDoubleComplex) ;
		
		//allocate memory
		lay.Weights = (cuDoubleComplex*)malloc ( WSIZE ); // 2D array of complex weights
			mCheckMallocWorked(lay.Weights)
		lay.XInputs = (cuDoubleComplex*)malloc( DSIZE ); //allocate a pointer to an array of outputs
			mCheckMallocWorked(lay.XInputs)
		lay.ZOutputs = (cuDoubleComplex*)malloc( NSIZE ); //allocate a pointer to an array of outputs
			mCheckMallocWorked(lay.ZOutputs)
		lay.Deltas = (cuDoubleComplex*)malloc( NSIZE ); //allocate a pointer to a parallel array of learned corrections
			mCheckMallocWorked(lay.Deltas)
		lReturn+=lay.iNeuronQty*lay.iDendriteQty;
   		lReturn+=lay.iNeuronQty;
	
		//init values
		for (int i=0; i <= lay.iDendriteQty; ++i){
			for (int k=0; k <= lay.iNeuronQty; ++k){ 
				lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)].x=(double)rand()/65535; // necessary to promote one operand to double to get a double result
				lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)].y=(double)rand()/65535;
				//lay.Deltas[IDX2C(i, k, lay.iDendriteQty+1)]=cdcInit;
			}
			// reset neuron 0 weights to null
			lay.Weights[IDX2C(i, 0, lay.iDendriteQty+1)] = cdcZero;
			// mark inputs as yet-unused
			lay.XInputs[i]=cdcInit;
		}
		lay.Weights[IDX2C(0, 0, lay.iDendriteQty+1)].x=1.0; // neuron 0, dendrite 0 interior weight should always be equal to 1+0i
		for (int k=0; k <= lay.iNeuronQty; ++k){
			// mark outputs and deltas as yet-unused
			lay.ZOutputs[k]=cdcInit;
			lay.Deltas[k]=cdcInit;
		}
	}
	return lReturn; //return how many weights and outputs allocated
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
	iReturn=BinaryFileHandleRead(sWeightSet, &fileInput);
	if (iReturn==0) // unable to open file
		++rSes.iErrors;
	else{ // file opened normally
		// file opening and reading are separated to allow for streams to be added later
		iReturn=cuNNLoadWeights(rSes, fileInput); // reads weights into layered structures
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
		//Team->LetSlack(rSes);
		//Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
		//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
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
		//Team->LetSlack(rSes);
		//Team->LetEvalSet(rSes, rSes.lSampleQtyReq, 'H');
		//rSes.dRMSE = RmseNN(rSes, rSes.lSampleQtyReq);
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
			AsciiFileHandleRead( afname, &ASCIN );
			BinaryFileHandleWrite( bfname, &BINOUT );
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

	DoEndItAll(rSes);
	printf("Simulation terminated after %d warning(s), %d operational error(s).\n", rSes.iWarnings, rSes.iErrors);
#ifdef _DEBUG
	printf("Waiting on keystroke...\n");
	_getch();
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

