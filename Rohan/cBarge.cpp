/* Includes, cuda */
#include "stdafx.h"

// BOOST libraries
#include <boost/date_time/gregorian/gregorian.hpp> //include all types plus i/o
using namespace boost::gregorian;


#include <iostream>
#include <fstream>
#include <iterator>
using namespace std;

// A helper function to simplify the main part.
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
    return os;
}
//#include <boost/filesystem.hpp>
//using namespace boost::filesystem;


extern int gDebugLvl, gTrace;


//////////////// class cBarge begins ////////////////

void cBarge::ShowMe()
{
	//ShowMeSes(* rSes, false);
	printf("Am stout barje.\n");
}


int cBarge::SetContext( rohanContext& rC)
{/// enables pointer access to master context struct
	rSes = &rC;
	rLearn = rC.rLearn;
	rNet = rC.rNet;
	//rSes.cbBarge=this;
	return 0;
}


int cBarge::SetDrover( class cDrover * cdDrover)
{/// enables pointer access to active Drover object
	Drover = cdDrover;
	return 0;
}


int cBarge::SetTeam( class cDeviceTeam * cdtTeam)
{/// enables pointer access to active Team object
	Team = cdtTeam;
	return 0;
}


int cBarge::SetProgOptions(struct rohanContext& rSes, int argc, char * argv[])
{mIDfunc /// Declare the supported options.
	int iReturn=0; // returns number of essential parameters present, or -1 if an error or prefunctory parameter like help or version is supplied.
    try {
		string config_file, eval_string, learn_string, net_arch, sample_file, tag_session, weight_file;
		
		// Declare a group of options that will be 
		// allowed only on command line
		po::options_description generic("Generic options");
		generic.add_options()
			("version,v", "print version string")
			("help,h", "produce help message")
			("config,c", po::value<string>(&config_file)->default_value("rohan.roh"), "name of a file of a configuration.")
            ;
		    
		// Declare a group of options that will be 
		// allowed both on command line and in
		// config file
		po::options_description config("Configuration");
		config.add_options()
            ("network,n", po::value<string>(&net_arch), "network sectors, inputs, 1st hidden layer, 2nd hidden layer, outputs")
			("samples,s", po::value<string>(&sample_file), "text file containing sample input-output sets")
			("weights,w", po::value<string>(&weight_file), ".wgt file containing complex weight values")
			("learn,l", po::value<string>(&eval_string), "train in pursuit of target RMSE given MAX criterion")
			("eval,e", po::value<string>(&learn_string), "evaluate samples and report")
			("tag,t", po::value<string>(&tag_session)->default_value("DefaultSession"), "tag session with an identifying string")
			;

		// Hidden options, will be allowed both on command line and
		// in config file, but will not be shown to the user.
		po::options_description hidden("Hidden options");
		hidden.add_options()
			("include-path,I", po::value< vector<string> >()->composing(), "include path")
			("input-file", po::value< vector<string> >(), "input file")
			; 

		po::options_description cmdline_options;
		cmdline_options.add(generic).add(config).add(hidden);

		po::options_description config_file_options;
		config_file_options.add(config).add(hidden);

		po::options_description visible("Allowed options");
		visible.add(generic).add(config);

        po::positional_options_description p;
        p.add("input-file", -1);
        
        //po::variables_map vm;
        store(po::command_line_parser(argc, argv).
              options(cmdline_options).positional(p).run(), vm);
        notify(vm);
        
        if (vm.count("help")) {
            cout << visible << "\n";
            return -1;
        }

        if (vm.count("version")) {
            cout << VERSION << ".\n";
			return -1;
        }

		if (vm.count("tag"))
		{
			cout << "Tagged session is : " ;
			cout << vm["tag"].as<string>() << "\n";
		}

		ifstream ifs(config_file.c_str());
        if (!ifs)
        {
            cout << "can not open config file: " << config_file << "\n";
            //return 0;
        }
        else
        {
            store(parse_config_file(ifs, config_file_options), vm);
            notify(vm);
        }
    
		if (vm.count("include-path"))
        {
            cout << "Include paths are: " 
                 << vm["include-path"].as< vector<string> >() << "\n";
        }

        if (vm.count("input-file"))
        {
            cout << "Input files are: " 
                 << vm["input-file"].as< vector<string> >() << "\n";
        }
	
		if (vm.count("config"))
        {
            cout << "Config files are: ";
            cout << vm["config"].as<string>() << "\n";
        }

		if (vm.count("network"))
        {
            cout << "Network architecture is: ";
            cout << vm["network"].as<string>() << "\n";
			++iReturn;
        }
		else cout << "No network specified!\n";

		if (vm.count("samples"))
        {
            cout << "Samples file is : ";
			cout << vm["samples"].as<string>() << "\n";
			++iReturn;
        }
		else cout << "No samples specified!\n";

		if (vm.count("weights"))
        {
			cout << "Weights file is : ";
			cout << vm["weights"].as<string>() << "\n";
        }
		else cout << "No weights specified.\n";

		if (vm.count("learn"))
        {
			cout << "Learn directive is : ";
			cout << vm["learn"].as<string>() << "\n";
        }

		//if (vm.count("eval"))
  //      {
		//	cout << "Eval directive is : ";
		//	cout << vm["eval"].as<string>() << "\n";
        //}

	}
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return -1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
		return -1;
    }

    return iReturn;
}


char sep_to_space(char c){
  return c == ',' || c == '<' || c == '>' ? ' ' : c;
}

//template <typename T>
void cBarge::OptionToIntVector(char * sOption, vector<int> & n)
{mIDfunc// converts strings to vectors of any numeric type
	string s;
	s=cBarge::vm[sOption].as<string>();
	transform(s.begin(), s.end(), s.begin(), sep_to_space);
	stringstream ss(s);
	copy(istream_iterator<int>(ss), istream_iterator<int>(),std::back_inserter(n));
}

void cBarge::OptionToDoubleVector(char * sOption, vector<double> & n)
{mIDfunc// converts strings to vectors of any numeric type
	string s;
	s=cBarge::vm[sOption].as<string>();
	transform(s.begin(), s.end(), s.begin(), sep_to_space);
	stringstream ss(s);
	copy(istream_iterator<double>(ss), istream_iterator<double>(),std::back_inserter(n));
}


int cBarge::ObtainGlobalSettings(struct rohanContext& rSes)
{mIDfunc /// sets initial and default value for globals and settings
	int iReturn=1;
	
	// Context settings
	// ERRORS and progress tracking
	gTrace=0; 
	gDebugLvl=0; 
	rSes.iWarnings=0; rSes.iErrors=0; 
	rSes.lSampleQtyReq=0;
	rSes.lMemStructAlloc=0;
	// eval related
	rSes.iSaveInputs=0 /* include inputs when saving evaluations */;
	rSes.iSaveOutputs=0 /* include desired outputs when saving evaluations */;
	rSes.iSaveSampleIndex=0 /* includes sample serials when saving evaluations */;
	rSes.iOutputFocus=1 /*! which output is under consideration (0=all) */;
	rSes.iEvalBlocks=128; 
	rSes.iEvalThreads=128; 
	// hardware related
		if (Team->CUDAverify(rSes)>=2.0){ // assigns .dMasterCalcVer, .deviceProp.major, .deviceCount
			cutilSafeCall( cudaSetDevice(rSes.iMasterCalcHw) ); /// all cuda calls to run on first device of highest compute capability device located
			if (gDebugLvl) cout << "CUDA present, device " << rSes.iMasterCalcHw << " selected." << endl;
		}
		else {
			if (rSes.dMasterCalcVer>1.0)
				fprintf(stderr, "Warning: CUDA hardware below Compute Capability 2.0.\n");
			else
				fprintf(stderr, "Warning: No CUDA hardware or no CUDA functions present.\n");
			rSes.iMasterCalcHw=-1;
			++rSes.iWarnings;
			iReturn=0;
		}
	// input handling
	rSes.iEvalMode=0;
	rSes.bConsoleUsed=false;
	rSes.bRInJMode=false; 
	rSes.bRMSEon=true; 
	strcpy( rSes.sLearnSet, vm["samples"].as<string>().c_str() );
	strcpy( rSes.sWeightSet, vm["weights"].as<string>().c_str() );
	// learning related
	rSes.lSamplesTrainable=-1;
	rSes.iOutputFocus=1;
	rSes.iBpropBlocks=1; 
	rSes.iBpropThreads=256; 
	rSes.dHostRMSE=0.0;
	rSes.dDevRMSE=0.0;
	rSes.dRMSE=0.0;
	rSes.dTargetRMSE=0.0;
	rSes.dMAX=0.0;
	rSes.iEpochLength=1000; 
	// network
	rSes.iContActivation=true; //rSes.rNet->iContActivation=true; 
		vector<int> v;
		//string s=vm["network"].as<string>();
		//OptionToNumericVectorT<int>( "network", v);
		OptionToIntVector( "network", v);
	rSes.iSectorQty = v.at(0);
	rSes.iInputQty = v.at(1);
	
	if( v.at(2) &&  v.at(3) &&  v.at(4) ){ // 384, 9, 36, 2, 1
		rSes.iFirstHiddenSize = v.at(2);
		rSes.iSecondHiddenSize = v.at(3);
		rSes.iOutputQty = v.at(4);
	}
	if( v.at(2) &&  v.at(3) && !v.at(4) ){ // 384, 9, 36, 2, 0
		rSes.iFirstHiddenSize = v.at(2);
		rSes.iSecondHiddenSize = 0;
		rSes.iOutputQty = v.at(3);
	}
	if( v.at(2) && !v.at(3) &&  v.at(4) ){ // 384, 9, 36, 0, 1
		rSes.iFirstHiddenSize = v.at(2);
		rSes.iSecondHiddenSize = 0;
		rSes.iOutputQty = v.at(4);
	}
	if( v.at(2) && !v.at(3) && !v.at(4) ){ // 384, 9, 36, 0, 0
		rSes.iFirstHiddenSize = v.at(2);
		rSes.iSecondHiddenSize = 0;
		rSes.iOutputQty = 1;
	}
	if(!v.at(2) &&  v.at(3) &&  v.at(4) ){ // 384, 9, 0, 2, 1
		rSes.iFirstHiddenSize = v.at(3);
		rSes.iSecondHiddenSize = 0;
		rSes.iOutputQty = v.at(4);
	}
	if(!v.at(2) &&  v.at(3) && !v.at(4) ){ // 384, 9, 0, 2, 0
		rSes.iFirstHiddenSize = v.at(3);
		rSes.iSecondHiddenSize = 0;
		rSes.iOutputQty = 1;
	}
	if(!v.at(2) && !v.at(3) &&  v.at(4) ){ // 384, 9, 0, 0, 1
		rSes.iFirstHiddenSize = 0;
		rSes.iSecondHiddenSize = 0;
		rSes.iOutputQty = v.at(4);
	}
	if(!v.at(2) && !v.at(3) && !v.at(4) ){ // 384, 9, 0, 0, 0
		rSes.iFirstHiddenSize = 0;
		rSes.iSecondHiddenSize = 0;
		rSes.iOutputQty = 1;
	}
	rSes.iLayerQty = (rSes.iFirstHiddenSize ? 1:0) + (rSes.iSecondHiddenSize ? 1:0) + 1;
	// record keeping, including tag
	char sPath[MAX_PATH], sHanPath[MAX_PATH];
	GetUserDocPath(sPath); // sPath now has "C:\users\documents"
	sprintf(rSes.sRohanVerPath, "%s\\Rohan_%s", sPath, VERSION); // .sRohanVerPath has "C:\users\documents\Rohan_0.9.3"
	strcpy( rSes.sSesName, vm["tag"].as<string>().c_str() );
	if(DirectoryEnsure(rSes.sRohanVerPath)){
		using namespace boost::posix_time; 
		ptime now = second_clock::local_time(); //use the clock
		// establish Rlog
		sprintf(sPath, "%s\\RohanLog.txt", rSes.sRohanVerPath); // sPath has full rlog pathname
		rSes.ofsRLog=new ofstream(sPath, std::ios::app|std::ios::out); 
		*(rSes.ofsRLog) << "\tSTART Rohan v" << VERSION << " Neural Network Application\n";
		// establish .han file 
		sprintf( sHanPath, "%s\\%s.han", rSes.sRohanVerPath, rSes.sSesName); //sHanPath has fill hanlog pathname
		rSes.ofsHanLog=new ofstream(sHanPath, std::ios::app|std::ios::out); 
		*(rSes.ofsHanLog) << "#\t" << now << "\tSTART Rohan v" << VERSION << " Neural Network Application\n" ;
		// establish bucket files
		AsciiFileHandleWrite(rSes.sRohanVerPath, "DevBucket.txt", &(rSes.deviceBucket));
		//fprintf(rSes.deviceBucket, "%s\tSTART Rohan v%s Neural Network Application\n", "to_simple_string(now)", VERSION);
		AsciiFileHandleWrite(rSes.sRohanVerPath, "HostBucket.txt", &(rSes.hostBucket));
		//fprintf(rSes.hostBucket, "%s\tSTART Rohan v%s Neural Network Application\n", "to_simple_string(now)", VERSION);
	}
	else {
		errPrintf("Directory %s could not be created\n", rSes.sRohanVerPath);
		++rSes.iWarnings;
	}
	// samples
	rSes.lSampleQty=0; // no samples loaded yet
	rSes.lSampleQtyReq=-1; // no samples loaded yet
	//rSes.iInputQty // part of network above
	//rSes.iOutputQty // part of network above

	// LEARNING SET
	//int iEvalMode /*! Defaults to discrete outputs but a value of 0 denotes Continuous outputs. */;
	rSes.rLearn->lSampleQty=0;
	rSes.rLearn->iValuesPerLine=0;
	rSes.rLearn->iInputQty=rSes.iInputQty;
	rSes.rLearn->iOutputQty=rSes.iOutputQty;
	//rohanSample *rSample /*! Array of rohanSample structures. */; //removed 10/24/11
	//FILE *fileInput /*! The filehandle used for reading the learning set from the given file-like object. */; 
	rSes.rLearn->bContInputs=false; //default
	rSes.rLearn->iContOutputs=false; //default
	rSes.rLearn->lSampleIdxReq=-1;
	//void* hostLearnPtr;
	
	// NETWORK STRUCT
	rSes.rNet->iSectorQty=rSes.iSectorQty; // 4
	rSes.rNet->kdiv2=rSes.iSectorQty/2; 
	rSes.rNet->iLayerQTY=rSes.iLayerQty+1; // 4
	rSes.rNet->iContActivation=rSes.iContActivation;
	rSes.rNet->dK_DIV_TWO_PI=rSes.iSectorQty/TWO_PI;
	rSes.rNet->two_pi_div_sect_qty=TWO_PI/rSes.iSectorQty; // 8

		
	if(gTrace) cout << "Tracing is ON.\n" ;
	if (gDebugLvl){
		cout << "Debug level is " << gDebugLvl << "\n" ;
		cout << "Session warning and session error counts reset.\n";
		rSes.rNet->iContActivation ? cout << "Activation default is CONTINUOUS.\n" : cout << "Activation default is DISCRETE.\n"; 
		// XX defaulting to false makes all kinds of heck on the GPU
		rSes.bRInJMode ? cout << "Reversed Input Order is ON.\n" : cout << "Reversed Input Order is OFF.\n"; 
		// this is working backward for some reason 2/08/11 // still fubared 3/7/12 XX
		rSes.bRMSEon ? cout << "RMSE stop condition is ON. XX\n" : cout << "RMSE stop condition is OFF. XX\n"; //
		cout << "Epoch length is " << rSes.iEpochLength << " iterations.\n";
		cout << rSes.iEvalBlocks << " EVAL Blocks per Kernel, " << rSes.iEvalThreads << " EVAL Threads per Block.\n";
		cout << rSes.iBpropBlocks << " BPROP Blocks per Kernel, " << rSes.iBpropThreads << " BPROP Threads per Block.\n";
		rSes.iEvalMode ? cout << "Continuous Inputs TRUE by DEFAULT.\n" : cout << "Continuous Inputs FALSE by DEFAULT.\n";
		rSes.iContActivation ? cout << "Continuous Outputs true by DEFAULT.\n" : cout << "Continuous Outputs false by DEFAULT.\n";
	}

	return iReturn;
}


int cBarge::ObtainSampleSet(struct rohanContext& rSes)
{mIDfunc /// loads the learning set to be worked with Ante-Loop
	try{
		int iReturn=0; 
		FILE *fileInput;
		// File handle for input
		iReturn=AsciiFileHandleRead(rSes.sLearnSet, &fileInput);
		if (iReturn==0) // unable to open file
			++rSes.iErrors;
		else{ // file opened normally
			// file opening and reading are separated to allow for streams to be added later
			int lLinesRead=DoLoadSampleSet(rSes, fileInput);
			if (lLinesRead) {
				printf("Parsed %d lines from %s\nStored %d samples, %d input values, %d output values each.\n", 
					lLinesRead, rSes.sLearnSet, rSes.rLearn->lSampleQty, rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
				//verify samples fall wihtin sector values
				if(CurateSectorValue(rSes)) {
					CompleteHostLearningSet(rSes);
				}
				else{
					return 0;
				} 
			}
			else {
				printf("No Samples Read by cuLoadSampleSet\n");
				iReturn=0;
			}
		}

		return iReturn;
	}
	catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return -1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
		return -1;
    }
}


int cBarge::DoLoadSampleSet(struct rohanContext& rSes, FILE *fileInput)
{mIDfunc/// pulls in values from .txt files, used for testing before main loop
	// Returns lines read in (0 = error)

	#define MAX_REC_LEN 65536 /* Maximum size of input buffer */

	int  lLinesQty=0,lMaxLines=256; /* countable number of lines, number of lines with memory allocation */
	char cThisLine[MAX_REC_LEN]; /* Contents of current line */
	int iArchLineIdx; // The line number that has the sample qty in it
	rSes.rLearn->iValuesPerLine=0; rSes.rLearn->lSampleQty=0;
	// reset quantities for counting later
	char **cLines = (char **)malloc(256 * sizeof (char *));
	// 1d array of lines in text file beginning with first line in position zero
	// learning set format allows for first or first and second lines to he paramteres rather than samples

	while (fgets(cThisLine, MAX_REC_LEN, fileInput)) { //each line is read in turn
		cLines[lLinesQty++] = _strdup(cThisLine); // each line is copied to a string in the array
		if (!(lMaxLines > lLinesQty)) {  // if alloated space is used up, double it.
			lMaxLines *= 2;
			void * temp = realloc(cLines, lMaxLines * sizeof (char *));
			if (!temp) {
				  for (int k=0;k<lLinesQty;++k) {
					  free(cLines[k]);
				  }
				  printf("Realloc ran out of space?  OH NOES! %s line %d\n", __FILE__, __LINE__);
				  return 0;
			} else {
				  cLines = (char **)temp;
			}
		}
	}
	fclose(fileInput); // close stream when fgets returns false (no more lines)
	// this should be a shrinking, and should never fail.
	cLines = (char **)realloc(cLines, lLinesQty * sizeof (char*));
		mCheckMallocWorked(cLines)
	// 0 means ContActivation
	// 1 means discrete, and values of 2+ indicate parameter has been omitted,
	// defaulting to discrete, and value is actually # of samples in file
	rSes.iEvalMode=atof(cLines[0])>1 ? 1 : 0; iArchLineIdx = 1 - rSes.iEvalMode;
	if (rSes.iEvalMode) 
		printf ("Discrete output values indicated.\n");
	else 
		printf("Continuous output values indicated.\n");

	char *sArch, *tok;

	sArch = _strdup(cLines[iArchLineIdx]); // make a sacrificial copy
	tok = strtok(sArch, " ,\t");
	//  condiitonal eliminated after move to config files and cli args - 6/27/12
	//if (sArch==NULL) // no additional params present
	//	printf("No params present; fetch from config file or command line (not yet supported XX).\n");
	//else {
		cuMakeArchValues(cLines[iArchLineIdx], rSes);
		rSes.rLearn->iInputQty=rSes.rNet->rLayer[0].iNeuronQty;
		rSes.rLearn->iOutputQty=rSes.rNet->rLayer[rSes.rNet->iLayerQTY-1].iNeuronQty;
		printf("%d inputs, %d output(s) specified.\n", rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
	//}
	
	rSes.rLearn->lSampleQty=rSes.lSampleQtyReq=rSes.lSampleQty=(int)atof(cLines[iArchLineIdx]); //record qty of samples
	printf("%d samples specified: ", rSes.rLearn->lSampleQty);
			
// Parses lines of text for input values and output value and stores them in dynamic int arrays
// returns # of inputs per line
	char *pch; 
	char  *cSample;

	int lCurrentLine=atof(cLines[0])>1 ? 1 : 2; //find which line the samples begin
	cSample=_strdup(cLines[2]); // strtok chops up the input string, so we must make a copy
	pch = strtok (cSample, " ,\t");
	while (pch != NULL) {// this loop counts the values present in a copy of line 2, which has to be a sample line
		pch = strtok (NULL, " ,\t"); ++rSes.rLearn->iValuesPerLine;
	}
	int iExcessValueQty = rSes.rLearn->iValuesPerLine - (rSes.rLearn->iInputQty + rSes.rLearn->iOutputQty );
	if(iExcessValueQty>0) {
		fprintf(stderr, "Warning: %d unused values in sample tuples.\n", iExcessValueQty);
		++rSes.iWarnings;
	}
	if(iExcessValueQty<0) {
		fprintf(stderr, "Error: %d values not found in sample tuples.\n", iExcessValueQty*-1);
		++rSes.iErrors;
	}
	/// allocate memory for tuple storage
	rSes.rLearn->dXInputs = (double*)malloc( (rSes.rLearn->iInputQty+1) * rSes.rLearn->lSampleQty * sizeof(double)); // scalar X input signal
		mCheckMallocWorked(rSes.rLearn->dXInputs)
	rSes.rLearn->dDOutputs=(double*)malloc( (rSes.rLearn->iOutputQty+1) * rSes.rLearn->lSampleQty * sizeof(double)); // scalar D correct output signal
		mCheckMallocWorked(rSes.rLearn->dDOutputs)
		rSes.lMemStructAlloc = rSes.lMemStructAlloc || RLEARNd; // flag existence of alllocation
	
	for (int s=0; s<rSes.rLearn->lSampleQty; ++s){ //iterate over the number of samples and malloc
		for (int k=0; k<=rSes.rLearn->iInputQty; ++k) // fill with uniform, bogus values
			//#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))
			rSes.rLearn->dXInputs[ IDX2C( k, s, rSes.rLearn->iInputQty+1 ) ]=-999.9;
		for (int k=0; k<=rSes.rLearn->iOutputQty; ++k) // fill with uniform, bogus values
			rSes.rLearn->dDOutputs[ IDX2C( k, s, rSes.rLearn->iOutputQty+1 ) ]=-888.8; 
		// parse and store sample values
		pch = strtok (cLines[lCurrentLine], " ,\t"); // get the first token on a line
		//fprintf(fShow, "%d: ", s);
		//if(rSes.bRInJMode){ // if flag for compatibility with older NN simulator is set XX
		//	for (int k=rSes.rLearn->iValuesPerLine; k>=1; --k){ // save it beginning with last position
		//		rSes.rLearn->dXInputs[ IDX2C( s, k, rSes.rLearn->iInputQty+1 ) ]=atof(pch); // convert and assign each value in a line
		//		pch = strtok (NULL, " ,\t");
		//	}
		//}
		//else{ // otherwise store things the usual way
			for (int k=1; k<=rSes.rLearn->iInputQty; ++k){ // save it beginning with position 1
				rSes.rLearn->dXInputs[ IDX2C( k, s, rSes.rLearn->iInputQty+1 ) ]=atof(pch); // convert and assign each value in a line
				//fprintf(fShow, "%s,%f,%d,%d\t", pch, atof(pch), k, IDX2C( k, s, rSes.rLearn->iInputQty+1) );
				pch = strtok (NULL, " ,\t");
			}
		for (int k=1; k<=rSes.rLearn->iOutputQty; ++k){
			rSes.rLearn->dDOutputs[ IDX2C( k, s, rSes.rLearn->iOutputQty+1 ) ]=atof(pch); // convert and assign each value in a line
			pch = strtok (NULL, " ,\t");
		}
		rSes.rLearn->dXInputs[ IDX2C( 0, s, rSes.rLearn->iInputQty+1 ) ]=0.0; // virtual input zero should always be zero
		rSes.rLearn->dDOutputs[ IDX2C( 0, s, rSes.rLearn->iOutputQty+1) ]=0.0; // output neuron zero should always produce sector 0 output
		free(cLines[lCurrentLine]);
		++lCurrentLine;
	}
	
	free(cLines[0]);
	if (iArchLineIdx) free(cLines[1]); // WEIRD MEMORY ERRORS? LOOK HERE XX
	// above line avoids double-freeing cLines[1] if it was used for a sample instead of the sample qty
	free(cLines);
 	return lLinesQty; // returns qty of lines read from file, not the same as quantity of samples
}


int cBarge::CurateSectorValue(struct rohanContext& rSes)
{mIDfunc /// compares sector qty to sample values for adequate magnitude
	int iOverK=0;
	// loop over samples for inputs
	for (int s=0; s<rSes.rLearn->lSampleQty; ++s){
		//fprintf( fShow, "%dX|", s);
		for (int i=0; i<=rSes.rLearn->iInputQty; ++i){
			if(rSes.rLearn->dXInputs[ IDX2C( i, s, rSes.rLearn->iInputQty+1 )]>=rSes.rNet->iSectorQty){
				// if any input values fall beyond the maximum sector value, alert and make recommendation
				fprintf(stderr, "Error: Sample #%d has value that exceeds sector qty %d; suggest increasing to %d!\n",
					s, rSes.rNet->iSectorQty, static_cast<int>(floor(rSes.rLearn->dXInputs[ IDX2C( i, s, rSes.rLearn->iInputQty+1 ) ]*1.33)+1));
				++iOverK;
			}
		}	
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ // now loop over output values
			if(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 )]>=rSes.rNet->iSectorQty){
				// if any output values fall beyond the maximum sector value, alert and make recommendation
				fprintf(stderr, "Error: Sample #%d has value that exceeds sector qty %d; suggest increasing to %d!\n",
					s, rSes.rNet->iSectorQty, static_cast<int>(floor(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ]*1.33)+1));
				++iOverK;
			}
		}
	}
	if (iOverK)	{ // any out-of-bounds values are a fatal error
		++rSes.iErrors;
		return 0;
	}

	return rSes.rLearn->lSampleQty; // return number of samples veified within parameters
}


int cBarge::CompleteHostLearningSet(struct rohanContext& rSes)
{mIDfunc //allocate and fill arrays of complx values converted from scalar samples, all in host memory
	int iReturn=0;
	int IQTY, OQTY, INSIZED, OUTSIZED, INSIZECX, OUTSIZECX;
	
	//setup dimension values
	IQTY = rSes.rLearn->iInputQty+1 ;
	INSIZED = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(double) ;
	INSIZECX = rSes.rLearn->lSampleQty * ( IQTY ) * sizeof(cuDoubleComplex) ;
	OQTY = rSes.rLearn->iOutputQty+1; 
	OUTSIZED = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(double);
	OUTSIZECX = rSes.rLearn->lSampleQty * ( OQTY ) * sizeof(cuDoubleComplex);

	// allocate remainder of host scalar arrays
	rSes.rLearn->dYEval=(double*)malloc( OUTSIZED ); // scalar Y evaluated output signal
		mCheckMallocWorked(rSes.rLearn->dYEval)
	rSes.rLearn->dAltYEval=(double*)malloc( OUTSIZED ); // alt method scalar output
		mCheckMallocWorked(rSes.rLearn->dAltYEval)
	rSes.rLearn->dSqrErr=(double*)malloc( OUTSIZED ); // array for RMSE calculation, changed to OUTSIZED 1/8/12
		mCheckMallocWorked(rSes.rLearn->dSqrErr)
	// allocate host complex arrays
	rSes.rLearn-> cdcXInputs  =(cuDoubleComplex*)malloc( INSIZECX ); // cx X Input signal
		mCheckMallocWorked(rSes.rLearn->cdcXInputs)
	rSes.rLearn-> cdcDOutputs =(cuDoubleComplex*)malloc( OUTSIZECX ); // cx D desired output signal
		mCheckMallocWorked(rSes.rLearn->cdcDOutputs)
	rSes.rLearn-> cdcYEval    =(cuDoubleComplex*)malloc( OUTSIZECX ); // cx evaluated Y output signal
		mCheckMallocWorked(rSes.rLearn->cdcYEval)
	rSes.rLearn-> cdcAltYEval =(cuDoubleComplex*)malloc( OUTSIZECX ); // cx alt evaluation Y output
		mCheckMallocWorked(rSes.rLearn->cdcAltYEval)
		rSes.lMemStructAlloc = rSes.lMemStructAlloc || RLEARNcdc; // flag existence of allocated structs

	for(int S=0;S<rSes.rLearn->lSampleQty; S++){
		for (int I=0;I< IQTY ; I++){
			rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )]
				= ConvScalarCx( rSes, rSes.rLearn-> dXInputs [IDX2C( I, S, IQTY )] ); // convert scalar inputs on host
		}
		for (int O=0;O< OQTY ; O++){
			rSes.rLearn-> dYEval      [IDX2C(  O, S, OQTY )] = S; 
			rSes.rLearn-> dAltYEval	  [IDX2C(  O, S, OQTY )] = -S;
			rSes.rLearn-> dSqrErr	  [IDX2C(  O, S, OQTY )] = O;
			rSes.rLearn-> cdcDOutputs [IDX2C(  O, S, OQTY )]
				= ConvScalarCx( rSes, rSes.rLearn->dDOutputs[IDX2C(  O, S, OQTY )] ); // convert cx desired outputs
			rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].x = O; 
			rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].y = S; 
			rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].x = -1*O;
			rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].y = -1*S;
		}
	}
	printf("Sample set size = %d\n", 8*3*OUTSIZED + 16*INSIZECX + 16*3*OUTSIZECX);
	
	return iReturn;
}


int cBarge::DoPrepareNetwork(struct rohanContext& rSes)
{mIDfunc /// sets up network poperties and data structures for use
	int iReturn=0;
	// on with it
	
	cuMakeNNStructures(rSes); // allocates memory and populates network structural arrays
	iReturn=BinaryFileHandleRead(rSes.sWeightSet, &rSes.rNet->fileInput);
	// file opening and reading are separated to allow for streams to be added later
	if (iReturn) {
		int lWeightsRead=cuNNLoadWeights(rSes, rSes.rNet->fileInput);
			if (lWeightsRead) printf("Parsed and assigned %d complex weights from %s\n", lWeightsRead, rSes.sWeightSet);
			else {
				fprintf(stderr, "Error: No Weights Read by cuNNLoadWeights\n");
				++rSes.iErrors;
				//printf("Waiting on keystroke...\n"); _getch(); return iReturn;
			}
	}
	else { // can't open, user random weights
		printf("Can't open %s, using random weights.\n", rSes.sWeightSet);
		cuRandomizeWeightsBlock(rSes); // populate network with random weight values
	}

// record fixed-length accumulation 
	struct rohanNetwork * rnSrc; //, * rnDest ;
	struct rohanLayer * rlSrc;
	int LQTY, LLAST, LSIZE; //, SECSIZE;
	rnSrc=(rSes.rNet);
	int SIZE = sizeof(*rSes.rNet);
	
	LQTY = rnSrc->iLayerQTY ; 
		LLAST = LQTY - 1 ;
	LSIZE = sizeof(rohanLayer) * LQTY ;
	//kind=cudaMemcpyHostToDevice;
	
	int DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE;
	NQTY = rnSrc->rLayer[0].iNeuronQty + 1 ; // neurons = outgoing signals
	NSIZE = NQTY * sizeof(cuDoubleComplex) ;
	rlSrc=&(rSes.rNet->rLayer[0]);
	
	// blank original values
	for (int i = 0; i<MAXWEIGHTS; ++i) rSes.rNet->Wt[i]=cdcZero;
	for (int i = 0; i<MAXNEURONS; ++i) rSes.rNet->Deltas[i]=rSes.rNet->Signals[i]=cdcZero;
	rSes.rNet->iNeuronQTY[0]=rSes.rLearn->iInputQty+1; // initialize with inputs for Layer Zero
	rSes.rNet->iDendrtOfst[0]=rSes.rNet->iDendrtQTY[0]=rSes.rNet->iNeuronOfst[0]=rSes.rNet->iWeightOfst[0]=rSes.rNet->iWeightQTY[0]=0;

	//printf("layer %d molded and filled?\n", 0);

	//for (int L=1; L<=LLAST; ++L){
	for (int L=1; L<MAXLAYERS; ++L){
		if (L<=LLAST){
			//setup dimension values
			DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = weights
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
			rlSrc=&(rSes.rNet->rLayer[L]);
		}else{
			// fill out unused layers with empty structures
			DQTY=rSes.rNet->iNeuronQTY[L-1]; //	dendrites = previous layer's neurons
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY=1; // neuonrs limited to Nzero
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = neurons * dendrites
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
		}
	
// track fixed-length accumulation - should move to net setup
		//rSes.rNet->iDendrtOfst[L]=rSes.rNet->iDendrtOfst[L-1]+rSes.rNet->iDendrtQTY[L-1];
		//rSes.rNet->iNeuronOfst[L]=rSes.rNet->iNeuronOfst[L-1]+rSes.rNet->iNeuronQTY[L-1];
		//rSes.rNet->iWeightOfst[L]=rSes.rNet->iWeightOfst[L-1]+rSes.rNet->iWeightQTY[L-1];
		rSes.rNet->iNeuronOfst[L] = rSes.rNet->iNeuronOfst[L-1] + ((7+rSes.rNet->iNeuronQTY[L-1])/8)*8;
		rSes.rNet->iWeightOfst[L] = rSes.rNet->iWeightOfst[L-1] + ((7+rSes.rNet->iWeightQTY[L-1])/8)*8;
		rSes.rNet->iDendrtQTY[L]=DQTY;
		rSes.rNet->iNeuronQTY[L]=NQTY;
		rSes.rNet->iWeightQTY[L]=WQTY;
		if(rSes.rNet->iWeightOfst[L]+rSes.rNet->iWeightQTY[L] > MAXWEIGHTS){
			++rSes.iErrors;
			fprintf(stderr, "MAXIMUM WEIGHTS EXCEEDED at layer %d!\n", L);
		}
		if(rSes.rNet->iNeuronOfst[L]+rSes.rNet->iNeuronQTY[L] > MAXNEURONS){
			++rSes.iErrors;
			fprintf(stderr, "MAXIMUM NEURONS EXCEEDED at layer %d!\n", L);
		}
		// copy each layer's weights into the fixed-length array beginning at an offset
		if (L<=LLAST){
			for(int i=0; i<WQTY; ++i)
				rSes.rNet->Wt[i+rSes.rNet->iWeightOfst[L]]=rlSrc->Weights[i];
		}else{
			for(int i=1; i<WQTY; ++i)
				rSes.rNet->Wt[i+rSes.rNet->iWeightOfst[L]]=cdcZero;
			rSes.rNet->Wt[0+rSes.rNet->iWeightOfst[L]]=cdcIdentity;
		}
		rSes.rNet->dINV_S[L]=1.0/(double)DQTY; //printf("%d %d %f\n", L, DQTY, rSes.rNet->dINV_S[L]);
	}
	rSes.rNet->dINV_S[0]=1.0; // setup final fixed-length pointer

	// store pointers to dev global structures
	cudaGetSymbolAddress( (void**)&rSes.rNet->gWt, "devNet");
		mCheckCudaWorked
	rSes.rNet->gDeltas=rSes.rNet->gSignals=rSes.rNet->gWt;
	rSes.rNet->gDeltas += offsetof(rohanNetwork, Deltas);
	rSes.rNet->gSignals += offsetof(rohanNetwork, Signals);
	rSes.rNet->gWt += offsetof(rohanNetwork, Wt);
	iReturn=cuSectorTableMake(rSes); // fill the table with values
	if (iReturn==0) {
		printf("Out of Memory in cuSectorTableMake\n");
		++rSes.iErrors;
	}
	return iReturn;
}


int cBarge::LayersToBlocks(struct rohanContext& rSes) //, struct rohanNetwork& Net)
{mIDfunc /// moves weight values from old layer structures to new block structures
	// record fixed-length accumulation 
	struct rohanNetwork * rnSrc; //, * rnDest ;
	struct rohanLayer * rlSrc;
	int iReturn=0;
	int LQTY, LLAST, LSIZE; //, SECSIZE;
	rnSrc=(rSes.rNet);
	int SIZE = sizeof(*rSes.rNet);
	
	LQTY = rnSrc->iLayerQTY ; 
		LLAST = LQTY - 1 ;
	LSIZE = sizeof(rohanLayer) * LQTY ;
	//kind=cudaMemcpyHostToDevice;
	
	int DQTY, NQTY, WQTY, DSIZE, NSIZE, WSIZE;
	NQTY = rnSrc->rLayer[0].iNeuronQty + 1 ; // neurons = outgoing signals
	NSIZE = NQTY * sizeof(cuDoubleComplex) ;
	rlSrc=&(rSes.rNet->rLayer[0]);
	
	//// blank original values
	//for (int i = 0; i<MAXWEIGHTS; ++i) rSes.rNet->Wt[i]=cdcZero;
	//for (int i = 0; i<MAXNEURONS; ++i) rSes.rNet->Deltas[i]=rSes.rNet->Signals[i]=cdcZero;
	//rSes.rNet->iNeuronQTY[0]=rSes.rLearn->iInputQty+1; // initialize with inputs for Layer Zero
	//rSes.rNet->iDendrtOfst[0]=rSes.rNet->iDendrtQTY[0]=rSes.rNet->iNeuronOfst[0]=rSes.rNet->iWeightOfst[0]=rSes.rNet->iWeightQTY[0]=0;

	//printf("layer %d molded and filled?\n", 0);

	//for (int L=1; L<=LLAST; ++L){
	for (int L=1; L<MAXLAYERS; ++L){
		if (L<=LLAST){
			//setup dimension values
			DQTY = rnSrc->rLayer[L].iDendriteQty + 1 ; // dendrites = incoming signals
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY = rnSrc->rLayer[L].iNeuronQty + 1 ; // neurons = outgoing signals
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = weights
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
			rlSrc=&(rSes.rNet->rLayer[L]);
		}else{
			// fill out unused layers with empty structures
			DQTY=rSes.rNet->iNeuronQTY[L-1]; //	dendrites = previous layer's neurons
			DSIZE = DQTY * sizeof(cuDoubleComplex) ;
			NQTY=1; // neuonrs limited to Nzero
			NSIZE = NQTY * sizeof(cuDoubleComplex) ;
			WQTY = DQTY * NQTY ; // weights = neurons * dendrites
			WSIZE = WQTY * sizeof(cuDoubleComplex) ;
		}
	
// track fixed-length accumulation - should move to net setup
		//rSes.rNet->iDendrtOfst[L]=rSes.rNet->iDendrtOfst[L-1]+rSes.rNet->iDendrtQTY[L-1];
		rSes.rNet->iNeuronOfst[L]=rSes.rNet->iNeuronOfst[L-1]+rSes.rNet->iNeuronQTY[L-1];
		rSes.rNet->iWeightOfst[L]=rSes.rNet->iWeightOfst[L-1]+rSes.rNet->iWeightQTY[L-1];
		rSes.rNet->iDendrtQTY[L]=DQTY;
		rSes.rNet->iNeuronQTY[L]=NQTY;
		rSes.rNet->iWeightQTY[L]=WQTY;
		if(rSes.rNet->iWeightOfst[L]+rSes.rNet->iWeightQTY[L] > MAXWEIGHTS){
			++rSes.iErrors;
			fprintf(stderr, "MAXIMUM WEIGHTS EXCEEDED at layer %d!\n", L);
		}
		if(rSes.rNet->iNeuronOfst[L]+rSes.rNet->iNeuronQTY[L] > MAXNEURONS){
			++rSes.iErrors;
			fprintf(stderr, "MAXIMUM NEURONS EXCEEDED at layer %d!\n", L);
		}
		// copy each layer's weights into the fixed-length array beginning at an offset
		if (L<=LLAST){
			for(int i=0; i<WQTY; ++i)
				rSes.rNet->Wt[i+rSes.rNet->iWeightOfst[L]]=rlSrc->Weights[i];
		}else{
			for(int i=1; i<WQTY; ++i)
				rSes.rNet->Wt[i+rSes.rNet->iWeightOfst[L]]=cdcZero;
			rSes.rNet->Wt[0+rSes.rNet->iWeightOfst[L]]=cdcIdentity;
		}
	}
	return iReturn;

}

int cBarge::LetWriteWeights(struct rohanContext& rSes)
{mIDfunc/// dump ASCII weight values to disk
	int lReturn;
	char sLog[255], sFileName[255];
	FILE *fileOutput; // File handle for output
	
	lReturn=AsciiFileHandleWrite(rSes.sRohanVerPath, "weightdump.txt", &fileOutput);
	AsciiWeightDump(rSes, fileOutput); 
	sprintf(sFileName, "%sWeight%d.txt", rSes.sSesName, (int)(rSes.dRMSE*100) ); //(int)(rSes.dDevRMSE*100)
	
	sprintf(sLog, "%d weights writen to %s\\%s", lReturn, rSes.sRohanVerPath, sFileName ); // document success and filename
	RLog(rSes, sLog);
	
	return lReturn;
}

int cBarge::LetWriteEvals(struct rohanContext& rSes, struct rohanLearningSet& rLearn)
{mIDfunc/// saves evaluated output values to disk
	int lReturn;
	FILE *fileOutput; // File handle for output
	char sFileAscii[255]; //="DefaultSession";
	
	sprintf(sFileAscii,"%s%d%s",rSes.sSesName, (int)(rSes.dRMSE*100), "Evals.txt"); // do not exceed 254 char file name
	lReturn=AsciiFileHandleWrite(rSes.sRohanVerPath, sFileAscii, &fileOutput);
	if(lReturn){
		for(int s=0; s<rSes.lSampleQtyReq; ++s){
			if(rSes.iSaveInputs){
				for(int i=1; i<=rLearn.iInputQty; ++i) // write inputs first
					fprintf(fileOutput, "%3.f, ", rLearn.dXInputs[IDX2C(i,s,rLearn.iInputQty+1)]);
			}
			if(rSes.iSaveOutputs){
				for(int i=1; i<=rLearn.iOutputQty; ++i) // write desired outputs second
					fprintf(fileOutput, "%7.3f, ", rLearn.dDOutputs[IDX2C(i,s,rLearn.iOutputQty+1)]);
			}
			for(int i=1; i<=rLearn.iOutputQty; ++i){ // write yielded outputs third
				fprintf(fileOutput, "%#7.3f", rLearn.dYEval[IDX2C(i,s,rLearn.iOutputQty+1)]);
				if (i<rLearn.iOutputQty)
					fprintf(fileOutput, ", "); // only put commas between outputs, not after
			}
			if(rSes.iSaveSampleIndex){ // write sample indexes last
				fprintf(fileOutput, ", %d", s);
			}
			fprintf(fileOutput, "\n"); // end each line with a newline
		}
		fclose(fileOutput);
	}

	if (lReturn)
		return rSes.lSampleQtyReq; // return number of sample evals recorded
	else
		return 0;
}

int cBarge::ShowDiagnostics()
{mIDfunc
	int iReturn=1;

	printf("cBarge diagnostics: ");
	if(Team==NULL)
		printf("No team!\n", iReturn=0);
	if(Drover==NULL)
		printf("No drover!\n", iReturn=0);
	if(rSes->rLearn->cdcXInputs==NULL)
		printf("No complex inputs at host!\n", iReturn=0);
	if(rSes->rLearn->dDOutputs==NULL)
		printf("No scalar outputs at host!\n", iReturn=0);
	if(rSes->rLearn->cdcDOutputs==NULL)
		printf("No complex outputs at host!\n", iReturn=0);
	if (rLearn==NULL)
		printf("No rLearn structure!\n", iReturn=0);
	else
		//printf("Holding %d samples w/ %d inputs, %d output(s)\n", *rLearn->lSampleQty, *rLearn->iInputQty, *rLearn->iOutputQty);
		printf("Barge is holding %d samples w/ %d inputs, %d output(s).\n", rSes->rLearn->lSampleQty, rSes->rLearn->iInputQty, rSes->rLearn->iOutputQty);
	
	return iReturn;
}


void cBarge::RLog(struct rohanContext& rSes, char * sLogEntry)
{mIDfunc // logs strings describing events, preceeded by the local time
	using namespace boost::posix_time; 
    ptime now = second_clock::local_time(); //use the clock 
    sLogEntry=strtok(sLogEntry, "\n"); // trim any trailing chars
	*(rSes.ofsRLog) << now << " " << sLogEntry  << endl;
	if(!rSes.bConsoleUsed)
		*(rSes.ofsHanLog) << "#\t " << sLogEntry  << endl; // all entries repeated as comments in .han file if any.
}

void cBarge::HanReport(struct rohanContext& rSes, char * sLogEntry)
{mIDfunc // logs strings describing events
	sLogEntry=strtok(sLogEntry, "\n"); // trim any trailing chars
	*(rSes.ofsHanLog) << sLogEntry  << endl;
}

int cBarge::DoCuFree(struct rohanContext &rSes)
{mIDfunc/// free allocated memory for all structures
	
	cuFreeNNTop(rSes); // free network topology structures
	cuFreeLearnSet(rSes); // free learning set structures
	
	return 0;
}


int cBarge::cuFreeNNTop(struct rohanContext &rSes)
{mIDfunc/// frees data structures related to network topology
	try {	int iFreed=0;
		if (rSes.lMemStructAlloc && RNETbdry) {//check
				if (rSes.rNet->cdcSectorBdry!=NULL){
					free( rSes.rNet->cdcSectorBdry );
				}
			else
				printf("can't free( rSes.rNet->cdcSectorBdry)\n");

			rSes.lMemStructAlloc = rSes.lMemStructAlloc && !RNETbdry; 
			++iFreed;
		}
		// layer components
		if (rSes.lMemStructAlloc && RNETlayers){
			free( rSes.rNet->rLayer[0].ZOutputs ); // Layer Zero has no need of weights!
			for (int i=1; i < rSes.rNet->iLayerQTY; ++i){ 
				struct rohanLayer& lay=rSes.rNet->rLayer[i];
			
				free( lay.Weights ); // free the weights
				free( lay.Deltas ); // free the backprop areas
				free( lay.XInputs ); // free the inputs
				free( lay.ZOutputs ); // free the outputs
			}
			free( rSes.rNet->rLayer ); // free empty layers
			rSes.lMemStructAlloc = rSes.lMemStructAlloc && !RNETlayers; 
			++iFreed;
		}
		if(iFreed)
			printf("Network structures freed.\n");
		return iFreed;
	}
	catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return -1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
		return -1;
    }
}


int cBarge::cuFreeLearnSet(struct rohanContext &rSes)
{mIDfunc/// free the learning set of samples
	try {
		int iFreed=0;
		if (rSes.lMemStructAlloc && RLEARNd) { //check
			free( rSes.rLearn->dXInputs ); 
			free( rSes.rLearn->dDOutputs );
			free( rSes.rLearn->dYEval ); 
			free( rSes.rLearn->dAltYEval ); 
			free( rSes.rLearn->dSqrErr );
			rSes.lMemStructAlloc = rSes.lMemStructAlloc && !RLEARNd; 
			++iFreed;
		}
		
		if (rSes.lMemStructAlloc && RLEARNcdc){ // check
			free( rSes.rLearn->cdcXInputs ); 
			free( rSes.rLearn->cdcDOutputs ); 
			free( rSes.rLearn->cdcYEval ); 
			free( rSes.rLearn->cdcAltYEval );
			rSes.lMemStructAlloc = rSes.lMemStructAlloc && !RLEARNcdc; 
			++iFreed;
		}
		if(iFreed)
			printf("Network structures freed.\n");
		return iFreed;
	}
	catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return -1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
		return -1;
    }
}
