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

// another helper 
char cBarge::sep_to_space(char c){
  return c == ',' || c == '<' || c == '>' ? ' ' : c;
}


//#include <boost/filesystem.hpp>
//using namespace boost::filesystem;


extern int gDebugLvl, gTrace;


//////////////// class cBarge begins ////////////////

cBarge::cBarge(rohanContext& rSes)
{
	rSes.Barge=this;
}

void cBarge::ShowMe()
{
	//ShowMeSes(* rSes, false);
	printf("Am stout barje.\n");
}


int cBarge::SetContext( rohanContext& rC)
{/// enables pointer access to internal objects
	rSes = &rC;
	rLearn = rC.rLearn;
	rNet = rC.rNet;
	//Barge=rC.Barge;
	Drover=rC.Drover;
	Ramp=rC.Ramp;
	Team=rC.Team;

	return 0;
}

void cBarge::ResetSettings(struct rohanContext& rSes, struct rohanNetwork& rNet, struct rohanLearningSet& rLearn)
{mIDfunc /// sets initial and default value for globals and settings

	// ERRORS and progress tracking
	gTrace=0; 
	gDebugLvl=0; 
	rSes.iWarnings=0; rSes.iErrors=0; 
	rSes.lMemStructAlloc=0;
	// eval related
	rSes.iSaveInputs=0 /* include inputs when saving evaluations */;
	rSes.iSaveOutputs=0 /* include desired outputs when saving evaluations */;
	rSes.iSaveSampleIndex=0 /* includes sample serials when saving evaluations */;
	rSes.iOutputFocus=1 /*! which output is under consideration (0=all) */;
	rSes.iEvalBlocks=128; 
	rSes.iEvalThreads=128; 
	// hardware related
	int iMasterCalcHw=-2;
	double dMasterCalcVer=0.0;
	int deviceCount=-1;
	// input handling
	rSes.iContReadMode=1; // default to discrete=1 (0=continuous)
	rSes.bConsoleUsed=false;
	rSes.bRInJMode=false; 
	rSes.bRMSEon=true; 
	strcpy(rSes.sLearnSet, "LS-NOT-SPECIFIED");
	strcpy(rSes.sWeightSet, "WTS-NOT-SPECIFIED");
	SetVerPath(rSes);
	sprintf(rSes.sDefaultConfig, "%s\\rohan.roh", rSes.sRohanVerPath);

// learning related
	rSes.lSamplesTrainable=-1;
	rSes.iOutputFocus=1;
	rSes.iBpropBlocks=1; 
	rSes.iBpropThreads=32; 
	rSes.dLastRmseB=999.9;
	rSes.dBestRmseB=999.9;
	rSes.dLastRmseG=999.9;
	rSes.dBestRmseG=999.9;
	rSes.dLastRmseS=999.9;
	rSes.dBestRmseS=999.9;
	rSes.dLastRmse3=999.9;
	rSes.dBestRmse3=999.9;
	rSes.dLastRMSE=999.9;
	rSes.dBestRMSE=999.9;
	rSes.dPrevLastRMSE=999.9;
	rSes.dPrevBestRMSE=999.9;
	rSes.dMAX=0.0;
	rSes.dTargetRMSE=0.0;
	rSes.iEpochLength=1000; 
	rSes.iEpochQty=100; 
	rSes.cEngagedModel=' ';
	// network related
	rSes.iContActivation=true; // default to true (quicker)
	rSes.iSectorQty=0;
	rSes.iFirstHiddenSize=0;
	rSes.iSecondHiddenSize=0;
	rSes.iLayerQty=0;
	// record keeping
	rSes.iLoggingEnabled=false;
	//rSes.ofsRLog=NULL;
	//rSes.ofsHanLog=NULL;
	//strcpy(rSes.sRohanVerPath, "NO-PATH-SET");
	strcpy(rSes.sSesName, "NO-TAG-SET");
	rSes.lSampleQty=-1;
	rSes.iInputQty=-1; 
	rSes.iOutputQty=-1;
	rSes.lSampleQtyReq=-1;
	
	//rLearn
	rLearn.lSampleQty=0;
	rLearn.iValuesPerLine=0;
	rLearn.iInputQty=0;
	rLearn.iOutputQty=0;
	rLearn.iContInputs=0; //default
	rLearn.iContOutputs=0; //default
	rLearn.lSampleIdxReq=-1;

	//rNet ?

}


int cBarge::SetProgOptions(struct rohanContext& rSes, int argc, _TCHAR * argv[])
{mIDfunc /// Declare the supported options.
	int iReturn=1 ; // returns 0 if version or help is specified, or if insufficient paramters for automated operation are given, 1 otherwise
		// console based operations not supported atm
    try {
		string config_file, eval_string, learn_string, net_arch, sample_file, tseries_string, conty_string, tag_session, weight_file;
		
		// Declare a group of options that will be 
		// allowed only on command line
		po::options_description generic("Generic options");
		generic.add_options()
			("version,v", "print version string")
			("help,h", "produce help message")
			("config,c", po::value<string>(&config_file)->default_value(rSes.sDefaultConfig), "name of a file of a configuration.")
            ;
		    
		// Declare a group of options that will be 
		// allowed both on command line and in
		// config file
		po::options_description config("Configuration");
		config.add_options()
            ("network,n", po::value<string>(&net_arch), "network sectors, inputs, 1st hidden layer, 2nd hidden layer, outputs")
			("tseries,z", po::value<string>(&tseries_string), "parameters for treating samples file as a time series")
			("samples,s", po::value<string>(&sample_file), "text file containing sample input-output sets")
			("conty,y", po::value<string>(&conty_string), "continuity/discreteness of inputs, outputs, internal activation, final activation")
			("weights,w", po::value<string>(&weight_file), ".wgt file containing complex weight values")
			("learn,l", po::value<string>(&eval_string), "train in pursuit of target RMSE given MAX criterion")
			("eval,e", po::value<string>(&learn_string), "evaluate samples and report")
			("tag,t", po::value<string>(&tag_session)->default_value("DefaultSession"), "tag session with an identifying string")
			("help,h", "produce help message")
			;

		// Hidden options, will be allowed both on command line and
		// in config file, but will not be shown to the user.
		po::options_description hidden("Hidden options");
		hidden.add_options()
			("include-path,I", po::value< vector<string> >()->composing(), "include path") // XX
			("input-file", po::value< vector<string> >(), "input file") //XX
			; 

		po::options_description cmdline_options;
		cmdline_options.add(generic).add(config).add(hidden);

		po::options_description config_file_options;
		config_file_options.add(config).add(hidden);

		po::options_description visible("Allowed options");
		visible.add(generic).add(config);
		//odpKeptVisible = &visible; // ptr to visible kept to show in case of bogus startup

        po::positional_options_description p;
        p.add("input-file", -1);
        
        //po::variables_map vm;
        store(po::command_line_parser(argc, argv).
              options(cmdline_options).positional(p).run(), vm);
        notify(vm);
        
        if (vm.count("help")) {
            cout << visible << "\n";
            return 0;
        }

        if (vm.count("version")) {
            cout << VERSION << ".\n";
			return 0;
        }

		ifstream ifs(config_file.c_str());
        if (!ifs)
        {
            cout << "can not open config file: " << config_file << "\n";
			cout << visible << "\n";
            return 0;
        }
        else
        {
            store(parse_config_file(ifs, config_file_options), vm);
            notify(vm);
        }
    
        if (vm.count("help")) {
            cout << visible << "\n";
            return 0;
        }

		//if (vm.count("include-path"))
  //      {
  //          cout << "Include paths are: " 
  //               << vm["include-path"].as< vector<string> >() << "\n";
  //      }

  //      if (vm.count("input-file"))
  //      {
  //          cout << "Input files are: " 
  //               << vm["input-file"].as< vector<string> >() << "\n";
  //      }
	
		//if (vm.count("config"))
		//{
		//	cout << "Config files are: ";
		//	cout << vm["config"].as<string>() << "\n";
		//}

		if (vm.count("network"))
        {
            //cout << "Network architecture is: ";
            //cout << vm["network"].as<string>() << "\n";
        }
		else {
			cout << "No network specified!\n";
			return 0;
		}

		if (vm.count("samples"))
        {
            //cout << "Samples file is : ";
			//cout << vm["samples"].as<string>() << "\n";
		}
		else {
			cout << "No samples specified!\n";
			return 0;
		}

		//if (vm.count("weights"))
  //      {
		//	//cout << "Weights file is : ";
		//	//cout << vm["weights"].as<string>() << "\n";
  //      }
		//else cout << "No weights specified, default to random values.\n";

		//if (vm.count("eval")==0 || vm.count("learn")==0)
		//{
		//	cout << "No directives given.\n";
		//	++rSes.iWarnings;
		//}

	}
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
		++rSes.iErrors;
        return 0;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
		++rSes.iErrors;
		return 0;
    }

    return iReturn;
}


int cBarge::BeginLogging(struct rohanContext& rSes)
{mIDfunc /// Prepares logging activities
	int iReturn=1;
	char sLog[255], sPath[MAX_PATH], sHanPath[MAX_PATH];

	SetVerPath(rSes); // set up default path
	GetTagOpt(rSes); //get tag setting, sanitized
	if(DirectoryEnsure(rSes.sRohanVerPath)){
		using namespace boost::posix_time; 
		ptime now = second_clock::local_time(); //use the clock
		std::ostringstream osNow ;
		osNow << now;
		// establish Rlog
		sprintf(sPath, "%s\\RohanLog.txt", rSes.sRohanVerPath); // sPath has full rlog pathname
		rSes.ofsRLog=new ofstream(sPath, std::ios::app|std::ios::out); 
		*(rSes.ofsRLog) << "\tSTART Rohan v" << VERSION << " Neural Network Simulator - " << AUTHORCREDIT << "\n";
		// establish .han file 
		sprintf( sHanPath, "%s\\%s.han", rSes.sRohanVerPath, rSes.sSesName); //sHanPath has fill hanlog pathname
		rSes.ofsHanLog=new ofstream(sHanPath, std::ios::out); 
		//rSes.ofsHanLog=new ofstream(sHanPath, std::ios::app|std::ios::out); 
		*(rSes.ofsHanLog) << "#\t" << now << "\tSTART Rohan v" << VERSION << " Neural Network Simulator - " << AUTHORCREDIT << "\n";
		rSes.iLoggingEnabled=true;
		RLog(rSes, USERF, "Logging initiated");
		{
			sprintf(sLog, " using %s\\", rSes.sRohanVerPath);
			RLog(rSes, USERF+ADMINF, sLog);
			sprintf(sLog, " using %s", vm["config"].as<string>().c_str() );
			RLog(rSes, USERF+ADMINF, sLog);
			sprintf(sLog, " begining session %s", rSes.sSesName);
			RLog(rSes, USERF+ADMINF, sLog);
		}
#ifdef _DEBUG
		// establish bucket files
		//rSes.Ramp->GetFileHandle(rSes.sRohanVerPath, "DevBucket.txt", 'w', 'a', &(rSes.deviceBucket));
		//fprintf(rSes.deviceBucket, "%s\tSTART Rohan v%s Neural Network Simulator\n", osNow.str().c_str(), VERSION);
		//rSes.Ramp->GetFileHandle(rSes.sRohanVerPath, "HostBucket.txt", 'w', 'a', &(rSes.hostBucket));
		//fprintf(rSes.hostBucket, "%s\tSTART Rohan v%s Neural Network Simulator\n", osNow.str().c_str(), VERSION);
#endif
		RLog(rSes, 0, __FUNCTION__);
	}
	else {
		sprintf(sLog, "Directory %s could not be created", rSes.sRohanVerPath); RLog(rSes, USERF+ERRORF, sLog);
	}

	return iReturn;
}

int cBarge::GetTagOpt(struct rohanContext& rSes)
{mIDfunc///parses tag program_option 
	int iReturn=1;
//# the tag value is a name used to identify the files Rohan generates during a session. Rohan's reply will be NahorNahor.han, reports will be NahorNahor937Evals.txt, etc.
//tag=NahorNahor
	// tag has default value, so it can always be called upon
	strcpy( rSes.sSesName, vm["tag"].as<string>().c_str() ); 
	Ramp->CharRemover(rSes.sSesName,'"');
	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}

int cBarge::GetEvalOpt(struct rohanContext& rSes)
{mIDfunc///parses eval directive program_option 
	int iReturn=1; vector<int> v; char sLog[255];
//# eval's value specification is <1/0 for numbering tuples in sequence, 1/0 
//# for including tuple's input values, 1/0 for including tuple' output values, 
//# qty of tuples to include in evaluation/prediction, beginning sample idx>. 
//# OPTIONAL
//eval=<1,1,1,10000,0>
	if(vm.count("eval")){
		if(VectorFromOption("eval", v, EVALPMQTY)){
			rSes.iSaveSampleIndex = v.at(0);
			rSes.iSaveInputs = v.at(1);
			rSes.iSaveOutputs = v.at(2);
			rSes.lSampleQtyReq = v.at(3);
			rSes.iEvalSkip = v.at(4);
		// other activities at cDrover::DoEvalOpt
			RLog(rSes, 0, __FUNCTION__);
		}
		else{
			RLog(rSes, GUIF, "eval=fail");
			sprintf(sLog, "bad eval directive: %s", vm["eval"].as<string>().c_str());
			RLog(rSes, WARNINGF, sLog);
		}
	}
	return iReturn;
}

int cBarge::GetTSeriesOpt(struct rohanContext& rSes)
{mIDfunc///parses samples specification program_option 
//("tseries,z", po::value<string>(&tseries_string), "parameters for treating samples file as a time series")
//tseries=<8000,0, 2,0, 1, 100,1>
	int iReturn=1;
	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
} 

int cBarge::GetSamplesOpt(struct rohanContext& rSes)
{mIDfunc///parses samples specification program_option 
	int iReturn=1;
//# samples' value specification = name of the text file that contains the comma, tab, or space delimited lines of samples
//samples=AirplanePsDN1W3S10k.txt
	if (vm.count("samples"))
		strcpy( rSes.sLearnSet, vm["samples"].as<string>().c_str() );
	Ramp->CharRemover(rSes.sLearnSet,'"');
	RLog(rSes, 0, rSes.sLearnSet);
	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
} 

int cBarge::GetContyOpt(struct rohanContext& rSes)
{mIDfunc///parses samples specification program_option 
//("conty,y", po::value<string>(&conty_string), "continuity/discreteness of inputs, outputs, internal activation, final activation")
//conty=<1,0,1,0>
	int iReturn=1;
	vector<int> v;
	char sLog[255];
	
	if (vm.count("conty")){
		if(VectorFromOption("conty", v, CONTYPMQTY) ){ // all params present, hopefully valid
			// perform Contying here, other activities at cDrover::DoContyOpt
			rSes.rLearn->iContInputs= v.at(0);
			rSes.rLearn->iContOutputs= v.at(1);
			rSes.iContActivation = v.at(2);
		}
		else{ // missing or extra parameters
			RLog(rSes, GUIF, "conty=fail");
			sprintf(sLog, "bad conty specification: %s", vm["conty"].as<string>().c_str());
			RLog(rSes, WARNINGF, sLog);
		}
	}
	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
} 

int cBarge::GetWeightsOpt(struct rohanContext& rSes)
{mIDfunc///parses weights specification program_option 
	int iReturn=1;
	//# weights' value  specification = name of the file that contains the binary representation of the neural weights
	//weights=AirplanePsDN1W3S10kRMSE876.wgt
	if (vm.count("weights")){
		strcpy( rSes.sWeightSet, vm["weights"].as<string>().c_str() );
		Ramp->CharRemover(rSes.sWeightSet, '"');
	}
	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}

int cBarge::GetLearnOpt(struct rohanContext& rSes)
{mIDfunc///parses learn directive program_option 
//# learn's value specification is < target RMSE, MAX, # of samples used, warp 
//# factor, iterations per epoch, qty of epochs for learning>. Warp 1 is always
//# safe, use other values at your own risk!! Training is saved at the end of
//# each epoch; epoch durations depend on network and sample sizes.
//
//learn=<9.5,10,10000,1,500,20>
	int iReturn=1; vector<double> v; char sLog[255];

	if(vm.count("learn")) {
		if(VectorFromOption("learn", v, LEARNPMQTY) ){ // all params present, hopefully valid
			rSes.dTargetRMSE = v.at(0);
			rSes.dMAX = v.at(1);
			rSes.lSampleQtyReq = (int)v.at(2);
		if( rSes.lSampleQtyReq==0) // if specifying 0, set to all w/o comment
			rSes.lSampleQtyReq=rSes.lSampleQty;
		if( rSes.lSampleQtyReq<0 || rSes.lSampleQtyReq>rSes.lSampleQty ) {// if specifying less than none or more than all
			rSes.lSampleQtyReq=rSes.lSampleQty;// set request to all
			sprintf(sLog, "%s: Bad sample req value, set to %d", __FUNCTION__, rSes.lSampleQtyReq );
			RLog(rSes, WARNINGF+USERF, sLog);
		}
			rSes.iBpropThreads = (int)v.at(3) * 32; // Warp factor 1, Mr Sulu!
			rSes.iEpochLength = (int)v.at(4);
			rSes.iEpochQty = (int)v.at(5);
			// other activities at cDrover::DoLearnOpt
		}
		else{ // missing or extra parameters
			RLog(rSes, GUIF, "learn=fail");
			sprintf(sLog, "bad learn directive: %s", vm["learn"].as<string>().c_str());
			RLog(rSes, WARNINGF, sLog);
		}
	}
	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}


int cBarge::GetNetworkOpt(struct rohanContext& rSes, struct rohanNetwork& rNet)
{mIDfunc///parses network specification program_option 
	char sLog[255];
	vector<int> v;
	
	//# network's value specification is < # sectors, # inputs, size of first hidden layer or 0, size of second hidden layer or 0, # of outputs >
	//network=<384,9,36,0,1>
	if(VectorFromOption("network", v, 5)){
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
			//printf(":%d, %d, %d:", rSes.iFirstHiddenSize, rSes.iSecondHiddenSize, rSes.iOutputQty);
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
		ivTopoNN.push_back(rSes.iInputQty);
		if(rSes.iLayerQty==1)sprintf(rSes.sNetString, "%d.%d (%d)", rSes.iInputQty, rSes.iOutputQty, rSes.iSectorQty);
		if(rSes.iLayerQty==2){
			sprintf(rSes.sNetString, "%d.%d.%d (%d)", rSes.iInputQty, rSes.iFirstHiddenSize, rSes.iOutputQty, rSes.iSectorQty);
			ivTopoNN.push_back(rSes.iFirstHiddenSize);
		}
		if(rSes.iLayerQty==3){
			sprintf(rSes.sNetString, "%d.%d.%d.%d (%d)", rSes.iInputQty, rSes.iFirstHiddenSize, rSes.iSecondHiddenSize, rSes.iOutputQty, rSes.iSectorQty);
			ivTopoNN.push_back(rSes.iFirstHiddenSize);
			ivTopoNN.push_back(rSes.iSecondHiddenSize);
		}
		ivTopoNN.push_back(rSes.iOutputQty);

		ivOffsetK.clear(); ivOffsetQ.clear(); 
		ivOffsetK.push_back(0); ivOffsetQ.push_back(0);
		
		for (int LAY=1; LAY < (int)ivTopoNN.size(); ++LAY) { // add new layer size to previous offset
			ivOffsetK.push_back( ivOffsetK.at(LAY-1) + (0+ivTopoNN.at(LAY)) * (1+ivTopoNN.at(LAY-1)) ); // lack of neuron zero makes layers smaller
			ivOffsetQ.push_back( ivOffsetQ.at(LAY-1) + (1+ivTopoNN.at(LAY)) * (1+ivTopoNN.at(LAY-1)) );  // inclusion of neuron zero makes layers larger
		}
	
		RLog(rSes, 0, __FUNCTION__);
		return true;
	}
	else{
		RLog(rSes, GUIF, "network=fail");
		sprintf(sLog, "bad network spec: %s", vm["network"].as<string>().c_str());
		RLog(rSes, ERRORF, sLog);
		return false;
	}
}

void cBarge::GetOptions(struct rohanContext& rSes, struct rohanNetwork& rNet, struct rohanLearningSet& rLearn)
{mIDfunc// Get*Opt functions should only update rSes members according to vm values
	GetNetworkOpt(rSes, *rSes.rNet); // needs to change
	GetEvalOpt(rSes);
	GetTSeriesOpt(rSes);
	GetSamplesOpt(rSes);
	GetContyOpt(rSes);
	GetWeightsOpt(rSes);
	GetLearnOpt(rSes);
	RLog(rSes, 0, __FUNCTION__);
}

int cBarge::GetHdweSet(struct rohanContext& rSes)
{mIDfunc///parses eval directive program_option 
	int iReturn=1;
	char sLog[255];

	Team->CUDAverify(rSes);
		if (rSes.dMasterCalcVer>=2.0){ // assigns .dMasterCalcVer, .deviceProp.major, .deviceCount
			cutilSafeCall( cudaSetDevice(rSes.iMasterCalcHw) ); /// all cuda calls to run on first device of highest compute capability device located
			sprintf(sLog, "CUDA present, device %d w/Compute Capability %.1f selected.", rSes.iMasterCalcHw, rSes.dMasterCalcVer);
			RLog(rSes, 0, sLog);
		}
		else {
			if (rSes.dMasterCalcVer>1.0)
				sprintf(sLog, "Warning: CUDA hardware below Compute Capability 2.0.");
			else
				sprintf(sLog, "Warning: No CUDA hardware or no CUDA functions present.");
			rSes.iMasterCalcHw=-1;
			RLog(rSes, WARNINGF, sLog);
			iReturn=0;
		}
	RLog(rSes, 0, __FUNCTION__);

	return iReturn;
}

void cBarge::PrepareContext(struct rohanContext& rSes)
{mIDfunc
	GetHdweSet(rSes);
	RLog(rSes, 0, __FUNCTION__);
}

void cBarge::PrepareNetSettings(struct rohanContext& rSes, struct rohanNetwork& rNet)
{mIDfunc
	//char sLog[255];

		// NETWORK STRUCT
		rNet.iSectorQty=rSes.iSectorQty;
		rNet.kdiv2=rSes.iSectorQty/2; 
		//rNet.iLayerQTY=rSes.iLayerQty+1;
		rNet.iContActivation=rSes.iContActivation;
		rNet.dK_DIV_TWO_PI=rSes.iSectorQty/TWO_PI;
		rNet.two_pi_div_sect_qty=TWO_PI/rSes.iSectorQty;
		rNet.iLayerQTY = (rSes.iFirstHiddenSize ? 1:0) + (rSes.iSecondHiddenSize ? 1:0) + 2;
		// polylayer model
		//rSes.rNet->rLayer=(struct rohanLayer*)malloc(rSes.rNet->iLayerQTY * sizeof (struct rohanLayer)); //point to array of layers
		//	mCheckMallocWorked(rSes.rNet->rLayer)
		//if(rSes.iLayerQty==1){
		//	rSes.rNet->rLayer[1].iNeuronQty = rSes.iOutputQty;
		//	rSes.rNet->rLayer[1].iDendriteQty=rSes.iInputQty;
		//	rSes.rNet->rLayer[0].iNeuronQty = rSes.iInputQty;
		//	rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		//}
		//if(rSes.iLayerQty==2){
		//	rSes.rNet->rLayer[2].iNeuronQty = rSes.iOutputQty;
		//	rSes.rNet->rLayer[2].iDendriteQty=rSes.iFirstHiddenSize;
		//	rSes.rNet->rLayer[1].iNeuronQty = rSes.iFirstHiddenSize;
		//	rSes.rNet->rLayer[1].iDendriteQty=rSes.iInputQty;
		//	rSes.rNet->rLayer[0].iNeuronQty = rSes.iInputQty;
		//	rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		//}
		//if(rSes.iLayerQty==3){
		//	rSes.rNet->rLayer[3].iNeuronQty = rSes.iOutputQty;
		//	rSes.rNet->rLayer[3].iDendriteQty=rSes.iSecondHiddenSize;
		//	rSes.rNet->rLayer[2].iNeuronQty = rSes.iSecondHiddenSize;
		//	rSes.rNet->rLayer[2].iDendriteQty=rSes.iFirstHiddenSize;
		//	rSes.rNet->rLayer[1].iNeuronQty = rSes.iFirstHiddenSize;
		//	rSes.rNet->rLayer[1].iDendriteQty=rSes.iInputQty;
		//	rSes.rNet->rLayer[0].iNeuronQty = rSes.iInputQty;
		//	rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		//}
		// fixed layer model
		if(rSes.iLayerQty==1){
			rNet.OutputLayer.iNeuronQty = rSes.iOutputQty;
			rNet.OutputLayer.iDendriteQty=rSes.iInputQty;
			rNet.InputLayer.iNeuronQty = rSes.iInputQty;
			rNet.InputLayer.iDendriteQty=0; // layer zero has no dendrites
		}
		if(rSes.iLayerQty==2){
			rNet.OutputLayer.iNeuronQty = rSes.iOutputQty;
			rNet.OutputLayer.iDendriteQty=rSes.iFirstHiddenSize;
			rNet.FirstHiddenLayer.iNeuronQty = rSes.iFirstHiddenSize;
			rNet.FirstHiddenLayer.iDendriteQty=rSes.iInputQty;
			rNet.InputLayer.iNeuronQty = rSes.iInputQty;
			rNet.InputLayer.iDendriteQty=0; // layer zero has no dendrites
		}
		if(rSes.iLayerQty==3){
			rNet.OutputLayer.iNeuronQty = rSes.iOutputQty;
			rNet.OutputLayer.iDendriteQty=rSes.iSecondHiddenSize;
			rNet.SecondHiddenLayer.iNeuronQty = rSes.iSecondHiddenSize;
			rNet.SecondHiddenLayer.iDendriteQty=rSes.iFirstHiddenSize;
			rNet.FirstHiddenLayer.iNeuronQty = rSes.iFirstHiddenSize;
			rNet.FirstHiddenLayer.iDendriteQty=rSes.iInputQty;
			rNet.InputLayer.iNeuronQty = rSes.iInputQty;
			rNet.InputLayer.iDendriteQty=0; // layer zero has no dendrites
		}
		// block model?
	//sprintf(sLog, "%s: rLearn.iInputQty=%d, .iOutputQty=%d", __FUNCTION__, rLearn.iInputQty, rLearn.iOutputQty);
	RLog(rSes, 0, __FUNCTION__);
}

void cBarge::PrepareLearnSettings(struct rohanContext& rSes, struct rohanLearningSet& rLearn)
{mIDfunc
	char sLog[255];
	rLearn.iInputQty=rSes.iInputQty;
	rLearn.iOutputQty=rSes.iOutputQty;
	sprintf(sLog, "%s: rLearn.iInputQty=%d, .iOutputQty=%d", __FUNCTION__, rLearn.iInputQty, rLearn.iOutputQty);
	RLog(rSes, USERF, sLog);
}

int cBarge::PrepareAllSettings(struct rohanContext& rSes)
{mIDfunc /// sets initial and default value for globals and settings
	int iReturn=1; string s;
	//FILE *fOut;
		
	// ERRORS and progress tracking
	ResetSettings(rSes, *rSes.rNet, *rSes.rLearn);
	if (!SetProgOptions( rSes, rSes.argc, rSes.argv )){
		DirectoryEnsure(rSes.sRohanVerPath); 
		//if(){	need to add check to make sure file exists before overwriting or appending
		//	Ramp->GetFileHandle(rSes.sRohanVerPath, "Rohan.roh", 'w', 'a', &fOut);
		//	fprintf(fOut, "help=help\n#\t%s\n", vm["config"].as<string>().c_str() );  fclose(fOut);
		//}
		return 0;
	}
	// Get*Opt functions should only update rSes members according to vm values
	GetTagOpt(rSes);
	if (!BeginLogging(rSes))
		return 0;
	GetOptions(rSes, *rSes.rNet, *rSes.rLearn);
	
	// further setting modification begins here
	PrepareContext(rSes);
	PrepareNetSettings(rSes, *rSes.rNet);
	PrepareLearnSettings(rSes, *rSes.rLearn);

	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}

int cBarge::ObtainSampleSet(struct rohanContext& rSes)
{mIDfunc /// loads the learning set to be worked with Ante-Loop
	try{
		//int iReturn=0; 
		char sLog[255];
		int iLinesRead=Ramp->LoadSampleSet(rSes, rSes.sLearnSet);
		if (iLinesRead) {
			sprintf(sLog, "Parsed %d lines from %s", 
				iLinesRead, rSes.sLearnSet);
			RLog(rSes, USERF, sLog);
			sprintf(sLog, "Stored %d samples, %d input values, %d output values each.", 
				rSes.rLearn->lSampleQty, rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
			RLog(rSes, USERF, sLog);
			//verify samples fall within sector values
			if(CurateSectorValue(rSes)) {
				CompleteHostLearningSet(rSes);
				//CompleteHostLearningSetGGG(rSes);
			}
			else{
				return 0;
			} 
		}
		else {
			RLog(rSes, ERRORF, "No Samples Read by LoadSampleSet");
			iLinesRead=0;
		}
		RLog(rSes, 0, __FUNCTION__);
		return iLinesRead;
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


int cBarge::CurateSectorValue(struct rohanContext& rSes)
{mIDfunc /// compares sector qty to sample values for adequate magnitude
	int iOverK=0; char sLog[255];
	// loop over samples for inputs
	for (int s=0; s<rSes.rLearn->lSampleQty; ++s){
		//fprintf( fShow, "%dX|", s);
		for (int i=0; i<=rSes.rLearn->iInputQty; ++i){
			if(rSes.rLearn->dXInputs[ IDX2C( i, s, rSes.rLearn->iInputQty+1 )]>=rSes.rNet->iSectorQty){
				// if any input values fall beyond the maximum sector value, alert and make recommendation
				sprintf(sLog, "Error: Sample #%d has value that exceeds sector qty %d; suggest increasing to %d!",
					s, rSes.rNet->iSectorQty, static_cast<int>(floor(rSes.rLearn->dXInputs[ IDX2C( i, s, rSes.rLearn->iInputQty+1 ) ]*1.33)+1));
				RLog(rSes, ERRORF, sLog);
				++iOverK;
			}
		}	
		for (int i=0; i<=rSes.rLearn->iOutputQty; ++i){ // now loop over output values
			if(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 )]>=rSes.rNet->iSectorQty){
				// if any output values fall beyond the maximum sector value, alert and make recommendation
				sprintf(sLog, "Error: Sample #%d has value that exceeds sector qty %d; suggest increasing to %d!",
					s, rSes.rNet->iSectorQty, static_cast<int>(floor(rSes.rLearn->dDOutputs[ IDX2C( i, s, rSes.rLearn->iOutputQty+1 ) ]*1.33)+1));
				RLog(rSes, ERRORF, sLog);
				++iOverK;
			}
		}
	}
	RLog(rSes, 0, __FUNCTION__);
	if (iOverK)	// any out-of-bounds values are a fatal error
		return 0;
	else
		return rSes.rLearn->lSampleQty; // return number of samples veified within parameters
}


int cBarge::CompleteHostLearningSet(struct rohanContext& rSes)
{mIDfunc //allocate and fill arrays of complex values converted from scalar samples, all in host memory
	// needs to be adapted to multimodel capability ZZ
	int iReturn=0; char sLog[255];
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
		rSes.lMemStructAlloc = rSes.lMemStructAlloc | RLEARNcdc; // flag existence of allocated structs

	for(int S=0;S<rSes.rLearn->lSampleQty; S++){
		//fprintf(rSes.hostBucket, "%d[\t", S );
		for (int I=0;I< IQTY ; I++){
			if(rSes.rLearn->iContInputs)
				rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )] = ConvPhaseCx ( rSes, rSes.rLearn-> dXInputs [IDX2C( I, S, IQTY )] ); // convert scalar inputs on host
			else
				rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )] = ConvSectorCx( rSes, rSes.rLearn-> dXInputs [IDX2C( I, S, IQTY )] ); // convert scalar inputs on host
			//fprintf(rSes.hostBucket, "%9.6f+%9.6f,%d>%d\t", rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )].x, rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )].y, I, IDX2C( I, S, IQTY) );
		}
		for (int O=0;O< OQTY ; O++){
			rSes.rLearn-> dYEval      [IDX2C(  O, S, OQTY )] = S; 
			rSes.rLearn-> dAltYEval	  [IDX2C(  O, S, OQTY )] = -S;
			rSes.rLearn-> dSqrErr	  [IDX2C(  O, S, OQTY )] = O;
			if(rSes.rLearn->iContOutputs)
				rSes.rLearn-> cdcDOutputs [IDX2C(  O, S, OQTY )] = ConvPhaseCx ( rSes, rSes.rLearn->dDOutputs[IDX2C(  O, S, OQTY )] ); // convert cx desired outputs
			else
				rSes.rLearn-> cdcDOutputs [IDX2C(  O, S, OQTY )] = ConvSectorCx( rSes, rSes.rLearn->dDOutputs[IDX2C(  O, S, OQTY )] ); // convert cx desired outputs
			//fprintf(rSes.hostBucket, ":%9.6f+%9.6f,%d>%d\t", rSes.rLearn-> cdcDOutputs [IDX2C( O, S, OQTY )].x, rSes.rLearn-> cdcDOutputs [IDX2C( O, S, OQTY )].y, O, IDX2C( O, S, OQTY) );
			rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].x = O; 
			rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].y = S; 
			rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].x = -1*O;
			rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].y = -1*S;
		}
		//fprintf(rSes.hostBucket, "\n");
	}
	sprintf(sLog, "Sample set size = %d\n", 8*3*OUTSIZED + 16*INSIZECX + 16*3*OUTSIZECX);
	RLog(rSes, USERF, sLog);
	RLog(rSes, 0, __FUNCTION__);
	
	return iReturn;
}


int cBarge::CompleteHostLearningSetGGG(struct rohanContext& rSes)
{mIDfunc //allocate and fill arrays of complx values converted from scalar samples, all in host memory
	// needs to be adapted to multimodel capability ZZ
	int iReturn=0; char sLog[255];
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
		rSes.lMemStructAlloc = rSes.lMemStructAlloc | RLEARNcdc; // flag existence of allocated structs

	//if(rSes.iContReadMode==0){
		for(int S=0;S<rSes.rLearn->lSampleQty; S++){
			fprintf(rSes.hostBucket, "%d[\t", S );
			for (int I=0;I< IQTY ; I++){
				if(rSes.rLearn->iContInputs)
					rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )] = ConvPhaseCx ( rSes, rSes.rLearn-> dXInputs [IDX2C( I, S, IQTY )] ); // convert scalar inputs on host
				else
					rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )] = ConvSectorCx( rSes, rSes.rLearn-> dXInputs [IDX2C( I, S, IQTY )] ); // convert scalar inputs on host
				fprintf(rSes.hostBucket, "%9.6f+%9.6f,%d>%d\t", rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )].x, rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )].y, I, IDX2C( I, S, IQTY) );
			}
			for (int O=0;O< OQTY ; O++){
				rSes.rLearn-> dYEval      [IDX2C(  O, S, OQTY )] = S; 
				rSes.rLearn-> dAltYEval	  [IDX2C(  O, S, OQTY )] = -S;
				rSes.rLearn-> dSqrErr	  [IDX2C(  O, S, OQTY )] = O;
				if(rSes.rLearn->iContOutputs)
					rSes.rLearn-> cdcDOutputs [IDX2C(  O, S, OQTY )] = ConvPhaseCx ( rSes, rSes.rLearn->dDOutputs[IDX2C(  O, S, OQTY )] ); // convert cx desired outputs
				else
					rSes.rLearn-> cdcDOutputs [IDX2C(  O, S, OQTY )] = ConvSectorCx( rSes, rSes.rLearn->dDOutputs[IDX2C(  O, S, OQTY )] ); // convert cx desired outputs
				fprintf(rSes.hostBucket, ":%9.6f+%9.6f,%d>%d\t", rSes.rLearn-> cdcDOutputs [IDX2C( O, S, OQTY )].x, rSes.rLearn-> cdcDOutputs [IDX2C( O, S, OQTY )].y, O, IDX2C( O, S, OQTY) );
				rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].x = O; 
				rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].y = S; 
				rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].x = -1*O;
				rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].y = -1*S;
			}
			fprintf(rSes.hostBucket, "\n");
		}
	//}
	//if(rSes.iContReadMode==1){
	//	for(int S=0;S<rSes.rLearn->lSampleQty; S++){
	//		for (int I=0;I< IQTY ; I++){
	//			rSes.rLearn-> cdcXInputs [IDX2C( I, S, IQTY )]
	//				= ConvSectorCx( rSes, rSes.rLearn-> dXInputs [IDX2C( I, S, IQTY )] ); // convert scalar inputs on host
	//		}
	//		for (int O=0;O< OQTY ; O++){
	//			rSes.rLearn-> dYEval      [IDX2C(  O, S, OQTY )] = S; 
	//			rSes.rLearn-> dAltYEval	  [IDX2C(  O, S, OQTY )] = -S;
	//			rSes.rLearn-> dSqrErr	  [IDX2C(  O, S, OQTY )] = O;
	//			rSes.rLearn-> cdcDOutputs [IDX2C(  O, S, OQTY )]
	//				= ConvSectorCx( rSes, rSes.rLearn->dDOutputs[IDX2C(  O, S, OQTY )] ); // convert cx desired outputs
	//			rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].x = O; 
	//			rSes.rLearn-> cdcYEval    [IDX2C(  O, S, OQTY )].y = S; 
	//			rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].x = -1*O;
	//			rSes.rLearn-> cdcAltYEval [IDX2C(  O, S, OQTY )].y = -1*S;
	//		}
	//	}
	//}
	sprintf(sLog, "Sample set size = %d\n", 8*3*OUTSIZED + 16*INSIZECX + 16*3*OUTSIZECX);
	RLog(rSes, USERF, sLog);
	RLog(rSes, 0, __FUNCTION__);
	
	return iReturn;
}


int cBarge::DoPrepareNetwork(struct rohanContext& rSes)
{mIDfunc /// sets up network poperties and data structures for use
	int iReturn=0; char sLog[255];
	// on with it

					cuMakeArchValues(rSes, *rSes.rNet);
					//rSes.rLearn->iInputQty=rSes.rNet->rLayer[0].iNeuronQty;
					//rSes.rLearn->iOutputQty=rSes.rNet->rLayer[rSes.rNet->iLayerQTY-1].iNeuronQty;

	cuMakeNNStructures(rSes); // allocates memory and populates network structural arrays
	iReturn=Ramp->LoadNNWeights(rSes, rSes.sWeightSet);

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
			sprintf(sLog, "MAXIMUM WEIGHTS EXCEEDED at layer %d!\n", L); RLog(rSes, ERRORF, sLog);
		}
		if(rSes.rNet->iNeuronOfst[L]+rSes.rNet->iNeuronQTY[L] > MAXNEURONS){
			sprintf(sLog, "MAXIMUM NEURONS EXCEEDED at layer %d!\n", L); RLog(rSes, ERRORF, sLog);
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
		RLog(rSes, ERRORF, "Out of Memory in cuSectorTableMake");
	}
	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}


int cBarge::cuMakeLayers(int iInputQty, char *sLayerSizes, struct rohanContext& rSes)
{mIDfunc
/// Parses a string to assign network architecture parameters for use by later functions. 
/// Returns neurons in last layer if successful, otherwise 0
	char *sArchDup, *sDummy, sLog[255];
	int iLayerQTY=1;

	sArchDup = _strdup(sLayerSizes); // strtok chops up the input string, so we must make a copy (or do we? - 6/15/10) (yes we do 8/22/10)
	sDummy = strtok (sArchDup, " ,\t");
	while (sDummy!=NULL) {// this loop counts the values present in a copy of sLayerSizes, representing neurons in each layer until a not-legal layer value is reached
		sDummy = strtok (NULL, " ,\t");
		++iLayerQTY; //count layers
	}
	rSes.rNet->rLayer=(struct rohanLayer*)malloc(iLayerQTY * sizeof (struct rohanLayer)); //point to array of layers
		mCheckMallocWorked(rSes.rNet->rLayer)
		rSes.lMemStructAlloc = rSes.lMemStructAlloc || RNETlayers;
	sprintf(sLog, "%d layers plus input layer allocated.\n", (iLayerQTY-1));RLog(rSes, 0, sLog);
	
	sArchDup=_strdup(sLayerSizes); // second pass
	sDummy = strtok(sArchDup, " ,\t");
	for (int i=0;i<iLayerQTY;++i) {// this loop stores neurons in each layer
		if (i) {
			rSes.rNet->rLayer[i].iNeuronQty = atoi(sDummy);
			rSes.rNet->rLayer[i].iDendriteQty=rSes.rNet->rLayer[i-1].iNeuronQty; //previous layer's neuron qty is dendrite qty
			sDummy = strtok (NULL, " ,\t");
		}
		else {
			rSes.rNet->rLayer[i].iNeuronQty = iInputQty; // layer zero has virtual neurons with outputs equal to inputs converted to phases
			rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		}
		sprintf (sLog, "Layer %d: %d nodes\n", i, rSes.rNet->rLayer[i].iNeuronQty);
		RLog(rSes, 0, sLog);
	}
	if (cuMakeNNStructures(rSes)) 
		RLog(rSes, 0, "Nodes allocated.");
	RLog(rSes, 0, __FUNCTION__);
	return rSes.rNet->rLayer[rSes.rNet->iLayerQTY-1].iNeuronQty;
}



int cBarge::cuMakeArchValues(struct rohanContext& rSes, struct rohanNetwork& rNet)
{mIDfunc/// [No longer parses] a string to assign network architecture parameters for use by later functions. [now uses program_options 6/27/12]
/// Returns neurons in last layer if successful, otherwise 0
	try {
		char sLog[255];

		// preamble
		sprintf(sLog, "Using weights in %s\n", rSes.sWeightSet);
		RLog(rSes, USERF+DEBUGF, sLog);
		if(rSes.rNet->iContActivation)
			RLog(rSes, 0, "Continuous activation mode specified"); 
		else
			RLog(rSes, 0, "Discrete activation mode specified"); 
		
		// first polylayer mem structures
		rSes.rNet->rLayer=(struct rohanLayer*)malloc(rSes.rNet->iLayerQTY * sizeof (struct rohanLayer)); //point to array of layers
			mCheckMallocWorked(rSes.rNet->rLayer)
			sprintf(sLog, "mallocated %d polylayers\n", rSes.rNet->iLayerQTY);
			RLog(rSes, 0, sLog);
		if(rSes.iLayerQty==1){
			rSes.rNet->rLayer[1].iNeuronQty = rSes.iOutputQty;
			rSes.rNet->rLayer[1].iDendriteQty=rSes.iInputQty;
			rSes.rNet->rLayer[0].iNeuronQty = rSes.iInputQty;
			rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		}
		if(rSes.iLayerQty==2){
			rSes.rNet->rLayer[2].iNeuronQty = rSes.iOutputQty;
			rSes.rNet->rLayer[2].iDendriteQty=rSes.iFirstHiddenSize;
			rSes.rNet->rLayer[1].iNeuronQty = rSes.iFirstHiddenSize;
			rSes.rNet->rLayer[1].iDendriteQty=rSes.iInputQty;
			rSes.rNet->rLayer[0].iNeuronQty = rSes.iInputQty;
			rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		}
		if(rSes.iLayerQty==3){
			rSes.rNet->rLayer[3].iNeuronQty = rSes.iOutputQty;
			rSes.rNet->rLayer[3].iDendriteQty=rSes.iSecondHiddenSize;
			rSes.rNet->rLayer[2].iNeuronQty = rSes.iSecondHiddenSize;
			rSes.rNet->rLayer[2].iDendriteQty=rSes.iFirstHiddenSize;
			rSes.rNet->rLayer[1].iNeuronQty = rSes.iFirstHiddenSize;
			rSes.rNet->rLayer[1].iDendriteQty=rSes.iInputQty;
			rSes.rNet->rLayer[0].iNeuronQty = rSes.iInputQty;
			rSes.rNet->rLayer[0].iDendriteQty=0; // layer zero has no dendrites
		}
		// block, fixed length structures
			rNet.iDendrtQTY[0]=0;
			rNet.iNeuronQTY[0]=rSes.iInputQty+1;
			rNet.iWeightQTY[0]=0;
		if(rSes.iLayerQty==1){
			rNet.iNeuronQTY[1]=rSes.iOutputQty+1;
			rNet.iNeuronQTY[2]=0;
			rNet.iNeuronQTY[3]=0;
		}
		if(rSes.iLayerQty==2){
			rNet.iNeuronQTY[1]=rSes.iFirstHiddenSize+1;
			rNet.iNeuronQTY[2]=rSes.iOutputQty+1;
		}
		if(rSes.iLayerQty==3){
			rNet.iNeuronQTY[1]=rSes.iFirstHiddenSize+1;
			rNet.iNeuronQTY[2]=rSes.iSecondHiddenSize+1;
		}
		rNet.iDendrtQTY[1]=rNet.iNeuronQTY[0];
		rNet.iDendrtQTY[2]=rNet.iNeuronQTY[1];
		rNet.iDendrtQTY[3]=rNet.iNeuronQTY[2];
		rNet.iWeightQTY[1]=rNet.iNeuronQTY[1]*rNet.iDendrtQTY[1];
		rNet.iWeightQTY[2]=rNet.iNeuronQTY[2]*rNet.iDendrtQTY[2];
		rNet.iWeightQTY[3]=rNet.iNeuronQTY[3]*rNet.iDendrtQTY[3];
		rNet.iDendrtOfst[0]=0;
		rNet.iNeuronOfst[0]=0;
		rNet.iWeightOfst[0]=0;
		rNet.iDendrtOfst[1]=rNet.iDendrtOfst[0]+rNet.iDendrtQTY[0];
		rNet.iNeuronOfst[1]=rNet.iNeuronOfst[0]+rNet.iNeuronQTY[0];
		rNet.iWeightOfst[1]=rNet.iWeightOfst[0]+rNet.iWeightQTY[0];
		rNet.iDendrtOfst[2]=rNet.iDendrtOfst[1]+rNet.iDendrtQTY[1];
		rNet.iNeuronOfst[2]=rNet.iNeuronOfst[1]+rNet.iNeuronQTY[1];
		rNet.iWeightOfst[2]=rNet.iWeightOfst[1]+rNet.iWeightQTY[1];
		rNet.iDendrtOfst[3]=rNet.iDendrtOfst[2]+rNet.iDendrtQTY[2];
		rNet.iNeuronOfst[3]=rNet.iNeuronOfst[2]+rNet.iNeuronQTY[2];
		rNet.iWeightOfst[3]=rNet.iWeightOfst[2]+rNet.iWeightQTY[2];
						
		// HN3 structures ?
		
		RLog(rSes, 0, "NN architecture made");
		RLog(rSes, 0, __FUNCTION__);
		
		return rSes.rNet->rLayer[rSes.rNet->iLayerQTY-1].iNeuronQty;
	}
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
		++rSes.iErrors;
        return 0;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
		++rSes.iErrors;
		return 0;
    }
}


int cBarge::cuMakeNNStructures(struct rohanContext &rSes)
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

// polylayer model

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
	RLog(rSes, 0, __FUNCTION__);
	return lReturn; //return how many weights and outputs allocated
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
	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}

int cBarge::ShowDiagnostics(struct rohanContext& rSes)
{mIDfunc
	int iReturn=1;
	char sLog[255];
	
	if(Team==NULL)
		RLog(rSes, ERRORF, "cBarge diagnostics: No team!");
	if(Drover==NULL)
		RLog(rSes, ERRORF, "cBarge diagnostics: No drover!");
	if(rSes.rLearn->cdcXInputs==NULL)
		RLog(rSes, ERRORF, "cBarge diagnostics: No complex inputs at host!");
	if(rSes.rLearn->dDOutputs==NULL)
		RLog(rSes, ERRORF, "cBarge diagnostics: No scalar outputs at host!");
	if(rSes.rLearn->cdcDOutputs==NULL)
		RLog(rSes, ERRORF, "cBarge diagnostics: No complex outputs at host!");
	if (rLearn==NULL)
		RLog(rSes, ERRORF, "cBarge diagnostics: No rLearn structure!");
	else{
		sprintf(sLog, "Barge is holding %d samples w/ %d inputs, %d output(s).", rSes.rLearn->lSampleQty, rSes.rLearn->iInputQty, rSes.rLearn->iOutputQty);
		RLog(rSes, 0, sLog);
	}
#ifdef _DEBUG
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
		rSes.iContReadMode ? cout << "Continuous Inputs TRUE by DEFAULT.\n" : cout << "Continuous Inputs FALSE by DEFAULT.\n";
		rSes.iContActivation ? cout << "Continuous Outputs true by DEFAULT.\n" : cout << "Continuous Outputs false by DEFAULT.\n";
	}
#endif
	RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}


int cBarge::cuMessage(cublasStatus csStatus, char *sName, char *sCodeFile, int iLine, char *sFunc)
{	
	char *sMsg;

	switch (csStatus) {
		case CUBLAS_STATUS_SUCCESS: sMsg=_strdup("operation completed successfully");
			break;
		case CUBLAS_STATUS_NOT_INITIALIZED: sMsg=_strdup("library not initialized");
			break;
		case CUBLAS_STATUS_ALLOC_FAILED: sMsg=_strdup("resource allocation failed");
			break;
		case CUBLAS_STATUS_INVALID_VALUE: sMsg=_strdup("unsupported numerical value was passed to function");
			break;
		case CUBLAS_STATUS_ARCH_MISMATCH: sMsg=_strdup("function requires an architectural feature absent from the architecture of the device");
			break;
		case CUBLAS_STATUS_MAPPING_ERROR: sMsg=_strdup("access to GPU memory space failed");
			break;
		case CUBLAS_STATUS_EXECUTION_FAILED: sMsg=_strdup("GPU program failed to execute");
			break;
		case CUBLAS_STATUS_INTERNAL_ERROR: sMsg=_strdup("an internal operation failed");
			break;
		default: sMsg=_strdup("unknown response");
	}
	fprintf(stderr,"%s %s line %i: CUBLAS %s: %s\n", sCodeFile, sFunc, iLine, sMsg, sName);
	return 0;
}


void cBarge::RLog(struct rohanContext& rSes, int iRank, char * sLogEntry)
{mIDfunc // logs strings describing events, preceeded by the local time
	using namespace boost::posix_time; 
	ptime now = second_clock::local_time(); //use the clock 
	string p;
		
	try{
		if( rSes.iLoggingEnabled ){
			p=strtok(sLogEntry, "\n"); // trim any trailing chars

			if(iRank & ERRORF)
				*(rSes.ofsRLog) << now << "\t" << iRank << "\t" << "error=" << p << endl; // all entries go to applog, regardless
			if(iRank & WARNINGF)
				*(rSes.ofsRLog) << now << "\t" << iRank << "\t" << "warning=" << p << endl; // all entries go to applog, regardless
			if(((iRank & ERRORF) & WARNINGF) == 0)
				*(rSes.ofsRLog) << now << "\t" << iRank << "\t" << p << endl; // all entries go to applog, regardless
			
			if(iRank & USERF){
				if(rSes.bConsoleUsed)
					cout << "#\t " << p << endl; // entries sent to console
				else
					*(rSes.ofsHanLog) << "#\t " << now << " " << p << endl; // entries repeated as comments in .han file if any.
			}

			if(iRank & GUIF){
				if(rSes.bConsoleUsed)
					cout << p << endl; // entries sent to console
				else
					*(rSes.ofsHanLog) << p << endl; // typically for 'report=filename937.txt'
			}

			if(iRank & ERRORF){
				++rSes.iErrors;
				if(rSes.bConsoleUsed)
					cerr << p << " " << now << endl; // entries sent to standard error reporting pipe or stream
				else
					*(rSes.ofsHanLog) << "error=" << p << endl; 
			}

			if(iRank & WARNINGF){
				++rSes.iWarnings;
				if(rSes.bConsoleUsed)
					cerr << p << " " << now << endl; // entries sent to standard error reporting pipe or stream
					//fprintf(stderr, "%s  %s\n", p.c_str(), now.date());
				else
					*(rSes.ofsHanLog) << "warning=" << p << endl; 
			}

			if(iRank & ADMINF){
				if(!rSes.bConsoleUsed)
					cout << p << endl; // entries sent to screen *if* running autonomously
			}

			if(iRank & DEBUGF){
				fprintf(stderr, "%s\n", p.c_str());
			}

			if(iRank & CODERF){
				// put some e-mail or other messaging in here?
			}
		}
	}
	catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        //return -1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
		//return -1;
    }
}


int cBarge::DoCuFree(struct rohanContext &rSes)
{mIDfunc/// free allocated memory for all structures
	
	cuFreeNNTop(rSes); // free network topology structures
	cuFreeLearnSet(rSes); // free learning set structures
	RLog(rSes, 0, __FUNCTION__);
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
				RLog(rSes, ERRORF, "can't free( rSes.rNet->cdcSectorBdry)");

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
			RLog(rSes, 0, "Network structures freed.");
		RLog(rSes, 0, __FUNCTION__);
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
			RLog(rSes, 0, "Network structures freed.");
		RLog(rSes, 0, __FUNCTION__);
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

void cBarge::LogFlush(struct rohanContext &rSes)
{mIDfunc/// flushes remaining log entries, hopefully getting all entries out before app ends or is suspended.
	if(rSes.iLoggingEnabled){
		if( rSes.ofsHanLog->is_open() )
			rSes.ofsHanLog->flush();
		if( rSes.ofsRLog->is_open() )
			rSes.ofsRLog->flush();
	}
}