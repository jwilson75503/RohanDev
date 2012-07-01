#ifndef CBARGE_H
#define CBARGE_H

// BOOST
#include <boost/program_options.hpp>
namespace po=boost::program_options;
using namespace boost::program_options;
// usings
using namespace std;

class cBarge
{/// Represents the load of data upon which the computational work is to be performed.
		struct rohanContext * rSes;
		struct rohanLearningSet * rLearn;
		struct rohanNetwork * rNet;
		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
		class cDeviceTeam * Team /*! The calculating "engine" currently in use. */;
//	private:
        static const char* const classname;
	public:
		cBarge( struct rohanContext& rSes){ SetContext(rSes); /*ShowMe();*/ } ; // end ctor
		void ShowMe();
		class variables_map vm /*! A variable map containing all program_options in use. */;
		int SetContext( struct rohanContext& rSes); // completed
			int SetDrover( class cDrover * cdDrover); // completed
			int SetTeam( class cDeviceTeam * cdtTeam); // completed
		int SetProgOptions(struct rohanContext& rSes, int argc, char * argv[]); // interprets command line options
		int ObtainGlobalSettings(struct rohanContext& rSes); /// sets initial and default value for globals and settings
		int ObtainSampleSet(struct rohanContext& rSes); /// chooses and loads the learning set to be worked with Ante-Loop
			int DoLoadSampleSet(struct rohanContext& rSes, FILE *fileInput); /// pulls in values from .txt files, used for testing before main loop
			int CurateSectorValue(struct rohanContext& rSes); /// compares sector qty to sample values for adequate magnitude
			int CompleteHostLearningSet(struct rohanContext& rSes); /// allocate, fill cx converted value & alt values, all in host memory
				//int cuDoubleComplex ConvScalarCx(struct rohanContext& rSes, int Scalar); // converts a scalar value to a returned complex coordinate)
			//int LetCplxCopySamples(struct rohanContext& rSes); //load complex samples into the parallel structures in the host memory
		int DoPrepareNetwork(struct rohanContext& rSes); /// sets up network poperties and data structures for use
		int LayersToBlocks(struct rohanContext& Ses); //, struct rohanNetwork& Net); /// moves weight values from old layer structures to new block structures
		int LetWriteWeights(struct rohanContext& rSes); /// saves weight values to disk
		int LetWriteEvals(struct rohanContext& rSes, struct rohanLearningSet& rLearn); /// saves evaluated output values to disk
		int ShowDiagnostics();
		//template <typename T> void OptionToNumericVectorT(char * sOption, vector<T> & n);
		void OptionToIntVector(char * sOption, vector<int> & n);
		void OptionToDoubleVector(char * sOption, vector<double> & n);
		void RLog(struct rohanContext& rSes, char * sLogEntry);
		void HanReport(struct rohanContext& rSes, char * sLogEntry);
		int DoCuFree(struct rohanContext &rSes);
			int cuFreeNNTop(struct rohanContext &rSes); /// frees data structures related to network topology
			int cuFreeLearnSet(struct rohanContext &rSes); /// free the learning set of samples
};


//int AnteLoop(struct rohanContext& rSes, int argc, char * argv[]);
//int GetGlobalSettings(struct rohanContext& rSes);
int BeginSession(struct rohanContext& rSes);
int GetNNTop(struct rohanContext& rSes);
	//int cuFreeNNTop(struct rohanContext &rSes);
int GetWeightSet(struct rohanContext& rSes);
int GetSampleSet(struct rohanContext& rSes);
int ReGetSampleSet(struct rohanContext& rSes);
int PrepareNetwork(struct rohanContext& rSes);
//void MainLoop(struct rohanContext& rSes);

//int InteractiveEvaluation(struct rohanContext& rSes);
//int InteractiveLearning(struct rohanContext& rSes);
//void PostLoop(struct rohanContext& rSes);


#endif
