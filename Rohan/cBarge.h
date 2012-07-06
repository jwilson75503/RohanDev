#ifndef CBARGE_H
#define CBARGE_H

// BOOST
#include <boost/program_options.hpp>
namespace po=boost::program_options;
using namespace boost::program_options;
// usings
using namespace std;

#include <iostream>
#include <fstream>
#include <iterator>
#include "stdafx.h"

class cBarge
{/// Represents the load of data upon which the computational work is to be performed.
		struct rohanContext * rSes;
		struct rohanLearningSet * rLearn;
		struct rohanNetwork * rNet;
		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
		class cTeam * Team /*! The calculating "engine" currently in use. */;
//	private:
        static const char* const classname;
	public:
		cBarge( struct rohanContext& rSes){ SetContext(rSes); /*ShowMe();*/ } ; // end ctor
		void ShowMe();
		class variables_map vm /*! A variable map containing all program_options in use. */;
		int SetContext( struct rohanContext& rSes); // completed
			int SetDrover( class cDrover * cdDrover); // completed
			int SetTeam( class cTeam * cdtTeam); // completed
		int SetProgOptions(struct rohanContext& rSes, int argc, char * argv[]); // interprets command line options
		int ObtainGlobalSettings(struct rohanContext& rSes); /// sets initial and default value for globals and settings
			void ResetContext(struct rohanContext& rSes);
			int GetSamplesOpt(struct rohanContext& rSes);
			int GetWeightsOpt(struct rohanContext& rSes);
			int GetNetworkOpt(struct rohanContext& rSes);
			int GetSessTagOpt(struct rohanContext& rSes);
		int ObtainSampleSet(struct rohanContext& rSes); /// chooses and loads the learning set to be worked with Ante-Loop
			int DoLoadSampleSet(struct rohanContext& rSes, FILE *fileInput); /// pulls in values from .txt files, used for testing before main loop
			int CurateSectorValue(struct rohanContext& rSes); /// compares sector qty to sample values for adequate magnitude
			int CompleteHostLearningSet(struct rohanContext& rSes); /// allocate, fill cx converted value & alt values, all in host memory
				//int cuDoubleComplex ConvScalarCx(struct rohanContext& rSes, int Scalar); // converts a scalar value to a returned complex coordinate)
			//int LetCplxCopySamples(struct rohanContext& rSes); //load complex samples into the parallel structures in the host memory
		int DoPrepareNetwork(struct rohanContext& rSes); /// sets up network poperties and data structures for use
		int cuMakeLayers(int iInputQty, char *sLayerSizes, struct rohanContext& rSes);
		int cuMakeArchValues(char *sMLMVNarch, struct rohanContext& rSes);
		int cuMakeNNStructures(struct rohanContext &rSes);
		int LayersToBlocks(struct rohanContext& Ses); //, struct rohanNetwork& Net); /// moves weight values from old layer structures to new block structures
		int BinaryFileHandleRead(char* sFileName, FILE** fileInput);
		int BinaryFileHandleWrite(char *sFileName, FILE **fileOutput);
		int AsciiFileHandleRead(char *sFileName, FILE **fileInput);
		int AsciiFileHandleWrite(char *sFilePath, char *sFileName, FILE **fileOutput);
		int cuNNLoadWeights(struct rohanContext &rSes, FILE *fileInput);
		int cuSaveNNWeights(struct rohanContext &rSes, FILE *fileOutput);
		int cuSaveNNWeightsASCII(struct rohanContext &rSes, FILE *fileOutput);
		int cuPreSaveNNWeights(struct rohanContext& rSes, char cVenue);
		int AsciiWeightDump(struct rohanContext& rSes, FILE *fileOutput);
		int LetWriteWeights(struct rohanContext& rSes); /// saves weight values to disk
		int LetWriteEvals(struct rohanContext& rSes, struct rohanLearningSet& rLearn); /// saves evaluated output values to disk
		int ShowDiagnostics(struct rohanContext& rSes);
		static char sep_to_space(char c);
		template <typename T>
		int VectorFromOption(char * sOption, vector<T> & n, int p);
		static void RLog(struct rohanContext& rSes, int iRank, char * sLogEntry);
		int DoCuFree(struct rohanContext &rSes);
			int cuFreeNNTop(struct rohanContext &rSes); /// frees data structures related to network topology
			int cuFreeLearnSet(struct rohanContext &rSes); /// free the learning set of samples
};

template <typename T>
int cBarge::VectorFromOption(char * sOption, vector<T> & n, int p)
	{mIDfunc/// converts strings to vectors of any (?) type
	// returns p if # of elements match, otherwise returns 0
	string s;

	s=cBarge::vm[sOption].as<string>();
	transform(s.begin(), s.end(), s.begin(), &cBarge::sep_to_space );
	stringstream ss(s);
	copy(istream_iterator<T>(ss), istream_iterator<T>(), std::back_inserter(n));
	if(p==n.size())
		return p; // evals to true (unless p is zero for some reason)
	else
		return 0; // evals to false
}

int AskSessionName(struct rohanContext& rSes);
int GetNNTop(struct rohanContext& rSes);
int GetWeightSet(struct rohanContext& rSes);
int GetSampleSet(struct rohanContext& rSes);
int ReGetSampleSet(struct rohanContext& rSes);
int PrepareNetwork(struct rohanContext& rSes);


#endif
