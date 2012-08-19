#ifndef CBARGE_H
#define CBARGE_H

// BOOST
#include <boost/program_options.hpp>
#include <boost/program_options.hpp>

namespace po=boost::program_options;
using namespace boost::program_options;
// usings
using namespace std;

#include <iostream>
#include <iterator>
#include "stdafx.h"

class cBarge
{/// Represents the load of data upon which the computational work is to be performed.
		struct rohanContext * rSes;
		struct rohanLearningSet * rLearn;
		struct rohanNetwork * rNet;
		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
		class cRamp * Ramp;
		class cTeam * Team /*! The calculating "engine" currently in use. */;

		vector<cuDoubleComplex> LastWtKebabB;
		vector<cuDoubleComplex> BestWtKebabB;
		vector<cuDoubleComplex> LastWtKebabP;
		vector<cuDoubleComplex> BestWtKebabP;
		vector<cuDoubleComplex> LastWtKebab3;
		vector<cuDoubleComplex> BestWtKebab3;
		vector<cuDoubleComplex> LastWtKebabS;
		vector<cuDoubleComplex> BestWtKebabS;
	public:
		vector<int> ivTopoNN; 
		vector<int> ivOffsetK; 
		vector<int> ivOffsetQ; 
		class variables_map vm /*! A variable map containing all program_options in use. */;
		cBarge( struct rohanContext& rSes) ; // end ctor
		void ShowMe();
		int SetContext( struct rohanContext& rSes); // completed
		int PrepareAllSettings(struct rohanContext& rSes); /// sets initial and default value for globals and settings
			void ResetSettings(struct rohanContext& rSes, struct rohanNetwork& rNet, struct rohanLearningSet& rLearn);
			int SetProgOptions(struct rohanContext& rSes, int argc, _TCHAR * argv[]); // interprets command line options
			int GetTagOpt(struct rohanContext& rSes);
			int BeginLogging(struct rohanContext& rSes);
			void GetOptions(struct rohanContext& rSes, struct rohanNetwork& rNet, struct rohanLearningSet& rLearn);
				int GetEvalOpt(struct rohanContext& rSes);
				int GetContyOpt(struct rohanContext& rSes);
				int GetSamplesOpt(struct rohanContext& rSes);
				int GetTSeriesOpt(struct rohanContext& rSes);
				int GetWeightsOpt(struct rohanContext& rSes);
				int GetLearnOpt(struct rohanContext& rSes);
				int GetNetworkOpt(struct rohanContext& rSes, struct rohanNetwork& rNet);
			void PrepareContext(struct rohanContext& rSes);
				int GetHdweSet(struct rohanContext& rSes);
			void PrepareNetSettings(struct rohanContext& rSes, struct rohanNetwork& rNet);
			void PrepareLearnSettings(struct rohanContext& rSes, struct rohanLearningSet& rLearn);
		int ObtainSampleSet(struct rohanContext& rSes); /// chooses and loads the learning set to be worked with by AnteLoop
			int CurateSectorValue(struct rohanContext& rSes); /// compares sector qty to sample values for adequate magnitude
			int CompleteHostLearningSet(struct rohanContext& rSes); /// allocate, fill cx converted value & alt values, all in host memory
			int CompleteHostLearningSetGGG(struct rohanContext& rSes); /// allocate, fill cx converted value & alt values, all in host memory
				//int cuDoubleComplex ConvSectorCx(struct rohanContext& rSes, int Scalar); // converts a scalar value to a returned complex coordinate)
			//int LetCplxCopySamples(struct rohanContext& rSes); //load complex samples into the parallel structures in the host memory
		int DoPrepareNetwork(struct rohanContext& rSes); /// sets up network poperties and data structures for use
		int cuMakeLayers(int iInputQty, char *sLayerSizes, struct rohanContext& rSes);
		int cuMakeArchValues(struct rohanContext& rSes, struct rohanNetwork& rNet);
		int cuMakeNNStructures(struct rohanContext &rSes);
		int LayersToBlocks(struct rohanContext& Ses); //, struct rohanNetwork& Net); /// moves weight values from old layer structures to new block structures
		int ShowDiagnostics(struct rohanContext& rSes);
		static char sep_to_space(char c);
		template <typename T>
		int VectorFromOption(char * sOption, vector<T> & n, int p);
		int cuMessage(cublasStatus csStatus, char *sName, char *sCodeFile, int iLine, char *sFunc);
		static void RLog(struct rohanContext& rSes, int iRank, char * sLogEntry);
		int DoCuFree(struct rohanContext &rSes);
			int cuFreeNNTop(struct rohanContext &rSes); /// frees data structures related to network topology
			int cuFreeLearnSet(struct rohanContext &rSes); /// free the learning set of samples
		void LogFlush(struct rohanContext &rSes);
private:
};

template <typename T>
int cBarge::VectorFromOption(char * sOption, vector<T> & n, int p)
	{mIDfunc/// converts strings to vectors of any (?) type
	// returns p if # of elements match, otherwise returns 0
	string s;

	if(vm.count(sOption)){ // only go ahead if there is something to be converted
		s=cBarge::vm[sOption].as<string>();
		transform(s.begin(), s.end(), s.begin(), &cBarge::sep_to_space );
		stringstream ss(s);
		copy(istream_iterator<T>(ss), istream_iterator<T>(), std::back_inserter(n));
		if(p==n.size())
			return p; // evals to true (unless p is zero for some reason)
		else
			return 0; // evals to false
	}
	else
		return 0;
}

int AskSessionName(struct rohanContext& rSes);
int GetNNTop(struct rohanContext& rSes);
int GetWeightSet(struct rohanContext& rSes);
int GetSampleSet(struct rohanContext& rSes);
int ReGetSampleSet(struct rohanContext& rSes);
int PrepareNetwork(struct rohanContext& rSes);
int SetVerPath( struct rohanContext& rSes );


#endif
