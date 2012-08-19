#ifndef CDROVER_H
#define CDROVER_H

class cDrover
{/// Provisional name for Controller/UI handler.
		struct rohanContext * rSes;
		struct rohanLearningSet * rLearn;
		struct rohanNetwork * rNet;
	private:
        class cTeam * Team /*! The calculating "engine" currently in use. */;
		class cRamp * Ramp ;
	public:
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		cDrover( rohanContext& rC); //, rohanLearningSet& rL, rohanNetwork& rN, cBarge& cB, cRamp& cR, cTeam& cT, int argc, _TCHAR * argv[]);// ctor in cDover.cpp
			void ShowMe();
			int SetContext( struct rohanContext& rC, int argc, _TCHAR * argv[]); // completed
		int DoAnteLoop(struct rohanContext& rSes, int argc, _TCHAR * argv[]); /// prepares all parameters and data structures necesary for learning and evaluation.
			int PrepareAllSettings(struct rohanContext& rSes); /// sets initial and default value for globals and settings
			int ShowDiagnostics(struct rohanContext& rSes, struct rohanNetwork& rNet); /// show some statistics, dump weights, and display warning and error counts
		int DoMainLoop(struct rohanContext& rSes); /// Trains a weight set to more closely reproduce the sampled outputs from the sampled inputs, and other options.
			void DoContyOpt(struct rohanContext& rSes);
			void DoLearnOpt(struct rohanContext& rSes);
			void DoEvalOpt(struct rohanContext& rSes);
		int DoPostLoop(struct rohanContext& rSes); /// Final operations including freeing of dynamically allocated memory are called from here. 
			int DoEndItAll(struct rohanContext& rSes); /// prepares for graceful ending of program
		
};

// from WinUnix.h
int DirectoryEnsure(char * sPath);
int GetUserDocPath(char * sPath);


#endif
