#ifndef CDROVER_H
#define CDROVER_H


class cDrover
{/// Provisional name for Controller/UI handler.
		struct rohanContext * rSes;
		struct rohanLearningSet * rLearn;
		struct rohanNetwork * rNet;
	private:
        static const char* const classname;
		class cDeviceTeam * Team /*! The calculating "engine" currently in use. */;
	public:
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		cDrover( rohanContext& rC, rohanLearningSet& rL, rohanNetwork& rN, cBarge& cB, cDeviceTeam& cdT);// { SetContext(rC, rL, rN); SetDroverBargeAndTeam(cB, cdT); /*ShowMe();*/ }; // end ctor
			void ShowMe();
			int SetContext( struct rohanContext& rC, struct rohanLearningSet& rL, struct rohanNetwork& rN); // completed
			int SetDroverBargeAndTeam( class cBarge& cbB, class cDeviceTeam& cdtT); // completed
		int DoAnteLoop(struct rohanContext& rSes, int argc, char * argv[]); /// prepares all parameters and data structures necesary for learning and evaluation.
			int ObtainGlobalSettings(struct rohanContext& rSes); /// sets initial and default value for globals and settings
			int AskSampleSetName(struct rohanContext& rSes) ;  /// chooses the learning set to be worked with Ante-Loop
			int ShowDiagnostics(struct rohanContext& rSes, struct rohanNetwork& rNet); /// show some statistics, dump weights, and display warning and error counts
		int DoMainLoop(struct rohanContext& rSes); /// Trains a weight set to more closely reproduce the sampled outputs from the sampled inputs, and other options.
			int DisplayMenu(int iMenuNum, struct rohanContext& rSes);
			int CLIbase(struct rohanContext& rSes);
			int GetWeightSet(struct rohanContext& rSes);
			int LetInteractiveEvaluation(struct rohanContext& rSes); 
			int LetInteractiveLearning(struct rohanContext& rSes);
			int LetUtilities(struct rohanContext& rSes);
		int DoPostLoop(struct rohanContext& rSes); /// Final operations including freeing of dynamically allocated memory are called from here. 
			int DoEndItAll(struct rohanContext& rSes); /// prepares for graceful ending of program
		int GetNNTop(struct rohanContext& rSes);
		void RLog(struct rohanContext& rSes, char * sLogEntry);
};

//int AnteLoop(struct rohanContext& rSes, int argc, char * argv[]);
//int GetGlobalSettings(struct rohanContext& rSes);
int BeginSession(struct rohanContext& rSes);

int GetWeightSet(struct rohanContext& rSes);
int GetSampleSet(struct rohanContext& rSes);
int ReGetSampleSet(struct rohanContext& rSes);
int PrepareNetwork(struct rohanContext& rSes);
int DirectoryEnsure(char * sPath);
int GetUserDocPath(char * sPath);


//int InteractiveEvaluation(struct rohanContext& rSes);
//int InteractiveLearning(struct rohanContext& rSes);
//void PostLoop(struct rohanContext& rSes);


#endif
