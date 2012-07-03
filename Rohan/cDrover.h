#ifndef CDROVER_H
#define CDROVER_H

class cDrover
{/// Provisional name for Controller/UI handler.
		struct rohanContext * rSes;
		struct rohanLearningSet * rLearn;
		struct rohanNetwork * rNet;
	private:
        static const char* const classname;
		class cTeam * Team /*! The calculating "engine" currently in use. */;
	public:
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		cDrover( rohanContext& rC, rohanLearningSet& rL, rohanNetwork& rN, cBarge& cB, cTeam& cT);// { SetContext(rC, rL, rN); SetDroverBargeAndTeam(cB, cdT); /*ShowMe();*/ }; // end ctor
			void ShowMe();
			int SetContext( struct rohanContext& rC, struct rohanLearningSet& rL, struct rohanNetwork& rN); // completed
			int SetDroverBargeAndTeam( class cBarge& cbB, class cTeam& cdtT); // completed
		int DoAnteLoop(struct rohanContext& rSes, int argc, char * argv[]); /// prepares all parameters and data structures necesary for learning and evaluation.
			int ObtainGlobalSettings(struct rohanContext& rSes); /// sets initial and default value for globals and settings
			int AskSampleSetName(struct rohanContext& rSes) ;  /// chooses the learning set to be worked with Ante-Loop
			int ShowDiagnostics(struct rohanContext& rSes, struct rohanNetwork& rNet); /// show some statistics, dump weights, and display warning and error counts
		int DoMainLoop(struct rohanContext& rSes); /// Trains a weight set to more closely reproduce the sampled outputs from the sampled inputs, and other options.
			int DisplayMenu(int iMenuNum, struct rohanContext& rSes);
				int MenuBase(struct rohanContext& rSes);
			int AskSessionName(struct rohanContext& rSes);
			int GetNNTop(struct rohanContext& rSes);
			int GetWeightSet(struct rohanContext& rSes);
			int LetInteractiveEvaluation(struct rohanContext& rSes); 
			int LetInteractiveLearning(struct rohanContext& rSes);
			int LetUtilities(struct rohanContext& rSes);
			void DoLearnOpt(struct rohanContext& rSes);
			void DoEvalOpt(struct rohanContext& rSes);
		int DoPostLoop(struct rohanContext& rSes); /// Final operations including freeing of dynamically allocated memory are called from here. 
			int DoEndItAll(struct rohanContext& rSes); /// prepares for graceful ending of program
		
};

int GetWeightSet(struct rohanContext& rSes);
int GetSampleSet(struct rohanContext& rSes);
int ReGetSampleSet(struct rohanContext& rSes);
int PrepareNetwork(struct rohanContext& rSes);
int DirectoryEnsure(char * sPath);
int GetUserDocPath(char * sPath);



#endif
