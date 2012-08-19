#ifndef CTEAM_H
#define CTEAM_H


class cTeam// revised to remove subclassing: public cTeam // cTeam members available at their original access
{/// A team of mighty stallions; parallel calculations in CUDA C running on the GPU
		struct rohanContext * rSes;
		struct rohanNetwork * rNet;
		struct rohanLearningSet * rLearn;
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
		class cRamp * Ramp;
		int lIterationQty /*! Iterate this many times over sample set and stop. */;
		char bHitched;
		char bTaut;
		char cEngagedModel;
		char cExpModel;
		char cControlModel;
	public:
		int SetContext( struct rohanContext& rSes); // completed
		int GetTrainables( struct rohanContext& rSes, int lSampleQtyReq);
		int SaveContext(int iMode);
		//int SaveWeights(int iMode);
		//int SaveEvaluation(int iMode);
		int SetStopCriteria( int lIterationQty, double dTargetRMSE); /// specifies conditions for end of training task
		int LetEvalSet( rohanContext& rS, char chMethod); /// Submits a subset of the samples available forevaluation.
	cTeam( struct rohanContext& rSes); /// ctor body in source
		void ShowMe() /*! diagnostic identity display on screen */;
// ENGAGING MODELS
public:	int LetHitch(struct rohanContext& rSes, char cModel) /*! copy data to dev mem and attach structures to team */;
public:	int LetUnHitch(struct rohanContext& rSes) /*! release dev mem structures */;
		int TransferContext(struct rohanContext& rSes, char Direction) /*! copy rSes 0D members to dev mem */;
	private:	int CopyNet(struct rohanContext& rSes, char Direction) /*! copy rNet members to dev mem */;
				int TransferNet(struct rohanContext& rSes, char Direction) /*! copy Net params to dev mem */;
				int CopyLearnSet(struct rohanContext& rSes, char Direction) /*! copy rLearn members to dev mem */;
public:	int LetTaut(struct rohanContext& rSes) /*! update dev mem from host for epoch */;
public:	int LetSlack(struct rohanContext& rSes) /*! update host mem with results of epoch calculations */;
	private:	int TransferLayers(struct rohanContext& rSes, char Direction);
				int TransferOutputs(struct rohanContext& rSes, char Direction);

// GETS
public:	char GetHitched();
		char GetTaut();
		double GetRmseNN(struct rohanContext& rSes, int o, char Option, char cModel) /*! checks sampled outputs vs evaluated outputs and calculates root mean squared error. */ ;

// SETTINGS
		double CUDAverify(struct rohanContext& rSes);
// MENU
		void CUDAShowProperties(struct rohanContext& rSes, int device, FILE* fShow);
// TESTING
public:
		double GetLastRmse(struct rohanContext& rSes, char model);
		int TeamTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTestType, int iTrials, int iBlocks, int iThreads, int iSampleQtyReq, int o);
		double RmseEvaluateTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iSampleQty); /// runs tests for RMSE and evaluation on both host and GPU
		int ClassifyTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iSampleQty); /// runs classification tests on both host and GPU
		double BackPropTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iThreads, int iSampleQty); /// runs tests for backward propagation on both host and GPU
		void SetTestModels(char xp, char control);
// MISC
public:
		void TeamLog(struct rohanContext& rSes, int iRank, char * sLogEntry); // dupe to call cBarge's RLog from CUDA C kernel launchers
		int LetTrainNNThresh( rohanContext& rSes, int o, char chMethod, double dTargetRMSE, int iEpochLength, int iEpochQty, char cModel);
		int LetTrainNNThreshGGG( rohanContext& rSes, int o, char chMethod, double dTargetRMSE, int iEpochLength, int iEpochQty, char cModel);
private:
		void UpdateRmseRecord(struct rohanContext& rSes, double lastRmse, double bestRmse, char cModel);
		double CalcRmseSerial(struct rohanContext& rSes, int o);
		int TrainNNThresh(struct rohanContext& rSes, int bChangeWeights);
		int TrainNNThreshGGG(struct rohanContext& rSes, int bChangeWeights);
};

#endif;