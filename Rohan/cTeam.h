#ifndef CTEAM_H
#define CTEAM_H


class cDeviceTeam// revised to remove subclassing: public cTeam // cTeam members available at their original access
{/// A team of mighty stallions; parallel calculations in CUDA C running on the GPU
		struct rohanContext * rSes;
		struct rohanNetwork * rNet;
		struct rohanLearningSet * rLearn;
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
		int lIterationQty /*! Iterate this many times ofr sample set and stop. */;
		char bHitched;
		char bTaut;
	public:
		int SetContext( struct rohanContext& rSes); // completed
		int SetNetwork( struct rohanNetwork& rNet); // completed
		int SetSamples( struct rohanLearningSet& rLearn); // completed
		int SetDrover( class cDrover * cdDrover); // completed
		int SetBarge( class cBarge * cbBarge); //completed
		int GetTrainables( struct rohanContext& rSes, int lSampleQtyReq);
		//int GetEvalSingleSample( struct rohanContext& rSes, int lSampleIdxReq, char chMethod); /// implemented in cHostTeam, cDeviceTeam
		//int LetBackpropSingleSample( rohanContext& rSes, int lSampleIdxReq, char chMethod); /// implemented in cHostTeam, cDeviceTeam
		//int LetTrainNNThresh( rohanContext& rSes, int lSampleQtyReq, char chMethod); // completed
		int SaveContext(int iMode);
		int SaveWeights(int iMode);
		int SaveEvaluation(int iMode);
		int SetStopCriteria( int lIterationQty, double dTargetRMSE); /// specifies conditions for end of training task
		int LetEvalSet( rohanContext& rS, char chMethod); /// Submits a subset of the samples available forevaluation.
	cDeviceTeam( struct rohanContext& rSes); /// ctor body in source
		void ShowMe() /*! diagnostic identity display on screen */;
		int LetHitch(struct rohanContext& rSes) /*! copy data to dev mem and attach structures to team */;
			int TransferContext(struct rohanContext& rSes, char Direction) /*! copy rSes 0D members to dev mem */;
			int CopyNet(struct rohanContext& rSes, char Direction) /*! copy rNet members to dev mem */;
				int TransferNet(struct rohanContext& rSes, char Direction) /*! copy Net params to dev mem */;
			int CopyLearnSet(struct rohanContext& rSes, char Direction) /*! copy rLearn members to dev mem */;
		double GetRmseNN(struct rohanContext& rSes, int o, char Option, char Venue) /*! checks sampled outputs vs evaluated outputs and calculates root mean squared error. */ ;
			int LetTaut(struct rohanContext& rSes) /*! update dev mem from host for epoch */;
				int TransferLayers(struct rohanContext& rSes, char Direction);
			int LetSlack(struct rohanContext& rSes) /*! update host mem with results of epoch calculations */;
				int TransferOutputs(struct rohanContext& rSes, char Direction);
		int LetUnHitch(struct rohanContext& rSes) /*! release dev mem structures */;
		int GetEvalSingleSample( struct rohanContext& rSes, int lSampleIdxReq, char chMethod) /*! calculates NN outputs for a given sample with GPU method */;
		int LetBackpropSingleSample( rohanContext& rSes, int lSampleIdxReq, int o, char chMethod) /*! procedure for training weights with MAX criterion */;
		int LetTrainNNThresh( rohanContext& rSes, int o, char chMethod, double dTargetRMSE, int iEpochLength, char Venue);
		char GetHitched();
		char GetTaut();
		void RLog(struct rohanContext& rSes, char * sLogEntry); // dupe to call cBarge's Rlog from CUDA C kernel launchers
		double RmseEvaluateTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iSampleQty); /// runs tests for RMSE and evaluation on both host and GPU
		int ClassifyTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iSampleQty); /// runs classification tests on both host and GPU
		double BackPropTest(struct rohanContext& rSes, struct rohanNetwork& rNet, int iTrials, int iThreads, int iSampleQty); /// runs tests for backward propagation on both host and GPU
		double CUDAverify(struct rohanContext& rSes);
			void CUDAShowProperties(struct rohanContext& rSes, int device, FILE* fShow);
};

#endif