#ifndef CRAMP_H
#define CRAMP_H

#pragma once
#include <fstream>
#include "stdafx.h"

class cRamp
{		struct rohanContext * rSes;
		struct rohanLearningSet * rLearn;
		struct rohanNetwork * rNet;
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
		class cTeam * Team /*! The calculating "engine" currently in use. */;

public:
	cRamp( struct rohanContext& rSes);
	~cRamp(void);
	int SetContext( struct rohanContext& rSes); // completed
	int LoadSampleSet(struct rohanContext& rSes, char *sFileName); /// pulls in values from .txt files, used for testing before main loop
	int cuLoadSamples(struct rohanContext& rSes, FILE *fileInput); /// pulls in values from .txt files, used for testing before main loop
			
	int GetFileHandle(char *sFilePath, char *sFileName, char cRW, char cAB, FILE **fileOutput);
	
	int LoadNNWeights(struct rohanContext &rSes, char *sFileName);
	int SaveNNWeights(struct rohanContext& rSes, char cModel);
	int cuRandomizeWeightsBlock(struct rohanContext& rSes);
	int LetWriteEvals(struct rohanContext& rSes, struct rohanLearningSet& rLearn, char * sFileAscii, char cModel); /// saves evaluated output values to disk
private:
	typedef std::vector<cuDoubleComplex> cxVec;
	int MoveKQbabWeights(struct rohanContext &rSes, FILE *fileHandle, char cKQ, char cRW, char cAB, cxVec& KQbab);
	int cuSaveNNWeightsBIN(struct rohanContext &rSes, FILE *fileOutput, char cModel);
	int cuSaveNNWeightsASCII(struct rohanContext &rSes, FILE *fileOutput, char cModel);
	int cuNNLoadWeights(struct rohanContext &rSes, FILE *fileInput);
	int cuRandomizeWeightsLayer(struct rohanContext& rSes);
	int TokenCounter(const char * String, char * Delimiters);
	int GetNthToken( char * String,  char * Delimiters, char Token[255], int N);
public:
	int CharRemover(char * String, char Remove);
};

#endif
