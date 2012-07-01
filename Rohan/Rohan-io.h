/* Includes, cuda */

int BinaryFileHandleRead(char* sFileName, FILE** fileInput)
;
int BinaryFileHandleWrite(char *sFileName, FILE **fileOutput)
;
int AsciiFileHandleRead(char *sFileName, FILE **fileInput)
;
int AsciiFileHandleWrite(char *sFilePath, char *sFileName, FILE **fileOutput)
;
int AsciiWeightDump(struct rohanContext& rSes, FILE *fileOutput)
;
int LoadNNWeights(int iLayerQTY, int iNeuronQty[], double ****dWeightsR, double ****dWeightsI, FILE *fileInput)
;
int cuMessage(cublasStatus csStatus, char *sName, char *sCodeFile, int iLine, char *sFunc)
;
int cuMakeLayers(int iInputQty, char *sLayerSizes, struct rohanContext& rSes)
;
int cuMakeArchValues(char *sMLMVNarch, struct rohanContext& rSes)
;
int cuLoadSampleSet(struct rohanContext& rSes, FILE *fileInput)
;
int cuReLoadSampleSet(struct rohanContext& rSes, FILE *fileInput)
;
int cuNNLoadWeights(struct rohanContext& rSes, FILE *fileInput)
;
int cuPreSaveNNWeights(struct rohanContext& rSes, char cVenue)
;
int cuSaveNNWeights(struct rohanContext& rSes, FILE *fileOutput)
;
int cuSaveNNWeightsASCII(struct rohanContext& rSes, FILE *fileOutput)
;
int devCopyArchValues(struct rohanContext& rSes)
;
int devCopySampleSet(struct rohanContext& rSes)
;
int devCopyNNWeights(struct rohanContext& rSes)
;
int devPrepareNetwork(struct rohanContext& rSes)
;
