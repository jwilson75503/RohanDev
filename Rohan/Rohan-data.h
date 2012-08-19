/* Includes, cuda */


int cuMakeNNStructures(struct rohanContext& rSes)
;
int cuSectorTableMake(struct rohanContext& rSes)
;
cuDoubleComplex CxActivate(const cuDoubleComplex Z, struct rohanNetwork& Net)
;
int cuConvertInputs16(struct rohanContext& rSes, int lSample, cuDoubleComplex * Sums)
;
int cuEvalMidTopLayers16(struct rohanContext& rSes, int lSample, cuDoubleComplex * Sums)
;
int cuOutputConvert16(struct rohanContext& rSes, int lSample, cuDoubleComplex * Sums)
;
int cuConvertInputs(struct rohanContext& rSes, int lSample)
;
int cuEvalMidTopLayers(struct rohanContext& rSes, int lSample)
;
int cuOutputConvert(struct rohanContext& rSes, int lSample)
;
int cuEvalNNLearnSet(struct rohanContext& rSes, int o)
;
////int cuFreeNNTop(struct rohanContext& rSes);
int cuFreeLearnSet(struct rohanContext& rSes)
;
int cuFree(struct rohanContext& rSes)
;


int devCopyNNStructures(struct rohanContext& rSes)
;
int devCopySectorTable(struct rohanContext& rSes)
;
cuDoubleComplex ConvPhaseCx(struct rohanContext& rSes, double Phase)
;
cuDoubleComplex ConvSectorCx(struct rohanContext& rSes, double Sector)
;
