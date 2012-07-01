/* Includes, cuda */
#ifndef ROHAN_LEARN_H
#define ROHAN_LEARN_H

////int devSetInputs(struct rohanContext& rSes, int iLayer);

////int devActivate(struct rohanContext& rSes, int iLayer);

////int devEvalSingleSample(struct rohanContext& rSes, int lSampleIdxReq);

int dualEvalSingleSample(struct rohanContext& rSes, int lSampleIdxReq)
;

int cuEvalSingleSampleBeta(struct rohanContext& Ses, int s, rohanNetwork& Net, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;

int cuEvalSingleOutput(rohanContext& rSes, int lSampleIdxReq, int iOutputIdxReq)
;

int OutputValidate(rohanContext& rSes)
;

////int devResetAllDeltasAndOutputs(struct rohanContext& rSes);

int cuResetAllDeltasAndOutputs(struct rohanContext& rSes)
;

////int devBackpropSingleSample(rohanContext& rSes, int lSampleIdxReq);

int dualBackpropSingleSample(struct rohanContext& rSes, int lSampleIdxReq)
;

int cuBackpropLearnSet(rohanContext& rSes, int s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;

int cuBackpropSingleSample(rohanContext& rSes, int s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;

int TrainNNThresh(struct rohanContext& rSes, int bChangeWeights)
;

double RmseNN(struct rohanContext& rSes, int o)
;

void cuCksum(struct rohanContext& rSes)
;

#endif