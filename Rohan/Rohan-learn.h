/* Includes, cuda */
#ifndef ROHAN_LEARN_H
#define ROHAN_LEARN_H

//int cuEvalSingleSampleBeta(struct rohanContext& Ses, int s, rohanNetwork& Net, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
int cuEvalSingleSampleBeta(struct rohanContext& rSes, int s, rohanNetwork& Net, rohanLearningSet& Learn, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;
int OutputValidate(rohanContext& rSes)
;
int cuResetAllDeltasAndOutputs(struct rohanContext& rSes)
;
int cuBackpropSingleSample(rohanContext& rSes, int s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;
int cuBackpropSingleSampleGGG(rohanContext& rSes, int s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
;

#endif