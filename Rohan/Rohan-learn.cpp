/* Includes, cuda */
#include "stdafx.h"

extern int gDebugLvl, gDevDebug, gTrace;


int cuEvalSingleSampleBeta(struct rohanContext& rSes, int s, rohanNetwork& Net, rohanLearningSet& Learn, int o, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{// Beta uses fixed length fields instead of nested pointer layers
	// o is currently used to control diagnostic output
	 /*! layer zero (inputs) is special. */
	int INROWLEN=Net.iNeuronQTY[0];//rSes.rLearn->iInputQty+1;
	for (int i=0; i<INROWLEN; ++i){
		Signals[Net.iNeuronOfst[0]+i]= XInputs[IDX2C( i, s, INROWLEN )];
	}
	 /*! middle and top layers. */
	for (int L=1; L<Net.iLayerQTY; ++L){
		//struct rohanLayer& lay = Net.rLayer[L];
		int LAY=L;
		int TRIB=L-1; // index of previous layer
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=0; k<iNeuronQTY; ++k){ //Neuron zero is not skipped, its output should be 1+0i as a check
			Zs[Net.iNeuronOfst[LAY]+k]=cdcZero;
			
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
			   			 Zs[Net.iNeuronOfst[LAY]+k] = 
				CxAddCx( Zs[Net.iNeuronOfst[LAY]+k] , 
					CxMultiplyCx(
						Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )],
						Signals[Net.iNeuronOfst[TRIB]+i] ) ) ;
			}
			// ACTIVATE //
			Signals[Net.iNeuronOfst[LAY]+k] = CxActivate( Zs[Net.iNeuronOfst[LAY]+k] , Net );
		}
	}
	 
	/*! last layer values are converted and stored here */
	int TOP = Net.iLayerQTY-1;
	int OUTROWLEN=Net.iNeuronQTY[TOP];// also equal to Learn.iOutputQty+1
	
	for (int i=0; i<OUTROWLEN; ++i){ // continuous conversion begins here 
		YEval[IDX2C( i, s, OUTROWLEN )]= Signals[Net.iNeuronOfst[TOP]+i] ; // store final complex output(s)
		dYEval[IDX2C( i, s, OUTROWLEN )]=FUnitCx( YEval[IDX2C( i, s, OUTROWLEN )] ) * Net.iSectorQty; // convert final complex outputs to sectors and store that
		if(rSes.rLearn->iContOutputs==false) // round off decimal if disc activation is set
			dYEval[IDX2C( i, s, OUTROWLEN )]=floor(dYEval[IDX2C( i, s, OUTROWLEN )]);
		double Delta = abs( Learn.dDOutputs[IDX2C( i, s, OUTROWLEN)] - dYEval[IDX2C( i, s, OUTROWLEN )] ); // delta = Desired - Yielded values
		if (Delta > Net.kdiv2 ) 
			Delta = Net.iSectorQty - Delta; // set delta to the lesser arc length
		Learn.dSqrErr[IDX2C( i, s, OUTROWLEN )] = Delta * Delta;
	}
	//rSes.Barge->RLog(rSes, 0, __FUNCTION__);
	/*! end of sample evaluation. */
	return true;
}


int OutputValidate(rohanContext& rSes)
{mIDfunc  /*! compares outputs by sample and by method to verify bulk outputs.. */
	int iReturn=0;
	int ROWLEN = rSes.rLearn->iOutputQty+1;
	double dDiff;
	
	for(int s=0; s<rSes.lSampleQtyReq; ++s){
		for(int i=0; i<=rSes.rLearn->iOutputQty; ++i){
			//printf("%d %d %f %f\t", s, i, rSes.rLearn->dYEval[ IDX2C( s, i, ROWLEN ) ] , rSes.rLearn->dAltYEval[ IDX2C( s, i, ROWLEN ) ] );
			dDiff=abs(
				rSes.rLearn->dYEval[    IDX2C( i, s, ROWLEN ) ] - 
				rSes.rLearn->dAltYEval[ IDX2C( i, s, ROWLEN ) ] 
				);			
				if( dDiff > 0.05){
					++iReturn;
				}
		}
	}
	char sLog[255];
	sprintf(sLog, "%s exits with %d output differences of %d samples compared.", __FUNCTION__, iReturn, rSes.lSampleQtyReq);
	rSes.Barge->RLog(rSes, 0, sLog);
	return iReturn; // return number of outputs that diverged
}


int cuResetAllDeltasAndOutputs(rohanContext& rSes)
{mIDfunc
	for (int L=1; L<rSes.rNet->iLayerQTY; ++L)  /*! reset outputs and deltas for full neuron layers. */
		for (int i = 0; i<=rSes.rNet->rLayer[L].iNeuronQty; ++i){
			rSes.rNet->rLayer[L].Deltas[i]=cdcZero;
			rSes.rNet->rLayer[L].ZOutputs[i]=cdcZero;
		}
	for (int i = 0; i< MAXNEURONS; ++i){
		rSes.rNet->Deltas[i]=cdcZero;
		rSes.rNet->Signals[i]=cdcZero;
		rSes.rNet->Zs[i]=cdcZero;
	}
	//rSes.Barge->RLog(rSes, 0, __FUNCTION__);
	
	return 0;
}


int cuBackpropSingleSample(rohanContext& rSes, int s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{ mIDfunc /*! propagates adjustment of weights backwards preceeding layers from the chosen network output. */
	// s is sample index
	int iReturn=0 /* returns number of weights adjusted */ ;
	int ROWLEN = rSes.rLearn->iOutputQty+1;
	/* clear all temp values BP0 */
	cuResetAllDeltasAndOutputs(rSes);
	/* re-evaluate sample to load temp values. BPI */
	cuEvalSingleSampleBeta(rSes, s, Net, *rSes.rLearn, 0, Signals, Zs, Wt, XInputs, YEval, dYEval);
	/* begin error calculation. BPII */
	cuDoubleComplex Deltastar /* measured error at the chosen network output. */ ;
	int TOP=Net.iLayerQTY-1;
	/* calc top layer deltas. */
	for(int i=1; i<Net.iNeuronQTY[TOP]; ++i){ //starting at 1 instead of 0 since 8/12
		 /* delta-star = D - Y = Desired output minus actual output from evaluation
		 /* D is the cplx coords of the sector of the desired answer		Y is the complex result of evaluation of the given sample, unactivated. */
		Deltastar = CxSubtractCx( 
						rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ], 
						Signals[Net.iNeuronOfst[TOP]+i] );
		 /* divide the correction; delta = alpha * delta-star / n+1 (but alpha is always 1 for now). */
		Deltas[Net.iNeuronOfst[TOP]+i] = CxMultiplyRl( Deltastar, Net.dINV_S[TOP] );
	}
	/* Now distribute the correction to lower layers if any. BPII.1 */
	if (Net.iLayerQTY>2){  /* remember layer 0 = inputs, layer 1 = bottom row, layer {2..iLayerQTY-2} = middle row, layer iLayerQTY-1 = top row. */
		for (int L=Net.iLayerQTY-1; L>1; --L){
			int LAY = L; /* setup access to layers. */
			int TRIB = L-1; /* trib for tributary.*/
			int iTributQTY=Net.iNeuronQTY[TRIB];
			int Sj=Net.iDendrtQTY[TRIB]; if (TRIB==1) Sj=1; // Sj=1 for firest hidden layer
			for (int i=1; i<Net.iNeuronQTY[LAY]; ++i) { // skip 0th neuron as its weights are either 1 (div identity) or 0 (div forbidden) and don't change anyway
				// k index must begin at 1, neuron zero not valid for correction
				for (int k=1; k<iTributQTY; ++k) { /* the contribution to ith neuron's kth tributary's delta = i's delta/i's weight k. */
								Deltas[Net.iNeuronOfst[TRIB]+k] 
					= CxAddCx ( Deltas[Net.iNeuronOfst[TRIB]+k] , 
						CxDivideCx( 
							Deltas[Net.iNeuronOfst[LAY]+i] , 
							Wt[IDX2C( Net.iWeightOfst[LAY]+k, i, iTributQTY )] ));
					++iReturn;
				}
			}
			// k index must begin at 1, neuron zero not valid for correction
			for (int k=1; k<iTributQTY; ++k) { /* contributions accumulated, now divide by dendrites+1. */
				Deltas[Net.iNeuronOfst[TRIB]+k] 
					= CxMultiplyRl( 
						Deltas[Net.iNeuronOfst[TRIB]+k] , 
						Net.dINV_S[TRIB] );
			}
		}
	}
	/* error distribution completed */
	/* and now update the weights BP III */
	/* adj weights on first hidden layer. */
		int FHID = 1;
		int SIG = 0;
		int iSignalQTY=rSes.rLearn->iInputQty+1;
		int iHidWidth=Net.iNeuronQTY[FHID];
	for (int k=1; k<iHidWidth; ++k){
		for (int i=0; i<iSignalQTY; ++i){  
			/* dW=d*xbar/s1/|z|= neuron's delta * input's conjugate / ( dendrites+1 * abs of input i ). */
					  Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )]
			=CxAddCx( Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )] , 
				CxDivideRl( 
					CxMultiplyCx( 
						Deltas[Net.iNeuronOfst[FHID]+k] , 
						CxConjugate( Signals[Net.iNeuronOfst[SIG]+i] ) 
					) , 
					CxAbs( Zs[Net.iNeuronOfst[FHID]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
				)
			);
			++iReturn;
		}
	}
	/* re-evaluate sample to update temp values. */
	cuEvalSingleSampleBeta(rSes, s, Net, *rSes.rLearn, 0 ,Signals, Zs, Wt, XInputs, YEval, dYEval);
	if (Net.iLayerQTY>2){
		 /* now use those outputs' conjugates and the deltas to adjust middle layers. BP III.1 */
		for (int L=2; L<Net.iLayerQTY-1; ++L){
			 /* setup access to layers. */
			//struct rohanLayer& lay = Net.rLayer[L]; 
			int LAY = L;
			//struct rohanLayer& trib = Net.rLayer[L-1] /* trib for tributary. */ ; 
			int TRIB = L-1;
			int iLayWidth=Net.iNeuronQTY[LAY];
			int iTribWidth=Net.iNeuronQTY[TRIB];
			for (int k=1; k<Net.iNeuronQTY[LAY]; ++k){
				for (int i=0; i<Net.iNeuronQTY[TRIB]; ++i){  
					/* the adjustment added to kth neuron's ith trib's weight = k's delta * complex conjugate of i's signal / (abs of k's previous-wt product-sum * dendrites+1)  . */
							  Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )]
					=CxAddCx( Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )] , 
						CxDivideRl( 
							CxMultiplyCx( 
								Deltas[Net.iNeuronOfst[LAY]+k] , 
								CxConjugate( Signals[Net.iNeuronOfst[TRIB]+i] ) 
							) ,
							( 
								CxAbs( Zs[Net.iNeuronOfst[LAY]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
							)
						)
					);
					++iReturn;
				}
			}
			/* layer is complete. */
			cuEvalSingleSampleBeta(rSes, s, Net, *rSes.rLearn, false, Signals, Zs, Wt, XInputs, YEval, dYEval);
		}
	}
	/* correct output layer BP III.3 */
	int SUB = TOP-1; 
	int iTopWidth=Net.iNeuronQTY[TOP];
	int iSubWidth=Net.iNeuronQTY[SUB];
			
	for (int k=1; k<Net.iNeuronQTY[TOP]; ++k){
		for (int i=0; i<Net.iNeuronQTY[SUB]; ++i){  
			/* For last layer only, adjustment to kth neuron's ith weight = k's delta * complex conjugate of i's signal / ( dendrites+1)  . */
					  Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )]
			=CxAddCx( Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )] , 
				CxMultiplyCx( 
					Deltas[Net.iNeuronOfst[TOP]+k] , 
					CxConjugate( Signals[Net.iNeuronOfst[SUB]+i] ) 
				)
			);  // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
			++iReturn;
		}
	}
	/* backprop is complete. */
	//rSes.Barge->RLog(rSes, 0, __FUNCTION__);
	
	return iReturn; /* number of weights updated. */
}

int cuBackpropSingleSampleGGG(rohanContext& rSes, int s, rohanNetwork& Net, cuDoubleComplex * Signals, cuDoubleComplex * Zs, cuDoubleComplex * Wt, cuDoubleComplex * Deltas, cuDoubleComplex * XInputs, cuDoubleComplex * YEval, double * dYEval )
{ mIDfunc /*! propagates adjustment of weights backwards preceeding layers from the chosen network output. */
	// s is sample index
	int iReturn=0 /* returns number of weights adjusted */ ;
	int ROWLEN = rSes.rLearn->iOutputQty+1;
	/* clear all temp values BP0 */
	cuResetAllDeltasAndOutputs(rSes);
	/* re-evaluate sample to load temp values. BPI */
	cuEvalSingleSampleBeta(rSes, s, Net, *rSes.rLearn, 0, Signals, Zs, Wt, XInputs, YEval, dYEval);
	/* begin error calculation. BPII */
	cuDoubleComplex Deltastar /* measured error at the chosen network output. */ ;
	int TOP=Net.iLayerQTY-1;
	fprintf(rSes.hostBucket,"\tBEGIN Backprop sample %d with %d outputs, TOP layer is %d of %d\n(full detail only on neuron 1 of each layer)\n", s, ROWLEN, TOP, Net.iLayerQTY);
	/* calc top layer deltas. */
	for(int i=1; i<Net.iNeuronQTY[TOP]; ++i){
		 /* delta-star = D - Y = Desired output minus actual output from evaluation
		 /* D is the cplx coords of the sector of the desired answer		Y is the complex result of evaluation of the given sample, unactivated. */
		Deltastar = CxSubtractCx( 
						rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ], 
						Signals[Net.iNeuronOfst[TOP]+i] );
		fprintf(rSes.hostBucket, "Output %d: d* %f+%f = D %f+%f - Y %f+%f\n", i, Deltastar.x, Deltastar.y, 
			rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].x, rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].y, 
			Signals[Net.iNeuronOfst[TOP]+i].x, Signals[Net.iNeuronOfst[TOP]+i].y );
		/* divide the correction; delta = alpha * delta-star / n+1 (but alpha is always 1 for now). */
		Deltas[Net.iNeuronOfst[TOP]+i] = CxMultiplyRl( Deltastar, Net.dINV_S[TOP] );
		fprintf(rSes.hostBucket, "d(%d,L%d) %f+%f = d* %f+%f * 1/%d %f\n", i, TOP,
			Deltas[Net.iNeuronOfst[TOP]+i].x, Deltas[Net.iNeuronOfst[TOP]+i].y, 
			Deltastar.x, Deltastar.y, (int)floor(.5+1/Net.dINV_S[TOP]), Net.dINV_S[TOP] );
	}
	/* Now distribute the correction to lower layers if any. BPII.1 */
	if (Net.iLayerQTY>2){  /* remember layer 0 = inputs, layer 1 = bottom row, layer {2..iLayerQTY-2} = middle row, layer iLayerQTY-1 = top row. */
		for (int L=Net.iLayerQTY-1; L>1; --L){
			int LAY = L; /* setup access to layers. */
			int TRIB = L-1; /* trib for tributary.*/
			fprintf(rSes.hostBucket, "Distribute correction layer %d to %d\n", LAY, TRIB);
			int iTributQTY=Net.iNeuronQTY[TRIB];
			int Sj=Net.iDendrtQTY[TRIB]; if (TRIB==1) Sj=1; // Sj=1 for firest hidden layer
			for (int i=1; i<Net.iNeuronQTY[LAY]; ++i) { // skip 0th neuron as its weights are either 1 (div identity) or 0 (div forbidden) and don't change anyway
				// k index must begin at 1, neuron zero not valid for correction
				for (int k=1; k<iTributQTY; ++k) { /* the contribution to ith neuron's kth tributary's delta = i's delta/i's weight k. */
					fprintf(rSes.hostBucket, "d(%d,L%d) %f+%f", k, TRIB, Deltas[Net.iNeuronOfst[TRIB]+k].x, Deltas[Net.iNeuronOfst[TRIB]+k].y);
								Deltas[Net.iNeuronOfst[TRIB]+k] 
					= CxAddCx ( Deltas[Net.iNeuronOfst[TRIB]+k] , 
						CxDivideCx( 
							Deltas[Net.iNeuronOfst[LAY]+i] , 
							Wt[IDX2C( Net.iWeightOfst[LAY]+k, i, iTributQTY )] ));
					++iReturn;
					fprintf(rSes.hostBucket, " += d(%d,L%d) %f+%f / w(%d,L%d,%d) %f+%f = %f+%f\n", i, LAY, Deltas[Net.iNeuronOfst[LAY]+i].x, Deltas[Net.iNeuronOfst[LAY]+i].y, 
						k, LAY, i, Wt[IDX2C( Net.iWeightOfst[LAY]+k, i, iTributQTY )].x, Wt[IDX2C( Net.iWeightOfst[LAY]+k, i, iTributQTY )].y,
						Deltas[Net.iNeuronOfst[TRIB]+k].x, Deltas[Net.iNeuronOfst[TRIB]+k].y);
				}
			}
			// k index must begin at 1, neuron zero not valid for correction
			for (int k=1; k<iTributQTY; ++k) { /* contributions accumulated, now divide by dendrites+1. */
				fprintf(rSes.hostBucket, "d(%d,L%d) %f+%f", k, TRIB, Deltas[Net.iNeuronOfst[TRIB]+k].x, Deltas[Net.iNeuronOfst[TRIB]+k].y);
				Deltas[Net.iNeuronOfst[TRIB]+k] 
					= CxMultiplyRl( 
						Deltas[Net.iNeuronOfst[TRIB]+k] , 
						Net.dINV_S[TRIB] );
				fprintf(rSes.hostBucket, " *= 1/%d %f = %f+%f\n", (int)floor(.5+1/Net.dINV_S[TRIB]), Net.dINV_S[TRIB], Deltas[Net.iNeuronOfst[TRIB]+k].x, Deltas[Net.iNeuronOfst[TRIB]+k].y);
			}
		}
	}
	/* error distribution completed */
	/* and now update the weights BP III */
	/* adj weights on first hidden layer. */
		int FHID = 1;
		int SIG = 0;
		int iSignalQTY=rSes.rLearn->iInputQty+1;
		int iHidWidth=Net.iNeuronQTY[FHID];
	fprintf(rSes.hostBucket, "Error distribution complete, adjustment of weights begin\n");
	for (int k=1; k<iHidWidth; ++k){
		for (int i=0; i<iSignalQTY; ++i){  
			/* dW=d*xbar/s1/|z|= neuron's delta * input's conjugate / ( dendrites+1 * abs of input i ). */
			if(k==1)fprintf(rSes.hostBucket, "W(%d,L%d,%d) %f+%f + ( d(%d,L%d) %f+%f * Xbar(%d) %f+%f ) / |Z(%d) %f+%f| %d", 
				k, FHID, i, Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )].x, Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )].y,
				k, FHID, Deltas[Net.iNeuronOfst[FHID]+k].x, Deltas[Net.iNeuronOfst[FHID]+k].y,
				i, Signals[Net.iNeuronOfst[SIG]+i].x, -1*Signals[Net.iNeuronOfst[SIG]+i].y,
				k, Zs[Net.iNeuronOfst[FHID]+k].x, Zs[Net.iNeuronOfst[FHID]+k].y, CxAbs( Zs[Net.iNeuronOfst[FHID]+k] ) ); 
				
					  Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )]
			=CxAddCx( Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )] , 
				CxDivideRl( 
					CxMultiplyCx( 
						Deltas[Net.iNeuronOfst[FHID]+k] , 
						CxConjugate( Signals[Net.iNeuronOfst[SIG]+i] ) 
					) , 
					CxAbs( Zs[Net.iNeuronOfst[FHID]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
				)
			);
			if(k==1)fprintf(rSes.hostBucket, " = ~W(%d,L%d,%d) %f+%f\n",
				k, FHID, i, Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )].x, Wt[IDX2C( Net.iWeightOfst[FHID]+i, k, iSignalQTY )].y );
			++iReturn;
		}
	}
	/* re-evaluate sample to update temp values. */
	fprintf(rSes.hostBucket, "Re-evaluate sample to update temp values.\n");
	cuEvalSingleSampleBeta(rSes, s, Net, *rSes.rLearn, 0 ,Signals, Zs, Wt, XInputs, YEval, dYEval);
	if (Net.iLayerQTY>2){
		 /* now use those outputs' conjugates and the deltas to adjust middle layers. BP III.1 */
		for (int L=2; L<Net.iLayerQTY-1; ++L){
			 /* setup access to layers. */
			//struct rohanLayer& lay = Net.rLayer[L]; 
			int LAY = L;
			//struct rohanLayer& trib = Net.rLayer[L-1] /* trib for tributary. */ ; 
			int TRIB = L-1;
			int iLayWidth=Net.iNeuronQTY[LAY];
			int iTribWidth=Net.iNeuronQTY[TRIB];
			for (int k=1; k<Net.iNeuronQTY[LAY]; ++k){
				for (int i=0; i<Net.iNeuronQTY[TRIB]; ++i){  
					/* the adjustment added to kth neuron's ith trib's weight = k's delta * complex conjugate of i's signal / (abs of k's previous-wt product-sum * dendrites+1)  . */
				if(k==1)fprintf(rSes.hostBucket, "W(%d,L%d,%d) %f+%f + ( d(%d,L%d) %f+%f * ~Ybar(%d) %f+%f ) / |Z(%d) %f+%f| %d", 
					k, LAY, i, Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )].x, Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )].y,
					k, LAY, Deltas[Net.iNeuronOfst[LAY]+k].x, Deltas[Net.iNeuronOfst[LAY]+k].y,
					i, Signals[Net.iNeuronOfst[TRIB]+i].x, -1*Signals[Net.iNeuronOfst[TRIB]+i].y,
					k, Zs[Net.iNeuronOfst[LAY]+k].x, Zs[Net.iNeuronOfst[LAY]+k].y, CxAbs( Zs[Net.iNeuronOfst[LAY]+k] ) ); 

							  Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )]
					=CxAddCx( Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )] , 
						CxDivideRl( 
							CxMultiplyCx( 
								Deltas[Net.iNeuronOfst[LAY]+k] , 
								CxConjugate( Signals[Net.iNeuronOfst[TRIB]+i] ) 
							) ,
							( 
								CxAbs( Zs[Net.iNeuronOfst[LAY]+k] ) // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
							)
						)
					);
					if(k==1)fprintf(rSes.hostBucket, " = ~W(%d,L%d,%d) %f+%f\n",
						k, LAY, i, Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )].x, Wt[IDX2C( Net.iWeightOfst[LAY]+i, k, iTribWidth )].y );
					++iReturn;
				}
			}
			/* layer is complete. */
			fprintf(rSes.hostBucket, "Re-evaluate sample to update temp values.");
			cuEvalSingleSampleBeta(rSes, s, Net, *rSes.rLearn, false, Signals, Zs, Wt, XInputs, YEval, dYEval);
		}
	}
	/* correct output layer BP III.3 */
	int SUB = TOP-1; 
	int iTopWidth=Net.iNeuronQTY[TOP];
	int iSubWidth=Net.iNeuronQTY[SUB];
			
	for (int k=1; k<Net.iNeuronQTY[TOP]; ++k){
		fprintf(rSes.hostBucket, "Begin final layer adjustment\n");
		for (int i=0; i<Net.iNeuronQTY[SUB]; ++i){  
			/* For last layer only, adjustment to kth neuron's ith weight = k's delta * complex conjugate of i's signal / ( dendrites+1)  . */
				if(k==1)fprintf(rSes.hostBucket, "W(%d,L%d,%d) %f+%f + ( d(%d,L%d) %f+%f * ~Ybar(%d) %f+%f )", 
					k, TOP, i, Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )].x, Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )].y,
					k, TOP, Deltas[Net.iNeuronOfst[TOP]+k].x, Deltas[Net.iNeuronOfst[TOP]+k].y,
					i, Signals[Net.iNeuronOfst[SUB]+i].x, -1*Signals[Net.iNeuronOfst[SUB]+i].y ); 

					  Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )]
			=CxAddCx( Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )] , 
				CxMultiplyCx( 
					Deltas[Net.iNeuronOfst[TOP]+k] , 
					CxConjugate( Signals[Net.iNeuronOfst[SUB]+i] ) 
				)
			);  // N+1 denominator factor is considered redundant - JAW & IA 2/27/12
			if(k==1)fprintf(rSes.hostBucket, " = ~W(%d,L%d,%d) %f+%f\n",
				k, TOP, i, Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )].x, Wt[IDX2C( Net.iWeightOfst[TOP]+i, k, iSubWidth )].y );
			++iReturn;
		}
	}
	/* backprop is complete. */
	fprintf(rSes.hostBucket, "Re-evaluate sample for final values.");
	cuEvalSingleSampleBeta(rSes, s, Net, *rSes.rLearn, false, Signals, Zs, Wt, XInputs, YEval, dYEval);
	for(int i=1; i<Net.iNeuronQTY[TOP]; ++i){
		Deltastar = CxSubtractCx( 
						rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ], 
						Signals[Net.iNeuronOfst[TOP]+i] );
		fprintf(rSes.hostBucket, "~Output %d: d* %f+%f = D %f+%f - ~Y %f+%f\n", i, Deltastar.x, Deltastar.y, 
			rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].x, rSes.rLearn->cdcDOutputs[ IDX2C( i, s, ROWLEN ) ].y, 
			Signals[Net.iNeuronOfst[TOP]+i].x, Signals[Net.iNeuronOfst[TOP]+i].y );
	}

	//rSes.Barge->RLog(rSes, 0, __FUNCTION__);
	
	return iReturn; /* number of weights updated. */
}

