/* Includes, cuda */
#include "stdafx.h"
#include <boost/timer/timer.hpp>

extern int gDebugLvl, gTrace;

int cuSectorTableMake(struct rohanContext &rSes)
{mIDfunc /// allocate and populate an array of complex coordinates for sectors on the unit circle of the complex plane
	double two_pi_div_sect_qty = TWO_PI/rSes.rNet->iSectorQty;

	rSes.rNet->cdcSectorBdry=(cuDoubleComplex*)malloc(rSes.rNet->iSectorQty * sizeof (cuDoubleComplex)); //point to array of cdc's
		mCheckMallocWorked(rSes.rNet->cdcSectorBdry)
	rSes.rNet->cdcAltSectorBdry=(cuDoubleComplex*)malloc(rSes.rNet->iSectorQty * sizeof (cuDoubleComplex)); //point to array of cdc's
		mCheckMallocWorked(rSes.rNet->cdcAltSectorBdry)
	for (int s=0; s<rSes.rNet->iSectorQty; ++s) {
		rSes.rNet->cdcSectorBdry[s].x=cos(s*two_pi_div_sect_qty);
		rSes.rNet->cdcSectorBdry[s].y=sin(s*two_pi_div_sect_qty);
		rSes.rNet->cdcAltSectorBdry[s]=cdcIdentity;
	}
	return rSes.rNet->iSectorQty;
}


cuDoubleComplex CxActivate(const cuDoubleComplex Z, struct rohanNetwork& Net)
{/// applies ContActivation or discrete activation function to cx neuron output and returns Phi(Z)
	/// This fn should be phased out in favor of a GPU device vector based fn
	cuDoubleComplex phi;
	if (Net.iContActivation) { // apply ContActivation activation function to weighted sum : phi(z)=z/|z|
		phi = CxDivideRl( Z, CxAbs( Z ) );
	}
	else {	// apply Discrete activation function to weighted sum : s=int(arctan(z)*k/2pi), phi(z)=(X(s),Y(s))
		double theta = atan2(Z.y, Z.x); // theta = arctan y/x
		int iSector = (int)((theta * Net.dK_DIV_TWO_PI) + Net.iSectorQty) % Net.iSectorQty;
		phi = Net.cdcSectorBdry[iSector];
	}

	if(!_finite(phi.x) || !_finite(phi.y))
		printf("CxActivate: bad value from %f+%f !\n", phi.x, phi.y);

	return phi;
}


int cuEvalNNLearnSet(struct rohanContext& rSes, int o)
{mIDfunc
/*! This will apply a MLMVN weight set to each sample of a learning set in turn and record the resulting final output for each.
 *  Discrete inputs and outputs are used. Real integers are convered via K-valued logic to complex coordinates,
 *  which are then product-summed by successive layers of neurons, then conveted back to integer output
 *  
 *  IMPORTANT: keep this code consistent with cuEvalSingleOutput in rohan-learn.cpp 
 */
	int lSamplesEvaled=0;
	// sample index, counts up
	double two_pi_div_sect_qty = TWO_PI/rSes.rNet->iSectorQty;
	// here beginneth ye main duty loop
	{
		for (int s=0; s<rSes.lSampleQtyReq; s+=1){
			lSamplesEvaled+=cuEvalSingleSampleBeta(rSes, s, *rSes.rNet, *rSes.rLearn, o, rSes.rNet->Signals, rSes.rNet->Zs, rSes.rNet->Wt, rSes.rLearn->cdcXInputs, rSes.rLearn->cdcYEval, rSes.rLearn->dYEval); // fixed-length method working 2/14/12
		}
	}
	return lSamplesEvaled; // return qty samples evaluated
}


cuDoubleComplex ConvSectorCx(struct rohanContext& rSes, double Sector)
{mIDfunc // converts a scalar value to a returned complex coordinate)

	cuDoubleComplex cdcReturn;
	
	if (Sector > rSes.rNet->iSectorQty){
		cdcReturn.x=666.6;
		cdcReturn.y=666.6;
	}
	else {
		double theta=Sector*rSes.rNet->two_pi_div_sect_qty;
		cdcReturn.x=cos( theta);
		cdcReturn.y=sin( theta);			
	}

	return cdcReturn;
}


cuDoubleComplex ConvPhaseCx(struct rohanContext& rSes, double Phase)
{mIDfunc // converts a scalar value to a returned complex coordinate)

	cuDoubleComplex cdcReturn;
	
	cdcReturn.x=cos( Phase);
	cdcReturn.y=sin( Phase);			
	
	return cdcReturn;
}
