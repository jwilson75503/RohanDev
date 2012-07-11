#include "cRamp.h"
/* Includes, cuda */
#include "stdafx.h"



extern int gDebugLvl, gDevDebug, gTrace;
extern float gElapsedTime, gKernelTimeTally;

//////////////// class cRamp begins ////////////////
cRamp::~cRamp(void)
{
}

int cRamp::SetContext( rohanContext& rC)
{/// enables pointer access to master context struct
	rSes = &rC;
	rLearn = rC.rLearn;
	rNet = rC.rNet;
	return 0;
}

int cRamp::SetBarge( class cBarge * cbBarge)
{mIDfunc/// enables pointer access to active Barge object
	Barge = cbBarge;
	return 0;
}


int cRamp::SetDrover( class cDrover * cdDrover)
{/// enables pointer access to active Drover object
	Drover = cdDrover;
	return 0;
}


int cRamp::SetTeam( class cTeam * ctTeam)
{/// enables pointer access to active Team object
	Team = ctTeam;
	return 0;
}


