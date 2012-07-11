#ifndef CRAMP_H
#define CRAMP_H

#pragma once

class cRamp
{		struct rohanContext * rSes;
		struct rohanLearningSet * rLearn;
		struct rohanNetwork * rNet;
		class cBarge * Barge /*! The data-holding "object" currently in use. */;
		class cDrover * Drover /*! The user-agent "driver" currently in use. */;
		class cTeam * Team /*! The calculating "engine" currently in use. */;

public:
	cRamp( struct rohanContext& rSes){ SetContext(rSes); } ; // end ctor
	~cRamp(void);
	int SetContext( struct rohanContext& rSes); // completed
	int SetBarge( class cBarge * cbBarge);
	int SetDrover( class cDrover * cdDrover);
	int SetTeam( class cTeam * ctTeam);
};

#endif
