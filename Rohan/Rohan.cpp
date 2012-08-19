/*! \mainpage Rohan Multivalued Neural Network Simulator
 *
 * \section intro_sec Introduction
 *
 * Rohan was developed by Jeff Wilson <jwilson@clueland.com> at the Texas A&M
 *
 * University - Texarkana Computational Intelligence Laboratory 
 *
 * < http://www.tamut.edu/CIL/ > under the direction of Dr Igor Aizenberg.
 *
 * Rohan is part of research funded by National Science Foundation grant #0925080.
 *
 */

// Rohan.cpp : Defines the entry point for the console application.

/* Includes */
#include "stdafx.h"


/// globals
int gDebugLvl=0, gDevDebug=0, gTrace=0;
float gElapsedTime=0.0, gKernelTimeTally=0.0;


int _tmain(int argc, _TCHAR* argv[])
{mIDfunc/// general program procedure is to setup preparations for the duty loop, execute it, then do housekeeping after
	try{
		fprintf(stdout, "Rohan %s Neural Net Simulator - %s\n", VERSION, AUTHORCREDIT);
		
		/* The session context holder, learning set, and network need to be structs since they are passed to CUDA C code that doesn't do classes */
		struct rohanContext rSes;
		struct rohanNetwork rNet;
		struct rohanLearningSet rLearn;
		
		rSes.rNet=&rNet;
		rSes.rLearn=&rLearn;
		
		// create class objects
		cBarge Barge(rSes); // the barge holds common data like the learning set and weights
		cDrover Drover(rSes); // the drover handles the user input and bosses the other objects
		cRamp Ramp(rSes); // the ramp loads and shuffles data 
		cTeam Team(rSes); // the horse team does the work of computation
		Drover.SetContext(rSes, argc, argv); 

		//FILE *fOut; fOut=fopen("Rohan.txt", "w");
		////Ramp.GetFileHandle(".", "Rohan.roh", 'w', 'a', &fOut);
		//for(int i=0; i<=argc; ++i)
		//	fprintf(fOut, "%s\n", argv[i]);
		//fclose(fOut);
		

		if ( Drover.DoAnteLoop(rSes, argc, argv) ) // prepare data structures and load parameters
			Drover.DoMainLoop(rSes); // proceed with operations based on session variables and external settings
		Drover.DoPostLoop(rSes); // terminates sim "gracefully"
		
		exit (0); // end of operations
	}
	catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        exit (1);
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
		exit (1);
    }
}
