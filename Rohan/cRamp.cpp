#include "cRamp.h"
/* Includes, cuda */
#include "stdafx.h"



extern int gDebugLvl, gDevDebug, gTrace;
extern float gElapsedTime, gKernelTimeTally;

//////////////// class cRamp begins ////////////////
cRamp::cRamp( struct rohanContext& rSes)
{
	rSes.Ramp=this;
}

cRamp::~cRamp(void)
{
}

int cRamp::SetContext( rohanContext& rC)
{/// enables pointer access to internal objects
	rSes = &rC;
	rLearn = rC.rLearn;
	rNet = rC.rNet;
	Barge=rC.Barge;
	Drover=rC.Drover;
	//Ramp=rC.Ramp;
	Team=rC.Team;

	return 0;
}

int cRamp::TokenCounter(const char * String, char * Delimiters)
{
	char * DupString, * Portion; int TokenCount=0;

	DupString=_strdup(String); // strtok chops up the input string, so we must make a copy
	Portion = strtok (DupString, Delimiters);
	while (Portion != NULL) {// this loop counts the tokens present in a copy of String
		Portion = strtok (NULL, Delimiters); 
		++TokenCount;
	}

	return TokenCount;
}

int cRamp::GetNthToken( char * String,  char * Delimiters, char Token[255], int N)
{
	char * DupString, * Portion; int TokenCount=1;

	DupString=_strdup(String); // strtok chops up the input string, so we must make a copy
	Portion = strtok (DupString, Delimiters);
	while (Portion != NULL && TokenCount<N) {// this loop counts the tokens present in a copy of String
		Portion = strtok (NULL, Delimiters); 
		++TokenCount;
	}
	sprintf(Token, "%s", Portion); // put the selected token in Token
	return TokenCount;
}
int cRamp::CharRemover(char * String, char Remove)
{ // strips unwanted character values from a string, null terminates the shortened string
	char * A, * B;

	A=B=String;
	while (*A){ // until A points to a null
		if(*A!=Remove) // copy non-removable chars
			*B++=*A;
		++A; // advance A
	}
	*B=*A; // copy final null
	return A-B; // return # of removed chars
}


int cRamp::GetFileHandle(char *sFilePath, char *sFileName, char cRW, char cAB, FILE **fileHandle)
{mIDfunc/// Opens a file in specified mode
	char sString[MAX_PATH], sMode[3];

	
	sMode[2]=NULL;
	sMode[1]=( (cAB=='b' || cAB=='B') ? 'b' : NULL );
	sMode[0]=cRW;
	
	if(cRW=='w' || cRW=='W'){
		if(DirectoryEnsure(sFilePath)){
			sprintf(sString, "%s\\%s", sFilePath, sFileName);
			{ // quote removal inserted 7/10/12
				CharRemover(sString, '"');
			} // end quote removal
			*fileHandle = fopen(sString, sMode);  /* Open in requested mode */
			if (*fileHandle == NULL) {
				fprintf(stderr, "Can't open %s for writing.\n", sString);
				return false;
			}
			else return true;
		}
		else{
			fprintf(stderr, "Can't open %s for writing.\n", sFilePath);
			return false;
		}
	}
	else{// read opens here
		char sPath[1024];

			{ // quote removal inserted 7/10/12
				CharRemover(sFileName, '"');
			} // end quote removal

		*fileHandle = fopen(sFileName, sMode);  /* Open in ASCII read mode */
		if (*fileHandle == NULL) {
			sprintf(sPath, "%s\\%s", sFilePath, sFileName);

			{ // quote removal inserted 7/10/12
				CharRemover(sPath, '"');
			} // end quote removal

			*fileHandle = fopen(sPath, sMode);  /* Open in requested mode */
			if (*fileHandle == NULL) {
				fprintf(stderr, "Can't open %s for reading.\n", sPath);
				return 0;
			}
			else return 1;
		}
		else return 1;
	}
}

int cRamp::LoadSampleSet(struct rohanContext& rSes, char *sFileName)
{mIDfunc/// pulls in values from .txt files, used for testing before main loop
		int iReturn=0; char sLog[255];
		FILE *fileInput;
		// File handle for input
		iReturn=GetFileHandle(rSes.sRohanVerPath, sFileName, 'r', 'a', &fileInput);
		if (iReturn==0) {// unable to open file
			sprintf(sLog, "Unable to open \"%s\" for reading", sFileName);
			Barge->RLog(rSes, ERRORF, sLog);
			return 0;
		}	
		else // file opened normally
			// file opening and reading are separated to allow for streams to be added later
			iReturn=cuLoadSamples(rSes, fileInput);

		return iReturn;
}
	
int cRamp::cuLoadSamples(struct rohanContext& rSes, FILE *fileInput)
{mIDfunc/// pulls in values from .txt files, used for testing before main loop
	// Returns lines read in (0 = error)
	// allocates space for rLearn.dXInputs and .dDOutputs and fills them with parsed, unconveted values

	char sLog[255], *pch, *cSample;

	#define MAX_REC_LEN 65536 /* Maximum size of input buffer */

	int  lLinesQty=0,lMaxLines=256; /* countable number of lines, number of lines with memory allocation */
	char cThisLine[MAX_REC_LEN]; /* Contents of current line */
	rSes.rLearn->iValuesPerLine=0; rSes.rLearn->lSampleQty=0;
	// reset quantities for counting later
	char **cLines = (char **)malloc(256 * sizeof (char *));
	// 1d array of lines in text file beginning with first line in position zero
	// learning set format allows for first or first and second lines to he paramteres rather than samples

	while (fgets(cThisLine, MAX_REC_LEN, fileInput)) { //each line is read in turn
		cLines[lLinesQty++] = _strdup(cThisLine); // each line is copied to a string in the array
		if (!(lMaxLines > lLinesQty)) {  // if alloated space is used up, double it.
			lMaxLines *= 2;
			void * temp = realloc(cLines, lMaxLines * sizeof (char *));
			if (!temp) {
				  for (int k=0;k<lLinesQty;++k) {
					  free(cLines[k]);
				  }
				  sprintf(sLog, "Realloc ran out of space?  OH NOES! %s line %d\n", __FILE__, __LINE__);
				  Barge->RLog(rSes, ERRORF, sLog);
				  return 0;
			} else {
				  cLines = (char **)temp;
			}
		}
	}
	fclose(fileInput); // close stream when fgets returns false (no more lines)
	// this should be a shrinking, and should never fail.
	cLines = (char **)realloc(cLines, lLinesQty * sizeof (char*));
		mCheckMallocWorked(cLines)
	double A, B, C; char cToken[255]; int lCurrentLine=0;
	// 0 means Continuous sample values
	// 1 means discrete, and values of 2+ indicate parameter has been omitted,
	// defaulting to discrete, and value is actually # of samples in file
	// rSes.iContReadMode=atof(cLines[0]);
	A=atof(cLines[0]);
	C=atof(cLines[1]);
	if(TokenCounter(cLines[0], " ,\t")>1){ // more than one value on first line ?

		GetNthToken(cLines[0], " ,\t", cToken, 2);
		B=atof(cToken);

		if(A!=floor(A)) { // first value is not an integer, header must have been omitted from a phase file
			lCurrentLine=0; // samples begin on line 0
			rSes.rLearn->lSampleQty=rSes.lSampleQtyReq=rSes.lSampleQty=lLinesQty; // set sample qty equal to lines
			rSes.iContReadMode=0; rSes.rLearn->iContInputs=1; rSes.rLearn->iContOutputs=0; // implied sample value continuity recorded
		}
		if(A==0) { // first value is 0
			lCurrentLine=1; // samples begin on line 1
			rSes.rLearn->lSampleQty=rSes.lSampleQtyReq=rSes.lSampleQty=(int)B; // set sample qty equal to 2nd parameter
			rSes.iContReadMode=0; rSes.rLearn->iContInputs=1; rSes.rLearn->iContOutputs=0; // implied sample value continuity recorded ZZ
		}
		if(A==1) { // first value is 1
			lCurrentLine=1; // samples begin on line 1
			rSes.rLearn->lSampleQty=rSes.lSampleQtyReq=rSes.lSampleQty=(int)B; // set sample qty equal to 2nd parameter
			rSes.iContReadMode=1; rSes.rLearn->iContInputs=0; rSes.rLearn->iContOutputs=0; // implied sample value continuity recorded
		}
		if(A>=2) { // first value is 2+
			lCurrentLine=1; // samples begin on line 1
			rSes.rLearn->lSampleQty=rSes.lSampleQtyReq=rSes.lSampleQty=(int)A; // set sample qty equal to 1st parameter, others on this line are ignored
			rSes.iContReadMode=1; rSes.rLearn->iContInputs=0; rSes.rLearn->iContOutputs=0; // implied sample value continuity recorded
		}
		
	}
	else {
		if(A==0) { // first value is 0
			lCurrentLine=2; // samples begin on line 2
			rSes.rLearn->lSampleQty=rSes.lSampleQtyReq=rSes.lSampleQty=(int)C; // set sample qty equal to 1st parameter on 2nd line
			rSes.iContReadMode=0; rSes.rLearn->iContInputs=1; rSes.rLearn->iContOutputs=0; // implied sample value continuity recorded
		}
		if(A==1) { // first value is 1
			lCurrentLine=2; // samples begin on line 2
			rSes.rLearn->lSampleQty=rSes.lSampleQtyReq=rSes.lSampleQty=(int)C; // set sample qty equal to 1st parameter on 2nd line
			rSes.iContReadMode=1; rSes.rLearn->iContInputs=0; rSes.rLearn->iContOutputs=0; // implied sample value continuity recorded
		}
		if(A>=2) { // first value is 2+
			lCurrentLine=1; // samples begin on line 1
			rSes.rLearn->lSampleQty=rSes.lSampleQtyReq=rSes.lSampleQty=(int)A; // set sample qty equal to 1st parameter
			rSes.iContReadMode=1; rSes.rLearn->iContInputs=0; rSes.rLearn->iContOutputs=0; // implied sample value continuity recorded
		}
	}
	sprintf(sLog, "%d read mode specified: \"%s\"", rSes.rLearn->lSampleQty, cLines[0]); Barge->RLog(rSes, 0, sLog);
	sprintf(sLog, "%d samples specified: \"%s\"", rSes.rLearn->lSampleQty, cLines[1]); Barge->RLog(rSes, 0, sLog);
	
	// Parses lines of text for input values and output value and stores them in dynamic int arrays
	// returns # of inputs per line
	
	
	// get # of values per sample line
	rSes.rLearn->iValuesPerLine=TokenCounter(cLines[2], " ,\t");
	int iExcessValueQty = rSes.rLearn->iValuesPerLine - (rSes.rLearn->iInputQty + rSes.rLearn->iOutputQty );
	if(iExcessValueQty>0) {
		sprintf(sLog, "Warning: %d unused values in sample tuples.\n", iExcessValueQty);
		Barge->RLog(rSes, WARNINGF, sLog);
	}
	if(iExcessValueQty<0) {
		sprintf(sLog, "Error: %d values not found in sample tuples.\n", iExcessValueQty*-1);
		Barge->RLog(rSes, ERRORF, sLog);
	}
	/// allocate memory for tuple storage
	rSes.rLearn->dXInputs = (double*)malloc( (rSes.rLearn->iInputQty+1) * rSes.rLearn->lSampleQty * sizeof(double)); // scalar X input signal
		mCheckMallocWorked(rSes.rLearn->dXInputs)
	rSes.rLearn->dDOutputs=(double*)malloc( (rSes.rLearn->iOutputQty+1) * rSes.rLearn->lSampleQty * sizeof(double)); // scalar D correct output signal
		mCheckMallocWorked(rSes.rLearn->dDOutputs)
		rSes.lMemStructAlloc = rSes.lMemStructAlloc | RLEARNd; // flag existence of alllocation
	
	cSample=_strdup(cLines[lCurrentLine+rSes.rLearn->lSampleQty-1]);
	for (int s=0; s<rSes.rLearn->lSampleQty; ++s){ //iterate over the number of samples and malloc
		//fprintf(rSes.hostBucket, "%d[\t", s );
		for (int k=0; k<=rSes.rLearn->iInputQty; ++k) // fill with uniform, bogus values
			//#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))
			rSes.rLearn->dXInputs[ IDX2C( k, s, rSes.rLearn->iInputQty+1 ) ]=-999.9;
		for (int k=0; k<=rSes.rLearn->iOutputQty; ++k) // fill with uniform, bogus values
			rSes.rLearn->dDOutputs[ IDX2C( k, s, rSes.rLearn->iOutputQty+1 ) ]=-888.8; 
		// parse and store sample values
		pch = strtok (cLines[lCurrentLine], " ,\t"); // get the first token on a line
		// bRINJ mode not supported yet XX
		//if(rSes.bRInJMode){ // if flag for compatibility with older NN simulator is set XX
		//	for (int k=rSes.rLearn->iValuesPerLine; k>=1; --k){ // save it beginning with last position
		//		rSes.rLearn->dXInputs[ IDX2C( s, k, rSes.rLearn->iInputQty+1 ) ]=atof(pch); // convert and assign each value in a line
		//		pch = strtok (NULL, " ,\t");
		//	}
		//}
		//else{ // otherwise store things the usual way
			for (int k=1; k<=rSes.rLearn->iInputQty; ++k){ // save it beginning with position 1
				rSes.rLearn->dXInputs[ IDX2C( k, s, rSes.rLearn->iInputQty+1 ) ]=atof(pch); // convert and assign each value in a line
				//fprintf(rSes.hostBucket, "%f,%d>%d\t", atof(pch), k, IDX2C( k, s, rSes.rLearn->iInputQty+1) );
				pch = strtok (NULL, " ,\t");
			}
		for (int k=1; k<=rSes.rLearn->iOutputQty; ++k){
			rSes.rLearn->dDOutputs[ IDX2C( k, s, rSes.rLearn->iOutputQty+1 ) ]=atof(pch); // convert and assign each value in a line
			//fprintf(rSes.hostBucket, ":%f,%d>%d\t", atof(pch), k, IDX2C( k, s, rSes.rLearn->iOutputQty+1) );
			pch = strtok (NULL, " ,\t");
		}
		//fprintf(rSes.hostBucket, "\n");
		rSes.rLearn->dXInputs[ IDX2C( 0, s, rSes.rLearn->iInputQty+1 ) ]=0.0; // virtual input zero should always be zero
		rSes.rLearn->dDOutputs[ IDX2C( 0, s, rSes.rLearn->iOutputQty+1) ]=0.0; // output neuron zero should always produce sector 0 output
		free(cLines[lCurrentLine]);
		++lCurrentLine;
	}
	sprintf(sLog, "Last tuple read from line %d: \"%s\"", lCurrentLine, cSample);
	Barge->RLog(rSes, 0, sLog);

	free(cLines[0]);
	//if (iArchLineIdx) free(cLines[1]); // WEIRD MEMORY ERRORS? LOOK HERE XX
	// above line avoids double-freeing cLines[1] if it was used for a sample instead of the sample qty
	free(cLines);
	Barge->RLog(rSes, 0, __FUNCTION__);
 	return lLinesQty; // returns qty of lines read from file, not the same as quantity of samples
}


int cRamp::MoveKQbabWeights(struct rohanContext &rSes, FILE *fileHandle, char cKQ, char cRW, char cAB, cxVec& KQbab)
{mIDfunc
	// reads and writes values between filehandles and weight kebab vectors
	// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	// cKQ specifies kebabs with neuron zero (Q) or not (K)
	// cRW controls loading/reading (R) or saving/writing (W)
	// cAB determines binary (B) or ASCII (A) file formats

	int lReturnValue=0;

	vector<int>& Topo=rSes.Barge->ivTopoNN;
	vector<int>& OffsetK=rSes.Barge->ivOffsetK;
	vector<int>& OffsetQ=rSes.Barge->ivOffsetQ;

	if(cRW=='W'){ // write weights to save KQbab
		for (int LAY=1; LAY < (int)Topo.size(); ++LAY){
			int iNeuronQTY=Topo.at(LAY)+1;
			int iSignalQTY=Topo.at(LAY-1)+1; // signal qty depends on size of previous layer
			for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
				for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
					cuDoubleComplex& way = ( cKQ=='K'
						? KQbab.at(IDX2C( OffsetK.at(LAY) + i, k-1, iSignalQTY )) // k-1 reflects neurons numbered from one instead of zero
						: KQbab.at(IDX2C( OffsetQ.at(LAY) + i, k, iSignalQTY )) ); // unmodified k reflects neurons numbered from zero instead of one
					{
						if(cAB=='B'){
							fwrite(&(way.x), sizeof(double), 1, fileHandle);
							fwrite(&(way.y), sizeof(double), 1, fileHandle);
						} else
							fprintf(fileHandle, "% 11f,% 11f,% d,% d,% d\n", way.x, way.y, LAY, k, i);
					}
					++lReturnValue;
				}
			}
		}
	}
	if(cRW=='R'){ // read weights to load KQbab
		if(cKQ=='K')
			KQbab.resize(OffsetK.at(Topo.size())); // resize weight kebab to requried size
		else
			KQbab.resize(OffsetQ.at(Topo.size())); // resize weight kebab to requried size
		
		for (int LAY=1; LAY < (int)Topo.size(); ++LAY){
			int iNeuronQTY=Topo.at(LAY)+1;
			int iSignalQTY=Topo.at(LAY-1)+1; // signal qty depends on size of previous layer
			for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
				for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
					if(cAB=='B'){ // read binary values
						cuDoubleComplex& way = ( cKQ=='K'
							? KQbab.at(IDX2C( OffsetK.at(LAY) + i, k-1, iSignalQTY )) // k-1 reflects neurons numbered from one instead of zero
							: KQbab.at(IDX2C( OffsetQ.at(LAY) + i, k, iSignalQTY )) ); // unmodified k reflects neurons numbered from zero instead of one
						fread(&(way.x), sizeof(double), 1, fileHandle); // count elements as we read weight values
						fread(&(way.y), sizeof(double), 1, fileHandle);
					} else { // ASCII values are read and then put where they go
						int readL, readK, readI; double readX, readY;
						fscanf(fileHandle, "%f, %f, %d, %d, %d", &readX, &readY, &readL, &readK, &readI);
						cuDoubleComplex& ray = ( cKQ=='K' 
							? KQbab.at(IDX2C( OffsetK.at(readL) + readI, readK-1, Topo.at(readL-1)+1 )) // k-1 reflects neurons numbered from one instead of zero
							: KQbab.at(IDX2C( OffsetQ.at(readL) + readI, readK, Topo.at(readL-1)+1 )) ); // unmodified k reflects neurons numbered from zero instead of one
						ray.x=readX;
						ray.y=readY;
					}
					++lReturnValue;
				}
			}
		}
		if(cKQ=='Q') { // follow up for Queen-sized reads
			for (int LAY=1; LAY < (int)Topo.size(); ++LAY){
				int iNeuronQTY=Topo.at(LAY)+1;
				int iSignalQTY=Topo.at(LAY-1)+1; // signal qty depends on size of previous layer
				int k=0; { // fake weights for neuron 0
					int i=0; { // internal weight is always 1+0i
						cuDoubleComplex& way = KQbab.at(IDX2C( OffsetQ.at(LAY) + i, k, iSignalQTY )); // unmodified k reflects neurons numbered from zero instead of one
						way.x=1.0;
						way.y=0.0;
					}
					for (int i=1; i<iSignalQTY; ++i){ // external weight are always 0+0i
						cuDoubleComplex& way = KQbab.at(IDX2C( OffsetQ.at(LAY) + i, k, iSignalQTY )); // unmodified k reflects neurons numbered from zero instead of one
						way.x=0.0;
						way.y=0.0;
					}
				}
			}
		}
	}
	char sLog[255];
	sprintf(sLog, "%s(rSes, fh, %c, %c, %c, kebab) moved %d of %d weights", __FUNCTION__, cKQ, cRW, cAB, lReturnValue, 
		( cKQ=='K' ? OffsetK.at(Topo.size()) : OffsetQ.at(Topo.size()) ));
	Barge->RLog(rSes, 0+USERF, sLog);

	return lReturnValue;
}


int cRamp::cuSaveNNWeightsBIN(struct rohanContext &rSes, FILE *fileOutput, char cModel)
{mIDfunc
// writes values to .wgt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	int lReturnValue=0;

	struct rohanNetwork& Net = *rSes.rNet;
	for (int LAY=1; LAY<Net.iLayerQTY; ++LAY){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				fwrite(&(way.x), sizeof(double), 1, fileOutput);
				fwrite(&(way.y), sizeof(double), 1, fileOutput);
				++lReturnValue;
			}
		}
	}
	fclose(fileOutput);
	Barge->RLog(rSes, 0, __FUNCTION__);
	return lReturnValue;
}

int cRamp::cuSaveNNWeightsASCII(struct rohanContext &rSes, FILE *fileOutput, char cModel)
{mIDfunc
// writes values to .txt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	int lReturnValue=0;

	struct rohanNetwork& Net = *rSes.rNet;
	for (int LAY=1; LAY<Net.iLayerQTY; ++LAY){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no weights for neuron 0
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				fprintf(fileOutput, "% 25.20g,% 25.20g,% d,% d,% d\n", way.x, way.y, LAY, k, i);
				++lReturnValue;
			}
		}
	}
		using namespace boost::posix_time; 
		ptime now = second_clock::local_time(); //use the clock
		std::ostringstream osNow ;
		osNow << now;
		fprintf(fileOutput, "#END WEIGHTS Rohan v%s Neural Network Simulator %s %s\n", VERSION, AUTHORCREDIT, osNow.str().c_str());
		fclose(fileOutput);
	fclose(fileOutput);
	Barge->RLog(rSes, 0, __FUNCTION__);
	return lReturnValue;
}

int cRamp::SaveNNWeights(struct rohanContext& rSes, char cModel)
{mIDfunc/// saves weights in binary and ASCII form
	// modified to use rSes.sRohanVerPath 6/21/12
	FILE *fileOutput;
	char sFileAscii[255], sFileName[255], sLog[255];

	sprintf(sFileName, "%sRmse%03d", rSes.sSesName, (int)(Team->GetLastRmse(rSes,'B')*100));

	strncpy(sFileAscii,sFileName,248); // do not exceed 254 char file name
	strcat(sFileName,".wgt");
	strcat(sFileAscii,"WGT.txt");

	//fileOutput = fopen(sFileName, "wb");  /* Open in BINARY mode */
	GetFileHandle(rSes.sRohanVerPath, sFileName, 'w', 'b', &fileOutput);
	if (fileOutput == NULL) {
		sprintf(sLog, "Can't open %s for writing.\n", sFileName); Barge->RLog(rSes, ERRORF, sLog);
	Barge->RLog(rSes, 0, __FUNCTION__);
		return 0;
	}
	else {
		int lWWrit=cuSaveNNWeightsBIN(rSes, fileOutput, cModel);
		//fileOutput = fopen(sFileAscii, "w");  /* Open in ASCII mode */
		GetFileHandle(rSes.sRohanVerPath, sFileAscii, 'w', 'a', &fileOutput);
		if (fileOutput == NULL) {
			sprintf(sLog, "Can't open %s for writing.\n", sFileAscii); Barge->RLog(rSes, ERRORF, sLog);
	Barge->RLog(rSes, 0, __FUNCTION__);
			return 0;
		}
		else {
			int lWWrit=cuSaveNNWeightsASCII(rSes, fileOutput, cModel);
			sprintf(sLog,"%d ASCII weights written to %s\n", lWWrit, sFileAscii); Barge->RLog(rSes, USERF, sLog);
			sprintf(sLog,"report=%s", sFileAscii); Barge->RLog(rSes, GUIF, sLog);
	Barge->RLog(rSes, 0, __FUNCTION__);
			return 1;
		}	
	}
}

int cRamp::LoadNNWeights(struct rohanContext &rSes, char *sFileName)
{mIDfunc/// loads double precision weights from binary .wgt file
	int iReturn=0; char sLog[255];
	// on with it

	iReturn=GetFileHandle(rSes.sRohanVerPath, rSes.sWeightSet, 'r', 'b', &rSes.rNet->fileInput);

	if (iReturn) {
		iReturn=cuNNLoadWeights(rSes, rSes.rNet->fileInput);
		if (iReturn) {
			sprintf(sLog, "Parsed and assigned %d complex weights from %s", iReturn, rSes.sWeightSet); Barge->RLog(rSes, 0, sLog);
		}
		else {
			sprintf(sLog, "No weights read by cuNNLoadWeights"); Barge->RLog(rSes, ERRORF, sLog);
		}
	}
	else { // can't open, user random weights
		sprintf(sLog, "Can't open %s, using random weights.\n", rSes.sWeightSet); Barge->RLog(rSes, WARNINGF, sLog);
		cuRandomizeWeightsBlock(rSes); // populate network with random weight values
	}
	Barge->RLog(rSes, 0, __FUNCTION__);
	return iReturn;
}

int cRamp::cuNNLoadWeights(struct rohanContext &rSes, FILE *fileInput)
{mIDfunc
// pulls in values from .wgt files
// weights are arranged in network order 8 bytes of real, 8 bytes of imaginary
	int lReturnValue=0, lElementsReturned=0;
	char sLog[255];

	for (int j=1; j < rSes.rNet->iLayerQTY; ++j){ //no weights for layer 0
		struct rohanLayer& lay = rSes.rNet->rLayer[j];
		for (int k=1; k <= lay.iNeuronQty; ++k){ // no weights for neuron 0
			for (int i=0; i <= lay.iDendriteQty; ++i){
				cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
				lElementsReturned+=fread(&(way.x), sizeof(double), 1, fileInput); // count elements as we read weight values
				lElementsReturned+=fread(&(way.y), sizeof(double), 1, fileInput);
				++lReturnValue;
			}
		}
	}
	fclose(fileInput);
	
	if(lElementsReturned != (lReturnValue*2) ){ // not enough data, raise an alarm
		sprintf(sLog, "WARNING! Read past end of weight data. Found %d doubles, needed %d (2 per complex weight).", lElementsReturned, lReturnValue*2);
		Barge->RLog(rSes, WARNINGF, sLog);
	}
	Barge->RLog(rSes, 0, __FUNCTION__);
	return lReturnValue;
}

int cRamp::cuRandomizeWeightsBlock(struct rohanContext &rSes)
{mIDfunc /// generates random weights in [-1..0..1]
	int lReturnValue=0;

	struct rohanNetwork& Net = *rSes.rNet;
	for (int LAY=1; LAY<Net.iLayerQTY; ++LAY){
		int iNeuronQTY=Net.iNeuronQTY[LAY];
		int iSignalQTY=Net.iDendrtQTY[LAY]; // signal qty depends on size of previous layer
		for (int k=1; k < iNeuronQTY; ++k){ // no randomization for neuron 0
			for (int i=0; i<iSignalQTY; ++i){ //walk weights on inputs from previous layer
				cuDoubleComplex& way = Net.Wt[IDX2C( Net.iWeightOfst[LAY] + i, k, iSignalQTY )];
				++lReturnValue;
				way.x=1.0-(2.0*(double)rand()/RAND_MAX); // range of values is -1.0 to +1.0
				way.y=1.0-(2.0*(double)rand()/RAND_MAX); // range of values is -1.0 to +1.0
				//printf("\t%f\t+ %fi\n", way.x, way.y);
			}
		}
	}
	printf("%d pseudo-random weights on [-1..0..1]\n",lReturnValue);
	Barge->RLog(rSes, 0, __FUNCTION__);
	return lReturnValue;
}


	
int cRamp::cuRandomizeWeightsLayer(struct rohanContext &rSes)
{mIDfunc /// generates random weights in [-1..0..1]
	int lReturnValue=0;

	for (int j=1; j < rSes.rNet->iLayerQTY; ++j){ //no weights for layer 0
		struct rohanLayer& lay = rSes.rNet->rLayer[j];
		for (int k=1; k <= lay.iNeuronQty; ++k){
			//printf("\n[%d,%d] ",j,k);
			for (int i=0; i <= lay.iDendriteQty; ++i){
				cuDoubleComplex& way = lay.Weights[IDX2C(i, k, lay.iDendriteQty+1)];
				++lReturnValue;
				way.x=1.0-(double)(2*rand()/RAND_MAX); // range of values is -1.0 to +1.0
				way.y=1.0-(double)(2*rand()/RAND_MAX); // range of values is -1.0 to +1.0
				//printf("%d.", i);
			}
		}
	}
	printf("%d pseudo-random weights on [-1..0..1]\n",lReturnValue);

	return lReturnValue;
}


//int cRamp::LetWriteEvals(struct rohanContext& rSes, struct rohanLearningSet& rLearn, char * sFileAscii, char cModel)
//{mIDfunc/// saves evaluated output values to disk
//	int lReturn;
//	FILE *fileOutput; // File handle for output
//	char sLog[255];
//
//	lReturn=GetFileHandle(rSes.sRohanVerPath, sFileAscii, 'w', 'a', &fileOutput);
//	if(lReturn){
//		for(int s=0; s<rSes.lSampleQtyReq; ++s){
//			if(rSes.iSaveInputs){
//				for(int i=1; i<=rLearn.iInputQty; ++i) // write inputs first
//					fprintf(fileOutput, "%3.f, ", rLearn.dXInputs[IDX2C(i,s,rLearn.iInputQty+1)]);
//			}
//			if(rSes.iSaveOutputs){
//				for(int i=1; i<=rLearn.iOutputQty; ++i) // write desired outputs second
//					fprintf(fileOutput, "%7.0f, ", rLearn.dDOutputs[IDX2C(i,s,rLearn.iOutputQty+1)]);
//			}
//			for(int i=1; i<=rLearn.iOutputQty; ++i){ // write yielded outputs third
//				if(cModel=='S') // serial outputs here, paralllel ouputs there
//					fprintf(fileOutput, "%7.0f", rLearn.dYEval[IDX2C(i,s,rLearn.iOutputQty+1)]);
//				else // with this branch I complete the original assigned functionality of Rohan, JAW 8/13/12
//					fprintf(fileOutput, "%7.0f", rLearn.dAltYEval[IDX2C(i,s,rLearn.iOutputQty+1)]);
//				if (i<rLearn.iOutputQty)
//					fprintf(fileOutput, ", "); // only put commas between outputs, not after
//			}
//			if(rSes.iSaveSampleIndex){ // write sample indexes last
//				fprintf(fileOutput, ", %7d", s);
//			}
//			fprintf(fileOutput, "\n"); // end each line with a newline
//		}
//		fclose(fileOutput);
//	}
//	// Log event
//	sprintf(sLog, "%d evals written to %s", rSes.lSampleQtyReq, sFileAscii ); // document success and filename
//	Barge->RLog(rSes, USERF+ADMINF, sLog);
//	Barge->RLog(rSes, GUIF, "eval=pass");
//	// include report for GUI
//	sprintf(sLog, "report=%s", sFileAscii ); // document success and filename
//	Barge->RLog(rSes, GUIF, sLog);
//	Barge->RLog(rSes, 0, __FUNCTION__);
//	if (lReturn)
//		return rSes.lSampleQtyReq; // return number of sample evals recorded
//	else
//		return 0;
//}
//

int cRamp::LetWriteEvals(struct rohanContext& rSes, struct rohanLearningSet& rLearn, char * sFileAscii, char cModel)
{mIDfunc/// saves evaluated output values to disk
	int lReturn;
	FILE *fileOutput; // File handle for output
	char sLog[255];

	lReturn=GetFileHandle(rSes.sRohanVerPath, sFileAscii, 'w', 'a', &fileOutput);
	if(lReturn){
		for(int s=0; s<rSes.lSampleQtyReq; ++s){
			if(rSes.iSaveInputs){
				for(int i=1; i<=rLearn.iInputQty; ++i) // write inputs first
					if ( rLearn.iContInputs)
						fprintf(fileOutput, "% 9.6f, ", rLearn.dXInputs[IDX2C(i,s,rLearn.iInputQty+1)]); // radian inputs
					else
						fprintf(fileOutput, "%5.0f, ", rLearn.dXInputs[IDX2C(i,s,rLearn.iInputQty+1)]); // sector inputs
				
			}
			if(rSes.iSaveOutputs){
				for(int i=1; i<=rLearn.iOutputQty; ++i) // write desired outputs second
					if(rLearn.iContOutputs)
						fprintf(fileOutput, "% 9.6f, ", rLearn.dDOutputs[IDX2C(i,s,rLearn.iOutputQty+1)]);
					else
						fprintf(fileOutput, "%5.0f, ", rLearn.dDOutputs[IDX2C(i,s,rLearn.iOutputQty+1)]);
			}

			for(int i=1; i<=rLearn.iOutputQty; ++i){ // write yielded outputs third
				if ( cModel=='S' &&  rLearn.iContOutputs)
					fprintf(fileOutput, "% 9.6f", rLearn.dYEval[IDX2C(i,s,rLearn.iOutputQty+1)]); // serial, radians
				if ( cModel=='S' && !rLearn.iContOutputs)
					fprintf(fileOutput, "%5.0f", rLearn.dYEval[IDX2C(i,s,rLearn.iOutputQty+1)]); // serial, sectors
				if ( cModel!='S' &&  rLearn.iContOutputs)
					fprintf(fileOutput, "% 9.6f", rLearn.dAltYEval[IDX2C(i,s,rLearn.iOutputQty+1)]); // parallel, radians
				if ( cModel!='S' && !rLearn.iContOutputs)
					fprintf(fileOutput, "%5.0f", rLearn.dAltYEval[IDX2C(i,s,rLearn.iOutputQty+1)]); // parallel, sectors

				if (i<rLearn.iOutputQty)
					fprintf(fileOutput, ", "); // only put commas between outputs, not after
			}	

			if(rSes.iSaveSampleIndex) // write sample indexes last
				fprintf(fileOutput, "  #%d", s);
			
			fprintf(fileOutput, "\n"); // end each line with a newline
		}
		using namespace boost::posix_time; 
		ptime now = second_clock::local_time(); //use the clock
		std::ostringstream osNow ;
		osNow << now;
		fprintf(fileOutput, "#END %s Rohan v%s Neural Network Simulator %s %s\n", sFileAscii, VERSION, AUTHORCREDIT, osNow.str().c_str());
		fclose(fileOutput);
	}
	// Log event
	sprintf(sLog, "%d evals written to %s", rSes.lSampleQtyReq, sFileAscii ); // document success and filename
	Barge->RLog(rSes, USERF+ADMINF, sLog);
	Barge->RLog(rSes, GUIF, "eval=pass");
	// include report for GUI
	sprintf(sLog, "report=%s", sFileAscii ); // document success and filename
	Barge->RLog(rSes, GUIF, sLog);
	Barge->RLog(rSes, 0, __FUNCTION__);
	if (lReturn)
		return rSes.lSampleQtyReq; // return number of sample evals recorded
	else
		return 0;
}
