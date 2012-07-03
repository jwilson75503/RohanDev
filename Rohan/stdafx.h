#ifndef STDAFX_H
#define STDAFX_H
// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently

#pragma once

#include "targetver.h"

/* standard libraries */
#include <conio.h> //for _getch 
#include <iostream>
#include <string>
#include <cctype>
#include <algorithm>
#include <vector>
#include <iterator>
#include <float.h>
#include <math.h>  //for M_PI
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>
#include <sys/timeb.h>

/* CUDA-relevant includes */
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <multithreading.h>

// Shared Utilities (QA Testing)
#include <shrUtils.h>
#include <shrQATest.h>

/* Project includes */
#include "cDrover.h"
#include "Complex-math.h"
#include "crc.h"
#include "cTeam.h"
#include "Rohan.h"
#include "Rohan-data.h"
#include "Rohan-kernel.h"
#include "Rohan-learn.h"
#ifndef CUDACONLY
#include "cBarge.h"
#endif

// Boost
#ifndef CUDACONLY
#include <boost/date_time/posix_time/posix_time.hpp> //include all types plus i/o
using namespace boost::posix_time;
#endif

// usings
using namespace std;
using std::cin;
using std::cout;

// defines
#define TWO_PI 6.283185307179586476925286766558
#define IDX2C( i, j, ld) ((i)+(( j )*( ld )))

// global declares
extern int gDebugLvl, gDevDebug, gTrace;
extern float gElapsedTime, gKernelTimeTally;

#endif
