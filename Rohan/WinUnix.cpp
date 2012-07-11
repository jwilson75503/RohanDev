// OS-specific items handled here

/* Includes, cuda */
#include "stdafx.h"

/* Windows specific */
#define SECURITY_WIN32
#include <security.h>
#include <windows.h>
#include <secext.h>
#include <shfolder.h>
#include <shlobj.h>


int GetUserDocPath(char * sPath)
{mIDfunc
//WCHAR path[MAX_PATH];
	HRESULT hr = SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, SHGFP_TYPE_CURRENT, sPath);
	return true;
}

int DirectoryEnsure(char * sPath)
{mIDfunc/// checks that a directory exists and creates it if not
	
	if ((GetFileAttributes(sPath)) == INVALID_FILE_ATTRIBUTES){
		//cout << "Directory doesn't exist\n";
		CreateDirectory(sPath, 0);
		//fprintf(stdout, "Directory %s created\n", sPath);
	}

	if ((GetFileAttributes(sPath)) == INVALID_FILE_ATTRIBUTES){
		return false;
	}
	else{
		return true;
	}
}


int SetVerPath( struct rohanContext& rSes )
{mIDfunc/// sets cwd to version-specific path in user's document folder
	char sPath[255];

	GetUserDocPath(sPath); // sPath now has "C:\users\documents"
	sprintf(rSes.sRohanVerPath, "%s\\Rohan-%s", sPath, VERSION); // .sRohanVerPath has "C:\users\documents\Rohan-0.9.4"
	
	return true;
}


int ResetCwd( struct rohanContext& rSes )
{mIDfunc/// sets cwd to version-specific path in user's document folder
	char sPath[255];

	GetUserDocPath(sPath);
	sprintf(rSes.cwd, "%s\\Rohan-%s", sPath, VERSION); // cwd has "C:\users\documents\Rohan-0.9.4"
	
	
	return true;
}
