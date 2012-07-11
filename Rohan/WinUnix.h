#ifndef WINUNIX_H
#define WINUNIX_H
// OS-specific items handled here

int GetUserDocPath(char * sPath)
;
int DirectoryEnsure(char * sPath)
;
int SetVerPath( struct rohanContext& rSes )
;
int ResetCwd( struct rohanContext& rSes )
;

#endif
