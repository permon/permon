
#include <permonsys.h>
#include <permon/private/permonimpl.h>
#include <petsc/private/logimpl.h>
#include <petsclog.h>

PetscBool FllopInfoEnabled = PETSC_FALSE;
PetscBool FllopObjectInfoEnabled = PETSC_FALSE;
PetscBool FllopTraceEnabled = PETSC_FALSE;
PetscBool FllopDebugEnabled = PETSC_FALSE;

PetscErrorCode _fllop_ierr;
PetscInt PeFuBe_i_ = 0;
char     PeFuBe_s_[128];
char     FLLOP_PathBuffer_Global[FLLOP_MAX_PATH_LEN];
char     FLLOP_ObjNameBuffer_Global[FLLOP_MAX_NAME_LEN];

#undef __FUNCT__
#define __FUNCT__ "FllopCreate"
PetscErrorCode FllopCreate(MPI_Comm comm,FLLOP *fllop_new)
{
  FLLOP fllop;

  PetscFunctionBegin;
  PetscValidPointer(fllop_new,2);
  TRY( PetscHeaderCreate(fllop,FLLOP_CLASSID,"FLLOP","FLLOP","FLLOP",comm,0,0) );
  *fllop_new = fllop;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopDestroy"
PetscErrorCode FllopDestroy(FLLOP *fllop)
{
  PetscFunctionBegin;
  if (!*fllop) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*fllop, FLLOP_CLASSID, 1);
  if (--((PetscObject) (*fllop))->refct > 0) {
    *fllop = 0;
    PetscFunctionReturn(0);
  }
  TRY( PetscHeaderDestroy(fllop) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMakePath"
PetscErrorCode FllopMakePath(const char *dir, mode_t mode)
{
    char *tmp = FLLOP_PathBuffer_Global;
    char *p = NULL;
    size_t len;
    PetscBool flg;
    
    PetscFunctionBegin;
    TRY( PetscSNPrintf(tmp, FLLOP_MAX_PATH_LEN, "%s", dir) );
    TRY( PetscStrlen(tmp, &len) );

    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;

    for (p = tmp + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, mode);
            *p = '/';
        }

    mkdir(tmp, mode);
    
    TRY( PetscTestDirectory(dir, 'x', &flg) );    
    if (!flg) FLLOP_SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Directory %s was not created properly.", dir);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopProcessInfoExclusions"
PetscErrorCode FllopProcessInfoExclusions(PetscClassId classid, const char *classname)
{
  char              logList[256];
  char              *str;
  PetscBool         opt;

  PetscFunctionBegin;
  if (FllopInfoEnabled) {
    TRY( PetscInfoActivateClass(classid) );
  } else {
    TRY( PetscInfoDeactivateClass(classid) );
  }
  
  /* Process info exclusions */
  TRY( PetscOptionsGetString(NULL,NULL, "-info_exclude", logList, 256, &opt) );
  if (opt) {
    TRY( PetscStrstr(logList, classname, &str) );
    if (str) {
      TRY( PetscInfoDeactivateClass(classid) );
    }
  }
  /* Process summary exclusions */
  TRY( PetscOptionsGetString(NULL,NULL, "-log_summary_exclude", logList, 256, &opt) );
  if (opt) {
    TRY( PetscStrstr(logList, classname, &str) );
    if (str) {
      TRY( PetscLogEventDeactivateClass(classid) );
    }
  }  
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopSetTrace"
PetscErrorCode FllopSetTrace(PetscBool flg)
{
  PetscFunctionBegin;
  FllopTraceEnabled = flg;  
  PetscFunctionReturn(0);  
}

#undef __FUNCT__
#define __FUNCT__ "FllopSetObjectInfo"
PetscErrorCode FllopSetObjectInfo(PetscBool flg)
{
  PetscFunctionBegin;
  FllopObjectInfoEnabled = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopSetDebug"
PetscErrorCode FllopSetDebug(PetscBool flg)
{
  PetscFunctionBegin;
  FllopDebugEnabled = flg;  
  PetscFunctionReturn(0);  
}

#undef __FUNCT__
#define __FUNCT__ "FllopPetscInfoDeactivateAll"
PetscErrorCode FllopPetscInfoDeactivateAll()
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = PETSC_SMALLEST_CLASSID+1; i < PETSC_SMALLEST_CLASSID+60; i++) TRY( PetscInfoDeactivateClass(i) );
  TRY( PetscInfoDeactivateClass(0) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopSetFromOptions"
/*@
   FllopSetFromOptions - Sets FLLOP options.

   Options Database Keys:
+  -fllop_object_info - print one-line info messages about matrices and vectors
.  -fllop_dump        - dump matrix data used by FLLOP
.  -fllop_trace       - trace crucial FLLOP functions
.  -fllop_debug       - enable FLLOP debug messages
-  -fllop_info        - enable info messages only from FLLOP
 
   Level: beginner

.seealso FllopInitialize()
@*/
PetscErrorCode FllopSetFromOptions()
{
  char           logList[256];
  char           *className;
  PetscBool      flg=PETSC_FALSE;
  PetscBool      info, excl, fllop_info;
  char           logname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  logname[0] = 0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FLLOP options", NULL);CHKERRQ(ierr);
  {
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-fllop_object_info", "print one-line info messages about matrices and vectors", NULL, PETSC_FALSE, &flg, NULL);CHKERRQ(ierr);
    ierr = FllopSetObjectInfo(flg);CHKERRQ(ierr);
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-fllop_trace",       "trace crucial FLLOP functions",                           NULL, PETSC_FALSE, &flg, NULL);CHKERRQ(ierr);
    ierr = FllopSetTrace(flg);CHKERRQ(ierr);
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-fllop_debug",       "enable FLLOP debug messages",                             NULL, PETSC_FALSE, &flg, NULL);CHKERRQ(ierr);
    ierr = FllopSetDebug(flg);CHKERRQ(ierr);    
    flg = PETSC_FALSE;
#if defined (PETSC_USE_INFO)
    ierr = PetscOptionsString("-fllop_info",      "enable info messages only from FLLOP",                    NULL, NULL, logname, 256, &fllop_info);CHKERRQ(ierr);
#endif     
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

#if defined (PETSC_USE_INFO)
  {
    info = PETSC_FALSE;
    ierr = PetscOptionsGetString(NULL,NULL, "-info", logname, 256, &info);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,NULL, "-info_exclude", logList, 256, &excl);CHKERRQ(ierr);
    
    if (fllop_info || info) {
      FllopInfoEnabled = PETSC_TRUE;
      ierr = PetscInfoAllow(PETSC_TRUE);CHKERRQ(ierr);
      if (logname[0]) {
        ierr = PetscInfoSetFile(logname,"w");CHKERRQ(ierr);
      }
    }
    
    if (!info) {
      TRY( FllopPetscInfoDeactivateAll() );
    }    

    if (excl) {
      ierr = PetscStrstr(logList, "petsc", &className);CHKERRQ(ierr);
      if (className) {
        TRY( FllopPetscInfoDeactivateAll() );
      }

      ierr = PetscStrstr(logList, "fllop", &className);CHKERRQ(ierr);
      if (className) FllopInfoEnabled = PETSC_FALSE;
    }

    if (FllopInfoEnabled) {
      ierr = PetscInfoActivateClass(FLLOP_CLASSID);CHKERRQ(ierr);
    } else {
      ierr = PetscInfoDeactivateClass(FLLOP_CLASSID);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}

/* based on EventRegLogGetEvent but does not throw an error if the event does not exist */
#undef __FUNCT__
#define __FUNCT__ "FllopEventRegLogGetEvent"
PetscErrorCode FllopEventRegLogGetEvent(PetscEventRegLog eventLog, const char name[], PetscLogEvent *event, PetscBool *exists)
{
  PetscBool      match;
  int            e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(event,3);
  *event = -1;
  *exists = PETSC_FALSE;
  for (e = 0; e < eventLog->numEvents; e++) {
    ierr = PetscStrcasecmp(eventLog->eventInfo[e].name, name, &match);CHKERRQ(ierr);
    if (match) {
      *event = e;
      *exists = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/* based on PetscLogEventGetId but does not throw an error if the event does not exist */
#undef __FUNCT__
#define __FUNCT__ "FllopPetscLogEventGetId"
PetscErrorCode  FllopPetscLogEventGetId(const char name[], PetscLogEvent *event, PetscBool *exists)
{
  PetscStageLog  stageLog;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = FllopEventRegLogGetEvent(stageLog->eventLog, name, event, exists);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopPetscObjectInheritName"
PetscErrorCode FllopPetscObjectInheritName(PetscObject obj,PetscObject orig,const char *suffix)
{
  size_t         len1=0,len2=0;
  char *name;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidHeader(orig,2);
  if (suffix) PetscValidCharPointer(suffix,3);
  TRY( PetscObjectName((PetscObject)orig) );
  if (obj==orig && !suffix) PetscFunctionReturn(0);
  TRY( PetscStrlen(orig->name,&len1) );
  TRY( PetscStrlen(suffix,&len2) );
  TRY( PetscMalloc((1+len1+len2)*sizeof(char),&name) );
  TRY( PetscStrcpy(name,orig->name) );
  TRY( PetscStrcat(name,suffix) );
  TRY( PetscFree(obj->name) );
  obj->name = name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopPetscObjectInheritPrefix"
PetscErrorCode FllopPetscObjectInheritPrefix(PetscObject obj,PetscObject orig,const char *suffix)
{
  size_t         len1=0,len2=0;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidHeader(orig,2);
  if (suffix) PetscValidCharPointer(suffix,3);
  TRY( PetscFree(obj->prefix) );

  if (!orig->prefix) {
    TRY( PetscStrallocpy(suffix,&obj->prefix) );
    PetscFunctionReturn(0);
  }

  TRY( PetscStrlen(orig->prefix,&len1) );
  TRY( PetscStrlen(suffix,&len2) );
  TRY( PetscMalloc((1+len1+len2)*sizeof(char),&obj->prefix) );
  TRY( PetscStrcpy(obj->prefix,orig->prefix) );
  TRY( PetscStrcat(obj->prefix,suffix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopPetscObjectInheritPrefixIfNotSet"
PetscErrorCode FllopPetscObjectInheritPrefixIfNotSet(PetscObject obj,PetscObject orig,const char *suffix)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidHeader(orig,2);
  if (suffix) PetscValidCharPointer(suffix,3);
  if (obj->prefix) PetscFunctionReturn(0);
  TRY( FllopPetscObjectInheritPrefix(obj,orig,suffix) );
  PetscFunctionReturn(0);
}
