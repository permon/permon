
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
  CHKERRQ(PetscHeaderCreate(fllop,FLLOP_CLASSID,"FLLOP","FLLOP","FLLOP",comm,0,0));
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
  CHKERRQ(PetscHeaderDestroy(fllop));
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
    CHKERRQ(PetscSNPrintf(tmp, FLLOP_MAX_PATH_LEN, "%s", dir));
    CHKERRQ(PetscStrlen(tmp, &len));

    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;

    for (p = tmp + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, mode);
            *p = '/';
        }

    mkdir(tmp, mode);
    
    CHKERRQ(PetscTestDirectory(dir, 'x', &flg));    
    if (!flg) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Directory %s was not created properly.", dir);
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
    CHKERRQ(PetscInfoActivateClass(classid));
  } else {
    CHKERRQ(PetscInfoDeactivateClass(classid));
  }
  
  /* Process info exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL, "-info_exclude", logList, 256, &opt));
  if (opt) {
    CHKERRQ(PetscStrstr(logList, classname, &str));
    if (str) {
      CHKERRQ(PetscInfoDeactivateClass(classid));
    }
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL, "-log_summary_exclude", logList, 256, &opt));
  if (opt) {
    CHKERRQ(PetscStrstr(logList, classname, &str));
    if (str) {
      CHKERRQ(PetscLogEventDeactivateClass(classid));
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
  for (i = PETSC_SMALLEST_CLASSID+1; i < PETSC_SMALLEST_CLASSID+60; i++) CHKERRQ(PetscInfoDeactivateClass(i));
  CHKERRQ(PetscInfoDeactivateClass(0));
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
  
  PetscFunctionBegin;
  logname[0] = 0;
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FLLOP options", NULL);
  {
    flg = PETSC_FALSE;
    CHKERRQ(PetscOptionsBool("-fllop_object_info", "print one-line info messages about matrices and vectors", NULL, PETSC_FALSE, &flg, NULL));
    CHKERRQ(FllopSetObjectInfo(flg));
    flg = PETSC_FALSE;
    CHKERRQ(PetscOptionsBool("-fllop_trace",       "trace crucial FLLOP functions",                           NULL, PETSC_FALSE, &flg, NULL));
    CHKERRQ(FllopSetTrace(flg));
    flg = PETSC_FALSE;
    CHKERRQ(PetscOptionsBool("-fllop_debug",       "enable FLLOP debug messages",                             NULL, PETSC_FALSE, &flg, NULL));
    CHKERRQ(FllopSetDebug(flg));    
    flg = PETSC_FALSE;
#if defined (PETSC_USE_INFO)
    CHKERRQ(PetscOptionsString("-fllop_info",      "enable info messages only from FLLOP",                    NULL, NULL, logname, 256, &fllop_info));
#endif     
  }
  PetscOptionsEnd();

#if defined (PETSC_USE_INFO)
  {
    info = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetString(NULL,NULL, "-info", logname, 256, &info));
    CHKERRQ(PetscOptionsGetString(NULL,NULL, "-info_exclude", logList, 256, &excl));
    
    if (fllop_info || info) {
      FllopInfoEnabled = PETSC_TRUE;
      CHKERRQ(PetscInfoAllow(PETSC_TRUE));
      if (logname[0]) {
        CHKERRQ(PetscInfoSetFile(logname,"w"));
      }
    }
    
    if (!info) {
      CHKERRQ(FllopPetscInfoDeactivateAll());
    }    

    if (excl) {
      CHKERRQ(PetscStrstr(logList, "petsc", &className));
      if (className) {
        CHKERRQ(FllopPetscInfoDeactivateAll());
      }

      CHKERRQ(PetscStrstr(logList, "fllop", &className));
      if (className) FllopInfoEnabled = PETSC_FALSE;
    }

    if (FllopInfoEnabled) {
      CHKERRQ(PetscInfoActivateClass(FLLOP_CLASSID));
    } else {
      CHKERRQ(PetscInfoDeactivateClass(FLLOP_CLASSID));
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

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(event,3);
  *event = -1;
  *exists = PETSC_FALSE;
  for (e = 0; e < eventLog->numEvents; e++) {
    CHKERRQ(PetscStrcasecmp(eventLog->eventInfo[e].name, name, &match));
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

  PetscFunctionBegin;
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  CHKERRQ(FllopEventRegLogGetEvent(stageLog->eventLog, name, event, exists));
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
  CHKERRQ(PetscObjectName((PetscObject)orig));
  if (obj==orig && !suffix) PetscFunctionReturn(0);
  CHKERRQ(PetscStrlen(orig->name,&len1));
  CHKERRQ(PetscStrlen(suffix,&len2));
  CHKERRQ(PetscMalloc((1+len1+len2)*sizeof(char),&name));
  CHKERRQ(PetscStrcpy(name,orig->name));
  CHKERRQ(PetscStrcat(name,suffix));
  CHKERRQ(PetscFree(obj->name));
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
  CHKERRQ(PetscFree(obj->prefix));

  if (!orig->prefix) {
    CHKERRQ(PetscStrallocpy(suffix,&obj->prefix));
    PetscFunctionReturn(0);
  }

  CHKERRQ(PetscStrlen(orig->prefix,&len1));
  CHKERRQ(PetscStrlen(suffix,&len2));
  CHKERRQ(PetscMalloc((1+len1+len2)*sizeof(char),&obj->prefix));
  CHKERRQ(PetscStrcpy(obj->prefix,orig->prefix));
  CHKERRQ(PetscStrcat(obj->prefix,suffix));
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
  CHKERRQ(FllopPetscObjectInheritPrefix(obj,orig,suffix));
  PetscFunctionReturn(0);
}
