#include <permonsys.h>
#include <permon/private/permonimpl.h>
#include <petsc/private/logimpl.h>
#include <petsclog.h>

PetscBool PermonInfoEnabled       = PETSC_FALSE;
PetscBool PermonObjectInfoEnabled = PETSC_FALSE;
PetscBool PermonTraceEnabled      = PETSC_FALSE;
PetscBool PermonDebugEnabled      = PETSC_FALSE;

PetscErrorCode _permon_ierr;
PetscInt       PeFuBe_i_ = 0;
char           PeFuBe_s_[128];
char           PERMON_PathBuffer_Global[PERMON_MAX_PATH_LEN];
char           PERMON_ObjNameBuffer_Global[PERMON_MAX_NAME_LEN];

#undef __FUNCT__
#define __FUNCT__ "PermonCreate"
PetscErrorCode PermonCreate(MPI_Comm comm, PERMON *permon_new)
{
  PERMON permon;

  PetscFunctionBegin;
  PetscAssertPointer(permon_new, 2);
  PetscCall(PetscHeaderCreate(permon, PERMON_CLASSID, "PERMON", "PERMON", "PERMON", comm, 0, 0));
  *permon_new = permon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonDestroy"
PetscErrorCode PermonDestroy(PERMON *permon)
{
  PetscFunctionBegin;
  if (!*permon) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*permon, PERMON_CLASSID, 1);
  if (--((PetscObject)(*permon))->refct > 0) {
    *permon = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscHeaderDestroy(permon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMakePath"
PetscErrorCode PermonMakePath(const char *dir, mode_t mode)
{
  char     *tmp = PERMON_PathBuffer_Global;
  char     *p   = NULL;
  size_t    len;
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(PetscSNPrintf(tmp, PERMON_MAX_PATH_LEN, "%s", dir));
  PetscCall(PetscStrlen(tmp, &len));

  if (tmp[len - 1] == '/') tmp[len - 1] = 0;

  for (p = tmp + 1; *p; p++)
    if (*p == '/') {
      *p = 0;
      mkdir(tmp, mode);
      *p = '/';
    }

  mkdir(tmp, mode);

  PetscCall(PetscTestDirectory(dir, 'x', &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Directory %s was not created properly.", dir);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonProcessInfoExclusions"
PetscErrorCode PermonProcessInfoExclusions(PetscClassId classid, const char *classname)
{
  char      logList[256];
  char     *str;
  PetscBool opt;

  PetscFunctionBegin;
  if (PermonInfoEnabled) {
    PetscCall(PetscInfoActivateClass(classid));
  } else {
    PetscCall(PetscInfoDeactivateClass(classid));
  }

  /* Process info exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-info_exclude", logList, 256, &opt));
  if (opt) {
    PetscCall(PetscStrstr(logList, classname, &str));
    if (str) { PetscCall(PetscInfoDeactivateClass(classid)); }
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_summary_exclude", logList, 256, &opt));
  if (opt) {
    PetscCall(PetscStrstr(logList, classname, &str));
    if (str) { PetscCall(PetscLogEventDeactivateClass(classid)); }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSetTrace"
PetscErrorCode PermonSetTrace(PetscBool flg)
{
  PetscFunctionBegin;
  PermonTraceEnabled = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSetObjectInfo"
PetscErrorCode PermonSetObjectInfo(PetscBool flg)
{
  PetscFunctionBegin;
  PermonObjectInfoEnabled = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSetDebug"
PetscErrorCode PermonSetDebug(PetscBool flg)
{
  PetscFunctionBegin;
  PermonDebugEnabled = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonPetscInfoDeactivateAll"
PetscErrorCode PermonPetscInfoDeactivateAll()
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = PETSC_SMALLEST_CLASSID + 1; i < PETSC_SMALLEST_CLASSID + 60; i++) PetscCall(PetscInfoDeactivateClass(i));
  PetscCall(PetscInfoDeactivateClass(0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonSetFromOptions"
/*@
   PermonSetFromOptions - Sets PERMON options.

   Options Database Keys:
+  -permon_object_info - print one-line info messages about matrices and vectors
.  -permon_dump        - dump matrix data used by PERMON
.  -permon_trace       - trace crucial PERMON functions
.  -permon_debug       - enable PERMON debug messages
-  -permon_info        - enable info messages only from PERMON

   Level: beginner

.seealso PermonInitialize()
@*/
PetscErrorCode PermonSetFromOptions()
{
  char      logList[256];
  char     *className;
  PetscBool flg = PETSC_FALSE;
  PetscBool info, excl, permon_info;
  char      logname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  logname[0] = 0;
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "PERMON options", NULL);
  {
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-permon_object_info", "print one-line info messages about matrices and vectors", NULL, PETSC_FALSE, &flg, NULL));
    PetscCall(PermonSetObjectInfo(flg));
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-permon_trace", "trace crucial PERMON functions", NULL, PETSC_FALSE, &flg, NULL));
    PetscCall(PermonSetTrace(flg));
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-permon_debug", "enable PERMON debug messages", NULL, PETSC_FALSE, &flg, NULL));
    PetscCall(PermonSetDebug(flg));
    flg = PETSC_FALSE;
#if defined(PETSC_USE_INFO)
    PetscCall(PetscOptionsString("-permon_info", "enable info messages only from PERMON", NULL, NULL, logname, 256, &permon_info));
#endif
  }
  PetscOptionsEnd();

#if defined(PETSC_USE_INFO)
  {
    info = PETSC_FALSE;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-info", logname, 256, &info));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-info_exclude", logList, 256, &excl));

    if (permon_info || info) {
      PermonInfoEnabled = PETSC_TRUE;
      PetscCall(PetscInfoAllow(PETSC_TRUE));
      if (logname[0]) { PetscCall(PetscInfoSetFile(logname, "w")); }
    }

    if (!info) { PetscCall(PermonPetscInfoDeactivateAll()); }

    if (excl) {
      PetscCall(PetscStrstr(logList, "petsc", &className));
      if (className) { PetscCall(PermonPetscInfoDeactivateAll()); }

      PetscCall(PetscStrstr(logList, "permon", &className));
      if (className) PermonInfoEnabled = PETSC_FALSE;
    }

    if (PermonInfoEnabled) {
      PetscCall(PetscInfoActivateClass(PERMON_CLASSID));
    } else {
      PetscCall(PetscInfoDeactivateClass(PERMON_CLASSID));
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* based on PetscLogEventGetId but does not throw an error if the event does not exist */
#undef __FUNCT__
#define __FUNCT__ "PermonPetscLogEventGetId"
PetscErrorCode PermonPetscLogEventGetId(const char name[], PetscLogEvent *event, PetscBool *exists)
{
  PetscFunctionBegin;
  *exists = PETSC_FALSE;
  PetscCall(PetscLogEventGetId(name, event));
  if (event) *exists = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonPetscObjectInheritName"
PetscErrorCode PermonPetscObjectInheritName(PetscObject obj, PetscObject orig, const char *suffix)
{
  size_t len1 = 0, len2 = 0;
  char  *name;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidHeader(orig, 2);
  if (suffix) PetscAssertPointer(suffix, 3);
  PetscCall(PetscObjectName((PetscObject)orig));
  if (obj == orig && !suffix) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscStrlen(orig->name, &len1));
  PetscCall(PetscStrlen(suffix, &len2));
  PetscCall(PetscMalloc((1 + len1 + len2) * sizeof(char), &name));
  PetscCall(PetscStrncpy(name, orig->name, sizeof(name)));
  PetscCall(PetscStrlcat(name, suffix, sizeof(name)));
  PetscCall(PetscFree(obj->name));
  obj->name = name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonPetscObjectInheritPrefix"
PetscErrorCode PermonPetscObjectInheritPrefix(PetscObject obj, PetscObject orig, const char *suffix)
{
  size_t len1 = 0, len2 = 0;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidHeader(orig, 2);
  if (suffix) PetscAssertPointer(suffix, 3);
  PetscCall(PetscFree(obj->prefix));

  if (!orig->prefix) {
    PetscCall(PetscStrallocpy(suffix, &obj->prefix));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrlen(orig->prefix, &len1));
  PetscCall(PetscStrlen(suffix, &len2));
  PetscCall(PetscMalloc((1 + len1 + len2) * sizeof(char), &obj->prefix));
  PetscCall(PetscStrncpy(obj->prefix, orig->prefix, sizeof(obj->prefix)));
  PetscCall(PetscStrlcat(obj->prefix, suffix, sizeof(obj->prefix)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonPetscObjectInheritPrefixIfNotSet"
PetscErrorCode PermonPetscObjectInheritPrefixIfNotSet(PetscObject obj, PetscObject orig, const char *suffix)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidHeader(orig, 2);
  if (suffix) PetscAssertPointer(suffix, 3);
  if (obj->prefix) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PermonPetscObjectInheritPrefix(obj, orig, suffix));
  PetscFunctionReturn(PETSC_SUCCESS);
}
