#pragma once

#include <petscksp.h>
#include <petsctime.h>

#if defined(PETSC_HAVE_SYS_TYPES_H)
  #include <sys/types.h>
  #include <sys/stat.h>
#endif

#define PERMON_EXTERN PETSC_EXTERN
#define PERMON_INTERN PETSC_INTERN

#include "permonpetscretro.h"

/*
  PERMON is a dummy class, defined in PermonInitialize,
  used to distinguish between PETSc and PERMON sys routines while using PetscInfo*,
*/
typedef struct _p_PERMON  *PERMON;
PERMON_EXTERN PetscClassId PERMON_CLASSID;
PERMON_EXTERN PERMON       permon;
PERMON_EXTERN PetscBool    PermonInfoEnabled, PermonObjectInfoEnabled, PermonDebugEnabled;

#ifdef NAME_MAX
  #define PERMON_MAX_NAME_LEN NAME_MAX + 1
#else
  #define PERMON_MAX_NAME_LEN 256
#endif
#define PERMON_MAX_PATH_LEN PETSC_MAX_PATH_LEN
PERMON_EXTERN char PERMON_PathBuffer_Global[PERMON_MAX_PATH_LEN];
PERMON_EXTERN char PERMON_ObjNameBuffer_Global[PERMON_MAX_NAME_LEN];

/* BEGIN Function-like Macros */
PERMON_EXTERN PetscErrorCode _permon_ierr;
#define PERMON_ASSERT(c, ...) \
  if (PetscUnlikely(!(c))) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, __VA_ARGS__);

#define PermonDebug(msg) \
  0; \
  do { \
    if (PermonDebugEnabled) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg)); \
  } while (0)
#define PermonDebug1(msg, a1) \
  0; \
  do { \
    if (PermonDebugEnabled) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1)); \
  } while (0)
#define PermonDebug2(msg, a1, a2) \
  0; \
  do { \
    if (PermonDebugEnabled) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1, a2)); \
  } while (0)
#define PermonDebug3(msg, a1, a2, a3) \
  0; \
  do { \
    if (PermonDebugEnabled) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1, a2, a3)); \
  } while (0)
#define PermonDebug4(msg, a1, a2, a3, a4) \
  0; \
  do { \
    if (PermonDebugEnabled) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1, a2, a3, a4)); \
  } while (0)
#define PermonDebug5(msg, a1, a2, a3, a4, a5) \
  0; \
  do { \
    if (PermonDebugEnabled) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1, a2, a3, a4, a5)); \
  } while (0)
#define PermonDebug6(msg, a1, a2, a3, a4, a5, a6) \
  0; \
  do { \
    if (PermonDebugEnabled) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1, a2, a3, a4, a5, a6)); \
  } while (0)

static inline PetscErrorCode PetscBoolGlobalAnd(MPI_Comm comm, PetscBool loc, PetscBool *glob)
{
  PetscFunctionBegin;
  PetscCallMPI(MPI_Allreduce(&loc, glob, 1, MPIU_BOOL, MPI_LAND, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscBoolGlobalOr(MPI_Comm comm, PetscBool loc, PetscBool *glob)
{
  PetscFunctionBegin;
  PetscCallMPI(MPI_Allreduce(&loc, glob, 1, MPIU_BOOL, MPI_LOR, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void FLLTIC(PetscLogDouble *t)
{
  PetscCallVoid(PetscTime(t));
}

static inline void FLLTOC(PetscLogDouble *t)
{
  PetscLogDouble toc_time;

  PetscCallVoid(PetscTime(&toc_time));
  *t = toc_time - *t;
}

PERMON_EXTERN PetscBool PermonTraceEnabled;
PERMON_EXTERN PetscInt  PeFuBe_i_;
PERMON_EXTERN char      PeFuBe_s_[128];

#define PermonTracedFunctionBegin \
  PetscLogDouble ttttt = 0.0; \
  PetscFunctionBegin;

#define PermonTraceBegin \
  if (PermonTraceEnabled) { \
    PeFuBe_s_[PeFuBe_i_]     = ' '; \
    PeFuBe_s_[PeFuBe_i_ + 1] = 0; \
    FLLTIC(&ttttt); \
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%d BEGIN FUNCTION %s\n", PeFuBe_s_, PeFuBe_i_, __FUNCT__)); \
    PeFuBe_i_++; \
  }

#define PetscFunctionBeginI \
  PermonTracedFunctionBegin; \
  PermonTraceBegin;

#define PetscFunctionReturnI(rrrrr) \
  { \
    if (PermonTraceEnabled) { \
      FLLTOC(&ttttt); \
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%d END   FUNCTION %s (%2.2f s)\n", PeFuBe_s_, --PeFuBe_i_, __FUNCT__, ttttt)); \
      PeFuBe_s_[PeFuBe_i_] = 0; \
      PetscFunctionReturn(rrrrr); \
    } else { \
      PetscFunctionReturn(rrrrr); \
    } \
  }

PERMON_EXTERN PetscErrorCode PermonInitialize(int *argc, char ***args, const char file[], const char help[]);
PERMON_EXTERN PetscErrorCode PermonFinalize();
PERMON_EXTERN PetscErrorCode PermonProcessInfoExclusions(PetscClassId id, const char *className);
PERMON_EXTERN PetscErrorCode PermonMakePath(const char *dir, mode_t mode);

PERMON_EXTERN PetscErrorCode PermonCreate(MPI_Comm comm, PERMON *permon_new);
PERMON_EXTERN PetscErrorCode PermonDestroy(PERMON *permon);

PERMON_EXTERN PetscErrorCode PermonSetObjectInfo(PetscBool flg);
PERMON_EXTERN PetscErrorCode PermonSetTrace(PetscBool flg);
PERMON_EXTERN PetscErrorCode PermonSetDebug(PetscBool flg);
PERMON_EXTERN PetscErrorCode PermonSetFromOptions();

PERMON_EXTERN PetscErrorCode PermonEventRegLogGetEvent(PetscEventRegLog eventLog, const char name[], PetscLogEvent *event, PetscBool *exists);
PERMON_EXTERN PetscErrorCode PermonPetscLogEventGetId(const char name[], PetscLogEvent *event, PetscBool *exists);

PERMON_EXTERN PetscErrorCode PermonPetscInfoDeactivateAll();

PERMON_EXTERN PetscErrorCode PermonPetscObjectInheritName(PetscObject dest, PetscObject orig, const char *suffix);
PERMON_EXTERN PetscErrorCode PermonPetscObjectInheritPrefix(PetscObject obj, PetscObject orig, const char *suffix);
PERMON_EXTERN PetscErrorCode PermonPetscObjectInheritPrefixIfNotSet(PetscObject obj, PetscObject orig, const char *suffix);
