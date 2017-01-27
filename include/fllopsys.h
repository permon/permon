#if !defined(__FLLOPSYS_H)
#define	__FLLOPSYS_H

#include <petscksp.h>
#include <petsctime.h>

#if defined(PETSC_HAVE_SYS_TYPES_H)
#include <sys/types.h>
#include <sys/stat.h>
#endif

//TODO remove all these FLLOP_* macros
#define FLLOP_SETERRQ( comm,n,s) SETERRQ(comm,n,s)
#define FLLOP_SETERRQ1(comm,n,s,a1) SETERRQ1(comm,n,s,a1)
#define FLLOP_SETERRQ2(comm,n,s,a1,a2) SETERRQ2(comm,n,s,a1,a2)
#define FLLOP_SETERRQ3(comm,n,s,a1,a2,a3) SETERRQ3(comm,n,s,a1,a2,a3)
#define FLLOP_SETERRQ4(comm,n,s,a1,a2,a3,a4) SETERRQ4(comm,n,s,a1,a2,a3,a4)
#define FLLOP_SETERRQ5(comm,n,s,a1,a2,a3,a4,a5) SETERRQ5(comm,n,s,a1,a2,a3,a4,a5)
#define FLLOP_EXTERN PETSC_EXTERN
#define FLLOP_INTERN PETSC_INTERN

#include "flloppetscretro.h"

/* 
  FLLOP is a dummy class, defined in FllopInitialize,
  used to distinguish between PETSc and FLLOP sys routines while using PetscInfo*,
*/
typedef struct _p_FLLOP* FLLOP;
/*
  StateContainer is simple structure containing PetscObjectState
*/
typedef struct _n_StateContainer *StateContainer;
FLLOP_EXTERN PetscClassId FLLOP_CLASSID;
FLLOP_EXTERN FLLOP fllop;
FLLOP_EXTERN PetscBool FllopInfoEnabled, FllopObjectInfoEnabled, FllopDebugEnabled;

#ifdef NAME_MAX
#define FLLOP_MAX_NAME_LEN NAME_MAX+1
#else
#define FLLOP_MAX_NAME_LEN 256
#endif
#define FLLOP_MAX_PATH_LEN  PETSC_MAX_PATH_LEN
FLLOP_EXTERN char FLLOP_PathBuffer_Global[FLLOP_MAX_PATH_LEN];
FLLOP_EXTERN char FLLOP_ObjNameBuffer_Global[FLLOP_MAX_NAME_LEN];

/* BEGIN Function-like Macros */
FLLOP_EXTERN PetscErrorCode _fllop_ierr;
#define TRY(f) do {_fllop_ierr = f;CHKERRQ(_fllop_ierr);} while (0)
#define FLLOP_SETERRQ_WORLD( n,s)                 FLLOP_SETERRQ(PETSC_COMM_WORLD,n,s)
#define FLLOP_SETERRQ_WORLD1(n,s,a1)              FLLOP_SETERRQ1(PETSC_COMM_WORLD,n,s,a1)
#define FLLOP_SETERRQ_WORLD2(n,s,a1,a2)           FLLOP_SETERRQ2(PETSC_COMM_WORLD,n,s,a1,a2)
#define FLLOP_SETERRQ_WORLD3(n,s,a1,a2,a3)        FLLOP_SETERRQ3(PETSC_COMM_WORLD,n,s,a1,a2,a3)
#define FLLOP_SETERRQ_WORLD4(n,s,a1,a2,a3,a4)     FLLOP_SETERRQ4(PETSC_COMM_WORLD,n,s,a1,a2,a3,a4)
#define FLLOP_SETERRQ_WORLD5(n,s,a1,a2,a3,a4,a5)  FLLOP_SETERRQ5(PETSC_COMM_WORLD,n,s,a1,a2,a3,a4,a5)
#define FLLOP_ASSERT( c,cstr)                     if (PetscUnlikely(!(c))) FLLOP_SETERRQ( PETSC_COMM_SELF,PETSC_ERR_PLIB, "Assertion failed: " cstr);
#define FLLOP_ASSERT1(c,cstr,a1)                  if (PetscUnlikely(!(c))) FLLOP_SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Assertion failed: " cstr, a1);
#define FLLOP_ASSERT2(c,cstr,a1,a2)               if (PetscUnlikely(!(c))) FLLOP_SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Assertion failed: " cstr, a1,a2);
#define FLLOP_ASSERT3(c,cstr,a1,a2,a3)            if (PetscUnlikely(!(c))) FLLOP_SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Assertion failed: " cstr, a1,a2,a3);
#define FLLOP_ASSERT4(c,cstr,a1,a2,a3,a4)         if (PetscUnlikely(!(c))) FLLOP_SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Assertion failed: " cstr, a1,a2,a3,a4);
#define FLLOP_ASSERT5(c,cstr,a1,a2,a3,a4,a5)      if (PetscUnlikely(!(c))) FLLOP_SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Assertion failed: " cstr, a1,a2,a3,a4,a5);

#define FllopDebug(msg)                       0; do { if (FllopDebugEnabled) PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg); } while(0)
#define FllopDebug1(msg,a1)                   0; do { if (FllopDebugEnabled) PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1); } while(0)
#define FllopDebug2(msg,a1,a2)                0; do { if (FllopDebugEnabled) PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1,a2); } while(0)
#define FllopDebug3(msg,a1,a2,a3)             0; do { if (FllopDebugEnabled) PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1,a2,a3); } while(0)
#define FllopDebug4(msg,a1,a2,a3,a4)          0; do { if (FllopDebugEnabled) PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1,a2,a3,a4); } while(0)
#define FllopDebug5(msg,a1,a2,a3,a4,a5)       0; do { if (FllopDebugEnabled) PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1,a2,a3,a4,a5); } while(0)
#define FllopDebug6(msg,a1,a2,a3,a4,a5,a6)    0; do { if (FllopDebugEnabled) PetscPrintf(PETSC_COMM_WORLD, "*** " __FUNCT__ ": " msg, a1,a2,a3,a4,a5,a6); } while(0)

PETSC_STATIC_INLINE PetscErrorCode PetscBoolGlobalAnd(MPI_Comm comm,PetscBool loc,PetscBool *glob)
{
  return MPI_Allreduce(&loc,glob,1,MPIU_BOOL,MPI_LAND,comm);
}

PETSC_STATIC_INLINE PetscErrorCode PetscBoolGlobalOr(MPI_Comm comm,PetscBool loc,PetscBool *glob)
{
  return MPI_Allreduce(&loc,glob,1,MPIU_BOOL,MPI_LOR,comm);
}

PETSC_STATIC_INLINE void FLLTIC(PetscLogDouble *t) {
    PetscTime(t);
}

PETSC_STATIC_INLINE void FLLTOC(PetscLogDouble *t) {
    PetscLogDouble toc_time;
    PetscTime(&toc_time);
    *t = toc_time - *t;
}

FLLOP_EXTERN PetscBool FllopTraceEnabled;
FLLOP_EXTERN PetscInt PeFuBe_i_;
FLLOP_EXTERN char     PeFuBe_s_[128];

#define FllopTracedFunctionBegin \
PetscLogDouble ttttt=0.0;\
PetscFunctionBegin;

#define FllopTraceBegin \
if (FllopTraceEnabled) {\
    PeFuBe_s_[PeFuBe_i_]=' ';\
    PeFuBe_s_[PeFuBe_i_+1]=0;\
    FLLTIC(&ttttt);\
    TRY( PetscPrintf(PETSC_COMM_WORLD,"%s%d BEGIN FUNCTION %s\n",PeFuBe_s_,PeFuBe_i_,__FUNCT__) );\
    PeFuBe_i_++;\
}

#define PetscFunctionBeginI \
FllopTracedFunctionBegin;\
FllopTraceBegin;

#define PetscFunctionReturnI(rrrrr) \
{\
if (FllopTraceEnabled) {\
    FLLTOC(&ttttt);\
    TRY( PetscPrintf(PETSC_COMM_WORLD,"%s%d END   FUNCTION %s (%2.2f s)\n",PeFuBe_s_,--PeFuBe_i_,__FUNCT__,ttttt) );\
    PeFuBe_s_[PeFuBe_i_]=0;\
    PetscFunctionReturn(rrrrr);\
} else {\
    PetscFunctionReturn(rrrrr);\
}\
}

FLLOP_EXTERN PetscErrorCode FllopInitialize(int *argc, char ***args, const char file[]);
FLLOP_EXTERN PetscErrorCode FllopFinalize();
FLLOP_EXTERN PetscErrorCode FllopProcessInfoExclusions(PetscClassId id, const char *className);
FLLOP_EXTERN PetscErrorCode FllopMakePath(const char *dir, mode_t mode);

FLLOP_EXTERN PetscErrorCode FllopCreate(MPI_Comm comm,FLLOP *fllop_new);
FLLOP_EXTERN PetscErrorCode FllopDestroy(FLLOP *fllop);

FLLOP_EXTERN PetscErrorCode FllopSetObjectInfo(PetscBool flg);
FLLOP_EXTERN PetscErrorCode FllopSetTrace(PetscBool flg);
FLLOP_EXTERN PetscErrorCode FllopSetDebug(PetscBool flg);
FLLOP_EXTERN PetscErrorCode FllopSetFromOptions();

FLLOP_EXTERN PetscErrorCode FllopEventRegLogGetEvent(PetscEventRegLog eventLog, const char name[], PetscLogEvent *event, PetscBool *exists);
FLLOP_EXTERN PetscErrorCode FllopPetscLogEventGetId(const char name[], PetscLogEvent *event, PetscBool *exists);

FLLOP_EXTERN PetscErrorCode FllopPetscInfoDeactivateAll();

FLLOP_EXTERN PetscErrorCode FllopPetscObjectInheritName(PetscObject dest,PetscObject orig,const char *suffix);
FLLOP_EXTERN PetscErrorCode FllopPetscObjectInheritPrefix(PetscObject obj,PetscObject orig,const char *suffix);
FLLOP_EXTERN PetscErrorCode FllopPetscObjectInheritPrefixIfNotSet(PetscObject obj,PetscObject orig,const char *suffix);
#endif
