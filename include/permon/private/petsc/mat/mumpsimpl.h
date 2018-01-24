/*
    Provides an interface to the MUMPS sparse solver
*/
#if !defined(__MUMPSIMPL_H)                                                     
#define __MUMPSIMPL_H    

EXTERN_C_BEGIN
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#include <cmumps_c.h>
#else
#include <zmumps_c.h>
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
#include <smumps_c.h>
#else
#include <dmumps_c.h>
#endif
#endif
EXTERN_C_END
#define JOB_INIT -1
#define JOB_FACTSYMBOLIC 1
#define JOB_FACTNUMERIC 2
#define JOB_SOLVE 3
#define JOB_END -2

/* calls to MUMPS */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define PetscMUMPS_c cmumps_c
#else
#define PetscMUMPS_c zmumps_c
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
#define PetscMUMPS_c smumps_c
#else
#define PetscMUMPS_c dmumps_c
#endif
#endif

/* declare MumpsScalar */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define MumpsScalar mumps_complex
#else
#define MumpsScalar mumps_double_complex
#endif
#else
#define MumpsScalar PetscScalar
#endif

/* macros s.t. indices match MUMPS documentation */
#define ICNTL(I) icntl[(I)-1]
#define CNTL(I) cntl[(I)-1]
#define INFOG(I) infog[(I)-1]
#define INFO(I) info[(I)-1]
#define RINFOG(I) rinfog[(I)-1]
#define RINFO(I) rinfo[(I)-1]

typedef struct {
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
  CMUMPS_STRUC_C id;
#else
  ZMUMPS_STRUC_C id;
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
  SMUMPS_STRUC_C id;
#else
  DMUMPS_STRUC_C id;
#endif
#endif

  MatStructure matstruc;
  PetscMPIInt  myid,size;
  PetscInt     *irn,*jcn,nz,sym;
  PetscScalar  *val;
  MPI_Comm     comm_mumps;
  PetscBool    isAIJ;
  PetscInt     ICNTL9_pre;           /* check if ICNTL(9) is changed from previous MatSolve */
  VecScatter   scat_rhs, scat_sol;   /* used by MatSolve() */
  Vec          b_seq,x_seq;
  PetscInt     ninfo,*info;          /* display INFO */
  PetscInt     sizeredrhs;
  PetscInt     *schur_pivots;
  PetscInt     schur_B_lwork;
  PetscScalar  *schur_work;
  PetscScalar  *schur_sol;
  PetscInt     schur_sizesol;
  PetscBool    schur_factored;
  PetscBool    schur_inverted;

  PetscErrorCode (*Destroy)(Mat);
  PetscErrorCode (*ConvertToTriples)(Mat, int, MatReuse, int*, int**, int**, PetscScalar**);
} Mat_MUMPS;

#endif
