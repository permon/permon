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

/* MUMPS uses MUMPS_INT for nonzero indices such as irn/jcn, irn_loc/jcn_loc and uses int64_t for
   number of nonzeros such as nnz, nnz_loc. We typedef MUMPS_INT to PetscMUMPSInt to follow the
   naming convention in PetscMPIInt, PetscBLASInt etc.
*/
typedef MUMPS_INT PetscMUMPSInt;

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

typedef struct Mat_MUMPS Mat_MUMPS;
struct Mat_MUMPS {
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

  MatStructure   matstruc;
  PetscMPIInt    myid,petsc_size;
  PetscMUMPSInt  *irn,*jcn;             /* the (i,j,v) triplets passed to mumps. */
  PetscScalar    *val,*val_alloc;       /* For some matrices, we can directly access their data array without a buffer. For others, we need a buffer. So comes val_alloc. */
  PetscInt64     nnz;                   /* number of nonzeros. The type is called selective 64-bit in mumps */
  PetscMUMPSInt  sym;
  MPI_Comm       mumps_comm;
  PetscMUMPSInt  ICNTL9_pre;            /* check if ICNTL(9) is changed from previous MatSolve */
  VecScatter     scat_rhs, scat_sol;    /* used by MatSolve() */
  Vec            b_seq,x_seq;
  PetscInt       ninfo,*info;           /* which INFO to display */
  PetscInt       sizeredrhs;
  PetscScalar    *schur_sol;
  PetscInt       schur_sizesol;
  PetscMUMPSInt  *ia_alloc,*ja_alloc;   /* work arrays used for the CSR struct for sparse rhs */
  PetscInt64     cur_ilen,cur_jlen;     /* current len of ia_alloc[], ja_alloc[] */
  PetscErrorCode (*ConvertToTriples)(Mat,PetscInt,MatReuse,Mat_MUMPS*);

  /* stuff used by petsc/mumps OpenMP support*/
  PetscBool      use_petsc_omp_support;
  PetscOmpCtrl   omp_ctrl;              /* an OpenMP controler that blocked processes will release their CPU (MPI_Barrier does not have this guarantee) */
  MPI_Comm       petsc_comm,omp_comm;   /* petsc_comm is petsc matrix's comm */
  PetscInt64     *recvcount;            /* a collection of nnz on omp_master */
  PetscMPIInt    tag,omp_comm_size;
  PetscBool      is_omp_master;         /* is this rank the master of omp_comm */
  MPI_Request    *reqs;
};

#endif
