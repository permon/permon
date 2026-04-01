/* This file is a stripped-down version of
   src/mat/impls/aij/mpi/mumps/mumps.c
   found in the PETSc source code.

   The original PETSc code is licensed under the BSD 2-Clause "Simplified" License.
   See the LICENSE file in this directory for full terms:
   ./LICENSE or https://gitlab.com/petsc/petsc/-/blob/main/LICENSE
*/

#pragma once

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  #include <cmumps_c.h>
  #include <zmumps_c.h>
  #include <smumps_c.h>
  #include <dmumps_c.h>
#else
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #include <cmumps_c.h>
      #define MUMPS_c     cmumps_c
      #define MumpsScalar CMUMPS_COMPLEX
    #else
      #include <zmumps_c.h>
      #define MUMPS_c     zmumps_c
      #define MumpsScalar ZMUMPS_COMPLEX
    #endif
  #else
    #if defined(PETSC_USE_REAL_SINGLE)
      #include <smumps_c.h>
      #define MUMPS_c     smumps_c
      #define MumpsScalar SMUMPS_REAL
    #else
      #include <dmumps_c.h>
      #define MUMPS_c     dmumps_c
      #define MumpsScalar DMUMPS_REAL
    #endif
  #endif
#endif
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define MUMPS_STRUC_C CMUMPS_STRUC_C
  #else
    #define MUMPS_STRUC_C ZMUMPS_STRUC_C
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    #define MUMPS_STRUC_C SMUMPS_STRUC_C
  #else
    #define MUMPS_STRUC_C DMUMPS_STRUC_C
  #endif
#endif
EXTERN_C_END

#define JOB_INIT         -1
#define JOB_NULL         0
#define JOB_FACTSYMBOLIC 1
#define JOB_FACTNUMERIC  2
#define JOB_SOLVE        3
#define JOB_END          -2

/* MUMPS uses MUMPS_INT for nonzero indices such as irn/jcn, irn_loc/jcn_loc and uses int64_t for
   number of nonzeros such as nnz, nnz_loc. We typedef MUMPS_INT to PetscMUMPSInt to follow the
   naming convention in PetscMPIInt, PetscBLASInt etc.
*/
typedef MUMPS_INT PetscMUMPSInt;

// An abstract type for specific MUMPS types {S,D,C,Z}MUMPS_STRUC_C.
//
// With the abstract (outer) type, we can write shared code. We call MUMPS through a type-to-be-determined inner field within the abstract type.
// Before/after calling MUMPS, we need to copy in/out fields between the outer and the inner, which seems expensive. But note that the large fixed size
// arrays within the types are directly linked. At the end, we only need to copy ~20 intergers/pointers, which is doable. See PreMumpsCall()/PostMumpsCall().
//
// Not all fields in the specific types are exposed in the abstract type. We only need those used by the PETSc/MUMPS interface.
// Notably, DMUMPS_COMPLEX* and DMUMPS_REAL* fields are now declared as void *. Their type will be determined by the the actual precision to be used.
// Also note that we added some *_len fields not in specific types to track sizes of those MumpsScalar buffers.
typedef struct {
  PetscPrecision precision;   // precision used by MUMPS
  void          *internal_id; // the data structure passed to MUMPS, whose actual type {S,D,C,Z}MUMPS_STRUC_C is to be decided by precision and PETSc's use of complex

  // aliased fields from internal_id, so that we can use XMUMPS_STRUC_C to write shared code across different precisions.
  MUMPS_INT  sym, par, job;
  MUMPS_INT  comm_fortran; /* Fortran communicator */
  MUMPS_INT *icntl;
  void      *cntl; // MumpsReal, fixed size array
  MUMPS_INT  n;
  MUMPS_INT  nblk;

  /* Assembled entry */
  MUMPS_INT8 nnz;
  MUMPS_INT *irn;
  MUMPS_INT *jcn;
  void      *a; // MumpsScalar, centralized input
  PetscCount a_len;

  /* Distributed entry */
  MUMPS_INT8 nnz_loc;
  MUMPS_INT *irn_loc;
  MUMPS_INT *jcn_loc;
  void      *a_loc; // MumpsScalar, distributed input
  PetscCount a_loc_len;

  /* Matrix by blocks */
  MUMPS_INT *blkptr;
  MUMPS_INT *blkvar;

  /* Ordering, if given by user */
  MUMPS_INT *perm_in;

  /* RHS, solution, ouptput data and statistics */
  void      *rhs, *redrhs, *rhs_sparse, *sol_loc, *rhs_loc;                 // MumpsScalar buffers
  PetscCount rhs_len, redrhs_len, rhs_sparse_len, sol_loc_len, rhs_loc_len; // length of buffers (in MumpsScalar) IF allocated in a different precision than PetscScalar

  MUMPS_INT *irhs_sparse, *irhs_ptr, *isol_loc, *irhs_loc;
  MUMPS_INT  nrhs, lrhs, lredrhs, nz_rhs, lsol_loc, nloc_rhs, lrhs_loc;
  // MUMPS_INT  nsol_loc; // introduced in MUMPS-5.7, but PETSc doesn't use it; would cause compile errors with the widely used 5.6. If you add it, must also update PreMumpsCall() and guard this with #if PETSC_PKG_MUMPS_VERSION_GE(5, 7, 0)
  MUMPS_INT  schur_lld;
  MUMPS_INT *info, *infog;   // fixed size array
  void      *rinfo, *rinfog; // MumpsReal, fixed size array

  /* Null space */
  MUMPS_INT *pivnul_list; // allocated by MUMPS!
  MUMPS_INT *mapping;     // allocated by MUMPS!

  /* Schur */
  MUMPS_INT  size_schur;
  MUMPS_INT *listvar_schur;
  void      *schur; // MumpsScalar
  PetscCount schur_len;

  /* For out-of-core */
  char *ooc_tmpdir; // fixed size array
  char *ooc_prefix; // fixed size array
} XMUMPS_STRUC_C;

// Make a companion MumpsScalar array (with a given PetscScalar array), to hold at least <n> MumpsScalars in the given <precision> and return the address at <ma>.
// <convert> indicates if we need to convert PetscScalars to MumpsScalars after allocating the MumpsScalar array.
// (For bravity, we use <ma> for array address and <m> for its length in MumpsScalar, though in code they should be <*ma> and <*m>)
// If <ma> already points to a buffer/array, on input <m> should be its length. Note the buffer might be freed if it is not big enough for this request.
//
// The returned array is a companion, so how it is created depends on if PetscScalar and MumpsScalar are the same.
// 1) If they are different, a separate array will be made and its length and address will be provided at <m> and <ma> on output.
// 2) Otherwise, <pa> will be returned in <ma>, and <m> will be zero on output.
//
//
//   Input parameters:
// + convert   - whether to do PetscScalar to MumpsScalar conversion
// . n         - length of the PetscScalar array
// . pa        - [n]], points to the PetscScalar array
// . precision - precision of MumpsScalar
// . m         - on input, length of an existing MumpsScalar array <ma> if any, otherwise *m is just zero.
// - ma        - on input, an existing MumpsScalar array if any.
//
//   Output parameters:
// + m  - length of the MumpsScalar buffer at <ma> if MumpsScalar is different from PetscScalar, otherwise 0
// . ma - the MumpsScalar array, which could be an alias of <pa> when the two types are the same.
//
//   Note:
//    New memory, if allocated, is done via PetscMalloc1(), and is owned by caller.
static PetscErrorCode MatMumpsMakeMumpsScalarArray(PetscBool convert, PetscCount n, const PetscScalar *pa, PetscPrecision precision, PetscCount *m, void **ma)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  const PetscPrecision mumps_precision = precision;
  PetscCheck(precision == PETSC_PRECISION_SINGLE || precision == PETSC_PRECISION_DOUBLE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported precicison (%d). Must be single or double", (int)precision);
  #if defined(PETSC_USE_COMPLEX)
  if (mumps_precision != PETSC_SCALAR_PRECISION) {
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      if (*m < n) {
        PetscCall(PetscFree(*ma));
        PetscCall(PetscMalloc1(n, (CMUMPS_COMPLEX **)ma));
        *m = n;
      }
      if (convert) {
        CMUMPS_COMPLEX *b = *(CMUMPS_COMPLEX **)ma;
        for (PetscCount i = 0; i < n; i++) {
          b[i].r = PetscRealPart(pa[i]);
          b[i].i = PetscImaginaryPart(pa[i]);
        }
      }
    } else {
      if (*m < n) {
        PetscCall(PetscFree(*ma));
        PetscCall(PetscMalloc1(n, (ZMUMPS_COMPLEX **)ma));
        *m = n;
      }
      if (convert) {
        ZMUMPS_COMPLEX *b = *(ZMUMPS_COMPLEX **)ma;
        for (PetscCount i = 0; i < n; i++) {
          b[i].r = PetscRealPart(pa[i]);
          b[i].i = PetscImaginaryPart(pa[i]);
        }
      }
    }
  }
  #else
  if (mumps_precision != PETSC_SCALAR_PRECISION) {
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      if (*m < n) {
        PetscCall(PetscFree(*ma));
        PetscCall(PetscMalloc1(n, (SMUMPS_REAL **)ma));
        *m = n;
      }
      if (convert) {
        SMUMPS_REAL *b = *(SMUMPS_REAL **)ma;
        for (PetscCount i = 0; i < n; i++) b[i] = pa[i];
      }
    } else {
      if (*m < n) {
        PetscCall(PetscFree(*ma));
        PetscCall(PetscMalloc1(n, (DMUMPS_REAL **)ma));
        *m = n;
      }
      if (convert) {
        DMUMPS_REAL *b = *(DMUMPS_REAL **)ma;
        for (PetscCount i = 0; i < n; i++) b[i] = pa[i];
      }
    }
  }
  #endif
  else
#endif
  {
    if (*m != 0) PetscCall(PetscFree(*ma)); // free existing buffer if any
    *ma = (void *)pa;                       // same precision, make them alias
    *m  = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PreMumpsCall(inner, outer, mumpsscalar) \
  do { \
    inner->job           = outer->job; \
    inner->n             = outer->n; \
    inner->nblk          = outer->nblk; \
    inner->nnz           = outer->nnz; \
    inner->irn           = outer->irn; \
    inner->jcn           = outer->jcn; \
    inner->a             = (mumpsscalar *)outer->a; \
    inner->nnz_loc       = outer->nnz_loc; \
    inner->irn_loc       = outer->irn_loc; \
    inner->jcn_loc       = outer->jcn_loc; \
    inner->a_loc         = (mumpsscalar *)outer->a_loc; \
    inner->blkptr        = outer->blkptr; \
    inner->blkvar        = outer->blkvar; \
    inner->perm_in       = outer->perm_in; \
    inner->rhs           = (mumpsscalar *)outer->rhs; \
    inner->redrhs        = (mumpsscalar *)outer->redrhs; \
    inner->rhs_sparse    = (mumpsscalar *)outer->rhs_sparse; \
    inner->sol_loc       = (mumpsscalar *)outer->sol_loc; \
    inner->rhs_loc       = (mumpsscalar *)outer->rhs_loc; \
    inner->irhs_sparse   = outer->irhs_sparse; \
    inner->irhs_ptr      = outer->irhs_ptr; \
    inner->isol_loc      = outer->isol_loc; \
    inner->irhs_loc      = outer->irhs_loc; \
    inner->nrhs          = outer->nrhs; \
    inner->lrhs          = outer->lrhs; \
    inner->lredrhs       = outer->lredrhs; \
    inner->nz_rhs        = outer->nz_rhs; \
    inner->lsol_loc      = outer->lsol_loc; \
    inner->nloc_rhs      = outer->nloc_rhs; \
    inner->lrhs_loc      = outer->lrhs_loc; \
    inner->schur_lld     = outer->schur_lld; \
    inner->size_schur    = outer->size_schur; \
    inner->listvar_schur = outer->listvar_schur; \
    inner->schur         = (mumpsscalar *)outer->schur; \
  } while (0)

#define PostMumpsCall(inner, outer) \
  do { \
    outer->pivnul_list = inner->pivnul_list; \
    outer->mapping     = inner->mapping; \
  } while (0)

// Cast a MumpsScalar array <ma[n]> in <mumps_precision> to a PetscScalar array at address <pa>.
//
// 1) If the two types are different, cast array elements.
// 2) Otherwise, this works as a memcpy; of course, if the two addresses are equal, it is a no-op.
static PetscErrorCode MatMumpsCastMumpsScalarArray(PetscCount n, PetscPrecision mumps_precision, const void *ma, PetscScalar *pa)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  if (mumps_precision != PETSC_SCALAR_PRECISION) {
  #if defined(PETSC_USE_COMPLEX)
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      PetscReal         *a = (PetscReal *)pa;
      const SMUMPS_REAL *b = (const SMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < 2 * n; i++) a[i] = b[i];
    } else {
      PetscReal         *a = (PetscReal *)pa;
      const DMUMPS_REAL *b = (const DMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < 2 * n; i++) a[i] = b[i];
    }
  #else
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      const SMUMPS_REAL *b = (const SMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < n; i++) pa[i] = b[i];
    } else {
      const DMUMPS_REAL *b = (const DMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < n; i++) pa[i] = b[i];
    }
  #endif
  } else
#endif
    PetscCall(PetscArraycpy((PetscScalar *)pa, (PetscScalar *)ma, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Entry for PETSc to call mumps
static inline PetscErrorCode PetscCallMumps_Private(XMUMPS_STRUC_C *outer)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  #if defined(PETSC_USE_COMPLEX)
  if (outer->precision == PETSC_PRECISION_SINGLE) {
    CMUMPS_STRUC_C *inner = (CMUMPS_STRUC_C *)outer->internal_id;
    PreMumpsCall(inner, outer, CMUMPS_COMPLEX);
    PetscCallExternalVoid("cmumps_c", cmumps_c(inner));
    PostMumpsCall(inner, outer);
  } else {
    ZMUMPS_STRUC_C *inner = (ZMUMPS_STRUC_C *)outer->internal_id;
    PreMumpsCall(inner, outer, ZMUMPS_COMPLEX);
    PetscCallExternalVoid("zmumps_c", zmumps_c(inner));
    PostMumpsCall(inner, outer);
  }
  #else
  if (outer->precision == PETSC_PRECISION_SINGLE) {
    SMUMPS_STRUC_C *inner = (SMUMPS_STRUC_C *)outer->internal_id;
    PreMumpsCall(inner, outer, SMUMPS_REAL);
    PetscCallExternalVoid("smumps_c", smumps_c(inner));
    PostMumpsCall(inner, outer);
  } else {
    DMUMPS_STRUC_C *inner = (DMUMPS_STRUC_C *)outer->internal_id;
    PreMumpsCall(inner, outer, DMUMPS_REAL);
    PetscCallExternalVoid("dmumps_c", dmumps_c(inner));
    PostMumpsCall(inner, outer);
  }
  #endif
#else
  MUMPS_STRUC_C *inner = (MUMPS_STRUC_C *)outer->internal_id;
  PreMumpsCall(inner, outer, MumpsScalar);
  PetscCallExternalVoid(PetscStringize(MUMPS_c), MUMPS_c(inner));
  PostMumpsCall(inner, outer);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* macros s.t. indices match MUMPS documentation */
#define ICNTL(I) icntl[(I) - 1]
#define INFOG(I) infog[(I) - 1]
#define INFO(I)  info[(I) - 1]

/* if using PETSc OpenMP support, we only call MUMPS on master ranks. Before/after the call, we change/restore CPUs the master ranks can run on */
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
  #define PetscMUMPS_c(mumps) \
    do { \
      if (mumps->use_petsc_omp_support) { \
        if (mumps->is_omp_master) { \
          PetscCall(PetscOmpCtrlOmpRegionOnMasterBegin(mumps->omp_ctrl)); \
          PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF)); \
          PetscCall(PetscCallMumps_Private(&mumps->id)); \
          PetscCall(PetscFPTrapPop()); \
          PetscCall(PetscOmpCtrlOmpRegionOnMasterEnd(mumps->omp_ctrl)); \
        } \
        PetscCall(PetscOmpCtrlBarrier(mumps->omp_ctrl)); \
        /* Global info is same on all processes so we Bcast it within omp_comm. Local info is specific      \
         to processes, so we only Bcast info[1], an error code and leave others (since they do not have   \
         an easy translation between omp_comm and petsc_comm). See MUMPS-5.1.2 manual p82.                   \
         omp_comm is a small shared memory communicator, hence doing multiple Bcast as shown below is OK. \
      */ \
        MUMPS_STRUC_C tmp; /* All MUMPS_STRUC_C types have same lengths on these info arrays */ \
        PetscCallMPI(MPI_Bcast(mumps->id.infog, PETSC_STATIC_ARRAY_LENGTH(tmp.infog), MPIU_MUMPSINT, 0, mumps->omp_comm)); \
        PetscCallMPI(MPI_Bcast(mumps->id.info, PETSC_STATIC_ARRAY_LENGTH(tmp.info), MPIU_MUMPSINT, 0, mumps->omp_comm)); \
        PetscCallMPI(MPI_Bcast(mumps->id.rinfog, PETSC_STATIC_ARRAY_LENGTH(tmp.rinfog), MPIU_MUMPSREAL(&mumps->id), 0, mumps->omp_comm)); \
        PetscCallMPI(MPI_Bcast(mumps->id.rinfo, PETSC_STATIC_ARRAY_LENGTH(tmp.rinfo), MPIU_MUMPSREAL(&mumps->id), 0, mumps->omp_comm)); \
      } else { \
        PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF)); \
        PetscCall(PetscCallMumps_Private(&mumps->id)); \
        PetscCall(PetscFPTrapPop()); \
      } \
    } while (0)
#else
  #define PetscMUMPS_c(mumps) \
    do { \
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF)); \
      PetscCall(PetscCallMumps_Private(&mumps->id)); \
      PetscCall(PetscFPTrapPop()); \
    } while (0)
#endif

typedef struct Mat_MUMPS Mat_MUMPS;
struct Mat_MUMPS {
  XMUMPS_STRUC_C id;

  MatStructure   matstruc;
  PetscMPIInt    myid, petsc_size;
  PetscMUMPSInt *irn, *jcn;       /* the (i,j,v) triplets passed to mumps. */
  PetscScalar   *val, *val_alloc; /* For some matrices, we can directly access their data array without a buffer. For others, we need a buffer. So comes val_alloc. */
  PetscCount     nnz;             /* number of nonzeros. The type is called selective 64-bit in mumps */
  PetscMUMPSInt  sym;
  MPI_Comm       mumps_comm;
  PetscMUMPSInt *ICNTL_pre;
  PetscReal     *CNTL_pre;
  PetscMUMPSInt  ICNTL9_pre;         /* check if ICNTL(9) is changed from previous MatSolve */
  VecScatter     scat_rhs, scat_sol; /* used by MatSolve() */
  PetscMUMPSInt  ICNTL20;            /* use centralized (0) or distributed (10) dense RHS */
  PetscMUMPSInt  ICNTL26;
  PetscMUMPSInt  lrhs_loc, nloc_rhs, *irhs_loc;
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
  PetscInt    *rhs_nrow, max_nrhs;
  PetscMPIInt *rhs_recvcounts, *rhs_disps;
  PetscScalar *rhs_loc, *rhs_recvbuf;
#endif
  Vec            b_seq, x_seq;
  PetscInt       ninfo, *info; /* which INFO to display */
  PetscInt       sizeredrhs;
  PetscScalar   *schur_sol;
  PetscInt       schur_sizesol;
  PetscScalar   *redrhs;              // buffer in PetscScalar in case MumpsScalar is in a different precision
  PetscMUMPSInt *ia_alloc, *ja_alloc; /* work arrays used for the CSR struct for sparse rhs */
  PetscCount     cur_ilen, cur_jlen;  /* current len of ia_alloc[], ja_alloc[] */
  PetscErrorCode (*ConvertToTriples)(Mat, PetscInt, MatReuse, Mat_MUMPS *);

  /* Support for MATNEST */
  PetscErrorCode (**nest_convert_to_triples)(Mat, PetscInt, MatReuse, Mat_MUMPS *);
  PetscCount  *nest_vals_start;
  PetscScalar *nest_vals;

  /* stuff used by petsc/mumps OpenMP support*/
  PetscBool    use_petsc_omp_support;
  PetscOmpCtrl omp_ctrl;             /* an OpenMP controller that blocked processes will release their CPU (MPI_Barrier does not have this guarantee) */
  MPI_Comm     petsc_comm, omp_comm; /* petsc_comm is PETSc matrix's comm */
  PetscCount  *recvcount;            /* a collection of nnz on omp_master */
  PetscMPIInt  tag, omp_comm_size;
  PetscBool    is_omp_master; /* is this rank the master of omp_comm */
  MPI_Request *reqs;
};
