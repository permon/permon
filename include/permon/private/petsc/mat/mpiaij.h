
#if !defined(__MPIAIJ_H)
#define __MPIAIJ_H

#include "aij.h"

typedef struct { /* used by MatCreateMPIAIJSumSeqAIJ for reusing the merged matrix */
  PetscLayout rowmap;
  PetscInt    **buf_ri,**buf_rj;
  PetscMPIInt *len_s,*len_r,*id_r;    /* array of length of comm->size, store send/recv matrix values */
  PetscMPIInt nsend,nrecv;
  PetscInt    *bi,*bj;    /* i and j array of the local portion of mpi C (matrix product) - rename to ci, cj! */
  PetscInt    *owners_co,*coi,*coj;    /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*duplicate)(Mat,MatDuplicateOption,Mat*);
} Mat_Merge_SeqsToMPI;

typedef struct { /* used by MatPtAP_MPIAIJ_MPIAIJ() and MatMatMult_MPIAIJ_MPIAIJ() */
  PetscInt    *startsj_s,*startsj_r;    /* used by MatGetBrowsOfAoCols_MPIAIJ */
  PetscScalar *bufa;                    /* used by MatGetBrowsOfAoCols_MPIAIJ */
  Mat         P_loc,P_oth;     /* partial B_seq -- intend to replace B_seq */
  PetscInt    *api,*apj;       /* symbolic i and j arrays of the local product A_loc*B_seq */
  PetscScalar *apv;
  MatReuse    reuse;           /* flag to skip MatGetBrowsOfAoCols_MPIAIJ() and MatMPIAIJGetLocalMat() in 1st call of MatPtAPNumeric_MPIAIJ_MPIAIJ() */
  PetscScalar *apa;            /* tmp array for store a row of A*P used in MatMatMult() */
  Mat         A_loc;           /* used by MatTransposeMatMult(), contains api and apj */
  Mat         Pt;              /* used by MatTransposeMatMult(), Pt = P^T */
  PetscBool   scalable;        /* flag determines scalable or non-scalable implementation */
  Mat         Rd,Ro,AP_loc,C_loc,C_oth;
  PetscInt    algType;         /* implementation algorithm */

  Mat_Merge_SeqsToMPI *merge;
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*duplicate)(Mat,MatDuplicateOption,Mat*);
  PetscErrorCode (*view)(Mat,PetscViewer);
} Mat_PtAPMPI;

typedef struct {
  Mat A,B;                             /* local submatrices: A (diag part),
                                           B (off-diag part) */
  PetscMPIInt size;                     /* size of communicator */
  PetscMPIInt rank;                     /* rank of proc in communicator */

  /* The following variables are used for matrix assembly */
  PetscBool   donotstash;               /* PETSC_TRUE if off processor entries dropped */
  MPI_Request *send_waits;              /* array of send requests */
  MPI_Request *recv_waits;              /* array of receive requests */
  PetscInt    nsends,nrecvs;           /* numbers of sends and receives */
  PetscScalar *svalues,*rvalues;       /* sending and receiving data */
  PetscInt    rmax;                     /* maximum message length */
#if defined(PETSC_USE_CTABLE)
  PetscTable colmap;
#else
  PetscInt *colmap;                     /* local col number of off-diag col */
#endif
  PetscInt *garray;                     /* global index of all off-processor columns */

  /* The following variables are used for matrix-vector products */
  Vec        lvec;                 /* local vector */
  Vec        diag;
  VecScatter Mvctx;                /* scatter context for vector */
  PetscBool  roworiented;          /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */
  PetscInt    *rowindices;         /* column indices for row */
  PetscScalar *rowvalues;          /* nonzero values in row */
  PetscBool   getrowactive;        /* indicates MatGetRow(), not restored */

  /* Used by MatDistribute_MPIAIJ() to allow reuse of previous matrix allocation  and nonzero pattern */
  PetscInt *ld;                    /* number of entries per row left of diagona block */

  /* Used by MatMatMult() and MatPtAP() */
  Mat_PtAPMPI *ptap;

  /* used by MatMatMatMult() */
  Mat_MatMatMatMult *matmatmatmult;

  /* Used by MPICUSP and MPICUSPARSE classes */
  void * spptr;

} Mat_MPIAIJ;

#endif
