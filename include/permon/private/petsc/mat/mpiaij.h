#ifndef __MPIAIJ_H
#define __MPIAIJ_H

#include "aij.h"

typedef struct { /* used by MatCreateMPIAIJSumSeqAIJ for reusing the merged matrix */
  PetscLayout  rowmap;
  PetscInt   **buf_ri, **buf_rj;
  PetscMPIInt *len_s, *len_r, *id_r; /* array of length of comm->size, store send/recv matrix values */
  PetscMPIInt  nsend, nrecv;
  PetscInt    *bi, *bj;               /* i and j array of the local portion of mpi C (matrix product) - rename to ci, cj! */
  PetscInt    *owners_co, *coi, *coj; /* i and j array of (p->B)^T*A*P - used in the communication */
} Mat_Merge_SeqsToMPI;

typedef struct {                                /* used by MatPtAPXXX_MPIAIJ_MPIAIJ() and MatMatMultXXX_MPIAIJ_MPIAIJ() */
  PetscInt              *startsj_s, *startsj_r; /* used by MatGetBrowsOfAoCols_MPIAIJ */
  PetscScalar           *bufa;                  /* used by MatGetBrowsOfAoCols_MPIAIJ */
  Mat                    P_loc, P_oth;          /* partial B_seq -- intend to replace B_seq */
  PetscInt              *api, *apj;             /* symbolic i and j arrays of the local product A_loc*B_seq */
  PetscScalar           *apv;
  MatReuse               reuse; /* flag to skip MatGetBrowsOfAoCols_MPIAIJ() and MatMPIAIJGetLocalMat() in 1st call of MatPtAPNumeric_MPIAIJ_MPIAIJ() */
  PetscScalar           *apa;   /* tmp array for store a row of A*P used in MatMatMult() */
  Mat                    A_loc; /* used by MatTransposeMatMult(), contains api and apj */
  ISLocalToGlobalMapping ltog;  /* mapping from local column indices to global column indices for A_loc */
  Mat                    Pt;    /* used by MatTransposeMatMult(), Pt = P^T */
  Mat                    Rd, Ro, AP_loc, C_loc, C_oth;
  PetscInt               algType; /* implementation algorithm */
  PetscSF                sf;      /* use it to communicate remote part of C */
  PetscInt              *c_othi, *c_rmti;

  Mat_Merge_SeqsToMPI *merge;
} Mat_APMPI;

typedef struct {
  Mat         A, B; /* local submatrices: A (diag part),
                                           B (off-diag part) */
  PetscMPIInt size; /* size of communicator */
  PetscMPIInt rank; /* rank of proc in communicator */

  /* The following variables are used for matrix assembly */
  PetscBool    donotstash;        /* PETSC_TRUE if off processor entries dropped */
  MPI_Request *send_waits;        /* array of send requests */
  MPI_Request *recv_waits;        /* array of receive requests */
  PetscInt     nsends, nrecvs;    /* numbers of sends and receives */
  PetscScalar *svalues, *rvalues; /* sending and receiving data */
  PetscInt     rmax;              /* maximum message length */
#if defined(PETSC_USE_CTABLE)
  PetscTable colmap;
#else
  PetscInt *colmap; /* local col number of off-diag col */
#endif
  PetscInt *garray; /* global index of all off-processor columns */

  /* The following variables are used for matrix-vector products */
  Vec        lvec; /* local vector */
  Vec        diag;
  VecScatter Mvctx;       /* scatter context for vector */
  PetscBool  roworiented; /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */
  PetscInt    *rowindices;   /* column indices for row */
  PetscScalar *rowvalues;    /* nonzero values in row */
  PetscBool    getrowactive; /* indicates MatGetRow(), not restored */

  PetscInt *ld; /* number of entries per row left of diagonal block */

  /* Used by device classes */
  void *spptr;

  /* MatSetValuesCOO() related stuff */
  PetscCount   coo_n;                      /* Number of COOs passed to MatSetPreallocationCOO)() */
  PetscSF      coo_sf;                     /* SF to send/recv remote values in MatSetValuesCOO() */
  PetscCount   Annz, Bnnz;                 /* Number of entries in diagonal A and off-diagonal B */
  PetscCount   Annz2, Bnnz2;               /* Number of unique remote entries belonging to A and B */
  PetscCount   Atot1, Atot2, Btot1, Btot2; /* Total local (tot1) and remote (tot2) entries (which might contain repeats) belonging to A and B */
  PetscCount  *Ajmap1, *Aperm1;            /* Lengths: [Annz+1], [Atot1]. Local entries to diag */
  PetscCount  *Bjmap1, *Bperm1;            /* Lengths: [Bnnz+1], [Btot1]. Local entries to offdiag */
  PetscCount  *Aimap2, *Ajmap2, *Aperm2;   /* Lengths: [Annz2], [Annz2+1], [Atot2]. Remote entries to diag */
  PetscCount  *Bimap2, *Bjmap2, *Bperm2;   /* Lengths: [Bnnz2], [Bnnz2+1], [Btot2]. Remote entries to offdiag */
  PetscCount  *Cperm1;                     /* [sendlen] Permutation to fill MPI send buffer. 'C' for communication */
  PetscScalar *sendbuf, *recvbuf;          /* Buffers for remote values in MatSetValuesCOO() */
  PetscInt     sendlen, recvlen;           /* Lengths (in unit of PetscScalar) of send/recvbuf */
} Mat_MPIAIJ;


#endif
