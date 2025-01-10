#pragma once

#include <petsc/private/matimpl.h>
#include <petsc/private/hashmapi.h>
#include <petsc/private/hashmapijv.h>

/*
 Used by MatCreateSubMatrices_MPIXAIJ_Local()
*/
typedef struct { /* used by MatCreateSubMatrices_MPIAIJ_SingleIS_Local() and MatCreateSubMatrices_MPIAIJ_Local */
  PetscInt   id; /* index of submats, only submats[0] is responsible for deleting some arrays below */
  PetscInt   nrqs, nrqr;
  PetscInt **rbuf1, **rbuf2, **rbuf3, **sbuf1, **sbuf2;
  PetscInt **ptr;
  PetscInt  *tmp;
  PetscInt  *ctr;
  PetscInt  *pa; /* proc array */
  PetscInt  *req_size, *req_source1, *req_source2;
  PetscBool  allcolumns, allrows;
  PetscBool  singleis;
  PetscInt  *row2proc; /* row to proc map */
  PetscInt   nstages;
#if defined(PETSC_USE_CTABLE)
  PetscHMapI cmap, rmap;
  PetscInt  *cmap_loc, *rmap_loc;
#else
  PetscInt *cmap, *rmap;
#endif
  PetscErrorCode (*destroy)(Mat);
} Mat_SubSppt;

/* Operations provided by MATSEQAIJ and its subclasses */
typedef struct {
  PetscErrorCode (*getarray)(Mat, PetscScalar **);
  PetscErrorCode (*restorearray)(Mat, PetscScalar **);
  PetscErrorCode (*getarrayread)(Mat, const PetscScalar **);
  PetscErrorCode (*restorearrayread)(Mat, const PetscScalar **);
  PetscErrorCode (*getarraywrite)(Mat, PetscScalar **);
  PetscErrorCode (*restorearraywrite)(Mat, PetscScalar **);
  PetscErrorCode (*getcsrandmemtype)(Mat, const PetscInt **, const PetscInt **, PetscScalar **, PetscMemType *);
} Mat_SeqAIJOps;

/*
    Struct header shared by SeqAIJ, SeqBAIJ and SeqSBAIJ matrix formats
*/
#define SEQAIJHEADER(datatype) \
  PetscBool         roworiented;  /* if true, row-oriented input, default */ \
  PetscInt          nonew;        /* 1 don't add new nonzeros, -1 generate error on new */ \
  PetscInt          nounused;     /* -1 generate error on unused space */ \
  PetscBool         singlemalloc; /* if true a, i, and j have been obtained with one big malloc */ \
  PetscInt          maxnz;        /* allocated nonzeros */ \
  PetscInt         *imax;         /* maximum space allocated for each row */ \
  PetscInt         *ilen;         /* actual length of each row */ \
  PetscInt         *ipre;         /* space preallocated for each row by user */ \
  PetscBool         free_imax_ilen; \
  PetscInt          reallocs;           /* number of mallocs done during MatSetValues() \
                                        as more values are set than were prealloced */ \
  PetscInt          rmax;               /* max nonzeros in any row */ \
  PetscBool         keepnonzeropattern; /* keeps matrix structure same in calls to MatZeroRows()*/ \
  PetscBool         ignorezeroentries; \
  PetscBool         free_ij;       /* free the column indices j and row offsets i when the matrix is destroyed */ \
  PetscBool         free_a;        /* free the numerical values when matrix is destroy */ \
  Mat_CompressedRow compressedrow; /* use compressed row format */ \
  PetscInt          nz;            /* nonzeros */ \
  PetscInt         *i;             /* pointer to beginning of each row */ \
  PetscInt         *j;             /* column values: j + i[k] - 1 is start of row k */ \
  PetscInt         *diag;          /* pointers to diagonal elements */ \
  PetscInt          nonzerorowcnt; /* how many rows have nonzero entries */ \
  PetscBool         free_diag; \
  datatype         *a;              /* nonzero elements */ \
  PetscScalar      *solve_work;     /* work space used in MatSolve */ \
  IS                row, col, icol; /* index sets, used for reorderings */ \
  PetscBool         pivotinblocks;  /* pivot inside factorization of each diagonal block */ \
  Mat               parent;         /* set if this matrix was formed with MatDuplicate(...,MAT_SHARE_NONZERO_PATTERN,....); \
                                         means that this shares some data structures with the parent including diag, ilen, imax, i, j */ \
  Mat_SubSppt      *submatis1;      /* used by MatCreateSubMatrices_MPIXAIJ_Local */ \
  Mat_SeqAIJOps     ops[1]          /* operations for SeqAIJ and its subclasses */

typedef struct {
  MatTransposeColoring matcoloring;
  Mat                  Bt_den;  /* dense matrix of B^T */
  Mat                  ABt_den; /* dense matrix of A*B^T */
  PetscBool            usecoloring;
} Mat_MatMatTransMult;

typedef struct { /* used by MatTransposeMatMult() */
  Mat At;        /* transpose of the first matrix */
  Mat mA;        /* maij matrix of A */
  Vec bt, ct;    /* vectors to hold locally transposed arrays of B and C */
  /* used by PtAP */
  void *data;
  PetscErrorCode (*destroy)(void *);
} Mat_MatTransMatMult;

typedef struct {
  PetscInt    *api, *apj; /* symbolic structure of A*P */
  PetscScalar *apa;       /* temporary array for storing one row of A*P */
} Mat_AP;

typedef struct {
  MatTransposeColoring matcoloring;
  Mat                  Rt;   /* sparse or dense matrix of R^T */
  Mat                  RARt; /* dense matrix of R*A*R^T */
  Mat                  ARt;  /* A*R^T used for the case -matrart_color_art */
  MatScalar           *work; /* work array to store columns of A*R^T used in MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense() */
  /* free intermediate products needed for PtAP */
  void *data;
  PetscErrorCode (*destroy)(void *);
} Mat_RARt;

typedef struct {
  Mat BC; /* temp matrix for storing B*C */
} Mat_MatMatMatMult;

/*
  MATSEQAIJ format - Compressed row storage (also called Yale sparse matrix
  format) or compressed sparse row (CSR).  The i[] and j[] arrays start at 0. For example,
  j[i[k]+p] is the pth column in row k.  Note that the diagonal
  matrix elements are stored with the rest of the nonzeros (not separately).
*/

/* Info about i-nodes (identical nodes) helper class for SeqAIJ */
typedef struct {
  MatScalar *bdiag, *ibdiag, *ssor_work; /* diagonal blocks of matrix used for MatSOR_SeqAIJ_Inode() */
  PetscInt   bdiagsize;                  /* length of bdiag and ibdiag */
  PetscBool  ibdiagvalid;                /* do ibdiag[] and bdiag[] contain the most recent values */

  PetscBool        use;
  PetscInt         node_count;       /* number of inodes */
  PetscInt        *size;             /* size of each inode */
  PetscInt         limit;            /* inode limit */
  PetscInt         max_limit;        /* maximum supported inode limit */
  PetscBool        checked;          /* if inodes have been checked for */
  PetscObjectState mat_nonzerostate; /* non-zero state when inodes were checked for */
} Mat_SeqAIJ_Inode;

typedef struct {
  SEQAIJHEADER(MatScalar);
  Mat_SeqAIJ_Inode inode;
  MatScalar       *saved_values; /* location for stashing nonzero values of matrix */

  PetscScalar *idiag, *mdiag, *ssor_work; /* inverse of diagonal entries, diagonal values and workspace for Eisenstat trick */
  PetscBool    idiagvalid;                /* current idiag[] and mdiag[] are valid */
  PetscScalar *ibdiag;                    /* inverses of block diagonals */
  PetscBool    ibdiagvalid;               /* inverses of block diagonals are valid. */
  PetscBool    diagonaldense;             /* all entries along the diagonal have been set; i.e. no missing diagonal terms */
  PetscScalar  fshift, omega;             /* last used omega and fshift */

  /* MatSetValuesCOO() related fields on host */
  PetscCount  coo_n; /* Number of entries in MatSetPreallocationCOO() */
  PetscCount  Atot;  /* Total number of valid (i.e., w/ non-negative indices) entries in the COO array */
  PetscCount *jmap;  /* perm[jmap[i]..jmap[i+1]) give indices of entries in v[] associated with i-th nonzero of the matrix */
  PetscCount *perm;  /* The permutation array in sorting (i,j) by row and then by col */

  /* MatSetValues() via hash related fields */
  PetscHMapIJV   ht;
  PetscInt      *dnz;
  struct _MatOps cops;
} Mat_SeqAIJ;

/*
  Frees the a, i, and j arrays from the XAIJ (AIJ, BAIJ, and SBAIJ) matrix types
*/
static inline PetscErrorCode MatSeqXAIJFreeAIJ(Mat AA, MatScalar **a, PetscInt **j, PetscInt **i)
{
  Mat_SeqAIJ *A = (Mat_SeqAIJ *)AA->data;

  PetscFunctionBegin;
  if (A->singlemalloc) {
    PetscCall(PetscFree3(*a, *j, *i));
  } else {
    if (A->free_a) PetscCall(PetscFree(*a));
    if (A->free_ij) PetscCall(PetscFree(*j));
    if (A->free_ij) PetscCall(PetscFree(*i));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
