
#if !defined(__AIJ_H)
#define __AIJ_H

#include <petsc/private/matimpl.h>
#include <petscctable.h>

/*
    Struct header shared by SeqAIJ, SeqBAIJ and SeqSBAIJ matrix formats
*/
#define SEQAIJHEADER(datatype)        \
  PetscBool roworiented;              /* if true, row-oriented input, default */ \
  PetscInt  nonew;                    /* 1 don't add new nonzeros, -1 generate error on new */ \
  PetscInt  nounused;                 /* -1 generate error on unused space */ \
  PetscBool singlemalloc;             /* if true a, i, and j have been obtained with one big malloc */ \
  PetscInt  maxnz;                    /* allocated nonzeros */ \
  PetscInt  *imax;                    /* maximum space allocated for each row */ \
  PetscInt  *ilen;                    /* actual length of each row */ \
  PetscBool free_imax_ilen;  \
  PetscInt  reallocs;                 /* number of mallocs done during MatSetValues() \
                                        as more values are set than were prealloced */\
  PetscInt          rmax;             /* max nonzeros in any row */ \
  PetscBool         keepnonzeropattern;   /* keeps matrix structure same in calls to MatZeroRows()*/ \
  PetscBool         ignorezeroentries; \
  PetscBool         free_ij;          /* free the column indices j and row offsets i when the matrix is destroyed */ \
  PetscBool         free_a;           /* free the numerical values when matrix is destroy */ \
  Mat_CompressedRow compressedrow;    /* use compressed row format */                      \
  PetscInt          nz;               /* nonzeros */                                       \
  PetscInt          *i;               /* pointer to beginning of each row */               \
  PetscInt          *j;               /* column values: j + i[k] - 1 is start of row k */  \
  PetscInt          *diag;            /* pointers to diagonal elements */                  \
  PetscInt          nonzerorowcnt;    /* how many rows have nonzero entries */             \
  PetscBool         free_diag;         \
  datatype          *a;               /* nonzero elements */                               \
  PetscScalar       *solve_work;      /* work space used in MatSolve */                    \
  IS                row, col, icol;   /* index sets, used for reorderings */ \
  PetscBool         pivotinblocks;    /* pivot inside factorization of each diagonal block */ \
  Mat               parent             /* set if this matrix was formed with MatDuplicate(...,MAT_SHARE_NONZERO_PATTERN,....);
                                         means that this shares some data structures with the parent including diag, ilen, imax, i, j */

typedef struct {
  MatTransposeColoring matcoloring;
  Mat                  Bt_den;       /* dense matrix of B^T */
  Mat                  ABt_den;      /* dense matrix of A*B^T */
  PetscBool            usecoloring;
  PetscErrorCode (*destroy)(Mat);
} Mat_MatMatTransMult;

typedef struct { /* for MatTransposeMatMult_SeqAIJ_SeqDense() */
  Mat          mA;           /* maij matrix of A */
  Vec          bt,ct;        /* vectors to hold locally transposed arrays of B and C */
  PetscErrorCode (*destroy)(Mat);
} Mat_MatTransMatMult;

typedef struct {
  PetscInt    *api,*apj;       /* symbolic structure of A*P */
  PetscScalar *apa;            /* temporary array for storing one row of A*P */
  PetscErrorCode (*destroy)(Mat);
} Mat_PtAP;

typedef struct {
  MatTransposeColoring matcoloring;
  Mat                  Rt;    /* sparse or dense matrix of R^T */
  Mat                  RARt;  /* dense matrix of R*A*R^T */
  Mat                  ARt;   /* A*R^T used for the case -matrart_color_art */
  MatScalar            *work; /* work array to store columns of A*R^T used in MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense() */
  PetscErrorCode (*destroy)(Mat);
} Mat_RARt;

typedef struct {
  Mat BC;               /* temp matrix for storing B*C */
  PetscErrorCode (*destroy)(Mat);
} Mat_MatMatMatMult;

/*
  MATSEQAIJ format - Compressed row storage (also called Yale sparse matrix
  format) or compressed sparse row (CSR).  The i[] and j[] arrays start at 0. For example,
  j[i[k]+p] is the pth column in row k.  Note that the diagonal
  matrix elements are stored with the rest of the nonzeros (not separately).
*/

/* Info about i-nodes (identical nodes) helper class for SeqAIJ */
typedef struct {
  MatScalar        *bdiag,*ibdiag,*ssor_work;        /* diagonal blocks of matrix used for MatSOR_SeqAIJ_Inode() */
  PetscInt         bdiagsize;                         /* length of bdiag and ibdiag */
  PetscBool        ibdiagvalid;                       /* do ibdiag[] and bdiag[] contain the most recent values */

  PetscBool        use;
  PetscInt         node_count;                     /* number of inodes */
  PetscInt         *size;                          /* size of each inode */
  PetscInt         limit;                          /* inode limit */
  PetscInt         max_limit;                      /* maximum supported inode limit */
  PetscBool        checked;                        /* if inodes have been checked for */
  PetscObjectState mat_nonzerostate;               /* non-zero state when inodes were checked for */
} Mat_SeqAIJ_Inode;

typedef struct {
  SEQAIJHEADER(MatScalar);
  Mat_SeqAIJ_Inode inode;
  MatScalar        *saved_values;             /* location for stashing nonzero values of matrix */

  PetscScalar *idiag,*mdiag,*ssor_work;       /* inverse of diagonal entries, diagonal values and workspace for Eisenstat trick */
  PetscBool   idiagvalid;                     /* current idiag[] and mdiag[] are valid */
  PetscScalar *ibdiag;                        /* inverses of block diagonals */
  PetscBool   ibdiagvalid;                    /* inverses of block diagonals are valid. */
  PetscScalar fshift,omega;                   /* last used omega and fshift */

  ISColoring coloring;                        /* set with MatADSetColoring() used by MatADSetValues() */

  PetscScalar       *matmult_abdense;    /* used by MatMatMult() */
  Mat_PtAP          *ptap;               /* used by MatPtAP() */
  Mat_MatMatMatMult *matmatmatmult;      /* used by MatMatMatMult() */
  Mat_RARt          *rart;               /* used by MatRARt() */
  Mat_MatMatTransMult *abt;              /* used by MatMatTransposeMult() */
} Mat_SeqAIJ;

#endif
