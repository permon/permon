
#if !defined(__DENSE_H)
#define __DENSE_H
#include "aij.h" /* Mat_MatTransMatMult is defined here */

/*
  MATSEQDENSE format - conventional dense Fortran storage (by columns)
*/

typedef struct {
  PetscScalar  *v;                /* matrix elements */
  PetscScalar  *unplacedarray;    /* if one called MatDensePlaceArray(), this is where it stashed the original */
  PetscBool    roworiented;       /* if true, row oriented input (default) */
  PetscInt     pad;               /* padding */
  PetscBLASInt *pivots;           /* pivots in LU factorization */
  PetscBLASInt lfwork;            /* length of work array in factorization */
  PetscScalar  *fwork;            /* work array in factorization */
  PetscBLASInt lda;               /* Lapack leading dimension of data */
  PetscBool    changelda;         /* change lda on resize? Default unless user set lda */
  PetscBLASInt Mmax,Nmax;         /* indicates the largest dimensions of data possible */
  PetscBool    user_alloc;        /* true if the user provided the dense data */
  PetscBool    unplaced_user_alloc; 
  Mat          ptapwork;          /* workspace (SeqDense matrix) for PtAP */

  Mat_MatTransMatMult *atb;       /* used by MatTransposeMatMult_SeqAIJ_SeqDense */
} Mat_SeqDense;

#endif
