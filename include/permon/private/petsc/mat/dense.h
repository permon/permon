#pragma once

#include "aij.h" /* Mat_MatTransMatMult is defined here */

/*
  MATSEQDENSE format - conventional dense Fortran storage (by columns)
*/

typedef struct {
  PetscScalar  *v;             /* matrix elements */
  PetscScalar  *unplacedarray; /* if one called MatDensePlaceArray(), this is where it stashed the original */
  PetscBool     roworiented;   /* if true, row oriented input (default) */
  PetscInt      pad;           /* padding */
  PetscBLASInt *pivots;        /* pivots in LU factorization */
  PetscBLASInt  lfwork;        /* length of work array in factorization */
  PetscScalar  *fwork;         /* work array in factorization */
  PetscScalar  *tau;           /* scalar factors of QR factorization */
  Vec           qrrhs;         /* RHS for solving with QR (solution vector can't hold copy of RHS) */
  PetscBLASInt  lda;           /* Lapack leading dimension of data */
  PetscBLASInt  rank;          /* numerical rank (of a QR factorized matrix) */
  PetscBool     user_alloc;    /* true if the user provided the dense data */
  PetscBool     unplaced_user_alloc;

  /* Support for MatDenseGetColumnVec and MatDenseGetSubMatrix */
  Mat                cmat;     /* matrix representation of a given subset of columns */
  Vec                cvec;     /* vector representation of a given column */
  const PetscScalar *ptrinuse; /* holds array to be restored (just a placeholder) */
  PetscInt           vecinuse; /* if cvec is in use (col = vecinuse-1) */
  PetscInt           matinuse; /* if cmat is in use (cbegin = matinuse-1) */
} Mat_SeqDense;
