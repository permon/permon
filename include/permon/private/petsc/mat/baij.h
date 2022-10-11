
#ifndef __BAIJ_H
#define __BAIJ_H
#include <petsc/private/matimpl.h>
#include "aij.h"

/*
  MATSEQBAIJ format - Block compressed row storage. The i[] and j[]
  arrays start at 0.
*/

/* This header is shared by the SeqSBAIJ matrix */
#define SEQBAIJHEADER \
  PetscInt     bs2;       /*  square of block size */ \
  PetscInt     mbs, nbs;  /* rows/bs, columns/bs */ \
  PetscScalar *mult_work; /* work array for matrix vector product*/ \
  PetscScalar *sor_workt; /* work array for SOR */ \
  PetscScalar *sor_work;  /* work array for SOR */ \
  MatScalar   *saved_values; \
\
  Mat sbaijMat; /* mat in sbaij format */ \
\
  MatScalar *idiag;     /* inverse of block diagonal  */ \
  PetscBool  idiagvalid /* if above has correct/current values */

typedef struct {
  SEQAIJHEADER(MatScalar);
  SEQBAIJHEADER;
} Mat_SeqBAIJ;

#endif
