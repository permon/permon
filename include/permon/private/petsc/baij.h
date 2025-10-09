/* This file is a stripped-down version of
   src/mat/impls/baij/seq/baij.h
   found in the PETSc source code.

   The original PETSc code is licensed under the BSD 2-Clause "Simplified" License.
   See the LICENSE file in this directory for full terms:
   ./LICENSE or https://gitlab.com/petsc/petsc/-/blob/main/LICENSE
*/

#pragma once
#include <petsc/private/matimpl.h>
#include "aij.h"
#include <petsc/private/hashmapijv.h>
#include <petsc/private/hashsetij.h>

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
  MatScalar *idiag;      /* inverse of block diagonal  */ \
  PetscBool  idiagvalid; /* if above has correct/current values */ \
  /* MatSetValues() via hash related fields */ \
  PetscHMapIJV   ht; \
  PetscInt      *dnz; \
  PetscHSetIJ    bht; \
  PetscInt      *bdnz; \
  struct _MatOps cops

typedef struct {
  SEQAIJHEADER(MatScalar);
  SEQBAIJHEADER;
} Mat_SeqBAIJ;
