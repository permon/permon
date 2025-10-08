/* This file is a stripped-down version of
   src/mat/impls/sbaij/seq/sbaij.h
   found in the PETSc source code.

   The original PETSc code is licensed under the BSD 2-Clause "Simplified" License.
   See the LICENSE file in this directory for full terms:
   ./LICENSE or https://gitlab.com/petsc/petsc/-/blob/main/LICENSE
*/

#pragma once
#include <petsc/private/matimpl.h>
#include "baij.h"

/*
  MATSEQSBAIJ format - Block compressed row storage. The i[] and j[]
  arrays start at 0.
*/

typedef struct {
  SEQAIJHEADER(MatScalar);
  SEQBAIJHEADER;
  PetscInt        *inew;               /* pointer to beginning of each row of reordered matrix */
  PetscInt        *jnew;               /* column values: jnew + i[k] is start of row k */
  MatScalar       *anew;               /* nonzero diagonal and superdiagonal elements of reordered matrix */
  PetscScalar     *solves_work;        /* work space used in MatSolves */
  PetscInt         solves_work_n;      /* size of solves_work */
  PetscInt        *a2anew;             /* map used for symm permutation */
  PetscBool        permute;            /* if true, a non-trivial permutation is used for factorization */
  PetscBool        ignore_ltriangular; /* if true, ignore the lower triangular values inserted by users */
  PetscBool        getrow_utriangular; /* if true, MatGetRow_SeqSBAIJ() is enabled to get the upper part of the row */
  Mat_SeqAIJ_Inode inode;
  unsigned short  *jshort;
  PetscBool        free_jshort;
} Mat_SeqSBAIJ;
