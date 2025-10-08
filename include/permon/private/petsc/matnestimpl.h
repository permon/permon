/* This file is a stripped-down version of
   src/mat/impls/nest/matnestimpl.h
   found in the PETSc source code.

   The original PETSc code is licensed under the BSD 2-Clause "Simplified" License.
   See the LICENSE file in this directory for full terms:
   ./LICENSE or https://gitlab.com/petsc/petsc/-/blob/main/LICENSE
*/

#pragma once

#include <petsc/private/matimpl.h>

struct MatNestISPair {
  IS *row, *col;
};

typedef struct {
  PetscInt             nr, nc; /* nr x nc blocks */
  Mat                **m;
  struct MatNestISPair isglobal;
  struct MatNestISPair islocal;
  Vec                 *left, *right;
  PetscInt            *row_len, *col_len;
  PetscObjectState    *nnzstate;
  PetscBool            splitassembly;
} Mat_Nest;
