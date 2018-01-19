#if !defined(__SHELL_H)
#define __SHELL_H

#include <petsc/private/matimpl.h>

struct _MatShellOps {
  /*   0 */
  PetscErrorCode (*mult)(Mat,Vec,Vec);
  /*   5 */
  PetscErrorCode (*multtranspose)(Mat,Vec,Vec);
  /*  10 */
  /*  15 */
  PetscErrorCode (*getdiagonal)(Mat,Vec);
  /*  20 */
  /*  24 */
  /*  29 */
  /*  34 */
  /*  39 */
  PetscErrorCode (*copy)(Mat,Mat,MatStructure);
  /*  44 */
  PetscErrorCode (*diagonalset)(Mat,Vec,InsertMode);
  /*  49 */
  /*  54 */
  /*  59 */
  PetscErrorCode (*destroy)(Mat);
  /*  64 */
  /*  69 */
  /*  74 */
  /*  79 */
  /*  84 */
  /*  89 */
  /*  94 */
  /*  99 */
  /* 104 */
  /* 109 */
  /* 114 */
  /* 119 */
  /* 124 */
  /* 129 */
  /* 134 */
  /* 139 */
  /* 144 */
};

typedef struct {
  struct _MatShellOps ops[1];

  PetscScalar vscale,vshift;
  Vec         dshift;
  Vec         left,right;
  Vec         dshift_owned,left_owned,right_owned;
  Vec         left_work,right_work;
  Vec         left_add_work,right_add_work;
  PetscBool   usingscaled;
  void        *ctx;
} Mat_Shell;

#endif

