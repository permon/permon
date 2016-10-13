#if !defined(__SHELL_H)
#define __SHELL_H

#include <petsc-private/matimpl.h>

typedef struct {
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*mult)(Mat,Vec,Vec);
  PetscErrorCode (*multtranspose)(Mat,Vec,Vec);
  PetscErrorCode (*getdiagonal)(Mat,Vec);

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

