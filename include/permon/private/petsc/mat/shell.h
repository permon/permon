#if !defined(__SHELL_H)
#define __SHELL_H

#include <petsc/private/matimpl.h>

struct _MatShellOps {
  /*  3 */ PetscErrorCode (*mult)(Mat,Vec,Vec);
  /*  5 */ PetscErrorCode (*multtranspose)(Mat,Vec,Vec);
  /* 17 */ PetscErrorCode (*getdiagonal)(Mat,Vec);
  /* 43 */ PetscErrorCode (*copy)(Mat,Mat,MatStructure);
  /* 60 */ PetscErrorCode (*destroy)(Mat);
};

typedef struct {
  struct _MatShellOps ops[1];

  PetscScalar vscale,vshift;
  Vec         dshift;
  Vec         left,right;
  Vec         left_work,right_work;
  Vec         left_add_work,right_add_work;
  Mat         axpy;
  PetscScalar axpy_vscale;
  PetscBool   managescalingshifts;                   /* The user will manage the scaling and shifts for the MATSHELL, not the default */
  void        *ctx;
} Mat_Shell;

#endif

