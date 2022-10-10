
#ifndef __NORMM_H
#define	__NORMM_H

typedef struct {
  Mat         A;
  Mat         D; /* local submatrix for diagonal part */
  Vec         w, left, right, leftwork, rightwork;
  PetscScalar scale;
} Mat_Normal;

#endif
