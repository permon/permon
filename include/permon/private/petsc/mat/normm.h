#pragma once

typedef struct {
  Mat         A;
  Mat         D; /* local submatrix for diagonal part */
  Vec         w, left, right, leftwork, rightwork;
  PetscScalar scale;
} Mat_Normal;
