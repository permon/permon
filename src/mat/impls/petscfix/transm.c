#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatIsImplicitTranspose"
PetscErrorCode MatIsImplicitTranspose(Mat A, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATTRANSPOSEVIRTUAL, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatTranspose_Transpose"
static PetscErrorCode PermonMatTranspose_Transpose(Mat A, MatTransposeType type, Mat *At_out)
{
  Mat At, Ate, Ae;

  PetscFunctionBegin;
  switch (type) {
  default: /* MAT_TRANSPOSE_CHEAPEST */
  case MAT_TRANSPOSE_EXPLICIT:
    PetscCall(MatTransposeGetMat(A, &At));
    PetscCall(PetscObjectReference((PetscObject)At));
    break;
  case MAT_TRANSPOSE_IMPLICIT:
    PetscCall(MatTransposeGetMat(A, &Ate));
    PetscCall(MatTranspose(Ate, MAT_INITIAL_MATRIX, &Ae));
    PetscCall(MatCreateTransposePermon(Ae, &At));
    PetscCall(MatDestroy(&Ae));
  }
  *At_out = At;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatTranspose_Default"
static PetscErrorCode PermonMatTranspose_Default(Mat A, MatTransposeType type, Mat *At_out)
{
  Mat At;

  PetscFunctionBegin;
  switch (type) {
  case MAT_TRANSPOSE_EXPLICIT:
    PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &At));
    break;
  default: /* MAT_TRANSPOSE_CHEAPEST */
  case MAT_TRANSPOSE_IMPLICIT:
    PetscCall(MatCreateTransposePermon(A, &At));
  }
  *At_out = At;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScale_TransposePermon"
PetscErrorCode MatDiagonalScale_TransposePermon(Mat At, Vec l, Vec r)
{
  Mat_Transpose *data = (Mat_Transpose *)At->data;

  PetscFunctionBegin;
  PetscCall(MatDiagonalScale(data->A, r, l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_TransposePermon"
PetscErrorCode MatDuplicate_TransposePermon(Mat mat, MatDuplicateOption op, Mat *M)
{
  Mat A = ((Mat_Transpose *)mat->data)->A;
  Mat A1;

  PetscFunctionBegin;
  PetscCall(MatDuplicate(A, op, &A1));
  PetscCall(PermonPetscObjectInheritName((PetscObject)A1, (PetscObject)A, NULL));
  PetscCall(MatCreateTransposePermon(A1, M));
  PetscCall(MatDestroy(&A1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateTransposePermon"
PetscErrorCode MatCreateTransposePermon(Mat A, Mat *At)
{
  PetscFunctionBegin;
  PetscCall(MatCreateTranspose(A, At));
  (*At)->ops->diagonalscale = MatDiagonalScale_TransposePermon;
  (*At)->ops->duplicate     = MatDuplicate_TransposePermon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatTranspose"
PetscErrorCode PermonMatTranspose(Mat A, MatTransposeType type, Mat *At_out)
{
  PetscErrorCode (*f)(Mat, MatTransposeType, Mat *);
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(A, type, 2);
  PetscAssertPointer(At_out, 3);

  /* try to find a type-specific implementation */
  PetscCall(PetscObjectQueryFunction((PetscObject)A, "PermonMatTranspose_C", &f));

  /* work-around for MATTRANSPOSE to avoid need of a new constructor */
  if (!f) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATTRANSPOSEVIRTUAL, &flg));
    if (flg) f = PermonMatTranspose_Transpose;
  }

  /* if no type-specific implementation is found, use the default one */
  if (!f) f = PermonMatTranspose_Default;

  /* call the implementation */
  PetscCall((*f)(A, type, At_out));

  if (!((PetscObject)(*At_out))->name) { PetscCall(PermonPetscObjectInheritName((PetscObject)*At_out, (PetscObject)A, "_T")); }
  PetscFunctionReturn(PETSC_SUCCESS);
}
