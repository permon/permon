
#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatIsImplicitTranspose"
PetscErrorCode MatIsImplicitTranspose(Mat A,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(flg,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATTRANSPOSEMAT,flg));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatTranspose_Transpose"
static PetscErrorCode PermonMatTranspose_Transpose(Mat A,MatTransposeType type,Mat *At_out)
{
  Mat At,Ate,Ae;

  PetscFunctionBegin;
  switch (type) {
    default: /* MAT_TRANSPOSE_CHEAPEST */
    case MAT_TRANSPOSE_EXPLICIT:
      CHKERRQ(MatTransposeGetMat(A,&At));
      CHKERRQ(PetscObjectReference((PetscObject)At));
      break;
    case MAT_TRANSPOSE_IMPLICIT:
      CHKERRQ(MatTransposeGetMat(A,&Ate));
      CHKERRQ(MatTranspose(Ate,MAT_INITIAL_MATRIX,&Ae));
      CHKERRQ(MatCreateTransposePermon(Ae,&At));
      CHKERRQ(MatDestroy(&Ae));
  }
  *At_out = At;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatTranspose_Default"
static PetscErrorCode PermonMatTranspose_Default(Mat A,MatTransposeType type,Mat *At_out)
{
  Mat At;

  PetscFunctionBegin;
  switch (type) {
    case MAT_TRANSPOSE_EXPLICIT:
      CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&At));
      break;
    default: /* MAT_TRANSPOSE_CHEAPEST */
    case MAT_TRANSPOSE_IMPLICIT:
      CHKERRQ(MatCreateTransposePermon(A,&At));
  }
  *At_out = At;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScale_TransposePermon"
PetscErrorCode MatDiagonalScale_TransposePermon(Mat At,Vec l,Vec r)
{
  Mat_Transpose *data = (Mat_Transpose*)At->data;

  PetscFunctionBegin;
  CHKERRQ(MatDiagonalScale(data->A,r,l));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_TransposePermon"
PetscErrorCode MatDuplicate_TransposePermon(Mat mat,MatDuplicateOption op,Mat *M)
{
  Mat A = ((Mat_Transpose*)mat->data)->A;
  Mat A1;

  PetscFunctionBegin;
  CHKERRQ(MatDuplicate(A,op,&A1));
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)A1,(PetscObject)A,NULL));
  CHKERRQ(MatCreateTransposePermon(A1,M));
  CHKERRQ(MatDestroy(&A1));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateTransposePermon"
PetscErrorCode MatCreateTransposePermon(Mat A,Mat *At)
{
  PetscFunctionBegin;
  CHKERRQ(MatCreateTranspose(A,At));
  (*At)->ops->diagonalscale = MatDiagonalScale_TransposePermon;
  (*At)->ops->duplicate     = MatDuplicate_TransposePermon;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatTranspose"
PetscErrorCode PermonMatTranspose(Mat A,MatTransposeType type,Mat *At_out)
{
  PetscErrorCode (*f)(Mat,MatTransposeType,Mat*);
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(A,type,2);
  PetscValidPointer(At_out,3);

  /* try to find a type-specific implementation */
  CHKERRQ(PetscObjectQueryFunction((PetscObject)A,"PermonMatTranspose_C",&f));

  /* work-around for MATTRANSPOSE to avoid need of a new constructor */
  if (!f) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATTRANSPOSEMAT,&flg));
    if (flg) f = PermonMatTranspose_Transpose;
  }
  
  /* if no type-specific implementation is found, use the default one */
  if (!f) f = PermonMatTranspose_Default;
  
  /* call the implementation */
  CHKERRQ((*f)(A,type,At_out));

  if (!((PetscObject)(*At_out))->name) {
    CHKERRQ(FllopPetscObjectInheritName((PetscObject)*At_out,(PetscObject)A,"_T"));
  }
  PetscFunctionReturn(0);
}
