
#include <private/fllopmatimpl.h>
#include <private/petscimpl.h>

#undef __FUNCT__  
#define __FUNCT__ "MatTransposeGetMat"
PetscErrorCode MatTransposeGetMat(Mat N,Mat *A)
{
  Mat_Transpose  *Na;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(N,MAT_CLASSID,1);
  PetscValidPointer(A,2);
  TRY( PetscObjectTypeCompare((PetscObject) N, MATTRANSPOSEMAT, &match) );
  if (!match) FLLOP_SETERRQ(((PetscObject)N)->comm, PETSC_ERR_SUP,"Requires MATTRANSPOSEMAT matrix as input");
      
  Na = (Mat_Transpose*)N->data;
  *A = Na->A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatIsImplicitTranspose"
PetscErrorCode MatIsImplicitTranspose(Mat A,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(flg,2);
  TRY( PetscObjectTypeCompare((PetscObject)A,MATTRANSPOSEMAT,flg) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatTranspose_Transpose"
static PetscErrorCode FllopMatTranspose_Transpose(Mat A,MatTransposeType type,Mat *At_out)
{
  Mat At,Ate,Ae;

  PetscFunctionBegin;
  switch (type) {
    default: /* MAT_TRANSPOSE_CHEAPEST */
    case MAT_TRANSPOSE_EXPLICIT:
      TRY( MatTransposeGetMat(A,&At) );
      TRY( PetscObjectReference((PetscObject)At) );
      break;
    case MAT_TRANSPOSE_IMPLICIT:
      TRY( MatTransposeGetMat(A,&Ate) );
      TRY( MatTranspose(Ate,MAT_INITIAL_MATRIX,&Ae) );
      TRY( MatCreateTransposePermon(Ae,&At) );
      TRY( MatDestroy(&Ae) );
  }
  *At_out = At;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatTranspose_Default"
static PetscErrorCode FllopMatTranspose_Default(Mat A,MatTransposeType type,Mat *At_out)
{
  Mat At;

  PetscFunctionBegin;
  switch (type) {
    case MAT_TRANSPOSE_EXPLICIT:
      TRY( MatTranspose(A,MAT_INITIAL_MATRIX,&At) );
      break;
    default: /* MAT_TRANSPOSE_CHEAPEST */
    case MAT_TRANSPOSE_IMPLICIT:
      TRY( MatCreateTransposePermon(A,&At) );
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
  TRY( MatDiagonalScale(data->A,r,l) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_TransposePermon"
PetscErrorCode MatDuplicate_TransposePermon(Mat mat,MatDuplicateOption op,Mat *M)
{
  Mat A = ((Mat_Transpose*)mat->data)->A;
  Mat A1;

  PetscFunctionBegin;
  TRY( MatDuplicate(A,op,&A1) );
  TRY( FllopPetscObjectInheritName((PetscObject)A1,(PetscObject)A,NULL) );
  TRY( MatCreateTransposePermon(A1,M) );
  TRY( MatDestroy(&A1) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateTransposePermon"
PetscErrorCode MatCreateTransposePermon(Mat A,Mat *At)
{
  PetscFunctionBegin;
  TRY( MatCreateTranspose(A,At) );
  (*At)->ops->diagonalscale = MatDiagonalScale_TransposePermon;
  (*At)->ops->duplicate     = MatDuplicate_TransposePermon;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatTranspose"
PetscErrorCode FllopMatTranspose(Mat A,MatTransposeType type,Mat *At_out)
{
  PetscErrorCode (*f)(Mat,MatTransposeType,Mat*);
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(A,type,2);
  PetscValidPointer(At_out,3);

  /* try to find a type-specific implementation */
  TRY( PetscObjectQueryFunction((PetscObject)A,"FllopMatTranspose_C",&f) );

  /* work-around for MATTRANSPOSE to avoid need of a new constructor */
  if (!f) {
    TRY( PetscObjectTypeCompare((PetscObject)A,MATTRANSPOSEMAT,&flg) );
    if (flg) f = FllopMatTranspose_Transpose;
  }
  
  /* if no type-specific implementation is found, use the default one */
  if (!f) f = FllopMatTranspose_Default;
  
  /* call the implementation */
  TRY( (*f)(A,type,At_out) );

  if (!((PetscObject)(*At_out))->name) {
    TRY( FllopPetscObjectInheritName((PetscObject)*At_out,(PetscObject)A,"_T") );
  }
  PetscFunctionReturn(0);
}
