
#include <permon/private/permonmatimpl.h>

#if defined(MatCreateNormal)
#undef MatCreateNormal
#endif

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Normal_permonfix"
PetscErrorCode MatMultAdd_Normal_permonfix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->right) {
    if (!Na->rightwork) {
      CHKERRQ(VecDuplicate(Na->right,&Na->rightwork));
    }
    CHKERRQ(VecPointwiseMult(Na->rightwork,Na->right,in));
    in   = Na->rightwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(VecScale(Na->w,Na->scale));
  if (Na->left) {
    if (v2 == v3) {
      if (!Na->leftwork) {
        CHKERRQ(VecDuplicate(Na->left,&Na->leftwork));
      }
      CHKERRQ(VecCopy(v2,Na->leftwork));
      CHKERRQ(MatMultTranspose(Na->A,Na->w,v3));
      CHKERRQ(VecPointwiseMult(v3,Na->left,v3));
      CHKERRQ(VecAXPY(v3,1.0,Na->leftwork));
    } else {
      CHKERRQ(MatMultTranspose(Na->A,Na->w,v3));
      CHKERRQ(VecPointwiseMult(v3,Na->left,v3));
      CHKERRQ(VecAXPY(v3,1.0,v2));
    }
  } else {
    CHKERRQ(MatMultTransposeAdd(Na->A,Na->w,v2,v3));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Normal_permonfix"
PetscErrorCode MatMultTransposeAdd_Normal_permonfix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->left) {
    if (!Na->leftwork) {
      CHKERRQ(VecDuplicate(Na->left,&Na->leftwork));
    }
    CHKERRQ(VecPointwiseMult(Na->leftwork,Na->left,in));
    in   = Na->leftwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(VecScale(Na->w,Na->scale));
  if (Na->right) {
    if (v2 == v3) {
      if (!Na->rightwork) {
        CHKERRQ(VecDuplicate(Na->right,&Na->rightwork));
      }
      CHKERRQ(VecCopy(v2,Na->rightwork));
      CHKERRQ(MatMultTranspose(Na->A,Na->w,v3));
      CHKERRQ(VecPointwiseMult(v3,Na->right,v3));
      CHKERRQ(VecAXPY(v3,1.0,Na->rightwork));
    } else {
      CHKERRQ(MatMultTranspose(Na->A,Na->w,v3));
      CHKERRQ(VecPointwiseMult(v3,Na->right,v3));
      CHKERRQ(VecAXPY(v3,1.0,v2));
    }
  } else {
    CHKERRQ(MatMultTransposeAdd(Na->A,Na->w,v2,v3));
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreateNormal_permonfix"
PetscErrorCode MatCreateNormal_permonfix(Mat A,Mat *N)
{
  PetscFunctionBegin;
  CHKERRQ(MatCreateNormal(A,N));
  (*N)->ops->multadd = MatMultAdd_Normal_permonfix;
  (*N)->ops->multtransposeadd = MatMultTransposeAdd_Normal_permonfix;
  CHKERRQ(MatSetUp(*N));
  PetscFunctionReturn(0);
}

#define MatCreateNormal(A,N) MatCreateNormal_permonfix(A,N)
