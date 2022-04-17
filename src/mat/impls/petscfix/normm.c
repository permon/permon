
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
      PetscCall(VecDuplicate(Na->right,&Na->rightwork));
    }
    PetscCall(VecPointwiseMult(Na->rightwork,Na->right,in));
    in   = Na->rightwork;
  }
  PetscCall(MatMult(Na->A,in,Na->w));
  PetscCall(VecScale(Na->w,Na->scale));
  if (Na->left) {
    if (v2 == v3) {
      if (!Na->leftwork) {
        PetscCall(VecDuplicate(Na->left,&Na->leftwork));
      }
      PetscCall(VecCopy(v2,Na->leftwork));
      PetscCall(MatMultTranspose(Na->A,Na->w,v3));
      PetscCall(VecPointwiseMult(v3,Na->left,v3));
      PetscCall(VecAXPY(v3,1.0,Na->leftwork));
    } else {
      PetscCall(MatMultTranspose(Na->A,Na->w,v3));
      PetscCall(VecPointwiseMult(v3,Na->left,v3));
      PetscCall(VecAXPY(v3,1.0,v2));
    }
  } else {
    PetscCall(MatMultTransposeAdd(Na->A,Na->w,v2,v3));
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
      PetscCall(VecDuplicate(Na->left,&Na->leftwork));
    }
    PetscCall(VecPointwiseMult(Na->leftwork,Na->left,in));
    in   = Na->leftwork;
  }
  PetscCall(MatMult(Na->A,in,Na->w));
  PetscCall(VecScale(Na->w,Na->scale));
  if (Na->right) {
    if (v2 == v3) {
      if (!Na->rightwork) {
        PetscCall(VecDuplicate(Na->right,&Na->rightwork));
      }
      PetscCall(VecCopy(v2,Na->rightwork));
      PetscCall(MatMultTranspose(Na->A,Na->w,v3));
      PetscCall(VecPointwiseMult(v3,Na->right,v3));
      PetscCall(VecAXPY(v3,1.0,Na->rightwork));
    } else {
      PetscCall(MatMultTranspose(Na->A,Na->w,v3));
      PetscCall(VecPointwiseMult(v3,Na->right,v3));
      PetscCall(VecAXPY(v3,1.0,v2));
    }
  } else {
    PetscCall(MatMultTransposeAdd(Na->A,Na->w,v2,v3));
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreateNormal_permonfix"
PetscErrorCode MatCreateNormal_permonfix(Mat A,Mat *N)
{
  PetscFunctionBegin;
  PetscCall(MatCreateNormal(A,N));
  (*N)->ops->multadd = MatMultAdd_Normal_permonfix;
  (*N)->ops->multtransposeadd = MatMultTransposeAdd_Normal_permonfix;
  PetscCall(MatSetUp(*N));
  PetscFunctionReturn(0);
}

#define MatCreateNormal(A,N) MatCreateNormal_permonfix(A,N)
