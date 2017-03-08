
#include <private/fllopmatimpl.h>

#if defined(MatCreateNormal)
#undef MatCreateNormal
#endif

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Normal_permonfix"
PetscErrorCode MatMultAdd_Normal_permonfix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->right) {
    if (!Na->rightwork) {
      ierr = VecDuplicate(Na->right,&Na->rightwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->rightwork,Na->right,in);CHKERRQ(ierr);
    in   = Na->rightwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = VecScale(Na->w,Na->scale);CHKERRQ(ierr);
  if (Na->left) {
    if (v2 == v3) {
      if (!Na->leftwork) {
        ierr = VecDuplicate(Na->left,&Na->leftwork);CHKERRQ(ierr);
      }
      ierr = VecCopy(v2,Na->leftwork);CHKERRQ(ierr);
      ierr = MatMultTranspose(Na->A,Na->w,v3);CHKERRQ(ierr);
      ierr = VecPointwiseMult(v3,Na->left,v3);CHKERRQ(ierr);
      ierr = VecAXPY(v3,1.0,Na->leftwork);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(Na->A,Na->w,v3);CHKERRQ(ierr);
      ierr = VecPointwiseMult(v3,Na->left,v3);CHKERRQ(ierr);
      ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
    }
  } else {
    ierr = MatMultTransposeAdd(Na->A,Na->w,v2,v3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Normal_permonfix"
PetscErrorCode MatMultTransposeAdd_Normal_permonfix(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->left) {
    if (!Na->leftwork) {
      ierr = VecDuplicate(Na->left,&Na->leftwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->leftwork,Na->left,in);CHKERRQ(ierr);
    in   = Na->leftwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = VecScale(Na->w,Na->scale);CHKERRQ(ierr);
  if (Na->right) {
    if (v2 == v3) {
      if (!Na->rightwork) {
        ierr = VecDuplicate(Na->right,&Na->rightwork);CHKERRQ(ierr);
      }
      ierr = VecCopy(v2,Na->rightwork);CHKERRQ(ierr);
      ierr = MatMultTranspose(Na->A,Na->w,v3);CHKERRQ(ierr);
      ierr = VecPointwiseMult(v3,Na->right,v3);CHKERRQ(ierr);
      ierr = VecAXPY(v3,1.0,Na->rightwork);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(Na->A,Na->w,v3);CHKERRQ(ierr);
      ierr = VecPointwiseMult(v3,Na->right,v3);CHKERRQ(ierr);
      ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
    }
  } else {
    ierr = MatMultTransposeAdd(Na->A,Na->w,v2,v3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreateNormal_permonfix"
PetscErrorCode MatCreateNormal_permonfix(Mat A,Mat *N)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatCreateNormal(A,N);CHKERRQ(ierr);
  (*N)->ops->multadd = MatMultAdd_Normal_permonfix;
  (*N)->ops->multtransposeadd = MatMultTransposeAdd_Normal_permonfix;
  ierr = MatSetUp(*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define MatCreateNormal(A,N) MatCreateNormal_permonfix(A,N)
