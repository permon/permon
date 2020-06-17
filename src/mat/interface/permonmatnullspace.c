
#include <permon/private/permonmatimpl.h>

PetscErrorCode MatSetNullSpaceMat(Mat mat, Mat R)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
#if defined(PETSC_USE_DEBUG)
  if (R) {
    PetscBool flg;

    PetscValidHeaderSpecific(R,MAT_CLASSID,2);
    TRY( MatCheckNullSpaceMat(mat, R, PETSC_DEFAULT, &flg) );
     if (!flg) SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONG, "R is unlikely to be nullspace of K. See -info output for details.");
  }
#endif
  ierr = PetscObjectCompose((PetscObject)mat, "NullSpace_Mat", (PetscObject)R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetNullSpaceMat(Mat mat, Mat *R)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(R,2);
  ierr = PetscObjectQuery((PetscObject)mat, "NullSpace_Mat", (PetscObject*)R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopyNullSpaceMat(Mat mat1, Mat mat2)
{
  Mat R;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat1,MAT_CLASSID,1);
  PetscValidHeaderSpecific(mat2,MAT_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)mat1, "NullSpace_Mat", (PetscObject*)&R);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)mat2, "NullSpace_Mat", (PetscObject)R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCheckNullSpaceMat(Mat K,Mat R,PetscReal tol,PetscBool *valid)
{
  Vec Rx,x,y;
  PetscReal normy;

  PetscFunctionBegin;
  if (tol == PETSC_DECIDE || tol == PETSC_DEFAULT) tol = 1e2*PETSC_SQRT_MACHINE_EPSILON;
  PetscValidHeaderSpecific(K,MAT_CLASSID,1);
  PetscValidHeaderSpecific(R,MAT_CLASSID,2);
  PetscValidLogicalCollectiveReal(K,tol,3);
  if (K->cmap->N != R->rmap->N) SETERRQ2(PetscObjectComm((PetscObject)K),PETSC_ERR_ARG_SIZ,"non-conforming global size of K and R: %D != %D",K->cmap->N,R->rmap->N);
  if (K->cmap->n != R->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"non-conforming local size of K and R: %D != %D",K->cmap->n,R->rmap->n);

  TRY( MatCreateVecs(K,NULL,&y) );
  TRY( MatCreateVecs(R,&x,&Rx) );
  TRY( VecSet(x,1.0) );
  TRY( MatMult(R,x,Rx) );
  TRY( MatMult(K,Rx,y) );
  TRY( VecNorm(y,NORM_2,&normy) );
  TRY( PetscInfo1(K,"||K*R*x|| = %.3e   [x is vector of all ones]\n",normy) );
  *valid = (normy > tol) ? PETSC_FALSE : PETSC_TRUE;
  if (!*valid) PetscInfo2(K, "R is unlikely to be nullspace of K, ||K*R*e|| = %.3e is greater than tolerance %.3e", normy, tol);
  TRY( VecDestroy(&Rx) );
  TRY( VecDestroy(&x) );
  TRY( VecDestroy(&y) );
  PetscFunctionReturn(0);
}
