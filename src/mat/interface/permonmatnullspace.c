
#include <permon/private/permonmatimpl.h>

PetscErrorCode MatSetNullSpaceMat(Mat mat, Mat R)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
#if defined(PETSC_USE_DEBUG)
  if (R) {
    PetscValidHeaderSpecific(R,MAT_CLASSID,2);
    TRY( MatCheckNullSpaceMat(mat, R, PETSC_DEFAULT) );
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

PetscErrorCode MatCheckNullSpaceMat(Mat K,Mat R,PetscReal tol)
{
  Vec d,x,y;
  PetscReal normd,normy;

  PetscFunctionBegin;
  if (tol == PETSC_DECIDE || tol == PETSC_DEFAULT) tol = PETSC_SMALL;
  PetscValidHeaderSpecific(K,MAT_CLASSID,1);
  PetscValidHeaderSpecific(R,MAT_CLASSID,2);
  PetscValidLogicalCollectiveReal(K,tol,3);
  if (K->cmap->N != R->rmap->N) SETERRQ2(PetscObjectComm((PetscObject)K),PETSC_ERR_ARG_SIZ,"non-conforming global size of K and R: %D != %D",K->cmap->N,R->rmap->N);
  if (K->cmap->n != R->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"non-conforming local size of K and R: %D != %D",K->cmap->n,R->rmap->n);

  TRY( MatCreateVecs(K,&d,&y) );
  TRY( MatCreateVecs(R,&x,NULL) );
  TRY( MatGetDiagonal(K,d) );
  TRY( VecNorm(d,NORM_2,&normd) );
  TRY( VecSetRandom(x,NULL) );
  TRY( MatMult(R,x,d) );
  TRY( MatMult(K,d,y) );
  TRY( VecNorm(y,NORM_2,&normy) );
  TRY( PetscInfo3(fllop,"||K*R*x|| = %.3e   ||diag(K)|| = %.3e    ||K*R*x|| / ||diag(K)|| = %.3e\n",normy,normd,normy/normd) );
  FLLOP_ASSERT1(normy / normd < tol, "||K*R*x|| / ||diag(K)|| < %.1e", tol);
  TRY( VecDestroy(&d) );
  TRY( VecDestroy(&x) );
  TRY( VecDestroy(&y) );
  PetscFunctionReturn(0);
}
