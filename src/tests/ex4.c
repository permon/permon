
/* Test VecInvalidate */
#include <permonvec.h>

int main(int argc,char **args)
{
  Vec            v;
  PetscInt       n = 4;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&args,(char *)0,(char *)0);if (ierr) return ierr;

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&v));
  CHKERRQ(VecSetSizes(v,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(v));
  CHKERRQ(VecSet(v,1.));

  CHKERRQ(VecIsInvalidated(v,&flg));
  if (flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec is invalid");
  CHKERRQ(VecInvalidate(v));
  CHKERRQ(VecIsInvalidated(v,&flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec is valid");
  CHKERRQ(VecView(v,NULL));
  CHKERRQ(VecSet(v,1.));
  CHKERRQ(VecIsInvalidated(v,&flg));
  if (flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec is invalid");

  CHKERRQ(VecDestroy(&v));
  ierr = PermonFinalize();
  return ierr;
}


/*TEST
  test:
TEST*/
