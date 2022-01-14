
/* Test VecInvalidate */
#include <permonvec.h>

int main(int argc,char **args)
{
  Vec            v;
  PetscInt       n = 4;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&args,(char *)0,(char *)0);if (ierr) return ierr;

  ierr = VecCreate(PETSC_COMM_WORLD,&v);CHKERRQ(ierr);
  ierr = VecSetSizes(v,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  ierr = VecSet(v,1.);CHKERRQ(ierr);

  ierr = VecIsInvalidated(v,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec is invalid");
  ierr = VecInvalidate(v);CHKERRQ(ierr);
  ierr = VecIsInvalidated(v,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec is valid");
  ierr = VecView(v,NULL);CHKERRQ(ierr);
  ierr = VecSet(v,1.);CHKERRQ(ierr);
  ierr = VecIsInvalidated(v,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec is invalid");

  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = PermonFinalize();
  return ierr;
}


/*TEST
  test:
TEST*/

