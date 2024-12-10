
/* Test VecInvalidate */
#include <permonvec.h>

int main(int argc,char **args)
{
  Vec            v;
  PetscInt       n = 4;
  PetscBool      flg;

  PetscCall(PermonInitialize(&argc,&args,(char *)0,(char *)0));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&v));
  PetscCall(VecSetSizes(v,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(v));
  PetscCall(VecSet(v,1.));

  PetscCall(VecIsInvalidated(v,&flg));
  if (flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec is invalid");
  PetscCall(VecInvalidate(v));
  PetscCall(VecIsInvalidated(v,&flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec is valid");
  PetscCall(VecView(v,NULL));
  PetscCall(VecSet(v,1.));
  PetscCall(VecIsInvalidated(v,&flg));
  if (flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec is invalid");

  PetscCall(VecDestroy(&v));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  test:
TEST*/
