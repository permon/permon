
/* Test nullspace detection */
#include <permonmat.h>

int main(int argc,char **args)
{
  Mat            A,Ainv;
  PetscInt       i,n = 5,row[2],col[2],rstart,rend;
  PetscReal      val[] = {1.0, -1.0, -1.0, 1.0};

  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&args,(char *)0,(char *)0);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));

  /* Construct rank-defficient matrix (without Dirichlet BC) */
  if (rend==n) rend--;
  for (i=rstart; i<rend; i++) {
    row[0] = i; row[1] = i+1;
    col[0] = i; col[1] = i+1;
    CHKERRQ(MatSetValues(A,2,row,2,col,val,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateInv(A,MAT_INV_MONOLITHIC,&Ainv));
  CHKERRQ(MatInvComputeNullSpace(Ainv));
  /* nullspace is checked automatically in MatInvComputeNullSpace() in debug mode */
  {
    Mat R;
    CHKERRQ(MatInvGetNullSpace(Ainv,&R));
    CHKERRQ(MatCheckNullSpace(A,R,PETSC_SMALL));
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&Ainv));
  ierr = PermonFinalize();
  return ierr;
}


/*TEST
  build:
    require: mumps
  test:
    nsize: {{1 2 4}}
TEST*/
