
/* Test nullspace detection */
#include <permonmat.h>

int main(int argc,char **args)
{
  Mat            A,Ainv;
  PetscInt       i,n = 5,row[2],col[2],rstart,rend;
  PetscReal      val[] = {1.0, -1.0, -1.0, 1.0};

  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&args,(char *)0,(char *)0);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);

  /* Construct rank-defficient matrix (without Dirichlet BC) */
  if (rend==n) rend--;
  for (i=rstart; i<rend; i++) {
    row[0] = i; row[1] = i+1;
    col[0] = i; col[1] = i+1;
    ierr = MatSetValues(A,2,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateInv(A,MAT_INV_MONOLITHIC,&Ainv);CHKERRQ(ierr);
  ierr = MatInvComputeNullSpace(Ainv);CHKERRQ(ierr);
  /* nullspace is checked automatically in MatInvComputeNullSpace() in debug mode */
  {
    Mat R;
    ierr = MatInvGetNullSpace(Ainv,&R);CHKERRQ(ierr);
    ierr = MatCheckNullSpace(A,R,PETSC_SMALL);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Ainv);CHKERRQ(ierr);
  ierr = PermonFinalize();
  return ierr;
}


/*TEST
  build:
    require: mumps
  test:
    nsize: {{1 2 4}}
TEST*/

