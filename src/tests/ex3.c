/* Test nullspace detection */
#include <permonmat.h>

int main(int argc, char **args)
{
  Mat       A, Ainv;
  PetscInt  i, n = 5, row[2], col[2], rstart, rend;
  PetscReal val[] = {1.0, -1.0, -1.0, 1.0};

  PetscCall(PermonInitialize(&argc, &args, (char *)0, (char *)0));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

  /* Construct rank-defficient matrix (without Dirichlet BC) */
  if (rend == n) rend--;
  for (i = rstart; i < rend; i++) {
    row[0] = i;
    row[1] = i + 1;
    col[0] = i;
    col[1] = i + 1;
    PetscCall(MatSetValues(A, 2, row, 2, col, val, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateInv(A, MAT_INV_MONOLITHIC, &Ainv));
  PetscCall(MatInvComputeNullSpace(Ainv));
  /* nullspace is checked automatically in MatInvComputeNullSpace() in debug mode */
  {
    Mat R;
    PetscCall(MatInvGetNullSpace(Ainv, &R));
    PetscCall(MatCheckNullSpace(A, R, PETSC_SMALL));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Ainv));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  build:
    require: mumps
  test:
    nsize: {{1 2 4}}
TEST*/
