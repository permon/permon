static char help[] = "Solves a tridiagonal system with lower bound specified as a linear inequality constraint. Uses dualization.\n\
Solves finite difference discretization of:\n\
-u''(x) = -15,  x in [0,1]\n\
u(0) = u(1) = 0\n\
s.t. u(x) >= sin(4*pi*x -pi/6)/2 -2\n\
Based on ex1.\n\
Input parameters include:\n\
  -n <mesh_n>   : number of mesh points\n\
  -spd          : mark Hessian SPD\n\
  -empty_nullsp : pass empty nullspace to QP\n";

/*
* Include "permonqps.h" so that we can use QPS solvers.  Note that this file
* automatically includes:
*   petscsys.h   - base PERMON routines
*   permonvec.h  - Vectors
*   permonmat.h  - Matrices
*   permonqppf.h - Projection Factory
*   permonqpc.h  - Quadratic Programming Constraints
*   permonqp.h   - Quadratic Programming objects and transformations
*   petsctao.h   - Toolkit for Advanced Optimization solvers
*/
#include <permonqps.h>

/* Lower bound (obstacle) function */
PetscReal fobst(PetscInt i, PetscInt n)
{
  PetscReal h = 1. / (n - 1);
  return PetscSinReal(4 * PETSC_PI * i * h - PETSC_PI / 6.) / 2 - 2;
}

int main(int argc, char **args)
{
  Vec       b, c, x;
  Mat       A, B, R = NULL;
  QP        qp;
  QPS       qps;
  PetscInt  i, n = 10, col[3], rstart, rend;
  PetscReal h, value[3];
  PetscBool converged, spd = PETSC_FALSE, empty_nullsp = PETSC_FALSE;

  PetscCall(PermonInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_nullsp", &empty_nullsp, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-spd", &spd, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup matrices and vectors
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  h = 1. / (n - 1);
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &c));

  if (!rstart) {
    rstart   = 1;
    i        = 0;
    value[0] = 1.0;
    PetscCall(MatSetValues(A, 1, &i, 1, &i, value, INSERT_VALUES));
    PetscCall(VecSetValue(b, i, 0, INSERT_VALUES));
  }
  if (rend == n) {
    rend     = n - 1;
    i        = n - 1;
    value[0] = 1.0;
    PetscCall(MatSetValues(A, 1, &i, 1, &i, value, INSERT_VALUES));
    PetscCall(VecSetValue(b, i, 0, INSERT_VALUES));
  }
  value[0] = -1.0;
  value[1] = 2.0;
  value[2] = -1.0;
  for (i = rstart; i < rend; i++) {
    col[0] = i - 1;
    col[1] = i;
    col[2] = i + 1;
    if (i == 1) col[0] = -1;     /* ignore the first value in the second row (Dirichlet BC) */
    if (i == n - 2) col[2] = -1; /* ignore the third value in the second to last row (Dirichlet BC) */
    PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
    PetscCall(VecSetValue(b, i, -15 * h * h * 2, INSERT_VALUES));
    PetscCall(VecSetValue(c, i, fobst(i, n), INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(VecAssemblyBegin(c));
  PetscCall(VecAssemblyEnd(c));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, 1, NULL, 0, NULL, &B));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(B, 1.0));

  if (empty_nullsp) {
    /* NOT RECOMMENDED: Empty null space matrix for dualization */
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, 0, 0, NULL, 0, NULL, &R));
    PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));
  }
  if (spd) {
    /* RECOMMENDED: Mark Hessian SPD - will skip nullspace computation in QPTDualize */
    PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QP: argmin 1/2 x'Ax -x'b s.t. c <= I*x
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(QPCreate(PETSC_COMM_WORLD, &qp));
  /* Set matrix representing QP operator */
  PetscCall(QPSetOperator(qp, A));

  /* Set right hand side */
  PetscCall(QPSetRhs(qp, b));
  /* Set initial guess.
  * THIS VECTOR WILL ALSO HOLD THE SOLUTION OF QP */
  PetscCall(QPSetInitialVector(qp, x));
  /* Set inequality constraint c <= Bx in the form -c >= -Bx*/
  PetscCall(VecScale(c, -1.0));
  PetscCall(MatScale(B, -1.0));
  PetscCall(QPSetIneq(qp, B, c));
  if (empty_nullsp) { PetscCall(QPSetOperatorNullSpace(qp, R)); }
  /* Dualize QP */
  PetscCall(QPTDualize(qp, MAT_INV_MONOLITHIC, MAT_REG_NONE));
  /* Set runtime options, e.g
  *   -qp_chain_view_kkt */
  PetscCall(QPSetFromOptions(qp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QPS, i.e. QP Solver
  *   Note the use of PetscObjectComm() to get the same comm as in qp object.
  *   We could specify the comm explicitly, in this case PETSC_COMM_WORLD.
  *   Also, all PERMON objects are PETSc objects as well :)
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(QPSCreate(PetscObjectComm((PetscObject)qp), &qps));
  /* Set QP to solve */
  PetscCall(QPSSetQP(qps, qp));
  /* Set runtime options for solver, e.g,
  *   -qps_type <type> -qps_rtol <relative tolerance> -qps_view_convergence */
  PetscCall(QPSSetFromOptions(qps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Solve QP
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(QPSSolve(qps));

  /* Check that QPS converged */
  PetscCall(QPIsSolved(qp, &converged));
  if (!converged) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "QPS did not converge!\n"));

  PetscCall(QPSDestroy(&qps));
  PetscCall(QPDestroy(&qp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&c));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  testset:
    suffix: 1
    requires: mumps
    filter: grep -e CONVERGED -e number -e "r ="
    args: -n 100 -qps_view_convergence -qp_chain_view_kkt -spd {{0 1}}
    test:
      nsize: 2
  testset:
    suffix: nullspace
    requires: mumps
    filter: grep -e CONVERGED -e number -e "r ="
    args: -n 100 -qps_view_convergence -qp_chain_view_kkt -empty_nullsp
    test:
    test:
      nsize: 3
TEST*/
