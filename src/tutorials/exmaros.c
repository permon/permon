#include "petscsys.h"
static char help[] = "Solves a tridiagonal system with lower bound.\n\
Solves finite difference discretization of:\n\
-u''(x) = -15,  x in [0,1]\n\
u(0) = u(1) = 0\n\
s.t. u(x) >= sin(4*pi*x -pi/6)/2 -2\n\
Input parameters include:\n\
  -n <mesh_n> : number of mesh points\n\
  -sol        : view solution vector\n\
  -draw_pause : number of seconds to pause, -1 implies until user input, see PetscDrawSetPause()\n";

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
#include <petscdraw.h>

/* Draw vector */

int main(int argc, char **args)
{
  Vec         b, x, c, lb, ub;
  Mat         A, B;
  QP          qp;
  QPS         qps;
  PetscBool   flg, converged, bounds = PETSC_FALSE;
  PetscViewer fd;                       /* viewer */
  char        file[PETSC_MAX_PATH_LEN]; /* input file name */
  char        name[PETSC_MAX_PATH_LEN]; /* input file name */

  PetscCall(PermonInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-name", name, sizeof(name), &flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate the problem name with the -name option");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-bounds", &bounds, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup matrices and vectors
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscSNPrintf(file,PETSC_MAX_PATH_LEN,"MM/%sA.bin",name));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  //PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(PetscSNPrintf(file,PETSC_MAX_PATH_LEN,"MM/%sBeq.bin",name));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  //PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatLoad(B, fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(PetscSNPrintf(file,PETSC_MAX_PATH_LEN,"MM/%sb.bin",name));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
  PetscCall(VecLoad(b, fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(PetscSNPrintf(file,PETSC_MAX_PATH_LEN,"MM/%sc.bin",name));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &c));
  PetscCall(VecLoad(c, fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(PetscSNPrintf(file,PETSC_MAX_PATH_LEN,"MM/%slb.bin",name));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &lb));
  PetscCall(VecLoad(lb, fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(PetscSNPrintf(file,PETSC_MAX_PATH_LEN,"MM/%sub.bin",name));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ub));
  PetscCall(VecLoad(ub, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QP: argmin 1/2 x'Ax -x'b s.t. c <= x
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(QPCreate(PETSC_COMM_WORLD, &qp));
  /* Set matrix representing QP operator */
  PetscCall(QPSetOperator(qp, A));
  /* Set right hand side */
  PetscCall(QPSetRhs(qp, b));
  /* Set initial guess.
  * THIS VECTOR WILL ALSO HOLD THE SOLUTION OF QP */
  PetscCall(MatCreateVecs(A, &x, NULL));
  PetscCall(QPSetInitialVector(qp, x));
  /* Set box constraints.
  * c <= x <= PETSC_INFINITY */
  if (bounds) PetscCall(QPSetBox(qp,NULL,lb,ub));
  PetscCall(QPSetEq(qp, B, c));
  /* Set runtime options, e.g
  *   -qp_chain_view_kkt */
  PetscCall(QPSetFromOptions(qp));

  PetscCall(QPTFromOptions(qp));

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
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  testset:
    suffix: 1
    filter: grep -e CONVERGED -e number -e "r ="
    args: -n 100 -qps_view_convergence -qp_chain_view_kkt
    test:
    test:
      nsize: 3
  testset:
    filter: grep -e CONVERGED -e number -e "r ="
    args: -n 100 -qps_view_convergence -qp_chain_view_kkt
    nsize: 2
    test:
      suffix: opt
      args: -qps_mpgp_expansion_type gf -qps_mpgp_expansion_length_type opt
    test:
      suffix: optapprox
      args: -qps_mpgp_expansion_type g -qps_mpgp_expansion_length_type optapprox
    test:
      suffix: bb
      args: -qps_mpgp_expansion_type gfgr -qps_mpgp_expansion_length_type bb
    test:
      suffix: projcg
      args: -qps_mpgp_expansion_type projcg
  testset:
    filter: grep -e CONVERGED -e "function/" -e Objective -e "r ="
    args: -n 100 -qps_view_convergence -qp_chain_view_kkt -qps_type tao -qps_tao_type blmvm
    test:
      suffix: blmvm_1
    test:
      suffix: blmvm_3
      nsize: 3
TEST*/
