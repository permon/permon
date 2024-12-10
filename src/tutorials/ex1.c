
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
PetscErrorCode viewDraw(Vec x) {
  PetscViewer    v1;
  PetscDraw      draw;
  PetscFunctionBeginUser;

  PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,380,400,160,&v1));
  PetscCall(PetscViewerDrawGetDraw(v1,0,&draw));
  PetscCall(PetscDrawSetDoubleBuffer(draw));
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCall(VecView(x,v1));
  PetscCall(PetscViewerDestroy(&v1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Lower bound (obstacle) function */
PetscReal fobst(PetscInt i,PetscInt n) {
  PetscReal h = 1./(n-1);
  return PetscSinReal(4*PETSC_PI*i*h-PETSC_PI/6.)/2 -2;
}

int main(int argc,char **args)
{
  Vec            b,c,x;
  Mat            A;
  QP             qp;
  QPS            qps;
  PetscInt       i,n = 10,col[3],rstart,rend;
  PetscReal      h,value[3];
  PetscBool      converged,viewSol=PETSC_FALSE;

  PetscCall(PermonInitialize(&argc,&args,(char *)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-sol",&viewSol,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup matrices and vectors
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  h = 1./(n-1);
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  PetscCall(MatCreateVecs(A,&x,&b));
  PetscCall(VecDuplicate(x,&c));

  if (!rstart) {
    rstart = 1;
    i      = 0; value[0] = 1.0;
    PetscCall(MatSetValues(A,1,&i,1,&i,value,INSERT_VALUES));
    PetscCall(VecSetValue(b,i,0,INSERT_VALUES));
  }
  if (rend == n) {
    rend = n-1;
    i    = n-1; value[0] = 1.0;
    PetscCall(MatSetValues(A,1,&i,1,&i,value,INSERT_VALUES));
    PetscCall(VecSetValue(b,i,0,INSERT_VALUES));
  }
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    if (i == 1)   col[0] = -1; /* ignore the first value in the second row (Dirichlet BC) */
    if (i == n-2) col[2] = -1; /* ignore the third value in the second to last row (Dirichlet BC) */
    PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
    PetscCall(VecSetValue(b,i,-15*h*h*2,INSERT_VALUES));
    PetscCall(VecSetValue(c,i,fobst(i,n),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(VecAssemblyBegin(c));
  PetscCall(VecAssemblyEnd(c));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QP: argmin 1/2 x'Ax -x'b s.t. c <= x
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(QPCreate(PETSC_COMM_WORLD,&qp));
  /* Set matrix representing QP operator */
  PetscCall(QPSetOperator(qp,A));
  /* Set right hand side */
  PetscCall(QPSetRhs(qp,b));
  /* Set initial guess.
  * THIS VECTOR WILL ALSO HOLD THE SOLUTION OF QP */
  PetscCall(QPSetInitialVector(qp,x));
  /* Set box constraints.
  * c <= x <= PETSC_INFINITY */
  PetscCall(QPSetBox(qp,NULL,c,NULL));
  /* Set runtime options, e.g
  *   -qp_chain_view_kkt */
  PetscCall(QPSetFromOptions(qp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QPS, i.e. QP Solver
  *   Note the use of PetscObjectComm() to get the same comm as in qp object.
  *   We could specify the comm explicitly, in this case PETSC_COMM_WORLD.
  *   Also, all PERMON objects are PETSc objects as well :)
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(QPSCreate(PetscObjectComm((PetscObject)qp),&qps));
  /* Set QP to solve */
  PetscCall(QPSSetQP(qps,qp));
  /* Set runtime options for solver, e.g,
  *   -qps_type <type> -qps_rtol <relative tolerance> -qps_view_convergence */
  PetscCall(QPSSetFromOptions(qps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Solve QP
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(QPSSolve(qps));

  /* Check that QPS converged */
  PetscCall(QPIsSolved(qp,&converged));
  if (!converged) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"QPS did not converge!\n"));
  if (viewSol) PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  if (viewSol) PetscCall(viewDraw(c));
  if (viewSol) PetscCall(viewDraw(x));

  PetscCall(QPSDestroy(&qps));
  PetscCall(QPDestroy(&qp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&c));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
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
