
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,380,400,160,&v1));
  CHKERRQ(PetscViewerDrawGetDraw(v1,0,&draw));
  CHKERRQ(PetscDrawSetDoubleBuffer(draw));
  CHKERRQ(PetscDrawSetFromOptions(draw));
  CHKERRQ(VecView(x,v1));
  CHKERRQ(PetscViewerDestroy(&v1));
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&args,(char *)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-sol",&viewSol,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup matrices and vectors
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  h = 1./(n-1);
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  CHKERRQ(MatCreateVecs(A,&x,&b));
  CHKERRQ(VecDuplicate(x,&c));

  if (!rstart) {
    rstart = 1;
    i      = 0; value[0] = 1.0;
    CHKERRQ(MatSetValues(A,1,&i,1,&i,value,INSERT_VALUES));
    CHKERRQ(VecSetValue(b,i,0,INSERT_VALUES));
  }
  if (rend == n) {
    rend = n-1;
    i    = n-1; value[0] = 1.0;
    CHKERRQ(MatSetValues(A,1,&i,1,&i,value,INSERT_VALUES));
    CHKERRQ(VecSetValue(b,i,0,INSERT_VALUES));
  }
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    if (i == 1)   col[0] = -1; /* ignore the first value in the second row (Dirichlet BC) */
    if (i == n-2) col[2] = -1; /* ignore the third value in the second to last row (Dirichlet BC) */
    CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
    CHKERRQ(VecSetValue(b,i,-15*h*h*2,INSERT_VALUES));
    CHKERRQ(VecSetValue(c,i,fobst(i,n),INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));
  CHKERRQ(VecAssemblyBegin(c));
  CHKERRQ(VecAssemblyEnd(c));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QP: argmin 1/2 x'Ax -x'b s.t. c <= x
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(QPCreate(PETSC_COMM_WORLD,&qp));
  /* Set matrix representing QP operator */
  CHKERRQ(QPSetOperator(qp,A));
  /* Set right hand side */
  CHKERRQ(QPSetRhs(qp,b));
  /* Set initial guess.
  * THIS VECTOR WILL ALSO HOLD THE SOLUTION OF QP */
  CHKERRQ(QPSetInitialVector(qp,x));
  /* Set box constraints.
  * c <= x <= PETSC_INFINITY */
  CHKERRQ(QPSetBox(qp,NULL,c,NULL));
  /* Set runtime options, e.g
  *   -qp_chain_view_kkt */
  CHKERRQ(QPSetFromOptions(qp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QPS, i.e. QP Solver
  *   Note the use of PetscObjectComm() to get the same comm as in qp object.
  *   We could specify the comm explicitly, in this case PETSC_COMM_WORLD.
  *   Also, all PERMON objects are PETSc objects as well :)
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(QPSCreate(PetscObjectComm((PetscObject)qp),&qps));
  /* Set QP to solve */
  CHKERRQ(QPSSetQP(qps,qp));
  /* Set runtime options for solver, e.g,
  *   -qps_type <type> -qps_rtol <relative tolerance> -qps_view_convergence */
  CHKERRQ(QPSSetFromOptions(qps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Solve QP
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(QPSSolve(qps));

  /* Check that QPS converged */
  CHKERRQ(QPIsSolved(qp,&converged));
  if (!converged) PetscPrintf(PETSC_COMM_WORLD,"QPS did not converge!\n");
  if (viewSol) CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  if (viewSol) CHKERRQ(viewDraw(c));
  if (viewSol) CHKERRQ(viewDraw(x));

  CHKERRQ(QPSDestroy(&qps));
  CHKERRQ(QPDestroy(&qp));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&c));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  ierr = PermonFinalize();
  return ierr;
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
