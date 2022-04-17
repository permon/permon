
static char help[] = "Solves a tridiagonal system with lower bound on the first half of the components.\n\
Solves finite difference discretization of:\n\
-u''(x) = -15,  x in [0,1]\n\
u(0) = u(1) = 0\n\
s.t. u(x) >= sin(4*pi*x -pi/6)/2 -2, x in [0,1/2]\n\
Based on ex1\n\
Input parameters include:\n\
  -n <mesh_n> : number of mesh points in both x and y-direction\n\
  -infinite   : use PETSC_NINFINITY to keep part of the domain unconstrained\n";

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
  IS             is = NULL;
  PetscInt       i,n = 10,col[3],isn,rstart,rend;
  PetscReal      h,value[3];
  PetscBool      converged,infinite=PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&args,(char *)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-infinite",&infinite,NULL));

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
  if (infinite) {
    CHKERRQ(VecDuplicate(x,&c));
  } else {
    isn = rend;
    if (n/2 < rend) {
      if (n/2 < rstart) {
        isn = 0;
      } else {
        isn = n/2;
      }
    }
    if (isn) isn = isn - rstart;
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,isn,rstart,1,&is));
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&c));
    CHKERRQ(VecSetSizes(c,isn,n/2));
    CHKERRQ(VecSetFromOptions(c));
  }

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
    if (i<n/2) {
        CHKERRQ(VecSetValue(c,i,fobst(i,n),INSERT_VALUES));
    } else if (infinite) {
        CHKERRQ(VecSetValue(c,i,PETSC_NINFINITY,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));
  CHKERRQ(VecAssemblyBegin(c));
  CHKERRQ(VecAssemblyEnd(c));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QP: argmin 1/2 x'Ax -x'b s.t. c_i <= x_i where i in [n/2,n]
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
  CHKERRQ(QPSetBox(qp,is,c,NULL));
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

  CHKERRQ(ISDestroy(&is));
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
  test:
    suffix: 1
    filter: grep -e CONVERGED -e number -e "r ="
    nsize: 3
    args: -n 100 -qps_view_convergence -qp_chain_view_kkt -infinite {{false true}separate output}}
TEST*/
