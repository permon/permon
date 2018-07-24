
static char help[] = "Solves a tridiagonal system with lower bound.\n\
Input parameters include:\n\
  -view_sol   : view solution vector\n\
  -n <mesh_n> : number of mesh points in both x and y-direction\n\n";

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
#include <math.h>
#define PI 3.14159265

/* Draw vector */
PetscErrorCode viewDraw(Vec x) {
  PetscViewer    v1;
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",80,380,400,160,&v1);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(v1,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = VecView(x,v1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&v1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Lower bound (obstacle) function */
PetscReal fobst(PetscInt i,PetscInt n) {
  PetscReal h = 1./(n-1);
  return sin(4*PI*i*h-PI/6.)/2 -2; 
}

#undef __FUNCT__
#define __FUNCT__ "main"
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-sol",&viewSol,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup matrices and vectors
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  h = 1./(n-1);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&c);CHKERRQ(ierr);

  if (!rstart) {
    rstart = 1;
    i      = 0; value[0] = 1.0;
    ierr   = MatSetValues(A,1,&i,1,&i,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecSetValue(b,i,0,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (rend == n) {
    rend = n-1;
    i    = n-1; value[0] = 1.0;
    ierr = MatSetValues(A,1,&i,1,&i,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(b,i,0,INSERT_VALUES);CHKERRQ(ierr);
  }
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(b,i,-15*h*h*2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(c,i,fobst(i,n),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(c);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(c);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QP: argmin 1/2 x'Ax -x'b s.t. c <= x
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = QPCreate(PETSC_COMM_WORLD,&qp);CHKERRQ(ierr);
  /* Set matrix representing QP operator */
  ierr = QPSetOperator(qp,A);CHKERRQ(ierr);
  /* Set right hand side */
  ierr = QPSetRhs(qp,b);CHKERRQ(ierr);
  /* Set initial guess. 
  * THIS VECTOR WILL ALSO HOLD THE SOLUTION OF QP */
  ierr = QPSetInitialVector(qp,x);CHKERRQ(ierr);
  /* Set box constraints.
  * c <= x <= PETSC_INFINITY */
  ierr = QPSetBox(qp,NULL,c,NULL);CHKERRQ(ierr);
  /* Set runtime options, e.g
  *   -qp_chain_view_kkt */
  ierr = QPSetFromOptions(qp);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QPS, i.e. QP Solver
  *   Note the use of PetscObjectComm() to get the same comm as in qp object.
  *   We could specify the comm explicitly, in this case PETSC_COMM_WORLD.
  *   Also, all PERMON objects are PETSc objects as well :)
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = QPSCreate(PetscObjectComm((PetscObject)qp),&qps);CHKERRQ(ierr);
  /* Set QP to solve */
  ierr = QPSSetQP(qps,qp);CHKERRQ(ierr);
  /* Set runtime options for solver, e.g,
  *   -qps_type <type> -qps_rtol <relative tolerance> -qps_view_convergence */
  ierr = QPSSetFromOptions(qps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Solve QP
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = QPSSolve(qps);CHKERRQ(ierr);

  /* Check that QPS converged */
  ierr = QPIsSolved(qp,&converged);CHKERRQ(ierr);
  if (!converged) PetscPrintf(PETSC_COMM_WORLD,"QPS did not converge!\n");
  if (viewSol) ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  if (viewSol) ierr = viewDraw(c);CHKERRQ(ierr);
  if (viewSol) ierr = viewDraw(x);CHKERRQ(ierr);

  ierr = QPSDestroy(&qps);CHKERRQ(ierr);
  ierr = QPDestroy(&qp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&c);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
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
    filter: grep -e CONVERGED -e "function/" -e Objective -e "r ="
    args: -n 100 -qps_view_convergence -qp_chain_view_kkt -qps_type tao -qps_tao_type blmvm
    test:
      suffix: blmvm_1
    test:
      suffix: blmvm_3
      nsize: 3
TEST*/

