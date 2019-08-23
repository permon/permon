
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

int main(int argc,char **args)
{
  Vec            b,lb,x;
  Mat            A,B;
  QP             qp;
  QPS            qps;
  PetscInt       i,n = 10,col[3],rstart,rend;
  PetscReal      h,value[3];
  char           file[4][PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      converged,flg;
  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&args,(char *)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f1",file[1],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f2",file[2],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f3",file[3],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  /* open binary file. Note that we use FILE_MODE_READ to indicate reading from this file */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  /* load the matrices and vectors; then destroy the viewer */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&lb);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(b,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[3],FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(lb,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QP: argmin 1/2 x'Ax -x'b s.t. lb <= x and Bx = 0
  *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = QPCreate(PETSC_COMM_WORLD,&qp);CHKERRQ(ierr);
  /* Set matrix representing QP operator */
  ierr = QPSetOperator(qp,A);CHKERRQ(ierr);
  /* Set right hand side */
  ierr = QPSetRhs(qp,b);CHKERRQ(ierr);
  /* Set initial guess
  * THIS VECTOR WILL ALSO HOLD THE SOLUTION OF QP */
  ierr = QPSetInitialVector(qp,x);CHKERRQ(ierr);
  /* Set box constraints
  * c <= x <= PETSC_INFINITY */
  ierr = QPSetBox(qp,NULL,lb,NULL);CHKERRQ(ierr);
  /* Set equality constraint */
  ierr = QPSetEq(qp,B,NULL);CHKERRQ(ierr);
  /* Set runtime options, e.g
  *   -qp_chain_view_kkt */
  ierr = QPSetFromOptions(qp);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  * Setup QPS, i.e., QP Solver
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

  ierr = QPSDestroy(&qps);CHKERRQ(ierr);
  ierr = QPDestroy(&qp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&lb);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
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

