static char help[] =  "This example illustrates the use of (T)FETI.\n\n\
Solves -u'' = sin(pi*u) on domain [0,1].\n\
Homogenous dirichlet boundaries on both sides\n\
Number of subdomains is the same as the nubmer of MPI ranks.\n\
-ne sets number of elements per subdomain; default 3.\n\
-dir_in_hess if true enforce dirichlet in Hessian, use Eq. constraint otherwise; default False.\n\
Exaple usage: \n\
 mpirun -n 4 ./ex1 -ne 7\n\
";

#include <permonqps.h>
#include <permonksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat A;
  KSP ksp;
  Vec solution,rhs;
  PetscReal Aloc[4] = {1,-1,-1,1};
  PetscReal bloc[2];
  PetscReal h;
  PetscInt ndofs_l,ndofs,ns,ne,ne_l=3;
  PetscInt *global_indices,idx[2];
  PetscInt i,its,rank;
  ISLocalToGlobalMapping l2g;
  IS dirichletIS;
  PetscBool dirInHess = PETSC_FALSE; /* Enforce Dirichlet BC in Hessian? */
  KSPConvergedReason reason;
  PetscErrorCode ierr;

  /* Init PERMON */
  ierr = PermonInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /* Get number of elems per subdomain and number of subdomains */
  ierr = PetscOptionsGetInt(NULL,NULL,"-ne",&ne_l,NULL);CHKERRQ(ierr); /* number of local elements */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&ns);CHKERRQ(ierr); /* number of subdomains */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ne = ns*ne_l;
  ndofs = ns*ne_l+1;
  ndofs_l = ne_l+1;
  h = 1.0/ne;

  /* Create l2g mapping*/
  ierr = PetscMalloc1(ndofs_l,&global_indices);CHKERRQ(ierr);
  for (i=0; i<ndofs_l; i++) {
    global_indices[i]=rank*ne_l +i;
  }
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,ndofs_l,global_indices,PETSC_OWN_POINTER,&l2g);
  /* Create MATIS object needed by KSPFETI */
  ierr = MatCreateIS(PETSC_COMM_WORLD,1,PETSC_DECIDE,PETSC_DECIDE,ndofs,ndofs,l2g,NULL,&A);CHKERRQ(ierr);
  ierr = MatISSetPreallocation(A,3,NULL,3,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&solution,&rhs);CHKERRQ(ierr);

  /* assemble global matrix */
  for (i=0; i<ne_l; i++) {
    bloc[0] = sin((rank*ne_l +i+ 0.5)*h*3.14159) *.5*h; 
    bloc[1] = bloc[1]; 
    idx[0] = i;
    idx[1] = i+1;
    ierr = MatSetValuesLocal(A,2,idx,2,idx,Aloc,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecSetValuesLocal(rhs,2,idx,bloc,ADD_VALUES);CHKERRQ(ierr);
  }
  /* Call assembly functions */
  ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* create and customize KSP for (T)FETI */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPFETI);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  /* Set Dirichlet BC */
  idx[0] = 0; idx[1]=ndofs-1;
  if (!rank) { 
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,1,idx,PETSC_COPY_VALUES,&dirichletIS);CHKERRQ(ierr);
    if (ns==1) {
      idx[1]=ndofs_l-1;
      ierr = ISCreateGeneral(PETSC_COMM_WORLD,2,idx,PETSC_COPY_VALUES,&dirichletIS);CHKERRQ(ierr);
    }
  } else if (rank == ns-1) {
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,1,&idx[1],PETSC_COPY_VALUES,&dirichletIS);CHKERRQ(ierr);
  } else {
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,0,idx,PETSC_COPY_VALUES,&dirichletIS);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetBool(NULL,NULL,"-dir_in_hess",&dirInHess,NULL);CHKERRQ(ierr); 
  ierr = KSPFETISetDirichlet(ksp,dirichletIS,FETI_GLOBAL_UNDECOMPOSED,!dirInHess);CHKERRQ(ierr);
  /* Values of Dirichlet BC are passed in solution */
  //ierr = VecSet(solution,1.0);CHKERRQ(ierr); 
  //ierr = KSPSetInitialGuessNonzero);CHKERRQ(ierr);
  
  /* Solve */
  ierr = KSPSolve(ksp,rhs,solution);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"PERMON FETI %s in %d iteration\n",KSPConvergedReasons[reason],its);CHKERRQ(ierr);

  /* Free workspace */
  ierr = ISDestroy(&dirichletIS);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
  ierr = VecDestroy(&solution);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  /* Quit PERMON */
  ierr = PermonFinalize();
  return ierr;
}

/*TEST
  testset:
    nsize: 4
    requires: mumps
    filter: grep -e CONVERGED -e "r ="
    args: -ne 7 -qp_chain_view_kkt -qpt_matis_to_diag_norm
    test:
      suffix: 1
    test:
      suffix: 2
      args: -dir_in_hess
    test:
      suffix: 3
      output_file: output/ex1_1.out
      args: -dual_qppf_redundancy 2
TEST*/

