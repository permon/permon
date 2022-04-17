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

  /* Init PERMON */
  PetscCall(PermonInitialize(&argc,&args,(char*)0,help));

  /* Get number of elems per subdomain and number of subdomains */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ne",&ne_l,NULL)); /* number of local elements */
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&ns)); /* number of subdomains */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  ne = ns*ne_l;
  ndofs = ns*ne_l+1;
  ndofs_l = ne_l+1;
  h = 1.0/ne;

  /* Create l2g mapping*/
  PetscCall(PetscMalloc1(ndofs_l,&global_indices));
  for (i=0; i<ndofs_l; i++) {
    global_indices[i]=rank*ne_l +i;
  }
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,ndofs_l,global_indices,PETSC_OWN_POINTER,&l2g));
  /* Create MATIS object needed by KSPFETI */
  PetscCall(MatCreateIS(PETSC_COMM_WORLD,1,PETSC_DECIDE,PETSC_DECIDE,ndofs,ndofs,l2g,l2g,&A));
  PetscCall(MatISSetPreallocation(A,3,NULL,3,NULL));
  PetscCall(MatCreateVecs(A,&solution,&rhs));

  /* assemble global matrix */
  for (i=0; i<ne_l; i++) {
    bloc[0] = sin((rank*ne_l +i+ 0.5)*h*3.14159) *.5*h; 
    bloc[1] = bloc[1]; 
    idx[0] = i;
    idx[1] = i+1;
    PetscCall(MatSetValuesLocal(A,2,idx,2,idx,Aloc,ADD_VALUES));
    PetscCall(VecSetValuesLocal(rhs,2,idx,bloc,ADD_VALUES));
  }
  /* Call assembly functions */
  PetscCall(VecAssemblyBegin(rhs));
  PetscCall(VecAssemblyEnd(rhs));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* create and customize KSP for (T)FETI */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetType(ksp,KSPFETI));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  /* Set Dirichlet BC */
  idx[0] = 0; idx[1]=ndofs-1;
  if (!rank) { 
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,1,idx,PETSC_COPY_VALUES,&dirichletIS));
    if (ns==1) {
      idx[1]=ndofs_l-1;
      PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,2,idx,PETSC_COPY_VALUES,&dirichletIS));
    }
  } else if (rank == ns-1) {
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,1,&idx[1],PETSC_COPY_VALUES,&dirichletIS));
  } else {
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,0,idx,PETSC_COPY_VALUES,&dirichletIS));
  }
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-dir_in_hess",&dirInHess,NULL)); 
  PetscCall(KSPFETISetDirichlet(ksp,dirichletIS,FETI_GLOBAL_UNDECOMPOSED,PetscNot(dirInHess)));
  /* Values of Dirichlet BC are passed in solution */
  //PetscCall(VecSet(solution,1.0)); 
  //PetscCall(KSPSetInitialGuessNonzero));
  
  /* Solve */
  PetscCall(KSPSolve(ksp,rhs,solution));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(KSPGetConvergedReason(ksp,&reason));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"PERMON FETI %s in %d iteration\n",KSPConvergedReasons[reason],its));

  /* Free workspace */
  PetscCall(ISDestroy(&dirichletIS));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
  PetscCall(VecDestroy(&solution));
  PetscCall(VecDestroy(&rhs));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  /* Quit PERMON */
  PetscCall(PermonFinalize());
  return 0;
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
