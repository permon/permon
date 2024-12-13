/* Test MatRedistributeRows() */
#include <permonmat.h>

int main(int argc,char **args)
{
  Mat               A,B,C;
  PetscInt          n = 5,nB,N,rstart;
  PetscMPIInt       rank,size;
  PetscBool         flg;
  IS                rperm;
  PetscViewer       viewer=NULL;
  PetscViewerFormat format;
  PetscRandom       rand;
  MPI_Comm          comm;

  PetscCall(PermonInitialize(&argc,&args,(char *)0,(char *)0));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nloc",&n,NULL));
  PetscCall(PetscOptionsCreateViewer(comm,NULL,NULL,"-view",&viewer,&format,NULL));

  PetscCall(PetscRandomCreate(comm,&rand));
  PetscCall(PetscRandomSetInterval(rand,0.0,10.0));

  PetscCall(MatCreateDense(comm,n,n,PETSC_DECIDE,PETSC_DECIDE,NULL,&A));
  PetscCall(MatGetSize(A,&N,NULL));
  PetscCall(MatSetRandom(A,rand));

  PetscCall(MatGetOwnershipRange(A,&rstart,NULL));
  PetscCall(ISCreateStride(comm,n,N-rstart,-1,&rperm));

  if (viewer) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(MatView(A,viewer));
    PetscCall(ISView(rperm,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
  }

  if (!rank) {
    nB = n + size - 1;
  } else {
    nB = n - 1;
  }
  PetscCall(MatCreateDense(comm,nB,nB,N,N,NULL,&B));
  PetscCall(MatRedistributeRows(A,rperm,1,B));

  if (viewer) {
    IS isr,isrg;

    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(MatView(B,viewer));
    PetscCall(MatGetOwnershipIS(B,&isr,NULL));
    PetscCall(ISOnComm(isr,comm,PETSC_USE_POINTER,&isrg));
    PetscCall(ISView(isrg,viewer));
    PetscCall(ISDestroy(&isr));
    PetscCall(ISDestroy(&isrg));
    PetscCall(PetscViewerPopFormat(viewer));
  }

  PetscCall(ISDestroy(&rperm));

  PetscCall(MatGetOwnershipRange(B,&rstart,NULL));
  PetscCall(ISCreateStride(comm,nB,N-rstart,-1,&rperm));

  PetscCall(MatCreateDense(comm,n,n,N,N,NULL,&C));
  PetscCall(MatRedistributeRows(B,rperm,1,C));

  if (viewer) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(ISView(rperm,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatView(C,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
  }

  PetscCall(MatEqual(A,C,&flg));
  PetscCheck(flg,comm, PETSC_ERR_PLIB, "C != A");

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(ISDestroy(&rperm));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  build:
    require: mumps
  test:
    nsize: {{1 2 4}}
    args: -nloc {{1 5 19}}
TEST*/
