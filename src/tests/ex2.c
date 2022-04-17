
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
  PetscErrorCode    ierr;

  ierr = PermonInitialize(&argc,&args,(char *)0,(char *)0);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRQ(MPI_Comm_rank(comm,&rank));
  CHKERRQ(MPI_Comm_size(comm,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nloc",&n,NULL));
  CHKERRQ(PetscOptionsGetViewer(comm,NULL,NULL,"-view",&viewer,&format,NULL));

  CHKERRQ(PetscRandomCreate(comm,&rand));
  CHKERRQ(PetscRandomSetInterval(rand,0.0,10.0));

  CHKERRQ(MatCreateDense(comm,n,n,PETSC_DECIDE,PETSC_DECIDE,NULL,&A));
  CHKERRQ(MatGetSize(A,&N,NULL));
  CHKERRQ(MatSetRandom(A,rand));

  CHKERRQ(MatGetOwnershipRange(A,&rstart,NULL));
  CHKERRQ(ISCreateStride(comm,n,N-rstart,-1,&rperm));

  if (viewer) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(MatView(A,viewer));
    CHKERRQ(ISView(rperm,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }

  if (!rank) {
    nB = n + size - 1;
  } else {
    nB = n - 1;
  }
  CHKERRQ(MatCreateDense(comm,nB,nB,N,N,NULL,&B));
  CHKERRQ(MatRedistributeRows(A,rperm,1,B));

  if (viewer) {
    IS isr,isrg;

    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(MatView(B,viewer));
    CHKERRQ(MatGetOwnershipIS(B,&isr,NULL));
    CHKERRQ(ISOnComm(isr,comm,PETSC_USE_POINTER,&isrg));
    CHKERRQ(ISView(isrg,viewer));
    CHKERRQ(ISDestroy(&isr));
    CHKERRQ(ISDestroy(&isrg));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }

  CHKERRQ(ISDestroy(&rperm));

  CHKERRQ(MatGetOwnershipRange(B,&rstart,NULL));
  CHKERRQ(ISCreateStride(comm,nB,N-rstart,-1,&rperm));

  CHKERRQ(MatCreateDense(comm,n,n,N,N,NULL,&C));
  CHKERRQ(MatRedistributeRows(B,rperm,1,C));

  if (viewer) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(ISView(rperm,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(MatView(C,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }

  CHKERRQ(MatEqual(A,C,&flg));
  if (!flg) SETERRQ(comm, PETSC_ERR_PLIB, "C != A");

  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(ISDestroy(&rperm));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  ierr = PermonFinalize();
  return ierr;
}


/*TEST
  build:
    require: mumps
  test:
    nsize: {{1 2 4}}
    args: -nloc {{1 5 19}}
TEST*/
