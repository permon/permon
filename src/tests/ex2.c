
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
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nloc",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm,NULL,NULL,"-view",&viewer,&format,NULL);CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,0.0,10.0);CHKERRQ(ierr);

  ierr = MatCreateDense(comm,n,n,PETSC_DECIDE,PETSC_DECIDE,NULL,&A);CHKERRQ(ierr);
  ierr = MatGetSize(A,&N,NULL);CHKERRQ(ierr);
  ierr = MatSetRandom(A,rand);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&rstart,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,n,N-rstart,-1,&rperm);CHKERRQ(ierr);

  if (viewer) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = MatView(A,viewer);CHKERRQ(ierr);
    ierr = ISView(rperm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }

  if (!rank) {
    nB = n + size - 1;
  } else {
    nB = n - 1;
  }
  ierr = MatCreateDense(comm,nB,nB,N,N,NULL,&B);CHKERRQ(ierr);
  ierr = MatRedistributeRows(A,rperm,1,B);CHKERRQ(ierr);

  if (viewer) {
    IS isr,isrg;

    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = MatView(B,viewer);CHKERRQ(ierr);
    ierr = MatGetOwnershipIS(B,&isr,NULL);CHKERRQ(ierr);
    ierr = ISOnComm(isr,comm,PETSC_USE_POINTER,&isrg);CHKERRQ(ierr);
    ierr = ISView(isrg,viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&isr);CHKERRQ(ierr);
    ierr = ISDestroy(&isrg);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }

  ierr = ISDestroy(&rperm);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B,&rstart,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,nB,N-rstart,-1,&rperm);CHKERRQ(ierr);

  ierr = MatCreateDense(comm,n,n,N,N,NULL,&C);CHKERRQ(ierr);
  ierr = MatRedistributeRows(B,rperm,1,C);CHKERRQ(ierr);

  if (viewer) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = ISView(rperm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatView(C,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }

  ierr = MatEqual(A,C,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(comm, PETSC_ERR_PLIB, "C != A");

  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = ISDestroy(&rperm);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
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

