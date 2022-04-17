
#include <permon/private/permonmatimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMult_OneRow"
PetscErrorCode MatMult_OneRow(Mat A, Vec x, Vec z) {
    Vec a;
    PetscScalar alpha;

    PetscFunctionBegin;
    CHKERRQ(MatShellGetContext(A, (void*) &a));
    CHKERRQ(VecDot(a,x,&alpha));
    CHKERRQ(VecSet(z,alpha));  /* z has length 1 */
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_OneRow"
PetscErrorCode MatMultAdd_OneRow(Mat A, Vec x, Vec w, Vec z) {
    PetscMPIInt rank;
    Vec a;
    PetscScalar alpha;
    const PetscScalar *warr;
    PetscScalar *zarr;

    PetscFunctionBegin;
    CHKERRQ(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));
    CHKERRQ(MatShellGetContext(A, (void*) &a));
    CHKERRQ(VecDot(a,x,&alpha));
    CHKERRQ(VecGetArrayRead(w,&warr));
    CHKERRQ(VecGetArray(z,&zarr));
    if (!rank) {
      zarr[0] = warr[0] + alpha; /* w and z have length 1 */
    }
    CHKERRQ(VecRestoreArrayRead(w,&warr));
    CHKERRQ(VecRestoreArray(z,&zarr));
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_OneRow"
PetscErrorCode MatMultTranspose_OneRow(Mat A, Vec x, Vec z) {
    PetscMPIInt rank;
    Vec a;
    PetscInt zero=0;
    PetscScalar xval;

    PetscFunctionBegin;
    CHKERRQ(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));
    CHKERRQ(MatShellGetContext(A, (void*) &a));
    if (!rank) {
      CHKERRQ(VecGetValues(x,1,&zero,&xval));  /* x has length 1 */
    }
    CHKERRQ(MPI_Bcast(&xval,1,MPIU_SCALAR,0,PetscObjectComm((PetscObject)A)));
    CHKERRQ(VecCopy(a,z));
    CHKERRQ(VecScale(z,xval));
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_OneRow"
PetscErrorCode MatMultTransposeAdd_OneRow(Mat A, Vec x, Vec w, Vec z) { 
    PetscMPIInt rank;
    Vec a;
    PetscInt zero=0;
    PetscScalar xval;

    PetscFunctionBegin;
    CHKERRQ(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));
    CHKERRQ(MatShellGetContext(A, (void*) &a));
    if (!rank) {
      CHKERRQ(VecGetValues(x,1,&zero,&xval));  /* x has length 1 */
    }
    CHKERRQ(MPI_Bcast(&xval,1,MPIU_SCALAR,0,PetscObjectComm((PetscObject)A)));
    if (w == z) {
      CHKERRQ(VecAXPY(z,xval,a));
    } else {
      CHKERRQ(VecWAXPY(z,xval,a,w));
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_OneRow"
PetscErrorCode MatDestroy_OneRow(Mat A) {
    Vec a;

    PetscFunctionBegin;
    CHKERRQ(MatShellGetContext(A, (void*) &a));
    CHKERRQ(VecDestroy(&a));
    CHKERRQ(MatShellSetContext(A, NULL));
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateOneRow"
PetscErrorCode MatCreateOneRow(Vec a, Mat *A_new)
{
  PetscInt n;
  Mat A;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetLocalSize(a,&n));
  CHKERRQ(MatCreateShellPermon(PetscObjectComm((PetscObject)a), PETSC_DECIDE, n, 1, PETSC_DECIDE, a, &A));
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)A,(PetscObject)a,NULL));
  CHKERRQ(PetscObjectReference((PetscObject)a));

  CHKERRQ(MatShellSetOperation(A,MATOP_DESTROY,(void(*)(void))MatDestroy_OneRow));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT,(void(*)(void))MatMult_OneRow));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_OneRow));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_OneRow));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))MatMultTransposeAdd_OneRow));
  *A_new = A;
  PetscFunctionReturn(0);
}
