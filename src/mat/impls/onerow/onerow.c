
#include <permon/private/permonmatimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMult_OneRow"
PetscErrorCode MatMult_OneRow(Mat A, Vec x, Vec z) {
    Vec a;
    PetscScalar alpha;

    PetscFunctionBegin;
    PetscCall(MatShellGetContext(A, (void*) &a));
    PetscCall(VecDot(a,x,&alpha));
    PetscCall(VecSet(z,alpha));  /* z has length 1 */
    PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));
    PetscCall(MatShellGetContext(A, (void*) &a));
    PetscCall(VecDot(a,x,&alpha));
    PetscCall(VecGetArrayRead(w,&warr));
    PetscCall(VecGetArray(z,&zarr));
    if (!rank) {
      zarr[0] = warr[0] + alpha; /* w and z have length 1 */
    }
    PetscCall(VecRestoreArrayRead(w,&warr));
    PetscCall(VecRestoreArray(z,&zarr));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_OneRow"
PetscErrorCode MatMultTranspose_OneRow(Mat A, Vec x, Vec z) {
    PetscMPIInt rank;
    Vec a;
    PetscInt zero=0;
    PetscScalar xval;

    PetscFunctionBegin;
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));
    PetscCall(MatShellGetContext(A, (void*) &a));
    if (!rank) {
      PetscCall(VecGetValues(x,1,&zero,&xval));  /* x has length 1 */
    }
    PetscCallMPI(MPI_Bcast(&xval,1,MPIU_SCALAR,0,PetscObjectComm((PetscObject)A)));
    PetscCall(VecCopy(a,z));
    PetscCall(VecScale(z,xval));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_OneRow"
PetscErrorCode MatMultTransposeAdd_OneRow(Mat A, Vec x, Vec w, Vec z) { 
    PetscMPIInt rank;
    Vec a;
    PetscInt zero=0;
    PetscScalar xval;

    PetscFunctionBegin;
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));
    PetscCall(MatShellGetContext(A, (void*) &a));
    if (!rank) {
      PetscCall(VecGetValues(x,1,&zero,&xval));  /* x has length 1 */
    }
    PetscCallMPI(MPI_Bcast(&xval,1,MPIU_SCALAR,0,PetscObjectComm((PetscObject)A)));
    if (w == z) {
      PetscCall(VecAXPY(z,xval,a));
    } else {
      PetscCall(VecWAXPY(z,xval,a,w));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_OneRow"
PetscErrorCode MatDestroy_OneRow(Mat A) {
    Vec a;

    PetscFunctionBegin;
    PetscCall(MatShellGetContext(A, (void*) &a));
    PetscCall(VecDestroy(&a));
    PetscCall(MatShellSetContext(A, NULL));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateOneRow"
PetscErrorCode MatCreateOneRow(Vec a, Mat *A_new)
{
  PetscInt n;
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(a,&n));
  PetscCall(MatCreateShellPermon(PetscObjectComm((PetscObject)a), PETSC_DECIDE, n, 1, PETSC_DECIDE, a, &A));
  PetscCall(FllopPetscObjectInheritName((PetscObject)A,(PetscObject)a,NULL));
  PetscCall(PetscObjectReference((PetscObject)a));

  PetscCall(MatShellSetOperation(A,MATOP_DESTROY,(void(*)(void))MatDestroy_OneRow));
  PetscCall(MatShellSetOperation(A,MATOP_MULT,(void(*)(void))MatMult_OneRow));
  PetscCall(MatShellSetOperation(A,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_OneRow));
  PetscCall(MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_OneRow));
  PetscCall(MatShellSetOperation(A,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))MatMultTransposeAdd_OneRow));
  *A_new = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}
