
#include <private/fllopmatimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMult_OneRow"
PetscErrorCode MatMult_OneRow(Mat A, Vec x, Vec z) {
    Vec a;
    PetscScalar alpha;

    PetscFunctionBegin;
    TRY( MatShellGetContext(A, (void*) &a) );
    TRY( VecDot(a,x,&alpha) );
    TRY( VecSet(z,alpha) );  /* z has length 1 */
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
    TRY( MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank) );
    TRY( MatShellGetContext(A, (void*) &a) );
    TRY( VecDot(a,x,&alpha) );
    TRY( VecGetArrayRead(w,&warr) );
    TRY( VecGetArray(z,&zarr) );
    if (!rank) {
      zarr[0] = warr[0] + alpha; /* w and z have length 1 */
    }
    TRY( VecRestoreArrayRead(w,&warr) );
    TRY( VecRestoreArray(z,&zarr) );
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
    TRY( MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank) );
    TRY( MatShellGetContext(A, (void*) &a) );
    if (!rank) {
      TRY( VecGetValues(x,1,&zero,&xval) );  /* x has length 1 */
    }
    TRY( MPI_Bcast(&xval,1,MPIU_SCALAR,0,PetscObjectComm((PetscObject)A)) );
    TRY( VecCopy(a,z));
    TRY( VecScale(z,xval) );
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
    TRY( MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank) );
    TRY( MatShellGetContext(A, (void*) &a) );
    if (!rank) {
      TRY( VecGetValues(x,1,&zero,&xval) );  /* x has length 1 */
    }
    TRY( MPI_Bcast(&xval,1,MPIU_SCALAR,0,PetscObjectComm((PetscObject)A)) );
    if (w == z) {
      TRY( VecAXPY(z,xval,a) );
    } else {
      TRY( VecWAXPY(z,xval,a,w) );
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_OneRow"
PetscErrorCode MatDestroy_OneRow(Mat A) {
    Vec a;

    PetscFunctionBegin;
    TRY( MatShellGetContext(A, (void*) &a) );
    TRY( VecDestroy(&a) );
    TRY( MatShellSetContext(A, NULL) );
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateOneRow"
PetscErrorCode MatCreateOneRow(Vec a, Mat *A_new)
{
  PetscInt n;
  Mat A;

  PetscFunctionBeginUser;
  TRY( VecGetLocalSize(a,&n) );
  TRY( MatCreateShellPermon(PetscObjectComm((PetscObject)a), PETSC_DECIDE, n, 1, PETSC_DECIDE, a, &A));
  TRY( FllopPetscObjectInheritName((PetscObject)A,(PetscObject)a,NULL) );
  TRY( PetscObjectReference((PetscObject)a) );

  TRY( MatShellSetOperation(A,MATOP_DESTROY,(void(*)(void))MatDestroy_OneRow) );
  TRY( MatShellSetOperation(A,MATOP_MULT,(void(*)(void))MatMult_OneRow) );
  TRY( MatShellSetOperation(A,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_OneRow) );
  TRY( MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_OneRow) );
  TRY( MatShellSetOperation(A,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))MatMultTransposeAdd_OneRow) );
  *A_new = A;
  PetscFunctionReturn(0);
}
