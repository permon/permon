#include <permonmat.h>
#include <../src/mat/impls/composite/permoncompositeimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatSumGetMat"
PetscErrorCode MatSumGetMat(Mat A,PetscInt i,Mat *Ai)
{
  PetscFunctionBegin;
  PetscCall(MatCompositeGetMat(A,i,Ai));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Sum"
PetscErrorCode MatMult_Sum(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next = shell->head;
  Vec               in,out;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->right) {
    if (!shell->rightwork) {
      PetscCall(VecDuplicate(shell->right,&shell->rightwork));
    }
    PetscCall(VecPointwiseMult(shell->rightwork,shell->right,in));
    in   = shell->rightwork;
  }
  PetscCall(MatMult(next->mat,in,y));
  while ((next = next->next)) {
    if (next->mat->ops->multadd) {
      PetscCall(MatMultAdd(next->mat,in,y,y));
    } else {
      if (!next->work) { /* should reuse previous work if the same size */
        PetscCall(MatCreateVecs(next->mat,NULL,&next->work));
      }
      out = next->work;
      PetscCall(MatMult(next->mat,in,out));
      PetscCall(VecAXPY(y,1.0,out));
    }
  }
  if (shell->left) {
    PetscCall(VecPointwiseMult(y,shell->left,y));
  }
  PetscCall(VecScale(y,shell->scale));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Sum"
PetscErrorCode MatMultTranspose_Sum(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next = shell->head;
  Vec               in;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->left) {
    if (!shell->leftwork) {
      PetscCall(VecDuplicate(shell->left,&shell->leftwork));
    }
    PetscCall(VecPointwiseMult(shell->leftwork,shell->left,in));
    in   = shell->leftwork;
  }
  PetscCall(MatMultTranspose(next->mat,in,y));
  while ((next = next->next)) {
    PetscCall(MatMultTransposeAdd(next->mat,in,y,y));
  }
  if (shell->right) {
    PetscCall(VecPointwiseMult(y,shell->right,y));
  }
  PetscCall(VecScale(y,shell->scale));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Sum"
PetscErrorCode MatMultAdd_Sum(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Composite *shell = (Mat_Composite *) A->data;

  PetscFunctionBegin;
  if (y != z) {
    PetscCall(MatMult_Sum(A,x,z));
    PetscCall(VecAXPY(z,1.0,y));
  } else {
    if (!shell->rightwork) {
      PetscCall(VecDuplicate(z,&shell->rightwork));
    }
    PetscCall(MatMult(A,x,shell->rightwork));
    PetscCall(VecAXPY(z,1.0,shell->rightwork));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Sum"
PetscErrorCode MatMultTransposeAdd_Sum(Mat A,Vec x,Vec y,Vec z)
{
  PetscFunctionBegin;
  PetscCall(MatMultTranspose_Sum(A, x, z));
  PetscCall(VecAXPY(z, 1.0, y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Sum"
FLLOP_EXTERN PetscErrorCode  MatCreate_Sum(Mat A)
{
  PetscErrorCode (*createComposite)(Mat);
  Mat_Composite  *composite;

  PetscFunctionBegin;
  PetscCall(PetscFunctionListFind(MatList,MATCOMPOSITE,(void(**)(void))&createComposite));
  PetscCall(createComposite(A));
  composite = (Mat_Composite*)A->data;

  A->ops->mult              = MatMult_Sum;
  A->ops->multtranspose     = MatMultTranspose_Sum;
  A->ops->multadd           = MatMultAdd_Sum;
  A->ops->multtransposeadd  = MatMultTransposeAdd_Sum;

  composite->type           = MAT_COMPOSITE_ADDITIVE;
  composite->head           = NULL;
  composite->tail           = NULL;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATSUM));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateSum"
/*@C
   MatCreateSum - Creates a matrix as the implicit sum of zero or more matrices.
   This class is an adoption of MATCOMPOSITE type in PETSc.

  Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  nmat - number of matrices to put in
-  mats - the matrices

   Output Parameter:
.  A - the matrix

   Level: advanced

   Notes:
     Alternative construction
$       MatCreate(comm,&mat);
$       MatSetType(mat,MATSUM);
$       MatCompositeAddMat(mat,mats[0]);
$       ....
$       MatCompositeAddMat(mat,mats[nmat-1]);
$       MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
$       MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCompositeMerge(), MatCompositeSetType(), MatCompositeType

@*/
PetscErrorCode  MatCreateSum(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat)
{
  PetscInt       m,n,M,N,i;

  PetscFunctionBegin;
  if (nmat < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must pass in at least one matrix");
  PetscValidPointer(mat,3);

  PetscCall(MatGetLocalSize(mats[0],PETSC_IGNORE,&n));
  PetscCall(MatGetLocalSize(mats[nmat-1],&m,PETSC_IGNORE));
  PetscCall(MatGetSize(mats[0],PETSC_IGNORE,&N));
  PetscCall(MatGetSize(mats[nmat-1],&M,PETSC_IGNORE));
  PetscCall(MatCreate(comm,mat));
  PetscCall(MatSetSizes(*mat,m,n,M,N));
  PetscCall(MatSetType(*mat,MATSUM));
  for (i=0; i<nmat; i++) {
    PetscCall(MatCompositeAddMat(*mat,mats[i]));
  }
  PetscCall(MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
