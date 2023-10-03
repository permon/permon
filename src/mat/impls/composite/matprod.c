#include <permonmat.h>
#include <../src/mat/impls/composite/permoncompositeimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatProdGetMat"
PetscErrorCode MatProdGetMat(Mat A,PetscInt i,Mat *Ai)
{
  PetscFunctionBegin;
  PetscCall(MatCompositeGetMat(A,i,Ai));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Prod"
PetscErrorCode MatMult_Prod(Mat A,Vec x,Vec y)
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
  while (next->next) {
    if (!next->work) { /* should reuse previous work if the same size */
      PetscCall(MatCreateVecs(next->mat,NULL,&next->work));
    }
    out = next->work;
    PetscCall(MatMult(next->mat,in,out));
    in   = out;
    next = next->next;
  }
  PetscCall(MatMult(next->mat,in,y));
  if (shell->left) {
    PetscCall(VecPointwiseMult(y,shell->left,y));
  }
  PetscCall(VecScale(y,shell->scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Prod"
PetscErrorCode MatMultTranspose_Prod(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  
  Mat_CompositeLink tail = shell->tail;
  Vec               in,out;

  PetscFunctionBegin;
  if (!tail) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->left) {
    if (!shell->leftwork) {
      PetscCall(VecDuplicate(shell->left,&shell->leftwork));
    }
    PetscCall(VecPointwiseMult(shell->leftwork,shell->left,in));
    in   = shell->leftwork;
  }
  while (tail->prev) {
    if (!tail->prev->work) { /* should reuse previous work if the same size */
      PetscCall(MatCreateVecs(tail->mat,&tail->prev->work,NULL));
    }
    out = tail->prev->work;
    PetscCall(MatMultTranspose(tail->mat,in,out));
    in   = out;
    tail = tail->prev;
  }
  PetscCall(MatMultTranspose(tail->mat,in,y));
  if (shell->right) {
    PetscCall(VecPointwiseMult(y,shell->right,y));
  }
  PetscCall(VecScale(y,shell->scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_Prod"
PetscErrorCode MatMultAdd_Prod(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->leftwork) { PetscCall(VecDuplicate(z,&shell->leftwork)); }
    PetscCall(MatMult_Prod(A,x,shell->leftwork));
    PetscCall(VecCopy(y,z));
    PetscCall(VecAXPY(z,1.0,shell->leftwork));
  } else {
    PetscCall(MatMult_Prod(A,x,z));
    PetscCall(VecAXPY(z,1.0,y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_Prod"
PetscErrorCode MatMultTransposeAdd_Prod(Mat A,Vec x,Vec y, Vec z)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->rightwork) { PetscCall(VecDuplicate(z,&shell->rightwork)); }
    PetscCall(MatMultTranspose_Prod(A,x,shell->rightwork));
    PetscCall(VecCopy(y,z));
    PetscCall(VecAXPY(z,1.0,shell->rightwork));
  } else {
    PetscCall(MatMultTranspose_Prod(A,x,z));
    PetscCall(VecAXPY(z,1.0,y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreate_Prod"
FLLOP_EXTERN PetscErrorCode  MatCreate_Prod(Mat A)
{
  PetscErrorCode (*createComposite)(Mat);
  Mat_Composite  *composite;

  PetscFunctionBegin;
  PetscCall(PetscFunctionListFind(MatList,MATCOMPOSITE,(void(**)(void))&createComposite));
  PetscCall(createComposite(A));
  composite = (Mat_Composite*)A->data;

  A->ops->mult               = MatMult_Prod;
  A->ops->multtranspose      = MatMultTranspose_Prod;
  A->ops->multadd            = MatMultAdd_Prod;
  A->ops->multtransposeadd   = MatMultTransposeAdd_Prod;
  A->ops->getdiagonal        = NULL;
  composite->type            = MAT_COMPOSITE_MULTIPLICATIVE;

  composite->type           = MAT_COMPOSITE_MULTIPLICATIVE;
  composite->head           = NULL;
  composite->tail           = NULL;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATPROD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateProd"
/*@C
   MatCreateProd - Creates a matrix as the implicit product of zero or more matrices.
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
$       MatSetType(mat,MATPROD);
$       MatCompositeAddMat(mat,mats[0]);
$       ....
$       MatCompositeAddMat(mat,mats[nmat-1]);
$       MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
$       MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);

     For the multiplicative form the product is mat[nmat-1]*mat[nmat-2]*....*mat[0]

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCompositeMerge(), MatCompositeSetType(), MatCompositeType

@*/
PetscErrorCode  MatCreateProd(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat)
{
  PetscInt       m,n,M,N,i;
  
  PetscFunctionBegin;
  if (nmat < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must pass in at least one matrix");
  PetscAssertPointer(mat,3);

  PetscCall(MatGetLocalSize(mats[0],PETSC_IGNORE,&n));
  PetscCall(MatGetLocalSize(mats[nmat-1],&m,PETSC_IGNORE));
  PetscCall(MatGetSize(mats[0],PETSC_IGNORE,&N));
  PetscCall(MatGetSize(mats[nmat-1],&M,PETSC_IGNORE));
  PetscCall(MatCreate(comm,mat));
  PetscCall(MatSetSizes(*mat,m,n,M,N));
  PetscCall(MatSetType(*mat,MATPROD));
  for (i=0; i<nmat; i++) {
    PetscCall(MatCompositeAddMat(*mat,mats[i]));
  }
  PetscCall(MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
