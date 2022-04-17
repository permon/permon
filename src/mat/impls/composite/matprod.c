#include <permonmat.h>
#include <../src/mat/impls/composite/permoncompositeimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatProdGetMat_Prod"
static PetscErrorCode MatProdGetMat_Prod(Mat A,PetscInt index,Mat *Ai)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink ilink;
  PetscInt          i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ilink  = shell->head;
  for (i=0; i<index; i++) {
    if (ilink) {
      ilink = ilink->next;
    } else {
      break;
    }
  }
  if (!ilink) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"partial matrix index out of range: %d",i);
  *Ai = ilink->mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProdGetMat"
PetscErrorCode MatProdGetMat(Mat A,PetscInt i,Mat *Ai)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(A,i,2);
  PetscValidPointer(Ai,3);
  PetscUseMethod(A,"MatProdGetMat_Prod_C",(Mat,PetscInt,Mat*),(A,i,Ai));
  PetscFunctionReturn(0);
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
      CHKERRQ(VecDuplicate(shell->right,&shell->rightwork));
    }
    CHKERRQ(VecPointwiseMult(shell->rightwork,shell->right,in));
    in   = shell->rightwork;
  }
  while (next->next) {
    if (!next->work) { /* should reuse previous work if the same size */
      CHKERRQ(MatCreateVecs(next->mat,NULL,&next->work));
    }
    out = next->work;
    CHKERRQ(MatMult(next->mat,in,out));
    in   = out;
    next = next->next;
  }
  CHKERRQ(MatMult(next->mat,in,y));
  if (shell->left) {
    CHKERRQ(VecPointwiseMult(y,shell->left,y));
  }
  CHKERRQ(VecScale(y,shell->scale));
  PetscFunctionReturn(0);
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
      CHKERRQ(VecDuplicate(shell->left,&shell->leftwork));
    }
    CHKERRQ(VecPointwiseMult(shell->leftwork,shell->left,in));
    in   = shell->leftwork;
  }
  while (tail->prev) {
    if (!tail->prev->work) { /* should reuse previous work if the same size */
      CHKERRQ(MatCreateVecs(tail->mat,&tail->prev->work,NULL));
    }
    out = tail->prev->work;
    CHKERRQ(MatMultTranspose(tail->mat,in,out));
    in   = out;
    tail = tail->prev;
  }
  CHKERRQ(MatMultTranspose(tail->mat,in,y));
  if (shell->right) {
    CHKERRQ(VecPointwiseMult(y,shell->right,y));
  }
  CHKERRQ(VecScale(y,shell->scale));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_Prod"
PetscErrorCode MatMultAdd_Prod(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->leftwork) { CHKERRQ(VecDuplicate(z,&shell->leftwork)); }
    CHKERRQ(MatMult_Prod(A,x,shell->leftwork));
    CHKERRQ(VecCopy(y,z));
    CHKERRQ(VecAXPY(z,1.0,shell->leftwork));
  } else {
    CHKERRQ(MatMult_Prod(A,x,z));
    CHKERRQ(VecAXPY(z,1.0,y));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_Prod"
PetscErrorCode MatMultTransposeAdd_Prod(Mat A,Vec x,Vec y, Vec z)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;  

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->rightwork) { CHKERRQ(VecDuplicate(z,&shell->rightwork)); }
    CHKERRQ(MatMultTranspose_Prod(A,x,shell->rightwork));
    CHKERRQ(VecCopy(y,z));
    CHKERRQ(VecAXPY(z,1.0,shell->rightwork));
  } else {
    CHKERRQ(MatMultTranspose_Prod(A,x,z));
    CHKERRQ(VecAXPY(z,1.0,y));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreate_Prod"
FLLOP_EXTERN PetscErrorCode  MatCreate_Prod(Mat A)
{
  PetscErrorCode (*createComposite)(Mat);
  Mat_Composite  *composite;

  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListFind(MatList,MATCOMPOSITE,(void(**)(void))&createComposite));
  CHKERRQ(createComposite(A));
  composite = (Mat_Composite*)A->data;

  A->ops->mult               = MatMult_Prod;
  A->ops->multtranspose      = MatMultTranspose_Prod;
  A->ops->multadd            = MatMultAdd_Prod;
  A->ops->multtransposeadd   = MatMultTransposeAdd_Prod;
  A->ops->getdiagonal        = NULL;
  composite->type            = MAT_COMPOSITE_MULTIPLICATIVE;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatProdGetMat_Prod_C",MatProdGetMat_Prod));

  composite->type           = MAT_COMPOSITE_MULTIPLICATIVE;
  composite->head           = NULL;
  composite->tail           = NULL;

  CHKERRQ(PetscObjectChangeTypeName((PetscObject)A,MATPROD));
  PetscFunctionReturn(0);
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
  PetscValidPointer(mat,3);

  CHKERRQ(MatGetLocalSize(mats[0],PETSC_IGNORE,&n));
  CHKERRQ(MatGetLocalSize(mats[nmat-1],&m,PETSC_IGNORE));
  CHKERRQ(MatGetSize(mats[0],PETSC_IGNORE,&N));
  CHKERRQ(MatGetSize(mats[nmat-1],&M,PETSC_IGNORE));
  CHKERRQ(MatCreate(comm,mat));
  CHKERRQ(MatSetSizes(*mat,m,n,M,N));
  CHKERRQ(MatSetType(*mat,MATPROD));
  for (i=0; i<nmat; i++) {
    CHKERRQ(MatCompositeAddMat(*mat,mats[i]));
  }
  CHKERRQ(MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
